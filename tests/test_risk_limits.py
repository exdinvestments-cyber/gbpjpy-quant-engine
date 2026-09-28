"""Phase 1F: period limits, boundaries, persistence, consecutive losses, drawdown, adjustments, open/reserved/aggregate
risk, exposure, external rulesets, weekend/news, kill switch, fail-closed, staleness, revalidation, idempotency,
concurrency."""

from __future__ import annotations

import threading
from decimal import Decimal as Dec

import pandas as pd
import pytest

from gbpjpy_engine.risk import (RESET_CONFIRMATION, BalanceAdjustment, ClosedTradeResult, ContractSpec, GenericRuleset,
                                JsonFileRiskStateStore, ResetAuthorisation, StaticRates, period_keys)
from risk_helpers import T0, account, engine, policy, position, proposal, rates
from trade_helpers import build, make_ctx

DAY = pd.Timedelta(days=1)


def loss(eng, rid, amount, ts, partial=False):
    return eng.record_closed_result(ClosedTradeResult(rid, f"POS-{rid}", None, Dec(amount), Dec(0), Dec(0), ts, "STOP",
                                                      partial=partial))


def ok(dec):
    return dec["decision"] == "RISK_APPROVED"


# ------------------------------------------------------------------ period limits
def test_daily_limit_blocks_and_binds():
    eng = engine()
    loss(eng, "a", "-120", T0 - pd.Timedelta(hours=1))
    loss(eng, "b", "-50", T0 - pd.Timedelta(minutes=30))
    dec = eng.approve(proposal(pid="P-1"), account(balance="9830"), T0)
    assert ok(dec) and dec["binding_constraint"] == "DAILY_REMAINING_RISK"
    assert Dec(dec["approved_trade"]["risk_amount"]) <= Dec("26.60")
    eng2 = engine()
    loss(eng2, "c", "-200", T0 - pd.Timedelta(hours=1))
    rej = eng2.approve(proposal(pid="P-2"), account(balance="9800"), T0)
    assert rej["rejection_category"] == "DAILY_LIMIT" and "DAILY_LIMIT_REACHED" in rej["reason_codes"]
    nxt = eng2.approve(proposal(pid="P-3", ts=T0 + DAY), account(balance="9800", ts=T0 + DAY), T0 + DAY)
    assert ok(nxt) and "DAILY_LIMIT_OK" in nxt["reason_codes"]


def test_daily_budget_includes_floating_and_open_risk():
    eng = engine()
    rej = eng.approve(proposal(), account(balance="10000", equity="9790", floating="-210"), T0)
    assert rej["rejection_category"] == "DAILY_LIMIT"
    eng2 = engine(policy(limits={"pyramiding_enabled": True, "max_same_direction_positions": 3, "max_symbol_positions": 3,
                                 "max_aggregate_risk_percent": 5.0, "max_symbol_risk_percent": 5.0}))
    big = position(volume="0.50", entry=190.3, stop=189.6, current_price=190.3)  # ~184 GBP remaining stop risk
    dec = eng2.approve(proposal(), account(positions=[big]), T0)
    assert dec["binding_constraint"] == "DAILY_REMAINING_RISK" and Dec(dec["approved_trade"]["risk_amount"]) < Dec("20")


def test_weekly_and_monthly_limits():
    eng = engine()
    for k, amt in enumerate(("-150", "-150", "-150")):
        loss(eng, f"w{k}", amt, T0 - DAY * (2 - k) - pd.Timedelta(hours=3))  # Mon..Wed of the same week
    rej = eng.approve(proposal(), account(balance="9550"), T0)
    assert rej["rejection_category"] in ("WEEKLY_LIMIT", "DAILY_LIMIT")
    eng_w = engine()
    loss(eng_w, "x1", "-190", T0 - DAY - pd.Timedelta(hours=3))
    loss(eng_w, "x2", "-190", T0 - 2 * DAY - pd.Timedelta(hours=3))
    loss(eng_w, "x3", "-30", T0 - pd.Timedelta(hours=1))
    w = eng_w.approve(proposal(), account(balance="9590"), T0)
    assert w["rejection_category"] == "WEEKLY_LIMIT" and "WEEKLY_LIMIT_REACHED" in w["reason_codes"]
    eng_m = engine(policy(limits={"monthly_loss_percent": 1.0}))
    loss(eng_m, "m", "-100", T0 - DAY - pd.Timedelta(hours=3))
    m = eng_m.approve(proposal(), account(balance="9900"), T0)
    assert m["rejection_category"] == "MONTHLY_LIMIT"
    off = engine(policy(limits={"monthly_loss_percent": 1.0, "monthly_enabled": False}))
    loss(off, "m", "-100", T0 - DAY - pd.Timedelta(hours=3))
    assert ok(off.approve(proposal(), account(balance="9900"), T0))


def test_day_week_month_boundaries_are_deterministic_and_dst_aware():
    assert period_keys(pd.Timestamp("2024-01-02 21:59", tz="UTC")).day == "2024-01-02"  # 16:59 EST
    assert period_keys(pd.Timestamp("2024-01-02 22:00", tz="UTC")).day == "2024-01-03"  # 17:00 EST roll
    assert period_keys(pd.Timestamp("2024-07-01 20:59", tz="UTC")).day == "2024-07-01"  # 16:59 EDT
    assert period_keys(pd.Timestamp("2024-07-01 21:00", tz="UTC")).day == "2024-07-02"  # 17:00 EDT roll
    assert period_keys(pd.Timestamp("2024-01-07 21:59", tz="UTC")).week == "2024-W01"  # Sunday before the roll
    assert period_keys(pd.Timestamp("2024-01-07 22:00", tz="UTC")).week == "2024-W02"  # Sunday roll = new week
    assert period_keys(pd.Timestamp("2024-01-31 22:30", tz="UTC")).month == "2024-02"
    assert period_keys(pd.Timestamp("2024-01-03 05:00", tz="Asia/Tokyo")) == period_keys(pd.Timestamp("2024-01-02 20:00", tz="UTC"))
    assert period_keys(pd.Timestamp("2024-01-02 21:00", tz="UTC"), "Europe/London", 0).day == "2024-01-02"
    with pytest.raises(ValueError):
        period_keys(pd.Timestamp("2024-01-02 21:00"))  # naive time never accepted


# ------------------------------------------------------------------ persistence / restart
def test_risk_state_survives_restart(tmp_path):
    path = tmp_path / "risk_state.json"
    e1 = engine(store=JsonFileRiskStateStore(path))
    assert ok(e1.approve(proposal(pid="P-1"), account(balance="10000"), T0))
    loss(e1, "r1", "-150", T0 + pd.Timedelta(minutes=30))
    for k in range(3):
        loss(e1, f"c{k}", "-1", T0 + pd.Timedelta(minutes=31 + k))
    e1.approve(proposal(pid="P-9"), account(balance="11000", ts=T0 + pd.Timedelta(minutes=50)), T0 + pd.Timedelta(minutes=50))
    e2 = engine(store=JsonFileRiskStateStore(path))  # "software restart"
    st = e2.state()
    assert st.periods["day"]["reference"] == "10000"  # the restart did not reset the daily reference
    assert st.consecutive_losses == 4 and st.hwm_equity == "11000" and "RA-P-1" in st.reservations  # P-9: pyramiding
    again = e2.approve(proposal(pid="P-1"), account(balance="10000"), T0)
    assert again.get("idempotent_replay") and len(e2.state().reservations) == 1


def test_corrupted_state_fails_closed_until_authorised_reset(tmp_path):
    path = tmp_path / "risk_state.json"
    e1 = engine(store=JsonFileRiskStateStore(path))
    e1.approve(proposal(pid="P-1"), account(), T0)
    path.write_text(path.read_text().replace('"ENABLED"', '"ENABLEX"'))
    e2 = engine(store=JsonFileRiskStateStore(path))
    dec = e2.approve(proposal(pid="P-2"), account(), T0)
    assert dec["rejection_category"] == "CORRUPTED_RISK_STATE" and "RISK_STATE_CORRUPTED" in dec["reason_codes"]
    assert list(tmp_path.glob("risk_state.json.corrupt*"))  # kept for inspection
    assert engine(store=JsonFileRiskStateStore(path)).approve(proposal(pid="P-3"), account(), T0)["rejection_category"] == "GLOBAL_HALT"
    assert e2.reset(ResetAuthorisation("ops", "restored", RESET_CONFIRMATION), T0)["status"] == "RESET"
    assert ok(e2.approve(proposal(pid="P-4"), account(), T0))


# ------------------------------------------------------------------ consecutive losses
def test_consecutive_losses_reduce_then_pause_never_increase():
    eng = engine()
    base = Dec(eng.approve(proposal(pid="P-0"), account(), T0)["approved_trade"]["risk_amount"])
    eng.release_reservation("RA-P-0", "test", T0)
    for k in range(3):
        loss(eng, f"l{k}", "-1", T0 + pd.Timedelta(minutes=k + 1))
    red = eng.approve(proposal(pid="P-1"), account(ts=T0 + pd.Timedelta(minutes=10)), T0 + pd.Timedelta(minutes=10))
    assert red["binding_constraint"] == "CONSECUTIVE_LOSS_CAP" and Dec(red["approved_trade"]["risk_amount"]) == base / 2
    eng.release_reservation("RA-P-1", "test", T0 + pd.Timedelta(minutes=10))
    for k in range(3, 5):
        loss(eng, f"l{k}", "-1", T0 + pd.Timedelta(minutes=k + 11))
    p = eng.approve(proposal(pid="P-2"), account(ts=T0 + pd.Timedelta(minutes=20)), T0 + pd.Timedelta(minutes=20))
    assert p["rejection_category"] == "CONSECUTIVE_LOSS_PAUSE" and eng.state().global_state == "PAUSED"
    assert eng.approve(proposal(pid="P-3"), account(ts=T0 + pd.Timedelta(hours=1)), T0 + pd.Timedelta(hours=1))[
        "rejection_category"] == "GLOBAL_PAUSE"
    t2 = T0 + DAY
    nxt = eng.approve(proposal(pid="P-4", ts=t2), account(ts=t2), t2)  # next trading day: lifted, still reduced
    assert ok(nxt) and Dec(nxt["approved_trade"]["risk_amount"]) == base / 2
    eng.record_closed_result(ClosedTradeResult("win", "POS-w", None, Dec("30"), Dec(0), Dec(0), t2 + pd.Timedelta(hours=1), "TARGET"))
    assert eng.state().consecutive_losses == 0
    man = engine(policy(consecutive={"pause_until": "MANUAL_RESET"}))
    for k in range(5):
        loss(man, f"l{k}", "-1", T0 + pd.Timedelta(minutes=k))
    man.approve(proposal(pid="P-5"), account(ts=T0 + pd.Timedelta(minutes=10)), T0 + pd.Timedelta(minutes=10))
    assert man.approve(proposal(pid="P-6", ts=t2), account(ts=t2), t2)["rejection_category"] == "GLOBAL_PAUSE"


# ------------------------------------------------------------------ drawdown & high-water marks
def test_drawdown_tiers_reduce_and_halt_persistently():
    eng = engine()
    eng.approve(proposal(pid="P-0"), account(balance="10000"), T0)
    eng.release_reservation("RA-P-0", "t", T0)
    tiers = {}
    for k, (bal, pid) in enumerate((("9400", "P-1"), ("8900", "P-2")), start=1):
        t = T0 + k * DAY  # losses realised on earlier days: today's budget is intact, the drawdown is not
        d = eng.approve(proposal(pid=pid, ts=t), account(balance=bal, ts=t), t)
        tiers[bal] = d
        if ok(d):
            eng.release_reservation(d["risk_approval_id"], "t", t)
    assert tiers["9400"]["audit"]["drawdown"]["state"] == "CAUTION" and "DRAWDOWN_CAUTION" in tiers["9400"]["reason_codes"]
    assert tiers["9400"]["binding_constraint"] == "DRAWDOWN_CAP"
    assert Dec(tiers["9400"]["approved_trade"]["risk_amount"]) == Dec("35.25")  # 0.5% x 9400 x 0.75
    assert tiers["8900"]["audit"]["drawdown"]["state"] == "DEFENSIVE"
    assert Dec(tiers["8900"]["approved_trade"]["risk_amount"]) == Dec("22.25")  # 0.5% x 8900 x 0.5
    h = engine()
    h.approve(proposal(pid="P-0"), account(balance="10000"), T0)
    t1 = T0 + DAY
    halt = h.approve(proposal(pid="P-1", ts=t1), account(balance="8400", ts=t1), t1)
    assert halt["rejection_category"] == "DRAWDOWN_HALT" and h.state().global_state == "HALTED"
    t2 = T0 + 2 * DAY
    assert h.approve(proposal(pid="P-2", ts=t2), account(balance="9900", ts=t2), t2)["rejection_category"] == "GLOBAL_HALT"
    assert h.state().hwm_equity == "10000" and h.state().hwm_balance == "10000"


def test_deposits_and_withdrawals_are_not_trading_results():
    eng = engine()
    eng.approve(proposal(pid="P-0"), account(balance="10000"), T0)
    eng.release_reservation("RA-P-0", "t", T0)
    eng.record_adjustment(BalanceAdjustment("dep1", "DEPOSIT", Dec("5000"), T0 + pd.Timedelta(minutes=1)))
    st = eng.state()
    assert st.hwm_equity == "15000" and st.periods["day"]["reference"] == "15000" and not st.ledger
    eng.record_adjustment(BalanceAdjustment("wd1", "WITHDRAWAL", Dec("-6000"), T0 + pd.Timedelta(minutes=2)))
    d = eng.approve(proposal(pid="P-1"), account(balance="9000", ts=T0 + pd.Timedelta(minutes=3)), T0 + pd.Timedelta(minutes=3))
    assert ok(d) and d["audit"]["drawdown"]["state"] == "NORMAL" and d["audit"]["periods"]["day"]["realised_net"] == "0"
    assert eng.record_adjustment(BalanceAdjustment("wd1", "WITHDRAWAL", Dec("-6000"), T0))["status"] == "DUPLICATE_IGNORED"


# ------------------------------------------------------------------ open / reserved / aggregate / exposure
def test_reserved_risk_and_aggregate_cap():
    pol = policy(limits={"pyramiding_enabled": True, "allow_opposite_direction": True, "max_same_direction_positions": 5,
                         "max_symbol_positions": 5, "max_aggregate_risk_percent": 0.8, "max_symbol_risk_percent": 0.8})
    eng = engine(pol)
    a = eng.approve(proposal(pid="P-1"), account(), T0)
    b = eng.approve(proposal(pid="P-2"), account(), T0)
    assert ok(a) and ok(b) and b["binding_constraint"] == "AGGREGATE_RISK_CAP"
    agg = Dec(b["approved_trade"]["aggregate_risk_after_trade"])
    assert agg <= Dec("80.00") and Dec(b["approved_trade"]["aggregate_risk_before_trade"]) == Dec(a["approved_trade"]["actual_risk_currency"])
    c = eng.approve(proposal(pid="P-3"), account(), T0)
    assert not ok(c) and "AGGREGATE_RISK_EXCEEDED" in c["reason_codes"] or c["rejection_category"] == "BROKER_MIN_VOLUME"
    rej = engine(policy(limits={**{"pyramiding_enabled": True, "max_same_direction_positions": 5, "max_symbol_positions": 5},
                               "max_aggregate_risk_percent": 0.8, "aggregate_policy": "REJECT"}))
    rej.approve(proposal(pid="P-1"), account(), T0)
    r2 = rej.approve(proposal(pid="P-2"), account(), T0)
    assert r2["rejection_category"] == "AGGREGATE_RISK_CAP" and "AGGREGATE_RISK_EXCEEDED" in r2["reason_codes"]
    assert eng.release_reservation("RA-P-1", "order not placed", T0)["status"] == "RELEASED"
    assert eng.release_reservation("RA-P-1", "again", T0)["status"] == "ALREADY_RELEASED"


def test_symbol_exposure_duplicates_pyramiding_and_losers():
    eng = engine()
    dup = position(trade_proposal_id="P-X", setup_id="S-1")
    assert eng.approve(proposal(pid="P-1"), account(positions=[dup]), T0)["rejection_category"] == "DUPLICATE_EXPOSURE"
    same = position(pid="POS-2")
    assert eng.approve(proposal(pid="P-2"), account(positions=[same]), T0)["rejection_category"] == "PYRAMIDING_DISABLED"
    loser = position(pid="POS-3", floating="-20")
    pyr = engine(policy(limits={"pyramiding_enabled": True, "max_same_direction_positions": 3, "max_symbol_positions": 3}))
    assert pyr.approve(proposal(pid="P-3"), account(positions=[loser]), T0)["rejection_category"] == "ADD_TO_LOSER_PROHIBITED"
    opp = position(pid="POS-4", direction="SHORT", entry=190.0, stop=190.5)
    assert eng.approve(proposal(pid="P-4"), account(positions=[opp]), T0)["rejection_category"] == "SYMBOL_EXPOSURE_LIMIT"
    nostop = position(pid="POS-5", direction="SHORT", stop=None)
    assert eng.approve(proposal(pid="P-5"), account(positions=[nostop]), T0)["rejection_category"] == "OPEN_RISK_UNKNOWN"
    first = eng.approve(proposal(pid="P-6"), account(), T0)
    assert ok(first) and eng.approve(proposal(pid="P-7"), account(), T0)["rejection_category"] == "PYRAMIDING_DISABLED"


# ------------------------------------------------------------------ external rulesets / weekend / news
def test_external_ruleset_envelope():
    rs = GenericRuleset(initial_balance=Dec("10000"), daily_loss_limit=Dec("500"), max_drawdown=Dec("1000"),
                        drawdown_mode="TRAILING", max_volume=Dec("0.05"), max_open_positions=1, consistency_max_day_share=0.3)
    eng = engine(external_policy=rs)
    d = eng.approve(proposal(), account(profile="EXTERNAL_RULESET"), T0)
    a = d["approved_trade"]
    assert ok(d) and a["external_policy_status"] == "OK" and Dec(a["approved_volume"]) == Dec("0.05")
    assert d["audit"]["external_policy"]["consistency"]["status"] in ("WITHIN", "OUTSIDE")
    assert engine().approve(proposal(), account(profile="EXTERNAL_RULESET"), T0)["rejection_category"] == "EXTERNAL_POLICY_MISSING"
    tight = engine(external_policy=GenericRuleset(initial_balance=Dec("10000"), max_drawdown=Dec("1000")))
    t = tight.approve(proposal(), account(balance="9030"), T0)
    assert t["binding_constraint"] == "EXTERNAL_POLICY_CAP"
    br = engine(external_policy=GenericRuleset(initial_balance=Dec("10000"), daily_loss_limit=Dec("100")))
    b = br.approve(proposal(), account(equity="9890", floating="-110"), T0)
    assert b["rejection_category"] == "EXTERNAL_POLICY" and br.state().global_state == "HALTED"


def test_weekend_and_news_policies():
    fri = pd.Timestamp("2024-01-05 19:00", tz="UTC")  # 14:00 New York, 3 h before the Friday roll
    p = proposal(ts=fri)
    assert ok(engine(rate_provider=rates(ts=fri - pd.Timedelta(minutes=5))).approve(p, account(ts=fri), fri))
    for pol_name in ("DISALLOW", "CLOSE_BEFORE_WEEKEND", "UNKNOWN_EXTERNAL_POLICY"):
        e = engine(policy(events={"weekend": pol_name}), rate_provider=rates(ts=fri - pd.Timedelta(minutes=5)))
        assert e.approve(p, account(ts=fri), fri)["rejection_category"] == "WEEKEND_POLICY", pol_name
    assert ok(engine(policy(events={"weekend": "DISALLOW"})).approve(proposal(), account(), T0))  # midweek
    imminent = dict(proposal(), news_status="EVENT_IMMINENT")
    assert engine(policy(events={"news": "BLOCK_BEFORE"})).approve(imminent, account(), T0)["rejection_category"] == "NEWS_POLICY"
    assert ok(engine(policy(events={"news": "BLOCK_AFTER"})).approve(imminent, account(), T0))
    red = engine(policy(events={"news": "REDUCE_RISK"})).approve(imminent, account(), T0)
    assert red["binding_constraint"] == "NEWS_RISK_CAP"
    unk = engine().approve(proposal(), account(), T0)
    assert ok(unk) and "NEWS_UNKNOWN" in unk["reason_codes"] and unk["approved_trade"]["news_risk_status"].startswith("UNKNOWN")
    assert engine(policy(events={"block_unknown_news": True})).approve(proposal(), account(), T0)["rejection_category"] == "NEWS_POLICY"


# ------------------------------------------------------------------ kill switch / manual halt
def test_manual_halt_persists_until_authorised_reset(tmp_path):
    path = tmp_path / "rs.json"
    e1 = engine(store=JsonFileRiskStateStore(path))
    e1.halt("dashboard emergency", "operator-1", T0)
    e2 = engine(store=JsonFileRiskStateStore(path))
    d = e2.approve(proposal(), account(), T0)
    assert d["rejection_category"] == "GLOBAL_HALT" and "GLOBAL_HALT" in d["reason_codes"]
    assert e2.reset(ResetAuthorisation("op", "fixed", "yes"), T0)["status"] == "REFUSED"
    assert e2.reset(ResetAuthorisation("", "fixed", RESET_CONFIRMATION), T0)["status"] == "REFUSED"
    assert engine(store=JsonFileRiskStateStore(path)).state().global_state == "HALTED"
    assert e2.reset(ResetAuthorisation("op", "fixed", RESET_CONFIRMATION), T0)["from"] == "HALTED"
    assert ok(e2.approve(proposal(), account(), T0))
    e2.pause("maintenance", "op", T0)
    assert e2.approve(proposal(pid="P-9"), account(), T0)["rejection_category"] == "GLOBAL_PAUSE"
    assert any(a["event"] == "AUTHORISED_RESET" for a in e2.state().audit)


# ------------------------------------------------------------------ fail closed / staleness / revalidation
def test_engine_fails_closed():
    p = proposal()
    assert engine(rate_provider=StaticRates()).approve(p, account(), T0)["rejection_category"] == "CONVERSION_UNAVAILABLE"
    assert "CONVERSION_RATE_MISSING" in engine(rate_provider=StaticRates()).approve(p, account(), T0)["reason_codes"]
    stale = engine(rate_provider=rates(ts=T0 - pd.Timedelta(hours=5), span_hours=0)).approve(p, account(), T0)
    assert "CONVERSION_RATE_STALE" in stale["reason_codes"]
    e = engine()
    bad = e.approve(p, account(balance="NaN"), T0)
    assert bad["rejection_category"] == "ACCOUNT_STATE_INVALID" and e.state().global_state == "HALTED"
    assert engine().approve(p, account(currency="gbp"), T0)["rejection_category"] == "ACCOUNT_STATE_INVALID"
    assert engine().approve(p, None, T0)["rejection_category"] == "ACCOUNT_STATE_INVALID"
    unknown_size = engine(contract=ContractSpec(contract_size=Dec("0")))
    assert unknown_size.approve(p, account(), T0)["rejection_category"] == "SYMBOL_SPEC_INVALID"
    assert unknown_size.state().global_state == "HALTED"
    for broken in ({"proposed_stop_price": float("nan")}, {"proposed_stop_price": p["executable_reference_price"]},
                   {"direction": "FLAT"}, {"decision": "REJECT_TRADE"}, {"symbol": "EURUSD"}):
        assert engine().approve(dict(p, **broken), account(), T0)["rejection_category"] == "PROPOSAL_INVALID", broken


def test_stale_or_future_account_state_is_rejected():
    assert engine().approve(proposal(), account(ts=T0 - pd.Timedelta(minutes=10)), T0)["rejection_category"] == "ACCOUNT_STATE_STALE"
    fut = engine().approve(proposal(), account(ts=T0 + pd.Timedelta(seconds=1)), T0)
    assert "ACCOUNT_STATE_FROM_FUTURE" in fut["reason_codes"]


def test_invalid_proposals_cannot_be_revived():
    rec = build(make_ctx())
    p = dict(rec.proposal)
    assert ok(engine().approve(p, account(), T0, construction=rec, construction_index=rec.index))
    t_late = T0 + pd.Timedelta(hours=2)
    rec.move(rec.index + 2, t_late.isoformat(), "EXPIRED", "entry candidate expired")
    assert ok(engine().approve(dict(p, trade_proposal_id="P-early"), account(), T0, construction=None))
    late = engine().approve(p, account(ts=t_late), t_late, construction=rec, construction_index=rec.index + 2)
    assert late["rejection_category"] == "PROPOSAL_NO_LONGER_VALID"
    moved = dict(p, proposed_stop_price=p["proposed_stop_price"] + 0.2)
    g = engine().approve(moved, account(), T0, construction=rec, construction_index=rec.index)
    assert "PROPOSAL_GEOMETRY_CHANGED" in g["reason_codes"]
    h4 = engine().approve(p, account(), T0, h4_permission_now="BLOCK_ALL")
    assert "H4_PERMISSION_REVOKED" in h4["reason_codes"]


# ------------------------------------------------------------------ idempotency & concurrency
def test_idempotent_approvals_and_results():
    eng = engine()
    a1 = eng.approve(proposal(pid="P-1"), account(), T0)
    a2 = eng.approve(proposal(pid="P-1"), account(), T0 + pd.Timedelta(seconds=5))
    assert a2["idempotent_replay"] and a2["risk_approval_id"] == a1["risk_approval_id"]
    assert len(eng.state().reservations) == 1
    assert loss(eng, "r", "-10", T0)["status"] == "RECORDED" and loss(eng, "r", "-10", T0)["status"] == "DUPLICATE_IGNORED"
    assert eng.state().consecutive_losses == 1


def _race(engines, n):
    out, barrier = [], threading.Barrier(n)

    def go(k):
        barrier.wait()
        out.append(engines[k % len(engines)].approve(proposal(pid=f"P-{k}"), account(), T0))

    th = [threading.Thread(target=go, args=(k,)) for k in range(n)]
    [t.start() for t in th]
    [t.join() for t in th]
    return out


def test_concurrent_candidates_cannot_share_the_same_budget(tmp_path):
    pol = policy(limits={"pyramiding_enabled": True, "allow_opposite_direction": True, "max_same_direction_positions": 50,
                         "max_symbol_positions": 50, "max_aggregate_risk_percent": 0.6, "max_symbol_risk_percent": 0.6,
                         "aggregate_policy": "REJECT"})
    res = _race([engine(pol)], 12)
    approved = [r for r in res if ok(r)]
    assert len(approved) == 1
    path = tmp_path / "shared.json"
    res2 = _race([engine(pol, store=JsonFileRiskStateStore(path)), engine(pol, store=JsonFileRiskStateStore(path))], 10)
    assert sum(ok(r) for r in res2) == 1
    st = engine(pol, store=JsonFileRiskStateStore(path)).state()
    assert len(st.reservations) == 1 and sum(Dec(r["amount"]) for r in st.reservations.values()) <= Dec("60")
