"""Phase 1D end-to-end behaviour: lifecycle, candidate object, bid/ask, spread, gap, deterioration, windows,
revalidation, H4 revocation, setup invalidation, room, abnormality, news, session, weekend, duplicates,
conflict/quality, counterfactuals, audit trail and synthetic edge cases."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.entry import (BarOpenQuotes, EntryConfig, EntryIntelligenceEngine, ExecutableQuote,
                                 IntrabarConfirmation, explain_entry_candidate)
from gbpjpy_engine.entry.config import CONFIRMATION_FAMILIES
from gbpjpy_engine.entry.confirmation import SHORT_NAME
from gbpjpy_engine.entry.execution import session_context
from gbpjpy_engine.reason_codes import ReasonCode

VALID = {c.value for c in ReasonCode}
LEGAL = {
    "NO_ENTRY": {"WAITING_FOR_CONFIRMATION"},
    "WAITING_FOR_CONFIRMATION": {"CONFIRMING", "EXPIRED", "INVALIDATED"},
    "CONFIRMING": {"CONFIRMING", "WAITING_FOR_CONFIRMATION", "ENTRY_CANDIDATE", "DEFERRED", "REJECTED", "EXPIRED",
                   "INVALIDATED"},
    "DEFERRED": {"DEFERRED", "ENTRY_CANDIDATE", "WAITING_FOR_CONFIRMATION", "REJECTED", "EXPIRED", "INVALIDATED"},
    "ENTRY_CANDIDATE": {"EXPIRED", "INVALIDATED"},
}
REQUIRED = ("entry_candidate_id", "setup_id", "timestamp", "symbol", "direction", "setup_family", "confirmation_family",
            "signal_price", "executable_reference_price", "spread_status", "spread_pips", "price_deterioration",
            "chase_risk_score", "extension_score", "freshness_score", "confirmation_quality_score",
            "entry_market_quality_score", "remaining_room_score", "barrier_density_score", "entry_conflict_score",
            "entry_quality_score", "h4_permission", "h1_setup_score", "session_context", "news_status", "reason_codes")
FORBIDDEN = ("position_size", "size", "lots", "lot_size", "risk_amount", "stop_loss", "take_profit", "sl", "tp",
             "order_ticket", "ticket", "order_id")


@pytest.fixture(scope="module")
def up(entry_results):
    return entry_results["entry_uptrend"]


@pytest.fixture(scope="module")
def down(entry_results):
    return entry_results["entry_downtrend"]


def accepted_cand(er, side=None):
    return [c for c in er.candidates if c.accepted is not None and (side is None or c.side == side)]


def run(h1, h4, cfg=None, **kw):
    return EntryIntelligenceEngine(cfg or EntryConfig()).run(h1, h4, **kw)


def edited(h1, table, edits):
    df = getattr(h1, table).copy()
    for (i, col), v in edits.items():
        df.at[i, col] = v
    return replace(h1, **{table: df})


def quote_bars(h1, edits):
    b = h1.features[["timestamp", "open", "high", "low", "close", "spread"]].copy()
    for (i, col), v in edits.items():
        b.at[i, col] = v
    return BarOpenQuotes(b)


def by_setup(er, setup_id):
    return next(c for c in er.candidates if c.setup_id == setup_id)


# ------------------------------------------------------------------ pipeline & state machine
def test_qualified_setups_feed_the_entry_engine(up, down):
    for _, _, h1, er in (up, down):
        q = [s for s in h1.setups if s.qualified_index is not None]
        assert sorted(c.setup_id for c in er.candidates) == sorted(s.setup_id for s in q)
        for c in er.candidates:
            stp = next(s for s in q if s.setup_id == c.setup_id)
            assert c.entry_candidate_id == f"E-{stp.setup_id}" and c.qualified_index == stp.qualified_index
            assert c.transitions[0][:5] == (stp.qualified_index, "CLOSE", c.qualified_at, "NO_ENTRY",
                                            "WAITING_FOR_CONFIRMATION")


def test_state_machine_transitions_are_legal_and_auditable(up, down):
    for _, _, _, er in (up, down):
        for c in er.candidates:
            prev_idx, prev_to = -1, "NO_ENTRY"
            for idx, phase, at, frm, to, reason in c.transitions:
                assert frm == prev_to and to in LEGAL[frm], (c.entry_candidate_id, frm, to)
                assert idx >= prev_idx and phase in ("OPEN", "CLOSE") and reason
                prev_idx, prev_to = idx, to
            assert c.state == prev_to and c.state in ("REJECTED", "EXPIRED", "INVALIDATED")
            assert c.state_at(c.transitions[-1][0]) == c.state and c.end_reason and c.end_category
            if c.confirmation is not None:  # confirmations are strictly after qualification
                assert c.confirmation_index > c.qualified_index


def test_one_candidate_per_setup_and_one_active_per_side(up, down):
    for _, _, _, er in (up, down):
        ids = [c.entry_candidate_id for c in er.candidates]
        assert len(ids) == len(set(ids)) == len({c.setup_id for c in er.candidates})
        assert all(len([o for o in c.opportunities if o["decision"] == "ACCEPT_ENTRY_CANDIDATE"]) <= 1
                   for c in er.candidates)
        for side in ("long", "short"):
            spans = sorted((c.qualified_index, c.end_index) for c in er.candidates if c.side == side)
            assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # never two active per side
        assert len(er.accepted) == len(accepted_cand(er))


# ------------------------------------------------------------------ the candidate object & prices
def test_accepted_long_and_short_candidate_objects(up, down):
    for (_, _, h1, er), side in ((up, "long"), (down, "short")):
        acc = accepted_cand(er, side)
        assert acc, f"no accepted {side} candidate"
        for c in acc:
            p = c.accepted
            assert all(k in p for k in REQUIRED)
            assert not any(k in p for k in FORBIDDEN)
            j, ci = c.accepted_index, c.confirmation_index
            f = h1.features
            assert j == ci + 1 and p["timestamp"] == f["timestamp"].iloc[j].isoformat()
            assert p["signal_price"] == pytest.approx(f["close"].iloc[ci])  # the confirmation close (reference only)
            spread = f["spread"].iloc[ci] * 0.01  # spread KNOWN at the decision = last completed bar's report
            if side == "long":  # buys the ASK: next open + spread on bid-based charts
                assert p["execution_side"] == "ASK"
                assert p["executable_reference_price"] == pytest.approx(f["open"].iloc[j] + spread)
            else:  # sells the BID
                assert p["execution_side"] == "BID"
                assert p["executable_reference_price"] == pytest.approx(f["open"].iloc[j])
            assert p["direction"] == side.upper() and p["h4_permission"] in ("ALLOW_LONG", "ALLOW_SHORT", "ALLOW_BOTH")
            assert p["slippage_status"] == "UNKNOWN" and p["news_status"] == "UNKNOWN"
            assert set(p["reason_codes"]) <= VALID and "ENTRY_CANDIDATE_ACCEPTED" in p["reason_codes"]
            assert "not a probability" in p["entry_quality_note"].lower() and "no position size" in p["not_an_order"]
            assert p["session_context"] == session_context(f["timestamp"].iloc[j], h1.config.sessions.sessions)


def test_multiple_barriers_from_both_timeframes_are_considered(up, down):
    seen_tf = set()
    for _, _, _, er in (up, down):
        for c in er.candidates:
            for o in c.opportunities:
                for cl in (o.get("metrics") or {}).get("barriers", {}).get("clusters", []):
                    seen_tf |= set(cl["timeframes"])
                    assert cl["distance_atr"] >= 0 and cl["n_members"] == len(cl["members"])
    assert seen_tf == {"H1", "H4"}


# ------------------------------------------------------------------ spread
def test_confirmation_with_excessive_spread_is_rejected(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    alt = run(h1, h4, quotes=quote_bars(h1, {(c.confirmation_index, "spread"): 12.0}))
    r = by_setup(alt, c.setup_id)
    assert r.state == "REJECTED" and r.end_category == "SPREAD" and r.accepted is None
    assert "SPREAD_TOO_HIGH" in r.opportunities[-1]["reason_codes"]
    cf = next(x for x in alt.counterfactual if x["setup_id"] == c.setup_id)
    assert cf["category"] == "SPREAD" and cf["hypothetical"]["spread_status"] in ("HIGH", "EXTREME")


def test_missing_spread_is_unknown_never_zero(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    alt = run(h1, h4, quotes=quote_bars(h1, {(c.confirmation_index, "spread"): np.nan}))
    p = by_setup(alt, c.setup_id).accepted
    assert p["spread_status"] == "UNKNOWN" and p["spread_pips"] is None and p["spread_assumed"] is True
    assert "SPREAD_UNKNOWN" in p["reason_codes"]
    assert p["executable_reference_price"] > p["signal_price"]  # a conservative spread was applied, not zero
    strict = EntryConfig(spread=replace(EntryConfig().spread, block_unknown=True))
    r = by_setup(run(h1, h4, strict, quotes=quote_bars(h1, {(c.confirmation_index, "spread"): np.nan})), c.setup_id)
    assert r.state == "REJECTED" and r.end_category == "SPREAD"


def test_spread_rejections_on_low_volatility_phase1c_data(h1_results):
    """Phase 1C synthetic H1 ATR is ~15 pips, so its 1.5-3.5 pip spreads are HIGH relative to ATR."""
    _, h4, h1 = h1_results["h1_uptrend_pullbacks"]
    er = run(h1, h4)
    spread_rej = [c for c in er.candidates if c.end_category == "SPREAD"]
    assert spread_rej and all(c.accepted is None for c in spread_rej)


# ------------------------------------------------------------------ gap & price deterioration
def test_gap_deterioration_rejects_and_gap_against_thesis_defers(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    ci, j = c.confirmation_index, c.confirmation_index + 1
    atr, close = h1.features["atr"].iloc[ci], h1.features["close"].iloc[ci]
    away = by_setup(run(h1, h4, quotes=quote_bars(h1, {(j, "open"): close + 0.8 * atr})), c.setup_id)
    last = away.opportunities[-1]
    assert away.state == "REJECTED" and last["category"] == "PRICE_DETERIORATION"
    assert last["metrics"]["deterioration"]["price_deterioration_atr"] >= 0.8
    assert last["metrics"]["gap"]["gap_class"] == "LARGE" and "EXECUTION_GAP" in last["reason_codes"]
    jump = by_setup(run(h1, h4, quotes=quote_bars(h1, {(j, "open"): close + 1.3 * atr})), c.setup_id)
    assert {"PRICE_DETERIORATION", "GAP"} <= {r["category"] for r in jump.opportunities[-1]["all_rejection_reasons"]}
    against = by_setup(run(h1, h4, quotes=quote_bars(h1, {(j, "open"): close - 0.7 * atr})), c.setup_id)
    assert against.opportunities[0]["decision"] == "DEFER" and against.opportunities[0]["category"] == "GAP"


# ------------------------------------------------------------------ abnormal movement / volatility
def test_extreme_volatility_rejects_and_abnormal_conditions_defer(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    p = c.confirmation_index
    ext = by_setup(run(edited(h1, "features", {(p, "range_atr"): 5.0}), h4), c.setup_id)
    assert ext.state == "REJECTED" and ext.end_category == "ABNORMAL"
    assert "EXECUTION_CONDITIONS_ABNORMAL" in ext.opportunities[-1]["reason_codes"]
    mild = by_setup(run(edited(h1, "features", {(p, "atr_ratio"): 1.95}), h4), c.setup_id)
    first = mild.opportunities[0]
    assert first["decision"] == "DEFER" and first["category"] == "ABNORMAL"
    assert mild.state_at(p + 1) in ("DEFERRED", "EXPIRED", "INVALIDATED")


# ------------------------------------------------------------------ windows / freshness
def test_entry_window_expires(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    closed = EntryConfig(lifecycle=replace(EntryConfig().lifecycle, window_bars=0))
    r = by_setup(run(h1, h4, closed), c.setup_id)
    assert r.state == "EXPIRED" and r.end_category == "WINDOW_EXPIRED"
    assert "ENTRY_WINDOW_EXPIRED" in r.opportunities[-1]["reason_codes"]
    # repeated deferrals exhaust the window instead of waiting indefinitely
    p = c.confirmation_index
    edits = {(i, "atr_ratio"): 1.95 for i in range(p, p + 4)}
    d = by_setup(run(edited(h1, "features", edits), h4), c.setup_id)
    decisions = [o["decision"] for o in d.opportunities]
    assert decisions[0] == "DEFER" and "ACCEPT_ENTRY_CANDIDATE" not in decisions
    if len(decisions) == 3:
        assert decisions[-1] == "EXPIRE"
    # accepted candidates do not stay available forever either
    assert all(x.state in ("EXPIRED", "INVALIDATED") and x.end_index >= x.accepted_index for x in accepted_cand(er))


def test_stale_signal_is_rejected(down):
    _, h4, h1, er = down
    c = accepted_cand(er, "short")[0]
    strict = EntryConfig(freshness=replace(EntryConfig().freshness, fresh_score=100.0, aging_score=99.9, stale_score=1.0))
    r = by_setup(run(h1, h4, strict), c.setup_id)
    assert r.state == "REJECTED" and r.end_category == "STALENESS"
    assert "SIGNAL_STALE" in r.opportunities[-1]["reason_codes"]


# ------------------------------------------------------------------ chase / extension / room / conflict / quality
def test_confirmation_after_price_already_ran_is_rejected_for_chasing(up, down):
    chased = [c for _, _, _, er in (up, down) for c in er.candidates if c.end_category == "CHASE"]
    assert chased
    for c in chased:
        m = c.opportunities[-1]["metrics"]["chase"]
        assert m["chase_class"] in ("HIGH", "EXTREME") and c.accepted is None
        assert any(code.startswith("CHASE_RISK_") for code in c.opportunities[-1]["reason_codes"])


def test_room_recheck_extension_conflict_and_quality_can_reject(up):
    _, h4, h1, er = up
    base = EntryConfig()
    confirmed = {c.setup_id for c in er.candidates if c.opportunities}
    cases = {
        "ROOM": replace(base, barriers=replace(base.barriers, min_remaining_room_score=101.0)),
        "CONFLICT": replace(base, scoring=replace(base.scoring, max_entry_conflict=-1.0)),
        "QUALITY": replace(base, scoring=replace(base.scoring, min_entry_quality=101.0)),
        "EXTENSION": replace(base, extension=replace(base.extension, overextended_score=-1.0)),
    }
    for cat, cfg in cases.items():
        alt = run(h1, h4, cfg)
        for sid in confirmed:
            r = by_setup(alt, sid)
            cats = {x["category"] for x in r.opportunities[-1].get("all_rejection_reasons") or []}
            assert r.state == "REJECTED" and cat in cats, (cat, sid, r.end_reason)


def test_entry_conflict_and_quality_families_are_reported(up):
    for c in accepted_cand(up[3]):
        p = c.accepted
        assert set(p["entry_quality_families"]) == {"CONFIRMATION", "TIMING", "FRESHNESS", "PRICE_QUALITY",
                                                    "MARKET_QUALITY", "STRUCTURAL_INTEGRITY", "ROOM",
                                                    "EXECUTION_CONDITIONS", "CONFLICT"}
        assert 0 <= p["entry_conflict_score"] <= 100 and 0 <= p["entry_quality_score"] <= 100
        assert p["entry_quality_families"]["CONFLICT"] == pytest.approx(100 - p["entry_conflict_score"])


# ------------------------------------------------------------------ revalidation: H4 & setup
def test_h4_permission_loss_after_confirmation_invalidates(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    j = c.accepted_index
    lost = by_setup(run(edited(h1, "frame", {(j, "h4_permission"): "BLOCK_ALL"}), h4), c.setup_id)
    assert lost.accepted is not None  # the acceptance happened before the change and is not rewritten
    assert lost.state == "INVALIDATED" and lost.end_category == "H4_PERMISSION_REVOKED" and lost.end_index == j
    rev = by_setup(run(edited(h1, "frame", {(c.confirmation_index, "h4_permission"): "ALLOW_SHORT"}), h4), c.setup_id)
    assert rev.state == "INVALIDATED" and rev.end_category == "H4_PERMISSION_REVOKED" and rev.accepted is None
    stale = by_setup(run(edited(h1, "frame", {(c.confirmation_index, "h4_context_status"): "STALE"}), h4), c.setup_id)
    assert stale.state == "INVALIDATED" and "H4 context STALE" in stale.end_reason


def test_phase1c_setup_invalidation_or_expiry_stops_entry_tracking(up, down):
    n = 0
    for _, _, h1, er in (up, down):
        for c in er.candidates:
            if c.end_category in ("SETUP_INVALIDATED", "SETUP_EXPIRED"):
                n += 1
                stp = next(s for s in h1.setups if s.setup_id == c.setup_id)
                assert c.end_index == stp.end_index and stp.state_at(c.end_index) in ("INVALIDATED", "EXPIRED")
                assert c.transitions[-1][0] == c.end_index
    assert n


def test_opposing_structure_break_invalidates(up):
    _, h4, h1, er = up
    c = next(x for x in er.candidates if x.confirmation is not None)
    ci = c.confirmation_index
    ev = next(b for b in h1.structure.breaks if b.direction == "bearish")
    fake = replace(ev, break_index=ci, event_id=10**6)
    alt_structure = replace(h1.structure, breaks=list(h1.structure.breaks) + [fake])
    r = by_setup(run(replace(h1, structure=alt_structure), h4), c.setup_id)
    assert r.state == "INVALIDATED" and r.end_reason.startswith("OPPOSING_STRUCTURE_BREAK") and r.end_index == ci


# ------------------------------------------------------------------ weekend / reopen
class ClosureQuotes:
    """Pretends the market closed between the close of bar ``k`` and the next price."""

    def __init__(self, base: BarOpenQuotes, k: int):
        self.base, self.k = base, k

    def quote_after(self, index):
        q = self.base.quote_after(index)
        if index == self.k and q is not None:
            return ExecutableQuote(q.time + pd.Timedelta(hours=49), q.price, q.spread, q.spread_source, q.source)
        return q


def test_confirmation_is_not_carried_over_a_market_closure(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    ci = c.confirmation_index
    r = by_setup(run(h1, h4, quotes=ClosureQuotes(BarOpenQuotes(h1.features), ci)), c.setup_id)
    first = r.opportunities[0]
    assert first["decision"] == "DEFER" and "WEEKEND_REOPEN_REVALIDATION" in first["reason_codes"]
    assert r.confirmation_history and r.confirmation_history[0]["index"] == ci
    back = [t for t in r.transitions if t[3] == "DEFERRED" and t[4] == "WAITING_FOR_CONFIRMATION"]
    assert back and "post-reopen" in back[0][5]
    if r.confirmation is not None:
        assert r.confirmation_index >= r.reconfirm_after


def test_reopen_bars_cannot_confirm_and_friday_cutoff(up, down):
    for _, _, _, er in (up, down):
        f = er.frame
        assert f["reopen_bar"].sum() > 0 and (f.loc[f["reopen_bar"], "timestamp"].dt.dayofweek == 0).all()
        guard = set(np.flatnonzero(f["reopen_guard"].to_numpy()))
        assert all(c.confirmation_index not in guard for c in er.candidates if c.confirmation is not None)
    _, h4, h1, er = up
    fri = [c for c in accepted_cand(er) if pd.Timestamp(c.confirmation["at"]).dayofweek == 4]
    assert fri
    c = fri[0]
    hour = pd.Timestamp(c.confirmation["at"]).hour
    cfg = EntryConfig(gap=replace(EntryConfig().gap, friday_cutoff_hour_utc=hour))
    r = by_setup(run(h1, h4, cfg), c.setup_id)
    assert r.opportunities[0]["decision"] == "DEFER" and "PRE_WEEKEND_CUTOFF" in r.opportunities[0]["reason_codes"]
    assert r.accepted is None


def _carryover_world(h1, side="long"):
    """Synthetic edge case: a setup qualified late on a Friday that Phase 1C keeps QUALIFIED over the weekend.

    (With default Phase 1C settings this does not happen naturally - the stale H4 context after the weekend already
    invalidates setups - so the Phase 1D protection is exercised on an edited copy.)"""
    f = h1.frame
    reopen = [i for i in range(1, len(f)) if f["timestamp"].iloc[i] - f["available_at"].iloc[i - 1] > pd.Timedelta(hours=2)]
    k = next(r for r in reopen if r > 1300)
    q = k - 3  # Friday 21:00 bar
    src = next(s for s in h1.setups if s.side == side and s.qualified_index is not None)
    stp = replace(src, setup_id=f"H1-{side.upper()}-CARRY-{q}", qualified_index=q,
                  transitions=[(q, "x", "NO_SETUP", "QUALIFIED", "edited: qualified before the weekend")],
                  end_index=None, end_reason=None, state="QUALIFIED",
                  invalidation_reference=float(h1.features["low"].iloc[q - 30:q + 30].min()) - 5.0)
    frame = f.copy()
    for i in range(q, q + 24):
        frame.at[i, "h4_context_status"] = "OK"
        frame.at[i, "h4_permission"] = "ALLOW_LONG"
        frame.at[i, "h1_blockers"] = []
    return replace(h1, frame=frame, setups=[stp]), stp, k


def test_weekend_carryover_setup_requires_post_reopen_confirmation(up):
    _, h4, h1, _ = up
    world, stp, reopen = _carryover_world(h1)
    base = EntryConfig()
    patient = replace(base, lifecycle=replace(base.lifecycle, max_wait_bars=40, run_away_atr=99.0))
    er = run(world, h4, patient)
    c = er.candidates[0]
    assert c.weekend_carryover and c.reconfirm_after == reopen + 1
    assert any(step == "WEEKEND/REOPEN REVALIDATION" for _, _, _, step, _ in c.audit)
    assert er.frame["reopen_bar"].iloc[reopen] and er.frame["reopen_guard"].iloc[reopen]
    if c.confirmation is not None:  # only post-reopen evidence may confirm
        assert c.confirmation_index >= reopen + 1
    # the pre-weekend bars alone could not have produced an entry: nothing is decided before the reopen
    assert all(o["index"] > reopen for o in c.opportunities)
    never = replace(patient, policy=replace(base.policy, policies=tuple((f, a, 100.1, 0.0) for f, a, _, _ in
                                                                        base.policy.policies)))
    nc = run(world, h4, never).candidates[0]
    assert nc.weekend_carryover and nc.confirmation is None and nc.accepted is None


# ------------------------------------------------------------------ news / session / slippage / intrabar
def test_news_unknown_is_recorded_not_blocking_unless_configured(up):
    _, h4, h1, er = up
    assert all(p["news_status"] == "UNKNOWN" and "NEWS_STATUS_UNKNOWN" in p["reason_codes"] for p in er.accepted)
    strict = EntryConfig(news=replace(EntryConfig().news, block_unknown=True))
    alt = run(h1, h4, strict)
    assert not alt.accepted and any(c.end_category == "NEWS" for c in alt.candidates)

    class Empty:
        def events(self, start, end, known_at):
            return []

    clear = run(h1, h4, news=Empty())
    assert clear.accepted and all(p["news_status"] == "CLEAR" for p in clear.accepted)


def test_intrabar_provider_is_optional_research_evidence_only(up):
    _, h4, h1, er = up
    calls = []

    class Provider:
        def confirm(self, direction, level, start, end, as_of):
            calls.append((start, end, as_of))
            return IntrabarConfirmation(True, None, {"bars": 12}, "fake_m5")

    alt = run(h1, h4, intrabar=Provider())
    assert calls and all(as_of == end for _, end, as_of in calls)
    assert [c.to_dict()["opportunities"] for c in alt.candidates] == [c.to_dict()["opportunities"] for c in er.candidates]
    assert all(c.confirmation["intrabar"]["source"] == "fake_m5" for c in alt.candidates if c.confirmation)
    assert all(c.confirmation["intrabar"]["source"] == "none" for c in er.candidates if c.confirmation)


# ------------------------------------------------------------------ edge case: perfect-looking setup, no confirmation
def test_perfect_looking_setup_without_confirmation_never_becomes_a_candidate(up):
    _, h4, h1, _ = up
    base = EntryConfig()
    none = replace(base, policy=replace(base.policy, policies=tuple((f, a, 100.1, 0.0) for f, a, _, _ in base.policy.policies)),
                   lifecycle=replace(base.lifecycle, max_wait_bars=2, run_away_atr=99.0))
    er = run(h1, h4, none)
    assert not er.accepted
    no_conf = [c for c in er.candidates if c.end_category == "NO_CONFIRMATION"]
    assert no_conf and all(c.opportunities == [] for c in no_conf)
    assert all(x["category"] == "NO_CONFIRMATION" and x["confirmed"] is False
               for x in er.counterfactual if x["setup_id"] in {c.setup_id for c in no_conf})


# ------------------------------------------------------------------ counterfactuals, audit, codes, storage
def test_rejected_and_ended_candidates_are_kept_as_counterfactuals(up, down):
    for _, _, _, er in (up, down):
        ended = [c for c in er.candidates if c.accepted is None]
        assert sorted(x["entry_candidate_id"] for x in er.counterfactual) == sorted(c.entry_candidate_id for c in ended)
        for x in er.counterfactual:
            assert x["category"] and x["reason"] and x["end_code"] in VALID
            if x["confirmed"] and x["end_state"] == "REJECTED":
                assert x["hypothetical"]["executable_reference_price"] is not None
                assert x["hypothetical"]["all_rejection_reasons"]


def test_audit_trail_explains_accepted_and_rejected_candidates(up):
    _, _, _, er = up
    acc = accepted_cand(er, "long")[0]
    steps = [a[3] for a in acc.audit]
    order = ["SETUP QUALIFIED", "WAITING FOR CONFIRMATION", "SPREAD CHECK", "ROOM CHECK PASSED", "CHASE CHECK PASSED",
             "ENTRY CANDIDATE ACCEPTED"]
    pos = [next(i for i, s in enumerate(steps) if s.startswith(o)) for o in order]
    assert pos == sorted(pos)
    assert any(s in ("BULLISH STRUCTURAL BREAK", "DISPLACEMENT CONFIRMED", "MOMENTUM REACCELERATION",
                     "BREAK-RETEST CONFIRMED") for s in steps)
    text = explain_entry_candidate(er, acc.entry_candidate_id)
    assert "ENTRY CANDIDATE ACCEPTED" in text and "not an order" in text and "ASK" in text
    rej = next(c for c in er.candidates if c.state == "REJECTED")
    rsteps = [a[3] for a in rej.audit]
    assert any(s.endswith("CHECK FAILED") for s in rsteps) and rsteps[-1] == "ENTRY REJECTED"
    assert "LAST DECISION: REJECT" in explain_entry_candidate(er, rej.entry_candidate_id)


def test_reason_codes_are_valid_everywhere(up, down, entry_results):
    for _, _, _, er in entry_results.values():
        for codes in er.frame["reason_codes"]:
            assert set(codes) <= VALID, set(codes) - VALID
        for c in er.candidates:
            for o in c.opportunities:
                assert set(o["reason_codes"]) <= VALID, set(o["reason_codes"]) - VALID
        for x in er.counterfactual:
            assert set(x["reason_codes"]) <= VALID


def test_research_frame_stores_every_confirmation_family_independently(up):
    _, _, h1, er = up
    f = er.frame
    assert len(f) == len(h1.frame) and list(f["timestamp"]) == list(h1.frame["timestamp"])
    for side in ("long", "short"):
        for fam in CONFIRMATION_FAMILIES:
            col = f"{side}_conf_{SHORT_NAME[fam]}_score"
            assert col in f.columns and f[col].between(0, 100).all()
    c = accepted_cand(er, "long")[0]
    row = f.iloc[c.confirmation_index]
    assert row["long_confirmation_family"] == c.confirmation["family"]
    assert "CONFIRMED" in row["long_entry_events"]
    assert f["long_entry_state"].iloc[c.accepted_index] in ("ENTRY_CANDIDATE", "EXPIRED", "INVALIDATED")


def test_fail_safe_invalidates_active_candidates(up):
    _, h4, h1, er = up
    c = accepted_cand(er, "long")[0]
    bad = EntryIntelligenceEngine().run(h1, h4, _fault_at=c.confirmation_index)
    r = by_setup(bad, c.setup_id)
    assert r.state == "INVALIDATED" and r.end_category == "ENTRY_CONTEXT_ERROR" and bad.errors
    tail = bad.frame.iloc[c.confirmation_index:]
    assert (tail["entry_error"] == "ENTRY_CONTEXT_ERROR").all()
    assert not [x for x in bad.accepted if x["decision_bar_index"] > c.confirmation_index]
