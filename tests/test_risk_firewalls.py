"""Phase 1F firewalls and stress: score/risk separation, no loss escalation, loss & win sequences, the £500k firewall,
look-ahead, determinism, pipeline integration, research mode, portability."""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
from dataclasses import replace
from decimal import Decimal as Dec
from pathlib import Path

import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine.risk import (AccountRiskEngine, AccountStateSource, ClosedTradeResult, ContractSpecSource, ConversionRate,
                                ExternalRiskPolicy, MarginModel, RateProvider, ResearchAccount, ResearchRunner, RiskOfRuinInputs,
                                RiskPolicy, StaticRates, compare_policies, monte_carlo_records, rates_from_bars, risk_of_ruin,
                                risk_policy_from_dict)
from gbpjpy_engine.risk.account import ClosedResultSink
from risk_helpers import T0, account, engine, policy, proposal, rates

PKG = Path(gbpjpy_engine.__file__).parent / "risk"
ROOT = Path(gbpjpy_engine.__file__).parents[2]


def ok(d):
    return d["decision"] == "RISK_APPROVED"


def names_and_strings(py: Path):
    tree = ast.parse(py.read_text())
    docs = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef))
            and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.keyword) and n.arg:
            out.add(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            out.add(n.value)
    return out


# ------------------------------------------------------------------ score / risk firewall
def test_quality_scores_cannot_change_exposure():
    base = proposal()
    variants = [dict(base, setup_score=s, entry_quality_score=s, trade_construction_quality_score=s, confidence=s)
                for s in (20.0, 60.0, 80.0, 95.0, 100.0)]
    outs = [engine().approve(v, account(), T0)["approved_trade"] for v in variants]
    assert len({(o["approved_volume"], o["risk_amount"], o["actual_risk_currency"]) for o in outs}) == 1
    for py in PKG.rglob("*.py"):
        bad = [t for t in names_and_strings(py) if re.search(r"score|quality|confidence", t, re.IGNORECASE)]
        assert not bad, (py.name, bad)


def test_no_multiplier_can_increase_risk():
    for bad in ({"drawdown": {"tiers": (("NORMAL", 0.0, 1.2),)}}, {"consecutive": {"reduce_factor": 1.5}},
                {"events": {"news_reduce_factor": 2.0}}):
        with pytest.raises(ValueError):
            policy(**bad).validate()


# ------------------------------------------------------------------ loss / win sequences
def weekdays(start, n):
    out, t = [], start
    while len(out) < n:
        if t.dayofweek < 5:
            out.append(t)
        t += pd.Timedelta(days=1)
    return out


def run_losses(pol, n_days=60, per_day=10):
    days = weekdays(pd.Timestamp("2024-01-03 10:00", tz="UTC"), n_days)
    eng = engine(pol, rate_provider=rates(ts=days[0] - pd.Timedelta(hours=1), span_hours=24 * 95))
    run = ResearchRunner(eng, ResearchAccount("SIM", "GBP", Dec("10000"), 30))
    pcts, cats, states, k = [], set(), set(), 0
    for day in days:
        for h in range(per_day):
            t = day + pd.Timedelta(hours=h)
            pid = f"P-{k}"
            k += 1
            d = run.propose(proposal(pid=pid, ts=t), t)
            if ok(d):
                a = d["approved_trade"]
                caps = d["audit"]["risk_budget"]["caps"]
                policy_cap = min(Dec(caps[c]) for c in ("PER_TRADE_CAP", "MAX_SINGLE_TRADE_CAP", "DRAWDOWN_CAP",
                                                        "CONSECUTIVE_LOSS_CAP") if c in caps)
                pcts.append((policy_cap / Dec(a["sizing_capital"])).quantize(Dec("0.0001")))
                assert Dec(a["actual_risk_currency"]) <= policy_cap
                states.add(a["drawdown_state"])
                run.close_in_r(pid, -1.0, t + pd.Timedelta(minutes=30))  # research-supplied synthetic loss
            else:
                cats.add(d["rejection_category"])
    return eng, pcts, cats, states


def test_loss_sequence_stress_never_escalates_and_protections_activate():
    eng, pcts, cats, states = run_losses(policy(consecutive={"pause_after": 0}))
    assert pcts and all(b <= a for a, b in zip(pcts, pcts[1:]))  # the policy risk % never increases after losses
    assert max(pcts) <= Dec("0.005")
    assert {"DAILY_LIMIT", "WEEKLY_LIMIT", "MONTHLY_LIMIT", "DRAWDOWN_HALT", "GLOBAL_HALT"} <= cats, cats
    assert {"NORMAL", "CAUTION", "DEFENSIVE"} <= states and eng.state().global_state == "HALTED"
    eng2, pcts2, cats2, _ = run_losses(RiskPolicy(), n_days=10)
    assert all(b <= a for a, b in zip(pcts2, pcts2[1:])) and "CONSECUTIVE_LOSS_PAUSE" in cats2


def test_win_sequence_compounds_currency_not_percentage():
    days = weekdays(pd.Timestamp("2024-01-03 10:00", tz="UTC"), 20)
    eng = engine(rate_provider=rates(ts=days[0] - pd.Timedelta(hours=1), span_hours=24 * 30))
    run = ResearchRunner(eng, ResearchAccount("SIM", "GBP", Dec("10000"), 30))
    amounts, pcts = [], []
    for k, t in enumerate(days):
        d = run.propose(proposal(pid=f"W-{k}", ts=t), t)
        a = d["approved_trade"]
        amounts.append(Dec(a["risk_amount"]))
        pcts.append(Dec(a["risk_amount"]) / Dec(a["sizing_capital"]))
        assert d["binding_constraint"] == "PER_TRADE_CAP"
        run.close_in_r(f"W-{k}", 2.0, t + pd.Timedelta(hours=1))
    assert amounts[-1] > amounts[0] and all(b >= a for a, b in zip(amounts, amounts[1:]))  # currency grows with equity
    assert all(Dec("0.00499") <= p <= Dec("0.005") for p in pcts)  # the percentage never rises after wins


# ------------------------------------------------------------------ £500k firewall
def test_profit_objectives_cannot_enter_the_risk_path():
    pat = re.compile(r"(profit|annual|monthly|weekly|daily)_?(target|goal|objective)|target_?(profit|equity|balance|return)"
                     r"|recover|500_?000|ambition|needed_to|required_return", re.IGNORECASE)
    for py in PKG.rglob("*.py"):
        bad = [t for t in names_and_strings(py) if pat.search(t)]
        assert not bad, (py.name, bad)
    for key in ("annual_profit_target", "profit_target", "target_equity", "recovery_amount"):
        with pytest.raises(KeyError):
            risk_policy_from_dict({"sizing": {key: 500000}})
        with pytest.raises(KeyError):
            risk_policy_from_dict({key: {"value": 500000}})
    assert set(inspect.signature(AccountRiskEngine.approve).parameters) == {
        "self", "proposal", "account", "as_of", "construction", "construction_index", "h4_permission_now"}
    p = proposal()
    ANNUAL_OBJECTIVE_GBP = 500_000  # exists only outside the decision path
    before = engine().approve(p, account(), T0)
    after = engine().approve(dict(p, annual_objective=ANNUAL_OBJECTIVE_GBP, needed_per_trade=ANNUAL_OBJECTIVE_GBP / 250),
                             account(), T0)
    keys = ("approved_volume", "risk_amount", "risk_cap_percent", "stop", "primary_target", "entry_reference")
    assert {k: before["approved_trade"][k] for k in keys} == {k: after["approved_trade"][k] for k in keys}
    assert before["approved_trade"]["stop"] == p["proposed_stop_price"]
    assert before["approved_trade"]["primary_target"] == p["primary_target"]["price"]


# ------------------------------------------------------------------ look-ahead & determinism
def test_future_rates_results_and_accounts_cannot_leak():
    fut = rates(extra=[ConversionRate("GBP", "JPY", Dec("100"), T0 + pd.Timedelta(minutes=1), "future")], span_hours=0)
    d = engine(rate_provider=fut).approve(proposal(), account(), T0)
    assert d["audit"]["conversion"]["rates"][0]["rate"] == "190"
    eng = engine()
    eng.record_closed_result(ClosedTradeResult("late", "POS-l", None, Dec("-300"), Dec(0), Dec(0),
                                               T0 + pd.Timedelta(hours=1), "STOP"))
    e = eng.approve(proposal(), account(), T0)
    assert ok(e) and e["audit"]["periods"]["day"]["realised_net"] == "0"  # a result closing after T0 is invisible at T0


def test_deterministic_risk_approval():
    outs = [json.dumps(engine().approve(proposal(), account(), T0), sort_keys=True, default=str) for _ in range(3)]
    assert len(set(outs)) == 1


# ------------------------------------------------------------------ pipeline integration & research mode
def test_real_phase1e_proposals_through_the_risk_engine(trade_results):
    for name, (_, _, h1, _, tr) in trade_results.items():
        r = rates_from_bars(h1.features)
        for rec in tr.constructions:
            if rec.decision != "PROPOSE_TRADE":
                continue
            eng = AccountRiskEngine(rates=r)  # one fresh account per proposal (no carried reservations)
            t = pd.Timestamp(rec.proposal["timestamp"])
            d = eng.approve(rec.proposal, account(ts=t), t, construction=rec, construction_index=rec.index)
            assert ok(d), d["rejection_reason"]
            a = d["approved_trade"]
            assert (a["stop"], a["primary_target"], a["entry_reference"]) == (
                rec.proposal["proposed_stop_price"], rec.proposal["primary_target"]["price"],
                rec.proposal["executable_reference_price"])
            assert Dec(a["actual_risk_currency"]) <= Dec(a["risk_amount"]) and a["account_currency"] == "GBP"
            t_end = pd.Timestamp(rec.transitions[-1][1])  # the proposal lapsed with its entry candidate
            late = eng.approve(dict(rec.proposal, trade_proposal_id=rec.trade_proposal_id + "-x"), account(ts=t_end), t_end,
                               construction=rec, construction_index=rec.end_index)
            assert late["rejection_category"] in ("PROPOSAL_NO_LONGER_VALID", "PROPOSAL_INVALID")


def test_research_mode_compares_policies_on_identical_signals(trade_results):
    _, _, h1, _, tr = trade_results["entry_downtrend"]
    props = [rec.proposal for rec in tr.constructions if rec.decision == "PROPOSE_TRADE"]
    events = []
    for p in props:
        t = pd.Timestamp(p["timestamp"])
        events += [("propose", p, t), ("close_r", p["trade_proposal_id"], -1.0, t + pd.Timedelta(hours=1))]
    r = rates_from_bars(h1.features)
    res = compare_policies(events, {"baseline": AccountRiskEngine(rates=r),
                                    "half": AccountRiskEngine(policy(sizing={"base_risk_percent": 0.25}), rates=r)},
                           lambda: ResearchAccount("SIM", "GBP", Dec("10000"), 30))
    b, h = res["baseline"], res["half"]
    assert [d["trade_proposal_id"] for d in b["decisions"]] == [d["trade_proposal_id"] for d in h["decisions"]]
    for x, y in zip(b["decisions"], h["decisions"]):
        if ok(x) and ok(y):
            assert Dec(y["approved_trade"]["risk_amount"]) < Dec(x["approved_trade"]["risk_amount"])
    recs = monte_carlo_records(b["decisions"])
    assert recs and all("outcomes must come from validated data" in r["note"] for r in recs)
    with pytest.raises(NotImplementedError):
        risk_of_ruin(RiskOfRuinInputs())


# ------------------------------------------------------------------ portability & docs
def test_risk_package_is_portable_and_places_nothing():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    forbidden = {"order_send", "ordersend", "place_order", "submit_order", "send_order", "ticket", "order_ticket",
                 "account_info", "positions_get", "symbol_info_tick", "copy_rates", "metatrader5", "martingale", "grid"}
    for py in PKG.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert {a.name.split(".")[0] for a in n.names} <= allowed, py.name
            elif isinstance(n, ast.ImportFrom) and n.level == 0:
                assert n.module.split(".")[0] in allowed, py.name
        toks = {t.lower() for t in names_and_strings(py)}
        assert not toks & forbidden, (py.name, toks & forbidden)
        assert not [t for t in names_and_strings(py) if re.search(r"\bMT[45]\b", t)], py.name
    for proto in (RateProvider, MarginModel, ExternalRiskPolicy, ContractSpecSource, AccountStateSource, ClosedResultSink):
        assert getattr(proto, "_is_protocol", False), proto
    assert {n for n in dir(AccountRiskEngine) if not n.startswith("_")} >= {"approve", "halt", "reset"}
    assert not {n for n in dir(AccountRiskEngine) if re.search("order|send|execute|close_position", n)}


def test_phase1f_documentation_and_policy_file():
    doc = (ROOT / "docs" / "PHASE_1F_ACCOUNT_RISK.md").read_text().lower()
    for token in ("not an order", "pip value", "round down", "not validated edge", "gap", "fail closed", "persist",
                  "concurrency", "score", "500", "leverage", "known limitations", "adapter"):
        assert token in doc, token
    iface = (ROOT / "docs" / "BROKER_NEUTRAL_INTERFACES.md").read_text()
    assert "## Account risk adapter requirements" in iface
    from gbpjpy_engine.config import describe_config
    from gbpjpy_engine.risk import load_risk_policy

    assert load_risk_policy(ROOT / "config" / "risk_default.yaml").to_dict() == RiskPolicy().to_dict()
    assert all("NOT VALIDATED EDGE" in r["doc"] for r in describe_config(RiskPolicy()))
    assert StaticRates().rate("GBP", "JPY", T0) is None
