"""Phase 1E pipeline integration: only accepted entry candidates, revalidation, proposal object, audit, retention,
direction statistics, volatility/session/news storage, account independence, MAE/MFE boundary."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine.reason_codes import ReasonCode
from gbpjpy_engine.trade import TRADE_STATES, TradeConstructionEngine, explain_trade
from gbpjpy_engine.trade.outcomes import compute_excursions

VALID = {c.value for c in ReasonCode}
PKG = Path(gbpjpy_engine.__file__).parent / "trade"
REQUIRED = ("trade_proposal_id", "entry_candidate_id", "setup_id", "timestamp", "symbol", "direction", "setup_family",
            "confirmation_family", "executable_reference_price", "stop_reference_type", "raw_invalidation_price",
            "stop_buffer", "proposed_stop_price", "stop_distance_pips", "stop_distance_atr", "stop_noise_risk_score",
            "stop_quality_score", "candidate_targets", "primary_target", "gross_R", "estimated_net_R",
            "target_reachability_score", "barrier_density_score", "trade_construction_quality_score",
            "trade_construction_conflict_score", "h4_permission", "setup_score", "entry_quality_score", "reason_codes",
            "cost_assumptions", "broker_constraints_status")
FORBIDDEN = ("lot", "lots", "lot_size", "volume", "position_size", "currency_risk", "risk_amount", "risk_percent",
             "percent_risk", "leverage", "account_balance", "equity", "margin", "ticket", "order_id")
LEGAL = {"NO_TRADE_CONSTRUCTION": {"EVALUATING_STOP", "INVALIDATED", "EXPIRED"},
         "EVALUATING_STOP": {"EVALUATING_TARGETS", "REJECTED", "INVALIDATED"},
         "EVALUATING_TARGETS": {"EVALUATING_RISK_REWARD", "REJECTED"},
         "EVALUATING_RISK_REWARD": {"PROPOSED", "REJECTED", "INVALIDATED"},
         "PROPOSED": {"EXPIRED", "INVALIDATED"}}


def all_recs(trade_results):
    return [(name, rec) for name, (*_, tr) in trade_results.items() for rec in tr.constructions]


def test_only_accepted_entry_candidates_reach_construction(trade_results):
    for _, (_, _, _, er, tr) in trade_results.items():
        assert sorted(r.entry_candidate_id for r in tr.constructions) == \
            sorted(c.entry_candidate_id for c in er.candidates if c.accepted is not None)
        assert all(r.trade_proposal_id == f"P-{r.entry_candidate_id}" for r in tr.constructions)
        assert len(tr.proposals) + len(tr.rejected) == len(tr.constructions)


def test_state_machine_transitions_are_legal_and_logged(trade_results):
    for _, rec in all_recs(trade_results):
        prev = "NO_TRADE_CONSTRUCTION"
        for idx, at, frm, to, reason in rec.transitions:
            assert frm == prev and to in LEGAL[frm] and to in TRADE_STATES and reason and at
            prev = to
        assert rec.state_at(rec.transitions[-1][0]) == rec.state


def test_long_and_short_proposals_with_complete_objects(trade_results):
    sides = set()
    for _, (_, _, h1, er, tr) in trade_results.items():
        for p in tr.proposals:
            sides.add(p["direction"])
            assert all(k in p for k in REQUIRED)
            assert not any(k in p for k in FORBIDDEN)
            cand = next(c for c in er.candidates if c.entry_candidate_id == p["entry_candidate_id"])
            assert p["executable_reference_price"] == cand.accepted["executable_reference_price"]
            d = 1 if p["direction"] == "LONG" else -1
            assert d * (p["executable_reference_price"] - p["proposed_stop_price"]) > 0
            assert d * (p["primary_target"]["price"] - p["executable_reference_price"]) > 0
            one_r = p["risk_unit"]["one_R_price"]
            assert one_r == pytest.approx(abs(p["executable_reference_price"] - p["proposed_stop_price"]), abs=1e-9)
            assert p["gross_R"] == pytest.approx(abs(p["primary_target"]["price"] - p["executable_reference_price"]) / one_r,
                                                 abs=1e-4)
            assert p["estimated_net_R"] < p["gross_R"] and p["gross_R"] >= p["minimum_net_R_required"]
            assert set(p["reason_codes"]) <= VALID and "TRADE_PROPOSED" in p["reason_codes"]
            assert p["broker_constraints_status"] == "UNKNOWN" and p["news_status"] == "UNKNOWN" and "NEWS_UNKNOWN" in p["reason_codes"]
            assert p["volatility"]["h1_regime"] and p["volatility"]["h4_regime"] and p["volatility"]["h1_atr_pips"] > 0
            assert p["session_context"] == cand.accepted["session_context"]
            assert p["symbol_spec"]["pip_size"] == 0.01 and p["proposed_stop_price"] == round(p["proposed_stop_price"], 3)
    assert sides == {"LONG", "SHORT"}


def test_h4_permission_revoked_before_proposal_invalidates(trade_results):
    _, h4, h1, er, tr = trade_results["entry_uptrend"]
    p = tr.proposals[0]
    rec = tr.construction(p["trade_proposal_id"])
    frame = h1.frame.copy()
    frame.at[rec.index - 1, "h4_permission"] = "BLOCK_ALL"
    alt = TradeConstructionEngine().run(er, replace(h1, frame=frame), h4)
    r = alt.construction(p["trade_proposal_id"])
    assert r.decision == "INVALIDATE" and r.category == "H4_PERMISSION_CHANGED" and "H4_PERMISSION_REVOKED" in r.reason_codes
    assert r.proposal is None and any(x["trade_proposal_id"] == r.trade_proposal_id for x in alt.rejected)


def test_setup_and_entry_revalidation(trade_results):
    _, h4, h1, er, tr = trade_results["entry_downtrend"]
    p = tr.proposals[0]
    rec = tr.construction(p["trade_proposal_id"])
    frame = h1.frame.copy()
    frame.at[rec.index - 1, "h1_blockers"] = ["SEVERE_H1_CHOP"]
    r = TradeConstructionEngine().run(er, replace(h1, frame=frame), h4).construction(p["trade_proposal_id"])
    assert r.decision == "INVALIDATE" and r.category == "MARKET_STATE"
    # proposals follow their entry candidate: when the candidate lapses the proposal lapses too (append-only)
    cand = next(c for c in er.candidates if c.entry_candidate_id == p["entry_candidate_id"])
    assert rec.decision == "PROPOSE_TRADE" and rec.state == cand.state and rec.end_index == cand.end_index
    to = [t[3] for t in rec.transitions]
    assert to[-2:] == ["PROPOSED", cand.state] and rec.transitions[-1][0] == cand.end_index >= rec.index


def test_rejections_are_retained_with_research_records(trade_results):
    for _, (*_, tr) in trade_results.items():
        for x in tr.rejected:
            assert x["category"] and x["reason"] and set(x["reason_codes"]) <= VALID
            assert "filter adds value" in x["note"]


def test_direction_statistics_are_kept_separately(trade_results):
    for _, (*_, tr) in trade_results.items():
        st = tr.direction_stats()
        assert set(st) == {"LONG", "SHORT"}
        assert st["LONG"]["constructions"] + st["SHORT"]["constructions"] == len(tr.constructions)


def test_audit_answers_every_question(trade_results):
    _, _, _, _, tr = trade_results["entry_uptrend"]
    text = explain_trade(tr, tr.proposals[0]["trade_proposal_id"])
    for q in ("WHY THIS STOP?", "WHY IS THE STOP NOT CLOSER?", "WHY THIS TARGET?", "WHY IS THE TARGET REALISTIC?",
              "WHAT BARRIERS EXIST?", "GROSS R", "ESTIMATED NET R", "WHAT IS UNKNOWN?", "WHY PROPOSE_TRADE?", "not an order"):
        assert q in text, q


def test_account_independence_and_no_sizing(trade_results):
    sig = inspect.signature(TradeConstructionEngine.run)
    assert not {"account", "balance", "equity", "risk_percent", "leverage"} & set(sig.parameters)
    assert {n for n in dir(TradeConstructionEngine) if not n.startswith("_")} == {"run"}
    _, h4, h1, er, tr = trade_results["entry_uptrend"]
    again = TradeConstructionEngine().run(er, h1, h4)
    assert again.proposals == tr.proposals  # nothing account-dependent exists to vary


def test_mae_mfe_only_after_construction(trade_results):
    for py in PKG.rglob("*.py"):
        if py.name == "outcomes.py":
            continue
        tree = ast.parse(py.read_text())
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        assert "outcomes" not in mods and ".outcomes" not in mods, py.name
        assert "compute_excursions" not in py.read_text() and "mae" not in {x.lower() for x in py.read_text().split()}
    sc, _, h1, _, tr = trade_results["entry_uptrend"]
    p = tr.proposals[0]
    t = pd.Timestamp(p["timestamp"])
    later = sc.h1[sc.h1["timestamp"] >= t].head(24)
    o = compute_excursions(p, later)
    assert o.bars_evaluated == 24 and o.mae_R >= 0 and o.mfe_R >= 0
    with pytest.raises(ValueError):
        compute_excursions(p, sc.h1[sc.h1["timestamp"] < t].tail(5))
