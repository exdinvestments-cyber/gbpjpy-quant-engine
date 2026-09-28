"""Phase 1E synthetic edge cases and the critical anti-R:R-manipulation tests (constructed contexts)."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest

from gbpjpy_engine.trade import TradeConfig
from gbpjpy_engine.trade import stops as stops_mod
from gbpjpy_engine.trade import targets as targets_mod
from gbpjpy_engine.trade.config import AsymmetryConfig
from trade_helpers import build, lvl, make_ctx

POOR_R = [lvl(190.60, 80.0, "H1"), lvl(190.65, 70.0, "H4")]  # only structure: 0.86 R


def with_min_r(v: float) -> TradeConfig:
    base = TradeConfig()
    return replace(base, asymmetry=AsymmetryConfig(min_net_r=tuple((f, v) for f, _ in base.asymmetry.min_net_r)))


def geometry(rec):
    p = rec.proposal
    return (p["stop_reference_type"], p["raw_invalidation_price"], p["proposed_stop_price"],
            [(t["label"], t["price"], t["gross_R"]) for t in p["candidate_targets"]],
            p["primary_target"]["label"] if p["primary_target"] else None)


# ------------------------------------------------------------------ synthetic edge cases
def test_excellent_entry_sensible_stop_clear_target_is_proposed():
    for d in (1, -1):
        r = build(make_ctx(d))
        p = r.proposal
        assert r.decision == "PROPOSE_TRADE" and r.state == "PROPOSED"
        assert [t[3] for t in r.transitions] == ["EVALUATING_STOP", "EVALUATING_TARGETS", "EVALUATING_RISK_REWARD", "PROPOSED"]
        assert p["primary_target"]["path"]["barrier_count"] == 0 and "TARGET_PATH_CLEAR" in r.reason_codes
        assert p["gross_R"] >= p["estimated_net_R"] >= p["minimum_net_R_required"]
        assert p["risk_unit"]["one_R_pips"] == pytest.approx(66.0)


def test_long_short_symmetry_of_geometry():
    lp, sp = build(make_ctx(1)).proposal, build(make_ctx(-1)).proposal
    for k in ("stop_distance_pips", "gross_R", "estimated_net_R"):
        assert lp[k] == pytest.approx(sp[k])
    assert lp["proposed_stop_price"] < lp["executable_reference_price"] < lp["primary_target"]["price"]
    assert sp["proposed_stop_price"] > sp["executable_reference_price"] > sp["primary_target"]["price"]


def test_excellent_entry_with_enormous_structural_stop_is_rejected():
    ctx = make_ctx(entry_quality=90.0, stop_refs={"STRUCTURAL_INVALIDATION": {"price": 188.50, "reason": "deep origin"}})
    r = build(ctx)
    assert r.decision == "REJECT_TRADE" and r.category == "STOP_TOO_WIDE" and "STOP_DISTANCE_EXCESSIVE" in r.reason_codes
    assert r.proposal["volatility_adequacy"] == "EXTREME" and r.proposal["stop_reference_price"] == 188.50


def test_no_realistic_target_is_rejected():
    assert build(make_ctx(levels=[])).category == "NO_REALISTIC_TARGET"
    far = build(make_ctx(levels=[lvl(199.00, 90.0, "H4")]))
    assert far.category == "TARGET_UNREALISTIC" and "TARGET_UNREALISTIC" in far.reason_codes
    wrong_side = build(make_ctx(levels=[lvl(188.00, 90.0), lvl(float("nan"), 90.0)]))  # below a long entry / NaN
    assert wrong_side.category == "NO_REALISTIC_TARGET" and wrong_side.proposal["candidate_targets"] == []


def test_attractive_nominal_r_but_barrier_congestion_is_rejected():
    ctx = make_ctx(levels=[lvl(190.10, 90.0, "H1"), lvl(190.12, 85.0, "H4", "h4_strong_resistance_zone"),
                           lvl(191.80, 80.0), lvl(191.85, 70.0, "H4"), lvl(192.90, 60.0, "H4")])
    r = build(ctx)
    lad = r.proposal["candidate_targets"]
    assert max(t["gross_R"] for t in lad) > 2.0  # nominally attractive
    assert all(t["path"]["congested"] for t in lad)
    assert r.decision == "REJECT_TRADE" and r.category == "BARRIER_CONGESTION" and "TARGET_PATH_CONGESTED" in r.reason_codes


def test_structural_target_with_poor_r_is_rejected():
    r = build(make_ctx(levels=POOR_R))
    assert r.decision == "REJECT_TRADE" and r.category == "INSUFFICIENT_RR" and "ASYMMETRY_INSUFFICIENT" in r.reason_codes
    assert r.proposal["gross_R"] < 1.5


def test_tight_stop_inside_ordinary_noise_is_rejected():
    ctx = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 189.80, "reason": "shallow pullback low"}},
                   chop=70.0, recent_extremes=[189.77] * 50, adverse_excursions=[0.30] * 40)
    r = build(ctx)
    assert r.decision == "REJECT_TRADE" and r.category == "STOP_INSIDE_NOISE" and "STOP_INSIDE_NOISE" in r.reason_codes
    assert r.proposal["stop_noise_risk_score"] >= 70


def test_costs_degrade_rr_and_spread_is_embedded():
    ctx = make_ctx(levels=[lvl(191.038, 80.0), lvl(191.05, 70.0, "H4")])
    r = build(ctx)
    p = r.proposal
    assert p["gross_R"] >= 1.5 > p["estimated_net_R"]
    assert r.category == "EXECUTION_COSTS" and "COSTS_DEGRADE_RR" in r.reason_codes
    cheap = build(make_ctx(levels=[lvl(191.038, 80.0), lvl(191.05, 70.0, "H4")], slippage_pips=0.0, slippage_status="MODELLED"),
                  costs=__import__("gbpjpy_engine.trade", fromlist=["FixedCommission"]).FixedCommission(0.01))
    assert cheap.proposal["estimated_net_R"] > p["estimated_net_R"]
    narrow, wide = build(make_ctx(-1)).proposal, build(make_ctx(-1, spread_pips=8.0)).proposal
    assert wide["gross_R"] < narrow["gross_R"]  # a SHORT pays the spread on its stop and target side


def test_unknown_transaction_costs_are_explicit():
    r = build(make_ctx())
    ca = r.proposal["cost_assumptions"]
    assert ca["status"] == "PARTIALLY_UNKNOWN" and "COSTS_UNKNOWN" in r.reason_codes
    assert {v["status"] for v in ca["components"].values()} >= {"UNKNOWN_ASSUMED", "UNKNOWN"}
    assert r.proposal["estimated_net_R"] < r.proposal["gross_R"]


def test_zero_distance_stop_and_numeric_corruption_reject_safely():
    cfg = replace(TradeConfig(), stop=replace(TradeConfig().stop, buffer_atr=0.0))
    at_entry = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 190.020, "reason": "at entry"}}, adverse_wicks=[0.0] * 5)
    r = build(at_entry, cfg)
    assert r.decision == "REJECT_TRADE" and r.category in ("NO_STRUCTURAL_STOP", "INVALID_STOP")
    tiny = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 190.019, "reason": "tick below"}}, adverse_wicks=[0.0] * 5,
                    structural_levels=[])
    t = build(tiny, cfg)
    assert t.category == "INVALID_STOP" and "STOP_BELOW_MINIMUM_DISTANCE" in t.reason_codes
    for bad in (dict(atr=float("nan")), dict(entry_price=float("inf")), dict(atr=0.0), dict(spread_pips=float("nan"))):
        x = build(make_ctx(**bad))
        assert x.decision == "REJECT_TRADE" and x.category == "NUMERIC_INVALID" and x.proposal is None
    assert build(make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": float("nan"), "reason": "corrupt"}})).category == \
        "NO_STRUCTURAL_STOP"
    nan_ref = build(make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": float("nan"), "reason": "corrupt"},
                                        "SWING_INVALIDATION": {"price": 189.55, "reason": "last swing"}}))
    assert nan_ref.decision == "PROPOSE_TRADE" and nan_ref.proposal["stop_reference_type"] == "SWING_INVALIDATION"
    assert np.isfinite(nan_ref.proposal["proposed_stop_price"])


def test_thesis_recheck_before_finalising_invalidates():
    r = build(make_ctx(), final_checks={"h4_permission_valid": False})
    assert r.decision == "INVALIDATE" and r.state == "INVALIDATED" and "THESIS_FAILED_BEFORE_ENTRY" in r.reason_codes
    lost = build(make_ctx(last_close=189.30))  # the last close already sits beyond the invalidation level
    assert lost.state == "INVALIDATED" and lost.proposal["final_checks"]["stop_thesis_relevant"] is False


# ------------------------------------------------------------------ CRITICAL: no R:R manufacturing
def test_minimum_r_never_changes_stop_or_targets_only_the_decision():
    ctx = make_ctx(levels=POOR_R + [lvl(192.90, 60.0, "H4")])
    results = {v: build(ctx, with_min_r(v)) for v in (0.25, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 10.0)}
    geos = {v: geometry(r) for v, r in results.items()}
    assert len({repr(g) for g in geos.values()}) == 1  # identical stop, targets and primary for every threshold
    gross = results[1.5].proposal["estimated_net_R"]
    for v, r in results.items():
        assert r.decision == ("PROPOSE_TRADE" if gross >= v else "REJECT_TRADE"), v


def test_a_closer_stop_that_would_pass_is_never_used():
    tight_swing = {"price": 189.85, "reason": "closer swing that would give 3R"}
    ctx = make_ctx(levels=POOR_R, stop_refs={"STRUCTURAL_INVALIDATION": {"price": 189.40, "reason": "pullback extreme"},
                                            "SWING_INVALIDATION": tight_swing, "ZONE_INVALIDATION": tight_swing})
    r = build(ctx)
    assert r.proposal["stop_reference_type"] == "STRUCTURAL_INVALIDATION" and r.proposal["raw_invalidation_price"] == 189.40
    assert r.decision == "REJECT_TRADE" and r.category == "INSUFFICIENT_RR"


def test_a_farther_target_that_would_pass_is_never_substituted():
    ctx = make_ctx(levels=POOR_R + [lvl(199.00, 95.0, "H4")])  # distant level offering ~13R but unrealistic
    r = build(ctx)
    far = next(t for t in r.proposal["candidate_targets"] if t["structural_level"] == 199.00)
    assert far["gross_R"] > 10 and not far["realistic"]
    assert r.proposal["primary_target"]["structural_level"] == 190.60
    assert r.decision == "REJECT_TRADE" and r.category == "INSUFFICIENT_RR"


def test_stop_and_targets_never_move_in_the_r_improving_direction():
    for d in (1, -1):
        for ctx in (make_ctx(d), make_ctx(d, spread_pips=5.0), make_ctx(d, adverse_wicks=[0.2] * 50)):
            p = build(ctx).proposal
            assert d * (p["raw_invalidation_price"] - p["proposed_stop_price"]) > 0  # buffer only widens
            for t in p["candidate_targets"]:
                assert d * (t["structural_level"] - t["price"]) > 0  # targets sit before their level


def test_stop_and_target_code_cannot_see_r_requirements():
    for mod in (stops_mod, targets_mod):
        src = inspect.getsource(mod)
        for token in ("min_net_r", "asymmetry", "gross_R", "estimated_net_R", "risk_distance"):
            assert token not in src, (mod.__name__, token)
    sig = inspect.signature(targets_mod.build_ladder)
    assert "stop" not in " ".join(sig.parameters)
