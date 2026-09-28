"""Phase 1E unit tests: symbol maths, stops, buffers, adequacy, noise, quality, targets, paths, reachability, R, costs,
broker constraints, management architecture, config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import gbpjpy_engine
from gbpjpy_engine.config import describe_config
from gbpjpy_engine.h1 import H1Config
from gbpjpy_engine.trade import (GBPJPY, BrokerStopConstraints, DisabledPolicy, ExitPlan, FixedCommission, ManagementContext,
                                 SymbolSpec, TradeConfig, UnknownCosts, load_trade_config, trade_config_from_dict)
from gbpjpy_engine.trade.construct import cost_model, r_values
from gbpjpy_engine.trade.stops import adequacy, buffer_stop, noise_risk, sanity, select_reference, stop_quality
from gbpjpy_engine.trade.targets import build_ladder, path_analysis, reachability, select_primary
from trade_helpers import build, lvl, make_ctx

CFG = TradeConfig()
ROOT = Path(gbpjpy_engine.__file__).parents[2]


# ------------------------------------------------------------------ symbol metadata / pip maths
def test_gbpjpy_pip_mathematics_are_centralised():
    s = GBPJPY
    assert (s.digits, s.point, s.pip_size, s.points_per_pip) == (3, 0.001, 0.01, 10)
    assert s.to_pips(0.66) == pytest.approx(66.0) and s.to_points(0.66) == pytest.approx(660.0)
    assert s.pips_to_price(25) == pytest.approx(0.25) and s.points_to_price(7) == pytest.approx(0.007)
    assert s.normalize(189.3605) == 189.36 and s.normalize(189.3615) == 189.362  # banker's rounding on exact decimals
    assert s.normalize(189.3609, "down") == 189.36 and s.normalize(189.3601, "up") == 189.361
    assert s.normalize_away(189.3609, 1, "stop") == 189.36 and s.normalize_away(190.6601, -1, "stop") == 190.661  # never tighter
    assert s.normalize_away(191.7859, 1, "target") == 191.785 and s.normalize_away(188.2341, -1, "target") == 188.235  # never farther
    assert s.pip_size == H1Config().data.pip_size  # one source of truth with the data configuration
    with pytest.raises(ValueError):
        SymbolSpec(point=0.01, digits=3).validate()
    with pytest.raises(ValueError):
        s.normalize(float("nan"))
    two = SymbolSpec(symbol="GBPJPY", digits=2, point=0.01, pip_size=0.01, source="broker_x")
    two.validate()
    assert two.points_per_pip == 1 and two.normalize(189.367) == 189.37


# ------------------------------------------------------------------ stop references
def test_family_specific_stop_policies():
    pols = {f: CFG.stop_policy(f) for f, _ in CFG.stop.policies}
    assert pols["TREND_PULLBACK_CONTINUATION"][0] == "STRUCTURAL_INVALIDATION"
    assert pols["BREAK_RETEST_CONTINUATION"][0] == "BREAK_RETEST_FAILURE"
    assert pols["LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"][0] == "RECLAIM_FAILURE"
    assert len(set(pols.values())) > 1
    ctx = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 189.40, "reason": "x"},
                              "BREAK_RETEST_FAILURE": {"price": 189.60, "reason": "retest"},
                              "RECLAIM_FAILURE": {"price": 189.20, "reason": "sweep"}})
    assert select_reference(ctx, CFG)["stop_reference_type"] == "STRUCTURAL_INVALIDATION"
    assert select_reference(replace(ctx, setup_family="BREAK_RETEST_CONTINUATION"), CFG)["stop_reference_type"] == "BREAK_RETEST_FAILURE"
    assert select_reference(replace(ctx, setup_family="LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"), CFG)["stop_reference_price"] == 189.20


def test_structural_reference_on_wrong_side_is_skipped_and_missing_structure_rejects():
    ctx = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 190.30, "reason": "above entry"},
                              "SWING_INVALIDATION": {"price": 189.55, "reason": "swing"}})
    ref = select_reference(ctx, CFG)
    assert ref["stop_reference_type"] == "SWING_INVALIDATION" and ref["policy_rank"] == 1
    r = build(make_ctx(stop_refs={}))
    assert r.decision == "REJECT_TRADE" and r.category == "NO_STRUCTURAL_STOP"


def test_reference_inside_noise_is_widened_never_tightened():
    ctx = make_ctx(stop_refs={"STRUCTURAL_INVALIDATION": {"price": 189.95, "reason": "tiny pullback"}},
                   structural_levels=[{"price": 189.95, "reason": "tiny"}, {"price": 189.70, "reason": "prior swing"},
                                      {"price": 189.10, "reason": "older"}])
    ref = select_reference(ctx, CFG)
    assert ref["stop_reference_type"] == "VOLATILITY_ADJUSTED_STRUCTURE" and ref["stop_reference_price"] == 189.70
    assert ref["volatility_adjusted_from"] == "STRUCTURAL_INVALIDATION"


# ------------------------------------------------------------------ buffer / adequacy / noise / quality / sanity
def test_stop_buffer_is_volatility_aware_and_bid_ask_correct():
    ctx = make_ctx()
    ref = select_reference(ctx, CFG)
    b = buffer_stop(ctx, ref, CFG, GBPJPY)
    assert b["raw_invalidation_price"] == 189.40 and b["proposed_stop_price"] == pytest.approx(189.36)
    assert b["buffer_components"]["atr_floor"] == pytest.approx(0.03) and b["buffer_components"]["trigger_side_spread"] == 0.0
    wide = buffer_stop(replace(ctx, adverse_wicks=[0.12] * 50), ref, CFG, GBPJPY)
    assert wide["proposed_stop_price"] == pytest.approx(189.28)  # larger observed wicks -> larger buffer
    calm = buffer_stop(replace(ctx, adverse_wicks=[0.0] * 50), ref, CFG, GBPJPY)
    assert calm["stop_buffer"] == pytest.approx(0.03)  # ATR floor
    s = make_ctx(-1)
    sb = buffer_stop(s, select_reference(s, CFG), CFG, GBPJPY)
    assert sb["buffer_components"]["trigger_side_spread"] == pytest.approx(0.02)  # SHORT stop triggers on the ASK
    assert sb["proposed_stop_price"] == pytest.approx(190.60 + 0.04 + 0.02)


def test_volatility_adequacy_classes_and_percentile():
    ctx = make_ctx()
    classes = [adequacy(ctx, ctx.entry_price - m * 0.30, CFG, GBPJPY)["volatility_adequacy"] for m in (0.3, 2.0, 3.0, 5.0)]
    assert classes == ["TOO_TIGHT", "NORMAL", "WIDE", "EXTREME"]
    a = adequacy(ctx, 189.36, CFG, GBPJPY)
    assert a["stop_distance_pips"] == pytest.approx(66.0) and a["stop_distance_atr"] == pytest.approx(2.2)
    assert a["stop_distance_percentile"] == 100.0  # 66 pips > every 60-pip 5-bar range


def test_stop_noise_risk_measures_ordinary_noise():
    ctx = make_ctx()
    far = noise_risk(ctx, 189.36, 0.04, CFG)["stop_noise_risk_score"]
    near = noise_risk(ctx, 189.90, 0.005, CFG)["stop_noise_risk_score"]
    touched = noise_risk(replace(ctx, recent_extremes=[189.37] * 10), 189.36, 0.04, CFG)
    assert near > far and touched["recent_level_touches"] == 10 and touched["stop_noise_risk_score"] > far
    assert "not a hit probability" in touched["note"]


def test_stop_quality_components():
    ctx = make_ctx()
    ref = select_reference(ctx, CFG)
    good = stop_quality(ref, adequacy(ctx, 189.36, CFG, GBPJPY), noise_risk(ctx, 189.36, 0.04, CFG), CFG)
    ext = stop_quality(ref, adequacy(ctx, 188.40, CFG, GBPJPY), noise_risk(ctx, 188.40, 0.04, CFG), CFG)
    assert good["stop_quality_score"] > ext["stop_quality_score"]
    assert set(good["components"]) == {"structural_relevance", "noise_clearance", "volatility_appropriateness", "distance",
                                       "setup_family_relevance"}


def test_stop_sanity_and_broker_constraints():
    ctx = make_ctx()
    unknown = BrokerStopConstraints("UNKNOWN")
    assert sanity(ctx, 189.36, CFG, GBPJPY, unknown) == []
    assert sanity(ctx, ctx.entry_price, CFG, GBPJPY, unknown) == ["ZERO_RISK_DISTANCE"]
    assert sanity(ctx, 190.30, CFG, GBPJPY, unknown) == ["STOP_ON_WRONG_SIDE_OF_ENTRY"]
    assert sanity(ctx, float("nan"), CFG, GBPJPY, unknown) == ["NON_FINITE_VALUE"]
    assert "STOP_BELOW_MINIMUM_DISTANCE" in sanity(ctx, 189.99, CFG, GBPJPY, unknown)
    assert "STOP_NOT_AT_SYMBOL_PRECISION" in sanity(ctx, 189.3605, CFG, GBPJPY, unknown)
    known = BrokerStopConstraints("KNOWN", min_stop_distance_points=1000.0, source="test")
    assert "STOP_INSIDE_BROKER_STOP_LEVEL" in sanity(ctx, 189.36, CFG, GBPJPY, known)
    r = build(ctx)
    assert r.proposal["broker_constraints_status"] == "UNKNOWN"
    rej = build(ctx, constraints=known)
    assert rej.decision == "REJECT_TRADE" and rej.category == "INVALID_STOP"


# ------------------------------------------------------------------ targets
def test_target_ladder_clusters_front_runs_and_records_every_field():
    ctx = make_ctx(levels=[lvl(191.80, 80.0, "H1"), lvl(191.85, 70.0, "H4"), lvl(192.90, 60.0, "H4"), lvl(193.90, 55.0, "H1"),
                           lvl(194.90, 55.0, "H1"), lvl(188.00, 90.0, "H1"), lvl(190.10, 40.0, "H1")])
    lad = build_ladder(ctx, CFG.target, GBPJPY)["ladder"]
    assert [t["label"] for t in lad] == ["T1", "T2", "T3"]  # max_targets
    t1 = lad[0]
    assert t1["structural_level"] == 191.80 and t1["price"] == pytest.approx(191.785)  # before the level, never beyond
    assert t1["timeframes"] == ["H1", "H4"] and t1["n_members"] == 2 and t1["strength"] == 80.0
    assert t1["barriers_before_target"] == 1  # the 190.10 level is too close to be a target but is a barrier
    for t in lad:
        for k in ("price", "type", "timeframe", "strength", "distance_pips", "barriers_before_target", "path",
                  "target_reachability_score", "target_quality_score"):
            assert k in t
        assert t["price"] > ctx.entry_price  # no target below a long entry


def test_short_targets_include_exit_side_spread():
    s = make_ctx(-1)
    t1 = build_ladder(s, CFG.target, GBPJPY)["ladder"][0]
    assert t1["price"] == pytest.approx(188.20 + 0.015 + 0.02)  # SHORT takes profit on the ASK


def test_path_analysis_and_target_quality_penalise_obstruction():
    clear = make_ctx()
    blocked = make_ctx(levels=[lvl(190.30, 90.0), lvl(190.40, 85.0, "H4"), lvl(191.80, 80.0), lvl(191.85, 70.0, "H4")])
    tc = build_ladder(clear, CFG.target, GBPJPY)["ladder"][0]
    lb = build_ladder(blocked, CFG.target, GBPJPY)["ladder"]
    far = next(t for t in lb if t["structural_level"] == 191.80)
    assert far["path"]["barrier_count"] >= 1 and far["path"]["max_barrier_strength"] >= 85.0
    assert far["target_quality_score"] < tc["target_quality_score"]
    p = path_analysis([], 1.0, CFG.target)
    assert p["barrier_count"] == 0 and p["barrier_density_score"] == 0.0 and not p["congested"]


def test_reachability_uses_entry_time_context_only():
    ctx = make_ctx()
    near, far = reachability(ctx, 1.0, CFG.target), reachability(ctx, 7.0, CFG.target)
    assert near["target_reachability_score"] > far["target_reachability_score"]
    weak = reachability(replace(ctx, h4_side_score=10.0, h1_primary_alignment=0.0, momentum_net=-30.0), 3.0, CFG.target)
    assert weak["target_reachability_score"] < reachability(ctx, 3.0, CFG.target)["target_reachability_score"]
    assert set(near["components"]) == {"distance", "trend_context", "momentum", "volatility"}


def test_primary_selection_is_explained_and_ignores_r():
    lad = build_ladder(make_ctx(), CFG.target, GBPJPY)["ladder"]
    best, why = select_primary(lad)
    assert best["label"] == "T1" and "R was not a selection input" in why
    assert select_primary([]) == (None, "NO_STRUCTURAL_TARGET")


# ------------------------------------------------------------------ R, costs
def test_gross_r_exact_directional_mathematics():
    r = r_values(1, 190.020, 189.360, 191.785, 0.0, GBPJPY)
    assert r["gross_R"] == pytest.approx(1.765 / 0.66, abs=1e-4)
    s = r_values(-1, 190.000, 190.660, 188.235, 0.0, GBPJPY)
    assert s["gross_R"] == pytest.approx(1.765 / 0.66, abs=1e-4)
    assert not r_values(1, 190.0, 190.0, 191.0, 0.0, GBPJPY)["valid"]  # zero risk
    assert not r_values(1, 190.0, 190.2, 191.0, 0.0, GBPJPY)["valid"]  # stop above a long entry
    assert r_values(1, 190.0, 189.5, 191.0, 1.0, GBPJPY)["estimated_net_R"] == pytest.approx((1.0 - 0.01) / (0.5 + 0.01), abs=1e-4)


def test_costs_unknown_are_explicit_never_zero_and_known_costs_are_used():
    ctx = make_ctx()
    unk = cost_model(ctx, CFG, UnknownCosts().estimate("GBPJPY", None, "LONG"))
    assert unk["status"] == "PARTIALLY_UNKNOWN" and unk["total_extra_pips"] > 0
    assert unk["components"]["slippage_entry"]["status"] == "UNKNOWN_ASSUMED"
    assert unk["components"]["commission_round_turn"]["status"] == "UNKNOWN_ASSUMED"
    assert unk["components"]["swap_financing"]["status"] == "UNKNOWN"
    known = cost_model(replace(ctx, slippage_pips=0.2, slippage_status="MODELLED"), CFG,
                       FixedCommission(0.6).estimate("GBPJPY", None, "LONG"))
    assert known["status"] == "KNOWN" and known["total_extra_pips"] == pytest.approx(1.0)
    r = build(ctx)
    p = r.proposal
    assert "COSTS_UNKNOWN" in r.reason_codes and p["estimated_net_R"] < p["gross_R"]
    assert p["cost_assumptions"]["status"] == "PARTIALLY_UNKNOWN"


# ------------------------------------------------------------------ management architecture
def test_partial_exit_break_even_trailing_and_time_exit_are_architecture_only():
    single = ExitPlan.single("T1")
    assert single.kind == "SINGLE_TARGET" and single.legs[0].fraction == 1.0
    multi = ExitPlan.split([("T1", 0.5), ("T2", 0.3), ("RUNNER", 0.2)])
    assert multi.kind == "MULTI_TARGET" and len(multi.legs) == 3
    with pytest.raises(ValueError):
        ExitPlan.split([("T1", 0.6), ("T2", 0.6)])
    ctx = ManagementContext(bars_in_trade=5, r_progress=1.2)
    for kind in ("break_even", "trailing", "time_exit"):
        pol = DisabledPolicy(kind)
        assert not pol.enabled and pol.evaluate(ctx).action == "NONE"
    m = build(make_ctx()).proposal["management"]
    assert m["exit_plan"]["kind"] == "SINGLE_TARGET" and m["exit_plan"]["legs"][0]["target"] == "T1"
    assert not m["break_even"]["enabled"] and not m["trailing"]["enabled"] and not m["time_exit"]["enabled"]
    assert {"R_ACHIEVED", "STRUCTURAL_PROGRESS", "TARGET_REACHED", "NEW_H1_STRUCTURE"} == set(m["break_even"]["available_triggers"])
    assert {"STRUCTURE", "ATR", "SWING", "FIXED_R"} == set(m["trailing"]["available_methods"])


# ------------------------------------------------------------------ config
def test_trade_config_documented_yaml_and_validation():
    assert all(r["doc"] and len(r["doc"]) > 10 for r in describe_config(TradeConfig()))
    assert load_trade_config(ROOT / "config" / "trade_default.yaml").to_dict() == TradeConfig().to_dict()
    with pytest.raises(ValueError):
        trade_config_from_dict({"cost": {"assumed_slippage_pips_per_side": 0.0}})  # unknown costs are never zero
    with pytest.raises(ValueError):
        trade_config_from_dict({"stop": {"policies": [["TREND_PULLBACK_CONTINUATION", ["FIXED_PIPS"]]]}})
    with pytest.raises(KeyError):
        trade_config_from_dict({"account": {"balance": 10000}})
    assert not any("lot" in r["name"] or "account" in r["name"] or "leverage" in r["name"] or "risk_percent" in r["name"]
                   for r in describe_config(TradeConfig()))
    assert np.isclose(TradeConfig().min_net_r("TREND_PULLBACK_CONTINUATION"), 1.5)
