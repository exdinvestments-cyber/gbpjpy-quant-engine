"""Phase 1B: room to move, conflict, quality, LONG/SHORT context, permission, hard blockers, fail-safety,
auditability."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ctx_helpers import layer, swing, zone
from gbpjpy_engine import H4Config, H4MarketIntelligenceEngine
from gbpjpy_engine.config import RoomConfig
from gbpjpy_engine.context.engine import ContextEngine
from gbpjpy_engine.context.permission import conflict, decide, directional_score, hard_blockers, quality
from gbpjpy_engine.context.room import room
from gbpjpy_engine.reason_codes import ReasonCode

CFG = H4Config()
VALID_CODES = {c.value for c in ReasonCode}


def ctx(direction=1, **over):
    bull = direction > 0
    cls = "BULLISH" if bull else "BEARISH"
    row = {"extension_state": "normal", "extension_direction": "up" if bull else "down", "structure_quality_score": 70.0,
           "volatility_shock": False, "trend_state": "bullish" if bull else "bearish", "volatility_regime": "normal",
           "shock_severity": 0.0, "extension_score": 40.0, "chop_score": 25.0, "directional_efficiency_percentile": 70.0,
           "close_location": 0.7 if bull else 0.3, "distance_to_support_atr": 3.0, "support_strength": 50.0,
           "distance_to_resistance_atr": 3.0, "resistance_strength": 50.0, "market_quality": "clean_trend",
           "data_quality_status": "OK", "warmup_complete": True, "regime": "BULL_TREND" if bull else "BEAR_TREND"}
    row.update(over.pop("row", {}))
    c = {"row": row, "primary": layer(cls, 85), "intermediate": layer(cls, 80), "immediate": layer(cls, 70),
         "bull_disp_recent": 65.0 if bull else 5.0, "bear_disp_recent": 5.0 if bull else 65.0, "break_ctx": None,
         "failed_bullish_recent": False, "failed_bearish_recent": False, "bull_sweep_score": 0.0, "bear_sweep_score": 0.0,
         "latest_sweep": None, "long_room": {"room_score": 80.0}, "short_room": {"room_score": 80.0},
         "deterioration_score": 10.0, "correction_active": True, "correction_class": "HEALTHY_CORRECTION",
         "retracement": {"retracement_band": "NORMAL"}, "recent_flip": None, "origin_reaction": None,
         "bars_since_gap": None, "stale": False}
    c.update(over)
    return c


def evaluate(c):
    conf, _, _ = conflict(c, CFG)
    q, _ = quality(c, CFG)
    ls, _ = directional_score("long", c, conf, CFG)
    ss, _ = directional_score("short", c, conf, CFG)
    bl = hard_blockers(c, CFG)
    return decide(c, ls, ss, conf, q, bl, CFG), ls, ss, conf, q


# ------------------------------------------------------------------ room
def test_room_uses_nearest_opposing_barrier_and_ignores_own_move():
    mem = [swing(0, "low", 0, 95.0), swing(1, "high", 10, 104.0)]
    zones = [zone(1, 106.0, 106.3, strength=80, age=100), zone(2, 100.8, 101.0, strength=90, age=3)]
    r = room("long", 100.0, 1.0, 200, zones, mem, [], 101.0, 195, exclude_from=190, strong_level=60, cfg=RoomConfig())
    # zone 2 and the range high were formed inside the current leg (index >= 190): not barriers
    assert r["nearest_barrier"]["kind"] == "confirmed_swing_high" and r["nearest_barrier"]["distance_atr"] == 4.0
    assert r["room_score"] == 100.0
    r2 = room("long", 105.5, 1.0, 200, zones, mem, [], 101.0, 195, 190, 60, RoomConfig())
    assert r2["nearest_barrier"]["kind"] == "strong_resistance_zone" and r2["room_score"] == 0.0
    r3 = room("short", 100.0, 1.0, 200, [], mem, [], float("nan"), 0, 190, 60, RoomConfig())
    assert r3["nearest_barrier"]["kind"] == "confirmed_swing_low" and r3["room_score"] == 100.0
    assert room("long", 200.0, 1.0, 200, [], [], [], float("nan"), 0, 190, 60, RoomConfig())["note"] == "NO_BARRIER_FOUND"


# ------------------------------------------------------------------ conflict / quality
def test_conflicts_detected_with_reasons():
    c = ctx(1, bear_disp_recent=75.0, row={"extension_state": "extremely_extended", "trend_state": "bearish"},
            failed_bullish_recent=True, intermediate=layer("BEARISH", 70), bear_sweep_score=60.0,
            long_room={"room_score": 5.0})
    score, hits, w = conflict(c, CFG)
    assert set(hits) == {"CONFLICT_PRIMARY_VS_DISPLACEMENT", "CONFLICT_TREND_VS_EXTENSION", "CONFLICT_STRUCTURE_VS_TREND_ENGINE",
                         "CONFLICT_STRUCTURE_VS_FAILED_BREAK", "CONFLICT_HIERARCHY_DISAGREEMENT", "CONFLICT_OPPOSING_SWEEP",
                         "CONFLICT_TREND_VS_OPPOSING_LEVEL"}
    import numpy as np

    assert score == pytest.approx(100 * (1 - np.prod([1 - w[h] for h in hits])), abs=0.01)  # noisy-OR
    assert 80 < score <= 100 and set(hits) <= VALID_CODES
    assert conflict(ctx(1), CFG)[0] == 0.0
    shock, hits2, _ = conflict(ctx(1, row={"volatility_shock": True}), CFG)
    assert "CONFLICT_STRUCTURE_VS_SHOCK" in hits2


def test_quality_families_exposed_and_bounded():
    q, fam = quality(ctx(1), CFG)
    assert set(fam) == {"STRUCTURE", "DISPLACEMENT", "MOMENTUM", "VOLATILITY", "LOCATION", "LIQUIDITY_CONTEXT",
                        "ROOM_TO_MOVE", "MARKET_QUALITY"}
    assert all(0 <= v <= 100 for v in fam.values()) and q > 60
    q_bad, fam_bad = quality(ctx(1, row={"chop_score": 90.0, "volatility_regime": "extreme"}, deterioration_score=90.0), CFG)
    assert q_bad < q and fam_bad["MARKET_QUALITY"] < fam["MARKET_QUALITY"]


def test_family_weighting_prevents_inflation():
    base, _ = quality(ctx(1), CFG)
    # saturating correlated displacement inputs cannot move quality beyond the family's weight
    sat, _ = quality(ctx(1, bull_disp_recent=100.0, bear_disp_recent=0.0), CFG)
    assert sat - base <= 100 * CFG.context_scoring.q_displacement + 1e-9


# ------------------------------------------------------------------ directional scores
def test_long_short_symmetry_but_separate():
    ls_b, ss_b = directional_score("long", ctx(1), 0, CFG)[0], directional_score("short", ctx(1), 0, CFG)[0]
    ls_s, ss_s = directional_score("long", ctx(-1), 0, CFG)[0], directional_score("short", ctx(-1), 0, CFG)[0]
    assert ls_b == pytest.approx(ss_s) and ss_b == pytest.approx(ls_s)
    assert ls_b > 55 > ss_b


def test_negative_evidence_reduces_long_score():
    base = directional_score("long", ctx(1), 0, CFG)[0]
    for over in ({"long_room": {"room_score": 10.0}}, {"row": {"extension_state": "extremely_extended"}},
                 {"failed_bullish_recent": True}, {"intermediate": layer("BEARISH", 70)},
                 {"row": {"market_quality": "severe_chop"}}):
        s, det = directional_score("long", ctx(1, **over), 0, CFG)
        assert s < base and det["penalties"], over
    s_conf, det = directional_score("long", ctx(1), 60.0, CFG)
    assert s_conf < base and det["penalties"]["context_conflict"] == pytest.approx(0.3)


# ------------------------------------------------------------------ permission decision
def test_allow_long_and_short():
    d, ls, *_ = evaluate(ctx(1))
    assert d["directional_permission"] == "ALLOW_LONG" and "LONG_PERMISSION_GRANTED" in d["permission_codes"]
    assert 0 < d["permission_confidence"] <= 100
    d2, *_ = evaluate(ctx(-1))
    assert d2["directional_permission"] == "ALLOW_SHORT"


def _two_way(regime):
    c = ctx(1, row={"regime": regime}, primary=layer("BULLISH", 85), intermediate=layer("NEUTRAL", 0),
            immediate=layer("NEUTRAL", 0))
    return c


def test_allow_both_only_in_range_regimes(monkeypatch):
    import gbpjpy_engine.context.permission as perm

    for regime, expected in (("RANGE", "ALLOW_BOTH"), ("BULL_TREND", "BLOCK_ALL")):
        c = _two_way(regime)
        d = perm.decide(c, 70.0, 68.0, 10.0, 70.0, [], CFG)
        assert d["directional_permission"] == expected
        if expected == "BLOCK_ALL":
            assert "TWO_WAY_CONTEXT_OUTSIDE_RANGE" in d["permission_codes"]
        else:
            assert "BOTH_DIRECTIONS_ALLOWED" in d["permission_codes"] and d["permission_confidence"] > 0
    # sufficiently separated -> only the stronger side
    assert perm.decide(_two_way("RANGE"), 80.0, 60.0, 10.0, 70.0, [], CFG)["directional_permission"] == "ALLOW_LONG"


def test_block_all_on_low_quality_high_conflict_or_weak_context():
    d = decide(ctx(1), 70.0, 10.0, 80.0, 70.0, [], CFG)
    assert d["directional_permission"] == "BLOCK_ALL" and "CONTEXT_CONFLICT_TOO_HIGH" in d["permission_codes"]
    d = decide(ctx(1), 70.0, 10.0, 10.0, 30.0, [], CFG)
    assert "CONTEXT_QUALITY_TOO_LOW" in d["permission_codes"]
    d = decide(ctx(1), 40.0, 10.0, 10.0, 70.0, [], CFG)
    assert d["directional_permission"] == "BLOCK_ALL" and "LONG_CONTEXT_INSUFFICIENT" in d["permission_codes"]
    assert any("long_context_score" in x for x in d["failed_requirements"]["long"])
    assert d["permission_confidence"] == 0.0
    d = decide(ctx(1, long_room={"room_score": 10.0}), 70.0, 10.0, 10.0, 70.0, [], CFG)
    assert any("long_room_score" in x for x in d["failed_requirements"]["long"])


def test_transition_regime_requires_more_evidence():
    c = ctx(1, row={"regime": "TRANSITION"})
    assert decide(c, 60.0, 10.0, 10.0, 70.0, [], CFG)["directional_permission"] == "BLOCK_ALL"
    assert decide(c, 70.0, 10.0, 10.0, 70.0, [], CFG)["directional_permission"] == "ALLOW_LONG"


@pytest.mark.parametrize("over,blocker", [
    ({"row": {"data_quality_status": "ERROR"}}, "INVALID_DATA"),
    ({"stale": True}, "STALE_DATA"),
    ({"row": {"warmup_complete": False}}, "INSUFFICIENT_HISTORY"),
    ({"bars_since_gap": 2}, "UNRESOLVED_DATA_GAP"),
    ({"row": {"shock_severity": 95.0}}, "EXTREME_VOLATILITY_SHOCK"),
    ({"row": {"market_quality": "severe_chop"}}, "SEVERE_CHOP"),
    ({"primary": layer("UNCLEAR"), "intermediate": layer("UNCLEAR")}, "UNCLASSIFIABLE_STRUCTURE"),
])
def test_hard_blockers_force_block_all(over, blocker):
    d, *_ = evaluate(ctx(1, **over))
    assert d["directional_permission"] == "BLOCK_ALL"
    assert f"BLOCKER_{blocker}" in d["permission_codes"] and "ALL_DIRECTIONS_BLOCKED" in d["permission_codes"]
    assert set(d["permission_codes"]) <= VALID_CODES


def test_blocker_configurable():
    cfg = H4Config()
    from dataclasses import replace

    cfg2 = replace(cfg, blockers=replace(cfg.blockers, block_severe_chop=False, gap_block_bars=0))
    assert "SEVERE_CHOP" not in hard_blockers(ctx(1, row={"market_quality": "severe_chop"}), cfg2)
    assert "UNRESOLVED_DATA_GAP" not in hard_blockers(ctx(1, bars_since_gap=1), cfg2)


# ------------------------------------------------------------------ engine-level behaviour
def test_scenario_permissions(scenario_results):
    for name, allowed in (("clean_uptrend", {"ALLOW_LONG", "BLOCK_ALL"}), ("clean_downtrend", {"ALLOW_SHORT", "BLOCK_ALL"}),
                          ("sideways_range", {"BLOCK_ALL", "ALLOW_BOTH"}), ("volatile_range", {"BLOCK_ALL", "ALLOW_BOTH"}),
                          ("low_volatility_compression", {"BLOCK_ALL", "ALLOW_BOTH"})):
        sc, res = scenario_results[name]
        cf = res.context.frame.iloc[sc.segment_start + 60:]
        assert set(cf["directional_permission"]) <= allowed, name
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    assert (cf["directional_permission"] == "ALLOW_LONG").mean() > 0.25
    assert (cf["long_context_score"] > cf["short_context_score"]).mean() > 0.95
    sc, res = scenario_results["clean_downtrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    assert (cf["directional_permission"] == "ALLOW_SHORT").mean() > 0.25


def test_warmup_and_shock_bars_blocked(scenario_results, rw_result):
    cf = rw_result.context.frame
    assert (cf["directional_permission"].iloc[:250] == "BLOCK_ALL").all()
    assert all("INSUFFICIENT_HISTORY" in b for b in cf["hard_blockers"].iloc[:250])
    sc, res = scenario_results["volatility_shock"]
    k = sc.events["shock_index"]
    assert res.context.frame["directional_permission"].iloc[k] == "BLOCK_ALL"
    assert "EXTREME_VOLATILITY_SHOCK" in res.context.frame["hard_blockers"].iloc[k]


def test_block_all_is_common_on_random_walk(rw_result):
    cf = rw_result.context.frame.iloc[250:]
    assert (cf["directional_permission"] == "BLOCK_ALL").mean() > 0.5


def test_stale_data_blocker_with_as_of(engine, rw_bars):
    last_close = rw_bars["timestamp"].iloc[-1] + pd.Timedelta(hours=4)
    fresh = engine.run(rw_bars, as_of=last_close + pd.Timedelta(hours=1))
    stale = engine.run(rw_bars, as_of=last_close + pd.Timedelta(hours=48))
    assert "STALE_DATA" not in fresh.context.frame["hard_blockers"].iloc[-1]
    assert "STALE_DATA" in stale.context.frame["hard_blockers"].iloc[-1]
    assert stale.context.frame["directional_permission"].iloc[-1] == "BLOCK_ALL"


def test_fail_safe_exception_blocks_all(rw_result):
    ctx_res = ContextEngine(rw_result.config).run(rw_result, _fault_at=500)
    cf = ctx_res.frame
    assert ctx_res.errors and ctx_res.errors[0]["index"] == 500
    assert (cf["directional_permission"].iloc[500:] == "BLOCK_ALL").all()
    assert all(b == ["CONTEXT_ERROR"] for b in cf["hard_blockers"].iloc[500:])
    assert (cf["permission_confidence"].iloc[500:] == 0).all()
    # bars before the fault are unaffected
    pd.testing.assert_series_equal(cf["directional_permission"].iloc[:500],
                                   rw_result.context.frame["directional_permission"].iloc[:500])
    assert ctx_res.explain(600)["blockers"] == ["CONTEXT_ERROR"]


def test_every_decision_is_explained(rw_result):
    cf = rw_result.context.frame
    for i in range(250, len(cf), 13):
        e = rw_result.context.explain(i)
        assert e["decision"] == cf["directional_permission"].iloc[i]
        assert e["reason_codes"] and set(e["reason_codes"]) <= VALID_CODES
        assert e["reason_codes"] == list(cf["permission_reason_codes"].iloc[i])
        if e["decision"] == "BLOCK_ALL":
            assert e["blockers"] or any(e["failed_requirements"].values()) or "TWO_WAY_CONTEXT_OUTSIDE_RANGE" in e["reason_codes"]
        else:
            assert e["evidence"], i
        assert set(cf["context_reason_codes"].iloc[i]) <= VALID_CODES
        json.dumps(e, default=str)


def test_explain_permission_api(scenario_results):
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame
    i = int(cf.index[cf["directional_permission"] == "ALLOW_LONG"][-1])
    e = res.explain_permission(res.features["timestamp"].iloc[i])
    assert e["decision"] == "ALLOW_LONG" and "LONG_PERMISSION_GRANTED" in e["reason_codes"]
    assert "PRIMARY_STRUCTURE_BULLISH" in e["evidence"]
    assert e["long_breakdown"]["families"] and "not a trade signal" in e["note"]


def test_direction_dependent_codes_are_attributed_to_the_right_side(scenario_results, rw_result):
    """E.g. HIGH_QUALITY_BREAK of a bullish break must never be cited as evidence for ALLOW_SHORT."""
    results = [rw_result] + [scenario_results[k][1] for k in ("clean_uptrend", "clean_downtrend", "trend_reversal")]
    checked = 0
    for res in results:
        cf = res.context.frame
        for i in range(250, len(cf)):
            e = res.context.explain(i)
            if e["decision"] not in ("ALLOW_LONG", "ALLOW_SHORT"):
                continue
            want = "bullish" if e["decision"] == "ALLOW_LONG" else "bearish"
            det = res.context.details[i]
            if "HIGH_QUALITY_BREAK" in e["evidence"]:
                assert det["breakout"]["direction"] == want
                checked += 1
            if "HEALTHY_PULLBACK" in e["evidence"] or "TREND_EARLY" in e["evidence"]:
                assert cf["structural_direction"].iloc[i] == (1 if want == "bullish" else -1)
    assert checked > 0
