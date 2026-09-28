"""Phase 1C: H1 structure, hierarchy, legs, pullbacks, displacement, momentum, volatility, chop, efficiency,
zones, confluence, liquidity, sweeps, breakouts, transitions, reclaims, rejection, patterns, compression,
location and room."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ctx_helpers import price_arrays
from gbpjpy_engine.features.structure import BreakTransition
from gbpjpy_engine.h1.config import H1ConfluenceConfig, H1PullbackConfig, H1RejectionConfig, H1TransitionConfig
from gbpjpy_engine.h1.signals import (candle_patterns, confluence, make_reclaim, make_transition, pullback,
                                      reference_impulse, rejection_score)

T0 = pd.Timestamp("2024-01-02", tz="UTC")
TF = pd.Timedelta(hours=1)


def seg(res, sc):
    return res.frame.iloc[sc.segment_start_h1:]


# ------------------------------------------------------------------ engine-level feature presence
def test_h1_structure_swings_and_timestamps(h1_results):
    sc, _, res = h1_results["h1_uptrend_pullbacks"]
    sw = res.structure.swings
    assert len(sw) > 20
    for s in sw:
        assert s.confirmed_at > s.occurred_at and s.confirm_index - s.pivot_index == 3
        assert s.confirmed_at == res.features["available_at"].iloc[s.confirm_index]
    assert set(res.features["structure_state"]) <= {"bullish", "bearish", "neutral", "transitional"}
    labels = {s.label for s in sw if s.label}
    assert {"HH", "HL"} <= labels


def test_h1_hierarchy_and_alignment_states(h1_results):
    sc, _, res = h1_results["h1_uptrend_pullbacks"]
    f = seg(res, sc)
    assert set(f["h1_primary_structure"]) <= {"BULLISH", "BEARISH", "NEUTRAL", "TRANSITIONAL", "RANGING", "UNCLEAR"}
    assert (f["h1_primary_structure"] == "BULLISH").mean() > 0.5
    permitted = (f["h4_permission"] == "ALLOW_LONG") & (f["h4_context_status"] == "OK")
    assert set(f.loc[permitted, "alignment_state"]) <= {"FULL_BULLISH_ALIGNMENT", "H4_BULL_H1_PULLBACK", "H1_TRANSITION",
                                                        "H1_RANGE", "H1_COUNTERTREND", "CONFLICTED"}
    assert (f.loc[permitted, "alignment_state"] == "H4_BULL_H1_PULLBACK").any()
    assert (f.loc[permitted, "alignment_state"] == "FULL_BULLISH_ALIGNMENT").any()
    assert (f.loc[f["h4_permission"] != "ALLOW_LONG", "alignment_state"] == "BLOCKED_BY_H4").all()
    stale = f["h4_context_status"] != "OK"
    assert (f.loc[stale, "alignment_state"] == "BLOCKED_BY_H4").all()  # stale H4 context never aligns
    sc2, _, res2 = h1_results["h1_downtrend_pullbacks"]
    f2 = seg(res2, sc2)
    assert (f2.loc[f2["h4_permission"] == "ALLOW_SHORT", "alignment_state"] == "H4_BEAR_H1_PULLBACK").any()


def test_h1_legs_impulse_correction(h1_results):
    sc, _, res = h1_results["h1_uptrend_pullbacks"]
    kinds = set()
    for d in res.details[sc.segment_start_h1:]:
        kinds |= {lg["classification"] for lg in d.get("legs", [])}
    assert kinds & {"STRONG_IMPULSE", "NORMAL_IMPULSE", "WEAK_IMPULSE"}
    assert kinds & {"HEALTHY_CORRECTION", "DEEP_CORRECTION", "CHOPPY_CORRECTION"}


def test_h1_market_quality_columns_bounded(h1_results):
    sc, _, res = h1_results["h1_range"]
    f = res.frame.iloc[300:]
    for col in ("bullish_displacement_score", "bearish_displacement_score", "h1_chop_score", "compression_score",
                "long_location_score", "short_location_score", "h1_long_room_score", "h1_short_room_score",
                "long_setup_score", "short_setup_score", "long_setup_confidence", "long_setup_conflict_score",
                "bullish_sweep_score", "bearish_sweep_score", "bullish_rejection_score", "long_pullback_quality"):
        v = f[col].dropna()
        assert ((v >= 0) & (v <= 100)).all(), col
    assert set(f["volatility_regime"]) <= {"VERY_LOW", "LOW", "NORMAL", "HIGH", "EXTREME", "UNKNOWN"}
    assert f["directional_efficiency"].between(0, 1).all()
    assert f["momentum_strength"].ge(0).all()
    assert f["h1_vs_h4_atr_pct"].dropna().between(0.1, 2.0).all()  # H1 ATR% smaller than H4 ATR%
    rng_chop = f["h1_chop_score"].mean()
    sc_u, _, res_u = h1_results["h1_uptrend_pullbacks"]
    assert seg(res_u, sc_u)["h1_chop_score"].mean() < rng_chop
    assert (seg(res_u, sc_u)["bullish_displacement_score"] >= 60).any()


def test_h1_zones_liquidity_breakouts(h1_results):
    sc, _, res = h1_results["h1_uptrend_pullbacks"]
    assert any(len(z) for z in res.zones_by_bar[sc.segment_start_h1:])
    assert res.sweep_events and {e.side for e in res.sweep_events} <= {"upside", "downside"}
    states = {t.to_state for b in res.structure.breaks for t in b.transitions}
    assert {"CANDIDATE", "CONFIRMED", "ACCEPTED"} <= states and (states & {"FAILED", "INVALIDATED"})
    f = seg(res, sc)
    assert f["breakout_state"].dropna().isin(["CANDIDATE", "CONFIRMED", "ACCEPTED", "FAILED", "INVALIDATED"]).all()
    assert res.transitions and all(t.direction in ("bullish", "bearish") for t in res.transitions)


def test_candle_patterns_are_features_only():
    df = pd.DataFrame({
        "open": [100.0, 100.5, 99.8, 100.0], "high": [100.6, 100.6, 101.0, 100.5],
        "low": [99.9, 99.7, 99.6, 99.1], "close": [100.5, 99.8, 100.9, 100.4],
        "close_location": [0.86, 0.11, 0.92, 0.92], "candle_class": ["x", "x", "strong_bullish_close", "x"],
    })
    p = candle_patterns(df)
    assert bool(p["pat_bearish_engulfing"].iloc[1]) and bool(p["pat_bullish_engulfing"].iloc[2])
    assert bool(p["pat_outside_bar"].iloc[2]) and bool(p["pat_strong_bullish_close"].iloc[2])
    assert bool(p["pat_bullish_pin"].iloc[3])


def test_patterns_never_create_setups_alone(h1_results):
    """A setup's qualification requires a family premise; pattern columns are not inputs to eligibility."""
    import inspect

    import gbpjpy_engine.h1.setups as setups

    assert "pat_" not in inspect.getsource(setups)


# ------------------------------------------------------------------ pullback
def _imp(start=100.0, end=104.0, end_index=10, direction="up"):
    return SimpleNamespace(start_price=start, end_price=end, end_index=end_index, distance=abs(end - start),
                           direction=direction, leg_id="1-2", distance_atr=4.0)


@pytest.mark.parametrize("low,state", [(103.8, "NO_PULLBACK"), (103.2, "SHALLOW_PULLBACK"), (102.2, "HEALTHY_PULLBACK"),
                                       (100.8, "DEEP_PULLBACK"), (99.9, "STRUCTURE_THREATENING_PULLBACK")])
def test_pullback_states(low, state):
    n = 16
    c = np.full(n, 103.95)
    c[:11] = np.linspace(100, 104, 11)
    lows = c - 0.05
    lows[13] = low
    c[13] = max(low + 0.1, 100.0)
    arr = price_arrays(c, c + 0.05, lows, c)
    arr["overlap_prev"] = np.full(n, 0.5)
    counter = np.zeros(n)
    counter[12:14] = [40.0, 30.0]
    pb = pullback("long", _imp(), 15, arr, counter, H1PullbackConfig())
    assert pb["pullback_state"] == state
    assert pb["depth_pct"] == pytest.approx((104 - low) / 4 * 100)
    assert 0 <= pb["pullback_quality"] <= 100 and pb["duration_bars"] == 5


def test_failed_pullback_and_quality_orders():
    n = 16
    c = np.full(n, 103.0)
    c[:11] = np.linspace(100, 104, 11)
    c[14:] = 99.5  # close below the impulse origin by > 0.25 ATR
    arr = price_arrays(c, c + 0.05, c - 0.05, c)
    arr["overlap_prev"] = np.full(n, 0.5)
    pb = pullback("long", _imp(), 15, arr, np.zeros(n), H1PullbackConfig())
    assert pb["pullback_state"] == "FAILED_PULLBACK_CONTEXT" and pb["components"]["structural_integrity"] == 0.0
    # controlled correction with fading counter momentum scores higher than a violent one
    calm = np.concatenate([np.linspace(100, 104, 11), np.linspace(103.8, 102.2, 5)])
    fast = np.concatenate([np.linspace(100, 104, 11), [102.2, 102.2, 102.2, 102.3, 102.3]])
    q = []
    for path, counter in ((calm, np.linspace(40, 5, 16)), (fast, np.full(16, 60.0))):
        a = price_arrays(path, path + 0.05, path - 0.05, path)
        a["overlap_prev"] = np.full(16, 0.6)
        q.append(pullback("long", _imp(), 15, a, counter, H1PullbackConfig())["pullback_quality"])
    assert q[0] > q[1]


def test_reference_impulse_requires_minimum_size():
    legs = [SimpleNamespace(direction="up", distance_atr=0.8), SimpleNamespace(direction="down", distance_atr=2.0)]
    assert reference_impulse(legs, "long", H1PullbackConfig()) is None
    legs.append(SimpleNamespace(direction="up", distance_atr=2.5))
    assert reference_impulse(legs, "long", H1PullbackConfig()) is legs[-1]


# ------------------------------------------------------------------ transition / reclaim / rejection
def _break(direction="bullish", level=102.0, b=12, before="bearish", mag_atr=0.6):
    t = T0 + TF * b
    ev = SimpleNamespace(direction=direction, structure_before=before, level=level, event_id=3, break_index=b,
                         magnitude_atr=mag_atr, atr_at_break=1.0, transitions=[])
    ev.transitions.append(BreakTransition(b, t + TF, None, "CANDIDATE", "x"))
    ev.transitions_known_at = lambda c, ev=ev: [x for x in ev.transitions if x.index <= c]
    return ev


def test_structural_transition_is_measurable():
    n = 16
    c = np.concatenate([np.linspace(104, 101, 8), np.linspace(101.2, 102.6, 8)])
    arr = price_arrays(c, c + 0.1, c - 0.1, c)
    legs = [SimpleNamespace(direction="down", distance_atr=3.0, end_index=7, start_index=0)]
    counter = np.concatenate([np.full(8, 60.0), np.linspace(40, 5, 8)])
    ev = _break()
    te = make_transition(ev, 12, legs, counter, 70.0, 0.0, arr, H1TransitionConfig(), 0)
    assert te is not None and te.direction == "bullish" and te.broken_level == 102.0
    r = te.evaluate(ev, 12)
    assert r["transition_score"] > 50 and r["confirmation_timestamp"] is None and r["break_state"] == "CANDIDATE"
    ev.transitions.append(BreakTransition(14, T0 + TF * 15, "CANDIDATE", "CONFIRMED", "held"))
    assert te.evaluate(ev, 13)["confirmation_timestamp"] is None  # point in time
    later = te.evaluate(ev, 14)
    assert later["confirmation_timestamp"] is not None and later["transition_score"] > r["transition_score"]
    ev.transitions.append(BreakTransition(15, T0 + TF * 16, "CONFIRMED", "FAILED", "lost"))
    assert te.evaluate(ev, 15)["transition_score"] == 0.0
    # continuation BOS and breaks without a prior counter move are not transitions
    assert make_transition(_break(before="bullish"), 12, legs, counter, 70, 0, arr, H1TransitionConfig(), 1) is None
    small = [SimpleNamespace(direction="down", distance_atr=0.5, end_index=7, start_index=0)]
    assert make_transition(_break(), 12, small, counter, 70, 0, arr, H1TransitionConfig(), 1) is None


def test_reclaim_score():
    c = np.array([102.5, 102.4, 102.2, 101.6, 101.5, 102.4, 102.8, 103.1])
    arr = price_arrays(c, c + 0.1, c - 0.1, c)
    ev = SimpleNamespace(direction="bearish", level=102.0, break_index=3, atr_at_break=1.0)
    rc = make_reclaim(ev, 5, arr, 65.0, 0.0, 0)
    assert rc.direction == "bullish" and rc.duration_lost_bars == 2 and rc.distance_lost_atr == pytest.approx(0.5)
    s5, s7 = rc.evaluate(5, arr)["reclaim_score"], rc.evaluate(7, arr)["reclaim_score"]
    assert 0 < s5 < s7  # follow-through adds evidence as it happens, never before
    arr2 = price_arrays(np.append(c, 101.5), np.append(c + 0.1, 101.6), np.append(c - 0.1, 101.4), np.append(c, 101.5))
    assert rc.evaluate(8, arr2)["reclaim_score"] == 0.0  # level lost again


def test_rejection_score_level_and_follow_through():
    n = 6
    c = np.array([100.0, 100.0, 99.9, 100.3, 100.6, 100.9])
    o = np.array([100.0, 100.0, 100.0, 99.95, 100.3, 100.6])
    low = np.minimum(o, c) - 0.02
    low[3] = 99.0
    arr = price_arrays(o, np.maximum(o, c) + 0.02, low, c)
    base = np.zeros(n)
    base[3] = 80.0
    at_level, info = rejection_score("long", 3, arr, base, [(98.9, 99.1)], H1RejectionConfig())
    no_level, _ = rejection_score("long", 3, arr, base, [], H1RejectionConfig())
    assert at_level > no_level and info["level_interaction"]
    later, info2 = rejection_score("long", 5, arr, base, [(98.9, 99.1)], H1RejectionConfig())
    assert later > at_level and info2["follow_through"] > 0


# ------------------------------------------------------------------ confluence
def test_confluence_clusters_do_not_double_count():
    cfg = H1ConfluenceConfig()
    lv = [{"lower": 99.8, "upper": 100.0, "source": "H4_ZONE", "timeframe": "H4", "strength": 70},
          {"lower": 99.85, "upper": 100.05, "source": "H1_ZONE", "timeframe": "H1", "strength": 60},
          {"lower": 99.9, "upper": 100.0, "source": "H1_ZONE", "timeframe": "H1", "strength": 60},
          {"lower": 95.0, "upper": 95.2, "source": "H1_ZONE", "timeframe": "H1", "strength": 90}]
    r = confluence(100.3, 0.5, lv, cfg)
    near = r["long"]
    assert near["n_members"] == 3 and near["timeframes"] == ["H1", "H4"] and r["h4_h1_confluence"]
    assert len(r["clusters"]) == 2
    only_h1 = confluence(100.3, 0.5, [lv[1], lv[2]], cfg)
    single = confluence(100.3, 0.5, [lv[1]], cfg)
    # a second copy of the same H1 information adds nothing
    assert only_h1["long_score"] == single["long_score"]
    assert r["long_score"] > only_h1["long_score"]
    assert confluence(100.3, 0.5, [lv[3]], cfg)["long"] is None  # far away: no interaction


def test_location_and_room_outputs(h1_results):
    sc, _, res = h1_results["h1_uptrend_pullbacks"]
    d = res.details[len(res.frame) - 5]
    for side in ("long", "short"):
        assert set(d["location"][side]["components"]) == {"confluence", "h1_range_position", "h4_range_position",
                                                          "break_retest_area", "liquidity_interaction", "recent_swing"}
        rm = d["room"][side]
        assert rm["room_score"] == pytest.approx(min(rm["h1_room_score"], rm["h4_room_score_in_h1"]))
        assert rm["limiting"] in ("H1", "H4")
