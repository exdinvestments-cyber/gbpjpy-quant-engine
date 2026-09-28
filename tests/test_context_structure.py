"""Phase 1B: full swing-history interpretation, structural hierarchy, legs, impulse vs correction."""

from __future__ import annotations

import numpy as np
import pytest

from ctx_helpers import swing, zigzag
from gbpjpy_engine.config import HierarchyConfig, LegConfig
from gbpjpy_engine.context.hierarchy import classify_hierarchy, classify_layer
from gbpjpy_engine.context.legs import LegEngine, SwingMemoryTracker
from gbpjpy_engine.synthetic import generate_scenario
from helpers import make_bars

H = HierarchyConfig()


def _uptrend_swings(n=16):
    prices, lo = [], 100.0
    for _ in range(n // 2):
        prices += [lo, lo + 4]
        lo += 2
    return zigzag(prices)


def test_primary_uses_full_history_and_reports_evidence():
    sw = _uptrend_swings()
    res = classify_layer("primary", sw, close=sw[-1].price - 1, atr=1.0, cfg=H, protect=True)
    assert res.classification == "BULLISH"
    assert res.swing_ids == [s.swing_id for s in sw]  # every swing in the window influenced it
    assert len(res.weights) == len(res.labels) and res.weights[-1] == pytest.approx(H.recency_weight_ratio * res.weights[0])
    assert res.bull_consistency > 0.9 and res.high_progression_atr > 0 and res.low_progression_atr > 0
    assert res.violations == 0 and res.avg_correction_ratio == pytest.approx(0.5)
    assert 60 <= res.confidence <= 100


def test_bearish_and_unclear_and_ranging():
    sw = zigzag([110, 106, 108, 104, 106, 102, 104, 100], start_kind="high")
    assert classify_layer("p", sw, 101, 1.0, H).classification == "BEARISH"
    assert classify_layer("p", sw[:2], 101, 1.0, H).classification == "UNCLEAR"
    rng = zigzag([100, 102, 100.2, 101.9, 100.1, 102.1, 99.9, 102.0])
    r = classify_layer("p", rng, 101, 1.0, H)
    assert r.classification == "RANGING" and r.confidence > 0


def test_few_bearish_candles_do_not_flip_primary():
    sw = _uptrend_swings()
    # price dips below the latest swing high but stays above the protected low
    close = sw[-2].price + 0.5  # above the last higher low
    h = classify_hierarchy(sw, close, 1.0, H)
    assert h["primary"].classification == "BULLISH"


def test_protected_low_break_makes_primary_transitional():
    sw = _uptrend_swings()
    protected = sw[-2].price  # the low that launched the latest HH
    h = classify_hierarchy(sw, protected - 1.0, 1.0, H)
    assert h["primary"].classification == "TRANSITIONAL"
    assert h["primary"].broken_from == "BULLISH" and h["primary"].protected_level == pytest.approx(protected)


def test_hierarchy_layers_can_differ():
    # long uptrend, then a bearish sequence in the latest swings (correction) and a provisional recovery
    prices = [100, 104, 102, 106, 104, 108, 106, 110, 108, 112, 110, 114, 111, 113.0, 110.5, 112.5]
    sw = zigzag(prices)
    h = classify_hierarchy(sw, close=112.9, atr=1.0, cfg=H)
    # larger bullish trend / correction in progress / early recovery attempt
    assert h["primary"].classification == "BULLISH"
    assert h["intermediate"].classification == "TRANSITIONAL"
    assert h["immediate"].provisional == "HH"
    assert h["immediate"].labels[-1] == "HH"
    for lr in h.values():
        assert 0 <= lr.confidence <= 100


def test_swing_memory_tracker_matches_structure_history(rw_result):
    st = rw_result.structure
    tr = SwingMemoryTracker(st.swings, st.history_size)
    for c in range(len(rw_result.features)):
        mem = tr.update(c)
        if c % 37 == 0:
            assert [s.swing_id for s in mem] == [s.swing_id for s in st.history_at(c)]


def _leg_engine(closes):
    bars = make_bars(closes)
    rng = (bars["high"] - bars["low"]).to_numpy()
    return bars, LegEngine(bars, np.full(len(bars), float(np.mean(rng))), np.full(len(bars), 0.3), LegConfig(), 0.01)


def test_leg_metrics_and_impulse_vs_correction():
    up = list(np.linspace(100, 106, 7))  # clean, efficient impulse
    chop = [106, 105.2, 105.8, 105.0, 105.6, 104.9, 105.3]  # overlapping correction
    closes = up + chop
    bars, eng = _leg_engine(closes)
    a = swing(0, "low", 0, bars["low"].iloc[0])
    b = swing(1, "high", 6, bars["high"].iloc[6], "HH")
    c = swing(2, "low", 11, bars["low"].iloc[11], "HL")
    imp = eng.confirmed_leg(a, b)
    cor = eng.confirmed_leg(b, c)
    assert imp.direction == "up" and cor.direction == "down"
    assert imp.duration_bars == 6 and imp.distance > 0 and imp.pips == pytest.approx(imp.distance / 0.01)
    for k in ("start_time", "end_time", "distance_atr", "avg_body_atr", "avg_range_atr", "efficiency", "overlap",
              "momentum", "velocity_atr", "max_adverse_atr", "max_favourable_atr"):
        assert getattr(imp, k) is not None
    assert imp.efficiency > cor.efficiency
    assert imp.impulse_score > cor.impulse_score
    assert imp.nature == "IMPULSIVE"
    eng.classify([imp, cor], ref_dir=1)
    assert imp.classification in ("STRONG_IMPULSE", "NORMAL_IMPULSE")
    assert cor.classification in ("HEALTHY_CORRECTION", "CHOPPY_CORRECTION", "DEEP_CORRECTION")
    assert cor.retracement_ratio == pytest.approx(cor.distance / imp.distance)


def test_failed_impulse_and_unclear_classification():
    closes = list(np.linspace(100, 104, 5)) + list(np.linspace(104, 101, 4)) + list(np.linspace(101, 103, 4))
    bars, eng = _leg_engine(closes)
    s0 = swing(0, "low", 0, 99.9)
    s1 = swing(1, "high", 4, 104.05, "HH")
    s2 = swing(2, "low", 8, 100.95, "HL")
    s3 = swing(3, "high", 12, 103.05, "LH")  # failed to exceed the prior high
    legs = [eng.confirmed_leg(s0, s1), eng.confirmed_leg(s1, s2), eng.confirmed_leg(s2, s3)]
    eng.classify(legs, ref_dir=1)
    assert legs[2].classification == "FAILED_IMPULSE"
    eng.classify(legs, ref_dir=0)
    assert {lg.classification for lg in legs} == {"UNCLEAR"}


def test_multiple_historical_legs_stored(scenario_results):
    sc, res = scenario_results["clean_uptrend"]
    c = len(res.features) - 1
    det = res.context.details[c]
    assert len(det["legs"]) >= 6
    assert det["legs"][-1]["is_active"] is True
    assert len(res.context.legs) > 20
    kinds = {lg["classification"] for lg in det["legs"]}
    assert kinds & {"STRONG_IMPULSE", "NORMAL_IMPULSE"}
    assert kinds & {"HEALTHY_CORRECTION", "DEEP_CORRECTION", "CHOPPY_CORRECTION"}


def test_uptrend_hierarchy_on_engine_output(scenario_results):
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    assert (cf["primary_structure"] == "BULLISH").mean() > 0.9
    assert (cf["intermediate_structure"] == "BULLISH").mean() > 0.8
    assert cf["primary_structure_confidence"].between(0, 100).all()
    sc2, res2 = scenario_results["clean_downtrend"]
    cf2 = res2.context.frame.iloc[sc2.segment_start + 60:]
    assert (cf2["primary_structure"] == "BEARISH").mean() > 0.9
