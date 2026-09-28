"""Phase 1B: premium/discount, retracement, trend maturity, momentum deterioration, compression, expansion."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ctx_helpers import price_arrays, swing, zigzag
from gbpjpy_engine.config import CompressionConfig, LocationContextConfig, MaturityConfig
from gbpjpy_engine.context.location import premium_discount, retracement
from gbpjpy_engine.context.maturity import ExpansionTracker, compression, deterioration, maturity, trend_run

L = LocationContextConfig()


@pytest.mark.parametrize("close,state", [(100.5, "DEEP_DISCOUNT"), (103.0, "DISCOUNT"), (105.0, "EQUILIBRIUM"),
                                         (107.0, "PREMIUM"), (109.5, "DEEP_PREMIUM")])
def test_premium_discount_bands(close, state):
    mem = [swing(0, "low", 0, 100.0), swing(1, "high", 6, 110.0)]
    r = premium_discount(mem, close, 1.0, L)
    assert r["premium_discount_state"] == state and r["range_valid"]
    assert r["structural_range_percentile"] == pytest.approx((close - 100) / 10 * 100)


def test_premium_discount_outside_and_invalid_ranges():
    mem = [swing(0, "low", 0, 100.0), swing(1, "high", 6, 110.0)]
    r = premium_discount(mem, 112.0, 1.0, L)
    assert r["range_position"] == "above" and r["structural_range_percentile"] == 100.0
    assert r["structural_range_percentile_raw"] == pytest.approx(120.0)
    narrow = [swing(0, "low", 0, 100.0), swing(1, "high", 6, 100.5)]
    assert premium_discount(narrow, 100.2, 1.0, L)["premium_discount_state"] == "UNDEFINED"


def _impulse(end_index=5, start=100.0, end=110.0, direction="up"):
    return SimpleNamespace(distance=abs(end - start), end_index=end_index, end_price=end, direction=direction, leg_id="0-1")


@pytest.mark.parametrize("low,band", [(108.0, "SHALLOW"), (105.5, "NORMAL"), (103.0, "DEEP"), (101.0, "VERY_DEEP")])
def test_retracement_bands(low, band):
    n = 12
    c = np.full(n, 109.0)
    arr = price_arrays(c, c + 0.2, np.where(np.arange(n) == 8, low, c - 0.2), c)
    r = retracement(_impulse(), arr, 10, 1.0, L)
    assert r["retracement_band"] == band
    assert r["retracement_depth_pct"] == pytest.approx((110 - low) / 10 * 100)
    assert r["retracement_duration_bars"] == 5 and r["retracement_atr"] == pytest.approx(110 - low)
    for k in ("retracement_velocity_atr", "retracement_efficiency", "retracement_current_pct"):
        assert r[k] is not None


def test_retracement_point_in_time():
    n = 12
    c = np.full(n, 109.0)
    lows = c - 0.2
    lows[9] = 101.0
    arr = price_arrays(c, c + 0.2, lows, c)
    assert retracement(_impulse(), arr, 8, 1.0, L)["retracement_band"] == "SHALLOW"  # deep low not yet printed
    assert retracement(_impulse(), arr, 9, 1.0, L)["retracement_band"] == "VERY_DEEP"


def _leg(direction, dist, vel=0.5, score=70, eff=0.6, ov=0.3, ratio=None, cls="NORMAL_IMPULSE", s=0, e=1):
    return SimpleNamespace(direction=direction, distance_atr=dist, velocity_atr=vel, impulse_score=score, efficiency=eff,
                           overlap=ov, retracement_ratio=ratio, classification=cls, is_active=False,
                           start_swing_id=s, end_swing_id=e)


def test_momentum_deterioration_detects_weakening():
    strong = [_leg("up", 4.0), _leg("up", 4.0), _leg("up", 3.8)]
    weak = [_leg("up", 4.0), _leg("up", 4.0), _leg("up", 1.5, vel=0.2, score=40, eff=0.3, ov=0.5)]
    cors_ok = [_leg("down", 1.0, ratio=0.3, cls="HEALTHY_CORRECTION"), _leg("down", 1.0, ratio=0.3, cls="HEALTHY_CORRECTION")]
    cors_deep = [_leg("down", 1.0, ratio=0.3, cls="HEALTHY_CORRECTION"), _leg("down", 2.0, ratio=0.8, cls="DEEP_CORRECTION")]
    s_ok, _ = deterioration(strong, cors_ok, strong + cors_ok, 0)
    s_bad, comps = deterioration(weak, cors_deep, weak + cors_deep, 1)
    assert s_ok < 15 < 60 <= s_bad
    assert comps["smaller_impulses"] == 1.0 and comps["deeper_corrections"] == 1.0 and comps["failed_extensions"] == 0.5


def test_trend_maturity_progression():
    cfg = MaturityConfig()
    short = zigzag([100, 104, 102, 106])
    long_ = zigzag([p for i in range(8) for p in (100 + 2 * i, 104 + 2 * i)])
    legs_s = [_leg("up", 4, s=0, e=1), _leg("down", 2, s=1, e=2), _leg("up", 4, s=2, e=3)]
    early = maturity(short, legs_s, 1, 18, 105.0, 1.0, "normal", "up", 0.0, cfg)
    # the first swing high has no earlier high to compare with (no label), so the run starts there: 1 impulse
    assert early["trend_maturity"] == "EARLY" and early["trend_impulses"] == 1
    legs_l = [_leg("up" if k % 2 == 0 else "down", 4, s=k, e=k + 1) for k in range(15)]
    late = maturity(long_, legs_l, 1, 200, long_[-1].price + 2, 1.0, "normal", "up", 0.0, cfg)
    assert late["trend_maturity"] in ("MATURE", "EXTENDED") and late["trend_impulses"] >= 5
    exh = maturity(long_, legs_l, 1, 200, long_[-1].price + 2, 1.0, "normal", "up", 80.0, cfg)
    assert exh["trend_maturity"] == "EXHAUSTION_RISK"
    assert maturity(long_, legs_l, 0, 200, 110, 1.0, "normal", "up", 0.0, cfg)["trend_maturity"] == "UNKNOWN"
    # the run's origin is the last swing without a consistent label (the first swing high has no prior high)
    assert trend_run(long_, 1)[0].swing_id == 1


def _comp_arrays(n=80, shrink=True):
    amp = np.linspace(1.0, 0.1, n) if shrink else np.full(n, 1.0)
    c = 100 + amp * np.sin(np.arange(n) * 1.3)
    o = np.concatenate([[c[0]], c[:-1]])
    arr = price_arrays(o, np.maximum(o, c) + 0.2 * amp, np.minimum(o, c) - 0.2 * amp, c)
    arr["atr_ratio"] = np.full(n, 0.7 if shrink else 1.1)
    arr["overlap_mean"] = np.full(n, 0.75 if shrink else 0.4)
    arr["eff"] = np.full(n, 0.1 if shrink else 0.5)
    return arr


def test_compression_score_higher_when_contracting():
    mem_c = zigzag([100, 104, 101, 103.5, 101.6, 103.0, 102.0, 102.6], atr=1.0)
    mem_t = zigzag([100, 104, 102, 106, 104, 108, 106, 110], atr=1.0)
    sc_c, comps = compression(mem_c, _comp_arrays(shrink=True), 79, CompressionConfig())
    sc_t, _ = compression(mem_t, _comp_arrays(shrink=False), 79, CompressionConfig())
    assert sc_c >= CompressionConfig().compression_threshold > sc_t
    assert comps["converging_structure"] == 1.0 and comps["amplitude_contraction"] > 0.5


def test_expansion_after_compression():
    tr = ExpansionTracker(CompressionConfig(), 40.0)
    brk = SimpleNamespace(direction="bullish", event_id=7, state_at=lambda c: "CONFIRMED" if c >= 12 else "CANDIDATE")
    out = None
    for c in range(10):
        out = tr.step(c, 70.0, 10.0, 5.0, [], {}, bar_range=0.2)
    assert out["expansion_state"] == "COMPRESSING" and out["compression_episode_bars"] == 10
    # small bar does not qualify even with displacement; big range bar with a break does
    assert tr.step(10, 70.0, 50.0, 0.0, [], {}, bar_range=0.25)["expansion_state"] == "COMPRESSING"
    out = tr.step(11, 30.0, 55.0, 0.0, [brk], {7: brk}, bar_range=0.8)
    assert out["expansion_state"] == "EXPANDING_BULLISH"
    ev = tr.events[0]
    assert ev.structural_break and ev.compression_duration >= 4 and ev.direction == "bullish"
    out = tr.step(12, 20.0, 30.0, 0.0, [], {7: brk}, bar_range=0.5)
    assert out["expansion_state"] == "EXPANSION_ACCEPTED" and out["latest_expansion"]["expansion_quality"] > 50
    assert ev.state_at(11) == "EXPANDING"  # history kept


def test_expansion_fades_without_break():
    tr = ExpansionTracker(CompressionConfig(), 40.0)
    for c in range(8):
        tr.step(c, 70.0, 0.0, 0.0, [], {}, bar_range=0.2)
    assert tr.step(8, 30.0, 0.0, 55.0, [], {}, bar_range=0.9)["expansion_state"] == "EXPANDING_BEARISH"
    states = [tr.step(c, 20.0, 0.0, 0.0, [], {}, bar_range=0.3)["expansion_state"] for c in range(9, 14)]
    assert states[-1] == "NONE" and tr.events[0].state_at(13) == "EXPANSION_FADED"


def test_maturity_and_compression_on_engine_output(scenario_results):
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    assert set(cf["trend_maturity"]) <= {"EARLY", "DEVELOPING", "MATURE", "EXTENDED", "EXHAUSTION_RISK"}
    assert cf["momentum_deterioration_score"].between(0, 100).all()
    sc2, res2 = scenario_results["low_volatility_compression"]
    late = res2.context.frame.iloc[sc2.segment_start + 120:]
    early_trend = cf["compression_score"].mean()
    assert late["compression_score"].mean() > early_trend
    assert (late["context_reason_codes"].map(lambda c: "COMPRESSION_PRESENT" in c)).any()
    assert (res2.context.frame["trend_maturity"].iloc[sc2.segment_start + 60:] == "UNKNOWN").mean() > 0.8
