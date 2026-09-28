"""Phase 1D confirmation families on small constructed inputs (completed bars only)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gbpjpy_engine.entry import EntryConfig
from gbpjpy_engine.entry.confirmation import (GROUPS, break_retest, compression_expansion, confirmation_quality,
                                              displacement, reacceleration, structural_break, sweep_reclaim)

CC = EntryConfig().confirmation


def bars(ohlc, atr=0.5, disp_up=0.0, disp_dn=0.0, multibar=True, momentum=None):
    o, h, l, c = (np.array(x, float) for x in zip(*ohlc))
    n = len(o)
    return {"open": o, "high": h, "low": l, "close": c, "atr": np.full(n, atr),
            "range_atr": (h - l) / atr, "disp": {1: np.full(n, float(disp_up)), -1: np.full(n, float(disp_dn))},
            "multibar": {1: np.full(n, multibar), -1: np.full(n, multibar)},
            "momentum_net": np.zeros(n) if momentum is None else np.asarray(momentum, float)}


def flat(n, p=100.0):
    return [(p, p + 0.1, p - 0.1, p)] * n


def brk(idx, level, direction="bullish", state="CANDIDATE", mag=0.6, eid=0, swing=0):
    return SimpleNamespace(event_id=eid, direction=direction, break_index=idx, level=level, atr_at_break=0.5,
                           swing_id=swing, magnitude_atr=mag, break_type="BOS", state_at=lambda c, s=state: s)


SWINGS = [SimpleNamespace(significance_atr=2.5)]


# ------------------------------------------------------------------ structural break
def test_structural_break_confirmation_scores_meaningful_break_only():
    data = flat(10) + [(100.0, 100.62, 99.98, 100.6)]  # strong bullish close through 100.3
    a = bars(data, disp_up=70.0)
    zones = [[] for _ in data]
    ok = structural_break(10, 1, 8, [brk(10, 100.3)], a, SWINGS, zones, CC, 3.0)
    assert ok["detected"] and ok["score"] >= 55 and ok["reference_level"] == 100.3
    assert ok["evidence"]["close_beyond_atr"] == 0.6
    # before qualification / wrong direction / failed breaks never confirm
    assert not structural_break(10, 1, 10, [brk(10, 100.3)], a, SWINGS, zones, CC, 3.0)["detected"]
    assert not structural_break(10, 1, 8, [brk(10, 100.3, "bearish")], a, SWINGS, zones, CC, 3.0)["detected"]
    assert not structural_break(10, 1, 8, [brk(10, 100.3, state="FAILED")], a, SWINGS, zones, CC, 3.0)["detected"]
    # insignificant micro-structure (tiny swing, marginal close) scores low
    micro = structural_break(10, 1, 8, [brk(10, 100.55, mag=0.1)], bars(data), [SimpleNamespace(significance_atr=0.5)],
                             zones, CC, 3.0)
    assert micro["score"] < ok["score"] and micro["score"] < 55
    # follow-through policy: requiring CONFIRMED caps a fresh CANDIDATE break
    strict = replace(CC, structural_min_state="CONFIRMED")
    assert structural_break(10, 1, 8, [brk(10, 100.3)], a, SWINGS, zones, strict, 3.0)["score"] <= 40.0


# ------------------------------------------------------------------ displacement
def test_giant_isolated_candle_is_not_automatically_a_good_entry():
    data = flat(10) + [(100.0, 101.6, 99.95, 101.55)]  # 3.3 ATR candle
    multi = displacement(10, 1, bars(data, disp_up=58.0, multibar=True), CC, 3)
    lone = displacement(10, 1, bars(data, disp_up=58.0, multibar=False), CC, 3)
    assert multi["score"] == 58.0 and not multi["components"]["isolated_giant_candle"]
    assert lone["components"]["isolated_giant_candle"] and lone["score"] == pytest.approx(58.0 * 0.6)
    assert lone["score"] < 55  # below every default policy minimum


# ------------------------------------------------------------------ break-retest
def test_break_retest_valid_vs_breakout_failure():
    base = flat(5) + [(100.0, 100.5, 99.95, 100.45), (100.45, 100.6, 100.3, 100.5),
                      (100.5, 100.52, 100.08, 100.2)]  # break of 100.0 at bar 5; retest low 100.08 at bar 7
    good = base + [(100.2, 100.75, 100.18, 100.72)]  # directional response away from the level
    a = bars(good, disp_up=50.0)
    ev = brk(5, 100.0, state="CONFIRMED")
    r = break_retest(8, 1, [ev], a, CC)
    assert r["detected"] and r["evidence"]["status"] == "VALID_RETEST"
    assert r["evidence"]["retest_index"] == 7 and r["evidence"]["penetration_atr"] == 0.0
    assert r["evidence"]["retest_duration_bars"] == 1 and r["score"] >= 55
    # awaiting response on the retest bar itself
    assert break_retest(7, 1, [ev], bars(base, disp_up=50.0), CC)["evidence"]["status"] == "AWAITING_RESPONSE"
    # a deep close back through the level is a breakout failure, not a retest
    fail = base + [(100.2, 100.25, 99.5, 99.55)]
    f = break_retest(8, 1, [ev], bars(fail), CC)
    assert not f["detected"] and f["evidence"]["status"] == "BREAKOUT_FAILURE" and f["score"] == 0.0
    assert not break_retest(8, 1, [brk(5, 100.0, state="FAILED")], a, CC)["detected"]


# ------------------------------------------------------------------ sweep + reclaim
def _sweep(idx, level=99.7):
    return SimpleNamespace(event_id=3, index=idx, implication="bullish", level=level, atr=0.5,
                           state_at=lambda c: "SWEPT_AND_REJECTED")


def test_sweep_alone_never_confirms_reclaim_required():
    data = flat(5) + [(99.9, 100.05, 99.5, 99.95)]  # downside sweep of 99.7 at bar 5
    a = bars(data + [(99.95, 100.0, 99.9, 99.98)], disp_up=45.0)
    assert not sweep_reclaim(5, 1, [_sweep(5)], [], [], bars(data, disp_up=90.0), 80.0, CC)["detected"]  # sweep bar itself
    weak = sweep_reclaim(6, 1, [_sweep(5)], [], [], a, 80.0, CC)  # no close above the sweep candle
    assert not weak["detected"] and weak["score"] <= 40.0
    good = bars(data + [(99.95, 100.4, 99.94, 100.35)], disp_up=55.0)
    ok = sweep_reclaim(6, 1, [_sweep(5)], [], [], good, 80.0, CC)
    assert ok["detected"] and ok["reference_level"] == 100.05 and ok["evidence"]["sweep_extreme"] == 99.5
    # reclaim without displacement or structural consequence is still not enough
    assert not sweep_reclaim(6, 1, [_sweep(5)], [], [], bars(data + [(99.95, 100.4, 99.94, 100.35)], disp_up=10.0), 80.0,
                             CC)["detected"]


# ------------------------------------------------------------------ momentum reacceleration
def test_momentum_reacceleration_requires_resumed_momentum():
    data = flat(6) + [(100.0, 100.1, 99.9, 100.05), (100.05, 100.2, 100.0, 100.18), (100.18, 100.5, 100.15, 100.48)]
    up = reacceleration(8, 1, bars(data, disp_up=60.0, momentum=[0, 0, 0, 0, 0, -20, -10, 5, 25]), 0.8, CC)
    assert up["detected"] and up["score"] >= 55
    assert up["components"]["body_progression"] == 1.0 and up["components"]["structural_progress"] == 1.0
    fading = reacceleration(8, 1, bars(data, disp_up=60.0, momentum=[0, 0, 0, 0, 0, -20, -10, 25, 5]), 0.8, CC)
    assert not fading["detected"] and fading["score"] <= 30.0
    against = reacceleration(8, -1, bars(data, disp_dn=0.0, momentum=[0, 0, 0, 0, 0, -20, -10, 5, 25]), 0.8, CC)
    assert not against["detected"]


# ------------------------------------------------------------------ compression -> expansion
def test_compression_expansion_and_extreme_expansion_chase_cap():
    box = [(100.0, 100.05, 99.95, 100.0)] * 8
    exp_bar = [(100.0, 100.6, 99.98, 100.58)]  # 0.62 range vs 0.1 box ranges, 1.24 ATR
    ev = SimpleNamespace(event_id=1, index=8, direction="bullish", compression_start=0, break_event_id=None,
                         state_at=lambda c: "EXPANDING")
    r = compression_expansion(8, 1, [ev], {}, bars(box + exp_bar), 80.0, CC)
    assert r["detected"] and r["reference_level"] == 100.05 and not r["evidence"]["extreme_expansion_chase"]
    assert r["components"]["range_expansion"] == 1.0
    huge = compression_expansion(8, 1, [ev], {}, bars(box + [(100.0, 101.8, 99.98, 101.75)]), 80.0, CC)
    assert huge["evidence"]["extreme_expansion_chase"] and huge["score"] <= 40.0
    assert not compression_expansion(8, -1, [ev], {}, bars(box + exp_bar), 80.0, CC)["detected"]


# ------------------------------------------------------------------ confirmation confluence
def test_confirmation_quality_does_not_double_count_correlated_evidence():
    s = {f: 0.0 for fams in GROUPS.values() for f in fams}
    s["DISPLACEMENT_CONFIRMATION"] = 70.0
    q1, info = confirmation_quality(s, "DISPLACEMENT_CONFIRMATION", 50.0, 0.6)
    same_group = dict(s, COMPRESSION_EXPANSION_CONFIRMATION=90.0)  # same evidence group as the primary
    q2, _ = confirmation_quality(same_group, "DISPLACEMENT_CONFIRMATION", 50.0, 0.6)
    other_group = dict(s, STRUCTURAL_BREAK_CONFIRMATION=90.0)
    q3, _ = confirmation_quality(other_group, "DISPLACEMENT_CONFIRMATION", 50.0, 0.6)
    assert q2 == q1 < q3
    assert info["primary_group"] == "DISPLACEMENT" and set(info["groups"]) == {*GROUPS, "MARKET_QUALITY"}
    assert confirmation_quality(s, None, 50.0, 0.6)[0] == 0.0
