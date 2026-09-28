"""Swing detection, confirmation timing, HH/HL labelling, BOS / CHoCH timing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.config import StructureConfig, SwingConfig
from gbpjpy_engine.features.structure import compute_structure
from helpers import make_bars

TF = pd.Timedelta(hours=4)
WICK = 0.05


def legs(start: float, anchors: list[float], bars_per_leg: int | list[int] = 4):
    """Piecewise-linear closes through anchors; returns closes and the bar index of each anchor."""
    closes = [start]
    idx = []
    per = bars_per_leg if isinstance(bars_per_leg, list) else [bars_per_leg] * len(anchors)
    for a, n in zip(anchors, per):
        s = closes[-1]
        closes += list(np.linspace(s, a, n + 1)[1:])
        idx.append(len(closes) - 1)
    return closes, idx


def run(closes, highs=None, lows=None, swing=None, structure=None):
    bars = make_bars(closes, highs=highs, lows=lows)
    n = len(bars)
    atr = pd.Series(np.ones(n))
    half = pd.Series(np.full(n, 0.5))
    return bars, compute_structure(bars, atr, half, half, swing or SwingConfig(), structure or StructureConfig(), TF)


def test_uptrend_swings_labels_and_confirmation_timing():
    closes, idx = legs(100.0, [104, 102, 106, 104, 108, 106, 110, 108.8])
    bars, res = run(closes)
    R = SwingConfig().right_bars
    highs = [s for s in res.swings if s.kind == "high" and s.removed_index is None]
    lows = [s for s in res.swings if s.kind == "low" and s.removed_index is None]
    assert [round(s.price, 2) for s in highs] == [104.05, 106.05, 108.05, 110.05]
    assert [round(s.price, 2) for s in lows] == [101.95, 103.95, 105.95]
    assert [s.label for s in highs] == [None, "HH", "HH", "HH"]
    assert [s.label for s in lows] == [None, "HL", "HL"]
    for s, anchor in zip(highs, [idx[0], idx[2], idx[4], idx[6]]):
        assert s.pivot_index == anchor
        assert s.confirm_index == anchor + R
        assert s.confirmed_at == bars["timestamp"].iloc[anchor + R] + TF
    f = res.frame
    assert f["structure_state"].iloc[-1] == "bullish"
    # before the 2nd low is confirmed the structure cannot yet be bullish
    second_low = lows[1]
    assert f["structure_state"].iloc[second_low.confirm_index - 1] != "bullish" or f["n_swings"].iloc[second_low.confirm_index - 1] >= 4


def test_downtrend_is_bearish():
    closes, _ = legs(110.0, [106, 108, 104, 106, 102, 104, 100, 101.2])
    _, res = run(closes)
    assert res.frame["structure_state"].iloc[-1] == "bearish"
    alive = res.swings_known_at(len(closes) - 1)
    assert alive[-1].label in ("LH", "LL")


def test_noise_is_not_structural():
    rng = np.random.default_rng(0)
    closes = 100 + 0.25 * np.sin(np.arange(80) * 1.3) + rng.normal(0, 0.03, 80)
    _, res = run(closes)
    # with ATR = 1 and min_swing_atr = 1, oscillations of +-0.25 must not create alternating swings
    assert len(res.swings_known_at(79)) <= 1


def test_min_swing_filter_is_configurable():
    closes = list(100 + 0.4 * np.sin(np.arange(80) * 2 * np.pi / 10))  # swing ~0.9 ATR
    _, strict = run(closes)
    _, loose = run(closes, swing=SwingConfig(min_swing_atr=0.5))
    assert len(loose.swings_known_at(79)) > len(strict.swings_known_at(79))


def test_bos_requires_meaningful_close_and_timing():
    closes, idx = legs(100.0, [104, 102, 106, 104])
    level = 106.05  # swing high (anchor + wick)
    n0 = len(closes)
    extra_c = [104.8, 105.6, 105.95, 106.15, 106.40, 106.9, 107.3]
    closes = closes + extra_c
    highs = list(np.maximum(np.concatenate([[closes[0]], closes[:-1]]), closes) + WICK)
    lows = list(np.minimum(np.concatenate([[closes[0]], closes[:-1]]), closes) - WICK)
    highs[n0 + 2] = 106.6  # wick-only excursion above the level: NOT a break
    bars, res = run(closes, highs=highs, lows=lows)
    brk = [b for b in res.breaks if b.direction == "bullish" and b.level == pytest.approx(level)]
    assert len(brk) == 1
    b = brk[0]
    # close 106.15 is only 0.10 ATR beyond -> not significant; 106.40 (0.35 ATR) is the break bar
    assert b.break_index == n0 + 4
    assert b.magnitude == pytest.approx(106.40 - level)
    assert b.magnitude_atr == pytest.approx(0.35)
    assert b.closed_beyond
    assert b.status == "CONFIRMED"
    assert b.status_history == [(n0 + 4, "CANDIDATE"), (n0 + 6, "CONFIRMED")]
    assert b.break_time == bars["timestamp"].iloc[n0 + 4]
    f = res.frame
    assert f["last_break_status"].iloc[n0 + 4] == "CANDIDATE"
    assert f["last_break_status"].iloc[n0 + 6] == "CONFIRMED"
    assert pd.isna(f["last_break_level"].iloc[n0 + 3]) or f["last_break_level"].iloc[n0 + 3] != pytest.approx(level)


def test_false_break_is_rejected():
    closes, _ = legs(100.0, [104, 102, 106, 104])
    n0 = len(closes)
    closes = closes + [105.0, 106.0, 106.6, 105.4, 105.0]  # close through, then straight back inside
    _, res = run(closes)
    b = [b for b in res.breaks if b.direction == "bullish"][-1]
    assert b.break_index == n0 + 2
    assert b.status == "INVALIDATED"
    assert b.status_history[-1] == (n0 + 3, "INVALIDATED")
    assert res.frame["last_break_status"].iloc[n0 + 2] == "CANDIDATE"
    assert res.frame["last_break_status"].iloc[n0 + 3] == "INVALIDATED"


def test_change_of_character_against_bullish_structure():
    closes, idx = legs(100.0, [104, 102, 106, 104, 108, 106, 110])
    n0 = len(closes)
    # drop hard below the last higher low (105.95)
    closes = closes + [109.0, 108.0, 107.0, 106.0, 105.4, 104.8, 104.5]
    _, res = run(closes)
    f = res.frame
    choch = [b for b in res.breaks if b.break_type == "CHOCH" and b.direction == "bearish"]
    assert choch, "expected a bearish change of character"
    b = choch[0]
    assert b.structure_before == "bullish"
    assert b.structure_after == "transitional"
    assert b.level == pytest.approx(105.95)
    assert f["structure_state"].iloc[b.break_index - 1] == "bullish"
    assert f["structure_state"].iloc[b.break_index] == "transitional"


def test_superseded_swing_is_recorded_not_rewritten():
    # high at 104, tiny dip (< 1 ATR), then higher high at 105: first high is superseded
    closes, _ = legs(100.0, [104, 103.6, 105, 102, 103], bars_per_leg=[4, 4, 4, 6, 4])
    _, res = run(closes)
    highs = [s for s in res.swings if s.kind == "high"]
    assert len(highs) >= 2
    old, new = highs[0], highs[1]
    assert old.removed_index == new.accepted_index
    assert old.replaced_by == new.swing_id
    f = res.frame
    # history still shows the old high until the moment it was superseded
    assert f["last_swing_high"].iloc[new.accepted_index - 1] == pytest.approx(old.price)
    assert f["last_swing_high"].iloc[new.accepted_index] == pytest.approx(new.price)


def test_structure_quality_higher_for_clean_trend():
    clean, _ = legs(100.0, [104, 102, 106, 104, 108, 106, 110, 108, 112, 110, 114])
    _, rc = run(clean)
    rng = np.random.default_rng(3)
    choppy = list(100 + np.cumsum(rng.choice([-1.2, 1.2], size=len(clean))) * 0.8)
    _, rch = run(choppy)
    assert rc.frame["structure_quality_score"].iloc[-1] > rch.frame["structure_quality_score"].iloc[-1]
    comps = [c for c in rc.frame.columns if c.startswith("sq_")]
    assert {"sq_clarity", "sq_impulse_pullback", "sq_persistence", "sq_overlap", "sq_wickiness"} <= set(comps)
