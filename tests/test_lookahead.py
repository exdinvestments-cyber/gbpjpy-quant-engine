"""Explicit anti-look-ahead tests.

The core property: the evaluation of H4 bar T must be a function of bars
0..T only.  We verify it two ways:

1. Truncation - running the engine on bars[:T+1] reproduces rows 0..T of the
   full run exactly.
2. Perturbation - replacing every bar after T with different data leaves rows
   0..T unchanged.

Plus targeted checks that swings, breaks and zones only become visible at
their confirmation / availability time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.synthetic import random_walk_bars
from helpers import make_bars

CUTS = [120, 260, 400, 555, 700]


def _assert_rows_equal(a: pd.DataFrame, b: pd.DataFrame, upto: int) -> None:
    assert list(a.columns) == list(b.columns)
    a = a.iloc[: upto + 1]
    b = b.iloc[: upto + 1]
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            same = (xv == yv) | (np.isnan(xv) & np.isnan(yv))
            assert same.all(), f"look-ahead contamination in column {col!r} at rows {np.flatnonzero(~same)[:5]}"
        else:
            xs = [repr(v) if not (isinstance(v, float) and np.isnan(v)) else "nan" for v in x]
            ys = [repr(v) if not (isinstance(v, float) and np.isnan(v)) else "nan" for v in y]
            bad = [i for i, (p, q) in enumerate(zip(xs, ys)) if p != q]
            assert not bad, f"look-ahead contamination in column {col!r} at rows {bad[:5]}: {xs[bad[0]]} vs {ys[bad[0]]}"


@pytest.mark.parametrize("cut", CUTS)
def test_truncation_invariance(engine, rw_bars, rw_result, cut):
    partial = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
    _assert_rows_equal(rw_result.features, partial.features, cut)
    # snapshots (including zones and reasons) identical
    assert partial.snapshot_at(cut).to_dict() == rw_result.snapshot_at(cut).to_dict()


@pytest.mark.parametrize("cut", [300, 500, 650])
def test_future_perturbation_cannot_change_past(engine, rw_bars, rw_result, cut):
    alt_future = random_walk_bars(n=len(rw_bars) + 5, seed=999)
    # splice: identical up to cut, then completely different (but continuous-in-time) future
    shift = rw_bars["close"].iloc[cut] - alt_future["open"].iloc[cut + 1]
    fut = alt_future.iloc[cut + 1 : len(rw_bars)].copy()
    for c in ("open", "high", "low", "close"):
        fut[c] = fut[c] + shift
    fut["timestamp"] = rw_bars["timestamp"].iloc[cut + 1 :].to_numpy()
    spliced = pd.concat([rw_bars.iloc[: cut + 1], fut], ignore_index=True)
    spliced.attrs = rw_bars.attrs
    res = engine.run(spliced)
    # future genuinely differs
    assert not np.allclose(res.features["close"].iloc[cut + 1 :], rw_result.features["close"].iloc[cut + 1 :])
    _assert_rows_equal(rw_result.features, res.features, cut)
    for i in (cut - 50, cut - 1, cut):
        assert res.snapshot_at(i).to_dict() == rw_result.snapshot_at(i).to_dict()


def test_swings_never_visible_before_confirmation(rw_result, engine):
    R = engine.config.swing.right_bars
    tf = pd.Timedelta(minutes=engine.config.data.timeframe_minutes)
    feats = rw_result.features
    assert rw_result.structure.swings, "expected swings in random walk data"
    for s in rw_result.structure.swings:
        assert s.confirm_index >= s.pivot_index + R
        assert s.confirmed_at == feats["timestamp"].iloc[s.confirm_index] + tf
        # the bar before confirmation cannot know this swing
        if s.confirm_index - 1 >= 0:
            known_before = rw_result.structure.swings_known_at(s.confirm_index - 1)
            assert s.swing_id not in {k.swing_id for k in known_before}
    # per-bar outputs: the reported last swing's confirmation is never after the bar's close
    for col in ("last_swing_high_confirmed_at", "last_swing_low_confirmed_at"):
        known = feats[col].notna()
        assert (feats.loc[known, col] <= feats.loc[known, "available_at"]).all()


def test_pivot_detected_with_right_bars_latency(engine):
    # clear high pivot at bar 30 (V shape up then down); ATR warm-up via preceding noise
    base = list(185 + 0.3 * np.sin(np.arange(30)))
    up = [base[-1] + 0.4 * k for k in range(1, 8)]
    down = [up[-1] - 0.4 * k for k in range(1, 12)]
    bars = make_bars(base + up + down)
    res = engine.run(bars)
    pivot_idx = 30 + 6  # top of the "up" leg
    highs = [s for s in res.structure.swings if s.kind == "high" and s.pivot_index == pivot_idx]
    assert highs, "expected the V-top to become a swing high"
    s = highs[0]
    assert s.confirm_index == pivot_idx + engine.config.swing.right_bars
    f = res.features
    assert f["last_swing_high"].iloc[s.confirm_index - 1] != pytest.approx(s.price)
    assert f["last_swing_high"].iloc[s.confirm_index] == pytest.approx(s.price)


def test_breaks_only_use_known_levels(rw_result):
    for b in rw_result.structure.breaks:
        sw = rw_result.structure.swings[b.swing_id]
        # the level must have been confirmed strictly before the breaking bar was processed
        assert sw.accepted_index < b.break_index
        assert b.available_at > sw.confirmed_at
        # status history never resolves before the break or on the break bar itself
        for idx, status in b.status_history[1:]:
            assert idx > b.break_index


def test_zones_built_only_from_known_information(rw_result):
    feats = rw_result.features
    for c in (250, 400, 600, 850):
        close_c = feats["close"].iloc[c]
        max_high = feats["high"].iloc[: c + 1].max()
        min_low = feats["low"].iloc[: c + 1].min()
        for z in rw_result.zones_by_bar[c]:
            # every zone must be derivable from prices seen up to c
            assert z.lower >= min_low - 1.0 and z.upper <= max_high + 1.0
            assert z.last_interaction_index is None or z.last_interaction_index <= c
        assert close_c > 0


def test_unclosed_bar_is_excluded(engine, rw_bars):
    last_open = rw_bars["timestamp"].iloc[-1]
    as_of = last_open + pd.Timedelta(hours=2)  # last bar still forming
    res = engine.run(rw_bars, as_of=as_of)
    assert len(res.features) == len(rw_bars) - 1
    assert res.features["available_at"].iloc[-1] <= as_of


def test_harness_detects_a_deliberately_leaky_feature(rw_bars):
    """Canary: the comparison used above must FAIL for a feature that peeks ahead."""

    def leaky(bars: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"centered_mean": bars["close"].rolling(5, center=True).mean()})

    full = leaky(rw_bars)
    part = leaky(rw_bars.iloc[:401])
    with pytest.raises(AssertionError, match="look-ahead contamination"):
        _assert_rows_equal(full, part, 400)


def test_pivot_price_not_used_by_features_before_confirmation(engine, rw_bars, rw_result):
    """Changing the bars that CONFIRM a pivot must not alter anything before confirmation."""
    s = next(sw for sw in rw_result.structure.swings if sw.pivot_index > 300 and sw.kind == "high")
    alt = rw_bars.copy()
    # destroy the pivot by lifting every right-side bar far above it
    for j in range(s.pivot_index + 1, s.confirm_index + 1):
        for c in ("open", "high", "low", "close"):
            alt.loc[j, c] = alt.loc[j, c] + 5.0
    res = engine.run(alt)
    # everything up to and including the pivot bar is unchanged ...
    _assert_rows_equal(rw_result.features, res.features, s.pivot_index)
    # ... and in the original run, the swing was unknown until its confirmation bar
    known_before = rw_result.structure.swings_known_at(s.confirm_index - 1)
    assert s.swing_id not in {k.swing_id for k in known_before}
    # in the altered run the pivot is never confirmed at that price
    assert not any(sw.pivot_index == s.pivot_index and sw.kind == s.kind for sw in res.structure.swings)
