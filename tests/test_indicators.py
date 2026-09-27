from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.features.indicators import (
    adx,
    atr,
    efficiency_ratio,
    ema,
    rolling_percentile,
    true_range,
    wilder_smooth,
)


def test_ema_matches_manual_recursion():
    x = pd.Series(np.linspace(100, 110, 40) + np.sin(np.arange(40)))
    p = 10
    e = ema(x, p)
    alpha = 2 / (p + 1)
    manual = [x.iloc[0]]
    for v in x.iloc[1:]:
        manual.append(alpha * v + (1 - alpha) * manual[-1])
    assert e.iloc[: p - 1].isna().all()
    np.testing.assert_allclose(e.iloc[p - 1 :].to_numpy(), np.array(manual)[p - 1 :], rtol=1e-12)


def test_wilder_smooth_seed_and_recursion():
    x = np.arange(1, 21, dtype=float)
    out = wilder_smooth(x, 5)
    assert np.isnan(out[:4]).all()
    assert out[4] == pytest.approx(3.0)  # mean of 1..5
    assert out[5] == pytest.approx((3.0 * 4 + 6) / 5)


def test_wilder_smooth_skips_leading_nan():
    x = np.array([np.nan, np.nan, 2, 2, 2, 2, 4], dtype=float)
    out = wilder_smooth(x, 3)
    assert np.isnan(out[:4]).all()
    assert out[4] == pytest.approx(2.0)
    assert out[6] == pytest.approx((2.0 * 2 + 4) / 3)


def test_true_range_and_atr():
    h = pd.Series([10.0, 11.0, 12.0, 11.5])
    l = pd.Series([9.0, 10.0, 10.5, 10.0])
    c = pd.Series([9.5, 10.8, 11.0, 10.2])
    tr = true_range(h, l, c)
    assert list(tr) == pytest.approx([1.0, 1.5, 1.5, 1.5])
    a = atr(h, l, c, 2)
    assert np.isnan(a.iloc[0])
    assert a.iloc[1] == pytest.approx(1.25)
    assert a.iloc[2] == pytest.approx((1.25 + 1.5) / 2)


def test_atr_constant_range():
    n = 60
    c = pd.Series(np.full(n, 190.0))
    a = atr(c + 0.25, c - 0.25, c, 14)
    assert a.iloc[-1] == pytest.approx(0.5)


def test_adx_uptrend_and_downtrend():
    n = 120
    base = pd.Series(150 + np.arange(n) * 0.3)
    up = adx(base + 0.2, base - 0.2, base, 14)
    assert up["plus_di"].iloc[-1] > up["minus_di"].iloc[-1]
    assert up["adx"].iloc[-1] > 40
    down_c = pd.Series(150 - np.arange(n) * 0.3)
    dn = adx(down_c + 0.2, down_c - 0.2, down_c, 14)
    assert dn["minus_di"].iloc[-1] > dn["plus_di"].iloc[-1]
    assert dn["adx"].iloc[-1] > 40


def test_adx_low_in_alternating_market():
    n = 200
    c = pd.Series(150 + 0.5 * np.sign(np.sin(np.arange(n) * np.pi / 2 + 0.1)))
    r = adx(c + 0.3, c - 0.3, c, 14)
    assert r["adx"].iloc[-1] < 20


def test_rolling_percentile_is_trailing():
    x = pd.Series(np.arange(1, 11, dtype=float))
    p = rolling_percentile(x, 5, 5)
    assert p.iloc[-1] == pytest.approx(100.0)  # latest is the max of its trailing window
    y = pd.Series([5, 4, 3, 2, 1], dtype=float)
    assert rolling_percentile(y, 5, 5).iloc[-1] == pytest.approx(20.0)


def test_efficiency_ratio_extremes():
    straight = pd.Series(np.arange(30, dtype=float))
    assert efficiency_ratio(straight, 10).iloc[-1] == pytest.approx(1.0)
    zigzag = pd.Series([0.0, 1.0] * 15)
    assert efficiency_ratio(zigzag, 10).iloc[-1] == pytest.approx(0.0)
    # 150 pips net over 700 pips travelled
    moves = [0.85, -0.55] * 5  # net 1.50 JPY, travelled 7.00 JPY
    path = np.concatenate([[0.0], np.cumsum(moves)])
    er = efficiency_ratio(pd.Series(path), 10).iloc[-1]
    assert er == pytest.approx(1.5 / 7.0)
