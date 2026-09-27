"""Causal indicator primitives.

Every function here is strictly causal: the output at position ``t`` depends
only on inputs at positions ``<= t``.  This is verified by the anti-look-ahead
tests (``tests/test_lookahead.py``) which truncate / perturb future data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average (alpha = 2/(period+1)), seeded at the first value.

    Values before ``period`` observations are NaN (warm-up), which avoids
    presenting an un-converged EMA as meaningful.
    """
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RMA, seeded with the simple mean of the first ``period`` valid values.

    Leading NaNs are skipped: smoothing starts at the first non-NaN input.
    ``rma[t] = (rma[t-1] * (period - 1) + x[t]) / period``
    """
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid) == 0:
        return out
    start = valid[0]
    seed_end = start + period
    if seed_end > len(x) or np.isnan(x[start:seed_end]).any():
        return out
    out[seed_end - 1] = x[start:seed_end].mean()
    for t in range(seed_end, len(x)):
        xt = x[t]
        out[t] = out[t - 1] if np.isnan(xt) else (out[t - 1] * (period - 1) + xt) / period
    return out


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = true_range(high, low, close)
    return pd.Series(wilder_smooth(tr.to_numpy(), period), index=high.index)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.DataFrame:
    """Wilder ADX with +DI and -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_dm[0] = np.nan
    minus_dm[0] = np.nan
    tr = true_range(high, low, close).to_numpy().copy()
    tr[0] = np.nan
    atr_w = wilder_smooth(tr, period)
    pdm = wilder_smooth(plus_dm, period)
    mdm = wilder_smooth(minus_dm, period)
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100.0 * pdm / atr_w
        minus_di = 100.0 * mdm / atr_w
        denom = plus_di + minus_di
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    dx = np.where(np.isnan(plus_di), np.nan, dx)
    adx_v = wilder_smooth(dx, period)
    return pd.DataFrame({"adx": adx_v, "plus_di": plus_di, "minus_di": minus_di}, index=high.index)


def rolling_percentile(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Percentile rank (0-100) of the current value within its trailing window (inclusive)."""
    return series.rolling(window, min_periods=min_periods).rank(pct=True) * 100.0


def efficiency_ratio(close: pd.Series, lookback: int) -> pd.Series:
    """|close_t - close_{t-n}| / sum_{i=t-n+1..t} |close_i - close_{i-1}|  (Kaufman)."""
    net = (close - close.shift(lookback)).abs()
    path = close.diff().abs().rolling(lookback, min_periods=lookback).sum()
    return (net / path.replace(0.0, np.nan)).clip(0.0, 1.0)


def scale01(x, start: float, end: float):
    """Linear map of x from [start, end] onto [0, 1] with clipping."""
    return np.clip((np.asarray(x, dtype=float) - start) / (end - start), 0.0, 1.0)
