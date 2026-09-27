"""Volatility engine and volatility-shock detection.

Volatility is classified with RELATIVE historical distributions (trailing
percentile of ATR as a percentage of price), not fixed pip thresholds,
because GBPJPY's volatility level drifts substantially through time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ShockConfig, VolatilityConfig
from .indicators import atr, rolling_percentile, scale01


def volatility_features(df: pd.DataFrame, cfg: VolatilityConfig) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    out = pd.DataFrame(index=df.index)
    out["atr"] = atr(h, l, c, cfg.atr_period)
    out["atr_prev"] = out["atr"].shift(1)
    out["atr_pct"] = out["atr"] / c * 100.0
    out["atr_percentile"] = rolling_percentile(out["atr_pct"], cfg.percentile_lookback, cfg.percentile_min_periods)
    out["atr_short"] = atr(h, l, c, cfg.atr_short_period)
    out["atr_long"] = atr(h, l, c, cfg.atr_long_period)
    out["atr_ratio"] = out["atr_short"] / out["atr_long"]

    ratio = out["atr_ratio"]
    out["volatility_trend"] = np.select(
        [ratio >= cfg.expansion_ratio, ratio <= cfg.contraction_ratio, ratio.notna()],
        ["expanding", "contracting", "stable"],
        default="unknown",
    )
    p = out["atr_percentile"]
    out["volatility_regime"] = np.select(
        [p.isna(), p <= cfg.very_low_pct, p <= cfg.low_pct, p < cfg.high_pct, p < cfg.extreme_pct],
        ["unknown", "very_low", "low", "normal", "high"],
        default="extreme",
    )
    return out


SHOCK_COMPONENTS = ("range", "gap", "wick", "displacement", "atr_expansion")


def shock_features(df: pd.DataFrame, vol: pd.DataFrame, candles: pd.DataFrame, cfg: ShockConfig) -> pd.DataFrame:
    """Per-bar abnormal-behaviour severity.

    All normalisation uses the ATR known BEFORE the bar (``atr_prev``) so the
    bar being assessed cannot dampen its own abnormality.
    ``shock_severity`` is the maximum component severity (transparent: the
    triggering component is stored in ``shock_driver``).
    """
    prev_atr = vol["atr_prev"]
    prev_close = df["close"].shift(1)
    comps = {
        "range": scale01(candles["candle_range"] / prev_atr, cfg.range_atr_start, cfg.range_atr_extreme),
        "gap": scale01((df["open"] - prev_close).abs() / prev_atr, cfg.gap_atr_start, cfg.gap_atr_extreme),
        "wick": scale01(
            np.maximum(candles["upper_wick"], candles["lower_wick"]) / prev_atr, cfg.wick_atr_start, cfg.wick_atr_extreme
        ),
        "displacement": scale01(
            (df["close"] - prev_close).abs() / prev_atr, cfg.displacement_atr_start, cfg.displacement_atr_extreme
        ),
        "atr_expansion": scale01(vol["atr_ratio"], cfg.expansion_ratio_start, cfg.expansion_ratio_extreme),
    }
    out = pd.DataFrame(index=df.index)
    for k, v in comps.items():
        out[f"shock_{k}"] = np.nan_to_num(v, nan=0.0) * 100.0
    mat = out[[f"shock_{k}" for k in SHOCK_COMPONENTS]].to_numpy()
    out["shock_severity"] = mat.max(axis=1).round(2)
    out["shock_driver"] = np.where(
        out["shock_severity"] > 0, np.array(SHOCK_COMPONENTS, dtype=object)[mat.argmax(axis=1)], "none"
    )
    valid = prev_atr.notna()
    out.loc[~valid, "shock_severity"] = 0.0
    out.loc[~valid, "shock_driver"] = "none"
    out["volatility_shock"] = (out["shock_severity"] >= cfg.shock_threshold) & valid
    return out
