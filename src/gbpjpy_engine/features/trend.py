"""Trend engine (EMA-based) and ADX/DI directional-strength features.

The trend engine is independent of the swing-structure classifier.  Its
output is ONE family score: EMA ordering, slopes, price position, separation,
baseline distance and alignment persistence are highly correlated, so they are
blended into a single ``trend_score`` rather than counted as independent
confirmations.  ADX/DI describe directional strength and never produce a
signal on their own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import AdxConfig, TrendConfig
from .indicators import adx, ema

TREND_WEIGHTS = {
    "order": 0.25,
    "slope": 0.25,
    "position": 0.15,
    "separation": 0.15,
    "distance": 0.05,
    "persistence": 0.15,
}


def trend_features(close: pd.Series, atr_s: pd.Series, cfg: TrendConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    e_f, e_m, e_s = ema(close, cfg.ema_fast), ema(close, cfg.ema_mid), ema(close, cfg.ema_slow)
    out["ema_fast"], out["ema_mid"], out["ema_slow"] = e_f, e_m, e_s
    L = cfg.slope_lookback
    per_bar = L * atr_s
    out["ema_fast_slope_atr"] = (e_f - e_f.shift(L)) / per_bar
    out["ema_mid_slope_atr"] = (e_m - e_m.shift(L)) / per_bar
    out["ema_slow_slope_atr"] = (e_s - e_s.shift(L)) / per_bar
    out["ema_separation_atr"] = (e_f - e_m) / atr_s
    out["price_to_ema_mid_atr"] = (close - e_m) / atr_s

    comp = pd.DataFrame(index=close.index)
    comp["order"] = (np.sign(e_f - e_m) + np.sign(e_m - e_s) + np.sign(e_f - e_s)) / 3.0
    comp["slope"] = (
        0.4 * np.tanh(out["ema_fast_slope_atr"] / cfg.slope_norm_atr_per_bar)
        + 0.4 * np.tanh(out["ema_mid_slope_atr"] / cfg.slope_norm_atr_per_bar)
        + 0.2 * np.tanh(out["ema_slow_slope_atr"] / cfg.slope_norm_atr_per_bar)
    )
    comp["position"] = (np.sign(close - e_f) + np.sign(close - e_m) + np.sign(close - e_s)) / 3.0
    comp["separation"] = np.tanh(out["ema_separation_atr"] / cfg.separation_norm_atr)
    comp["distance"] = np.tanh(out["price_to_ema_mid_atr"] / cfg.distance_norm_atr)
    comp["persistence"] = np.sign(e_f - e_m).rolling(cfg.persistence_window, min_periods=cfg.persistence_window).mean()
    for k in TREND_WEIGHTS:
        out[f"trend_c_{k}"] = comp[k]

    score = sum(comp[k] * w for k, w in TREND_WEIGHTS.items()) * 100.0
    out["trend_score"] = score.round(2)
    out["trend_strength"] = score.abs().round(2)
    out["trend_state"] = np.select(
        [
            score.isna(),
            score >= cfg.strong_threshold,
            score >= cfg.trend_threshold,
            score <= -cfg.strong_threshold,
            score <= -cfg.trend_threshold,
        ],
        ["unknown", "strong_bullish", "bullish", "strong_bearish", "bearish"],
        default="neutral",
    )
    return out


def adx_features(df: pd.DataFrame, cfg: AdxConfig) -> pd.DataFrame:
    a = adx(df["high"], df["low"], df["close"], cfg.period)
    out = pd.DataFrame(index=df.index)
    out["adx"] = a["adx"]
    out["plus_di"] = a["plus_di"]
    out["minus_di"] = a["minus_di"]
    out["adx_slope"] = a["adx"] - a["adx"].shift(cfg.slope_lookback)
    out["di_spread"] = a["plus_di"] - a["minus_di"]
    out["directional_persistence"] = (
        np.sign(out["di_spread"]).rolling(cfg.persistence_window, min_periods=cfg.persistence_window).mean()
    )
    out["adx_state"] = np.select(
        [out["adx"].isna(), out["adx"] >= cfg.strong_adx, out["adx"] < cfg.weak_adx],
        ["unknown", "directional", "weak"],
        default="moderate",
    )
    return out
