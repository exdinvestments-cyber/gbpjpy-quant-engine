"""Directional efficiency and range/chop detection.

``chop_score`` (0-100, higher = choppier) is a weighted mean of normalised
components, each in [0, 1]:

  efficiency       1 - efficiency, scaled: low net displacement vs path travelled
  choppiness       Choppiness Index (38 -> 0, 62 -> 1)
  overlap          mean overlap between consecutive candles
  crossings        EMA-baseline crossings in the window
  flat_baseline    EMA50 slope close to zero
  compressed_emas  EMA20/EMA50 separation small in ATR
  weak_adx         low ADX
  break_conflict   structural breaks in both directions
  structure_instability  frequent structure-state changes (quality window)
  small_displacement  |net move| small relative to ATR * sqrt(window)

Several components are correlated (efficiency / choppiness / displacement);
they are deliberately given modest individual weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ChopConfig, EfficiencyConfig
from .indicators import efficiency_ratio, rolling_percentile, scale01, true_range

CHOP_WEIGHTS = {
    "efficiency": 0.18,
    "choppiness": 0.10,
    "overlap": 0.12,
    "crossings": 0.14,
    "flat_baseline": 0.08,
    "compressed_emas": 0.08,
    "weak_adx": 0.10,
    "break_conflict": 0.08,
    "structure_instability": 0.07,
    "small_displacement": 0.05,
}


def efficiency_features(close: pd.Series, cfg: EfficiencyConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    er = efficiency_ratio(close, cfg.lookback)
    out["directional_efficiency"] = er
    out["directional_efficiency_signed"] = er * np.sign(close - close.shift(cfg.lookback))
    out["directional_efficiency_percentile"] = rolling_percentile(er, cfg.percentile_lookback, cfg.percentile_min_periods)
    return out


def choppiness_index(df: pd.DataFrame, period: int) -> pd.Series:
    tr = true_range(df["high"], df["low"], df["close"])
    s = tr.rolling(period, min_periods=period).sum()
    rng = df["high"].rolling(period, min_periods=period).max() - df["low"].rolling(period, min_periods=period).min()
    return 100.0 * np.log10(s / rng.replace(0.0, np.nan)) / np.log10(period)


def chop_features(
    df: pd.DataFrame,
    atr_s: pd.Series,
    trend: pd.DataFrame,
    adx_df: pd.DataFrame,
    candles: pd.DataFrame,
    efficiency: pd.DataFrame,
    structure_frame: pd.DataFrame,
    cfg: ChopConfig,
) -> pd.DataFrame:
    close = df["close"]
    w = cfg.window
    out = pd.DataFrame(index=df.index)
    out["choppiness_index"] = choppiness_index(df, cfg.choppiness_period)
    out["overlap_mean"] = candles["overlap_prev"].rolling(w, min_periods=w).mean()
    side = np.sign(close - trend["ema_mid"])
    crossings = (side != side.shift(1)) & side.notna() & side.shift(1).notna() & (side != 0)
    out["baseline_crossings"] = crossings.astype(float).rolling(w, min_periods=w).sum()
    out["net_displacement_atr"] = (close - close.shift(w)) / atr_s

    up = structure_frame["breaks_up_window"].astype(float)
    dn = structure_frame["breaks_down_window"].astype(float)
    tot = up + dn

    comp = pd.DataFrame(index=df.index)
    comp["efficiency"] = scale01(0.45 - efficiency["directional_efficiency"], 0.0, 0.35)
    comp["choppiness"] = scale01(out["choppiness_index"], 38.2, 61.8)
    comp["overlap"] = scale01(out["overlap_mean"], 0.35, 0.80)
    comp["crossings"] = scale01(out["baseline_crossings"], 0.0, 6.0)
    comp["flat_baseline"] = 1.0 - scale01(trend["ema_mid_slope_atr"].abs(), 0.0, cfg.flat_slope_atr_per_bar)
    comp["compressed_emas"] = 1.0 - scale01(trend["ema_separation_atr"].abs(), 0.0, cfg.compressed_separation_atr)
    comp["weak_adx"] = 1.0 - scale01(adx_df["adx"], 15.0, 30.0)
    comp["break_conflict"] = np.where(tot > 0, 2.0 * np.minimum(up, dn) / tot.replace(0, np.nan), 0.0)
    comp["structure_instability"] = scale01(structure_frame["sq_state_changes"].astype(float), 1.0, 5.0)
    comp["small_displacement"] = 1.0 - scale01(out["net_displacement_atr"].abs() / np.sqrt(w), 0.0, 1.5)

    # preserve NaN where inputs are not yet available
    inputs_ok = (
        efficiency["directional_efficiency"].notna()
        & out["choppiness_index"].notna()
        & out["overlap_mean"].notna()
        & trend["ema_mid_slope_atr"].notna()
        & adx_df["adx"].notna()
        & out["net_displacement_atr"].notna()
    )
    for k in CHOP_WEIGHTS:
        out[f"chop_c_{k}"] = comp[k].where(inputs_ok)
    score = sum(comp[k] * wt for k, wt in CHOP_WEIGHTS.items()) / sum(CHOP_WEIGHTS.values()) * 100.0
    out["chop_score"] = score.where(inputs_ok).round(2)

    ts = trend["trend_strength"]
    cs = out["chop_score"]
    out["market_quality"] = np.select(
        [
            cs.isna() | ts.isna(),
            cs >= cfg.severe_threshold,
            cs >= cfg.range_threshold,
            (cs < cfg.clean_threshold) & (ts >= cfg.clean_trend_strength),
            ts >= cfg.noisy_trend_strength,
        ],
        ["unknown", "severe_chop", "range", "clean_trend", "trend_with_noise"],
        default="transition",
    )
    return out
