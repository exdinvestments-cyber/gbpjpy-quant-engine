"""Timeframe-generic causal feature pipeline.

Extracted verbatim from the Phase 1A H4 engine so the same, already-tested
feature stack can be reused for other timeframes (Phase 1C H1) without
duplicating it.  The H4 engine calls this function and its output is
bit-identical to the pre-extraction implementation.

``cfg`` is duck-typed: any config object exposing the Phase 1A sections
(``data``, ``engine``, ``swing``, ``structure``, ``trend``, ``adx``,
``volatility``, ``shock``, ``momentum``, ``candle``, ``chop``, ``efficiency``,
``levels``, ``round_numbers``, ``range_location``, ``extension``, ``sessions``)
works; ``cfg.data.timeframe_minutes`` sets the bar duration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.validation import DataQualityReport
from .candles import candle_features
from .chop import chop_features, efficiency_features
from .context import extension_features, range_location_features, round_number_features, session_features, time_features
from .levels import compute_levels
from .momentum import momentum_features
from .structure import compute_structure
from .trend import adx_features, trend_features
from .volatility import shock_features, volatility_features

_ERROR_TYPES = {
    "MISSING_PRICE", "NONPOSITIVE_PRICE", "INVALID_OHLC", "DUPLICATE_TIMESTAMP", "OUT_OF_ORDER", "NEGATIVE_SPREAD",
}
# Warnings that only say volume/spread are unavailable do not degrade price-based features.
_BENIGN_FLAGS = {"MISSING_VOLUME", "MISSING_SPREAD", "ZERO_VOLUME"}


def _bar_status(flags: list[str]) -> str:
    """ERROR > WARNING (price/time integrity concern) > INFO (only volume/spread unavailable) > OK."""
    if any(f in _ERROR_TYPES for f in flags):
        return "ERROR"
    if any(f not in _BENIGN_FLAGS for f in flags):
        return "WARNING"
    return "INFO" if flags else "OK"


def compute_feature_frame(df: pd.DataFrame, cfg, report: DataQualityReport):
    """Return (features, structure_result, zones_by_bar, zone_book) for canonical bars ``df``."""
    tf = pd.Timedelta(minutes=cfg.data.timeframe_minutes)

    vol = volatility_features(df, cfg.volatility)
    atr_s = vol["atr"]
    candles = candle_features(df, vol["atr_prev"], cfg.candle)
    shock = shock_features(df, vol, candles, cfg.shock)
    trend = trend_features(df["close"], atr_s, cfg.trend)
    adx_df = adx_features(df, cfg.adx)
    mom = momentum_features(df, atr_s, cfg.momentum)
    eff = efficiency_features(df["close"], cfg.efficiency)

    W = cfg.structure.quality_window_bars
    overlap_mean = candles["overlap_prev"].rolling(W, min_periods=5).mean()
    wick_mean = candles["wick_ratio"].rolling(W, min_periods=5).mean()
    structure = compute_structure(df, atr_s, overlap_mean, wick_mean, cfg.swing, cfg.structure, tf)
    chop = chop_features(df, atr_s, trend, adx_df, candles, eff, structure.frame, cfg.chop)
    levels, zones_by_bar, zone_book = compute_levels(df, atr_s, structure, cfg.levels, return_book=True)
    rn = round_number_features(df["close"], atr_s, cfg.round_numbers, cfg.data)
    rloc = range_location_features(df, cfg.range_location)
    ext = extension_features(df, trend, atr_s, cfg.extension)
    sess = session_features(df["timestamp"], cfg.sessions, tf)
    tfeat = time_features(df["timestamp"])

    base = df.copy()
    base.insert(1, "available_at", df["timestamp"] + tf)
    base.insert(0, "symbol", cfg.data.symbol)
    feats = pd.concat(
        [base, candles, vol, shock, trend, adx_df, mom, eff, structure.frame, chop, levels, rn, rloc, ext, sess, tfeat],
        axis=1,
    )
    feats = feats.loc[:, ~feats.columns.duplicated()].copy()

    warm = np.arange(len(df)) >= cfg.engine.warmup_bars
    feats["warmup_complete"] = warm
    flags = report.bar_flags if report.bar_flags else [[] for _ in range(len(df))]
    feats["data_quality_flags"] = [list(f) for f in flags]
    feats["data_quality_status"] = [_bar_status(fl) for fl in flags]
    return feats, structure, zones_by_bar, zone_book
