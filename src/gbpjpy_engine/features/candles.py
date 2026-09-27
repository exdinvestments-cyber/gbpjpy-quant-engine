"""Per-candle descriptive geometry.  Descriptive features only - never signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CandleConfig


def candle_features(df: pd.DataFrame, prev_atr: pd.Series, cfg: CandleConfig) -> pd.DataFrame:
    """Candle geometry for each bar.

    ``prev_atr`` must be the ATR known BEFORE the bar (ATR shifted by one), so
    that a huge bar is not normalised by an ATR it has itself inflated.
    """
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    safe = rng.replace(0.0, np.nan)
    body = (c - o).abs()
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    out = pd.DataFrame(index=df.index)
    out["candle_range"] = rng
    out["candle_body"] = body
    out["body_range_pct"] = (body / safe * 100).fillna(0.0)
    out["upper_wick"] = upper
    out["lower_wick"] = lower
    out["upper_wick_pct"] = (upper / safe * 100).fillna(0.0)
    out["lower_wick_pct"] = (lower / safe * 100).fillna(0.0)
    out["range_atr"] = rng / prev_atr
    out["close_location"] = ((c - l) / safe).fillna(0.5)
    out["candle_direction"] = np.sign(c - o).astype(int)

    br = out["body_range_pct"] / 100
    loc = out["close_location"]
    uw = out["upper_wick_pct"] / 100
    lw = out["lower_wick_pct"] / 100
    abnormal = (out["range_atr"] >= cfg.abnormal_range_atr).fillna(False)
    strong_bull = (c > o) & (br >= cfg.strong_body_ratio) & (loc >= cfg.strong_close_location)
    strong_bear = (c < o) & (br >= cfg.strong_body_ratio) & (loc <= 1 - cfg.strong_close_location)
    rejection_up = (lw >= cfg.rejection_wick_ratio) & (br <= cfg.rejection_max_body)  # rejected lower prices
    rejection_down = (uw >= cfg.rejection_wick_ratio) & (br <= cfg.rejection_max_body)
    indecision = br <= cfg.indecision_body_ratio

    labels = np.select(
        [strong_bull, strong_bear, rejection_up, rejection_down, indecision, c > o, c < o],
        [
            "strong_bullish_close",
            "strong_bearish_close",
            "bullish_rejection",
            "bearish_rejection",
            "indecision",
            "weak_bullish_close",
            "weak_bearish_close",
        ],
        default="indecision",
    )
    out["candle_class"] = labels
    out["abnormal_expansion"] = abnormal.astype(bool)

    # overlap with previous candle (fraction of the smaller range) - used by chop & structure quality
    ov = (np.minimum(h, h.shift(1)) - np.maximum(l, l.shift(1))).clip(lower=0.0)
    denom = np.minimum(rng, rng.shift(1)).replace(0.0, np.nan)
    out["overlap_prev"] = (ov / denom).clip(0.0, 1.0)
    out["wick_ratio"] = ((upper + lower) / safe).fillna(1.0)
    return out
