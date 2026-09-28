"""Objective multi-candle H4 displacement.

A single large candle is never enough: STRONG/EXTREME classes additionally
require a net multi-bar move of ``min_multi_bar_atr``, at least
``min_directional_closes`` directional closes and window efficiency of at
least ``min_efficiency``; otherwise the score is capped just below STRONG.

All normalisation uses ATR known BEFORE the window started (``atr`` shifted
by ``window``) so a displacement cannot dampen itself.

Components (bullish side shown; bearish mirrored), each in [0, 1]:

  body        0.15  signed body of the current bar / prior ATR     (0.3 -> 1.5)
  close       0.10  close location in bar                          (0.5 -> 0.9)
  range       0.05  range / prior ATR, only when closing strong    (0.8 -> 2.5)
  net_move    0.20  net close-to-close move over the window / ATR  (0.5 -> 2.5)
  efficiency  0.12  window directional efficiency                  (0.3 -> 0.8)
  low_overlap 0.08  1 - mean candle overlap in window              (0.3 -> 0.8)
  closes      0.10  share of directional closes in window          (0.34 -> 1)
  break       0.10  structural break in the same direction within the window
  percentile  0.10  |net move| percentile vs trailing history       (60 -> 98)

Velocity (net move per bar) is the same information as net_move and is stored
but not weighted separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DisplacementConfig
from ..features.indicators import rolling_percentile, scale01

WEIGHTS = {"body": 0.15, "close": 0.10, "range": 0.05, "net_move": 0.20, "efficiency": 0.12,
           "low_overlap": 0.08, "closes": 0.10, "break": 0.10, "percentile": 0.10}
CLASSES = ("NONE", "WEAK", "MODERATE", "STRONG", "EXTREME")


def _classify(score: np.ndarray, cfg: DisplacementConfig) -> np.ndarray:
    return np.select(
        [score >= cfg.extreme_score, score >= cfg.strong_score, score >= cfg.moderate_score, score >= cfg.weak_score],
        ["EXTREME", "STRONG", "MODERATE", "WEAK"], default="NONE",
    )


def displacement_features(feats: pd.DataFrame, cfg: DisplacementConfig) -> pd.DataFrame:
    w = cfg.window
    o, h, l, c = feats["open"], feats["high"], feats["low"], feats["close"]
    atr_ref = feats["atr"].shift(w)
    rng = (h - l)
    net = c - c.shift(w)
    path = c.diff().abs().rolling(w, min_periods=w).sum()
    eff = (net.abs() / path.replace(0.0, np.nan)).clip(0, 1).fillna(0.0)
    net_atr = net / atr_ref
    ov = feats["overlap_prev"].rolling(w, min_periods=1).mean()
    dirs = np.sign(c - o)
    up_closes = (dirs > 0).astype(float).rolling(w, min_periods=w).sum()
    dn_closes = (dirs < 0).astype(float).rolling(w, min_periods=w).sum()
    pct = rolling_percentile(net_atr.abs(), cfg.percentile_lookback, 50)
    body_atr = (c - o) / atr_ref
    loc = feats["close_location"]
    range_atr = rng / atr_ref
    bsb = feats["bars_since_break"]
    brk_dir = feats["last_break_direction"]
    recent_break = bsb.notna() & (bsb <= w - 1)

    out = pd.DataFrame(index=feats.index)
    out["disp_net_move_atr"] = net_atr
    out["disp_efficiency"] = eff
    out["disp_velocity_atr"] = net_atr / w
    out["disp_percentile"] = pct
    for side, sgn in (("bullish", 1.0), ("bearish", -1.0)):
        closes = up_closes if sgn > 0 else dn_closes
        loc_side = loc if sgn > 0 else 1.0 - loc
        comps = {
            "body": scale01(sgn * body_atr, 0.3, 1.5),
            "close": scale01(loc_side, 0.5, 0.9),
            "range": scale01(range_atr, 0.8, 2.5) * (loc_side >= 0.6),
            "net_move": scale01(sgn * net_atr, 0.5, 2.5),
            "efficiency": scale01(eff, 0.3, 0.8) * (sgn * net > 0),
            "low_overlap": scale01(1.0 - ov, 0.3, 0.8) * (sgn * net > 0),
            "closes": scale01(closes / w, 0.34, 1.0),
            "break": (recent_break & (brk_dir == side)).astype(float),
            "percentile": scale01(pct.fillna(0.0), 60.0, 98.0) * (sgn * net > 0),
        }
        score = sum(np.nan_to_num(np.asarray(comps[k], dtype=float)) * wt for k, wt in WEIGHTS.items()) * 100.0
        multi_ok = (
            (sgn * net_atr >= cfg.min_multi_bar_atr) & (closes >= cfg.min_directional_closes) & (eff >= cfg.min_efficiency)
        ).fillna(False).to_numpy()
        score = np.where(multi_ok, score, np.minimum(score, cfg.strong_score - 0.01))
        valid = atr_ref.notna().to_numpy() & net_atr.notna().to_numpy()
        score = np.where(valid, np.round(score, 2), 0.0)
        out[f"{side}_displacement_score"] = score
        out[f"{side}_displacement_class"] = _classify(score, cfg)
        out[f"{side}_displacement_multibar"] = multi_ok
    return out
