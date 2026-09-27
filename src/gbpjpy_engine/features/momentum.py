"""Price-derived momentum.

Components (each in [-1, 1]):
  roc          mean of tanh(ROC_n / (ATR * sqrt(n)) / roc_norm) over roc_periods
  velocity     tanh(displacement over velocity_window / window / ATR / velocity_norm)
  body         trailing mean of signed body/range
  consecutive  signed count of consecutive same-direction closes / cap

Bullish score = 100 * sum(w * max(c, 0)); bearish = 100 * sum(w * max(-c, 0)).
ROC and velocity overlap heavily; they share the "displacement" weight budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import MomentumConfig
from .indicators import rolling_percentile

MOMENTUM_WEIGHTS = {"roc": 0.35, "velocity": 0.20, "body": 0.25, "consecutive": 0.20}


def _consecutive(close: pd.Series, cap: int) -> pd.Series:
    d = np.sign(close.diff().to_numpy())
    out = np.zeros(len(d))
    run = 0.0
    for i, s in enumerate(d):
        if np.isnan(s) or s == 0:
            run = 0.0
        elif run == 0 or np.sign(run) == s:
            run = run + s
        else:
            run = s
        out[i] = np.clip(run, -cap, cap) / cap
    return pd.Series(out, index=close.index)


def momentum_features(df: pd.DataFrame, atr_s: pd.Series, cfg: MomentumConfig) -> pd.DataFrame:
    c, o, h, l = df["close"], df["open"], df["high"], df["low"]
    out = pd.DataFrame(index=df.index)
    rocs = []
    for n in cfg.roc_periods:
        r = (c - c.shift(n)) / (atr_s * np.sqrt(n))
        out[f"roc_{n}_norm"] = r
        rocs.append(np.tanh(r / cfg.roc_norm))
    comp = pd.DataFrame(index=df.index)
    comp["roc"] = sum(rocs) / len(rocs)
    v = cfg.velocity_window
    out["impulse_velocity_atr"] = (c - c.shift(v)) / v / atr_s
    comp["velocity"] = np.tanh(out["impulse_velocity_atr"] / cfg.velocity_norm)
    signed_body = ((c - o) / (h - l).replace(0.0, np.nan)).fillna(0.0)
    comp["body"] = signed_body.rolling(cfg.body_window, min_periods=cfg.body_window).mean()
    comp["consecutive"] = _consecutive(c, cfg.consecutive_cap)
    for k in MOMENTUM_WEIGHTS:
        out[f"mom_c_{k}"] = comp[k]

    valid = comp.notna().all(axis=1)
    bull = sum(comp[k].clip(lower=0) * w for k, w in MOMENTUM_WEIGHTS.items()) * 100.0
    bear = sum((-comp[k]).clip(lower=0) * w for k, w in MOMENTUM_WEIGHTS.items()) * 100.0
    out["bullish_momentum_score"] = bull.where(valid).round(2)
    out["bearish_momentum_score"] = bear.where(valid).round(2)
    net = out["bullish_momentum_score"] - out["bearish_momentum_score"]
    out["momentum_score"] = net.round(2)
    out["momentum_percentile"] = rolling_percentile(net.abs(), cfg.history_lookback, 50)

    delta = net - net.shift(cfg.accel_lookback)
    out["momentum_change"] = delta.round(2)
    direction = np.where(net > 0, "bullish", "bearish")
    accel = np.select(
        [
            net.isna() | delta.isna(),
            net.abs() < cfg.flat_threshold,
            (np.sign(delta) == np.sign(net)) & (delta.abs() >= cfg.accel_threshold),
            (np.sign(delta) != np.sign(net)) & (delta.abs() >= cfg.accel_threshold),
        ],
        ["unknown", "flat", "accelerating", "decelerating"],
        default="steady",
    )
    out["momentum_acceleration"] = accel
    out["momentum_state"] = np.where(
        np.isin(accel, ["unknown", "flat"]), accel, np.char.add(np.char.add(direction.astype(str), "_"), accel.astype(str))
    )
    return out
