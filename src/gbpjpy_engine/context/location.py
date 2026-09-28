"""Structural range location (premium/discount) and the retracement engine.

Premium/discount: the valid confirmed structural range is the latest confirmed
swing high and swing low in memory, provided the high is above the low and the
range is at least ``min_range_atr``.  ``structural_range_percentile`` =
(close - low) / (high - low) x 100 (raw value kept; the class uses the value
clipped to 0-100 and ``range_position`` says whether price is inside, above or
below).  Bands (configurable, descriptive): DEEP_DISCOUNT < 20 <= DISCOUNT < 45
<= EQUILIBRIUM <= 55 < PREMIUM <= 80 < DEEP_PREMIUM.  Location never implies
"discount = buy" or "premium = sell".

Retracement: the ACTIVE IMPULSE is the latest confirmed leg in the prevailing
structural direction.  Retracement is measured from its end pivot using only
bars after that pivot up to the current bar: deepest retracement %, current
retracement %, ATR distance, duration, velocity and directional efficiency.
Bands SHALLOW < 30 <= NORMAL < 55 <= DEEP < 80 <= VERY_DEEP are descriptive and
make no Fibonacci claim.
"""

from __future__ import annotations

import numpy as np

from ..config import LocationContextConfig
from ..features.structure import Swing


def premium_discount(memory: list[Swing], close: float, atr: float, cfg: LocationContextConfig) -> dict:
    hi = next((s for s in reversed(memory) if s.kind == "high"), None)
    lo = next((s for s in reversed(memory) if s.kind == "low"), None)
    out = {"structural_range_high": None, "structural_range_low": None, "structural_range_percentile": None,
           "structural_range_percentile_raw": None, "premium_discount_state": "UNDEFINED", "range_position": None,
           "range_valid": False}
    if hi is None or lo is None or not np.isfinite(atr) or atr <= 0:
        return out
    width = hi.price - lo.price
    out["structural_range_high"], out["structural_range_low"] = float(hi.price), float(lo.price)
    if width <= 0 or width / atr < cfg.min_range_atr:
        return out
    raw = (close - lo.price) / width * 100.0
    pct = float(np.clip(raw, 0.0, 100.0))
    if pct < cfg.deep_discount_pct:
        st = "DEEP_DISCOUNT"
    elif pct < cfg.discount_pct:
        st = "DISCOUNT"
    elif pct <= cfg.premium_pct:
        st = "EQUILIBRIUM"
    elif pct <= cfg.deep_premium_pct:
        st = "PREMIUM"
    else:
        st = "DEEP_PREMIUM"
    out.update({"structural_range_percentile": round(pct, 2), "structural_range_percentile_raw": round(float(raw), 2),
                "premium_discount_state": st, "range_valid": True,
                "range_position": "inside" if 0 <= raw <= 100 else ("above" if raw > 100 else "below")})
    return out


def retracement(impulse, arrays: dict, c: int, atr: float, cfg: LocationContextConfig) -> dict:
    out = {"retracement_depth_pct": None, "retracement_current_pct": None, "retracement_atr": None,
           "retracement_duration_bars": None, "retracement_velocity_atr": None, "retracement_efficiency": None,
           "retracement_band": "NONE", "active_impulse_leg": None}
    if impulse is None or impulse.distance <= 0 or not np.isfinite(atr) or atr <= 0:
        return out
    p = impulse.end_index
    if c <= p:
        return out
    up = impulse.direction == "up"
    lows, highs, closes = arrays["low"][p + 1:c + 1], arrays["high"][p + 1:c + 1], arrays["close"][p:c + 1]
    extreme = float(np.min(lows)) if up else float(np.max(highs))
    depth = (impulse.end_price - extreme) if up else (extreme - impulse.end_price)
    cur = (impulse.end_price - closes[-1]) if up else (closes[-1] - impulse.end_price)
    depth_pct = max(depth, 0.0) / impulse.distance * 100.0
    path = float(np.sum(np.abs(np.diff(closes))))
    dur = c - p
    if depth_pct < cfg.shallow_retracement_pct:
        band = "SHALLOW"
    elif depth_pct < cfg.normal_retracement_pct:
        band = "NORMAL"
    elif depth_pct < cfg.deep_retracement_pct:
        band = "DEEP"
    else:
        band = "VERY_DEEP"
    out.update({
        "retracement_depth_pct": round(depth_pct, 2), "retracement_current_pct": round(cur / impulse.distance * 100.0, 2),
        "retracement_atr": round(max(depth, 0.0) / atr, 4), "retracement_duration_bars": int(dur),
        "retracement_velocity_atr": round(max(depth, 0.0) / atr / dur, 4),
        "retracement_efficiency": round(abs(closes[-1] - closes[0]) / path, 4) if path > 0 else 0.0,
        "retracement_band": band, "active_impulse_leg": impulse.leg_id,
    })
    return out
