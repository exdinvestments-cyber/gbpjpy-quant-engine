"""Room-to-move: ATR-normalised distance to the nearest meaningful opposing
barrier.  No reward:risk is computed - no entry or stop exists in Phase 1B.

LONG barriers (at/above the close): strong resistance-type zones
(strength >= ``levels.strong_level_score``; distance 0 when price is inside
one), confirmed swing highs in the structural memory, valid supply-style
displacement-origin zones and the trailing range high.  SHORT barriers mirror.

Barriers formed entirely during the CURRENT active leg (e.g. the running high
of an advancing move, or a zone built only from it) are excluded: they are the
move's own extreme, not prior opposing structure.

room_score = scale(distance, min_room_atr -> 0, full_room_atr -> 100); with no
barrier the score is 100 and ``NO_BARRIER_FOUND`` is recorded.
"""

from __future__ import annotations

import numpy as np

from ..config import RoomConfig
from ..features.indicators import scale01


def room(direction: str, close: float, atr: float, c: int, zones, memory, origin_zones, range_extreme: float,
         range_extreme_index: int, exclude_from: int, strong_level: float, cfg: RoomConfig) -> dict:
    up = direction == "long"
    barriers = []
    if not np.isfinite(atr) or atr <= 0:
        return {"room_score": 0.0, "nearest_barrier": None, "barriers": [], "note": "ATR unavailable"}
    for z in zones:
        if z.strength < strong_level or (c - z.age_bars) >= exclude_from:
            continue
        if up and z.upper > close and (z.lower > close or z.zone_type == "resistance"):
            barriers.append(("strong_resistance_zone", max(z.lower, close), z.strength))
        elif not up and z.lower < close and (z.upper < close or z.zone_type == "support"):
            barriers.append(("strong_support_zone", min(z.upper, close), z.strength))
    for s in memory:
        if up and s.kind == "high" and s.price > close:
            barriers.append(("confirmed_swing_high", s.price, None))
        elif not up and s.kind == "low" and s.price < close:
            barriers.append(("confirmed_swing_low", s.price, None))
    for z in origin_zones:
        if up and z.direction == "supply" and z.upper > close:
            barriers.append(("supply_origin_zone", max(z.lower, close), z.displacement_score))
        elif not up and z.direction == "demand" and z.lower < close:
            barriers.append(("demand_origin_zone", min(z.upper, close), z.displacement_score))
    if np.isfinite(range_extreme) and range_extreme_index < exclude_from and (
        (up and range_extreme > close) or (not up and range_extreme < close)
    ):
        barriers.append(("range_high" if up else "range_low", range_extreme, None))
    out = [{"kind": k, "price": float(p), "distance_atr": float(abs(p - close) / atr), "strength": s} for k, p, s in barriers]
    out.sort(key=lambda b: (b["distance_atr"], b["kind"]))
    if not out:
        return {"room_score": 100.0, "nearest_barrier": None, "barriers": [], "note": "NO_BARRIER_FOUND"}
    score = 100.0 * float(scale01(out[0]["distance_atr"], cfg.min_room_atr, cfg.full_room_atr))
    return {"room_score": round(score, 2), "nearest_barrier": out[0], "barriers": out[:5], "note": None}
