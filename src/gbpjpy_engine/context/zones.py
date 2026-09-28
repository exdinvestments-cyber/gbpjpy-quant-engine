"""Displacement-origin zones, zone freshness, support/resistance role reversal
and dynamic (current-volatility) zone context.

Displacement-origin zones (no subjective order blocks)
------------------------------------------------------
When displacement in one direction first reaches ``min_displacement_score``
(a STRONG multi-candle displacement episode starts) at bar ``c``, the zone is
the high-low envelope of the ``origin_bars`` bars immediately preceding the
displacement window.  It is created at the close of ``c`` (when the
displacement is known).  Bullish -> "demand" zone, bearish -> "supply" zone.
Boundaries are fixed at creation and never resized.

Freshness: interactions are counted as episodes of bars trading into the zone
from the displacement side; penetration depth is the deepest trade into the
zone as a fraction of its width; reaction strength is the best close away from
the zone (ATR) after an interaction; a close through the far side INVALIDATES
the zone.  Classes FRESH / LIGHTLY_TESTED / TESTED / HEAVILY_TESTED /
INVALIDATED are descriptive; no superiority of fresh zones is assumed.

Role reversal (on the Phase 1A.1 support/resistance zones)
----------------------------------------------------------
ORIGINAL -> BROKEN (close beyond the far side) -> ACCEPTED (``acceptance_closes``
consecutive closes beyond) -> RETESTED (a bar trades back into the zone and
closes on the new side within ``retest_window_bars``) -> FLIPPED (subsequent
close ``reaction_atr`` away).  A close back on the original side from any
intermediate state records FAILED_FLIP and returns to ORIGINAL.  Crossing a
level alone never flips its role.  Zones are tracked while they are part of the
emitted zone map (top ``max_zones``); a zone that drops out is simply not
updated on those bars.

Dynamic context
---------------
For every zone the ORIGINAL boundaries are kept; current context is reported
separately: current ATR, width / current ATR, distance / current ATR and
current ATR / ATR at formation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import OriginZoneConfig, RoleReversalConfig
from ..features.indicators import scale01


@dataclass
class OriginZone:
    zone_id: int
    direction: str  # "demand" (bullish origin) | "supply" (bearish origin)
    lower: float
    upper: float
    created_index: int
    created_at: str
    origin_indices: list
    origin_structure: str
    displacement_score: float
    formation_atr: float
    interactions: int = 0
    max_penetration: float = 0.0
    reaction_atr: float = 0.0
    last_interaction_index: int | None = None
    touching: bool = False
    invalidated_index: int | None = None
    retired_index: int | None = None

    def freshness(self, cfg: OriginZoneConfig) -> str:
        if self.invalidated_index is not None:
            return "INVALIDATED"
        if self.interactions == 0:
            return "FRESH"
        if self.interactions <= cfg.lightly_tested:
            return "LIGHTLY_TESTED"
        if self.interactions <= cfg.tested:
            return "TESTED"
        return "HEAVILY_TESTED"

    def view(self, c: int, close: float, atr: float, cfg: OriginZoneConfig) -> dict:
        width = self.upper - self.lower
        if close > self.upper:
            dist = close - self.upper
        elif close < self.lower:
            dist = self.lower - close
        else:
            dist = 0.0
        return {
            "zone_id": self.zone_id, "direction": self.direction, "lower": self.lower, "upper": self.upper,
            "created_at": self.created_at, "age_bars": c - self.created_index, "origin_structure": self.origin_structure,
            "displacement_strength": self.displacement_score, "freshness": self.freshness(cfg),
            "revisits": self.interactions, "penetration_depth": round(self.max_penetration, 4),
            "reaction_strength_atr": round(self.reaction_atr, 4),
            "invalidation_status": "INVALIDATED" if self.invalidated_index is not None else "VALID",
            "formation_atr": self.formation_atr,
            **dynamic_context(self.lower, self.upper, dist, atr, self.formation_atr),
        }


def dynamic_context(lower: float, upper: float, distance: float, atr_now: float, formation_atr: float | None) -> dict:
    ok = np.isfinite(atr_now) and atr_now > 0
    return {
        "current_atr": float(atr_now) if ok else None,
        "current_width_atr": float((upper - lower) / atr_now) if ok else None,
        "current_distance_atr": float(distance / atr_now) if ok else None,
        "volatility_vs_formation": float(atr_now / formation_atr) if ok and formation_atr else None,
    }


class OriginZoneTracker:
    def __init__(self, arrays: dict, disp: dict, structure_state: np.ndarray, window: int, cfg: OriginZoneConfig):
        self.a = arrays
        self.disp = disp  # {"bullish": scores, "bearish": scores}
        self.state = structure_state
        self.window = window
        self.cfg = cfg
        self.zones: list[OriginZone] = []
        self.active: list[OriginZone] = []

    def step(self, c: int) -> list[OriginZone]:
        a, cfg = self.a, self.cfg
        h, l, cl, atr = a["high"][c], a["low"][c], a["close"][c], a["atr"][c]
        # update existing zones with bar c
        keep = []
        for z in self.active:
            if c - z.created_index > cfg.max_age_bars:
                z.retired_index = c
                continue
            width = max(z.upper - z.lower, 1e-9)
            if z.direction == "demand":
                touch = l <= z.upper
                depth = (z.upper - l) / width
                through = cl < z.lower
                away = (cl - z.upper) / atr if np.isfinite(atr) and atr > 0 else 0.0
            else:
                touch = h >= z.lower
                depth = (h - z.lower) / width
                through = cl > z.upper
                away = (z.lower - cl) / atr if np.isfinite(atr) and atr > 0 else 0.0
            if touch:
                if not z.touching:
                    z.interactions += 1
                    z.reaction_atr = 0.0
                z.last_interaction_index = c
                z.max_penetration = max(z.max_penetration, float(depth))
            elif z.last_interaction_index is not None:
                z.reaction_atr = max(z.reaction_atr, float(away))
            z.touching = bool(touch)
            if through:
                z.invalidated_index = c
                continue
            keep.append(z)
        self.active = keep
        # create a zone when a qualifying displacement episode starts at c
        w = self.window
        for side, direction in (("bullish", "demand"), ("bearish", "supply")):
            sc = self.disp[side]
            if sc[c] >= cfg.min_displacement_score and (c == 0 or sc[c - 1] < cfg.min_displacement_score):
                ws = c - w + 1
                i0 = ws - cfg.origin_bars
                if i0 < 0 or not np.isfinite(atr) or atr <= 0:
                    continue
                lo = float(np.min(a["low"][i0:ws]))
                hi = float(np.max(a["high"][i0:ws]))
                if (hi - lo) / atr > cfg.max_width_atr:
                    continue
                z = OriginZone(
                    zone_id=len(self.zones), direction=direction, lower=lo, upper=hi, created_index=c,
                    created_at=(a["ts"][c] + a["tf"]).isoformat(), origin_indices=list(range(i0, ws)),
                    origin_structure=str(self.state[ws - 1]), displacement_score=float(sc[c]), formation_atr=float(atr),
                )
                self.zones.append(z)
                self.active.append(z)
        return self.active


@dataclass
class RoleTrack:
    zone_id: int
    original_role: str
    state: str = "ORIGINAL"
    new_role: str | None = None
    break_index: int | None = None
    accept_count: int = 0
    accepted_index: int | None = None
    retest_index: int | None = None
    reaction_atr: float = 0.0
    confidence: float = 0.0
    transitions: list = field(default_factory=list)  # (index, from, to, reason)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class RoleReversalTracker:
    """Objective S/R role reversal on the Phase 1A.1 zone map."""

    def __init__(self, arrays: dict, cfg: RoleReversalConfig):
        self.a = arrays
        self.cfg = cfg
        self.tracks: dict[int, RoleTrack] = {}
        self.flips: list[dict] = []

    def _set(self, t: RoleTrack, c: int, to: str, reason: str) -> None:
        t.transitions.append((c, t.state, to, reason))
        t.state = to

    def step(self, c: int, zones) -> None:
        a, cfg = self.a, self.cfg
        cl, h, l, atr = a["close"][c], a["high"][c], a["low"][c], a["atr"][c]
        for z in zones:
            t = self.tracks.get(z.zone_id)
            if t is None:
                self.tracks[z.zone_id] = RoleTrack(z.zone_id, z.zone_type)
                continue
            orig_res = t.original_role == "resistance"
            beyond_new = cl > z.upper if orig_res else cl < z.lower
            back_original = cl < z.lower if orig_res else cl > z.upper
            if t.state in ("BROKEN", "ACCEPTED", "RETESTED") and back_original:
                self._set(t, c, "FAILED_FLIP", "close back on the original side")
                self._set(t, c, "ORIGINAL", "reset after failed flip")
                t.accept_count, t.reaction_atr = 0, 0.0
                continue
            if t.state == "ORIGINAL":
                if beyond_new:
                    t.break_index, t.accept_count = c, 1
                    self._set(t, c, "BROKEN", "close beyond the zone's far side")
            elif t.state == "BROKEN":
                t.accept_count = t.accept_count + 1 if beyond_new else t.accept_count
                if t.accept_count >= cfg.acceptance_closes:
                    t.accepted_index = c
                    self._set(t, c, "ACCEPTED", f"{t.accept_count} closes beyond the zone")
            elif t.state == "ACCEPTED":
                if c - (t.accepted_index or c) > cfg.retest_window_bars:
                    self._set(t, c, "ORIGINAL", "no retest within window")
                    continue
                probe = l <= z.upper if orig_res else h >= z.lower
                if probe and beyond_new:
                    t.retest_index = c
                    self._set(t, c, "RETESTED", "traded back into the zone and closed on the new side")
            elif t.state == "RETESTED":
                away = ((cl - z.upper) if orig_res else (z.lower - cl)) / atr if np.isfinite(atr) and atr > 0 else 0.0
                t.reaction_atr = max(t.reaction_atr, float(away))
                if t.reaction_atr >= cfg.reaction_atr:
                    t.new_role = "support" if orig_res else "resistance"
                    t.confidence = round(100.0 * (
                        0.3 * float(scale01(t.accept_count, 1, 4)) + 0.3 * float(scale01(t.reaction_atr, 0.5, 2.0))
                        + 0.2 * z.strength / 100.0 + 0.2), 2)
                    self._set(t, c, "FLIPPED", f"reaction {t.reaction_atr:.2f} ATR after retest")
                    self.flips.append({
                        "zone_id": z.zone_id, "index": c, "at": (a["ts"][c] + a["tf"]).isoformat(),
                        "original_role": t.original_role, "new_role": t.new_role, "lower": z.lower, "upper": z.upper,
                        "break_index": t.break_index, "accepted_index": t.accepted_index,
                        "retest_index": t.retest_index, "reaction_atr": round(t.reaction_atr, 4),
                        "confidence": t.confidence,
                        "type": "RESISTANCE_TO_SUPPORT" if orig_res else "SUPPORT_TO_RESISTANCE",
                    })
            elif t.state == "FLIPPED":
                # the flipped zone now plays its new role; a later close through it restarts tracking
                if back_original:
                    self._set(t, c, "ORIGINAL", "new role lost")
                    t.original_role = t.new_role or t.original_role
