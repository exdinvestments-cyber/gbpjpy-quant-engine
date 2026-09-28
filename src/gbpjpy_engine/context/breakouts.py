"""Breakout quality, acceptance vs rejection, failed-break and false-break-risk
analysis on top of the Phase 1A.1 break lifecycle.

Everything here is evaluated AT a bar ``c`` using only:

* ``BreakEvent.transitions_known_at(c)`` - the lifecycle as known at ``c``
  (never the final ``state`` of the object)
* bars ``break_index .. c`` of price data, the displacement series and swings
  accepted by ``c``
* the zone map as it was just BEFORE the break (``zones_by_bar[break_index-1]``)
  for level importance

Follow-through is recomputed from closes up to ``c``; the event's own
``max_follow_through_atr`` (its final value) is deliberately not used.

Intrabar extension point
------------------------
Lifecycle and acceptance use completed H4 closes only.  ``IntrabarEvidence``
(see ``data.interfaces.IntrabarProvider``) is accepted as an optional input
so a future lower-timeframe/tick provider can add evidence such as
"traded back through the level intrabar" without changing this interface.
When it is ``None`` (always, today) nothing intrabar is assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import BreakoutContextConfig
from ..features.indicators import scale01
from ..features.structure import ACCEPTED, CANDIDATE, CONFIRMED, FAILED, INVALIDATED, BreakEvent

LIFECYCLE_SUPPORT = {CANDIDATE: 0.5, CONFIRMED: 0.7, ACCEPTED: 1.0, FAILED: 0.0, INVALIDATED: 0.0}
QUALITY_WEIGHTS = {"importance": 0.15, "close_beyond": 0.15, "body": 0.10, "close_location": 0.10,
                   "displacement": 0.15, "efficiency": 0.10, "follow_through": 0.15, "lifecycle": 0.10}
RISK_WEIGHTS = {"weak_penetration": 0.14, "poor_close": 0.10, "rejection_wick": 0.10, "weak_displacement": 0.12,
                "poor_follow_through": 0.14, "high_chop": 0.10, "low_efficiency": 0.08, "opposing_structure": 0.12,
                "rapid_return": 0.10}
FAIL_WEIGHTS = {"return_depth": 0.25, "short_time_beyond": 0.15, "small_excursion": 0.15, "close_inside": 0.15,
                "opposing_displacement": 0.15, "structural_consequence": 0.15}


@dataclass
class IntrabarEvidence:
    """Placeholder for future lower-timeframe evidence (not produced by any provider yet)."""

    traded_back_through_level: bool | None = None
    time_beyond_level_minutes: float | None = None
    source: str = "none"


@dataclass
class BreakContext:
    event_id: int
    direction: str
    break_type: str
    level: float
    break_time: str
    state: str
    previous_state: str | None
    transition_reason: str
    confirmed_at: str | None
    accepted_at: str | None
    failed_at: str | None
    invalidated_at: str | None
    bars_since_break: int
    max_excursion_atr: float
    current_beyond_atr: float
    closes_beyond: int
    quality_components: dict
    breakout_quality_score: float
    breakout_quality_class: str
    acceptance_state: str
    acceptance_evidence: list
    rejection_evidence: list
    failure_score: float
    failure_components: dict
    false_break_risk_score: float
    false_break_risk_reasons: list
    risk_components: dict = field(default_factory=dict)
    intrabar_source: str = "none"

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class BreakoutAnalyzer:
    def __init__(self, arrays: dict, swings, zones_by_bar, cfg: BreakoutContextConfig, strong_level: float,
                 monitor_bars: int):
        self.a = arrays
        self.swings = swings
        self.zones_by_bar = zones_by_bar
        self.cfg = cfg
        self.strong_level = strong_level
        self.monitor = monitor_bars

    def _importance(self, ev: BreakEvent) -> float:
        sw = self.swings[ev.swing_id]
        sig = sw.significance_atr if sw.significance_atr is not None else 1.0
        zs = self.zones_by_bar[ev.break_index - 1] if ev.break_index >= 1 else []
        zone_strength = max((z.strength for z in zs if z.lower - 1e-9 <= ev.level <= z.upper + 1e-9), default=0.0)
        return 0.6 * float(scale01(sig, 0.5, self.cfg.importance_norm_atr)) + 0.4 * zone_strength / 100.0

    def evaluate(self, ev: BreakEvent, c: int, intrabar: IntrabarEvidence | None = None) -> BreakContext:
        a, cfg = self.a, self.cfg
        known = ev.transitions_known_at(c)
        state = known[-1].to_state
        b = ev.break_index
        d = 1.0 if ev.direction == "bullish" else -1.0
        atr_b = ev.atr_at_break if ev.atr_at_break > 0 else 1.0
        atr_c = a["atr"][c] if np.isfinite(a["atr"][c]) and a["atr"][c] > 0 else atr_b
        closes = a["close"][b:c + 1]
        beyond = d * (closes - ev.level) / atr_b
        ft = float(np.max(beyond))
        cur = float(d * (a["close"][c] - ev.level) / atr_c)
        n_beyond = int(np.sum(beyond[1:] > 0))
        k = c - b
        o, h, l, cl = a["open"][b], a["high"][b], a["low"][b], a["close"][b]
        rng = h - l if h > l else 1e-9
        body_ratio = abs(cl - o) / rng
        loc = (cl - l) / rng if d > 0 else (h - cl) / rng
        against_wick = ((h - max(o, cl)) if d > 0 else (min(o, cl) - l)) / rng
        side = "bullish" if d > 0 else "bearish"
        opp = "bearish" if d > 0 else "bullish"
        disp_b = a[f"{side}_disp"][b] / 100.0
        eff_b = a["disp_eff"][b]

        qc = {
            "importance": self._importance(ev),
            "close_beyond": float(scale01(ev.magnitude_atr, 0.25, 1.0)),
            "body": float(scale01(body_ratio, 0.4, 0.8)),
            "close_location": float(scale01(loc, 0.5, 0.9)),
            "displacement": float(disp_b),
            "efficiency": float(scale01(eff_b, 0.3, 0.8)),
            "follow_through": float(scale01(ft, 0.25, 1.5)),
            "lifecycle": LIFECYCLE_SUPPORT[state],
        }
        quality = round(100.0 * sum(qc[q] * w for q, w in QUALITY_WEIGHTS.items()), 2)
        if quality >= cfg.high_quality_score:
            qclass = "HIGH_QUALITY_BREAK"
        elif quality >= cfg.moderate_quality_score:
            qclass = "MODERATE_BREAK"
        elif quality >= cfg.weak_quality_score:
            qclass = "WEAK_BREAK"
        else:
            qclass = "FALSE_BREAK_CANDIDATE"

        # ---- acceptance vs rejection -------------------------------------
        acc, rej = [], []
        if n_beyond >= cfg.acceptance_min_closes:
            acc.append("multiple_closes_beyond_level")
        tol = cfg.retest_tolerance_atr * atr_b
        for j in range(b + 1, c + 1):
            probe = a["low"][j] <= ev.level + tol if d > 0 else a["high"][j] >= ev.level - tol
            if probe and d * (a["close"][j] - ev.level) > 0:
                acc.append("successful_retest")
                break
        opp_kind = "low" if d > 0 else "high"
        if any(s.kind == opp_kind and b < s.accepted_index <= c and d * (s.price - ev.level) > 0 for s in self.swings_after(b, c)):
            acc.append("new_structure_beyond_level")
        if k >= 1 and ft >= 1.0 and cur > 0:
            acc.append("continued_displacement")
        if state == ACCEPTED:
            acc.append("lifecycle_accepted")
        opp_disp = float(np.max(a[f"{opp}_disp"][b + 1:c + 1])) / 100.0 if k >= 1 else 0.0
        if k >= 1 and cur < 0:
            rej.append("close_back_inside")
        if opp_disp >= 0.6:
            rej.append("opposing_displacement")
        if k >= 1 and cur < 0 and k <= 2:
            rej.append("rapid_reversal")
        if k >= 4 and ft < 0.5:
            rej.append("failed_follow_through")
        if state in (FAILED, INVALIDATED):
            rej.append(f"lifecycle_{state.lower()}")
        if state in (FAILED, INVALIDATED) or len(rej) >= 2:
            acceptance = "REJECTING"
        elif k == 0:
            acceptance = "UNRESOLVED"
        elif not rej and (state == ACCEPTED or len(acc) >= 3):
            acceptance = "ACCEPTING"
        elif acc and len(acc) > len(rej):
            acceptance = "PARTIALLY_ACCEPTING"
        else:
            acceptance = "UNRESOLVED"
        if intrabar is not None and intrabar.traded_back_through_level:
            rej.append("intrabar_trade_back_through_level")

        # ---- failure score --------------------------------------------------
        consequence = any(
            other.break_index > b and other.break_index <= c and other.direction != ev.direction for other in self._later(ev, c)
        )
        fc = {
            "return_depth": float(scale01(-float(np.min(beyond)) if len(beyond) else 0.0, 0.0, 1.0)),
            "short_time_beyond": 1.0 - float(scale01(n_beyond, 1.0, 8.0)),
            "small_excursion": 1.0 - float(scale01(ft, 0.5, 2.0)),
            "close_inside": 1.0 if cur < 0 else 0.0,
            "opposing_displacement": opp_disp,
            "structural_consequence": 1.0 if consequence else 0.0,
        }
        failing = state in (FAILED, INVALIDATED) or (k >= 1 and cur < 0)
        failure = round(100.0 * sum(fc[x] * w for x, w in FAIL_WEIGHTS.items()), 2) if failing else 0.0

        # ---- false-break risk ----------------------------------------------
        zs = self.zones_by_bar[c]
        close_c = a["close"][c]
        opposing = any(
            z.strength >= self.strong_level
            and ((d > 0 and z.lower >= close_c and z.lower - close_c <= cfg.opposing_zone_atr * atr_c)
                 or (d < 0 and z.upper <= close_c and close_c - z.upper <= cfg.opposing_zone_atr * atr_c))
            for z in zs
        )
        rapid = 0.0
        if ft > 0 and k >= 1:
            rapid = float(np.clip(1.0 - max(d * (close_c - ev.level) / atr_b, 0.0) / ft, 0.0, 1.0))
        rc = {
            "weak_penetration": 1.0 - float(scale01(ev.magnitude_atr, 0.25, 1.0)),
            "poor_close": 1.0 - float(scale01(loc, 0.5, 0.9)),
            "rejection_wick": float(scale01(against_wick, 0.2, 0.5)),
            "weak_displacement": 1.0 - float(disp_b),
            "poor_follow_through": (1.0 - float(scale01(ft, 0.25, 1.0))) if k >= 2 else 0.5,
            "high_chop": float(scale01(a["chop"][c] if np.isfinite(a["chop"][c]) else 50.0, 40.0, 75.0)),
            "low_efficiency": 1.0 - float(scale01(a["eff"][c] if np.isfinite(a["eff"][c]) else 0.3, 0.2, 0.5)),
            "opposing_structure": 1.0 if opposing else 0.0,
            "rapid_return": rapid,
        }
        if state in (FAILED, INVALIDATED):
            risk, reasons = 100.0, ["break_already_" + state.lower()]
        else:
            risk = round(100.0 * sum(rc[x] * w for x, w in RISK_WEIGHTS.items()), 2)
            reasons = [x for x, v in rc.items() if v >= 0.6]

        def iso(t):
            return t.isoformat() if t is not None else None

        first = {t.to_state: t for t in reversed(known)}
        return BreakContext(
            event_id=ev.event_id, direction=ev.direction, break_type=ev.break_type, level=float(ev.level),
            break_time=ev.break_time.isoformat(), state=state, previous_state=known[-1].from_state,
            transition_reason=known[-1].reason,
            confirmed_at=iso(first[CONFIRMED].at) if CONFIRMED in first else None,
            accepted_at=iso(first[ACCEPTED].at) if ACCEPTED in first else None,
            failed_at=iso(first[FAILED].at) if FAILED in first else None,
            invalidated_at=iso(first[INVALIDATED].at) if INVALIDATED in first else None,
            bars_since_break=int(k), max_excursion_atr=round(ft, 4), current_beyond_atr=round(cur, 4),
            closes_beyond=n_beyond, quality_components={q: round(v, 4) for q, v in qc.items()},
            breakout_quality_score=quality, breakout_quality_class=qclass, acceptance_state=acceptance,
            acceptance_evidence=acc, rejection_evidence=rej, failure_score=failure,
            failure_components={x: round(v, 4) for x, v in fc.items()}, false_break_risk_score=risk,
            false_break_risk_reasons=reasons, risk_components={x: round(v, 4) for x, v in rc.items()},
            intrabar_source=intrabar.source if intrabar is not None else "none",
        )

    # helpers set by the context engine (point-in-time views)
    def swings_after(self, b: int, c: int):
        return [s for s in self._recent_swings if b < s.accepted_index <= c]

    def _later(self, ev: BreakEvent, c: int):
        return [x for x in self._recent_breaks if x.event_id != ev.event_id and x.break_index <= c]

    def set_views(self, recent_swings, recent_breaks) -> None:
        self._recent_swings = recent_swings
        self._recent_breaks = recent_breaks
