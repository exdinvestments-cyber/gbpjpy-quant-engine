"""Point-in-time swing memory and the structural leg engine.

``SwingMemoryTracker`` replays the Phase 1A.1 swing records bar by bar using
their ``accepted_index`` / ``removed_index`` so the context layer holds exactly
the swing window the structure engine held at each bar (verified against
``StructureResult.history_at`` in the tests) without re-detecting swings.

A structural LEG runs between two consecutive confirmed swings.  Its metrics
only use bars between the two pivots, all of which precede the end swing's
confirmation, so a leg is fully known at the end swing's ``confirmed_at``.
The ACTIVE leg runs from the latest confirmed swing to the current close.

Impulse vs correction
---------------------
``impulse_score`` (0-100) blends several independent views of the move, so no
single indicator decides:

  efficiency 0.25  |net| / path of closes
  distance   0.20  ATR-normalised size (1 -> 0, 4 ATR -> 1)
  overlap    0.15  low candle overlap
  body       0.15  mean body/range (body dominance)
  velocity   0.15  ATR per bar (0.15 -> 0, 0.6 -> 1)
  adverse    0.10  small maximum adverse excursion relative to the move

``nature`` = IMPULSIVE / MIXED / CORRECTIVE from the score.  ``classification``
additionally uses the leg's direction relative to the prevailing structure at
the evaluation bar (so it is a point-in-time interpretation).
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import LegConfig
from ..features.indicators import scale01
from ..features.structure import Swing

IMPULSE_WEIGHTS = {"efficiency": 0.25, "distance": 0.20, "overlap": 0.15, "body": 0.15, "velocity": 0.15, "adverse": 0.10}


class SwingMemoryTracker:
    """Rebuilds the structure engine's rolling swing memory incrementally."""

    def __init__(self, swings: list[Swing], size: int):
        self.size = size
        self.by_accept: dict[int, list[Swing]] = defaultdict(list)
        for s in swings:
            self.by_accept[s.accepted_index].append(s)
        self.alive: list[Swing] = []

    def update(self, c: int) -> list[Swing]:
        for s in self.by_accept.get(c, ()):
            if self.alive and self.alive[-1].removed_index == c and self.alive[-1].replaced_by == s.swing_id:
                self.alive.pop()
            self.alive.append(s)
        return self.alive[-self.size:]


@dataclass
class Leg:
    leg_id: str
    direction: str  # "up" | "down"
    start_swing_id: int
    end_swing_id: int | None  # None for the active leg
    start_index: int
    end_index: int
    start_time: str
    end_time: str
    confirmed_at: str | None
    start_price: float
    end_price: float
    distance: float
    pips: float
    duration_bars: int
    atr_ref: float
    distance_atr: float
    avg_body_atr: float
    avg_range_atr: float
    efficiency: float
    overlap: float
    momentum: float
    velocity_atr: float
    max_adverse_atr: float
    max_favourable_atr: float
    impulse_score: float
    nature: str
    end_label: str | None
    is_active: bool
    classification: str = "UNCLEAR"
    retracement_ratio: float | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class LegEngine:
    def __init__(self, df: pd.DataFrame, atr: np.ndarray, overlap: np.ndarray, cfg: LegConfig, pip_size: float):
        self.o = df["open"].to_numpy(float)
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.c = df["close"].to_numpy(float)
        self.ts = list(df["timestamp"])
        self.atr = atr
        self.ov = overlap
        self.cfg = cfg
        self.pip = pip_size
        self._cache: dict[tuple, Leg] = {}

    # ------------------------------------------------------------------
    def _metrics(self, start: Swing, end_index: int, end_price: float, direction: str, end_swing: Swing | None,
                 confirmed_at) -> Leg:
        i0, i1 = start.pivot_index, max(end_index, start.pivot_index + 1)
        sl = slice(i0 + 1, i1 + 1)
        atr_seg = self.atr[sl]
        atr_ref = float(np.nanmean(atr_seg)) if np.isfinite(atr_seg).any() else float("nan")
        if not np.isfinite(atr_ref) or atr_ref <= 0:
            atr_ref = float(np.nanmax(self.atr[: i1 + 1])) if np.isfinite(self.atr[: i1 + 1]).any() else 1.0
        d = 1.0 if direction == "up" else -1.0
        dist = d * (end_price - start.price)
        closes = self.c[i0:i1 + 1]
        path = float(np.sum(np.abs(np.diff(closes)))) if len(closes) > 1 else 0.0
        eff = float(abs(closes[-1] - closes[0]) / path) if path > 0 else 0.0
        rng = self.h[sl] - self.l[sl]
        body = np.abs(self.c[sl] - self.o[sl])
        signed_body = np.where(rng > 0, d * (self.c[sl] - self.o[sl]) / np.where(rng > 0, rng, 1.0), 0.0)
        ov = self.ov[sl]
        overlap = float(np.nanmean(ov)) if np.isfinite(ov).any() else 0.5
        dur = i1 - i0
        if direction == "up":
            run_ext = np.maximum.accumulate(self.h[sl])
            adverse = float(np.max(run_ext - self.l[sl])) if dur else 0.0
            fav = float(np.max(self.h[sl]) - start.price) if dur else 0.0
        else:
            run_ext = np.minimum.accumulate(self.l[sl])
            adverse = float(np.max(self.h[sl] - run_ext)) if dur else 0.0
            fav = float(start.price - np.min(self.l[sl])) if dur else 0.0
        dist_atr = dist / atr_ref
        body_ratio = float(np.mean(np.where(rng > 0, body / np.where(rng > 0, rng, 1.0), 0.0))) if dur else 0.0
        velocity = dist_atr / dur if dur else 0.0
        adverse_rel = (adverse / atr_ref) / max(abs(dist_atr), 1e-9)
        comps = {
            "efficiency": float(scale01(eff, 0.25, 0.70)),
            "distance": float(scale01(dist_atr, 1.0, 4.0)),
            "overlap": float(scale01(1.0 - overlap, 0.25, 0.65)),
            "body": float(scale01(body_ratio, 0.35, 0.70)),
            "velocity": float(scale01(velocity, 0.15, 0.60)),
            "adverse": 1.0 - float(scale01(adverse_rel, 0.20, 0.60)),
        }
        score = 100.0 * sum(comps[k] * w for k, w in IMPULSE_WEIGHTS.items())
        if dist <= 0:
            score = 0.0
        nature = "IMPULSIVE" if score >= self.cfg.impulsive_score else ("CORRECTIVE" if score < self.cfg.corrective_score else "MIXED")
        end_id = end_swing.swing_id if end_swing is not None else None
        return Leg(
            leg_id=f"{start.swing_id}-{end_id if end_id is not None else 'active'}", direction=direction,
            start_swing_id=start.swing_id, end_swing_id=end_id, start_index=i0, end_index=i1,
            start_time=self.ts[i0].isoformat(), end_time=self.ts[i1].isoformat(),
            confirmed_at=confirmed_at.isoformat() if confirmed_at is not None else None,
            start_price=float(start.price), end_price=float(end_price), distance=float(dist),
            pips=float(dist / self.pip), duration_bars=int(dur), atr_ref=atr_ref, distance_atr=float(dist_atr),
            avg_body_atr=float(np.mean(body) / atr_ref) if dur else 0.0,
            avg_range_atr=float(np.mean(rng) / atr_ref) if dur else 0.0,
            efficiency=eff, overlap=overlap, momentum=float(np.mean(signed_body)) if dur else 0.0,
            velocity_atr=float(velocity), max_adverse_atr=float(adverse / atr_ref), max_favourable_atr=float(fav / atr_ref),
            impulse_score=round(float(score), 2), nature=nature, end_label=end_swing.label if end_swing else None,
            is_active=end_swing is None,
        )

    def confirmed_leg(self, a: Swing, b: Swing) -> Leg:
        key = (a.swing_id, b.swing_id)
        if key not in self._cache:
            direction = "up" if b.kind == "high" else "down"
            self._cache[key] = self._metrics(a, b.pivot_index, b.price, direction, b, b.confirmed_at)
        leg = self._cache[key]
        return copy.copy(leg)  # shallow copy (scalar fields only): classification is per evaluation bar

    def active_leg(self, last: Swing, c: int) -> Leg:
        direction = "up" if last.kind == "low" else "down"
        if c <= last.pivot_index:
            end_idx, end_price = last.pivot_index, last.price
        elif direction == "up":
            seg = self.h[last.pivot_index + 1:c + 1]
            k = int(np.argmax(seg))
            end_idx, end_price = last.pivot_index + 1 + k, float(max(seg[k], last.price))
        else:
            seg = self.l[last.pivot_index + 1:c + 1]
            k = int(np.argmin(seg))
            end_idx, end_price = last.pivot_index + 1 + k, float(min(seg[k], last.price))
        leg = self._metrics(last, end_idx, end_price, direction, None, None)
        leg.duration_bars = int(c - last.pivot_index)
        return leg

    # ------------------------------------------------------------------
    def classify(self, legs: list[Leg], ref_dir: int) -> None:
        """Classify legs in place relative to the prevailing structural direction (+1/-1/0)."""
        cfg = self.cfg
        prev: Leg | None = None
        for leg in legs:
            ldir = 1 if leg.direction == "up" else -1
            if prev is not None and prev.distance > 0:
                leg.retracement_ratio = float(leg.distance / prev.distance)
            if ref_dir == 0:
                leg.classification = "UNCLEAR"
            elif ldir == ref_dir:
                failed = (not leg.is_active) and leg.end_label in (("LH", "EH") if ldir > 0 else ("HL", "EL"))
                if failed:
                    leg.classification = "FAILED_IMPULSE"
                elif leg.impulse_score >= cfg.strong_impulse_score:
                    leg.classification = "STRONG_IMPULSE"
                elif leg.impulse_score >= cfg.normal_impulse_score:
                    leg.classification = "NORMAL_IMPULSE"
                else:
                    leg.classification = "WEAK_IMPULSE"
            else:
                if leg.efficiency < cfg.choppy_efficiency or leg.overlap > cfg.choppy_overlap:
                    leg.classification = "CHOPPY_CORRECTION"
                elif leg.retracement_ratio is not None and leg.retracement_ratio > cfg.healthy_correction_ratio:
                    leg.classification = "DEEP_CORRECTION"
                else:
                    leg.classification = "HEALTHY_CORRECTION"
            prev = leg
