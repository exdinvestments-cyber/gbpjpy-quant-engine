"""Objective H4 support/resistance zone map (incremental).

Zones are maintained incrementally by ``ZoneBook``.  At each bar ``c`` only
information known at the close of ``c`` is used:

* level sources: confirmed swing highs/lows accepted by ``c`` whose pivot lies
  inside the lookback (including later-superseded swings), structural break
  levels with break bar in the lookback, and the trailing range high/low
* sources are clustered with single linkage (tolerance
  ``cluster_atr_mult * ATR``); a new source that lands within tolerance of an
  existing zone UPDATES it (or MERGES the zones it bridges), otherwise it
  CREATES a zone
* sources AGE OUT when they leave the lookback; a zone left without sources
  EXPIRES, and a zone whose remaining sources separate is split
* touches, interaction episodes, rejections and side-to-side breaks are kept
  as sliding-window counters updated with each new bar, instead of being
  recounted over the whole lookback every bar
* a zone that price has crossed ``invalidate_after_breaks`` times inside the
  lookback is INVALIDATED and its sources retired

Zone geometry (tolerance and minimum half-width) uses the ATR at the moment
of the zone's last membership change and is then held fixed; the reference
implementation ``compute_levels_rebuild`` re-clusters with the current ATR on
every bar.  With a constant ATR (and invalidation disabled) the two are
identical, which the test-suite verifies bar by bar.

Every lifecycle event is recorded in ``ZoneBook.events`` with the bar index at
which it became known; nothing already emitted is rewritten.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import LevelConfig
from .structure import StructureResult


LEVEL_COLUMNS = [
    "nearest_support", "support_lower", "support_upper", "support_strength",
    "distance_to_support", "distance_to_support_atr",
    "nearest_resistance", "resistance_lower", "resistance_upper", "resistance_strength",
    "distance_to_resistance", "distance_to_resistance_atr", "n_zones",
]


@dataclass
class Zone:
    lower: float
    upper: float
    midpoint: float
    zone_type: str  # support | resistance | (inside: price within zone)
    sources: list
    n_sources: int
    interactions: int
    touch_bars: int
    last_interaction_index: int | None
    last_interaction_time: str | None
    bars_since_interaction: int | None
    age_bars: int
    rejections: int
    rejection_strength: float
    breaks: int
    distance: float
    distance_atr: float
    strength: float
    zone_id: int = -1
    created_index: int | None = None
    bars_since_created: int | None = None
    status: str = "active"
    # Phase 1B (additive): ATR used when the zone's geometry was fixed, for current-volatility context.
    formation_atr: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _cluster(points: list[tuple[float, str, int]], tol: float) -> list[list[tuple[float, str, int]]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1], p[2]))
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def compute_levels_rebuild(
    df: pd.DataFrame,
    atr_s: pd.Series,
    structure: StructureResult,
    cfg: LevelConfig,
) -> tuple[pd.DataFrame, list[list[Zone]]]:
    """REFERENCE implementation: rebuilds every zone from scratch at every bar.

    Kept for verification.  With a constant ATR and break-based invalidation
    disabled, the incremental ``compute_levels`` must reproduce it exactly
    (``tests/test_levels_incremental.py``).
    """
    n = len(df)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    atr = atr_s.to_numpy(float)
    ts = list(df["timestamp"])
    LB = cfg.lookback_bars
    RX = cfg.range_extreme_lookback

    swings = structure.swings
    breaks = structure.breaks
    sw_acc = np.array([s.accepted_index for s in swings], dtype=int)
    sw_piv = np.array([s.pivot_index for s in swings], dtype=int)
    br_idx = np.array([b.break_index for b in breaks], dtype=int)
    rows = []
    zones_by_bar: list[list[Zone]] = []

    for c in range(n):
        a = atr[c]
        if np.isnan(a) or a <= 0:
            zones_by_bar.append([])
            rows.append({})
            continue
        start = max(0, c - LB + 1)
        pts: list[tuple[float, str, int]] = []
        # swings known at c (accepted by c) with pivot inside the lookback;
        # superseded swings still mark a price that was once structural
        for k in np.flatnonzero((sw_acc <= c) & (sw_piv >= start)):
            s = swings[k]
            pts.append((s.price, f"swing_{s.kind}", s.pivot_index))
        for k in np.flatnonzero((br_idx <= c) & (br_idx >= start)):
            b = breaks[k]
            pts.append((b.level, "break_level", b.break_index))
        rs = max(0, c - RX + 1)
        if c - rs + 1 >= min(RX, 20):
            hi_idx = rs + int(np.argmax(h[rs:c + 1]))
            lo_idx = rs + int(np.argmin(l[rs:c + 1]))
            pts.append((float(h[hi_idx]), "range_high", hi_idx))
            pts.append((float(l[lo_idx]), "range_low", lo_idx))

        tol = cfg.cluster_atr_mult * a
        hw = cfg.min_half_width_atr * a
        hs, ls, cs = h[start:c + 1], l[start:c + 1], cl[start:c + 1]
        zones: list[Zone] = []
        for cluster in _cluster(pts, tol):
            prices = [p[0] for p in cluster]
            lo, hi = min(prices), max(prices)
            mid = (lo + hi) / 2
            lo, hi = min(lo, mid - hw), max(hi, mid + hw)
            touched = (hs >= lo) & (ls <= hi)
            touch_bars = int(touched.sum())
            # interaction episodes = runs of consecutive touching bars
            episodes = int(touched[0]) + int(np.sum(touched[1:] & ~touched[:-1])) if touch_bars else 0
            last_touch = int(np.flatnonzero(touched)[-1]) + start if touch_bars else None
            support_rej = touched & (ls <= hi) & (cs > hi)
            resist_rej = touched & (hs >= lo) & (cs < lo)
            rejections = int(support_rej.sum() + resist_rej.sum())
            side = np.where(cs > hi, 1, np.where(cs < lo, -1, 0))
            nz = side[side != 0]
            zone_breaks = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
            first_idx = min(p[2] for p in cluster)
            sources = sorted({p[1] for p in cluster})
            n_swing_src = sum(p[1].startswith("swing") for p in cluster)
            rej_strength = 100.0 * rejections / touch_bars if touch_bars else 0.0

            price = cl[c]
            if price > hi:
                ztype, dist = "support", price - hi
            elif price < lo:
                ztype, dist = "resistance", lo - price
            else:
                ztype, dist = ("support" if price >= mid else "resistance"), 0.0

            recency = 1.0 - ((c - last_touch) / LB) if last_touch is not None else 0.0
            strength = (
                30.0 * min(episodes / 4.0, 1.0)
                + 25.0 * (rej_strength / 100.0)
                + 15.0 * min(len(sources) / 3.0, 1.0)
                + 10.0 * min(n_swing_src / 3.0, 1.0)
                + 10.0 * max(recency, 0.0)
                + 10.0 * (1.0 - min(zone_breaks / 3.0, 1.0))
            )
            zones.append(
                Zone(
                    lower=float(lo), upper=float(hi), midpoint=float(mid), zone_type=ztype, sources=sources,
                    n_sources=len(cluster), interactions=episodes, touch_bars=touch_bars,
                    last_interaction_index=last_touch,
                    last_interaction_time=ts[last_touch].isoformat() if last_touch is not None else None,
                    bars_since_interaction=(c - last_touch) if last_touch is not None else None,
                    age_bars=int(c - first_idx), rejections=rejections, rejection_strength=round(rej_strength, 2),
                    breaks=zone_breaks, distance=float(dist), distance_atr=float(dist / a),
                    strength=round(min(max(float(strength), 0.0), 100.0), 2), formation_atr=float(a),
                )
            )
        zones.sort(key=lambda z: (-z.strength, z.distance))
        zones = zones[: cfg.max_zones]
        zones.sort(key=lambda z: z.midpoint)
        zones_by_bar.append(zones)

        sup = [z for z in zones if z.zone_type == "support"]
        res = [z for z in zones if z.zone_type == "resistance"]
        ns = min(sup, key=lambda z: (z.distance, -z.midpoint)) if sup else None
        nr = min(res, key=lambda z: (z.distance, z.midpoint)) if res else None
        rows.append(
            {
                "nearest_support": ns.midpoint if ns else np.nan,
                "support_lower": ns.lower if ns else np.nan,
                "support_upper": ns.upper if ns else np.nan,
                "support_strength": ns.strength if ns else np.nan,
                "distance_to_support": ns.distance if ns else np.nan,
                "distance_to_support_atr": ns.distance_atr if ns else np.nan,
                "nearest_resistance": nr.midpoint if nr else np.nan,
                "resistance_lower": nr.lower if nr else np.nan,
                "resistance_upper": nr.upper if nr else np.nan,
                "resistance_strength": nr.strength if nr else np.nan,
                "distance_to_resistance": nr.distance if nr else np.nan,
                "distance_to_resistance_atr": nr.distance_atr if nr else np.nan,
                "n_zones": len(zones),
            }
        )
    frame = pd.DataFrame(rows, index=df.index, columns=LEVEL_COLUMNS)
    return frame, zones_by_bar


# ---------------------------------------------------------------------------
# Incremental engine
# ---------------------------------------------------------------------------


@dataclass
class _LiveZone:
    zone_id: int
    points: dict  # pid -> (price, source, idx)
    raw_lo: float
    raw_hi: float
    lower: float
    upper: float
    mid: float
    created_index: int
    tol: float
    # sliding-window counters over bars [start, c]
    touch_bars: int = 0
    rejections: int = 0
    episodes: int = 0
    flips: int = 0
    last_touch: int | None = None
    sides: deque = field(default_factory=deque)  # (bar, side) for bars closing outside the zone


@dataclass(frozen=True)
class ZoneEvent:
    index: int
    event: str  # created | updated | merged | split | expired | invalidated
    zone_id: int
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


class ZoneBook:
    """Incrementally maintained support/resistance zones."""

    def __init__(self, df: pd.DataFrame, atr_s: pd.Series, structure: StructureResult, cfg: LevelConfig):
        self.cfg = cfg
        self.h = df["high"].to_numpy(float)
        self.l = df["low"].to_numpy(float)
        self.cl = df["close"].to_numpy(float)
        self.atr = atr_s.to_numpy(float)
        self.ts = list(df["timestamp"])
        self.swings = structure.swings
        self.breaks = structure.breaks
        self.zones: dict[int, _LiveZone] = {}
        self.point_zone: dict = {}  # pid -> zone_id
        self.retired: set = set()
        self.expiry: list = []  # heap of (idx, pid)
        self.events: list[ZoneEvent] = []
        self.invalidated: list[dict] = []
        self._next_id = 0
        self._sw_ptr = 0
        self._br_ptr = 0
        self._range_pids: dict[str, tuple] = {}
        self._start = 0
        self._initialised = False

    # -- helpers ---------------------------------------------------------------
    def _touch(self, j: int, z: _LiveZone) -> tuple[bool, int, int]:
        """(touched, rejection, side) of bar j relative to zone z."""
        hj, lj, cj = self.h[j], self.l[j], self.cl[j]
        touched = hj >= z.lower and lj <= z.upper
        side = 1 if cj > z.upper else (-1 if cj < z.lower else 0)
        rej = 1 if touched and side != 0 else 0
        return touched, rej, side

    def _build_stats(self, z: _LiveZone, c: int) -> None:
        start = self._start
        hs, ls, cs = self.h[start:c + 1], self.l[start:c + 1], self.cl[start:c + 1]
        touched = (hs >= z.lower) & (ls <= z.upper)
        z.touch_bars = int(touched.sum())
        z.episodes = int(touched[0]) + int(np.sum(touched[1:] & ~touched[:-1])) if z.touch_bars else 0
        z.last_touch = int(np.flatnonzero(touched)[-1]) + start if z.touch_bars else None
        side = np.where(cs > z.upper, 1, np.where(cs < z.lower, -1, 0))
        z.rejections = int(np.sum(touched & (side != 0)))
        nzi = np.flatnonzero(side != 0)
        z.sides = deque(zip((nzi + start).tolist(), side[nzi].tolist()))
        nz = side[nzi]
        z.flips = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0

    def _new_zone(self, pts: list, c: int, tol: float, hw: float, zone_id: int | None = None,
                  created_index: int | None = None) -> _LiveZone:
        prices = [p[1][0] for p in pts]
        lo, hi = min(prices), max(prices)
        mid = (lo + hi) / 2
        if zone_id is None:
            zone_id = self._next_id
            self._next_id += 1
        z = _LiveZone(
            zone_id=zone_id, points=dict(pts), raw_lo=lo, raw_hi=hi, lower=min(lo, mid - hw), upper=max(hi, mid + hw),
            mid=mid, created_index=c if created_index is None else created_index, tol=tol,
        )
        self._build_stats(z, c)
        for pid in z.points:
            self.point_zone[pid] = zone_id
        return z

    # -- per-bar update --------------------------------------------------------
    def _advance_window(self, c: int, new_start: int) -> None:
        """Slide every zone's counters from window [start, c-1] to [new_start, c]."""
        prev_start = self._start
        for z in self.zones.values():
            touched, rej, side = self._touch(c, z)
            if touched:
                prev_touched = c - 1 >= prev_start and self._touch(c - 1, z)[0]
                if not prev_touched:
                    z.episodes += 1
                z.touch_bars += 1
                z.last_touch = c
            z.rejections += rej
            if side != 0:
                if z.sides and z.sides[-1][1] != side:
                    z.flips += 1
                z.sides.append((c, side))
            for j in range(prev_start, new_start):  # bars leaving the window
                tj, rj, _ = self._touch(j, z)
                if tj:
                    z.touch_bars -= 1
                    if not (j + 1 <= c and self._touch(j + 1, z)[0]):
                        z.episodes -= 1
                z.rejections -= rj
                while z.sides and z.sides[0][0] < new_start:
                    _, s0 = z.sides.popleft()
                    if z.sides and z.sides[0][1] != s0:
                        z.flips -= 1
            if z.last_touch is not None and z.last_touch < new_start:
                z.last_touch = None
        self._start = new_start

    def _pending_points(self, c: int) -> tuple[list, list]:
        """(points to add, pids to remove) at bar c."""
        start = self._start
        add, remove = [], []
        while self._sw_ptr < len(self.swings) and self.swings[self._sw_ptr].accepted_index <= c:
            s = self.swings[self._sw_ptr]
            self._sw_ptr += 1
            if s.pivot_index >= start:
                add.append((("s", s.swing_id), (s.price, f"swing_{s.kind}", s.pivot_index)))
        while self._br_ptr < len(self.breaks) and self.breaks[self._br_ptr].break_index <= c:
            b = self.breaks[self._br_ptr]
            self._br_ptr += 1
            if b.break_index >= start:
                add.append((("b", b.event_id), (b.level, "break_level", b.break_index)))
        rx = self.cfg.range_extreme_lookback
        rs = max(0, c - rx + 1)
        if c - rs + 1 >= min(rx, 20):
            hi_idx = rs + int(np.argmax(self.h[rs:c + 1]))
            lo_idx = rs + int(np.argmin(self.l[rs:c + 1]))
            for key, idx, price in (("range_high", hi_idx, self.h[hi_idx]), ("range_low", lo_idx, self.l[lo_idx])):
                pid = (key, idx)
                old = self._range_pids.get(key)
                if old != pid:
                    if old is not None:
                        remove.append(old)
                    self._range_pids[key] = pid
                    add.append((pid, (float(price), key, idx)))
        while self.expiry and self.expiry[0][0] < start:
            _, pid = heapq.heappop(self.expiry)
            remove.append(pid)
        return add, remove

    def _apply_membership(self, c: int, add: list, remove: list) -> None:
        cfg = self.cfg
        a = self.atr[c]
        tol = cfg.cluster_atr_mult * a
        hw = cfg.min_half_width_atr * a
        dirty: set[int] = set()
        shrunk: set[int] = set()
        for pid in remove:
            zid = self.point_zone.pop(pid, None)
            if zid is not None and zid in self.zones:
                self.zones[zid].points.pop(pid, None)
                dirty.add(zid)
                shrunk.add(zid)
        fresh = []
        for pid, pt in add:
            if pid in self.retired:
                continue
            fresh.append((pid, pt))
            heapq.heappush(self.expiry, (pt[2], pid))
            for zid, z in self.zones.items():
                if z.raw_lo - tol <= pt[0] <= z.raw_hi + tol:
                    dirty.add(zid)
        if not dirty and not fresh:
            return
        old = {zid: self.zones.pop(zid) for zid in sorted(dirty)}
        pool = [(pid, pt) for z in old.values() for pid, pt in z.points.items()] + fresh
        clusters = _cluster([(pt[0], pt[1], pt[2], pid) for pid, pt in pool], tol)
        claimed: set[int] = set()
        for cl in clusters:
            pts = [(p[3], (p[0], p[1], p[2])) for p in cl]
            pids = {p[3] for p in cl}
            overlap = sorted(((len(pids & set(z.points)), -zid) for zid, z in old.items() if pids & set(z.points)), reverse=True)
            parent = next((-negid for _, negid in overlap if -negid not in claimed), None)
            if parent is not None:
                claimed.add(parent)
                pz = old[parent]
                if parent not in shrunk and set(pz.points) == pids:
                    self.zones[parent] = pz  # membership unchanged (e.g. neighbour activity only)
                    continue
                z = self._new_zone(pts, c, tol, hw, zone_id=parent, created_index=pz.created_index)
                merged = [-negid for _, negid in overlap if -negid != parent]
                if merged:
                    for m in merged:
                        claimed.add(m)
                        self.events.append(ZoneEvent(c, "merged", m, f"merged into zone {parent}"))
                self.events.append(ZoneEvent(c, "updated", parent, f"{len(pts)} sources, [{z.lower:.3f}, {z.upper:.3f}]"))
            else:
                z = self._new_zone(pts, c, tol, hw)
                self.events.append(ZoneEvent(c, "created", z.zone_id, f"{len(pts)} sources, [{z.lower:.3f}, {z.upper:.3f}]"))
            self.zones[z.zone_id] = z
        for zid in old:
            if zid not in claimed:
                self.events.append(ZoneEvent(c, "expired", zid, "all sources aged out of the lookback"))

    def _invalidate(self, c: int) -> None:
        k = self.cfg.invalidate_after_breaks
        if not k:
            return
        for zid in [zid for zid, z in self.zones.items() if z.flips >= k]:
            z = self.zones.pop(zid)
            for pid in z.points:
                self.point_zone.pop(pid, None)
                self.retired.add(pid)
            detail = f"price crossed the zone {z.flips} times within the lookback"
            self.events.append(ZoneEvent(c, "invalidated", zid, detail))
            self.invalidated.append({"zone_id": zid, "index": c, "at": self.ts[c].isoformat(), "lower": z.lower,
                                     "upper": z.upper, "created_index": z.created_index, "reason": detail})

    def step(self, c: int) -> list[Zone]:
        a = self.atr[c]
        if np.isnan(a) or a <= 0:
            return []
        new_start = max(0, c - self.cfg.lookback_bars + 1)
        if not self._initialised:
            self._start = new_start
            self._initialised = True
        else:
            self._advance_window(c, new_start)
        add, remove = self._pending_points(c)
        self._apply_membership(c, add, remove)
        self._invalidate(c)
        return self._emit(c)

    def _emit(self, c: int) -> list[Zone]:
        cfg = self.cfg
        a = self.atr[c]
        LB = cfg.lookback_bars
        price = self.cl[c]
        scored = []
        for z in sorted(self.zones.values(), key=lambda q: (q.raw_lo, q.raw_hi)):
            srcs = {p[1] for p in z.points.values()}
            n_swing = sum(p[1].startswith("swing") for p in z.points.values())
            rej_strength = 100.0 * z.rejections / z.touch_bars if z.touch_bars else 0.0
            recency = 1.0 - ((c - z.last_touch) / LB) if z.last_touch is not None else 0.0
            strength = (
                30.0 * min(z.episodes / 4.0, 1.0)
                + 25.0 * (rej_strength / 100.0)
                + 15.0 * min(len(srcs) / 3.0, 1.0)
                + 10.0 * min(n_swing / 3.0, 1.0)
                + 10.0 * max(recency, 0.0)
                + 10.0 * (1.0 - min(z.flips / 3.0, 1.0))
            )
            if price > z.upper:
                dist = price - z.upper
            elif price < z.lower:
                dist = z.lower - price
            else:
                dist = 0.0
            scored.append((round(min(max(float(strength), 0.0), 100.0), 2), dist, z, srcs, rej_strength))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out = []
        for strength, dist, z, srcs, rej_strength in scored[: cfg.max_zones]:
            if price > z.upper:
                ztype = "support"
            elif price < z.lower:
                ztype = "resistance"
            else:
                ztype = "support" if price >= z.mid else "resistance"
            lt = z.last_touch
            out.append(Zone(
                lower=float(z.lower), upper=float(z.upper), midpoint=float(z.mid), zone_type=ztype,
                sources=sorted(srcs), n_sources=len(z.points), interactions=z.episodes, touch_bars=z.touch_bars,
                last_interaction_index=lt, last_interaction_time=self.ts[lt].isoformat() if lt is not None else None,
                bars_since_interaction=(c - lt) if lt is not None else None,
                age_bars=int(c - min(p[2] for p in z.points.values())), rejections=z.rejections,
                rejection_strength=round(rej_strength, 2), breaks=z.flips, distance=float(dist),
                distance_atr=float(dist / a), strength=strength, zone_id=z.zone_id,
                created_index=z.created_index, bars_since_created=c - z.created_index, status="active",
                formation_atr=float(z.tol / cfg.cluster_atr_mult) if cfg.cluster_atr_mult > 0 else None,
            ))
        out.sort(key=lambda q: q.midpoint)
        return out


def _level_row(zones: list[Zone]) -> dict:
    sup = [z for z in zones if z.zone_type == "support"]
    res = [z for z in zones if z.zone_type == "resistance"]
    ns = min(sup, key=lambda z: (z.distance, -z.midpoint)) if sup else None
    nr = min(res, key=lambda z: (z.distance, z.midpoint)) if res else None
    return {
        "nearest_support": ns.midpoint if ns else np.nan,
        "support_lower": ns.lower if ns else np.nan,
        "support_upper": ns.upper if ns else np.nan,
        "support_strength": ns.strength if ns else np.nan,
        "distance_to_support": ns.distance if ns else np.nan,
        "distance_to_support_atr": ns.distance_atr if ns else np.nan,
        "nearest_resistance": nr.midpoint if nr else np.nan,
        "resistance_lower": nr.lower if nr else np.nan,
        "resistance_upper": nr.upper if nr else np.nan,
        "resistance_strength": nr.strength if nr else np.nan,
        "distance_to_resistance": nr.distance if nr else np.nan,
        "distance_to_resistance_atr": nr.distance_atr if nr else np.nan,
        "n_zones": len(zones),
    }


def compute_levels(
    df: pd.DataFrame,
    atr_s: pd.Series,
    structure: StructureResult,
    cfg: LevelConfig,
    return_book: bool = False,
):
    """Incremental zone map.  Returns (frame, zones_by_bar[, ZoneBook])."""
    book = ZoneBook(df, atr_s, structure, cfg)
    zones_by_bar: list[list[Zone]] = []
    rows = []
    for c in range(len(df)):
        zs = book.step(c)
        zones_by_bar.append(zs)
        rows.append(_level_row(zs) if book.atr[c] > 0 else {})
    frame = pd.DataFrame(rows, index=df.index, columns=LEVEL_COLUMNS)
    if return_book:
        return frame, zones_by_bar, book
    return frame, zones_by_bar
