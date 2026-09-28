"""Price-action liquidity-reference map and sweep detection.

Terminology is deliberately modest: the engine cannot see resting orders.  A
reference is a POTENTIAL_LIQUIDITY_REFERENCE derived from price action only:

* SWING_LIQUIDITY          confirmed swing highs/lows (known from confirmation)
* STRUCTURAL_LIQUIDITY_REFERENCE  previous structural-break levels (the broken
                           level becomes a reference on the other side of price)
* clusters of references within volatility-aware tolerances:
  EQUAL_HIGHS / EQUAL_LOWS (``equal_tol_atr``) and NEAR_EQUAL_* (``near_equal_tol_atr``)
* STRUCTURAL_EXTREME flag for the highest / lowest live reference

When a bar trades beyond live references they are marked TAKEN and a
``SweepEvent`` starts in state UNRESOLVED.  Its state then evolves from later
completed closes only (append-only transitions, like the break lifecycle):

  UNRESOLVED -> SWEEP_AND_REJECT   all closes back inside for ``resolve_bars``
  UNRESOLVED -> BREAK_AND_ACCEPT   a close beyond by ``accept_close_atr`` (from the next bar)
  SWEEP_AND_REJECT -> BREAK_AND_ACCEPT  reference later reclaimed within ``monitor_bars``

An UPSIDE sweep (highs taken, then rejected) implies BEARISH context
(``bearish_sweep_score``); a DOWNSIDE sweep implies BULLISH context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import LiquidityConfig
from ..features.indicators import scale01

UNRESOLVED = "UNRESOLVED"
SWEEP_AND_REJECT = "SWEEP_AND_REJECT"
BREAK_AND_ACCEPT = "BREAK_AND_ACCEPT"
SWEEP_WEIGHTS = {"penetration": 0.15, "wick": 0.15, "close_position": 0.15, "rejection": 0.15,
                 "importance": 0.15, "confirmation": 0.15, "structural_effect": 0.10}


@dataclass
class LiquidityRef:
    ref_id: str
    side: str  # "upside" (above price, highs) | "downside"
    price: float
    kind: str  # SWING_LIQUIDITY | STRUCTURAL_LIQUIDITY_REFERENCE
    source_index: int
    created_index: int
    significance: float = 1.0
    category: str = "POTENTIAL_LIQUIDITY_REFERENCE"


@dataclass
class SweepEvent:
    event_id: int
    side: str  # "upside" | "downside"
    implication: str  # "bearish" for upside sweeps, "bullish" for downside sweeps
    level: float
    ref_ids: list
    ref_kind: str
    cluster_size: int
    index: int
    time: str
    atr: float
    penetration_atr: float
    wick_ratio: float
    close_position: float
    rejection_atr: float
    transitions: list = field(default_factory=list)  # (index, from, to, reason)

    def state_at(self, c: int) -> str | None:
        st = None
        for idx, _, to, _ in self.transitions:
            if idx <= c:
                st = to
        return st

    def snapshot_at(self, c: int, score: float | None = None) -> dict:
        known = [t for t in self.transitions if t[0] <= c]
        return {
            "event_id": self.event_id, "side": self.side, "implication": self.implication, "level": self.level,
            "reference_kind": self.ref_kind, "cluster_size": self.cluster_size, "time": self.time,
            "bars_since": c - self.index, "penetration_atr": round(self.penetration_atr, 4),
            "wick_ratio": round(self.wick_ratio, 4), "close_position": round(self.close_position, 4),
            "rejection_atr": round(self.rejection_atr, 4), "state": known[-1][2] if known else None,
            "transitions": [{"index": t[0], "from": t[1], "to": t[2], "reason": t[3]} for t in known],
            "sweep_score": score,
        }


class LiquidityMap:
    def __init__(self, arrays: dict, swings, breaks, cfg: LiquidityConfig):
        self.a = arrays
        self.cfg = cfg
        self.swings_by_acc: dict[int, list] = {}
        for s in swings:
            self.swings_by_acc.setdefault(s.accepted_index, []).append(s)
        self.breaks_by_idx: dict[int, list] = {}
        for b in breaks:
            self.breaks_by_idx.setdefault(b.break_index, []).append(b)
        self.active: dict[str, LiquidityRef] = {}
        self.events: list[SweepEvent] = []
        self.monitored: list[SweepEvent] = []

    # ------------------------------------------------------------------
    def _add_refs(self, c: int) -> None:
        for s in self.swings_by_acc.get(c, ()):
            side = "upside" if s.kind == "high" else "downside"
            self.active[f"S{s.swing_id}"] = LiquidityRef(
                f"S{s.swing_id}", side, float(s.price), "SWING_LIQUIDITY", s.pivot_index, c,
                float(s.significance_atr) if s.significance_atr is not None else 1.0,
            )
        for b in self.breaks_by_idx.get(c, ()):
            side = "downside" if b.direction == "bullish" else "upside"
            self.active[f"B{b.event_id}"] = LiquidityRef(
                f"B{b.event_id}", side, float(b.level), "STRUCTURAL_LIQUIDITY_REFERENCE", c, c, 1.0,
            )

    def _expire(self, c: int) -> None:
        start = c - self.cfg.lookback_bars + 1
        for rid in [r for r, ref in self.active.items() if ref.source_index < start]:
            del self.active[rid]

    def clusters(self, side: str, atr: float) -> list[dict]:
        refs = sorted((r for r in self.active.values() if r.side == side), key=lambda r: (r.price, r.ref_id))
        out, cur = [], []
        for r in refs:
            if cur and r.price - cur[-1].price > self.cfg.near_equal_tol_atr * atr:
                out.append(cur)
                cur = []
            cur.append(r)
        if cur:
            out.append(cur)
        res = []
        for cl in out:
            span = cl[-1].price - cl[0].price
            n = len(cl)
            if n >= 2 and span <= self.cfg.equal_tol_atr * atr:
                ctype = "EQUAL_HIGHS" if side == "upside" else "EQUAL_LOWS"
            elif n >= 2:
                ctype = "NEAR_EQUAL_HIGHS" if side == "upside" else "NEAR_EQUAL_LOWS"
            else:
                ctype = cl[0].kind
            res.append({"type": ctype, "side": side, "price": float(np.mean([r.price for r in cl])),
                        "low": cl[0].price, "high": cl[-1].price, "size": n, "ref_ids": [r.ref_id for r in cl],
                        "kinds": sorted({r.kind for r in cl})})
        if res:
            ext = max(res, key=lambda x: x["high"]) if side == "upside" else min(res, key=lambda x: x["low"])
            ext["structural_extreme"] = True
        return res

    # ------------------------------------------------------------------
    def step(self, c: int) -> None:
        a, cfg = self.a, self.cfg
        atr = a["atr"][c]
        h, l, o, cl = a["high"][c], a["low"][c], a["open"][c], a["close"][c]
        # 1) advance existing sweep events with this bar's close
        still = []
        for ev in self.monitored:
            k = c - ev.index
            d = 1.0 if ev.side == "upside" else -1.0
            beyond = d * (cl - ev.level)
            st = ev.transitions[-1][2]
            if k >= 1 and np.isfinite(atr) and beyond >= cfg.accept_close_atr * atr and st != BREAK_AND_ACCEPT:
                reason = "close beyond reference" if st == UNRESOLVED else "reference later reclaimed"
                ev.transitions.append((c, st, BREAK_AND_ACCEPT, reason))
            elif st == UNRESOLVED and k >= cfg.resolve_bars:
                closes = a["close"][ev.index:c + 1]
                if np.all(d * (closes - ev.level) <= 0):
                    ev.transitions.append((c, st, SWEEP_AND_REJECT, f"all closes back inside for {k} bars"))
            if k < cfg.monitor_bars and ev.transitions[-1][2] != BREAK_AND_ACCEPT:
                still.append(ev)
        self.monitored = still

        # 2) penetration of live references by this bar
        if np.isfinite(atr) and atr > 0:
            rng = h - l if h > l else 1e-9
            for side in ("upside", "downside"):
                pen = cfg.min_penetration_atr * atr
                taken = [r for r in self.active.values() if r.side == side and
                         ((side == "upside" and h > r.price + pen) or (side == "downside" and l < r.price - pen))]
                if not taken:
                    continue
                clusters = {tuple(x["ref_ids"]): x for x in self.clusters(side, atr)}
                size_of = {rid: x["size"] for x in clusters.values() for rid in x["ref_ids"]}
                anchor = max(taken, key=lambda r: (size_of.get(r.ref_id, 1), r.significance,
                                                  r.price if side == "upside" else -r.price, r.ref_id))
                for r in taken:
                    del self.active[r.ref_id]
                if side == "upside":
                    penetration = (h - anchor.price) / atr
                    wick = (h - max(o, cl)) / rng
                    close_pos = 1.0 - (cl - l) / rng
                    rejection = (anchor.price - cl) / atr
                else:
                    penetration = (anchor.price - l) / atr
                    wick = (min(o, cl) - l) / rng
                    close_pos = (cl - l) / rng
                    rejection = (cl - anchor.price) / atr
                ev = SweepEvent(
                    event_id=len(self.events), side=side, implication="bearish" if side == "upside" else "bullish",
                    level=float(anchor.price), ref_ids=[r.ref_id for r in taken], ref_kind=anchor.kind,
                    cluster_size=int(size_of.get(anchor.ref_id, 1)), index=c, time=a["ts"][c].isoformat(),
                    atr=float(atr), penetration_atr=float(penetration), wick_ratio=float(wick),
                    close_position=float(close_pos), rejection_atr=float(rejection),
                )
                ev.transitions.append((c, None, UNRESOLVED, "reference traded through; outcome not yet known"))
                self.events.append(ev)
                self.monitored.append(ev)

        # 3) references known at the close of c
        self._add_refs(c)
        self._expire(c)

    # ------------------------------------------------------------------
    def score(self, ev: SweepEvent, c: int, opposite_break_after: bool) -> float:
        st = ev.state_at(c)
        if st == BREAK_AND_ACCEPT:
            return 0.0
        cfg = self.cfg
        last_close = self.a["close"][c]
        inside = (last_close <= ev.level) if ev.side == "upside" else (last_close >= ev.level)
        comps = {
            "penetration": 1.0 - float(scale01(ev.penetration_atr, 0.75, cfg.max_sweep_penetration_atr)),
            "wick": float(scale01(ev.wick_ratio, 0.3, 0.7)),
            "close_position": float(scale01(ev.close_position, 0.5, 0.9)),
            "rejection": float(scale01(ev.rejection_atr, 0.0, 0.5)),
            "importance": {1: 0.4, 2: 0.7}.get(ev.cluster_size, 1.0) if ev.ref_kind == "SWING_LIQUIDITY" else 0.6,
            "confirmation": 1.0 if st == SWEEP_AND_REJECT else (0.5 if inside else 0.0),
            "structural_effect": 1.0 if opposite_break_after else 0.0,
        }
        return round(100.0 * sum(comps[k] * w for k, w in SWEEP_WEIGHTS.items()), 2)
