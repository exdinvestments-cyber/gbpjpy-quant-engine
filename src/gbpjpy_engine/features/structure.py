"""Confirmed swings, market structure, breaks of structure and structure quality.

Look-ahead protection
---------------------
The engine walks forward bar by bar.  At bar ``c`` it only reads arrays up to
index ``c``.  A fractal pivot at bar ``i`` needs ``right_bars`` later bars, so
it is evaluated at ``c = i + right_bars`` and stamped with
``confirmed_at = close_time[c]``.  Every per-bar output is written at the end
of processing bar ``c`` and never revisited, so a pivot can never appear in
the state of a bar that closed before it was confirmed.

Meaningful swings
-----------------
A candidate pivot becomes a structural swing only if it alternates with the
previous swing and is at least ``min_swing_atr`` x ATR (ATR at confirmation)
away from it.  A same-type candidate that is more extreme than the latest
swing *supersedes* it (e.g. a higher high before any meaningful pullback).
Superseding is recorded with ``removed_index`` so historical states still show
the swing that was known at the time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import StructureConfig, SwingConfig
from .indicators import scale01

BULL_LABELS = ("HH", "HL")
BEAR_LABELS = ("LH", "LL")

QUALITY_WEIGHTS = {
    "clarity": 0.20,
    "impulse_pullback": 0.15,
    "impulse_size": 0.05,
    "break_conflict": 0.10,
    "reversal_frequency": 0.10,
    "swing_spacing": 0.05,
    "persistence": 0.15,
    "overlap": 0.10,
    "wickiness": 0.10,
}


@dataclass
class Swing:
    swing_id: int
    kind: str  # "high" | "low"
    pivot_index: int
    pivot_time: pd.Timestamp
    price: float
    confirm_index: int
    confirmed_at: pd.Timestamp  # close time of the confirming bar = availability time
    atr_at_confirmation: float
    label: str | None
    accepted_index: int
    removed_index: int | None = None
    replaced_by: int | None = None

    def alive_at(self, c: int) -> bool:
        return self.accepted_index <= c and (self.removed_index is None or self.removed_index > c)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pivot_time"] = self.pivot_time.isoformat()
        d["confirmed_at"] = self.confirmed_at.isoformat()
        return d


@dataclass
class BreakEvent:
    event_id: int
    direction: str  # "bullish" | "bearish"
    break_type: str  # "BOS" | "CHOCH" | "BREAKOUT"
    level: float
    swing_id: int
    level_time: pd.Timestamp
    break_index: int
    break_time: pd.Timestamp
    available_at: pd.Timestamp
    close: float
    magnitude: float
    magnitude_atr: float
    wick_beyond_atr: float
    closed_beyond: bool
    structure_before: str
    structure_after: str | None = None
    status: str = "pending"  # pending | confirmed | rejected
    status_history: list = field(default_factory=list)  # [(bar_index, status)]

    def status_at(self, c: int) -> str | None:
        st = None
        for idx, s in self.status_history:
            if idx <= c:
                st = s
        return st

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("level_time", "break_time", "available_at"):
            d[k] = getattr(self, k).isoformat()
        return d


@dataclass
class StructureResult:
    frame: pd.DataFrame
    swings: list[Swing]
    breaks: list[BreakEvent]

    def swings_known_at(self, c: int) -> list[Swing]:
        return [s for s in self.swings if s.alive_at(c)]

    def breaks_known_at(self, c: int) -> list[BreakEvent]:
        return [b for b in self.breaks if b.break_index <= c]


def _swing_state(alive: list[Swing]) -> tuple[str, bool, str | None, str | None]:
    highs = [s for s in alive if s.kind == "high"]
    lows = [s for s in alive if s.kind == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "neutral", False, (highs[-1].label if highs else None), (lows[-1].label if lows else None)
    hl, ll = highs[-1].label, lows[-1].label
    if hl == "HH" and ll == "HL":
        st = "bullish"
    elif hl == "LH" and ll == "LL":
        st = "bearish"
    elif hl == "HH" and ll == "LL":
        st = "transitional"
    else:
        st = "neutral"
    return st, True, hl, ll


def _direction_of(state: str) -> int:
    return 1 if state == "bullish" else (-1 if state == "bearish" else 0)


def compute_structure(
    df: pd.DataFrame,
    atr_s: pd.Series,
    overlap_mean: pd.Series,
    wick_mean: pd.Series,
    swing_cfg: SwingConfig,
    cfg: StructureConfig,
    timeframe: pd.Timedelta,
) -> StructureResult:
    n = len(df)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c_arr = df["close"].to_numpy(float)
    atr = atr_s.to_numpy(float)
    ov = overlap_mean.to_numpy(float)
    wk = wick_mean.to_numpy(float)
    ts = list(df["timestamp"])
    L, R = swing_cfg.left_bars, swing_cfg.right_bars
    W = cfg.quality_window_bars

    swings: list[Swing] = []
    alive: list[Swing] = []
    breaks: list[BreakEvent] = []
    broken_swing_ids: set[int] = set()
    states: list[str] = []

    cols: dict[str, list] = {k: [] for k in (
        "structure_state", "swing_structure_state", "structure_defined",
        "last_swing_high", "last_swing_high_time", "last_swing_high_confirmed_at", "last_high_label",
        "last_swing_low", "last_swing_low_time", "last_swing_low_confirmed_at", "last_low_label",
        "n_swings", "last_break_type", "last_break_direction", "last_break_level", "last_break_time",
        "last_break_magnitude", "last_break_atr", "last_break_status", "last_break_structure_before",
        "last_break_structure_after", "bars_since_break", "breaks_up_window", "breaks_down_window",
        "breaks_rejected_window",
    )}
    qcols: dict[str, list] = {f"sq_{k}": [] for k in QUALITY_WEIGHTS}
    qcols["structure_quality_score"] = []
    qcols["sq_avg_impulse_atr"] = []
    qcols["sq_avg_pullback_atr"] = []
    qcols["sq_impulse_pullback_ratio"] = []
    qcols["sq_avg_swing_spacing"] = []
    qcols["sq_state_changes"] = []

    def close_time(i: int) -> pd.Timestamp:
        return ts[i] + timeframe

    def set_status(ev: BreakEvent, idx: int, status: str) -> None:
        ev.status = status
        ev.status_history.append((idx, status))

    def label_for(kind: str, price: float, atr_ref: float) -> str | None:
        prev = next((s for s in reversed(alive) if s.kind == kind), None)
        if prev is None:
            return None
        tol = swing_cfg.equal_level_atr * atr_ref
        if kind == "high":
            return "EH" if abs(price - prev.price) <= tol else ("HH" if price > prev.price else "LH")
        return "EL" if abs(price - prev.price) <= tol else ("HL" if price > prev.price else "LL")

    def consider(kind: str, i: int, c: int) -> None:
        atr_ref = atr[c]
        if np.isnan(atr_ref):
            return
        price = h[i] if kind == "high" else l[i]
        last = alive[-1] if alive else None
        if last is not None and last.kind == kind:
            more_extreme = price > last.price if kind == "high" else price < last.price
            if not more_extreme:
                return
            alive.pop()
            last.removed_index = c
            new_id = len(swings)
            last.replaced_by = new_id
        elif last is not None and abs(price - last.price) < swing_cfg.min_swing_atr * atr_ref:
            return
        sw = Swing(
            swing_id=len(swings), kind=kind, pivot_index=i, pivot_time=ts[i], price=float(price),
            confirm_index=c, confirmed_at=close_time(c), atr_at_confirmation=float(atr_ref),
            label=label_for(kind, price, atr_ref), accepted_index=c,
        )
        swings.append(sw)
        alive.append(sw)

    for c in range(n):
        a = atr[c]
        # 1) resolve pending breaks using this bar's close
        for ev in breaks:
            if ev.status != "pending" or ev.break_index >= c:
                continue
            back_inside = c_arr[c] < ev.level if ev.direction == "bullish" else c_arr[c] > ev.level
            if back_inside:
                set_status(ev, c, "rejected")
            elif c - ev.break_index >= cfg.break_confirm_bars:
                set_status(ev, c, "confirmed")

        # 2) detect new breaks against levels known at the end of bar c-1
        prev_state = states[-1] if states else "neutral"
        new_events: list[BreakEvent] = []
        if not np.isnan(a):
            last_high = next((s for s in reversed(alive) if s.kind == "high"), None)
            last_low = next((s for s in reversed(alive) if s.kind == "low"), None)
            for sw, direction in ((last_high, "bullish"), (last_low, "bearish")):
                if sw is None or sw.swing_id in broken_swing_ids:
                    continue
                if direction == "bullish":
                    beyond = c_arr[c] - sw.price
                    wick = h[c] - sw.price
                else:
                    beyond = sw.price - c_arr[c]
                    wick = sw.price - l[c]
                if beyond < cfg.break_min_atr * a:
                    continue
                pd_dir = _direction_of(prev_state)
                d = 1 if direction == "bullish" else -1
                btype = "BOS" if pd_dir == d else ("CHOCH" if pd_dir == -d else "BREAKOUT")
                ev = BreakEvent(
                    event_id=len(breaks), direction=direction, break_type=btype, level=sw.price,
                    swing_id=sw.swing_id, level_time=sw.pivot_time, break_index=c, break_time=ts[c],
                    available_at=close_time(c), close=float(c_arr[c]), magnitude=float(beyond),
                    magnitude_atr=float(beyond / a), wick_beyond_atr=float(wick / a), closed_beyond=True,
                    structure_before=prev_state,
                )
                set_status(ev, c, "pending")
                breaks.append(ev)
                new_events.append(ev)
                broken_swing_ids.add(sw.swing_id)

        # 3) swing confirmations: pivot at i = c - R is confirmable now
        i = c - R
        if i - L >= 0:
            is_high = h[i] > h[i - L:i].max() and h[i] >= h[i + 1:c + 1].max()
            is_low = l[i] < l[i - L:i].min() and l[i] <= l[i + 1:c + 1].min()
            order = ["high", "low"]
            if alive and alive[-1].kind == "high":
                order = ["low", "high"]
            for kind in order:
                if (kind == "high" and is_high) or (kind == "low" and is_low):
                    consider(kind, i, c)

        # 4) structure state
        swing_state, defined, hl, ll = _swing_state(alive)
        state = swing_state
        recent = [
            ev for ev in breaks
            if ev.status != "rejected" and c - ev.break_index <= cfg.transition_memory_bars
        ]
        last_ev = recent[-1] if recent else None
        if last_ev is not None:
            ev_dir = 1 if last_ev.direction == "bullish" else -1
            sdir = _direction_of(swing_state)
            if sdir == -ev_dir:
                state = "transitional"
            elif sdir == 0 and last_ev.break_type in ("CHOCH", "BREAKOUT"):
                state = "transitional"
        states.append(state)
        for ev in new_events:
            ev.structure_after = state

        # 5) record per-bar outputs
        lh = next((s for s in reversed(alive) if s.kind == "high"), None)
        lw = next((s for s in reversed(alive) if s.kind == "low"), None)
        cols["structure_state"].append(state)
        cols["swing_structure_state"].append(swing_state)
        cols["structure_defined"].append(defined)
        cols["last_swing_high"].append(lh.price if lh else np.nan)
        cols["last_swing_high_time"].append(lh.pivot_time if lh else pd.NaT)
        cols["last_swing_high_confirmed_at"].append(lh.confirmed_at if lh else pd.NaT)
        cols["last_high_label"].append(hl)
        cols["last_swing_low"].append(lw.price if lw else np.nan)
        cols["last_swing_low_time"].append(lw.pivot_time if lw else pd.NaT)
        cols["last_swing_low_confirmed_at"].append(lw.confirmed_at if lw else pd.NaT)
        cols["last_low_label"].append(ll)
        cols["n_swings"].append(len(alive))
        lb = breaks[-1] if breaks else None
        cols["last_break_type"].append(lb.break_type if lb else None)
        cols["last_break_direction"].append(lb.direction if lb else None)
        cols["last_break_level"].append(lb.level if lb else np.nan)
        cols["last_break_time"].append(lb.break_time if lb else pd.NaT)
        cols["last_break_magnitude"].append(lb.magnitude if lb else np.nan)
        cols["last_break_atr"].append(lb.magnitude_atr if lb else np.nan)
        cols["last_break_status"].append(lb.status if lb else None)
        cols["last_break_structure_before"].append(lb.structure_before if lb else None)
        cols["last_break_structure_after"].append(lb.structure_after if lb else None)
        cols["bars_since_break"].append(c - lb.break_index if lb else np.nan)
        win = [ev for ev in breaks if c - ev.break_index < W]
        cols["breaks_up_window"].append(sum(ev.direction == "bullish" for ev in win))
        cols["breaks_down_window"].append(sum(ev.direction == "bearish" for ev in win))
        cols["breaks_rejected_window"].append(sum(ev.status == "rejected" for ev in win))

        # 6) structure quality
        q = _quality(alive[-cfg.quality_swings:], win, states[-W:], state, ov[c], wk[c])
        for k, v in q.items():
            qcols[k].append(v)

    frame = pd.DataFrame({**cols, **qcols}, index=df.index)
    return StructureResult(frame=frame, swings=swings, breaks=breaks)


def _quality(recent: list[Swing], win: list[BreakEvent], state_hist: list[str], state: str, overlap: float, wick: float) -> dict:
    labels = [s.label for s in recent if s.label is not None]
    bull_n = sum(lbl in BULL_LABELS for lbl in labels)
    bear_n = sum(lbl in BEAR_LABELS for lbl in labels)
    n_lab = len(labels)
    clarity = max(bull_n, bear_n) / n_lab if n_lab >= 2 else 0.0
    dominant = 1 if bull_n > bear_n else (-1 if bear_n > bull_n else 0)

    legs = [
        (recent[j].price - recent[j - 1].price) / recent[j].atr_at_confirmation
        for j in range(1, len(recent))
        if recent[j].atr_at_confirmation > 0
    ]
    imp = [abs(x) for x in legs if np.sign(x) == dominant and dominant != 0]
    pb = [abs(x) for x in legs if np.sign(x) != dominant and dominant != 0]
    avg_imp = float(np.mean(imp)) if imp else np.nan
    avg_pb = float(np.mean(pb)) if pb else np.nan
    if imp and pb and avg_pb > 0:
        ratio = avg_imp / avg_pb
        ip = float(scale01(ratio, 1.0, 2.5))
    else:
        ratio = np.nan
        ip = 0.0
    imp_size = float(scale01(avg_imp, 1.0, 4.0)) if imp else 0.0

    spacing = float(np.mean(np.diff([s.pivot_index for s in recent]))) if len(recent) >= 2 else np.nan
    spacing_score = float(scale01(spacing, 2.0, 8.0)) if not np.isnan(spacing) else 0.0

    up = sum(ev.direction == "bullish" for ev in win)
    down = sum(ev.direction == "bearish" for ev in win)
    rejected = sum(ev.status == "rejected" for ev in win)
    total = up + down
    if total:
        conflict = 2.0 * min(up, down) / total
        conflict_score = (1.0 - conflict) * (1.0 - 0.5 * rejected / total)
    else:
        conflict_score = 0.5

    changes = sum(1 for a, b in zip(state_hist, state_hist[1:]) if a != b)
    reversal_score = 1.0 - float(scale01(changes, 1.0, 5.0))
    if state in ("bullish", "bearish"):
        persistence = sum(s == state for s in state_hist) / len(state_hist)
    else:
        persistence = 0.0
    overlap_score = 1.0 - float(scale01(overlap, 0.30, 0.80)) if not np.isnan(overlap) else 0.0
    wick_score = 1.0 - float(scale01(wick, 0.30, 0.70)) if not np.isnan(wick) else 0.0

    comps = {
        "clarity": clarity,
        "impulse_pullback": ip,
        "impulse_size": imp_size,
        "break_conflict": conflict_score,
        "reversal_frequency": reversal_score,
        "swing_spacing": spacing_score,
        "persistence": persistence,
        "overlap": overlap_score,
        "wickiness": wick_score,
    }
    score = 100.0 * sum(comps[k] * w for k, w in QUALITY_WEIGHTS.items()) / sum(QUALITY_WEIGHTS.values())
    out = {f"sq_{k}": round(float(v), 4) for k, v in comps.items()}
    out["structure_quality_score"] = round(score, 2)
    out["sq_avg_impulse_atr"] = avg_imp
    out["sq_avg_pullback_atr"] = avg_pb
    out["sq_impulse_pullback_ratio"] = ratio
    out["sq_avg_swing_spacing"] = spacing
    out["sq_state_changes"] = changes
    return out
