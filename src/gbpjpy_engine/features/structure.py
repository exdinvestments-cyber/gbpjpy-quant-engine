"""Confirmed swings, market structure, break lifecycle and structure quality.

Look-ahead protection
---------------------
The engine walks forward bar by bar.  At bar ``c`` it only reads arrays up to
index ``c``.  A fractal pivot at bar ``i`` needs ``right_bars`` later bars, so
it is evaluated at ``c = i + right_bars``.  Every swing therefore carries two
explicit timestamps:

* ``occurred_at``  - open time of the pivot bar (when the extreme printed)
* ``confirmed_at`` - close time of bar ``c`` (when the swing became KNOWABLE)

Every per-bar output is written at the end of processing bar ``c`` and never
revisited, so a pivot can never appear in the state of a bar that closed
before it was confirmed.

Structural memory
-----------------
The engine keeps a rolling window of the ``swing_history_size`` most recent
confirmed structural swings (working memory).  ``StructureResult.history_at(c)``
returns exactly the window that was in memory at bar ``c``.

Meaningful swings
-----------------
A candidate pivot becomes a structural swing only if it alternates with the
previous swing and is at least ``min_swing_atr`` x ATR (ATR at confirmation)
away from it.  A same-type candidate that is more extreme than the latest
swing *supersedes* it.  Superseding is recorded (``removed_index``) instead of
rewriting history, so earlier states still show the swing known at the time.

Break lifecycle
---------------
CANDIDATE -> CONFIRMED -> ACCEPTED, with CANDIDATE -> INVALIDATED and
CONFIRMED/ACCEPTED -> FAILED.  Transitions are appended with the bar at which
they became known; ``BreakEvent.state_at(c)`` reproduces the point-in-time
state, and the per-bar frame records the state as it was known at each bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import StructureConfig, SwingConfig
from .indicators import scale01

BULL_LABELS = ("HH", "HL")
BEAR_LABELS = ("LH", "LL")

# break lifecycle states
CANDIDATE = "CANDIDATE"
CONFIRMED = "CONFIRMED"
ACCEPTED = "ACCEPTED"
FAILED = "FAILED"
INVALIDATED = "INVALIDATED"
BREAK_STATES = (CANDIDATE, CONFIRMED, ACCEPTED, FAILED, INVALIDATED)
TERMINAL_STATES = (FAILED, INVALIDATED)
LIVE_STATES = (CANDIDATE, CONFIRMED, ACCEPTED)

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


def _iso(t) -> str | None:
    return None if t is None or pd.isna(t) else pd.Timestamp(t).isoformat()


@dataclass
class Swing:
    swing_id: int
    kind: str  # "high" | "low"
    pivot_index: int
    occurred_at: pd.Timestamp  # open time of the pivot bar
    price: float
    confirm_index: int
    confirmed_at: pd.Timestamp  # close time of the confirming bar = availability time
    atr_at_confirmation: float
    label: str | None  # structural classification: HH/LH/EH or HL/LL/EL (None for the first)
    significance_atr: float | None  # distance from the previous opposite swing in ATR (None for the first)
    accepted_index: int
    removed_index: int | None = None  # bar at which superseded by a more extreme same-type swing
    replaced_by: int | None = None

    @property
    def pivot_time(self) -> pd.Timestamp:  # backwards-compatible alias of occurred_at
        return self.occurred_at

    @property
    def confirmation_lag_bars(self) -> int:
        return self.confirm_index - self.pivot_index

    def alive_at(self, c: int) -> bool:
        """Confirmed by bar c and not yet superseded at bar c."""
        return self.accepted_index <= c and (self.removed_index is None or self.removed_index > c)

    def to_dict(self) -> dict:
        return {
            "swing_id": self.swing_id,
            "type": self.kind,
            "classification": self.label,
            "price": self.price,
            "occurred_at": _iso(self.occurred_at),
            "confirmed_at": _iso(self.confirmed_at),
            "pivot_index": self.pivot_index,
            "confirm_index": self.confirm_index,
            "confirmation_lag_bars": self.confirmation_lag_bars,
            "significance_atr": self.significance_atr,
            "atr_at_confirmation": self.atr_at_confirmation,
            "accepted_index": self.accepted_index,
            "removed_index": self.removed_index,
            "replaced_by": self.replaced_by,
        }


@dataclass(frozen=True)
class BreakTransition:
    index: int  # bar at whose close the transition became known
    at: pd.Timestamp  # that bar's close time
    from_state: str | None
    to_state: str
    reason: str

    def to_dict(self) -> dict:
        return {"index": self.index, "at": _iso(self.at), "from_state": self.from_state,
                "to_state": self.to_state, "reason": self.reason}


@dataclass
class BreakEvent:
    event_id: int
    direction: str  # "bullish" | "bearish"
    break_type: str  # "BOS" | "CHOCH" | "BREAKOUT"
    level: float
    swing_id: int
    level_occurred_at: pd.Timestamp
    level_confirmed_at: pd.Timestamp
    break_index: int
    break_time: pd.Timestamp  # open time of the break bar
    available_at: pd.Timestamp  # close time of the break bar (when the break became known)
    close: float
    magnitude: float
    magnitude_atr: float
    atr_at_break: float
    wick_beyond_atr: float
    closed_beyond: bool
    structure_before: str
    structure_after: str | None = None
    max_follow_through_atr: float = 0.0
    monitoring_ended_index: int | None = None
    transitions: list = field(default_factory=list)

    # -- current (latest known) lifecycle view --------------------------------
    @property
    def state(self) -> str:
        return self.transitions[-1].to_state

    @property
    def previous_state(self) -> str | None:
        return self.transitions[-1].from_state

    @property
    def transition_reason(self) -> str:
        return self.transitions[-1].reason

    @property
    def status(self) -> str:  # alias
        return self.state

    @property
    def status_history(self) -> list[tuple[int, str]]:
        return [(t.index, t.to_state) for t in self.transitions]

    def _time_of(self, state: str):
        for t in self.transitions:
            if t.to_state == state:
                return t.at
        return None

    @property
    def confirmed_at(self):
        return self._time_of(CONFIRMED)

    @property
    def accepted_at(self):
        return self._time_of(ACCEPTED)

    @property
    def failed_at(self):
        return self._time_of(FAILED)

    @property
    def invalidated_at(self):
        return self._time_of(INVALIDATED)

    # -- point-in-time view ----------------------------------------------------
    def transitions_known_at(self, c: int) -> list[BreakTransition]:
        return [t for t in self.transitions if t.index <= c]

    def state_at(self, c: int) -> str | None:
        known = self.transitions_known_at(c)
        return known[-1].to_state if known else None

    def status_at(self, c: int) -> str | None:  # alias
        return self.state_at(c)

    def snapshot_at(self, c: int) -> dict | None:
        """The break exactly as it was known at the close of bar ``c``."""
        known = self.transitions_known_at(c)
        if not known:
            return None

        def first(state):
            return next((_iso(t.at) for t in known if t.to_state == state), None)

        return {
            "event_id": self.event_id, "direction": self.direction, "break_type": self.break_type,
            "level": self.level, "swing_id": self.swing_id,
            "level_occurred_at": _iso(self.level_occurred_at), "level_confirmed_at": _iso(self.level_confirmed_at),
            "break_time": _iso(self.break_time), "available_at": _iso(self.available_at),
            "magnitude": self.magnitude, "magnitude_atr": self.magnitude_atr,
            "structure_before": self.structure_before, "structure_after": self.structure_after,
            "state": known[-1].to_state, "previous_state": known[-1].from_state,
            "transition_reason": known[-1].reason,
            "confirmed_at": first(CONFIRMED), "accepted_at": first(ACCEPTED),
            "failed_at": first(FAILED), "invalidated_at": first(INVALIDATED),
        }

    def to_dict(self) -> dict:
        d = self.snapshot_at(10**12) or {}
        d.update({
            "close": self.close, "atr_at_break": self.atr_at_break, "wick_beyond_atr": self.wick_beyond_atr,
            "closed_beyond": self.closed_beyond, "max_follow_through_atr": self.max_follow_through_atr,
            "monitoring_ended_index": self.monitoring_ended_index, "break_index": self.break_index,
            "transitions": [t.to_dict() for t in self.transitions],
        })
        return d


@dataclass
class StructureResult:
    frame: pd.DataFrame
    swings: list[Swing]
    breaks: list[BreakEvent]
    history_size: int = 16

    def swings_known_at(self, c: int) -> list[Swing]:
        """All confirmed, non-superseded swings known at bar c (full, unbounded list)."""
        return [s for s in self.swings if s.alive_at(c)]

    def history_at(self, c: int, n: int | None = None) -> list[Swing]:
        """The rolling structural memory (most recent ``n`` swings) as held at bar c."""
        n = n or self.history_size
        return self.swings_known_at(c)[-n:]

    def breaks_known_at(self, c: int) -> list[BreakEvent]:
        return [b for b in self.breaks if b.break_index <= c]


def _swing_state(alive: list[Swing]) -> tuple[str, bool, str | None, str | None]:
    last_high = next((s for s in reversed(alive) if s.kind == "high"), None)
    last_low = next((s for s in reversed(alive) if s.kind == "low"), None)
    n_high = sum(s.kind == "high" for s in alive)
    n_low = len(alive) - n_high
    hl = last_high.label if last_high else None
    ll = last_low.label if last_low else None
    if n_high < 2 or n_low < 2:
        return "neutral", False, hl, ll
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
    H = cfg.swing_history_size

    swings: list[Swing] = []
    memory: list[Swing] = []  # rolling structural memory (most recent H swings)
    breaks: list[BreakEvent] = []
    monitored: list[BreakEvent] = []  # breaks whose lifecycle is still being evaluated
    broken_swing_ids: set[int] = set()
    states: list[str] = []

    cols: dict[str, list] = {k: [] for k in (
        "structure_state", "swing_structure_state", "structure_defined",
        "last_swing_high", "last_swing_high_occurred_at", "last_swing_high_confirmed_at", "last_high_label",
        "last_swing_low", "last_swing_low_occurred_at", "last_swing_low_confirmed_at", "last_low_label",
        "n_swings", "swing_sequence", "hist_hh", "hist_hl", "hist_lh", "hist_ll", "hist_avg_significance_atr",
        "last_break_type", "last_break_direction", "last_break_level", "last_break_time",
        "last_break_magnitude", "last_break_atr", "last_break_status", "last_break_previous_status",
        "last_break_transition_reason", "last_break_confirmed_at", "last_break_failed_at",
        "last_break_structure_before", "last_break_structure_after", "bars_since_break",
        "breaks_up_window", "breaks_down_window", "breaks_invalidated_window", "breaks_failed_window",
    )}
    qcols: dict[str, list] = {f"sq_{k}": [] for k in QUALITY_WEIGHTS}
    for k in ("structure_quality_score", "sq_avg_impulse_atr", "sq_avg_pullback_atr",
              "sq_impulse_pullback_ratio", "sq_avg_swing_spacing", "sq_state_changes"):
        qcols[k] = []

    def close_time(i: int) -> pd.Timestamp:
        return ts[i] + timeframe

    def transition(ev: BreakEvent, idx: int, to_state: str, reason: str) -> None:
        prev = ev.transitions[-1].to_state if ev.transitions else None
        ev.transitions.append(BreakTransition(idx, close_time(idx), prev, to_state, reason))

    def label_for(kind: str, price: float, atr_ref: float) -> str | None:
        prev = next((s for s in reversed(memory) if s.kind == kind), None)
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
        last = memory[-1] if memory else None
        if last is not None and last.kind == kind:
            more_extreme = price > last.price if kind == "high" else price < last.price
            if not more_extreme:
                return
            memory.pop()
            last.removed_index = c
            last.replaced_by = len(swings)
        elif last is not None and abs(price - last.price) < swing_cfg.min_swing_atr * atr_ref:
            return
        opposite = memory[-1] if memory and memory[-1].kind != kind else None
        sw = Swing(
            swing_id=len(swings), kind=kind, pivot_index=i, occurred_at=ts[i], price=float(price),
            confirm_index=c, confirmed_at=close_time(c), atr_at_confirmation=float(atr_ref),
            label=label_for(kind, price, atr_ref),
            significance_atr=float(abs(price - opposite.price) / atr_ref) if opposite is not None else None,
            accepted_index=c,
        )
        swings.append(sw)
        memory.append(sw)
        if len(memory) > H:
            memory.pop(0)  # leaves working memory; the swing record itself is never altered

    for c in range(n):
        a = atr[c]
        # 1) advance break lifecycles with this bar's close (only information up to c)
        still: list[BreakEvent] = []
        for ev in monitored:
            if ev.break_index >= c:
                still.append(ev)
                continue
            d = 1.0 if ev.direction == "bullish" else -1.0
            beyond = d * (c_arr[c] - ev.level)  # >0: still beyond the level
            if ev.atr_at_break > 0:
                ev.max_follow_through_atr = max(ev.max_follow_through_atr, beyond / ev.atr_at_break)
            st = ev.state
            age = c - ev.break_index
            if st == CANDIDATE:
                if beyond < 0:
                    transition(ev, c, INVALIDATED, "closed back through level within confirmation window")
                elif age >= cfg.break_confirm_bars:
                    transition(ev, c, CONFIRMED, f"held beyond level for {cfg.break_confirm_bars} bars")
            elif st in (CONFIRMED, ACCEPTED):
                if not np.isnan(a) and -beyond >= cfg.break_fail_atr * a:
                    transition(ev, c, FAILED, f"closed {(-beyond / a):.2f} ATR back through level after {st.lower()}")
                elif (
                    st == CONFIRMED
                    and age >= cfg.break_accept_bars
                    and ev.max_follow_through_atr >= cfg.break_accept_atr
                ):
                    transition(ev, c, ACCEPTED,
                               f"held {age} bars with {ev.max_follow_through_atr:.2f} ATR follow-through")
            if ev.state in TERMINAL_STATES:
                ev.monitoring_ended_index = c
            elif age >= cfg.break_monitor_bars:
                ev.monitoring_ended_index = c  # lifecycle frozen in its last state
            else:
                still.append(ev)
        monitored = still

        # 2) detect new breaks against levels known at the end of bar c-1
        prev_state = states[-1] if states else "neutral"
        new_events: list[BreakEvent] = []
        if not np.isnan(a):
            last_high = next((s for s in reversed(memory) if s.kind == "high"), None)
            last_low = next((s for s in reversed(memory) if s.kind == "low"), None)
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
                dd = 1 if direction == "bullish" else -1
                btype = "BOS" if pd_dir == dd else ("CHOCH" if pd_dir == -dd else "BREAKOUT")
                ev = BreakEvent(
                    event_id=len(breaks), direction=direction, break_type=btype, level=sw.price,
                    swing_id=sw.swing_id, level_occurred_at=sw.occurred_at, level_confirmed_at=sw.confirmed_at,
                    break_index=c, break_time=ts[c], available_at=close_time(c), close=float(c_arr[c]),
                    magnitude=float(beyond), magnitude_atr=float(beyond / a), atr_at_break=float(a),
                    wick_beyond_atr=float(wick / a), closed_beyond=True, structure_before=prev_state,
                    max_follow_through_atr=float(beyond / a),
                )
                transition(ev, c, CANDIDATE, f"closed {beyond / a:.2f} ATR beyond swing {direction.replace('ish', '')} level")
                breaks.append(ev)
                monitored.append(ev)
                new_events.append(ev)
                broken_swing_ids.add(sw.swing_id)

        # 3) swing confirmations: pivot at i = c - R is confirmable now
        i = c - R
        if i - L >= 0:
            is_high = h[i] > h[i - L:i].max() and h[i] >= h[i + 1:c + 1].max()
            is_low = l[i] < l[i - L:i].min() and l[i] <= l[i + 1:c + 1].min()
            order = ["high", "low"]
            if memory and memory[-1].kind == "high":
                order = ["low", "high"]
            for kind in order:
                if (kind == "high" and is_high) or (kind == "low" and is_low):
                    consider(kind, i, c)

        # 4) structure state (swing sequence, overridden by a live counter-structure break)
        swing_state, defined, hl, ll = _swing_state(memory)
        state = swing_state
        last_ev = None
        for ev in reversed(breaks):
            if c - ev.break_index > cfg.transition_memory_bars:
                break
            if ev.state in LIVE_STATES:
                last_ev = ev
                break
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

        # 5) record per-bar outputs (as known at the close of bar c)
        lh = next((s for s in reversed(memory) if s.kind == "high"), None)
        lw = next((s for s in reversed(memory) if s.kind == "low"), None)
        labels = [s.label for s in memory if s.label]
        sig = [s.significance_atr for s in memory if s.significance_atr is not None]
        cols["structure_state"].append(state)
        cols["swing_structure_state"].append(swing_state)
        cols["structure_defined"].append(defined)
        cols["last_swing_high"].append(lh.price if lh else np.nan)
        cols["last_swing_high_occurred_at"].append(lh.occurred_at if lh else pd.NaT)
        cols["last_swing_high_confirmed_at"].append(lh.confirmed_at if lh else pd.NaT)
        cols["last_high_label"].append(hl)
        cols["last_swing_low"].append(lw.price if lw else np.nan)
        cols["last_swing_low_occurred_at"].append(lw.occurred_at if lw else pd.NaT)
        cols["last_swing_low_confirmed_at"].append(lw.confirmed_at if lw else pd.NaT)
        cols["last_low_label"].append(ll)
        cols["n_swings"].append(len(memory))
        cols["swing_sequence"].append(">".join(labels))
        cols["hist_hh"].append(labels.count("HH"))
        cols["hist_hl"].append(labels.count("HL"))
        cols["hist_lh"].append(labels.count("LH"))
        cols["hist_ll"].append(labels.count("LL"))
        cols["hist_avg_significance_atr"].append(float(np.mean(sig)) if sig else np.nan)
        lb = breaks[-1] if breaks else None
        cols["last_break_type"].append(lb.break_type if lb else None)
        cols["last_break_direction"].append(lb.direction if lb else None)
        cols["last_break_level"].append(lb.level if lb else np.nan)
        cols["last_break_time"].append(lb.break_time if lb else pd.NaT)
        cols["last_break_magnitude"].append(lb.magnitude if lb else np.nan)
        cols["last_break_atr"].append(lb.magnitude_atr if lb else np.nan)
        cols["last_break_status"].append(lb.state if lb else None)
        cols["last_break_previous_status"].append(lb.previous_state if lb else None)
        cols["last_break_transition_reason"].append(lb.transition_reason if lb else None)
        cols["last_break_confirmed_at"].append(lb.confirmed_at if lb and lb.confirmed_at is not None else pd.NaT)
        cols["last_break_failed_at"].append(lb.failed_at if lb and lb.failed_at is not None else pd.NaT)
        cols["last_break_structure_before"].append(lb.structure_before if lb else None)
        cols["last_break_structure_after"].append(lb.structure_after if lb else None)
        cols["bars_since_break"].append(c - lb.break_index if lb else np.nan)
        win = []
        for ev in reversed(breaks):
            if c - ev.break_index >= W:
                break
            win.append(ev)
        win.reverse()
        cols["breaks_up_window"].append(sum(ev.direction == "bullish" for ev in win))
        cols["breaks_down_window"].append(sum(ev.direction == "bearish" for ev in win))
        cols["breaks_invalidated_window"].append(sum(ev.state == INVALIDATED for ev in win))
        cols["breaks_failed_window"].append(sum(ev.state == FAILED for ev in win))

        # 6) structure quality
        q = _quality(memory[-cfg.quality_swings:], win, states[-W:], state, ov[c], wk[c])
        for k, v in q.items():
            qcols[k].append(v)

    frame = pd.DataFrame({**cols, **qcols}, index=df.index)
    return StructureResult(frame=frame, swings=swings, breaks=breaks, history_size=H)


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
    unsuccessful = sum(ev.state in TERMINAL_STATES for ev in win)  # invalidated or failed breaks = noise
    total = up + down
    if total:
        conflict = 2.0 * min(up, down) / total
        conflict_score = (1.0 - conflict) * (1.0 - 0.5 * unsuccessful / total)
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
