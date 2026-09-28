"""Trend maturity, momentum deterioration, structural compression and
expansion-after-compression.

Maturity (direction = prevailing structural direction)
------------------------------------------------------
The current move starts at the swing preceding the longest run of
direction-consistent labels at the end of the swing memory (so the horizon is
bounded by ``swing_history_size`` - documented limitation).  Inputs: with-trend
impulses, corrections, cumulative distance from the trend origin (ATR),
duration, Phase 1A extension, momentum deterioration and failed continuation
attempts.  ``maturity_score`` = mean of three 0-1 scalings (impulse count,
distance, duration).  Classes: EARLY, DEVELOPING, MATURE, EXTENDED,
EXHAUSTION_RISK (mature/extended with deterioration), UNKNOWN (no direction).
An extended market is NEVER a counter-trend signal.

Momentum deterioration (0-100): equal-weighted evidence of smaller impulses,
deeper corrections, slower velocity, weaker impulse quality (displacement),
lower efficiency, more overlap, failed impulses and failed/invalidated breaks
in the trend direction.

Compression (0-100): swing-amplitude contraction, falling short/long ATR,
contracting candle ranges, high overlap, converging swings (LH + HL) and low
efficiency.  No directional implication.

Expansion: a compression episode (``min_compression_bars`` bars at/above the
threshold) followed - during it or within ``expansion_window_bars`` after it -
by MODERATE+ displacement or a new structural break creates an expansion
event - provided the bar's range is at least ``expansion_range_ratio`` x the
mean range during the compression episode (ATR-normalised displacement alone
is not enough once ATR itself has compressed).  Events are tracked for
``expansion_track_bars`` against the linked break's lifecycle; an expansion
without an accepted break is marked EXPANSION_FADED after
``expansion_window_bars``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import CompressionConfig, MaturityConfig
from ..features.indicators import scale01
from ..features.structure import ACCEPTED, CONFIRMED, FAILED, INVALIDATED

BULL = ("HH", "HL")
BEAR = ("LH", "LL")


def trend_run(memory, direction: int):
    """Swings of the current direction-consistent run (including its origin swing)."""
    if direction == 0 or not memory:
        return []
    ok = BULL if direction > 0 else BEAR
    k = len(memory)
    while k > 0 and memory[k - 1].label in ok:
        k -= 1
    start = max(k - 1, 0)
    return memory[start:]


def deterioration(impulses, corrections, recent_legs, failed_breaks_in_dir: int) -> tuple[float, dict]:
    comps = {}
    if len(impulses) >= 2:
        last, prev = impulses[-1], impulses[-4:-1]
        pd_ = np.mean([x.distance_atr for x in prev])
        pv = np.mean([x.velocity_atr for x in prev])
        comps["smaller_impulses"] = float(scale01(1 - last.distance_atr / pd_, 0.0, 0.5)) if pd_ > 0 else 0.0
        comps["slower_velocity"] = float(scale01(1 - last.velocity_atr / pv, 0.0, 0.5)) if pv > 0 else 0.0
        comps["reduced_displacement"] = float(scale01((np.mean([x.impulse_score for x in prev]) - last.impulse_score) / 100, 0.0, 0.3))
        comps["reduced_efficiency"] = float(scale01(np.mean([x.efficiency for x in prev]) - last.efficiency, 0.0, 0.3))
        comps["more_overlap"] = float(scale01(last.overlap - np.mean([x.overlap for x in prev]), 0.0, 0.2))
    ratios = [x.retracement_ratio for x in corrections if x.retracement_ratio is not None]
    if len(ratios) >= 2:
        comps["deeper_corrections"] = float(scale01(ratios[-1] - np.mean(ratios[:-1]), 0.0, 0.4))
    comps["failed_impulses"] = 1.0 if any(x.classification == "FAILED_IMPULSE" for x in recent_legs[-4:]) else 0.0
    comps["failed_extensions"] = float(scale01(failed_breaks_in_dir, 0.0, 2.0))
    if len(impulses) < 2:
        # without two completed impulses only failure evidence is meaningful
        comps = {k: v for k, v in comps.items() if k in ("failed_impulses", "failed_extensions", "deeper_corrections")}
    score = 100.0 * float(np.mean(list(comps.values()))) if comps else 0.0
    return round(score, 2), {k: round(v, 4) for k, v in comps.items()}


def maturity(memory, legs, direction: int, c: int, close: float, atr: float, extension_state: str,
             extension_direction: str, deterioration_score: float, cfg: MaturityConfig) -> dict:
    if direction == 0 or not np.isfinite(atr) or atr <= 0:
        return {"trend_maturity": "UNKNOWN", "maturity_score": None, "trend_impulses": 0, "trend_corrections": 0,
                "trend_distance_atr": None, "trend_duration_bars": None, "trend_origin_swing": None}
    run = trend_run(memory, direction)
    ids = {s.swing_id for s in run}
    in_run = [lg for lg in legs if not lg.is_active and lg.start_swing_id in ids and lg.end_swing_id in ids]
    imp = [lg for lg in in_run if (1 if lg.direction == "up" else -1) == direction]
    cor = [lg for lg in in_run if (1 if lg.direction == "up" else -1) != direction]
    origin = run[0] if run else None
    dist = direction * (close - origin.price) / atr if origin else 0.0
    dur = c - origin.pivot_index if origin else 0
    score = float(np.mean([
        scale01(len(imp), 1, cfg.impulses_full), scale01(dist, 3.0, cfg.distance_full_atr), scale01(dur, 20, cfg.duration_full_bars),
    ]))
    same_dir_ext = extension_state == "extremely_extended" and extension_direction == ("up" if direction > 0 else "down")
    if len(imp) <= 1 and score < cfg.mature_score:
        cls = "EARLY"
    elif score < cfg.developing_score:
        cls = "EARLY"
    elif score < cfg.mature_score:
        cls = "DEVELOPING"
    elif score < cfg.extended_score and not same_dir_ext:
        cls = "MATURE"
    else:
        cls = "EXTENDED"
    if cls in ("MATURE", "EXTENDED") and deterioration_score >= cfg.exhaustion_deterioration:
        cls = "EXHAUSTION_RISK"
    return {"trend_maturity": cls, "maturity_score": round(score, 4), "trend_impulses": len(imp),
            "trend_corrections": len(cor), "trend_distance_atr": round(float(dist), 4),
            "trend_duration_bars": int(dur), "trend_origin_swing": origin.swing_id if origin else None}


def compression(memory, row_arrays: dict, c: int, cfg: CompressionConfig) -> tuple[float, dict]:
    a = row_arrays
    win = memory[-cfg.amplitude_swings:]
    comps = {}
    amps = [abs(win[k].price - win[k - 1].price) / win[k].atr_at_confirmation for k in range(1, len(win))
            if win[k].atr_at_confirmation > 0]
    if len(amps) >= 4:
        half = len(amps) // 2
        first, second = np.mean(amps[:half]), np.mean(amps[half:])
        comps["amplitude_contraction"] = float(scale01((first - second) / first, 0.0, 0.5)) if first > 0 else 0.0
    ratio = a["atr_ratio"][c]
    comps["atr_decline"] = float(scale01(1.0 - ratio, 0.0, 0.3)) if np.isfinite(ratio) else 0.0
    s0 = max(0, c - cfg.range_short + 1)
    l0 = max(0, c - cfg.range_long + 1)
    rng = a["high"] - a["low"]
    if c - l0 + 1 >= cfg.range_long:
        comps["range_contraction"] = float(scale01(1.0 - np.mean(rng[s0:c + 1]) / np.mean(rng[l0:c + 1]), 0.0, 0.4))
    ov = a["overlap_mean"][c]
    comps["overlap"] = float(scale01(ov, 0.45, 0.80)) if np.isfinite(ov) else 0.0
    hl = next((s.label for s in reversed(win) if s.kind == "high"), None)
    ll = next((s.label for s in reversed(win) if s.kind == "low"), None)
    comps["converging_structure"] = (0.5 if hl in ("LH", "EH") else 0.0) + (0.5 if ll in ("HL", "EL") else 0.0)
    eff = a["eff"][c]
    comps["low_efficiency"] = 1.0 - float(scale01(eff, 0.15, 0.45)) if np.isfinite(eff) else 0.0
    weights = {"amplitude_contraction": 0.20, "atr_decline": 0.15, "range_contraction": 0.20, "overlap": 0.15,
               "converging_structure": 0.15, "low_efficiency": 0.15}
    used = {k: w for k, w in weights.items() if k in comps}
    score = 100.0 * sum(comps[k] * w for k, w in used.items()) / sum(used.values())
    return round(score, 2), {k: round(v, 4) for k, v in comps.items()}


@dataclass
class ExpansionEvent:
    event_id: int
    index: int
    direction: str
    compression_start: int
    compression_duration: int
    displacement_score: float
    structural_break: bool
    break_event_id: int | None
    transitions: list = field(default_factory=list)

    def state_at(self, c: int) -> str | None:
        st = None
        for idx, to, _ in self.transitions:
            if idx <= c:
                st = to
        return st


class ExpansionTracker:
    def __init__(self, cfg: CompressionConfig, moderate_disp: float):
        self.cfg = cfg
        self.moderate = moderate_disp
        self.run = 0
        self.episode_start: int | None = None
        self.episode_end: int | None = None
        self.episode_len = 0
        self.events: list[ExpansionEvent] = []
        self.ranges: list[float] = []  # bar ranges of the current/last compression run

    def step(self, c: int, comp_score: float, bull_disp: float, bear_disp: float, new_breaks, breaks_by_id,
             bar_range: float = float("nan")) -> dict:
        cfg = self.cfg
        compressed = comp_score >= cfg.compression_threshold
        base_range = float(np.mean(self.ranges)) if self.ranges else float("nan")
        if compressed:
            if self.run == 0:
                self.ranges = []
            self.ranges.append(bar_range)
            self.run += 1
            if self.run >= cfg.min_compression_bars:
                self.episode_start = c - self.run + 1
                self.episode_len = self.run
                self.episode_end = None
        else:
            if self.run >= cfg.min_compression_bars:
                self.episode_end = c - 1
            self.run = 0
        in_episode = self.run >= cfg.min_compression_bars
        recent_end = self.episode_end is not None and c - self.episode_end <= cfg.expansion_window_bars
        expanding_range = np.isfinite(base_range) and base_range > 0 and bar_range >= cfg.expansion_range_ratio * base_range
        if (in_episode or recent_end) and self.episode_start is not None and expanding_range:
            direction = None
            brk = new_breaks[0] if new_breaks else None
            if brk is not None:
                direction = brk.direction
            elif max(bull_disp, bear_disp) >= self.moderate:
                direction = "bullish" if bull_disp >= bear_disp else "bearish"
            if direction is not None:
                ev = ExpansionEvent(
                    event_id=len(self.events), index=c, direction=direction, compression_start=self.episode_start,
                    compression_duration=self.episode_len, displacement_score=float(bull_disp if direction == "bullish" else bear_disp),
                    structural_break=brk is not None, break_event_id=brk.event_id if brk is not None else None,
                )
                ev.transitions.append((c, "EXPANDING", "displacement/break out of compression"))
                self.events.append(ev)
                self.run, self.episode_start, self.episode_end = 0, None, None
                in_episode = False
        # evolve the latest event from its linked break (point-in-time) and report
        state, latest = ("COMPRESSING" if in_episode else "NONE"), None
        if self.events and c - self.events[-1].index <= cfg.expansion_track_bars:
            ev = self.events[-1]
            cur = ev.state_at(c)
            if ev.break_event_id is not None:
                bst = breaks_by_id[ev.break_event_id].state_at(c)
                if bst in (ACCEPTED, CONFIRMED) and cur == "EXPANDING":
                    ev.transitions.append((c, "EXPANSION_ACCEPTED", f"linked break {bst}"))
                elif bst in (FAILED, INVALIDATED) and cur != "EXPANSION_FAILED":
                    ev.transitions.append((c, "EXPANSION_FAILED", f"linked break {bst}"))
            cur = ev.state_at(c)
            if cur == "EXPANDING" and c - ev.index > cfg.expansion_window_bars:
                ev.transitions.append((c, "EXPANSION_FADED", "no accepted structural break within the expansion window"))
                cur = "EXPANSION_FADED"
            quality = 100.0 * float(np.mean([ev.displacement_score / 100.0, 1.0 if ev.structural_break else 0.0,
                                             {"EXPANSION_ACCEPTED": 1.0, "EXPANSION_FAILED": 0.0}.get(cur, 0.5)]))
            latest = {"event_id": ev.event_id, "index": ev.index, "direction": ev.direction,
                      "compression_duration": ev.compression_duration, "displacement_score": ev.displacement_score,
                      "structural_break": ev.structural_break, "state": cur, "expansion_quality": round(quality, 2)}
            if not in_episode and cur != "EXPANSION_FADED":
                state = {"EXPANDING": f"EXPANDING_{ev.direction.upper()}"}.get(cur, cur)
        return {"expansion_state": state, "latest_expansion": latest,
                "compression_episode_bars": self.run if in_episode else 0}
