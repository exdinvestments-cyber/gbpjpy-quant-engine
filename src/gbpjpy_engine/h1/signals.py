"""H1 signal engines: pullback, structural transition (change of character),
reclaim, rejection quality, descriptive candle patterns, H4/H1 zone
confluence and H1 room to move.

All functions read only data up to the evaluation bar ``c`` and H4 objects
through the aligned (completed) H4 bar index.  Scores are descriptive
baselines - none has been validated against outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..context.room import room as h4_style_room
from ..features.indicators import rolling_percentile, scale01
from ..features.structure import ACCEPTED, CANDIDATE, CONFIRMED, FAILED, INVALIDATED

LIFECYCLE_WEIGHT = {CANDIDATE: 0.5, CONFIRMED: 0.8, ACCEPTED: 1.0, FAILED: 0.0, INVALIDATED: 0.0}


def _sgn(side: str) -> int:
    return 1 if side == "long" else -1


# ---------------------------------------------------------------------------
# Candle patterns (features only - never a setup on their own)
# ---------------------------------------------------------------------------
def candle_patterns(feats: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = feats["open"], feats["high"], feats["low"], feats["close"]
    po, pc, ph, pl = o.shift(1), c.shift(1), h.shift(1), l.shift(1)
    body = (c - o).abs()
    rng = (h - l).replace(0.0, np.nan)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    out = pd.DataFrame(index=feats.index)
    out["pat_bullish_engulfing"] = ((c > o) & (pc < po) & (c >= po) & (o <= pc)).fillna(False)
    out["pat_bearish_engulfing"] = ((c < o) & (pc > po) & (c <= po) & (o >= pc)).fillna(False)
    out["pat_inside_bar"] = ((h <= ph) & (l >= pl)).fillna(False)
    out["pat_outside_bar"] = ((h > ph) & (l < pl)).fillna(False)
    out["pat_bullish_pin"] = ((lower >= 2 * body) & (lower / rng >= 0.5) & (feats["close_location"] >= 0.5)).fillna(False)
    out["pat_bearish_pin"] = ((upper >= 2 * body) & (upper / rng >= 0.5) & (feats["close_location"] <= 0.5)).fillna(False)
    out["pat_strong_bullish_close"] = feats["candle_class"] == "strong_bullish_close"
    out["pat_strong_bearish_close"] = feats["candle_class"] == "strong_bearish_close"
    return out


# ---------------------------------------------------------------------------
# Rejection quality (base score per bar; level interaction + follow-through added per evaluation bar)
# ---------------------------------------------------------------------------
def rejection_base(feats: pd.DataFrame, lookback: int) -> pd.DataFrame:
    o, c = feats["open"], feats["close"]
    rng = (feats["high"] - feats["low"]).replace(0.0, np.nan)
    body = (c - o).abs()
    lower = np.minimum(o, c) - feats["low"]
    upper = feats["high"] - np.maximum(o, c)
    atrp = feats["atr_prev"]
    out = pd.DataFrame(index=feats.index)
    for side, wick, loc in (("bullish", lower, feats["close_location"]), ("bearish", upper, 1.0 - feats["close_location"])):
        wick_atr = wick / atrp
        pct = rolling_percentile(wick_atr, lookback, 30)
        comps = [
            (0.25, scale01(wick / rng, 0.3, 0.7)),
            (0.20, scale01(wick / np.maximum(body, 0.1 * rng), 1.0, 3.0)),
            (0.20, scale01(loc, 0.5, 0.9)),
            (0.20, scale01(wick_atr, 0.3, 1.2)),
            (0.15, scale01(pct.fillna(0.0), 50.0, 95.0)),
        ]
        s = sum(w * np.nan_to_num(np.asarray(x, dtype=float)) for w, x in comps) * 100.0
        out[f"{side}_rejection_base"] = np.round(s, 2)
    return out


def rejection_score(side: str, c: int, a: dict, base: np.ndarray, levels: list[tuple[float, float]], cfg) -> tuple[float, dict]:
    """Best recent rejection (last ``recent_bars``) with level interaction and follow-through measured to ``c``."""
    d = _sgn(side)
    best, info = 0.0, {}
    atr = a["atr"][c]
    if not np.isfinite(atr) or atr <= 0:
        return 0.0, {}
    tol = cfg.level_tolerance_atr * atr
    for j in range(max(0, c - cfg.recent_bars + 1), c + 1):
        extreme = a["low"][j] if d > 0 else a["high"][j]
        touch = any(lo - tol <= extreme <= hi + tol for lo, hi in levels)
        ft = float(scale01(d * (a["close"][c] - a["close"][j]) / atr, 0.0, 1.0)) if j < c else 0.0
        s = base[j] * (0.6 + 0.2 * touch + 0.2 * ft)
        if s > best:
            best, info = s, {"bar": j, "base": float(base[j]), "level_interaction": bool(touch), "follow_through": round(ft, 4)}
    return round(float(best), 2), info


# ---------------------------------------------------------------------------
# Pullback engine
# ---------------------------------------------------------------------------
def reference_impulse(conf_legs, side: str, cfg):
    want = "up" if side == "long" else "down"
    for leg in reversed(conf_legs[-cfg.impulse_lookback_legs:]):
        if leg.direction == want and leg.distance_atr >= cfg.min_impulse_atr:
            return leg
    return None


def pullback(side: str, impulse, c: int, a: dict, momentum_counter: np.ndarray, cfg) -> dict:
    """Correction of ``impulse`` measured on bars after its end pivot up to ``c``."""
    out = {"side": side, "pullback_state": "NO_PULLBACK", "pullback_quality": 50.0, "depth_pct": 0.0,
           "current_depth_pct": 0.0, "depth_atr": 0.0, "duration_bars": 0, "velocity_atr": 0.0, "efficiency": None,
           "overlap": None, "structural_damage": "none", "counter_momentum_deterioration": 0.0, "impulse_leg": None,
           "correction_extreme": None, "impulse_origin": None, "components": {}}
    atr = a["atr"][c]
    if impulse is None or not np.isfinite(atr) or atr <= 0 or c <= impulse.end_index:
        return out
    d = _sgn(side)
    p = impulse.end_index
    lows, highs, closes = a["low"][p + 1:c + 1], a["high"][p + 1:c + 1], a["close"][p:c + 1]
    extreme = float(np.min(lows)) if d > 0 else float(np.max(highs))
    depth = d * (impulse.end_price - extreme)
    depth_pct = max(depth, 0.0) / impulse.distance * 100.0
    cur_pct = d * (impulse.end_price - closes[-1]) / impulse.distance * 100.0
    dur = c - p
    path = float(np.sum(np.abs(np.diff(closes))))
    eff = abs(closes[-1] - closes[0]) / path if path > 0 else 0.0
    ov = a["overlap_prev"][p + 1:c + 1]
    overlap = float(np.nanmean(ov)) if np.isfinite(ov).any() else 0.5
    velocity = max(depth, 0.0) / atr / max(dur, 1)
    origin = impulse.start_price
    beyond_origin = d * (origin - a["close"][c]) / atr  # >0: close beyond the impulse origin
    ext_idx = p + 1 + (int(np.argmin(lows)) if d > 0 else int(np.argmax(highs)))
    counter = momentum_counter[p + 1:c + 1]
    peak = float(np.nanmax(counter)) if np.isfinite(counter).any() else 0.0
    now = float(momentum_counter[c]) if np.isfinite(momentum_counter[c]) else 0.0
    deterioration = float(np.clip(1.0 - now / peak, 0.0, 1.0)) if peak > 0 else 0.0
    if beyond_origin >= cfg.failed_buffer_atr:
        state, damage = "FAILED_PULLBACK_CONTEXT", "impulse origin broken"
    elif depth_pct >= cfg.deep_pct:
        state, damage = "STRUCTURE_THREATENING_PULLBACK", "impulse fully retraced"
    elif depth_pct < cfg.no_pullback_pct:
        state, damage = "NO_PULLBACK", "none"
    elif depth_pct < cfg.shallow_pct:
        state, damage = "SHALLOW_PULLBACK", "none"
    elif depth_pct < cfg.healthy_pct:
        state, damage = "HEALTHY_PULLBACK", "none"
    else:
        state, damage = "DEEP_PULLBACK", "deep retracement"
    lo, hi = cfg.ideal_depth_low, cfg.ideal_depth_high
    depth_c = 1.0 if lo <= depth_pct <= hi else float(np.clip(1.0 - (lo - depth_pct) / lo if depth_pct < lo
                                                              else 1.0 - (depth_pct - hi) / (100.0 - hi), 0.0, 1.0))
    comps = {
        "controlled_velocity": 1.0 - float(scale01(velocity, 0.15, cfg.max_velocity_atr)),
        "reasonable_depth": depth_c,
        "counter_momentum_declining": deterioration,
        "structural_integrity": float(scale01(-beyond_origin, 0.0, 2.0)) if state != "FAILED_PULLBACK_CONTEXT" else 0.0,
        "corrective_overlap": 1.0 - float(scale01(overlap, 0.55, 0.85)),
        "orderly_path": 1.0 - float(scale01(eff, 0.6, 0.95)),
    }
    weights = {"controlled_velocity": 0.20, "reasonable_depth": 0.20, "counter_momentum_declining": 0.20,
               "structural_integrity": 0.20, "corrective_overlap": 0.10, "orderly_path": 0.10}
    quality = 100.0 * sum(comps[k] * w for k, w in weights.items())
    if state == "NO_PULLBACK":
        quality = 50.0
    out.update({
        "pullback_state": state, "pullback_quality": round(quality, 2), "depth_pct": round(depth_pct, 2),
        "current_depth_pct": round(cur_pct, 2), "depth_atr": round(max(depth, 0.0) / atr, 4), "duration_bars": int(dur),
        "velocity_atr": round(velocity, 4), "efficiency": round(eff, 4), "overlap": round(overlap, 4),
        "structural_damage": damage, "counter_momentum_deterioration": round(deterioration, 4),
        "impulse_leg": impulse.leg_id, "correction_extreme": extreme, "correction_extreme_index": ext_idx,
        "impulse_origin": float(origin), "bars_since_extreme": int(c - ext_idx),
        "components": {k: round(v, 4) for k, v in comps.items()},
    })
    return out


# ---------------------------------------------------------------------------
# Structural transition (change of character) and reclaim events
# ---------------------------------------------------------------------------
@dataclass
class TransitionEvent:
    event_id: int
    direction: str  # "bullish" | "bearish"
    break_event_id: int
    broken_level: float
    index: int
    time: str
    structure_before: str
    counter_leg_atr: float
    stall_bars: int
    counter_deterioration: float
    break_atr: float
    displacement: float
    reasons: list = field(default_factory=list)

    def evaluate(self, ev, c: int) -> dict:
        known = ev.transitions_known_at(c)
        state = known[-1].to_state
        conf_at = next((t.at.isoformat() for t in known if t.to_state == CONFIRMED), None)
        comps = {
            "break_magnitude": float(scale01(self.break_atr, 0.25, 1.0)),
            "displacement": self.displacement / 100.0,
            "counter_momentum_deterioration": self.counter_deterioration,
            "counter_structure_stalled": float(scale01(self.stall_bars, 1, 6)),
            "prior_counter_move": float(scale01(self.counter_leg_atr, 1.0, 3.0)),
            "lifecycle": LIFECYCLE_WEIGHT[state],
        }
        w = {"break_magnitude": 0.15, "displacement": 0.20, "counter_momentum_deterioration": 0.15,
             "counter_structure_stalled": 0.15, "prior_counter_move": 0.10, "lifecycle": 0.25}
        score = 0.0 if state in (FAILED, INVALIDATED) else 100.0 * sum(comps[k] * wt for k, wt in w.items())
        return {"transition_direction": self.direction, "transition_score": round(score, 2),
                "broken_level": self.broken_level, "break_state": state, "confirmation_timestamp": conf_at,
                "detected_at": self.time, "bars_since": c - self.index, "components": {k: round(v, 4) for k, v in comps.items()},
                "reason_codes": list(self.reasons), "event_id": self.event_id}


def make_transition(ev, c: int, conf_legs, counter_mom: np.ndarray, bull_disp: float, bear_disp: float, a: dict,
                    cfg, event_id: int) -> TransitionEvent | None:
    """A break against / out of the prevailing H1 structure after a measurable counter move."""
    d = 1 if ev.direction == "bullish" else -1
    if ev.structure_before == ("bullish" if d > 0 else "bearish"):
        return None  # continuation BOS, not a transition
    counter = [lg for lg in conf_legs[-4:] if (1 if lg.direction == "up" else -1) == -d]
    if not counter:
        return None
    leg = counter[-1]
    if leg.distance_atr < cfg.min_counter_leg_atr:
        return None
    seg = a["low"][leg.end_index:c + 1] if d > 0 else a["high"][leg.end_index:c + 1]
    ext_rel = int(np.argmin(seg)) if d > 0 else int(np.argmax(seg))
    stall = (c - leg.end_index) - ext_rel
    cm = counter_mom[max(leg.start_index, 0):c + 1]
    peak = float(np.nanmax(cm)) if np.isfinite(cm).any() else 0.0
    now = float(counter_mom[c]) if np.isfinite(counter_mom[c]) else 0.0
    det = float(np.clip(1.0 - now / peak, 0.0, 1.0)) if peak > 0 else 0.0
    reasons = ["H1_BULLISH_TRANSITION" if d > 0 else "H1_BEARISH_TRANSITION"]
    return TransitionEvent(
        event_id=event_id, direction=ev.direction, break_event_id=ev.event_id, broken_level=float(ev.level), index=c,
        time=(a["ts"][c] + a["tf"]).isoformat(), structure_before=ev.structure_before, counter_leg_atr=float(leg.distance_atr),
        stall_bars=int(stall), counter_deterioration=det, break_atr=float(ev.magnitude_atr),
        displacement=float(bull_disp if d > 0 else bear_disp), reasons=reasons,
    )


@dataclass
class ReclaimEvent:
    event_id: int
    direction: str  # direction of the reclaim (opposite of the failed break)
    level: float
    lost_index: int
    reclaim_index: int
    time: str
    distance_lost_atr: float
    duration_lost_bars: int
    close_quality: float
    displacement: float

    def evaluate(self, c: int, a: dict) -> dict:
        d = 1 if self.direction == "bullish" else -1
        atr = a["atr"][c]
        closes = a["close"][self.reclaim_index:c + 1]
        strength = d * (a["close"][self.reclaim_index] - self.level) / a["atr"][self.reclaim_index]
        ft = float(np.max(d * (closes - self.level))) / atr if np.isfinite(atr) and atr > 0 else 0.0
        comps = {
            "reclaim_strength": float(scale01(strength, 0.0, 0.8)),
            "close_quality": self.close_quality,
            "displacement": self.displacement / 100.0,
            "follow_through": float(scale01(ft, 0.25, 1.5)),
            "brief_loss": 1.0 - float(scale01(self.duration_lost_bars, 2, 10)),
            "loss_size": float(scale01(self.distance_lost_atr, 0.1, 1.0)),
        }
        w = {"reclaim_strength": 0.25, "close_quality": 0.15, "displacement": 0.20, "follow_through": 0.20,
             "brief_loss": 0.10, "loss_size": 0.10}
        still = d * (a["close"][c] - self.level) > 0
        score = 100.0 * sum(comps[k] * wt for k, wt in w.items()) if still else 0.0
        return {"reclaim_direction": self.direction, "reclaim_score": round(score, 2), "level": self.level,
                "distance_lost_atr": round(self.distance_lost_atr, 4), "duration_lost_bars": self.duration_lost_bars,
                "reclaimed_at": self.time, "bars_since": c - self.reclaim_index,
                "components": {k: round(v, 4) for k, v in comps.items()}, "event_id": self.event_id}


def make_reclaim(ev, c: int, a: dict, bull_disp: float, bear_disp: float, event_id: int) -> ReclaimEvent:
    """``ev`` (a break) just became FAILED/INVALIDATED at ``c``: the lost level is reclaimed in the other direction."""
    d_break = 1 if ev.direction == "bullish" else -1
    closes = a["close"][ev.break_index:c]
    lost = float(np.max(d_break * (closes - ev.level))) / ev.atr_at_break if len(closes) else 0.0
    d = -d_break
    rng = a["high"][c] - a["low"][c]
    loc = (a["close"][c] - a["low"][c]) / rng if rng > 0 else 0.5
    return ReclaimEvent(
        event_id=event_id, direction="bullish" if d > 0 else "bearish", level=float(ev.level), lost_index=ev.break_index,
        reclaim_index=c, time=(a["ts"][c] + a["tf"]).isoformat(), distance_lost_atr=lost,
        duration_lost_bars=int(c - ev.break_index), close_quality=float(scale01(loc if d > 0 else 1 - loc, 0.5, 0.9)),
        displacement=float(bull_disp if d > 0 else bear_disp),
    )


# ---------------------------------------------------------------------------
# H4 / H1 zone confluence (clustered - the same price information counts once)
# ---------------------------------------------------------------------------
def confluence(close: float, atr: float, levels: list[dict], cfg) -> dict:
    out = {"clusters": [], "long": None, "short": None, "long_score": 0.0, "short_score": 0.0, "h4_h1_confluence": False}
    if not np.isfinite(atr) or atr <= 0 or not levels:
        return out
    tol = cfg.cluster_tolerance_atr * atr
    lv = sorted(levels, key=lambda x: (x["lower"], x["upper"], x["source"]))
    clusters, cur = [], None
    for x in lv:
        if cur is not None and x["lower"] <= cur["upper"] + tol:
            cur["upper"] = max(cur["upper"], x["upper"])
            cur["members"].append(x)
        else:
            cur = {"lower": x["lower"], "upper": x["upper"], "members": [x]}
            clusters.append(cur)
    res = []
    for cl in clusters:
        types = sorted({m["source"] for m in cl["members"]})
        tfs = sorted({m["timeframe"] for m in cl["members"]})
        strength = max((m.get("strength") or 0.0) for m in cl["members"])
        mid = (cl["lower"] + cl["upper"]) / 2
        dist = 0.0 if cl["lower"] <= close <= cl["upper"] else min(abs(close - cl["lower"]), abs(close - cl["upper"]))
        score = 100.0 * (0.30 * ("H4" in tfs) + 0.30 * ("H1" in tfs) + 0.20 * min(len(types) - 1, 2) / 2 + 0.20 * strength / 100)
        res.append({"lower": cl["lower"], "upper": cl["upper"], "midpoint": mid, "source_types": types,
                    "timeframes": tfs, "n_members": len(cl["members"]), "max_strength": strength,
                    "distance_atr": dist / atr, "confluence_score": round(score, 2),
                    "side": "below" if cl["upper"] < close else ("above" if cl["lower"] > close else "inside")})
    near = [r for r in res if r["distance_atr"] <= cfg.near_atr]
    sup = [r for r in near if r["side"] in ("below", "inside")]
    resist = [r for r in near if r["side"] in ("above", "inside")]
    if sup:
        s = min(sup, key=lambda r: (r["distance_atr"], -r["confluence_score"]))
        out["long"], out["long_score"] = s, s["confluence_score"]
    if resist:
        r = min(resist, key=lambda r: (r["distance_atr"], -r["confluence_score"]))
        out["short"], out["short_score"] = r, r["confluence_score"]
    out["clusters"] = sorted(res, key=lambda r: r["distance_atr"])[:6]
    out["h4_h1_confluence"] = any(len(r["timeframes"]) == 2 for r in near)
    return out


# ---------------------------------------------------------------------------
# H1 room (H1 barriers via the Phase 1B room function + the aligned H4 barrier)
# ---------------------------------------------------------------------------
def h1_room(side: str, close: float, atr: float, c: int, zones, memory, origin_zones, rng_ext: float, rng_idx: int,
            exclude_from: int, strong_level: float, room_cfg, h4_barrier_price) -> dict:
    direction = "long" if side == "long" else "short"
    r1 = h4_style_room(direction, close, atr, c, zones, memory, origin_zones, rng_ext, rng_idx, exclude_from,
                       strong_level, room_cfg)
    h4_score, h4_dist = 100.0, None
    if h4_barrier_price is not None and np.isfinite(atr) and atr > 0:
        dist = (h4_barrier_price - close) if side == "long" else (close - h4_barrier_price)
        h4_dist = max(dist, 0.0) / atr
        h4_score = 100.0 * float(scale01(h4_dist, room_cfg.min_room_atr, room_cfg.full_room_atr))
    score = min(r1["room_score"], h4_score)
    return {"room_score": round(score, 2), "h1_room_score": r1["room_score"], "h1_nearest_barrier": r1["nearest_barrier"],
            "h4_barrier_price": h4_barrier_price, "h4_barrier_distance_atr_h1": h4_dist, "h4_room_score_in_h1": round(h4_score, 2),
            "limiting": "H4" if h4_score < r1["room_score"] else "H1"}
