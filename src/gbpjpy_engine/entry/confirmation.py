"""Entry confirmation families (Phase 1D).

Each family is evaluated independently at the CLOSE of H1 bar ``c`` for one
side and returns a score in [0, 100] plus the evidence it used.  Results are
stored per bar for every family so later validation can test them one by one;
no family is assumed to be as useful as another.

Only completed H1 bars are used.  When a single H1 bar contains both a level
interaction and a subsequent move, the order of those intrabar events is NOT
known and is never assumed: every family reasons about closes, extremes and
bodies of completed bars only.

Point-in-time rules: breaks are read through ``transitions_known_at(c)`` /
``state_at(c)`` (never their final lifecycle fields), sweeps and expansions
through ``state_at(c)``, and only bars ``<= c`` are touched.
"""

from __future__ import annotations

import numpy as np

from ..features.indicators import scale01
from ..features.structure import ACCEPTED, CANDIDATE, CONFIRMED, FAILED, INVALIDATED
from .config import CONFIRMATION_FAMILIES

LIFECYCLE = {CANDIDATE: 0.5, CONFIRMED: 0.8, ACCEPTED: 1.0}
STATE_RANK = {CANDIDATE: 0, CONFIRMED: 1, ACCEPTED: 2}
GROUPS = {
    "STRUCTURE": ("STRUCTURAL_BREAK_CONFIRMATION",),
    "DISPLACEMENT": ("DISPLACEMENT_CONFIRMATION", "COMPRESSION_EXPANSION_CONFIRMATION"),
    "RECLAIM_RETEST": ("BREAK_RETEST_CONFIRMATION", "SWEEP_RECLAIM_CONFIRMATION"),
    "MOMENTUM": ("MOMENTUM_REACCELERATION_CONFIRMATION",),
}
GROUP_WEIGHTS = {"STRUCTURE": 0.30, "DISPLACEMENT": 0.25, "RECLAIM_RETEST": 0.20, "MOMENTUM": 0.15, "MARKET_QUALITY": 0.10}
FAMILY_GROUP = {f: g for g, fams in GROUPS.items() for f in fams}
SHORT_NAME = {"STRUCTURAL_BREAK_CONFIRMATION": "structural_break", "DISPLACEMENT_CONFIRMATION": "displacement",
              "BREAK_RETEST_CONFIRMATION": "break_retest", "SWEEP_RECLAIM_CONFIRMATION": "sweep_reclaim",
              "MOMENTUM_REACCELERATION_CONFIRMATION": "reacceleration",
              "COMPRESSION_EXPANSION_CONFIRMATION": "compression_expansion"}


def _empty(family: str, why: str) -> dict:
    return {"family": family, "score": 0.0, "detected": False, "reference_level": None, "components": {},
            "evidence": {"note": why}}


def _dirloc(a, j, d) -> float:
    rng = a["high"][j] - a["low"][j]
    loc = (a["close"][j] - a["low"][j]) / rng if rng > 0 else 0.5
    return loc if d > 0 else 1.0 - loc


def _body_ratio(a, j, d) -> float:
    rng = a["high"][j] - a["low"][j]
    return max(d * (a["close"][j] - a["open"][j]), 0.0) / rng if rng > 0 else 0.0


def _round(comps: dict) -> dict:
    return {k: round(float(v), 4) for k, v in comps.items()}


# ---------------------------------------------------------------------------
def structural_break(c: int, d: int, q: int, breaks, a: dict, swings, zones_by_bar, cfg, importance_norm_atr: float) -> dict:
    fam = "STRUCTURAL_BREAK_CONFIRMATION"
    want = "bullish" if d > 0 else "bearish"
    best = None
    for ev in breaks:
        b = ev.break_index
        if ev.direction != want or not (q < b <= c) or c - b >= cfg.structural_lookback_bars:
            continue
        state = ev.state_at(c)
        if state in (None, FAILED, INVALIDATED):
            continue
        atr_b = ev.atr_at_break if ev.atr_at_break > 0 else a["atr"][b]
        sw = swings[ev.swing_id]
        sig = sw.significance_atr if sw.significance_atr is not None else 1.0
        zs = zones_by_bar[b - 1] if b >= 1 else []
        zone_strength = max((z.strength for z in zs if z.lower - 1e-9 <= ev.level <= z.upper + 1e-9), default=0.0)
        extreme = a["high"][b] if d > 0 else a["low"][b]
        ft = float(np.max(d * (a["close"][b:c + 1] - ev.level))) / atr_b
        comps = {
            "importance": 0.6 * float(scale01(sig, 0.5, importance_norm_atr)) + 0.4 * zone_strength / 100.0,
            "close_beyond": float(scale01(ev.magnitude_atr, 0.25, 1.0)),
            "penetration": float(scale01(d * (extreme - ev.level) / atr_b, 0.25, 1.5)),
            "body_quality": float(scale01(_body_ratio(a, b, d), 0.4, 0.8)),
            "close_location": float(scale01(_dirloc(a, b, d), 0.5, 0.9)),
            "displacement": float(a["disp"][d][b]) / 100.0,
            "follow_through": float(scale01(ft, 0.25, 1.5)),
            "lifecycle": LIFECYCLE[state],
        }
        w = {"importance": 0.20, "close_beyond": 0.20, "penetration": 0.05, "body_quality": 0.15, "close_location": 0.10,
             "displacement": 0.15, "follow_through": 0.05, "lifecycle": 0.10}
        score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
        meets_state = STATE_RANK[state] >= STATE_RANK[cfg.structural_min_state]
        cand = {"family": fam, "score": round(score if meets_state else min(score, 40.0), 2), "detected": True,
                "reference_level": float(ev.level), "components": _round(comps),
                "evidence": {"break_event_id": ev.event_id, "break_index": b, "broken_level": float(ev.level),
                             "break_type": ev.break_type, "state": state, "level_significance_atr": sig,
                             "level_zone_strength": zone_strength, "close_beyond_atr": round(float(ev.magnitude_atr), 4),
                             "penetration_atr": round(float(d * (extreme - ev.level) / atr_b), 4),
                             "follow_through_atr": round(ft, 4), "meets_min_state": meets_state}}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best or _empty(fam, "no qualifying H1 structural break since qualification")


# ---------------------------------------------------------------------------
def displacement(c: int, d: int, a: dict, cfg, window: int) -> dict:
    fam = "DISPLACEMENT_CONFIRMATION"
    score = float(a["disp"][d][c])
    multibar = bool(a["multibar"][d][c])
    isolated = (not multibar) and np.isfinite(a["range_atr"][c]) and a["range_atr"][c] >= cfg.giant_candle_atr
    if isolated:
        score *= 0.6
    ref_j = max(0, c - window)
    return {"family": fam, "score": round(score, 2), "detected": score > 0, "reference_level": float(a["close"][ref_j]),
            "components": {"phase1b_displacement": round(float(a["disp"][d][c]), 2), "multi_bar": multibar,
                           "isolated_giant_candle": bool(isolated)},
            "evidence": {"window_start_close": float(a["close"][ref_j]), "range_atr": round(float(a["range_atr"][c]), 4)
                         if np.isfinite(a["range_atr"][c]) else None}}


# ---------------------------------------------------------------------------
def break_retest(c: int, d: int, breaks, a: dict, cfg) -> dict:
    fam = "BREAK_RETEST_CONFIRMATION"
    want = "bullish" if d > 0 else "bearish"
    best = None
    for ev in breaks:
        b = ev.break_index
        if ev.direction != want or not (b < c) or c - b > cfg.retest_max_bars:
            continue
        state = ev.state_at(c)
        atr_b = ev.atr_at_break if ev.atr_at_break > 0 else a["atr"][b]
        lvl = ev.level
        tol = cfg.retest_tolerance_atr * atr_b
        closes = a["close"][b + 1:c + 1]
        failed = state in (FAILED, INVALIDATED) or (len(closes) and np.min(d * (closes - lvl)) < -cfg.retest_max_penetration_atr * atr_b)
        ext = a["low"] if d > 0 else a["high"]
        touches = [j for j in range(b + 1, c + 1) if d * (ext[j] - lvl) <= tol]
        info = {"break_event_id": ev.event_id, "break_index": b, "break_level": float(lvl), "state": state}
        if failed:
            cand = {"family": fam, "score": 0.0, "detected": False, "reference_level": float(lvl), "components": {},
                    "evidence": {**info, "status": "BREAKOUT_FAILURE"}}
        elif state not in (CONFIRMED, ACCEPTED) or not touches:
            cand = {"family": fam, "score": 0.0, "detected": False, "reference_level": float(lvl), "components": {},
                    "evidence": {**info, "status": "NO_RETEST" if not touches else "BREAK_NOT_ACCEPTED"}}
        else:
            j0 = touches[0]
            seg = ext[j0:c + 1]
            retest_extreme = float(np.min(seg) if d > 0 else np.max(seg))
            penetration = max(d * (lvl - retest_extreme), 0.0) / atr_b
            distance = d * (retest_extreme - lvl) / atr_b
            atr_c = a["atr"][c] if np.isfinite(a["atr"][c]) and a["atr"][c] > 0 else atr_b
            response_move = d * (a["close"][c] - retest_extreme) / atr_c
            responding = d * (a["close"][c] - a["open"][c]) > 0 and d * (a["close"][c] - lvl) > 0 and response_move >= 0.5
            comps = {
                "break_quality": 0.5 * float(scale01(ev.magnitude_atr, 0.25, 1.0)) + 0.5 * LIFECYCLE.get(state, 0.0),
                "acceptance": 1.0 if state == ACCEPTED else 0.7,
                "controlled_retest": 1.0 - float(scale01(penetration, 0.0, cfg.retest_max_penetration_atr)),
                "retest_proximity": 1.0 - float(scale01(abs(distance), 0.0, cfg.retest_tolerance_atr)),
                "response": (0.5 * float(scale01(response_move, 0.5, 1.5)) + 0.25 * float(scale01(_dirloc(a, c, d), 0.5, 0.9))
                             + 0.25 * float(a["disp"][d][c]) / 100.0) if responding else 0.0,
            }
            w = {"break_quality": 0.25, "acceptance": 0.15, "controlled_retest": 0.20, "retest_proximity": 0.10, "response": 0.30}
            score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
            cand = {"family": fam, "score": round(score if responding else min(score, 40.0), 2), "detected": bool(responding),
                    "reference_level": float(lvl), "components": _round(comps),
                    "evidence": {**info, "status": "VALID_RETEST" if responding else "AWAITING_RESPONSE",
                                 "retest_index": j0, "retest_distance_atr": round(float(distance), 4),
                                 "penetration_atr": round(float(penetration), 4), "retest_duration_bars": int(c - j0),
                                 "response_move_atr": round(float(response_move), 4)}}
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best or _empty(fam, "no recent H1 break in this direction to retest")


# ---------------------------------------------------------------------------
def sweep_reclaim(c: int, d: int, sweeps, reclaims, breaks, a: dict, sweep_score_now: float, cfg) -> dict:
    fam = "SWEEP_RECLAIM_CONFIRMATION"
    want = "bullish" if d > 0 else "bearish"
    ev = None
    for s in reversed(sweeps):
        if s.index >= c:
            continue
        if c - s.index > cfg.sweep_max_bars:
            break
        if s.implication == want and s.state_at(c) != "BREAK_AND_ACCEPT":
            ev = s
            break
    if ev is None:
        return _empty(fam, "no recent rejected liquidity sweep implying this direction")
    e = ev.index
    reclaim_level = float(a["high"][e] if d > 0 else a["low"][e])  # the sweep candle's opposite extreme
    atr = a["atr"][c] if np.isfinite(a["atr"][c]) and a["atr"][c] > 0 else ev.atr
    beyond = d * (a["close"][c] - reclaim_level) / atr
    rc = [r for r in reclaims if r.direction == want and e < r.reclaim_index <= c]
    rc_score = max((r.evaluate(c, a)["reclaim_score"] for r in rc), default=0.0)
    consequence = any(b.direction == want and e < b.break_index <= c and b.state_at(c) not in (FAILED, INVALIDATED)
                      for b in breaks)
    disp = float(a["disp"][d][c]) / 100.0
    reclaimed = beyond > 0 or rc_score >= 50.0
    comps = {
        "sweep_quality": float(sweep_score_now or 0.0) / 100.0,
        "reclaim_quality": max(float(scale01(beyond, 0.0, 0.75)) if beyond > 0 else 0.0, rc_score / 100.0),
        "displacement": disp,
        "structural_consequence": 1.0 if consequence else 0.0,
    }
    w = {"sweep_quality": 0.30, "reclaim_quality": 0.30, "displacement": 0.20, "structural_consequence": 0.20}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    ok = reclaimed and (disp >= 0.4 or consequence)
    return {"family": fam, "score": round(score if ok else min(score, 40.0), 2), "detected": bool(ok),
            "reference_level": reclaim_level, "components": _round(comps),
            "evidence": {"sweep_event_id": ev.event_id, "sweep_index": e, "swept_level": float(ev.level),
                         "sweep_extreme": float(a["low"][e] if d > 0 else a["high"][e]), "reclaim_level": reclaim_level,
                         "close_beyond_reclaim_atr": round(float(beyond), 4), "reclaim_event_score": rc_score,
                         "note": "a sweep alone never confirms: reclaim plus displacement or structure is required"}}


# ---------------------------------------------------------------------------
def reacceleration(c: int, d: int, a: dict, counter_deterioration: float, cfg) -> dict:
    fam = "MOMENTUM_REACCELERATION_CONFIRMATION"
    n = cfg.reacceleration_lookback
    if c < n + 1:
        return _empty(fam, "insufficient bars")
    mom = d * a["momentum_net"]
    m_now, m_prev = mom[c], mom[c - 1]
    resumed = bool(np.isfinite(m_now) and np.isfinite(m_prev) and m_now > 0 and m_now > m_prev)
    bodies = d * (a["close"][c - 2:c + 1] - a["open"][c - 2:c + 1])
    prog = 1.0 if bodies[2] > bodies[1] > 0 else (0.5 if bodies[2] > 0 and bodies[2] > bodies[1] else 0.0)
    closes = float(np.mean(bodies > 0))
    prior = a["close"][c - n:c]
    struct = 1.0 if d * (a["close"][c] - (np.max(prior) if d > 0 else np.min(prior))) > 0 else 0.0
    path = float(np.sum(np.abs(np.diff(a["close"][c - n:c + 1]))))
    net = d * (a["close"][c] - a["close"][c - n])
    eff = net / path if path > 0 else 0.0
    comps = {
        "counter_momentum_deterioration": float(np.clip(counter_deterioration, 0, 1)),
        "body_progression": prog,
        "directional_closes": closes,
        "displacement": float(a["disp"][d][c]) / 100.0,
        "structural_progress": struct,
        "efficiency": float(scale01(eff, 0.3, 0.8)),
    }
    w = {"counter_momentum_deterioration": 0.20, "body_progression": 0.15, "directional_closes": 0.15,
         "displacement": 0.20, "structural_progress": 0.15, "efficiency": 0.15}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    origin = float(np.min(a["low"][c - n:c + 1]) if d > 0 else np.max(a["high"][c - n:c + 1]))
    return {"family": fam, "score": round(score if resumed else min(score, 30.0), 2), "detected": resumed,
            "reference_level": origin, "components": _round(comps),
            "evidence": {"momentum_now": round(float(m_now), 4) if np.isfinite(m_now) else None,
                         "momentum_prev": round(float(m_prev), 4) if np.isfinite(m_prev) else None,
                         "momentum_resumed": resumed, "reacceleration_origin": origin}}


# ---------------------------------------------------------------------------
def compression_expansion(c: int, d: int, expansions, breaks_by_id, a: dict, h4_confidence: float, cfg) -> dict:
    fam = "COMPRESSION_EXPANSION_CONFIRMATION"
    want = "bullish" if d > 0 else "bearish"
    ev = None
    for x in reversed(expansions):
        if x.index > c:
            continue
        if c - x.index > cfg.expansion_max_bars:
            break
        if x.direction == want and x.state_at(c) in ("EXPANDING", "EXPANSION_ACCEPTED"):
            ev = x
            break
    if ev is None:
        return _empty(fam, "no recent expansion out of compression in this direction")
    e, cs = ev.index, ev.compression_start
    rng = a["high"] - a["low"]
    base = float(np.mean(rng[cs:e])) if e > cs else float("nan")
    ratio = rng[e] / base if np.isfinite(base) and base > 0 else 0.0
    bst = breaks_by_id[ev.break_event_id].state_at(c) if ev.break_event_id is not None else None
    consequence = {ACCEPTED: 1.0, CONFIRMED: 1.0, CANDIDATE: 0.5}.get(bst, 0.0)
    comps = {
        "range_expansion": float(scale01(ratio, 1.3, 2.5)),
        "body_quality": float(scale01(_body_ratio(a, e, d), 0.4, 0.8)),
        "direction": float(scale01(_dirloc(a, e, d), 0.5, 0.9)),
        "structural_consequence": consequence,
        "h4_permission": float(np.clip(h4_confidence / 100.0, 0, 1)),
    }
    w = {"range_expansion": 0.25, "body_quality": 0.15, "direction": 0.15, "structural_consequence": 0.30, "h4_permission": 0.15}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    extreme = np.isfinite(a["range_atr"][e]) and a["range_atr"][e] >= cfg.extreme_expansion_atr
    if extreme:
        score = min(score, 40.0)
    box = float(np.max(a["high"][cs:e]) if d > 0 else np.min(a["low"][cs:e])) if e > cs else float(a["open"][e])
    return {"family": fam, "score": round(score, 2), "detected": True, "reference_level": box, "components": _round(comps),
            "evidence": {"expansion_event_id": ev.event_id, "expansion_index": e, "compression_start": cs,
                         "range_ratio": round(float(ratio), 4), "linked_break_state": bst, "compression_boundary": box,
                         "extreme_expansion_chase": bool(extreme)}}


# ---------------------------------------------------------------------------
def confirmation_quality(scores: dict, primary: str | None, market_quality: float, w_primary: float) -> tuple[float, dict]:
    """Primary family + support from the OTHER evidence groups (max within a group; the primary's group excluded)."""
    groups = {g: max(scores[f] for f in fams) for g, fams in GROUPS.items()}
    groups["MARKET_QUALITY"] = float(np.clip(market_quality, 0, 100))
    if primary is None:
        return 0.0, {"groups": {g: round(v, 2) for g, v in groups.items()}, "primary": None, "support": 0.0}
    pg = FAMILY_GROUP[primary]
    others = {g: wt for g, wt in GROUP_WEIGHTS.items() if g != pg}
    support = sum(groups[g] * wt for g, wt in others.items()) / sum(others.values())
    q = w_primary * scores[primary] + (1.0 - w_primary) * support
    return round(float(q), 2), {"groups": {g: round(v, 2) for g, v in groups.items()}, "primary": primary,
                                "primary_group": pg, "support": round(float(support), 2)}


def evaluate_families(c, d, q, ctx, cfg) -> dict:
    """All six families for one side at bar ``c`` (``q`` = qualification bar; structure must be newer than it)."""
    a = ctx["a"]
    return {
        "STRUCTURAL_BREAK_CONFIRMATION": structural_break(c, d, q, ctx["recent_breaks"], a, ctx["swings"], ctx["zones_by_bar"],
                                                          cfg, ctx["importance_norm_atr"]),
        "DISPLACEMENT_CONFIRMATION": displacement(c, d, a, cfg, ctx["disp_window"]),
        "BREAK_RETEST_CONFIRMATION": break_retest(c, d, ctx["recent_breaks"], a, cfg),
        "SWEEP_RECLAIM_CONFIRMATION": sweep_reclaim(c, d, ctx["sweeps"], ctx["reclaims"], ctx["recent_breaks"], a,
                                                    ctx["sweep_score"][d], cfg),
        "MOMENTUM_REACCELERATION_CONFIRMATION": reacceleration(c, d, a, ctx["counter_det"][d], cfg),
        "COMPRESSION_EXPANSION_CONFIRMATION": compression_expansion(c, d, ctx["expansions"], ctx["breaks_by_id"], a,
                                                                    ctx["h4_confidence"], cfg),
    }


__all__ = ["CONFIRMATION_FAMILIES", "GROUPS", "confirmation_quality", "evaluate_families"]
