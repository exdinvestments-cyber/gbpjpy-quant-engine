"""Context conflict, context quality, LONG/SHORT context scores, hard blockers
and DIRECTIONAL PERMISSION.

Permission states: ALLOW_LONG, ALLOW_SHORT, ALLOW_BOTH, BLOCK_ALL.  They say
what the future H1 engine may SEARCH for.  They are not trade signals and
``permission_confidence`` must never be used for position sizing.

Conflict (0-100)
----------------
Each detected contradiction has a documented weight ``w``; the score is a
noisy-OR ``100 x (1 - prod(1 - w))`` so several conflicts accumulate without
exceeding 100 and one conflict cannot be double counted.

Quality (0-100)
---------------
Eight families (STRUCTURE, DISPLACEMENT, MOMENTUM, VOLATILITY, LOCATION,
LIQUIDITY_CONTEXT, ROOM_TO_MOVE, MARKET_QUALITY) each produce ONE score;
the families are weighted (``context_scoring.q_*``).  Correlated inputs live in
the same family so they cannot inflate confidence.

Directional scores (0-100, computed and stored separately for LONG and SHORT)
-----------------------------------------------------------------------------
Positive families: structure hierarchy, continuation behaviour, recent
displacement, level behaviour (support/resistance, role reversal, origin-zone
reaction), rejected liquidity sweeps, room.  Negative evidence is applied as
documented multiplicative reductions (insufficient room, extreme extension,
recent failed break, transition against the direction, severe chop, conflict,
volatility shock/extreme volatility).  The same rules are applied to both
sides, but the two scores are always kept separate for later research.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..config import H4Config

CONFLICT_WEIGHTS = {
    "CONFLICT_PRIMARY_VS_DISPLACEMENT": 0.35,
    "CONFLICT_TREND_VS_OPPOSING_LEVEL": 0.25,
    "CONFLICT_STRUCTURE_VS_FAILED_BREAK": 0.30,
    "CONFLICT_TREND_VS_EXTENSION": 0.20,
    "CONFLICT_STRUCTURE_VS_SHOCK": 0.35,
    "CONFLICT_HIERARCHY_DISAGREEMENT": 0.20,
    "CONFLICT_STRUCTURE_VS_TREND_ENGINE": 0.20,
    "CONFLICT_OPPOSING_SWEEP": 0.15,
    "CONFLICT_FALSE_BREAK_RISK": 0.15,
}
HARD_BLOCKERS = ("INVALID_DATA", "STALE_DATA", "INSUFFICIENT_HISTORY", "UNRESOLVED_DATA_GAP",
                 "EXTREME_VOLATILITY_SHOCK", "SEVERE_CHOP", "UNCLASSIFIABLE_STRUCTURE", "CONTEXT_ERROR")
GAP_FLAGS = ("MISSING_BARS", "EXTENDED_MARKET_CLOSURE", "ABNORMAL_PRICE_GAP", "TIMEZONE_ALIGNMENT_SHIFT")


def _f(x, default=0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def structural_direction(ctx: dict) -> int:
    p, i = ctx["primary"].direction, ctx["intermediate"].direction
    return p if p != 0 else i


# ---------------------------------------------------------------------------
def conflict(ctx: dict, cfg: H4Config) -> tuple[float, list[str], dict]:
    row = ctx["row"]
    D = structural_direction(ctx)
    hits = []
    strong = cfg.displacement.strong_score
    if D > 0 and ctx["bear_disp_recent"] >= strong or D < 0 and ctx["bull_disp_recent"] >= strong:
        hits.append("CONFLICT_PRIMARY_VS_DISPLACEMENT")
    if D > 0 and ctx["long_room"]["room_score"] < 15 or D < 0 and ctx["short_room"]["room_score"] < 15:
        hits.append("CONFLICT_TREND_VS_OPPOSING_LEVEL")
    if D > 0 and ctx["failed_bullish_recent"] or D < 0 and ctx["failed_bearish_recent"]:
        hits.append("CONFLICT_STRUCTURE_VS_FAILED_BREAK")
    if D != 0 and row.get("extension_state") == "extremely_extended" and \
            row.get("extension_direction") == ("up" if D > 0 else "down"):
        hits.append("CONFLICT_TREND_VS_EXTENSION")
    strong_structure = _f(row.get("structure_quality_score")) >= 65 or ctx["primary"].confidence >= 60
    if strong_structure and bool(row.get("volatility_shock")):
        hits.append("CONFLICT_STRUCTURE_VS_SHOCK")
    pd_, idr = ctx["primary"].direction, ctx["intermediate"].direction
    if pd_ != 0 and idr != 0 and pd_ != idr:
        hits.append("CONFLICT_HIERARCHY_DISAGREEMENT")
    ts = row.get("trend_state")
    if D > 0 and ts in ("bearish", "strong_bearish") or D < 0 and ts in ("bullish", "strong_bullish"):
        hits.append("CONFLICT_STRUCTURE_VS_TREND_ENGINE")
    if D > 0 and ctx["bear_sweep_score"] >= 50 or D < 0 and ctx["bull_sweep_score"] >= 50:
        hits.append("CONFLICT_OPPOSING_SWEEP")
    bc = ctx["break_ctx"]
    if bc is not None and D != 0 and bc.state in ("CANDIDATE", "CONFIRMED", "ACCEPTED") and \
            (bc.direction == "bullish") == (D > 0) and bc.false_break_risk_score >= cfg.breakout_context.high_false_break_risk:
        hits.append("CONFLICT_FALSE_BREAK_RISK")
    prod = 1.0
    for h in hits:
        prod *= 1.0 - CONFLICT_WEIGHTS[h]
    score = round(100.0 * (1.0 - prod), 2)
    return score, hits, {h: CONFLICT_WEIGHTS[h] for h in hits}


# ---------------------------------------------------------------------------
def quality(ctx: dict, cfg: H4Config) -> tuple[float, dict]:
    row, sc = ctx["row"], cfg.context_scoring
    p, i = ctx["primary"], ctx["intermediate"]
    agree = 0.6 if (p.direction and i.direction and p.direction != i.direction) else 1.0
    fam = {}
    fam["STRUCTURE"] = (0.6 * (p.confidence + i.confidence) / 2.0 * agree + 0.4 * _f(row.get("structure_quality_score")))
    fam["DISPLACEMENT"] = 50.0 + 0.5 * abs(ctx["bull_disp_recent"] - ctx["bear_disp_recent"])
    fam["MOMENTUM"] = 100.0 - ctx["deterioration_score"]
    vol = {"normal": 100.0, "low": 80.0, "high": 80.0, "very_low": 55.0, "extreme": 35.0}.get(row.get("volatility_regime"), 30.0)
    if bool(row.get("volatility_shock")):
        vol = min(vol, 100.0 - _f(row.get("shock_severity")))
    fam["VOLATILITY"] = vol
    fam["LOCATION"] = float(np.clip(100.0 - max(0.0, _f(row.get("extension_score"), 50.0) - 50.0) * 1.2, 0, 100))
    liq = 60.0
    bc = ctx["break_ctx"]
    if bc is not None:
        liq = {"ACCEPTING": 80.0, "REJECTING": 80.0, "PARTIALLY_ACCEPTING": 65.0, "UNRESOLVED": 45.0}[bc.acceptance_state]
    sw = ctx["latest_sweep"]
    if sw is not None:
        liq = min(liq, {"SWEEP_AND_REJECT": 80.0, "BREAK_AND_ACCEPT": 75.0, "UNRESOLVED": 45.0}.get(sw["state"], 60.0))
    fam["LIQUIDITY_CONTEXT"] = liq
    D = structural_direction(ctx)
    lr, sr = ctx["long_room"]["room_score"], ctx["short_room"]["room_score"]
    fam["ROOM_TO_MOVE"] = lr if D > 0 else (sr if D < 0 else max(lr, sr))
    fam["MARKET_QUALITY"] = 0.7 * (100.0 - _f(row.get("chop_score"), 50.0)) + 0.3 * _f(row.get("directional_efficiency_percentile"), 50.0)
    weights = {"STRUCTURE": sc.q_structure, "DISPLACEMENT": sc.q_displacement, "MOMENTUM": sc.q_momentum,
               "VOLATILITY": sc.q_volatility, "LOCATION": sc.q_location, "LIQUIDITY_CONTEXT": sc.q_liquidity,
               "ROOM_TO_MOVE": sc.q_room, "MARKET_QUALITY": sc.q_market_quality}
    fam = {k: round(float(np.clip(v, 0, 100)), 2) for k, v in fam.items()}
    score = round(sum(fam[k] * w for k, w in weights.items()) / sum(weights.values()), 2)
    return score, fam


# ---------------------------------------------------------------------------
def directional_score(side: str, ctx: dict, conflict_score: float, cfg: H4Config) -> tuple[float, dict]:
    row, sc = ctx["row"], cfg.context_scoring
    d = 1 if side == "long" else -1
    want = "BULLISH" if d > 0 else "BEARISH"

    def layer(lr) -> float:
        return lr.confidence / 100.0 if lr.classification == want else 0.0

    fam = {}
    fam["structure"] = 0.5 * layer(ctx["primary"]) + 0.3 * layer(ctx["intermediate"]) + 0.2 * layer(ctx["immediate"])
    cont = 0.0
    D = structural_direction(ctx)
    if D == d and ctx["correction_active"]:
        cont = {"SHALLOW": 0.8, "NORMAL": 0.8, "DEEP": 0.3, "VERY_DEEP": 0.0}.get(ctx["retracement"]["retracement_band"], 0.0)
        if ctx["correction_class"] == "CHOPPY_CORRECTION":
            cont = min(cont, 0.4)
    bc = ctx["break_ctx"]
    if bc is not None and (bc.direction == "bullish") == (d > 0):
        cont = max(cont, {"ACCEPTING": 1.0, "PARTIALLY_ACCEPTING": 0.6}.get(bc.acceptance_state, 0.0))
    fam["continuation"] = cont
    fam["displacement"] = (ctx["bull_disp_recent"] if d > 0 else ctx["bear_disp_recent"]) / 100.0
    lvl = 0.0
    near, strong = cfg.levels.near_level_atr, cfg.levels.strong_level_score
    loc = _f(row.get("close_location"), 0.5)
    if d > 0 and _f(row.get("distance_to_support_atr"), 99) <= near and _f(row.get("support_strength")) >= strong and loc >= 0.5:
        lvl = 0.8
    if d < 0 and _f(row.get("distance_to_resistance_atr"), 99) <= near and _f(row.get("resistance_strength")) >= strong and loc <= 0.5:
        lvl = 0.8
    flip = ctx["recent_flip"]
    if flip is not None and flip["type"] == ("RESISTANCE_TO_SUPPORT" if d > 0 else "SUPPORT_TO_RESISTANCE"):
        lvl = max(lvl, flip["confidence"] / 100.0)
    if ctx["origin_reaction"] == ("demand" if d > 0 else "supply"):
        lvl = max(lvl, 0.7)
    fam["level_behaviour"] = lvl
    fam["liquidity"] = (ctx["bull_sweep_score"] if d > 0 else ctx["bear_sweep_score"]) / 100.0
    room = ctx["long_room"] if d > 0 else ctx["short_room"]
    fam["room"] = room["room_score"] / 100.0
    weights = {"structure": sc.d_structure, "continuation": sc.d_continuation, "displacement": sc.d_displacement,
               "level_behaviour": sc.d_level_behaviour, "liquidity": sc.d_liquidity, "room": sc.d_room}
    raw = 100.0 * sum(fam[k] * w for k, w in weights.items()) / sum(weights.values())

    pen = {}
    if room["room_score"] < cfg.room.sufficient_room_score:
        pen["insufficient_room"] = sc.penalty_opposing_level
    if row.get("extension_state") == "extremely_extended" and row.get("extension_direction") == ("up" if d > 0 else "down"):
        pen["extreme_extension"] = sc.penalty_extension
    if (ctx["failed_bullish_recent"] if d > 0 else ctx["failed_bearish_recent"]):
        pen["recent_failed_break"] = sc.penalty_failed_break
    against = "BEARISH" if d > 0 else "BULLISH"
    if ctx["primary"].broken_from == want or ctx["intermediate"].classification == against:
        pen["transition_against"] = sc.penalty_transition
    if row.get("market_quality") == "severe_chop":
        pen["severe_chop"] = sc.penalty_severe_chop
    if conflict_score > 0:
        pen["context_conflict"] = sc.penalty_conflict * conflict_score / 100.0
    if row.get("volatility_regime") == "extreme":
        pen["extreme_volatility"] = 1.0 - cfg.bias.extreme_vol_factor
    if bool(row.get("volatility_shock")):
        pen["volatility_shock"] = 1.0 - cfg.bias.shock_factor
    mult = 1.0
    for v in pen.values():
        mult *= 1.0 - v
    score = round(float(np.clip(raw * mult, 0, 100)), 2)
    return score, {"families": {k: round(v, 4) for k, v in fam.items()}, "raw": round(raw, 2),
                   "penalties": {k: round(v, 4) for k, v in pen.items()}, "multiplier": round(mult, 4)}


# ---------------------------------------------------------------------------
def hard_blockers(ctx: dict, cfg: H4Config) -> list[str]:
    row, b = ctx["row"], cfg.blockers
    out = []
    if b.block_invalid_data and row.get("data_quality_status") == "ERROR":
        out.append("INVALID_DATA")
    if ctx.get("stale"):
        out.append("STALE_DATA")
    if b.block_insufficient_history and not bool(row.get("warmup_complete")):
        out.append("INSUFFICIENT_HISTORY")
    if b.gap_block_bars and ctx["bars_since_gap"] is not None and ctx["bars_since_gap"] < b.gap_block_bars:
        out.append("UNRESOLVED_DATA_GAP")
    if _f(row.get("shock_severity")) >= b.shock_block_severity:
        out.append("EXTREME_VOLATILITY_SHOCK")
    if b.block_severe_chop and row.get("market_quality") == "severe_chop":
        out.append("SEVERE_CHOP")
    if b.block_unclassifiable_structure and ctx["primary"].classification == "UNCLEAR" and \
            ctx["intermediate"].classification == "UNCLEAR":
        out.append("UNCLASSIFIABLE_STRUCTURE")
    return out


def decide(ctx: dict, long_score: float, short_score: float, conflict_score: float, quality_score: float,
           blockers: list[str], cfg: H4Config) -> dict[str, Any]:
    pc = cfg.permission
    regime = ctx["row"].get("regime")
    thr = pc.transition_min_context_score if regime in ("TRANSITION", "UNCLEAR") else pc.min_context_score
    lr, sr = ctx["long_room"]["room_score"], ctx["short_room"]["room_score"]
    req = {
        "context_score_min": thr, "room_score_min": pc.min_room_score,
        "conflict_max": pc.max_conflict, "quality_min": pc.min_quality,
    }
    failed = {"long": [], "short": []}
    for side, score, room in (("long", long_score, lr), ("short", short_score, sr)):
        if score < thr:
            failed[side].append(f"{side}_context_score {score:.1f} < {thr:.1f}")
        if room < pc.min_room_score:
            failed[side].append(f"{side}_room_score {room:.1f} < {pc.min_room_score:.1f}")
    common = []
    if conflict_score > pc.max_conflict:
        common.append(f"context_conflict_score {conflict_score:.1f} > {pc.max_conflict:.1f}")
    if quality_score < pc.min_quality:
        common.append(f"context_quality_score {quality_score:.1f} < {pc.min_quality:.1f}")
    codes = []
    if blockers:
        decision = "BLOCK_ALL"
        codes = ["ALL_DIRECTIONS_BLOCKED"] + [f"BLOCKER_{b}" for b in blockers]
    else:
        long_ok = not failed["long"] and not common
        short_ok = not failed["short"] and not common
        if long_ok and short_ok:
            if abs(long_score - short_score) >= pc.both_separation:
                decision = "ALLOW_LONG" if long_score > short_score else "ALLOW_SHORT"
            elif regime in pc.allow_both_regimes:
                decision = "ALLOW_BOTH"
            else:
                decision = "BLOCK_ALL"
                codes.append("TWO_WAY_CONTEXT_OUTSIDE_RANGE")
        elif long_ok:
            decision = "ALLOW_LONG"
        elif short_ok:
            decision = "ALLOW_SHORT"
        else:
            decision = "BLOCK_ALL"
        if decision == "BLOCK_ALL":
            codes.insert(0, "ALL_DIRECTIONS_BLOCKED")
            if any(x.startswith("context_conflict") for x in common):
                codes.append("CONTEXT_CONFLICT_TOO_HIGH")
            if any(x.startswith("context_quality") for x in common):
                codes.append("CONTEXT_QUALITY_TOO_LOW")
            if failed["long"]:
                codes.append("LONG_CONTEXT_INSUFFICIENT")
            if failed["short"]:
                codes.append("SHORT_CONTEXT_INSUFFICIENT")
        else:
            codes.append({"ALLOW_LONG": "LONG_PERMISSION_GRANTED", "ALLOW_SHORT": "SHORT_PERMISSION_GRANTED",
                          "ALLOW_BOTH": "BOTH_DIRECTIONS_ALLOWED"}[decision])

    def conf(score: float, room: float) -> float:
        parts = [score / 100.0, quality_score / 100.0, 1.0 - conflict_score / 100.0, max(room, 1.0) / 100.0]
        return round(100.0 * float(np.prod(parts)) ** 0.25, 2)

    if decision == "ALLOW_LONG":
        confidence = conf(long_score, lr)
    elif decision == "ALLOW_SHORT":
        confidence = conf(short_score, sr)
    elif decision == "ALLOW_BOTH":
        confidence = min(conf(long_score, lr), conf(short_score, sr))
    else:
        confidence = 0.0
    return {"directional_permission": decision, "permission_confidence": confidence, "permission_codes": codes,
            "requirements": req, "failed_requirements": {"long": failed["long"] + common, "short": failed["short"] + common},
            "regime": regime}
