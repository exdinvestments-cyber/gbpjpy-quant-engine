"""Stop placement = thesis invalidation (Phase 1E).

The stop answers ONE question: "where is the objective point at which this
trading thesis is wrong?"  It is chosen from structural references known at
proposal time, in a family-specific preference order, and then buffered
OUTWARD by observed noise.  Nothing in this module knows about targets,
reward or the minimum-R requirement, so the stop cannot be tightened to
manufacture R:R (tested).

Bid/ask: a LONG stop is a sell that triggers on the BID, a SHORT stop is a buy
that triggers on the ASK.  Structural levels come from chart prices
(``price_basis``), so the buffer includes the spread component needed to
express the level on the triggering side of the book.
"""

from __future__ import annotations

import math

import numpy as np

from ..features.indicators import scale01

RELEVANCE = {"STRUCTURAL_INVALIDATION": 1.0, "BREAK_RETEST_FAILURE": 1.0, "RECLAIM_FAILURE": 1.0,
             "SWING_INVALIDATION": 0.8, "ZONE_INVALIDATION": 0.7, "VOLATILITY_ADJUSTED_STRUCTURE": 0.7}
ADEQUACY_QUALITY = {"NORMAL": 1.0, "WIDE": 0.6, "TOO_TIGHT": 0.3, "EXTREME": 0.0}


def _finite(*xs) -> bool:
    return all(x is not None and isinstance(x, (int, float, np.floating)) and math.isfinite(float(x)) for x in xs)


def trigger_spread_component(direction: int, spread_price: float, basis: str, what: str) -> float:
    """Spread needed to express a chart-basis level on the side of the book that triggers the order.

    stop:   LONG sells on the BID, SHORT buys on the ASK.   target: LONG sells on the BID, SHORT buys on the ASK.
    Returns the distance to move the level AWAY from the entry for stops and TOWARD the entry for targets."""
    triggers_on = "bid" if direction > 0 else "ask"
    if basis == triggers_on:
        return 0.0
    if basis == "mid":
        return spread_price / 2.0
    return spread_price


def select_reference(ctx, cfg) -> dict | None:
    """First reference of the family policy that exists on the correct side of the entry; widened (never
    tightened) to the next structure when it sits inside ordinary noise."""
    d, e, atr = ctx.direction, ctx.entry_price, ctx.atr
    prefs = cfg.stop_policy(ctx.setup_family)
    chosen = None
    for rank, t in enumerate(prefs):
        ref = ctx.stop_refs.get(t)
        if ref is None or not _finite(ref.get("price")):
            continue
        if d * (e - ref["price"]) > 0:
            chosen = {"stop_reference_type": t, "stop_reference_price": float(ref["price"]),
                      "structural_reason": ref["reason"], "policy_rank": rank, "policy": list(prefs),
                      "volatility_adjusted_from": None}
            break
    if chosen is None:
        return None
    if _finite(atr) and atr > 0 and d * (e - chosen["stop_reference_price"]) < cfg.stop.min_structure_atr * atr:
        farther = sorted((x for x in ctx.structural_levels
                          if _finite(x["price"]) and d * (chosen["stop_reference_price"] - x["price"]) > 0
                          and d * (e - x["price"]) >= cfg.stop.min_structure_atr * atr),
                         key=lambda x: d * (e - x["price"]))
        if farther:
            nxt = farther[0]
            chosen = {**chosen, "stop_reference_type": "VOLATILITY_ADJUSTED_STRUCTURE",
                      "stop_reference_price": float(nxt["price"]),
                      "structural_reason": f"preferred {chosen['stop_reference_type']} at "
                                           f"{chosen['stop_reference_price']:.3f} lies inside ordinary noise "
                                           f"(< {cfg.stop.min_structure_atr} ATR); next structure beyond it: {nxt['reason']}",
                      "volatility_adjusted_from": chosen["stop_reference_type"]}
    return chosen


def buffer_stop(ctx, ref: dict, cfg, spec) -> dict:
    d, atr = ctx.direction, ctx.atr
    wicks = np.asarray(ctx.adverse_wicks, float)
    wicks = wicks[np.isfinite(wicks)]
    wick_q = float(np.quantile(wicks, cfg.stop.wick_quantile)) if len(wicks) else 0.0
    atr_buf = cfg.stop.buffer_atr * atr
    noise_buf = max(atr_buf, wick_q)
    spread_c = trigger_spread_component(d, spec.pips_to_price(ctx.spread_pips), ctx.price_basis, "stop")
    raw = ref["stop_reference_price"]
    stop = spec.normalize_away(raw - d * (noise_buf + spread_c), d, "stop")
    return {"raw_invalidation_price": raw, "stop_buffer": abs(stop - raw), "stop_buffer_pips": round(spec.to_pips(abs(stop - raw)), 2),
            "buffer_components": {"atr_floor": atr_buf, "wick_quantile": wick_q, "noise_buffer": noise_buf,
                                  "trigger_side_spread": spread_c, "spread_assumed": ctx.spread_assumed},
            "proposed_stop_price": stop}


def adequacy(ctx, stop: float, cfg, spec) -> dict:
    d, atr = ctx.direction, ctx.atr
    dist = d * (ctx.entry_price - stop)
    r5 = np.asarray(ctx.ranges5, float)
    r5 = r5[np.isfinite(r5)]
    pct = float(np.mean(r5 <= dist) * 100.0) if len(r5) else None
    m = dist / atr if atr > 0 else float("inf")
    s = cfg.stop
    cls = "TOO_TIGHT" if m < s.tight_atr else ("NORMAL" if m <= s.normal_atr else ("WIDE" if m <= s.wide_atr else "EXTREME"))
    return {"stop_distance_pips": round(spec.to_pips(dist), 2), "stop_distance_atr": round(m, 4),
            "stop_distance_percentile": round(pct, 2) if pct is not None else None, "volatility_adequacy": cls}


def noise_risk(ctx, stop: float, buffer_price: float, cfg) -> dict:
    """Does the stop sit inside ordinary observed noise?  (Not a prediction that it will be hit.)"""
    d, atr = ctx.direction, ctx.atr
    dist = d * (ctx.entry_price - stop)
    exc = np.asarray(ctx.adverse_excursions, float)
    exc = exc[np.isfinite(exc)]
    env = float(np.quantile(exc, 0.9)) if len(exc) else atr
    ext = np.asarray(ctx.recent_extremes, float)
    touches = int(np.sum(np.abs(ext[np.isfinite(ext)] - stop) <= 0.15 * atr)) if atr > 0 else 0
    spread = ctx.spread_pips * ctx.pip_size
    comps = {
        "inside_typical_excursion": 1.0 - float(scale01(dist / env if env > 0 else 9.0, 0.8, 2.0)),
        "small_vs_atr": 1.0 - float(scale01(dist / atr if atr > 0 else 0.0, 0.5, 1.5)),
        "local_chop": float(scale01(ctx.chop if _finite(ctx.chop) else 50.0, 40.0, 75.0)),
        "spread_share": float(scale01(spread / dist if dist > 0 else 1.0, 0.05, 0.25)),
        "level_interactions": float(scale01(touches, 1, 5)),
        "thin_buffer": 1.0 - float(scale01(buffer_price / atr if atr > 0 else 0.0, 0.05, 0.3)),
    }
    w = {"inside_typical_excursion": 0.30, "small_vs_atr": 0.20, "local_chop": 0.15, "spread_share": 0.10,
         "level_interactions": 0.15, "thin_buffer": 0.10}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    return {"stop_noise_risk_score": round(score, 2), "components": {k: round(v, 4) for k, v in comps.items()},
            "typical_adverse_excursion_q90": round(env, 5), "recent_level_touches": touches,
            "note": "measures whether the stop sits inside ordinary observed noise; not a hit probability"}


def stop_quality(ref: dict, adeq: dict, noise: dict, cfg) -> dict:
    fam_rel = 1.0 if ref["policy_rank"] == 0 and ref["volatility_adjusted_from"] is None else \
        (0.6 if ref["volatility_adjusted_from"] else 0.7)
    comps = {
        "structural_relevance": RELEVANCE[ref["stop_reference_type"]],
        "noise_clearance": 1.0 - noise["stop_noise_risk_score"] / 100.0,
        "volatility_appropriateness": ADEQUACY_QUALITY[adeq["volatility_adequacy"]],
        "distance": 1.0 - float(scale01(adeq["stop_distance_atr"], cfg.stop.normal_atr, cfg.stop.wide_atr + 1.0)),
        "setup_family_relevance": fam_rel,
    }
    w = {"structural_relevance": 0.30, "noise_clearance": 0.25, "volatility_appropriateness": 0.20, "distance": 0.10,
         "setup_family_relevance": 0.15}
    return {"stop_quality_score": round(100.0 * sum(comps[k] * wt for k, wt in w.items()), 2),
            "components": {k: round(v, 4) for k, v in comps.items()}}


def sanity(ctx, stop: float, cfg, spec, broker) -> list[str]:
    """Reasons the stop is impossible/nonsensical (empty list = sane)."""
    d = ctx.direction
    bad = []
    if not _finite(stop, ctx.entry_price, ctx.atr):
        return ["NON_FINITE_VALUE"]
    dist = d * (ctx.entry_price - stop)
    if dist == 0:
        return ["ZERO_RISK_DISTANCE"]
    if dist < 0:
        return ["STOP_ON_WRONG_SIDE_OF_ENTRY"]
    if spec.to_pips(dist) < cfg.stop.min_stop_pips:
        bad.append("STOP_BELOW_MINIMUM_DISTANCE")
    if broker.status == "KNOWN" and broker.min_stop_distance_points is not None and \
            spec.to_points(dist) < broker.min_stop_distance_points:
        bad.append("STOP_INSIDE_BROKER_STOP_LEVEL")
    if abs(spec.normalize(stop) - stop) > spec.point / 1000:
        bad.append("STOP_NOT_AT_SYMBOL_PRECISION")
    return bad
