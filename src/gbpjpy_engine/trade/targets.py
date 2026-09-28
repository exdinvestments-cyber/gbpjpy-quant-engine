"""Structural targets, target ladder, barrier-path analysis, reachability and
target quality (Phase 1E).

Targets come ONLY from existing market structure known at proposal time
(H1/H4 swings, zones, H4 room barriers incl. supply/demand origin zones, the
H1 range extreme).  Overlapping H1/H4 levels are clustered and counted once.
A target is placed slightly BEFORE its structural level (front-run, plus the
spread component needed on the exit side of the book) - never beyond it.

Nothing here knows the stop, the risk distance or the minimum-R requirement,
so a target cannot be pushed farther to manufacture R:R (tested).  Primary
target selection uses structural legitimacy, reachability and the barrier
path only.
"""

from __future__ import annotations

import math

import numpy as np

from ..features.indicators import scale01
from .stops import trigger_spread_component

VOL_OK = {"normal": 1.0, "high": 0.8, "low": 0.7, "very_low": 0.5, "extreme": 0.3}


def cluster_levels(ctx, cfg) -> list[dict]:
    """Opposing structure beyond the entry, clustered (overlapping H1/H4 levels counted once), nearest first."""
    d, e, atr = ctx.direction, ctx.entry_price, ctx.atr
    lv = [x for x in ctx.levels if x.get("price") is not None and math.isfinite(x["price"]) and d * (x["price"] - e) >= 0]
    lv.sort(key=lambda x: (d * (x["price"] - e), x["type"]))
    tol = cfg.cluster_tolerance_atr * atr
    clusters = []
    for x in lv:
        if clusters and abs(x["price"] - clusters[-1]["far"]) <= tol:
            clusters[-1]["far"] = x["price"]
            clusters[-1]["members"].append(x)
        else:
            clusters.append({"near": x["price"], "far": x["price"], "members": [x]})
    out = []
    for cl in clusters:
        dist = d * (cl["near"] - e)
        out.append({"level": float(cl["near"]), "distance": float(dist), "distance_atr": float(dist / atr),
                    "strength": float(max(m["strength"] for m in cl["members"])),
                    "timeframes": sorted({m["timeframe"] for m in cl["members"]}),
                    "types": sorted({m["type"] for m in cl["members"]}), "n_members": len(cl["members"]),
                    "members": [{k: m[k] for k in ("type", "timeframe", "price", "strength")} for m in cl["members"]]})
    return out


def path_analysis(clusters: list[dict], upto: float, cfg) -> dict:
    """Barriers strictly between the entry and a target level (distance < ``upto``)."""
    before = [c for c in clusters if c["distance"] < upto - 1e-12]
    tot = 0.0
    for c in before:
        if c["distance_atr"] <= cfg.density_window_atr:
            tot += (0.5 + 0.5 * c["strength"] / 100.0) * (1.0 - c["distance_atr"] / cfg.density_window_atr) \
                   * (1.0 + 0.25 * (len(c["timeframes"]) - 1))
    density = 100.0 * min(1.0, tot / cfg.density_norm)
    strong = [c for c in before if c["strength"] >= cfg.strong_barrier]
    nearest = before[0] if before else None
    return {"barrier_count": len(before), "strong_barrier_count": len(strong),
            "max_barrier_strength": max((c["strength"] for c in before), default=0.0),
            "barrier_density_score": round(density, 2),
            "nearest_barrier": {k: nearest[k] for k in ("level", "types", "timeframes", "strength")} if nearest else None,
            "distance_to_nearest_barrier_atr": round(nearest["distance_atr"], 4) if nearest else None,
            "congested": len(strong) >= cfg.congested_strong_barriers or density >= cfg.congested_density}


def reachability(ctx, distance_atr: float, cfg) -> dict:
    """Information available at entry only.  Barriers are scored separately (path quality) - not counted twice."""
    d = ctx.direction
    mom = ctx.momentum_net if ctx.momentum_net is not None and math.isfinite(ctx.momentum_net) else 0.0
    comps = {
        "distance": 1.0 - float(scale01(distance_atr, cfg.reach_full_atr, cfg.reach_zero_atr)),
        "trend_context": 0.5 * float(np.clip(ctx.h4_side_score / 100.0, 0, 1)) + 0.5 * ctx.h1_primary_alignment,
        "momentum": float(scale01(d * mom, -20.0, 40.0)),
        "volatility": VOL_OK.get(str(ctx.vol_regime_h1).lower(), 0.5),
    }
    w = {"distance": 0.45, "trend_context": 0.25, "momentum": 0.15, "volatility": 0.15}
    return {"target_reachability_score": round(100.0 * sum(comps[k] * wt for k, wt in w.items()), 2),
            "components": {k: round(v, 4) for k, v in comps.items()},
            "note": "structure/context known at entry only - never future price data"}


def build_ladder(ctx, cfg, spec) -> dict:
    """Structural target ladder T1..Tn with path, reachability and quality (no stop / R input)."""
    d, e, atr = ctx.direction, ctx.entry_price, ctx.atr
    clusters = cluster_levels(ctx, cfg)
    exit_spread = trigger_spread_component(d, spec.pips_to_price(ctx.spread_pips), ctx.price_basis, "target")
    ladder = []
    for cl in clusters:
        if cl["distance_atr"] < cfg.min_target_atr:
            continue
        price = spec.normalize_away(cl["level"] - d * (cfg.front_run_atr * atr + exit_spread), d, "target")
        if d * (price - e) <= 0:
            continue
        path = path_analysis(clusters, cl["distance"], cfg)
        reach = reachability(ctx, cl["distance_atr"], cfg)
        legit = min(1.0, cl["strength"] / 100.0 * (1.0 + 0.15 * (len(cl["timeframes"]) - 1)))
        path_q = 100.0 - path["barrier_density_score"]
        quality = 0.40 * 100.0 * legit + 0.35 * reach["target_reachability_score"] + 0.25 * path_q
        label = f"T{len(ladder) + 1}"
        ladder.append({
            "label": label, "price": price, "structural_level": cl["level"], "type": "|".join(cl["types"]),
            "types": cl["types"], "timeframes": cl["timeframes"], "timeframe": "+".join(cl["timeframes"]),
            "strength": cl["strength"], "n_members": cl["n_members"], "members": cl["members"],
            "distance_pips": round(spec.to_pips(d * (price - e)), 2), "distance_atr": round(d * (price - e) / atr, 4),
            "barriers_before_target": path["barrier_count"], "path": path,
            "target_reachability_score": reach["target_reachability_score"], "reachability_components": reach["components"],
            "structural_legitimacy": round(100.0 * legit, 2), "path_quality": round(path_q, 2),
            "target_quality_score": round(quality, 2),
            "realistic": reach["target_reachability_score"] >= cfg.min_reachability and cl["distance_atr"] <= cfg.reach_zero_atr,
            "placement": f"{cfg.front_run_atr} ATR before the structural level"
                         + (" plus exit-side spread" if exit_spread else ""),
        })
        if len(ladder) >= cfg.max_targets:
            break
    return {"clusters": clusters, "ladder": ladder}


def select_primary(ladder: list[dict]) -> tuple[dict | None, str]:
    """Highest target quality among realistic, uncongested targets (ties -> nearer).  R is NOT an input."""
    eligible = [t for t in ladder if t["realistic"] and not t["path"]["congested"]]
    if not eligible:
        if not ladder:
            return None, "NO_STRUCTURAL_TARGET"
        if all(not t["realistic"] for t in ladder):
            return None, "TARGET_UNREALISTIC"
        return None, "BARRIER_CONGESTION"
    best = max(eligible, key=lambda t: (t["target_quality_score"], -t["distance_atr"]))
    why = (f"{best['label']} chosen: highest target quality {best['target_quality_score']:.1f} among realistic, "
           f"uncongested structural targets (legitimacy {best['structural_legitimacy']:.0f}, reachability "
           f"{best['target_reachability_score']:.0f}, path quality {best['path_quality']:.0f}); R was not a selection input")
    return best, why
