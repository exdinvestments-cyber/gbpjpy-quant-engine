"""Entry timing and quality scores (Phase 1D).

All scores are descriptive, untested baselines.  None is a probability, none
was fitted to outcomes, and ``entry_quality_score`` must never be used to size
a position.
"""

from __future__ import annotations

import numpy as np

from ..features.indicators import scale01

CHASE_CLASSES = ("LOW", "MODERATE", "HIGH", "EXTREME")
QUALITY_KEYS = {"CONFIRMATION": "w_confirmation", "TIMING": "w_timing", "FRESHNESS": "w_freshness",
                "PRICE_QUALITY": "w_price_quality", "MARKET_QUALITY": "w_market_quality",
                "STRUCTURAL_INTEGRITY": "w_structural_integrity", "ROOM": "w_room",
                "EXECUTION_CONDITIONS": "w_execution_conditions", "CONFLICT": "w_conflict"}
CONFLICT_WEIGHTS = {
    "MOMENTUM_DETERIORATING": 0.25,
    "PRICE_EXTENDED": 0.25,
    "ROOM_COLLAPSED": 0.30,
    "POOR_SPREAD": 0.20,
    "ABNORMAL_VOLATILITY": 0.25,
    "H4_CONTEXT_WEAKENED": 0.20,
    "OPPOSING_BARRIER_CLUSTER": 0.20,
}


def _r(x, nd=4):
    return round(float(x), nd)


def freshness(bars_since_conf: int, bars_since_qual: int, move_conf_atr: float, move_qual_atr: float, hours_since_conf: float,
              fcfg) -> dict:
    comps = {
        "confirmation_age": float(scale01(bars_since_conf, 0.0, fcfg.stale_confirmation_bars)),
        "qualification_age": float(scale01(bars_since_qual, 1.0, fcfg.stale_qualification_bars)),
        "move_since_confirmation": float(scale01(move_conf_atr, 0.0, fcfg.stale_move_since_confirmation_atr)),
        "move_since_qualification": float(scale01(move_qual_atr, 0.5, fcfg.stale_move_since_qualification_atr)),
        "wall_clock_age": float(scale01(hours_since_conf, 1.0, fcfg.stale_hours)),
    }
    w = {"confirmation_age": 0.30, "qualification_age": 0.15, "move_since_confirmation": 0.20,
         "move_since_qualification": 0.15, "wall_clock_age": 0.20}
    score = 100.0 * (1.0 - sum(comps[k] * wt for k, wt in w.items()))
    if score >= fcfg.fresh_score:
        cls = "FRESH"
    elif score >= fcfg.aging_score:
        cls = "AGING"
    elif score >= fcfg.stale_score:
        cls = "STALE"
    else:
        cls = "EXPIRED"
    return {"signal_freshness_score": round(score, 2), "freshness_class": cls, "components": {k: _r(v) for k, v in comps.items()},
            "bars_since_confirmation": int(bars_since_conf), "bars_since_qualification": int(bars_since_qual),
            "move_since_confirmation_atr": _r(move_conf_atr), "move_since_qualification_atr": _r(move_qual_atr),
            "hours_since_confirmation": _r(hours_since_conf, 2)}


def chase_risk(d: int, ref: float, conf_ref: float | None, setup_location: float | None, impulse_atr: float | None,
               nearest_barrier_atr: float | None, atr: float, ccfg) -> dict:
    """Has price already travelled too far from where the setup/confirmation made sense?"""
    atr = atr if np.isfinite(atr) and atr > 0 else np.nan
    from_conf = d * (ref - conf_ref) / atr if conf_ref is not None and np.isfinite(atr) else 0.0
    from_setup = d * (ref - setup_location) / atr if setup_location is not None and np.isfinite(atr) else 0.0
    travelled = (from_setup / impulse_atr) if impulse_atr and impulse_atr > 0 else None
    comps = {
        "distance_from_confirmation_reference": float(scale01(from_conf, 0.5, 2.0)),
        "distance_from_setup_location": float(scale01(from_setup, 1.0, 3.5)),
        "recent_impulse_travelled": float(scale01(travelled, 0.5, 1.0)) if travelled is not None else 0.5,
        # distance to the opposing barrier cluster (room itself is scored separately - counted once here)
        "proximity_to_opposing_structure": (1.0 - float(scale01(nearest_barrier_atr, 0.5, 2.0)))
        if nearest_barrier_atr is not None else 0.0,
    }
    w = {"distance_from_confirmation_reference": 0.30, "distance_from_setup_location": 0.25,
         "recent_impulse_travelled": 0.25, "proximity_to_opposing_structure": 0.20}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    cls = "EXTREME" if score >= ccfg.extreme_score else ("HIGH" if score >= ccfg.high_score else
                                                          ("MODERATE" if score >= ccfg.moderate_score else "LOW"))
    return {"chase_risk_score": round(score, 2), "chase_class": cls, "components": {k: _r(v) for k, v in comps.items()},
            "distance_from_confirmation_reference_atr": _r(from_conf), "distance_from_setup_location_atr": _r(from_setup),
            "recent_impulse_travelled_pct": _r(100 * travelled, 2) if travelled is not None else None,
            "rejects": CHASE_CLASSES.index(cls) >= CHASE_CLASSES.index(ccfg.reject_at)}


def entry_extension(d: int, ref: float, ema_fast: float, structural_origin: float | None, impulse_atr_10: float,
                    ext_pct: float, ext_dir: str, h4_ext_score: float, h4_ext_dir: str, atr: float, xcfg) -> dict:
    atr = atr if np.isfinite(atr) and atr > 0 else np.nan
    same = "up" if d > 0 else "down"
    base = d * (ref - ema_fast) / atr if np.isfinite(ema_fast) and np.isfinite(atr) else 0.0
    orig = d * (ref - structural_origin) / atr if structural_origin is not None and np.isfinite(atr) else 0.0
    comps = {
        "distance_from_h1_baseline": float(scale01(base, 0.5, 2.5)),
        "distance_from_structural_origin": float(scale01(orig, 1.5, 5.0)),
        "recent_impulse_size": float(scale01(impulse_atr_10 if np.isfinite(impulse_atr_10) else 0.0, 2.0, 6.0)),
        "historical_percentile": float(scale01(ext_pct, 50.0, 95.0)) if (np.isfinite(ext_pct) and ext_dir == same) else 0.0,
        "h4_extension_context": float(scale01(h4_ext_score, 50.0, 90.0)) if (np.isfinite(h4_ext_score) and h4_ext_dir == same) else 0.0,
    }
    w = {"distance_from_h1_baseline": 0.30, "distance_from_structural_origin": 0.20, "recent_impulse_size": 0.20,
         "historical_percentile": 0.15, "h4_extension_context": 0.15}
    score = 100.0 * sum(comps[k] * wt for k, wt in w.items())
    state = "OVEREXTENDED" if score >= xcfg.overextended_score else ("EXTENDED" if score >= xcfg.extended_score else "NOT_EXTENDED")
    return {"entry_extension_score": round(score, 2), "extension_state": state,
            "components": {k: _r(v) for k, v in comps.items()}, "distance_from_baseline_atr": _r(base),
            "distance_from_structural_origin_atr": _r(orig),
            "note": "timing protection only - extension is not assumed to predict reversal"}


def entry_market_quality(chop: float, eff_pct: float, vol_regime: str, gap_atr: float) -> dict:
    vol_q = {"NORMAL": 100.0, "LOW": 75.0, "HIGH": 70.0, "VERY_LOW": 50.0, "EXTREME": 20.0}.get(str(vol_regime).upper(), 40.0)
    comps = {
        "low_chop": 100.0 - (chop if np.isfinite(chop) else 50.0),
        "directional_efficiency": eff_pct if np.isfinite(eff_pct) else 50.0,
        "volatility_suitability": vol_q,
        "gap_behaviour": 100.0 * (1.0 - float(scale01(gap_atr, 0.1, 1.0))),
    }
    w = {"low_chop": 0.35, "directional_efficiency": 0.25, "volatility_suitability": 0.25, "gap_behaviour": 0.15}
    return {"entry_market_quality_score": round(sum(comps[k] * wt for k, wt in w.items()), 2),
            "components": {k: round(v, 2) for k, v in comps.items()},
            "note": "execution-time conditions only (confirmation, spread and deterioration are separate families)"}


def entry_conflict(hits: list[str]) -> float:
    prod = 1.0
    for h in hits:
        prod *= 1.0 - CONFLICT_WEIGHTS[h]
    return round(100.0 * (1.0 - prod), 2)


def entry_quality(fams: dict, conflict: float, scfg) -> float:
    w = {k: getattr(scfg, v) for k, v in QUALITY_KEYS.items()}
    q = sum(float(np.clip(fams[k], 0, 100)) * wt for k, wt in w.items()) / sum(w.values())
    return round(float(np.clip(q * (1.0 - scfg.conflict_penalty * conflict / 100.0), 0, 100)), 2)
