"""Execution-time conditions for Phase 1D: spread, bid/ask executable
reference, gap, price deterioration, abnormal movement, stacked barriers,
session context and news status.

Honesty rules
-------------
* The SIGNAL price is the confirmation bar's close on the chart basis.  It is
  a reference, not a fill: nobody can trade at a candle close after seeing it.
* The EXECUTABLE reference is the first price logically available AFTER
  confirmation (historically: the next H1 open), moved to the side of the book
  the trade would use - LONG buys the ASK, SHORT sells the BID.
* The spread used is the spread KNOWN at that moment (the report of the bar
  that just closed).  Missing spread is UNKNOWN - it is never treated as zero;
  the reference then uses an explicitly flagged conservative assumption.
* Slippage is reported as UNKNOWN unless a researcher supplies a model.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from ..features.context import _session_window_utc
from ..features.indicators import scale01
from .interfaces import IMPORTANCE, SlippageContext

SPREAD_LEVELS = ("NORMAL", "ELEVATED", "HIGH", "EXTREME")


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------
def to_pips(spread, ecfg, pip_size: float) -> float | None:
    if spread is None or not np.isfinite(spread):
        return None
    if ecfg.spread_unit == "pips":
        return float(spread)
    if ecfg.spread_unit == "points":
        return float(spread) / ecfg.points_per_pip
    return float(spread) / pip_size


def _level(v: float, a: float, b: float, c: float) -> int:
    return 0 if v <= a else (1 if v <= b else (2 if v <= c else 3))


def assess_spread(spread_pips: float | None, history_pips: np.ndarray, atr: float, pip_size: float, cfg) -> dict:
    """Spread status (NORMAL/ELEVATED/HIGH/EXTREME/UNKNOWN) = worst of absolute, ATR-relative and distribution tests."""
    if spread_pips is None:
        return {"spread_status": "UNKNOWN", "spread_pips": None, "spread_quality_score": None, "spread_atr_fraction": None,
                "spread_percentile": None, "spread_median_ratio": None, "tests": {}, "blocked": cfg.block_unknown,
                "note": "spread unavailable - NOT assumed to be zero"}
    hist = history_pips[np.isfinite(history_pips)]
    frac = spread_pips * pip_size / atr if np.isfinite(atr) and atr > 0 else None
    tests = {"absolute": _level(spread_pips, cfg.normal_max_pips, cfg.elevated_max_pips, cfg.high_max_pips)}
    if frac is not None:
        tests["atr_relative"] = _level(frac, cfg.normal_max_atr_fraction, cfg.elevated_max_atr_fraction, cfg.high_max_atr_fraction)
    ratio = pct = None
    if len(hist) >= cfg.history_min:
        med = float(np.median(hist))
        ratio = spread_pips / med if med > 0 else None
        pct = float(np.mean(hist <= spread_pips) * 100.0)
        if ratio is not None:
            tests["distribution"] = _level(ratio, cfg.normal_max_ratio, cfg.elevated_max_ratio, cfg.high_max_ratio)
    status = SPREAD_LEVELS[max(tests.values())]
    sev = max(float(scale01(spread_pips, 1.0, cfg.high_max_pips)),
              float(scale01(frac, 0.03, cfg.high_max_atr_fraction)) if frac is not None else 0.0,
              float(scale01(ratio, 1.0, cfg.high_max_ratio)) if ratio is not None else 0.0)
    return {"spread_status": status, "spread_pips": round(spread_pips, 3), "spread_quality_score": round(100.0 * (1 - sev), 2),
            "spread_atr_fraction": round(frac, 5) if frac is not None else None,
            "spread_percentile": round(pct, 2) if pct is not None else None,
            "spread_median_ratio": round(ratio, 4) if ratio is not None else None,
            "tests": {k: SPREAD_LEVELS[v] for k, v in tests.items()}, "blocked": status in cfg.block_statuses, "note": None}


def spread_code(status: str) -> str:
    return {"NORMAL": "SPREAD_ACCEPTABLE", "ELEVATED": "SPREAD_ELEVATED", "HIGH": "SPREAD_TOO_HIGH",
            "EXTREME": "SPREAD_TOO_HIGH", "UNKNOWN": "SPREAD_UNKNOWN"}[status]


# ---------------------------------------------------------------------------
# Bid / ask executable reference
# ---------------------------------------------------------------------------
def executable_reference(direction: int, chart_price: float, spread_pips: float | None, ecfg, pip_size: float) -> dict:
    """LONG executes on the ASK, SHORT on the BID.  ``chart_price`` is on ``ecfg.price_basis``."""
    assumed = spread_pips is None
    sp = (ecfg.unknown_spread_assumption_pips if assumed else spread_pips) * pip_size
    basis = ecfg.price_basis
    if basis == "bid":
        bid, ask = chart_price, chart_price + sp
    elif basis == "ask":
        bid, ask = chart_price - sp, chart_price
    else:
        bid, ask = chart_price - sp / 2.0, chart_price + sp / 2.0
    price = ask if direction > 0 else bid
    return {"executable_reference_price": float(price), "execution_side": "ASK" if direction > 0 else "BID",
            "bid_reference": float(bid), "ask_reference": float(ask), "chart_price_basis": basis,
            "spread_used_pips": round(sp / pip_size, 3), "spread_assumed": assumed,
            "note": ("spread UNKNOWN: conservative assumed spread applied, flagged" if assumed else None)}


def slippage(model, direction: int, t, spread_pips, atr, pip_size) -> dict:
    est = model.estimate(SlippageContext("LONG" if direction > 0 else "SHORT", t, spread_pips,
                                         (atr / pip_size) if np.isfinite(atr) else None))
    return {"slippage_status": est.status, "slippage_estimate_pips": est.pips, "slippage_model": est.model,
            "note": est.note or ("actual live slippage is unknown" if est.status == "UNKNOWN" else "")}


# ---------------------------------------------------------------------------
# Gap and price deterioration
# ---------------------------------------------------------------------------
def gap_info(direction: int, prev_close: float, next_price: float, atr: float, gcfg) -> dict:
    gap = next_price - prev_close
    gap_atr = abs(gap) / atr if np.isfinite(atr) and atr > 0 else 0.0
    toward = direction * gap  # >0: price moved away in the trade direction (worse fill); <0: against the thesis
    if gap_atr < gcfg.small_gap_atr:
        cls = "NONE"
    elif gap_atr < gcfg.defer_gap_atr:
        cls = "SMALL"
    elif gap_atr < gcfg.reject_gap_atr:
        cls = "LARGE"
    else:
        cls = "EXTREME"
    return {"gap_price": round(float(gap), 5), "gap_atr": round(float(gap_atr), 4), "gap_class": cls,
            "gap_direction": "WITH_TRADE" if toward > 0 else ("AGAINST_THESIS" if toward < 0 else "NONE")}


def deterioration(direction: int, signal_price: float, exec_ref: float, chart_next: float, atr: float, pip_size: float,
                  dcfg) -> dict:
    det = direction * (exec_ref - signal_price)
    det_atr = det / atr if np.isfinite(atr) and atr > 0 else 0.0
    move = direction * (chart_next - signal_price)
    sev = 100.0 * float(scale01(det_atr, dcfg.acceptable_atr * 0.5, dcfg.reject_atr))
    status = "ACCEPTABLE" if det_atr <= dcfg.acceptable_atr else ("DETERIORATING" if det_atr < dcfg.reject_atr else "EXCESSIVE")
    window = {"ACCEPTABLE": "AVAILABLE", "DETERIORATING": "DETERIORATING", "EXCESSIVE": "EXPIRED"}[status]
    return {"price_deterioration_pips": round(det / pip_size, 2) + 0.0, "price_deterioration_atr": round(float(det_atr), 4) + 0.0,
            "price_deterioration_score": round(sev, 2), "deterioration_status": status, "window_status": window,
            "components_pips": {"market_move": round(move / pip_size, 2),
                                "spread_and_basis": round((det - move) / pip_size, 2)},
            "note": "adverse = positive; score 0 = none, 100 = severe"}


# ---------------------------------------------------------------------------
# Abnormal movement
# ---------------------------------------------------------------------------
def abnormality(range_atr: float, atr_ratio: float, gap_atr: float, spread_ratio: float | None, shock: float, acfg) -> dict:
    comps = {
        "extreme_candle": float(scale01(range_atr if np.isfinite(range_atr) else 0.0, acfg.range_atr_start, acfg.range_atr_full)),
        "volatility_expansion": float(scale01(atr_ratio if np.isfinite(atr_ratio) else 1.0, acfg.atr_ratio_start, acfg.atr_ratio_full)),
        "gap": float(scale01(gap_atr, acfg.gap_atr_start, acfg.gap_atr_full)),
        "spread_explosion": float(scale01(spread_ratio, acfg.spread_ratio_start, acfg.spread_ratio_full)) if spread_ratio else 0.0,
        "price_discontinuity": float(np.clip((shock if np.isfinite(shock) else 0.0) / 100.0, 0, 1)),
    }
    score = 100.0 * max(comps.values())
    return {"execution_abnormality_score": round(score, 2), "components": {k: round(v, 4) for k, v in comps.items()},
            "drivers": [k for k, v in comps.items() if v >= 0.6],
            "status": "ABNORMAL" if score >= acfg.defer_score else "NORMAL"}


# ---------------------------------------------------------------------------
# Barrier stacking (H1 + H4, clustered)
# ---------------------------------------------------------------------------
def stacked_barriers(direction: int, ref: float, atr: float, h1_zones, h1_swings, h4_barriers, bcfg, room_cfg) -> dict:
    out = {"clusters": [], "remaining_room_score": 0.0, "barrier_density_score": 100.0, "nearest": None,
           "cluster_nearby": False}
    if not np.isfinite(atr) or atr <= 0:
        out["note"] = "ATR unavailable"
        return out
    raw = []
    for z in h1_zones:
        if z.strength < bcfg.min_zone_strength:
            continue
        if direction > 0 and z.upper > ref:
            raw.append(("H1", f"h1_{z.zone_type}_zone", max(z.lower, ref), float(z.strength)))
        elif direction < 0 and z.lower < ref:
            raw.append(("H1", f"h1_{z.zone_type}_zone", min(z.upper, ref), float(z.strength)))
    for s in h1_swings:
        if (direction > 0 and s.kind == "high" and s.price > ref) or (direction < 0 and s.kind == "low" and s.price < ref):
            sig = s.significance_atr if s.significance_atr is not None else 1.0
            raw.append(("H1", f"h1_swing_{s.kind}", float(s.price), round(100.0 * float(scale01(sig, 0.5, 3.0)), 2)))
    for b in h4_barriers or []:
        p = b.get("price")
        if p is None or not np.isfinite(p) or direction * (p - ref) < 0:
            continue
        raw.append(("H4", f"h4_{b.get('kind')}", float(p), float(b["strength"]) if b.get("strength") is not None else 50.0))
    raw.sort(key=lambda x: (direction * (x[2] - ref), x[1]))
    tol = bcfg.cluster_tolerance_atr * atr
    clusters = []
    for tf, kind, price, strength in raw:
        if clusters and abs(price - clusters[-1]["far_price"]) <= tol:
            cl = clusters[-1]
            cl["far_price"] = price
            cl["members"].append({"timeframe": tf, "type": kind, "price": round(price, 5), "strength": strength})
        else:
            clusters.append({"price": price, "far_price": price,
                             "members": [{"timeframe": tf, "type": kind, "price": round(price, 5), "strength": strength}]})
    res = []
    for cl in clusters[:bcfg.max_barriers]:
        dist = abs(cl["price"] - ref)
        res.append({"price": round(cl["price"], 5), "distance": round(dist, 5), "distance_atr": round(dist / atr, 4),
                    "strength": max(m["strength"] for m in cl["members"]),
                    "timeframes": sorted({m["timeframe"] for m in cl["members"]}),
                    "types": sorted({m["type"] for m in cl["members"]}), "n_members": len(cl["members"]),
                    "members": cl["members"]})
    total = 0.0
    for r in res:
        if r["distance_atr"] <= bcfg.density_window_atr:
            total += (0.5 + 0.5 * r["strength"] / 100.0) * (1.0 - r["distance_atr"] / bcfg.density_window_atr) \
                     * (1.0 + 0.25 * (len(r["timeframes"]) - 1))
    nearest = res[0] if res else None
    room_score = 100.0 if nearest is None else 100.0 * float(scale01(nearest["distance_atr"], room_cfg.min_room_atr,
                                                                     room_cfg.full_room_atr))
    return {"clusters": res, "nearest": nearest, "remaining_room_score": round(room_score, 2),
            "barrier_density_score": round(100.0 * min(1.0, total / bcfg.density_norm), 2),
            "cluster_nearby": any(r["distance_atr"] <= bcfg.nearby_atr and (r["n_members"] > 1 or len(r["timeframes"]) > 1)
                                  for r in res),
            "note": "NO_BARRIER_FOUND" if nearest is None else None}


# ---------------------------------------------------------------------------
# Session context at execution time (DST-aware via zoneinfo)
# ---------------------------------------------------------------------------
def session_context(t: pd.Timestamp, sessions) -> dict:
    tt = pd.Timestamp(t).tz_convert("UTC").to_pydatetime()
    active = []
    for name, tz, start, end in sessions:
        for k in (-1, 0, 1):
            day = datetime.combine((tt + pd.Timedelta(days=k).to_pytimedelta()).date(), datetime.min.time())
            s, e = _session_window_utc(tz, day, start, end)
            if s <= tt < e:
                active.append(name)
                break
    names = [s[0] for s in sessions]
    overlaps = [f"{a}_{b}_overlap" for i, a in enumerate(names) for b in names[i + 1:] if a in active and b in active]
    return {"time_utc": pd.Timestamp(t).isoformat(), "active_sessions": active,
            "session_overlap": overlaps, "session_label": "|".join(active) if active else "off_peak",
            "day_of_week": pd.Timestamp(t).day_name(), "hour_utc": int(pd.Timestamp(t).hour),
            "note": "descriptive only - no session is assumed to be profitable"}


# ---------------------------------------------------------------------------
# News status
# ---------------------------------------------------------------------------
def news_status(provider, t: pd.Timestamp, ncfg) -> dict:
    if provider is None:
        return {"news_status": "UNKNOWN", "provider": None, "events": [], "blocked": ncfg.block_unknown,
                "note": "no economic-calendar provider configured; no events are invented"}
    lo = t - pd.Timedelta(minutes=ncfg.post_event_minutes)
    hi = t + pd.Timedelta(minutes=ncfg.pre_event_minutes)
    evs = [e for e in provider.events(lo, hi, known_at=t)
           if (e.known_since is None or e.known_since <= t) and e.currency in (*ncfg.currencies, "GLOBAL")
           and IMPORTANCE.get(e.importance, 0) >= IMPORTANCE[ncfg.min_importance] and lo <= e.time <= hi]
    rows = [{"time": e.time.isoformat(), "currency": e.currency, "importance": e.importance, "name": e.name,
             "category": e.category, "minutes_until": round((e.time - t).total_seconds() / 60.0, 1)} for e in evs]
    status = "CLEAR"
    if any(r["minutes_until"] >= 0 for r in rows):
        status = "EVENT_IMMINENT"
    elif rows:
        status = "EVENT_RECENT"
    return {"news_status": status, "provider": type(provider).__name__, "events": rows,
            "blocked": ncfg.block_on_event and status != "CLEAR", "note": None}
