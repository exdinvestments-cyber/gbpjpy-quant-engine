"""Trade construction: risk first, reward second (Phase 1E).

Order of work for one accepted entry candidate:

1. EVALUATING_STOP      - structural invalidation level (family policy), noise
                          buffer, sanity, volatility adequacy, noise risk,
                          stop quality.  No target or R information exists yet.
2. EVALUATING_TARGETS   - structural target ladder, barrier paths,
                          reachability, quality, primary selection.  No stop,
                          R or minimum-R information is used.
3. EVALUATING_RISK_REWARD - 1R = |entry - stop|; gross and cost-adjusted R per
                          target; minimum-asymmetry gate on the PRIMARY target.
                          If the natural structure does not provide enough
                          asymmetry the trade is REJECTED - the stop is never
                          moved closer and the target never moved farther.
4. PROPOSED / REJECTED  (INVALIDATED / EXPIRED when the thesis or candidate
                          fails first).

A PROPOSED TRADE IS NOT AN ORDER.  There is no volume, lot size, currency
risk, percentage risk or leverage anywhere in this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from ..features.indicators import scale01
from ..snapshot import to_jsonable
from .interfaces import UNKNOWN_CONSTRAINTS, UnknownCosts, default_management
from .stops import adequacy, buffer_stop, noise_risk, sanity, select_reference, stop_quality
from .targets import build_ladder, select_primary

TRADE_STATES = ("NO_TRADE_CONSTRUCTION", "EVALUATING_STOP", "EVALUATING_TARGETS", "EVALUATING_RISK_REWARD",
                "PROPOSED", "REJECTED", "INVALIDATED", "EXPIRED")
CONFLICT_WEIGHTS = {
    "EXCELLENT_ENTRY_HUGE_STOP": 0.30,
    "GOOD_RR_UNREALISTIC_TARGET": 0.30,
    "TARGET_DENSE_BARRIERS": 0.25,
    "STOP_IN_ORDINARY_NOISE": 0.30,
    "COSTS_DESTROY_ASYMMETRY": 0.30,
    "H4_CONTEXT_WEAKENING": 0.20,
}
STOP_CODE = {"STRUCTURAL_INVALIDATION": "STRUCTURAL_STOP_VALID", "SWING_INVALIDATION": "SWING_STOP_VALID",
             "ZONE_INVALIDATION": "ZONE_STOP_VALID", "RECLAIM_FAILURE": "RECLAIM_FAILURE_STOP_VALID",
             "BREAK_RETEST_FAILURE": "BREAK_RETEST_STOP_VALID",
             "VOLATILITY_ADJUSTED_STRUCTURE": "VOLATILITY_ADJUSTED_STOP_VALID"}


@dataclass
class TradeContext:
    """Everything trade construction may know - all of it available at the proposal time."""

    direction: int  # +1 LONG, -1 SHORT
    setup_family: str
    confirmation_family: str
    entry_price: float  # Phase 1D executable reference (ASK for long, BID for short)
    price_basis: str  # chart basis of structural levels: bid | mid | ask
    spread_pips: float  # spread known at the decision (or the flagged Phase 1D assumption)
    spread_assumed: bool
    atr: float
    pip_size: float
    last_close: float
    stop_refs: dict  # stop type -> {"price", "reason"} (None when unavailable)
    structural_levels: list  # all structural invalidation candidates (for volatility adjustment - widening only)
    levels: list  # opposing structure (target / barrier candidates): {price, type, timeframe, strength}
    adverse_wicks: list  # recent adverse wick lengths (price)
    adverse_excursions: list  # recent 5-bar adverse excursions from a close (price)
    ranges5: list  # recent 5-bar ranges (price) for the stop-distance percentile
    recent_extremes: list  # recent adverse extremes (lows for long) to count level interactions
    chop: float
    momentum_net: float | None
    h4_side_score: float
    h4_side_score_at_qualification: float
    h4_permission_confidence: float
    h1_primary_alignment: float  # 1 aligned / 0.5 neutral / 0 opposed
    vol_regime_h1: str
    vol_regime_h4: str | None
    entry_quality: float
    setup_score: float | None
    slippage_pips: float | None  # per fill; None = unknown
    slippage_status: str
    meta: dict = field(default_factory=dict)  # ids, timestamps, session, news, permission ... (audit only)


@dataclass
class TradeConstruction:
    trade_proposal_id: str
    entry_candidate_id: str
    setup_id: str
    direction: str
    index: int
    at: str
    state: str = "NO_TRADE_CONSTRUCTION"
    transitions: list = field(default_factory=list)  # (index, at, from, to, reason)
    decision: str | None = None
    category: str | None = None
    reason: str | None = None
    reason_codes: list = field(default_factory=list)
    proposal: dict | None = None  # the full proposed trade (also filled for rejections: research record)
    end_index: int | None = None

    def move(self, i, at, to, reason):
        self.transitions.append((i, at, self.state, to, reason))
        self.state = to

    def state_at(self, i: int) -> str:
        st = "NO_TRADE_CONSTRUCTION"
        for idx, _, _, to, _ in self.transitions:
            if idx <= i:
                st = to
        return st

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transitions"] = [{"index": t[0], "at": t[1], "from": t[2], "to": t[3], "reason": t[4]} for t in self.transitions]
        return d


def _finite(*xs) -> bool:
    return all(x is not None and math.isfinite(float(x)) for x in xs)


def cost_model(ctx, cfg, cost_est) -> dict:
    """Cost components in pips.  Unknown components use flagged CONSERVATIVE assumptions - never zero."""
    c = cfg.cost
    slip_known = ctx.slippage_status == "MODELLED" and ctx.slippage_pips is not None
    slip = float(ctx.slippage_pips) if slip_known else c.assumed_slippage_pips_per_side
    com_known = cost_est.status == "KNOWN" and cost_est.commission_pips_round_turn is not None
    com = float(cost_est.commission_pips_round_turn) if com_known else c.assumed_commission_pips_round_turn
    comps = {
        "spread": {"status": "ASSUMED" if ctx.spread_assumed else "KNOWN", "pips": round(ctx.spread_pips, 3),
                   "treatment": "embedded in the entry/stop/target prices (bid/ask-correct), at the spread known at "
                                "the decision; the spread at exit is unknown"},
        "slippage_entry": {"status": "KNOWN" if slip_known else "UNKNOWN_ASSUMED", "pips": slip},
        "slippage_exit": {"status": "KNOWN" if slip_known else "UNKNOWN_ASSUMED", "pips": slip},
        "commission_round_turn": {"status": "KNOWN" if com_known else "UNKNOWN_ASSUMED", "pips": com},
        "swap_financing": {"status": "UNKNOWN", "pips": None, "treatment": "not modelled: holding period unknown"},
    }
    known = not ctx.spread_assumed and slip_known and com_known
    return {"components": comps, "total_extra_pips": 2 * slip + com, "status": "KNOWN" if known else "PARTIALLY_UNKNOWN",
            "note": "estimated_net_R uses these conservative assumptions for unknown costs; swap is excluded and unknown"}


def r_values(d, entry, stop, target, costs_pips, spec) -> dict:
    risk = d * (entry - stop)
    reward = d * (target - entry)
    if not _finite(risk, reward) or risk <= 0:
        return {"valid": False}
    extra = spec.pips_to_price(costs_pips)
    gross = reward / risk
    net = (reward - extra) / (risk + extra)
    return {"valid": True, "gross_R": round(gross, 4), "estimated_net_R": round(net, 4),
            "risk_distance": risk, "reward_distance": reward}


def noisy_or(hits) -> float:
    prod = 1.0
    for h in hits:
        prod *= 1.0 - CONFLICT_WEIGHTS[h]
    return round(100.0 * (1.0 - prod), 2)


def construct(ctx: TradeContext, rec: TradeConstruction, cfg, spec, broker=None, costs=None, final_checks=None) -> TradeConstruction:
    broker = broker or UNKNOWN_CONSTRAINTS
    costs = costs or UnknownCosts()
    i, at, d = rec.index, rec.at, ctx.direction
    codes: list[str] = []
    rs = []  # research record pieces

    def reject(category, reason, extra_codes=(), decision="REJECT_TRADE", to="REJECTED", payload=None):
        rec.move(i, at, to, reason)
        rec.decision, rec.category, rec.reason, rec.end_index = decision, category, reason, i
        tail = {"REJECTED": "TRADE_REJECTED", "INVALIDATED": "TRADE_INVALIDATED", "EXPIRED": "TRADE_EXPIRED"}[to]
        rec.reason_codes = list(dict.fromkeys(codes + list(extra_codes) + [tail]))
        rec.proposal = to_jsonable(dict(payload, reason_codes=rec.reason_codes, decision=decision, rejection_category=category,
                                        rejection_reason=reason)) if payload is not None else None
        return rec

    # ---- numeric preconditions --------------------------------------------------
    if not _finite(ctx.entry_price, ctx.atr, ctx.spread_pips) or ctx.atr <= 0:
        rec.move(i, at, "EVALUATING_STOP", "numeric preconditions")
        return reject("NUMERIC_INVALID", "non-finite entry, ATR or spread", ["NUMERIC_INVALID"])

    # ---- 1. stop ------------------------------------------------------------------
    rec.move(i, at, "EVALUATING_STOP", "structural invalidation first (no target or R information yet)")
    ref = select_reference(ctx, cfg)
    if ref is None:
        return reject("NO_STRUCTURAL_STOP", "no structural invalidation reference on the correct side of the entry",
                      ["NO_STRUCTURAL_STOP"])
    buf = buffer_stop(ctx, ref, cfg, spec)
    stop = buf["proposed_stop_price"]
    codes += [STOP_CODE[ref["stop_reference_type"]], "STOP_BUFFER_APPLIED"]
    bad = sanity(ctx, stop, cfg, spec, broker)
    base = {"stop": {**ref, **buf}}
    if bad:
        return reject("INVALID_STOP", "stop sanity failed: " + ",".join(bad), ["INVALID_STOP_GEOMETRY"] + bad, payload=base)
    adeq = adequacy(ctx, stop, cfg, spec)
    noise = noise_risk(ctx, stop, buf["stop_buffer"], cfg)
    sq = stop_quality(ref, adeq, noise, cfg)
    stop_info = {**ref, **buf, **adeq, **noise, **sq, "stop_quality_components": sq["components"],
                 "noise_components": noise["components"],
                 "why_not_closer": (f"the thesis is only invalidated beyond {ref['stop_reference_price']:.3f} "
                                    f"({ref['stop_reference_type']}); a closer stop would sit inside the structure it must "
                                    f"protect, and the buffer ({buf['stop_buffer_pips']} pips) covers observed noise "
                                    "(wick quantile / ATR floor) plus the trigger-side spread")}
    base["stop"] = stop_info

    # ---- 2. targets -----------------------------------------------------------------
    rec.move(i, at, "EVALUATING_TARGETS", "structural targets from existing structure (no stop, R or minimum-R input)")
    lad = build_ladder(ctx, cfg.target, spec)
    ladder = lad["ladder"]
    primary, why = select_primary(ladder)
    codes += ["STRUCTURAL_TARGET_FOUND"] if ladder else []
    codes += ["H1_TARGET_FOUND"] if any("H1" in t["timeframes"] for t in ladder) else []
    codes += ["H4_TARGET_FOUND"] if any("H4" in t["timeframes"] for t in ladder) else []

    # ---- 3. risk / reward ------------------------------------------------------------
    rec.move(i, at, "EVALUATING_RISK_REWARD", "1R = |entry - stop|; asymmetry judged on the fixed geometry")
    cm = cost_model(ctx, cfg, costs.estimate(ctx.meta.get("symbol", "GBPJPY"), ctx.meta.get("timestamp"),
                                             "LONG" if d > 0 else "SHORT"))
    codes.append("COSTS_KNOWN" if cm["status"] == "KNOWN" else "COSTS_UNKNOWN")
    for t in ladder:
        r = r_values(d, ctx.entry_price, stop, t["price"], cm["total_extra_pips"], spec)
        t.update({"gross_R": r.get("gross_R"), "estimated_net_R": r.get("estimated_net_R"), "r_valid": r["valid"]})
    one_r = d * (ctx.entry_price - stop)
    min_r = cfg.min_net_r(ctx.setup_family)
    pr = next((t for t in ladder if primary is not None and t["label"] == primary["label"]), None)
    path = pr["path"] if pr else None
    # ---- conflict & quality (on the natural geometry) ---------------------------------
    hits = []
    if ctx.entry_quality >= 65 and adeq["volatility_adequacy"] in ("WIDE", "EXTREME"):
        hits.append("EXCELLENT_ENTRY_HUGE_STOP")
    if pr and pr["gross_R"] is not None and pr["gross_R"] >= min_r and pr["target_reachability_score"] < 50:
        hits.append("GOOD_RR_UNREALISTIC_TARGET")
    if path and path["barrier_density_score"] >= 50:
        hits.append("TARGET_DENSE_BARRIERS")
    if noise["stop_noise_risk_score"] >= cfg.stop.noise_warn_score:
        hits.append("STOP_IN_ORDINARY_NOISE")
    if pr and pr["gross_R"] is not None and pr["gross_R"] >= min_r > pr["estimated_net_R"]:
        hits.append("COSTS_DESTROY_ASYMMETRY")
    if ctx.h4_side_score < ctx.h4_side_score_at_qualification - 10:
        hits.append("H4_CONTEXT_WEAKENING")
    conflict = noisy_or(hits)
    fams = {
        "ENTRY_QUALITY": ctx.entry_quality, "STOP_QUALITY": sq["stop_quality_score"],
        "TARGET_QUALITY": (0.55 * pr["structural_legitimacy"] + 0.45 * pr["target_reachability_score"]) if pr else 0.0,
        "ASYMMETRY": 100.0 * float(scale01(pr["estimated_net_R"], 1.0, 3.0)) if pr and pr["r_valid"] else 0.0,
        "PATH_QUALITY": (100.0 - path["barrier_density_score"]) if path else 0.0,
        "MARKET_CONTEXT": 0.6 * ctx.h4_side_score + 0.4 * ctx.h4_permission_confidence,
    }
    fams = {k: round(float(np.clip(v, 0, 100)), 2) for k, v in fams.items()}
    sc = cfg.scoring
    w = {"ENTRY_QUALITY": sc.w_entry, "STOP_QUALITY": sc.w_stop, "TARGET_QUALITY": sc.w_target,
         "ASYMMETRY": sc.w_asymmetry, "PATH_QUALITY": sc.w_path, "MARKET_CONTEXT": sc.w_context}
    quality = sum(fams[k] * wt for k, wt in w.items()) / sum(w.values()) * (1.0 - sc.conflict_penalty * conflict / 100.0)
    quality = round(float(np.clip(quality, 0, 100)), 2)
    codes += [f"TRADE_CONFLICT_{h}" for h in hits]

    payload = {
        "trade_proposal_id": rec.trade_proposal_id, "entry_candidate_id": rec.entry_candidate_id, "setup_id": rec.setup_id,
        "timestamp": at, "symbol": spec.symbol, "direction": rec.direction, "setup_family": ctx.setup_family,
        "confirmation_family": ctx.confirmation_family,
        "executable_reference_price": ctx.entry_price,
        "stop_reference_type": ref["stop_reference_type"], "stop_reference_price": ref["stop_reference_price"],
        "structural_reason": ref["structural_reason"],
        "raw_invalidation_price": buf["raw_invalidation_price"], "stop_buffer": round(buf["stop_buffer"], 6),
        "stop_buffer_pips": buf["stop_buffer_pips"], "proposed_stop_price": stop,
        "stop_distance_pips": adeq["stop_distance_pips"], "stop_distance_atr": adeq["stop_distance_atr"],
        "stop_distance_percentile": adeq["stop_distance_percentile"], "volatility_adequacy": adeq["volatility_adequacy"],
        "stop_noise_risk_score": noise["stop_noise_risk_score"], "stop_quality_score": sq["stop_quality_score"],
        "risk_unit": {"one_R_price": round(one_r, 6), "one_R_pips": round(spec.to_pips(one_r), 2),
                      "definition": "1R = |executable entry - proposed stop| (account-independent)"},
        "candidate_targets": ladder,
        "primary_target": pr, "primary_target_reason": why,
        "gross_R": pr["gross_R"] if pr else None, "estimated_net_R": pr["estimated_net_R"] if pr else None,
        "minimum_net_R_required": min_r,
        "target_reachability_score": pr["target_reachability_score"] if pr else None,
        "barrier_density_score": path["barrier_density_score"] if path else None,
        "trade_construction_quality_score": quality, "trade_quality_families": fams,
        "trade_construction_conflict_score": conflict, "trade_conflict_hits": hits,
        "h4_permission": ctx.meta.get("h4_permission"), "setup_score": ctx.setup_score, "entry_quality_score": ctx.entry_quality,
        "cost_assumptions": cm, "broker_constraints_status": broker.status,
        "broker_constraints": {"min_stop_distance_points": broker.min_stop_distance_points,
                               "freeze_level_points": broker.freeze_level_points, "source": broker.source},
        "stop_detail": stop_info,
        "volatility": {"h1_regime": ctx.vol_regime_h1, "h4_regime": ctx.vol_regime_h4,
                       "h1_atr_price": ctx.atr, "h1_atr_pips": round(spec.to_pips(ctx.atr), 2),
                       "note": "stored for research; risk is not changed because volatility is high"},
        "session_context": ctx.meta.get("session_context"), "news_status": ctx.meta.get("news_status"),
        "management": default_management(cfg.management, pr["label"] if pr else None),
        "symbol_spec": spec.to_dict(),
        "not_an_order": "no volume, lot size, currency risk, percentage risk, leverage or order exists in Phase 1E",
    }
    if ctx.meta.get("news_status") in (None, "UNKNOWN"):
        codes.append("NEWS_UNKNOWN")

    # ---- decisions: natural geometry only; failure => REJECT (no re-search) ------------
    if adeq["volatility_adequacy"] == "EXTREME":
        return reject("STOP_TOO_WIDE", f"structural stop needs {adeq['stop_distance_atr']:.2f} ATR "
                                       f"({adeq['stop_distance_pips']} pips) - excessive for current GBPJPY volatility",
                      ["STOP_DISTANCE_EXCESSIVE"], payload=payload)
    if noise["stop_noise_risk_score"] >= cfg.stop.noise_reject_score:
        return reject("STOP_INSIDE_NOISE", f"stop sits inside ordinary noise (noise risk {noise['stop_noise_risk_score']:.0f})",
                      ["STOP_INSIDE_NOISE"], payload=payload)
    if pr is None:
        cat = {"NO_STRUCTURAL_TARGET": "NO_REALISTIC_TARGET", "TARGET_UNREALISTIC": "TARGET_UNREALISTIC",
               "BARRIER_CONGESTION": "BARRIER_CONGESTION"}[why]
        code = {"NO_REALISTIC_TARGET": "TARGET_UNREALISTIC", "TARGET_UNREALISTIC": "TARGET_UNREALISTIC",
                "BARRIER_CONGESTION": "TARGET_PATH_CONGESTED"}[cat]
        return reject(cat, {"NO_REALISTIC_TARGET": "no structural target beyond the entry",
                            "TARGET_UNREALISTIC": "no structural target is realistically reachable",
                            "BARRIER_CONGESTION": "every realistic target sits behind congested barriers"}[cat],
                      [code], payload=payload)
    codes.append("TARGET_PATH_CLEAR" if path["barrier_count"] == 0 else "TARGET_PATH_BARRIERS_PRESENT")
    if not pr["r_valid"]:
        return reject("NUMERIC_INVALID", "invalid risk/reward geometry", ["NUMERIC_INVALID"], payload=payload)
    if pr["estimated_net_R"] < min_r:
        if pr["gross_R"] >= min_r:
            return reject("EXECUTION_COSTS", f"gross R {pr['gross_R']:.2f} passes but estimated net R "
                                             f"{pr['estimated_net_R']:.2f} < {min_r} after costs",
                          ["COSTS_DEGRADE_RR", "ASYMMETRY_INSUFFICIENT"], payload=payload)
        return reject("INSUFFICIENT_RR", f"natural structure offers only {pr['estimated_net_R']:.2f} net R "
                                         f"(gross {pr['gross_R']:.2f}) < {min_r}; stop and target are not moved",
                      ["ASYMMETRY_INSUFFICIENT"], payload=payload)
    codes.append("ASYMMETRY_ACCEPTABLE")
    if conflict > sc.max_conflict:
        return reject("CONFLICT", f"trade construction conflict {conflict:.1f} > {sc.max_conflict}",
                      ["TRADE_CONFLICT_TOO_HIGH"], payload=payload)
    if quality < sc.min_quality:
        return reject("LOW_QUALITY", f"trade construction quality {quality:.1f} < {sc.min_quality}",
                      ["TRADE_QUALITY_TOO_LOW"], payload=payload)
    # ---- 4. thesis re-check immediately before finalising ------------------------------
    fc = dict(final_checks or {})
    fc["stop_thesis_relevant"] = d * (ctx.last_close - ref["stop_reference_price"]) > 0
    fc["target_available"] = d * (pr["price"] - ctx.entry_price) > 0
    payload["final_checks"] = fc
    failed = [k for k, v in fc.items() if not v]
    if failed:
        return reject("THESIS_FAILED", "thesis re-check failed before finalising: " + ",".join(failed),
                      ["THESIS_FAILED_BEFORE_ENTRY"], decision="INVALIDATE", to="INVALIDATED", payload=payload)
    rec.move(i, at, "PROPOSED", why)
    rec.decision, rec.category, rec.reason = "PROPOSE_TRADE", None, "all trade-construction checks passed"
    rec.reason_codes = list(dict.fromkeys(codes + ["TRADE_PROPOSED"]))
    payload["reason_codes"] = rec.reason_codes
    payload["decision"] = "PROPOSE_TRADE"
    rec.proposal = to_jsonable(payload)
    return rec
