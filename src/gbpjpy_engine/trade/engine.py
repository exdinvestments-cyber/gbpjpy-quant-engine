"""Phase 1E Trade Construction Engine (orchestration).

Consumes the Phase 1D ``EntryResult`` (with the Phase 1C setup result and the
H4 result underneath) and turns every ACCEPTED entry candidate - and only
those - into a PROPOSED TRADE or a retained rejection.  A proposed trade is
NOT an order: there is no volume, lot size, currency risk, percentage risk,
leverage or platform contact.

Timing: construction happens at the moment the entry candidate was accepted -
the first executable price after the close of bar ``p`` (index ``j = p + 1``,
phase OPEN).  It uses information known at the close of ``p`` plus the
Phase 1D executable reference - never later highs, lows, swings, volatility,
news or outcomes (tested in tests/test_trade_lookahead.py).  MAE/MFE can only
be computed afterwards by ``trade.outcomes`` (never imported here).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field

import numpy as np

from ..features.indicators import scale01
from ..snapshot import to_jsonable
from .config import TradeConfig
from .construct import TradeConstruction, TradeContext, construct
from .interfaces import UNKNOWN_CONSTRAINTS, UnknownCosts
from .symbol import GBPJPY, SymbolSpec


def _f(v, default=float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


@dataclass
class TradeResult:
    constructions: list = field(default_factory=list)  # every TradeConstruction (proposed or not)
    proposals: list = field(default_factory=list)  # proposed-trade objects
    rejected: list = field(default_factory=list)  # retained rejections / invalidations / expiries (research)
    config: TradeConfig | None = None
    symbol: SymbolSpec | None = None

    def construction(self, trade_proposal_id: str) -> TradeConstruction:
        return next(c for c in self.constructions if c.trade_proposal_id == trade_proposal_id)

    def direction_stats(self) -> dict:
        """Long and short trade-construction statistics, kept separate (no assumed symmetry)."""
        out = {}
        for side in ("LONG", "SHORT"):
            recs = [c for c in self.constructions if c.direction == side]
            props = [c.proposal for c in recs if c.decision == "PROPOSE_TRADE"]
            cats: dict = {}
            for c in recs:
                if c.decision != "PROPOSE_TRADE":
                    cats[c.category] = cats.get(c.category, 0) + 1

            def mean(key, rows):
                v = [r[key] for r in rows if r and r.get(key) is not None]
                return round(float(np.mean(v)), 4) if v else None

            with_geom = [c.proposal for c in recs if c.proposal]
            out[side] = {"constructions": len(recs), "proposed": len(props), "not_proposed_by_category": cats,
                         "mean_gross_R_proposed": mean("gross_R", props), "mean_net_R_proposed": mean("estimated_net_R", props),
                         "mean_stop_atr_all": mean("stop_distance_atr", with_geom),
                         "mean_stop_noise_all": mean("stop_noise_risk_score", with_geom)}
        return out


class TradeConstructionEngine:
    def __init__(self, config: TradeConfig | None = None, symbol: SymbolSpec | None = None):
        self.cfg = config or TradeConfig()
        self.cfg.validate()
        self.spec = symbol or GBPJPY
        self.spec.validate()

    # ------------------------------------------------------------------
    def run(self, entry_result, h1_result, h4_result, broker=None, costs=None) -> TradeResult:
        cfg, spec = self.cfg, self.spec
        if abs(spec.pip_size - h1_result.config.data.pip_size) > 1e-12:
            raise ValueError("symbol metadata pip size disagrees with the data configuration")
        costs = costs or UnknownCosts()
        feats, fr = h1_result.features, h1_result.frame
        a = {k: feats[k].to_numpy(float) for k in ("open", "high", "low", "close", "atr")}
        a["vol"] = feats["volatility_regime"].to_numpy(object)
        frame = {k: fr[k].to_numpy(object) for k in ("h4_index", "h4_permission", "h4_context_status", "h1_blockers",
                                                      "h1_primary_structure", "h1_chop_score", "momentum_net",
                                                      "h4_long_context_score", "h4_short_context_score",
                                                      "h4_permission_confidence") if k in fr.columns}
        swings = sorted(h1_result.structure.swings, key=lambda s: (s.accepted_index, s.swing_id))
        sw_idx = [s.accepted_index for s in swings]
        h4sw = sorted(h4_result.structure.swings, key=lambda s: (s.accepted_index, s.swing_id))
        h4sw_idx = [s.accepted_index for s in h4sw]
        setups = {s.setup_id: s for s in h1_result.setups}
        S = {"a": a, "fr": frame, "swings": swings, "sw_idx": sw_idx, "h4sw": h4sw, "h4sw_idx": h4sw_idx, "h1": h1_result,
             "h4": h4_result, "setups": setups}
        res = TradeResult(config=cfg, symbol=spec)
        accepted = sorted((c for c in entry_result.candidates if c.accepted is not None), key=lambda c: c.accepted_index)
        for cand in accepted:
            j, p = cand.accepted_index, cand.accepted_index - 1
            at = cand.accepted["timestamp"]
            rec = TradeConstruction(trade_proposal_id=f"P-{cand.entry_candidate_id}", entry_candidate_id=cand.entry_candidate_id,
                                    setup_id=cand.setup_id, direction="LONG" if cand.side == "long" else "SHORT", index=j, at=at)
            bad = self._revalidate(cand, j, p, S)
            if bad is not None:
                to, category, reason, code = bad
                rec.move(j, at, to, reason)
                rec.decision = "INVALIDATE" if to == "INVALIDATED" else "EXPIRE"
                rec.category, rec.reason, rec.end_index = category, reason, j
                rec.reason_codes = [code, "TRADE_INVALIDATED" if to == "INVALIDATED" else "TRADE_EXPIRED"]
            else:
                try:
                    ctx = self._context(cand, p, S)
                    broker_c = broker.stop_constraints(spec.symbol, cand.accepted["timestamp"]) if broker else UNKNOWN_CONSTRAINTS
                    final = {"entry_candidate_valid": True, "h4_permission_valid": True,
                             "room_acceptable": _f(cand.accepted.get("remaining_room_score"), 0.0) >= 30.0}
                    construct(ctx, rec, cfg, spec, broker_c, costs, final)
                except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as exc:  # numeric safety
                    rec.move(j, at, "REJECTED", f"numeric failure: {type(exc).__name__}: {exc}")
                    rec.decision, rec.category, rec.reason, rec.end_index = "REJECT_TRADE", "NUMERIC_INVALID", str(exc), j
                    rec.reason_codes = ["NUMERIC_INVALID", "TRADE_REJECTED"]
            if rec.state == "PROPOSED":
                self._follow(rec, cand)
                res.proposals.append(rec.proposal)
            else:
                res.rejected.append(to_jsonable({"trade_proposal_id": rec.trade_proposal_id, "entry_candidate_id":
                                                 rec.entry_candidate_id, "setup_id": rec.setup_id, "direction": rec.direction,
                                                 "index": rec.index, "at": rec.at, "decision": rec.decision,
                                                 "category": rec.category, "reason": rec.reason,
                                                 "reason_codes": rec.reason_codes, "research_record": rec.proposal,
                                                 "note": "retained so later research can test whether this filter adds value"}))
            res.constructions.append(rec)
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _entry_state_at_open(cand, j) -> str:
        st = "NO_ENTRY"
        for idx, phase, _, _, to, _ in cand.transitions:
            if idx < j or (idx == j and phase == "OPEN"):
                st = to
        return st

    def _revalidate(self, cand, j, p, S):
        st = self._entry_state_at_open(cand, j)
        if st != "ENTRY_CANDIDATE":
            return ("EXPIRED" if st == "EXPIRED" else "INVALIDATED", "ENTRY_NOT_VALID", f"entry candidate state {st}",
                    "ENTRY_CANDIDATE_NOT_VALID")
        if cand.accepted.get("freshness_class") not in ("FRESH", "AGING"):
            return "EXPIRED", "CANDIDATE_EXPIRED", "entry signal no longer fresh", "ENTRY_CANDIDATE_NOT_VALID"
        fr = S["fr"]
        perm, status = fr["h4_permission"][p], fr["h4_context_status"][p]
        allowed = ("ALLOW_LONG", "ALLOW_BOTH") if cand.side == "long" else ("ALLOW_SHORT", "ALLOW_BOTH")
        if status != "OK" or perm not in allowed:
            return "INVALIDATED", "H4_PERMISSION_CHANGED", f"H4 permission {perm} / context {status} at proposal time", \
                "H4_PERMISSION_REVOKED"
        if S["setups"][cand.setup_id].state_at(p) != "QUALIFIED":
            return "INVALIDATED", "SETUP_INVALIDATED", "Phase 1C setup no longer QUALIFIED", "SETUP_INVALIDATED"
        blockers = fr["h1_blockers"][p]
        if isinstance(blockers, (list, tuple)) and len(blockers):
            return "INVALIDATED", "MARKET_STATE", "H1 hard blocker: " + ",".join(blockers), "SETUP_INVALIDATED"
        return None

    def _follow(self, rec, cand) -> None:
        """After the proposal: it lapses/invalidates with its entry candidate (append-only)."""
        if cand.end_index is not None and cand.state in ("EXPIRED", "INVALIDATED"):
            t = cand.transitions[-1]
            rec.move(cand.end_index, t[2], cand.state, f"entry candidate {cand.state.lower()}: {cand.end_reason}")
            rec.end_index = cand.end_index

    # ------------------------------------------------------------------
    def _known_swings(self, S, p, h4=False):
        lst, idx = (S["h4sw"], S["h4sw_idx"]) if h4 else (S["swings"], S["sw_idx"])
        hi = bisect_right(idx, p)
        return [s for s in lst[max(0, hi - 64):hi] if s.removed_index is None or s.removed_index > p]

    def _context(self, cand, p, S) -> TradeContext:
        cfg, spec = self.cfg, self.spec
        a, fr, h1, h4 = S["a"], S["fr"], S["h1"], S["h4"]
        d = 1 if cand.side == "long" else -1
        acc = cand.accepted
        last = cand.opportunities[-1]["metrics"]
        entry = float(acc["executable_reference_price"])
        atr = float(a["atr"][p])
        stp = S["setups"][cand.setup_id]
        conf = cand.confirmation or {}
        ev = conf.get("evidence") or {}
        # ---- structural invalidation references (all known at the close of p) -----------
        refs = {}
        if stp.invalidation_reference is not None:
            refs["STRUCTURAL_INVALIDATION"] = {"price": float(stp.invalidation_reference),
                                               "reason": f"Phase 1C {stp.family} premise level (setup {stp.setup_id})"}
        h1sw = self._known_swings(S, p)[-h1.config.structure.swing_history_size:]
        kind = "low" if d > 0 else "high"
        sw = next((s for s in reversed(h1sw) if s.kind == kind and d * (entry - s.price) > 0), None)
        if sw is not None:
            refs["SWING_INVALIDATION"] = {"price": float(sw.price), "reason": f"most recent confirmed H1 swing {kind} "
                                                                              f"(swing {sw.swing_id}, {sw.label})"}
        zmin = cfg.stop.zone_min_strength
        zones = [z for z in h1.zones_by_bar[p] if z.strength >= zmin and (z.upper < entry if d > 0 else z.lower > entry)]
        if zones:
            z = max(zones, key=lambda z: z.upper) if d > 0 else min(zones, key=lambda z: z.lower)
            refs["ZONE_INVALIDATION"] = {"price": float(z.lower if d > 0 else z.upper),
                                         "reason": f"far edge of the nearest H1 {z.zone_type} zone (strength {z.strength:.0f})"}
        if conf.get("family") == "SWEEP_RECLAIM_CONFIRMATION" and ev.get("sweep_extreme") is not None:
            refs["RECLAIM_FAILURE"] = {"price": float(ev["sweep_extreme"]), "reason": "extreme of the swept liquidity (reclaim fails beyond it)"}
        elif stp.family == "LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION" and stp.invalidation_reference is not None:
            refs["RECLAIM_FAILURE"] = {"price": float(stp.invalidation_reference), "reason": "Phase 1C sweep extreme"}
        if conf.get("family") == "BREAK_RETEST_CONFIRMATION" and ev.get("retest_index") is not None:
            seg = slice(int(ev["retest_index"]), cand.confirmation_index + 1)
            x = float(np.min(a["low"][seg]) if d > 0 else np.max(a["high"][seg]))
            refs["BREAK_RETEST_FAILURE"] = {"price": x, "reason": f"retest extreme of the broken level {ev.get('break_level')}"}
        elif stp.family == "BREAK_RETEST_CONTINUATION" and stp.invalidation_reference is not None:
            refs["BREAK_RETEST_FAILURE"] = {"price": float(stp.invalidation_reference), "reason": "Phase 1C broken level minus tolerance"}
        structural = [{"price": v["price"], "reason": f"{k}: {v['reason']}"} for k, v in refs.items()]
        structural += [{"price": float(s.price), "reason": f"confirmed H1 swing {s.kind} {s.swing_id}"} for s in h1sw
                       if s.kind == kind]
        structural += [{"price": float(z.lower if d > 0 else z.upper), "reason": f"H1 {z.zone_type} zone edge"} for z in zones]
        # ---- opposing structure (target / barrier candidates) ------------------------------
        opp = "high" if d > 0 else "low"
        levels = []
        for s in h1sw:
            if s.kind == opp:
                sig = s.significance_atr if s.significance_atr is not None else 1.0
                levels.append({"price": float(s.price), "type": f"h1_swing_{opp}", "timeframe": "H1",
                               "strength": round(100.0 * float(scale01(sig, 0.5, 3.0)), 2)})
        tmin = cfg.target.zone_min_strength
        for z in h1.zones_by_bar[p]:
            if z.strength < tmin:
                continue
            if d > 0 and z.upper > entry and (z.lower > entry or z.zone_type == "resistance"):
                levels.append({"price": float(max(z.lower, entry)), "type": f"h1_{z.zone_type}_zone", "timeframe": "H1",
                               "strength": float(z.strength)})
            elif d < 0 and z.lower < entry and (z.upper < entry or z.zone_type == "support"):
                levels.append({"price": float(min(z.upper, entry)), "type": f"h1_{z.zone_type}_zone", "timeframe": "H1",
                               "strength": float(z.strength)})
        k = int(_f(fr["h4_index"][p], -1))
        if k >= 0:
            room = ((h4.context.details[k].get("room") or {}).get(cand.side) or {}) if h4.context else {}
            for b in room.get("barriers") or []:
                levels.append({"price": float(b["price"]), "type": f"h4_{b['kind']}", "timeframe": "H4",
                               "strength": float(b["strength"]) if b.get("strength") is not None else 50.0})
            for s in self._known_swings(S, k, h4=True)[-h4.config.structure.swing_history_size:]:
                if s.kind == opp:
                    levels.append({"price": float(s.price), "type": f"h4_swing_{opp}", "timeframe": "H4", "strength": 60.0})
        rx = h1.config.levels.range_extreme_lookback
        r0 = max(0, p - rx + 1)
        ext = float(np.max(a["high"][r0:p + 1]) if d > 0 else np.min(a["low"][r0:p + 1]))
        levels.append({"price": ext, "type": f"h1_range_{opp}", "timeframe": "H1", "strength": 50.0})
        # ---- noise distributions (bars <= p only) -----------------------------------------
        L = cfg.stop.wick_lookback
        w0 = max(0, p - L + 1)
        o, h, lo, c = a["open"][w0:p + 1], a["high"][w0:p + 1], a["low"][w0:p + 1], a["close"][w0:p + 1]
        wicks = (np.minimum(o, c) - lo) if d > 0 else (h - np.maximum(o, c))
        exc = []
        for t in range(w0, p - 4):
            if d > 0:
                exc.append(max(a["close"][t] - float(np.min(a["low"][t + 1:t + 6])), 0.0))
            else:
                exc.append(max(float(np.max(a["high"][t + 1:t + 6])) - a["close"][t], 0.0))
        P = cfg.stop.percentile_lookback
        p0 = max(0, p - P + 1)
        r5 = [float(np.max(a["high"][t - 4:t + 1]) - np.min(a["low"][t - 4:t + 1])) for t in range(max(p0, 4), p + 1)]
        prim = str(fr["h1_primary_structure"][p])
        align = 1.0 if prim == ("BULLISH" if d > 0 else "BEARISH") else (0.0 if prim == ("BEARISH" if d > 0 else "BULLISH") else 0.5)
        key = "h4_long_context_score" if d > 0 else "h4_short_context_score"
        h4_vol = str(h4.features["volatility_regime"].iloc[k]) if k >= 0 else None
        return TradeContext(
            direction=d, setup_family=cand.setup_family, confirmation_family=conf.get("family"), entry_price=entry,
            price_basis=last["executable"]["chart_price_basis"], spread_pips=float(last["executable"]["spread_used_pips"]),
            spread_assumed=bool(last["executable"]["spread_assumed"]), atr=atr, pip_size=spec.pip_size,
            last_close=float(a["close"][p]), stop_refs=refs, structural_levels=structural, levels=levels,
            adverse_wicks=list(wicks), adverse_excursions=exc, ranges5=r5, recent_extremes=list(lo if d > 0 else h),
            chop=_f(fr["h1_chop_score"][p], 50.0), momentum_net=_f(fr["momentum_net"][p], 0.0),
            h4_side_score=_f(fr[key][p], 0.0), h4_side_score_at_qualification=_f(fr[key][cand.qualified_index], 0.0),
            h4_permission_confidence=_f(fr["h4_permission_confidence"][p], 0.0), h1_primary_alignment=align,
            vol_regime_h1=str(a["vol"][p]), vol_regime_h4=h4_vol, entry_quality=float(acc["entry_quality_score"]),
            setup_score=acc.get("h1_setup_score"), slippage_pips=acc.get("slippage_estimate_pips"),
            slippage_status=acc.get("slippage_status", "UNKNOWN"),
            meta={"symbol": spec.symbol, "timestamp": acc["timestamp"], "session_context": acc.get("session_context"),
                  "news_status": acc.get("news_status"), "h4_permission": fr["h4_permission"][p]},
        )
