"""Phase 1B H4 Context & Directional Permission Engine (orchestration).

Consumes a completed Phase 1A/1A.1 ``H4AnalysisResult`` and walks forward bar
by bar.  At bar ``c`` it reads only:

* feature row ``c`` (every Phase 1A feature is causal and truncation-tested)
* swings accepted by ``c`` (``SwingMemoryTracker``), break lifecycles as known
  at ``c`` (``transitions_known_at``), the zone map emitted at ``c``
* price arrays up to ``c`` and its own state built from earlier bars

It never recomputes Phase 1A features and never modifies Phase 1A outputs.

Fail-safe: any exception while evaluating a bar sets the ``CONTEXT_ERROR``
hard blocker (``BLOCK_ALL``) for that bar AND every later bar of the run,
because stateful trackers may be inconsistent after a failure.  An exception
can therefore never become permission.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import H4Config
from ..features.structure import ACCEPTED, CANDIDATE, CONFIRMED, FAILED, INVALIDATED
from .breakouts import BreakoutAnalyzer
from .displacement import displacement_features
from .hierarchy import classify_hierarchy
from .legs import LegEngine, SwingMemoryTracker
from .liquidity import LiquidityMap
from .location import premium_discount, retracement
from .maturity import ExpansionTracker, compression, deterioration, maturity, trend_run
from .permission import GAP_FLAGS, conflict, decide, directional_score, hard_blockers, quality, structural_direction
from .room import room
from .state_machine import ContextStateMachine, observe
from .zones import OriginZoneTracker, RoleReversalTracker, dynamic_context

logger = logging.getLogger(__name__)

IMPULSE_CLASSES = ("STRONG_IMPULSE", "NORMAL_IMPULSE", "WEAK_IMPULSE", "FAILED_IMPULSE")
LAYER_CODES = {"BULLISH": "BULLISH", "BEARISH": "BEARISH", "RANGING": "RANGING", "TRANSITIONAL": "TRANSITIONAL",
               "NEUTRAL": "NEUTRAL", "UNCLEAR": "UNCLEAR"}

# codes supporting / opposing each direction (for permission explanations)
LONG_SUPPORT = {"PRIMARY_STRUCTURE_BULLISH", "INTERMEDIATE_STRUCTURE_BULLISH", "IMMEDIATE_STRUCTURE_BULLISH",
                "STRONG_BULLISH_DISPLACEMENT", "BULLISH_BREAK_ACCEPTED", "DOWNSIDE_LIQUIDITY_SWEEP",
                "BULLISH_ROLE_REVERSAL", "DEMAND_ZONE_REACTION", "STRONG_SUPPORT_NEARBY", "SUFFICIENT_LONG_ROOM",
                "LOW_CONTEXT_CONFLICT"}
SHORT_SUPPORT = {"PRIMARY_STRUCTURE_BEARISH", "INTERMEDIATE_STRUCTURE_BEARISH", "IMMEDIATE_STRUCTURE_BEARISH",
                 "STRONG_BEARISH_DISPLACEMENT", "BEARISH_BREAK_ACCEPTED", "UPSIDE_LIQUIDITY_SWEEP",
                 "BEARISH_ROLE_REVERSAL", "SUPPLY_ZONE_REACTION", "STRONG_RESISTANCE_NEARBY", "SUFFICIENT_SHORT_ROOM",
                 "LOW_CONTEXT_CONFLICT"}
# codes whose side depends on a direction known at the bar (break, structure or expansion direction)
SUPPORT_IF_SAME_DIRECTION = {"HIGH_QUALITY_BREAK", "HEALTHY_PULLBACK", "TREND_EARLY", "EXPANSION_FROM_COMPRESSION"}
LONG_AGAINST = {"PRIMARY_STRUCTURE_BEARISH", "STRONG_BEARISH_DISPLACEMENT", "BULLISH_FAILED_BREAK", "INSUFFICIENT_LONG_ROOM",
                "STRONG_RESISTANCE_NEARBY", "UPSIDE_LIQUIDITY_SWEEP", "HIGH_CONTEXT_CONFLICT", "LOW_CONTEXT_QUALITY",
                "DEEP_PULLBACK", "STRUCTURAL_TRANSITION", "PROTECTED_LOW_BROKEN"}
SHORT_AGAINST = {"PRIMARY_STRUCTURE_BULLISH", "STRONG_BULLISH_DISPLACEMENT", "BEARISH_FAILED_BREAK", "INSUFFICIENT_SHORT_ROOM",
                 "STRONG_SUPPORT_NEARBY", "DOWNSIDE_LIQUIDITY_SWEEP", "HIGH_CONTEXT_CONFLICT", "LOW_CONTEXT_QUALITY",
                 "DEEP_PULLBACK", "STRUCTURAL_TRANSITION", "PROTECTED_HIGH_BROKEN"}
AGAINST_IF_SAME_DIRECTION = {"TREND_EXTENDED", "EXHAUSTION_RISK", "MOMENTUM_DETERIORATING", "HIGH_FALSE_BREAK_RISK"}


@dataclass
class ContextResult:
    frame: pd.DataFrame
    details: list
    legs: dict = field(default_factory=dict)
    sweep_events: list = field(default_factory=list)
    origin_zones: list = field(default_factory=list)
    role_reversals: list = field(default_factory=list)
    failed_breaks: list = field(default_factory=list)
    expansions: list = field(default_factory=list)
    unexpected_transitions: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def explain(self, i: int) -> dict:
        return self.details[i]["permission_explanation"]


def _finite(x) -> bool:
    return x is not None and np.isfinite(x)


class ContextEngine:
    def __init__(self, config: H4Config | None = None):
        self.cfg = config or H4Config()

    # ------------------------------------------------------------------
    def run(self, result, as_of: pd.Timestamp | None = None, _fault_at: int | None = None) -> ContextResult:
        cfg = self.cfg
        feats = result.features
        n = len(feats)
        tf = pd.Timedelta(minutes=cfg.data.timeframe_minutes)
        disp = displacement_features(feats, cfg.displacement)
        a = {
            "open": feats["open"].to_numpy(float), "high": feats["high"].to_numpy(float),
            "low": feats["low"].to_numpy(float), "close": feats["close"].to_numpy(float),
            "atr": feats["atr"].to_numpy(float), "ts": list(feats["timestamp"]), "tf": tf,
            "chop": feats["chop_score"].to_numpy(float), "eff": feats["directional_efficiency"].to_numpy(float),
            "atr_ratio": feats["atr_ratio"].to_numpy(float), "overlap_mean": feats["overlap_mean"].to_numpy(float),
            "bullish_disp": disp["bullish_displacement_score"].to_numpy(float),
            "bearish_disp": disp["bearish_displacement_score"].to_numpy(float),
            "disp_eff": disp["disp_efficiency"].to_numpy(float),
        }
        records = feats.to_dict("records")
        st = result.structure
        breaks = st.breaks
        breaks_by_id = {b.event_id: b for b in breaks}
        breaks_at: dict[int, list] = {}
        for b in breaks:
            breaks_at.setdefault(b.break_index, []).append(b)
        tracker = SwingMemoryTracker(st.swings, cfg.structure.swing_history_size)
        legs = LegEngine(feats, a["atr"], feats["overlap_prev"].to_numpy(float), cfg.legs, cfg.data.pip_size)
        analyzer = BreakoutAnalyzer(a, st.swings, result.zones_by_bar, cfg.breakout_context,
                                    cfg.levels.strong_level_score, cfg.structure.break_monitor_bars)
        liq = LiquidityMap(a, st.swings, breaks, cfg.liquidity)
        origin = OriginZoneTracker(a, {"bullish": a["bullish_disp"], "bearish": a["bearish_disp"]},
                                   feats["structure_state"].to_numpy(object), cfg.displacement.window, cfg.origin_zones)
        roles = RoleReversalTracker(a, cfg.role_reversal)
        expansion = ExpansionTracker(cfg.compression, cfg.displacement.moderate_score)
        machine = ContextStateMachine(cfg.context_state.flip_window_bars)

        rows, details, failed_records, errors = [], [], [], []
        known_breaks: list = []
        accepted_swings: list = []
        last_gap = None
        degraded = False
        rx = cfg.levels.range_extreme_lookback
        for c in range(n):
            row = records[c]
            # bookkeeping that must happen every bar (cheap, cannot fail on valid input)
            flags = row.get("data_quality_flags") or []
            if any(f in GAP_FLAGS for f in flags):
                last_gap = c
            if degraded:
                rows.append(self._error_row(row, "context state unreliable after an earlier failure"))
                details.append({"permission_explanation": rows[-1]["_explanation"], "error": "degraded"})
                continue
            try:
                if _fault_at is not None and c == _fault_at:
                    raise RuntimeError("injected fault (test)")
                out, det, new_failed = self._bar(
                    c, row, a, disp, records, tracker, legs, analyzer, liq, origin, roles, expansion, machine,
                    known_breaks, accepted_swings, breaks_at, breaks_by_id, st, result.zones_by_bar, last_gap,
                    as_of, n, rx,
                )
                failed_records.extend(new_failed)
            except Exception as exc:  # fail safe: never convert an error into permission
                degraded = True
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("context evaluation failed at bar %d (%s); BLOCK_ALL from here on", c, msg)
                errors.append({"index": c, "error": msg, "traceback": traceback.format_exc(limit=3)})
                out = self._error_row(row, msg)
                det = {"permission_explanation": out["_explanation"], "error": msg}
            rows.append(out)
            details.append(det)
        for r in rows:
            r.pop("_explanation", None)
        frame = pd.DataFrame(rows, index=feats.index)
        return ContextResult(
            frame=frame, details=details, legs={k: v.to_dict() for k, v in legs._cache.items()},
            sweep_events=liq.events, origin_zones=origin.zones, role_reversals=roles.flips,
            failed_breaks=failed_records, expansions=expansion.events, unexpected_transitions=machine.unexpected,
            errors=errors,
        )

    # ------------------------------------------------------------------
    def _error_row(self, row: dict, msg: str) -> dict:
        expl = {"decision": "BLOCK_ALL", "confidence": 0.0, "blockers": ["CONTEXT_ERROR"], "evidence": [],
                "against": [], "failed_requirements": {}, "reason_codes": ["ALL_DIRECTIONS_BLOCKED", "BLOCKER_CONTEXT_ERROR"],
                "error": msg}
        return {"timestamp": row["timestamp"], "directional_permission": "BLOCK_ALL", "permission_confidence": 0.0,
                "hard_blockers": ["CONTEXT_ERROR"], "permission_reason_codes": ["ALL_DIRECTIONS_BLOCKED", "BLOCKER_CONTEXT_ERROR"],
                "context_reason_codes": [], "context_error": msg, "context_state": "BLOCKED", "_explanation": expl}

    # ------------------------------------------------------------------
    def _bar(self, c, row, a, disp, records, tracker, legs, analyzer, liq, origin, roles, expansion, machine,
             known_breaks, accepted_swings, breaks_at, breaks_by_id, st, zones_by_bar, last_gap, as_of, n, rx):
        cfg = self.cfg
        memory = tracker.update(c)
        accepted_swings.extend(tracker.by_accept.get(c, ()))
        new_breaks = breaks_at.get(c, [])
        known_breaks.extend(new_breaks)
        close, atr = a["close"][c], a["atr"][c]

        # ---- hierarchy & legs -------------------------------------------
        hier = classify_hierarchy(memory, close, atr, cfg.hierarchy)
        ctx = {"row": row, "primary": hier["primary"], "intermediate": hier["intermediate"], "immediate": hier["immediate"]}
        D = structural_direction(ctx)
        conf_legs = [legs.confirmed_leg(memory[k - 1], memory[k]) for k in range(1, len(memory))]
        active = legs.active_leg(memory[-1], c) if memory and _finite(atr) else None
        all_legs = conf_legs + ([active] if active is not None else [])
        legs.classify(all_legs, D)
        adir = (1 if active.direction == "up" else -1) if active is not None else 0
        correction_active = active is not None and D != 0 and adir != D
        with_structure = active is not None and D != 0 and adir == D
        prev_same = next((s for s in reversed(memory[:-1]) if active is not None and
                          s.kind == ("high" if adir > 0 else "low")), None)
        beyond_extreme = bool(with_structure and prev_same is not None and adir * (active.end_price - prev_same.price) > 0)
        impulse = None
        if correction_active and conf_legs and (1 if conf_legs[-1].direction == "up" else -1) == D:
            impulse = conf_legs[-1]
        elif with_structure and active.distance > 0:
            impulse = active
        retr = retracement(impulse, a, c, atr, cfg.location_context)
        pdz = premium_discount(memory, close, atr, cfg.location_context)

        # ---- displacement ------------------------------------------------
        lo = max(0, c - cfg.displacement.recent_bars + 1)
        bull_recent = float(np.max(a["bullish_disp"][lo:c + 1]))
        bear_recent = float(np.max(a["bearish_disp"][lo:c + 1]))

        # ---- breakouts ---------------------------------------------------
        analyzer.set_views(accepted_swings[-64:], known_breaks[-64:])
        monitor = cfg.structure.break_monitor_bars
        latest = known_breaks[-1] if known_breaks and c - known_breaks[-1].break_index <= monitor else None
        break_ctx = analyzer.evaluate(latest, c) if latest is not None else None
        mem_bars = cfg.structure.transition_memory_bars
        failed_bull = failed_bear = False
        new_failed = []
        for ev in reversed(known_breaks):
            if c - ev.break_index > max(monitor, mem_bars):
                break
            s_now = ev.state_at(c)
            if s_now in (FAILED, INVALIDATED) and c - ev.break_index <= mem_bars:
                if ev.direction == "bullish":
                    failed_bull = True
                else:
                    failed_bear = True
            if any(t.index == c and t.to_state in (FAILED, INVALIDATED) for t in ev.transitions):
                bc = analyzer.evaluate(ev, c)
                new_failed.append({
                    "event_id": ev.event_id, "attempted_direction": ev.direction, "level": ev.level,
                    "state": s_now, "failed_at": (a["ts"][c] + a["tf"]).isoformat(), "failed_index": c,
                    "max_excursion_atr": bc.max_excursion_atr, "time_beyond_level_bars": bc.closes_beyond,
                    "close_behaviour_atr": bc.current_beyond_atr, "rejection_evidence": bc.rejection_evidence,
                    "opposing_displacement": bc.failure_components.get("opposing_displacement"),
                    "structural_consequence": bool(bc.failure_components.get("structural_consequence")),
                    "failure_score": bc.failure_score,
                })

        # ---- liquidity ---------------------------------------------------
        liq.step(c)
        bull_sweep = bear_sweep = 0.0
        latest_sweep = None
        mem_s = cfg.liquidity.sweep_memory_bars
        for ev in reversed(liq.events):
            if c - ev.index > mem_s:
                break
            opp_after = any(b.break_index > ev.index and b.direction == ev.implication for b in known_breaks[-32:])
            sc = liq.score(ev, c, opp_after)
            if latest_sweep is None:
                latest_sweep = ev.snapshot_at(c, sc)
            if ev.implication == "bullish" and bull_sweep == 0.0 and ev.state_at(c) != "BREAK_AND_ACCEPT":
                bull_sweep = sc
            if ev.implication == "bearish" and bear_sweep == 0.0 and ev.state_at(c) != "BREAK_AND_ACCEPT":
                bear_sweep = sc
        liq_context = {"nearest_upside": None, "nearest_downside": None, "equal_high_clusters": 0, "equal_low_clusters": 0}
        if _finite(atr) and atr > 0:
            up_cl = [x for x in liq.clusters("upside", atr) if x["low"] > close]
            dn_cl = [x for x in liq.clusters("downside", atr) if x["high"] < close]
            if up_cl:
                u = min(up_cl, key=lambda x: x["low"])
                liq_context["nearest_upside"] = {**u, "distance_atr": round((u["low"] - close) / atr, 4)}
            if dn_cl:
                d_ = max(dn_cl, key=lambda x: x["high"])
                liq_context["nearest_downside"] = {**d_, "distance_atr": round((close - d_["high"]) / atr, 4)}
            liq_context["equal_high_clusters"] = sum(x["type"] == "EQUAL_HIGHS" for x in up_cl)
            liq_context["equal_low_clusters"] = sum(x["type"] == "EQUAL_LOWS" for x in dn_cl)

        # ---- zones -------------------------------------------------------
        zones = zones_by_bar[c]
        active_origin = origin.step(c)
        roles.step(c, zones)
        recent_flip = roles.flips[-1] if roles.flips and c - roles.flips[-1]["index"] <= cfg.role_reversal.retest_window_bars else None
        views = [z.view(c, close, atr, cfg.origin_zones) for z in active_origin]
        supply = [v for v in views if v["direction"] == "supply" and v["upper"] >= close]
        demand = [v for v in views if v["direction"] == "demand" and v["lower"] <= close]
        nearest_supply = min(supply, key=lambda v: (v["current_distance_atr"] or 0, v["zone_id"])) if supply else None
        nearest_demand = min(demand, key=lambda v: (v["current_distance_atr"] or 0, v["zone_id"])) if demand else None
        origin_reaction = None
        best = -1
        for z in active_origin:
            if z.interactions >= 1 and z.reaction_atr >= 0.5 and z.last_interaction_index is not None and \
                    c - z.last_interaction_index <= 10 and z.last_interaction_index > best:
                best, origin_reaction = z.last_interaction_index, z.direction
        sr_context = [
            {"zone_id": z.zone_id, "zone_type": z.zone_type, "lower": z.lower, "upper": z.upper, "strength": z.strength,
             "formation_atr": z.formation_atr, **dynamic_context(z.lower, z.upper, z.distance, atr, z.formation_atr),
             "role": roles.tracks[z.zone_id].state if z.zone_id in roles.tracks else "ORIGINAL"}
            for z in sorted(zones, key=lambda q: q.distance)[:6]
        ]

        # ---- maturity / momentum / compression ---------------------------
        run = trend_run(memory, D)
        ids = {s.swing_id for s in run}
        run_legs = [lg for lg in conf_legs if lg.start_swing_id in ids and lg.end_swing_id in ids]
        imps = [lg for lg in run_legs if (1 if lg.direction == "up" else -1) == D]
        cors = [lg for lg in run_legs if (1 if lg.direction == "up" else -1) != D]
        failed_dir = sum(1 for ev in known_breaks[-16:] if c - ev.break_index <= mem_bars and
                         ev.state_at(c) in (FAILED, INVALIDATED) and (ev.direction == "bullish") == (D > 0)) if D else 0
        det_score, det_comps = deterioration(imps, cors, conf_legs, failed_dir) if D else (0.0, {})
        mat = maturity(memory, conf_legs, D, c, close, atr, row.get("extension_state"), row.get("extension_direction"),
                       det_score, cfg.maturity)
        comp_score, comp_comps = compression(memory, a, c, cfg.compression)
        exp = expansion.step(c, comp_score, a["bullish_disp"][c], a["bearish_disp"][c], new_breaks, breaks_by_id,
                             bar_range=float(a["high"][c] - a["low"][c]))

        # ---- room -------------------------------------------------------
        r0 = max(0, c - rx + 1)
        hi_i = r0 + int(np.argmax(a["high"][r0:c + 1]))
        lo_i = r0 + int(np.argmin(a["low"][r0:c + 1]))
        excl = active.start_index + 1 if active is not None else c + 1
        long_room = room("long", close, atr, c, zones, memory, active_origin, a["high"][hi_i], hi_i, excl,
                         cfg.levels.strong_level_score, cfg.room)
        short_room = room("short", close, atr, c, zones, memory, active_origin, a["low"][lo_i], lo_i, excl,
                          cfg.levels.strong_level_score, cfg.room)

        # ---- scoring and permission ---------------------------------------
        ctx.update({
            "bull_disp_recent": bull_recent, "bear_disp_recent": bear_recent, "break_ctx": break_ctx,
            "failed_bullish_recent": failed_bull, "failed_bearish_recent": failed_bear,
            "bull_sweep_score": bull_sweep, "bear_sweep_score": bear_sweep, "latest_sweep": latest_sweep,
            "long_room": long_room, "short_room": short_room, "deterioration_score": det_score,
            "correction_active": correction_active, "correction_class": active.classification if correction_active else None,
            "retracement": retr, "recent_flip": recent_flip, "origin_reaction": origin_reaction,
            "bars_since_gap": (c - last_gap) if last_gap is not None else None,
            "stale": bool(as_of is not None and c == n - 1 and
                          (pd.Timestamp(as_of) - (a["ts"][c] + a["tf"])) > pd.Timedelta(hours=cfg.blockers.stale_after_hours)),
            "expansion": exp, "active_leg": active, "active_leg_with_structure": with_structure,
            "active_leg_beyond_extreme": beyond_extreme,
        })
        conf_score, conf_hits, conf_w = conflict(ctx, cfg)
        qual_score, qual_fam = quality(ctx, cfg)
        long_score, long_detail = directional_score("long", ctx, conf_score, cfg)
        short_score, short_detail = directional_score("short", ctx, conf_score, cfg)
        blockers = hard_blockers(ctx, cfg)
        ctx["blockers"] = blockers
        dec = decide(ctx, long_score, short_score, conf_score, qual_score, blockers, cfg)

        codes = self._codes(ctx, D, break_ctx, bull_recent, bear_recent, bull_sweep, bear_sweep, recent_flip,
                            origin_reaction, long_room, short_room, mat, det_score, comp_score, exp, conf_score,
                            conf_hits, qual_score, row)
        bdir = (1 if break_ctx.direction == "bullish" else -1) if break_ctx is not None else 0
        latest_exp = exp["latest_expansion"]
        edir = (1 if latest_exp["direction"] == "bullish" else -1) if latest_exp else 0
        code_dirs = {"HIGH_QUALITY_BREAK": bdir, "HIGH_FALSE_BREAK_RISK": bdir, "HEALTHY_PULLBACK": D, "TREND_EARLY": D,
                     "TREND_EXTENDED": D, "EXHAUSTION_RISK": D, "MOMENTUM_DETERIORATING": D,
                     "EXPANSION_FROM_COMPRESSION": edir}
        explanation = self._explain(dec, codes, blockers, long_score, short_score, conf_score, qual_score,
                                    long_detail, short_detail, code_dirs)
        observed, why = observe(ctx)
        state = machine.step(c, observed, why, (a["ts"][c] + a["tf"]).isoformat())

        bc = break_ctx
        out = {
            "timestamp": row["timestamp"],
            "primary_structure": hier["primary"].classification,
            "primary_structure_confidence": hier["primary"].confidence,
            "intermediate_structure": hier["intermediate"].classification,
            "intermediate_structure_confidence": hier["intermediate"].confidence,
            "immediate_structure": hier["immediate"].classification,
            "immediate_structure_confidence": hier["immediate"].confidence,
            "structural_direction": D,
            "active_leg_id": active.leg_id if active else None,
            "active_leg_direction": active.direction if active else None,
            "active_leg_classification": active.classification if active else None,
            "active_leg_impulse_score": active.impulse_score if active else np.nan,
            "last_leg_classification": conf_legs[-1].classification if conf_legs else None,
            "last_leg_impulse_score": conf_legs[-1].impulse_score if conf_legs else np.nan,
            "retracement_depth": retr["retracement_depth_pct"] if retr["retracement_depth_pct"] is not None else np.nan,
            "retracement_band": retr["retracement_band"],
            "bullish_displacement_score": float(a["bullish_disp"][c]),
            "bearish_displacement_score": float(a["bearish_disp"][c]),
            "bullish_displacement_class": disp["bullish_displacement_class"].iloc[c],
            "bearish_displacement_class": disp["bearish_displacement_class"].iloc[c],
            "recent_bullish_displacement": bull_recent, "recent_bearish_displacement": bear_recent,
            "breakout_event_id": bc.event_id if bc else None,
            "breakout_direction": bc.direction if bc else None,
            "breakout_state": bc.state if bc else None,
            "breakout_quality_score": bc.breakout_quality_score if bc else np.nan,
            "breakout_quality_class": bc.breakout_quality_class if bc else None,
            "acceptance_state": bc.acceptance_state if bc else None,
            "failure_score": bc.failure_score if bc else 0.0,
            "false_break_risk_score": bc.false_break_risk_score if bc else np.nan,
            "bullish_sweep_score": bull_sweep, "bearish_sweep_score": bear_sweep,
            "latest_sweep_side": latest_sweep["side"] if latest_sweep else None,
            "latest_sweep_state": latest_sweep["state"] if latest_sweep else None,
            "nearest_supply_zone": nearest_supply["upper"] if nearest_supply else np.nan,
            "nearest_supply_zone_lower": nearest_supply["lower"] if nearest_supply else np.nan,
            "nearest_demand_zone": nearest_demand["lower"] if nearest_demand else np.nan,
            "nearest_demand_zone_upper": nearest_demand["upper"] if nearest_demand else np.nan,
            "supply_zone_freshness": nearest_supply["freshness"] if nearest_supply else None,
            "demand_zone_freshness": nearest_demand["freshness"] if nearest_demand else None,
            "latest_role_reversal": recent_flip["type"] if recent_flip else None,
            "structural_range_percentile": pdz["structural_range_percentile"] if pdz["structural_range_percentile"] is not None else np.nan,
            "premium_discount_state": pdz["premium_discount_state"],
            "trend_maturity": mat["trend_maturity"],
            "maturity_score": mat["maturity_score"] if mat["maturity_score"] is not None else np.nan,
            "momentum_deterioration_score": det_score,
            "compression_score": comp_score, "expansion_state": exp["expansion_state"],
            "long_room_score": long_room["room_score"], "short_room_score": short_room["room_score"],
            "context_conflict_score": conf_score, "context_quality_score": qual_score,
            "long_context_score": long_score, "short_context_score": short_score,
            "directional_permission": dec["directional_permission"],
            "permission_confidence": dec["permission_confidence"],
            "permission_reason_codes": explanation["reason_codes"],
            "hard_blockers": list(blockers),
            "context_reason_codes": codes,
            "context_state": state["context_state"], "previous_context_state": state["previous_context_state"],
            "bars_in_state": state["bars_in_state"], "state_changed_at": state["state_changed_at"],
            "state_transition_expected": state["state_transition_expected"],
            "state_changes_in_window": state["state_changes_in_window"],
            "context_error": None,
            "_explanation": explanation,
        }
        det = {
            "hierarchy": {k: v.to_dict() for k, v in hier.items()},
            "legs": [lg.to_dict() for lg in all_legs[-8:]],
            "active_leg": active.to_dict() if active else None,
            "retracement": retr, "premium_discount": pdz,
            "displacement": {"bullish": float(a["bullish_disp"][c]), "bearish": float(a["bearish_disp"][c]),
                             "bullish_class": out["bullish_displacement_class"], "bearish_class": out["bearish_displacement_class"],
                             "net_move_atr": float(disp["disp_net_move_atr"].iloc[c]) if _finite(disp["disp_net_move_atr"].iloc[c]) else None,
                             "recent_bullish": bull_recent, "recent_bearish": bear_recent},
            "breakout": bc.to_dict() if bc else None,
            "liquidity": {**liq_context, "latest_sweep": latest_sweep, "bullish_sweep_score": bull_sweep,
                          "bearish_sweep_score": bear_sweep},
            "origin_zones": {"nearest_supply": nearest_supply, "nearest_demand": nearest_demand, "active": views[:8]},
            "sr_zone_context": sr_context, "role_reversal": recent_flip,
            "maturity": mat, "momentum_deterioration": {"score": det_score, "components": det_comps},
            "compression": {"score": comp_score, "components": comp_comps}, "expansion": exp,
            "room": {"long": long_room, "short": short_room},
            "conflict": {"score": conf_score, "hits": conf_hits, "weights": conf_w},
            "quality": {"score": qual_score, "families": qual_fam},
            "long_context": {"score": long_score, **long_detail}, "short_context": {"score": short_score, **short_detail},
            "decision": dec, "state": state, "permission_explanation": explanation,
        }
        return out, det, new_failed

    # ------------------------------------------------------------------
    def _codes(self, ctx, D, bc, bull_recent, bear_recent, bull_sweep, bear_sweep, flip, origin_reaction,
               long_room, short_room, mat, det, comp, exp, conf, conf_hits, qual, row) -> list[str]:
        cfg = self.cfg
        codes: list[str] = []
        p, i, im = ctx["primary"], ctx["intermediate"], ctx["immediate"]
        codes.append(f"PRIMARY_STRUCTURE_{LAYER_CODES[p.classification]}")
        if i.classification in ("BULLISH", "BEARISH"):
            codes.append(f"INTERMEDIATE_STRUCTURE_{i.classification}")
        if im.classification in ("BULLISH", "BEARISH"):
            codes.append(f"IMMEDIATE_STRUCTURE_{im.classification}")
        for lr in (p, i):
            if lr.broken_from == "BULLISH":
                codes.append("PROTECTED_LOW_BROKEN")
            elif lr.broken_from == "BEARISH":
                codes.append("PROTECTED_HIGH_BROKEN")
        if p.classification == "TRANSITIONAL" or i.classification == "TRANSITIONAL":
            codes.append("STRUCTURAL_TRANSITION")
        if ctx["correction_active"]:
            band = ctx["retracement"]["retracement_band"]
            if ctx["correction_class"] == "CHOPPY_CORRECTION":
                codes.append("CHOPPY_PULLBACK")
            elif band in ("DEEP", "VERY_DEEP") or ctx["correction_class"] == "DEEP_CORRECTION":
                codes.append("DEEP_PULLBACK")
            else:
                codes.append("HEALTHY_PULLBACK")
        strong = cfg.displacement.strong_score
        if bull_recent >= strong:
            codes.append("STRONG_BULLISH_DISPLACEMENT")
        if bear_recent >= strong:
            codes.append("STRONG_BEARISH_DISPLACEMENT")
        if bc is not None:
            if bc.acceptance_state == "ACCEPTING":
                codes.append("BULLISH_BREAK_ACCEPTED" if bc.direction == "bullish" else "BEARISH_BREAK_ACCEPTED")
            if bc.acceptance_state == "REJECTING":
                codes.append("BREAK_REJECTING")
            if bc.breakout_quality_class == "HIGH_QUALITY_BREAK" and bc.state not in (FAILED, INVALIDATED):
                codes.append("HIGH_QUALITY_BREAK")
            if bc.state not in (FAILED, INVALIDATED) and bc.false_break_risk_score >= cfg.breakout_context.high_false_break_risk:
                codes.append("HIGH_FALSE_BREAK_RISK")
        if ctx["failed_bullish_recent"]:
            codes.append("BULLISH_FAILED_BREAK")
        if ctx["failed_bearish_recent"]:
            codes.append("BEARISH_FAILED_BREAK")
        if bull_sweep >= 50:
            codes.append("DOWNSIDE_LIQUIDITY_SWEEP")
        if bear_sweep >= 50:
            codes.append("UPSIDE_LIQUIDITY_SWEEP")
        if ctx["latest_sweep"] is not None and ctx["latest_sweep"]["state"] == "BREAK_AND_ACCEPT":
            codes.append("LIQUIDITY_BREAK_ACCEPTED")
        if flip is not None:
            codes.append("BULLISH_ROLE_REVERSAL" if flip["type"] == "RESISTANCE_TO_SUPPORT" else "BEARISH_ROLE_REVERSAL")
        if origin_reaction == "demand":
            codes.append("DEMAND_ZONE_REACTION")
        elif origin_reaction == "supply":
            codes.append("SUPPLY_ZONE_REACTION")
        near, sl = cfg.levels.near_level_atr, cfg.levels.strong_level_score
        def fnum(k, d):
            v = row.get(k)
            return d if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        if fnum("distance_to_support_atr", 99) <= near and fnum("support_strength", 0) >= sl:
            codes.append("STRONG_SUPPORT_NEARBY")
        if fnum("distance_to_resistance_atr", 99) <= near and fnum("resistance_strength", 0) >= sl:
            codes.append("STRONG_RESISTANCE_NEARBY")
        suff = cfg.room.sufficient_room_score
        codes.append("SUFFICIENT_LONG_ROOM" if long_room["room_score"] >= suff else "INSUFFICIENT_LONG_ROOM")
        codes.append("SUFFICIENT_SHORT_ROOM" if short_room["room_score"] >= suff else "INSUFFICIENT_SHORT_ROOM")
        if mat["trend_maturity"] == "EARLY":
            codes.append("TREND_EARLY")
        if mat["trend_maturity"] in ("EXTENDED", "EXHAUSTION_RISK"):
            codes.append("TREND_EXTENDED")
        if mat["trend_maturity"] == "EXHAUSTION_RISK":
            codes.append("EXHAUSTION_RISK")
        if det >= cfg.maturity.deterioration_high:
            codes.append("MOMENTUM_DETERIORATING")
        if comp >= cfg.compression.compression_threshold:
            codes.append("COMPRESSION_PRESENT")
        if exp["expansion_state"].startswith("EXPANDING") or exp["expansion_state"] == "EXPANSION_ACCEPTED":
            codes.append("EXPANSION_FROM_COMPRESSION")
        sc = cfg.context_scoring
        if conf >= sc.high_conflict:
            codes.append("HIGH_CONTEXT_CONFLICT")
        elif conf < sc.low_conflict:
            codes.append("LOW_CONTEXT_CONFLICT")
        codes.extend(conf_hits)
        if qual < sc.low_quality:
            codes.append("LOW_CONTEXT_QUALITY")
        return list(dict.fromkeys(codes))

    def _explain(self, dec, codes, blockers, long_score, short_score, conf, qual, ld, sd, code_dirs) -> dict:
        decision = dec["directional_permission"]

        def partition(side):
            sd_ = 1 if side == "long" else -1
            support = LONG_SUPPORT if side == "long" else SHORT_SUPPORT
            against = LONG_AGAINST if side == "long" else SHORT_AGAINST
            ev = [x for x in codes if x in support or (x in SUPPORT_IF_SAME_DIRECTION and code_dirs.get(x) == sd_)]
            ag = [x for x in codes if x in against or (x in AGAINST_IF_SAME_DIRECTION and code_dirs.get(x) == sd_)]
            ag += [x for x in codes if x.startswith("CONFLICT_")]
            return ev, list(dict.fromkeys(ag))

        if decision == "ALLOW_LONG":
            ev, ag = partition("long")
        elif decision == "ALLOW_SHORT":
            ev, ag = partition("short")
        elif decision == "ALLOW_BOTH":
            e1, a1 = partition("long")
            e2, a2 = partition("short")
            ev, ag = list(dict.fromkeys(e1 + e2)), list(dict.fromkeys(a1 + a2))
        else:
            ev = []
            ag = [x for x in codes if x in (LONG_AGAINST | SHORT_AGAINST) or x.startswith("CONFLICT_")]
        reason_codes = list(dict.fromkeys(dec["permission_codes"] + ev + (ag if decision == "BLOCK_ALL" else [])))
        return {
            "decision": decision, "confidence": dec["permission_confidence"], "evidence": ev, "against": ag,
            "blockers": list(blockers), "failed_requirements": dec["failed_requirements"], "requirements": dec["requirements"],
            "scores": {"long_context_score": long_score, "short_context_score": short_score,
                       "context_conflict_score": conf, "context_quality_score": qual},
            "long_breakdown": ld, "short_breakdown": sd, "regime": dec["regime"], "reason_codes": reason_codes,
            "note": "Directional permission defines what the H1 engine may SEARCH for. It is not a trade signal "
                    "and must not be used for position sizing.",
        }
