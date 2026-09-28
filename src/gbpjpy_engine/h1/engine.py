"""Phase 1C H1 Setup Intelligence Engine (orchestration).

Answers: "is a sufficiently high-quality setup developing in a direction the
H4 engine permits?"  A setup - even a QUALIFIED one - is not a trade; there is
no entry, stop, target, size or order anywhere in Phase 1C.

Point-in-time rules (tested in tests/test_h1_lookahead.py):

* H1 features come from the same causal Phase 1A pipeline (``compute_feature_frame``)
* H4 context is attached by ``align_h4_to_h1``: only the latest H4 bar whose
  close is <= the H1 bar's close; H4 objects (zones, context details) are read
  through that H4 index only
* swings, breaks, sweeps, zones, expansions and setups are replayed bar by bar
  with append-only histories; nothing already emitted is rewritten

Fail-safe: data errors raise (like the H4 engine); an alignment violation or
any exception during a bar blocks actionable setups for that bar and every
later bar (``H1_CONTEXT_ERROR`` / ``H4_ALIGNMENT_ERROR``).
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..context.breakouts import BreakoutAnalyzer
from ..context.displacement import displacement_features
from ..context.hierarchy import classify_layer
from ..context.legs import LegEngine, SwingMemoryTracker
from ..context.liquidity import LiquidityMap
from ..context.location import premium_discount
from ..context.maturity import ExpansionTracker, compression
from ..context.zones import OriginZoneTracker
from ..data.interfaces import assert_canonical
from ..data.model import DataIntegrityError, exclude_unclosed_bars
from ..data.validation import validate_bars
from ..features.indicators import scale01
from ..features.pipeline import compute_feature_frame
from ..features.structure import FAILED, INVALIDATED
from ..snapshot import to_jsonable
from .alignment import AlignmentError, align_h4_to_h1
from .config import H1Config
from .setups import FAMILIES, PULLBACK_OK, SetupTracker, families_for_side, quality_families, score_side, setup_conflict
from .signals import (candle_patterns, confluence, h1_room, make_reclaim, make_transition, pullback, reference_impulse,
                      rejection_base, rejection_score)

logger = logging.getLogger(__name__)
GAP_FLAGS = ("MISSING_BARS", "EXTENDED_MARKET_CLOSURE", "ABNORMAL_PRICE_GAP", "TIMEZONE_ALIGNMENT_SHIFT")
SIDES = ("long", "short")


@dataclass
class H1SetupResult:
    features: pd.DataFrame  # H1 Phase 1A-style feature table (causal)
    frame: pd.DataFrame  # per-H1-bar setup intelligence (research storage)
    details: list
    alignment: pd.DataFrame
    structure: object
    zones_by_bar: list
    setups: list = field(default_factory=list)
    qualified_setups: list = field(default_factory=list)
    counterfactual: list = field(default_factory=list)
    transitions: list = field(default_factory=list)
    reclaims: list = field(default_factory=list)
    sweep_events: list = field(default_factory=list)
    expansions: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    report: object = None
    config: H1Config | None = None

    def explain(self, i: int, side: str) -> dict:
        return self.details[i]["explanation"][side]

    def explain_setup(self, setup_id: str) -> dict:
        """Why did this setup qualify / fail?  (full audit trail)"""
        st = next(s for s in self.setups if s.setup_id == setup_id)
        out = st.to_dict()
        if st.qualified_index is not None:
            out["qualification_explanation"] = self.details[st.qualified_index]["explanation"][st.side]
        if st.end_index is not None:
            out["end_explanation"] = self.details[st.end_index]["explanation"][st.side]
        return to_jsonable(out)

    def export(self, path) -> object:
        from ..logging_utils import export_frame

        return export_frame(self.frame, path)


class H1SetupEngine:
    def __init__(self, config: H1Config | None = None):
        self.cfg = config or H1Config()
        self.cfg.validate()

    # ------------------------------------------------------------------
    def run(self, h1_bars: pd.DataFrame, h4_result, as_of: pd.Timestamp | None = None,
            _fault_at: int | None = None) -> H1SetupResult:
        cfg = self.cfg
        assert_canonical(h1_bars)
        if as_of is not None:
            h1_bars = exclude_unclosed_bars(h1_bars, as_of, 60)
        report = validate_bars(h1_bars, cfg.data)
        if report.has_errors and cfg.engine.fail_on_data_errors:
            raise DataIntegrityError(f"H1 data integrity errors: {report.counts()}", report=report)
        df = h1_bars.reset_index(drop=True)
        feats, structure, zones_by_bar, _ = compute_feature_frame(df, cfg, report)
        n = len(feats)
        tf = pd.Timedelta(minutes=60)
        disp = displacement_features(feats, cfg.displacement)
        pats = candle_patterns(feats)
        rej = rejection_base(feats, cfg.rejection.significance_lookback)
        wick_mean = feats["wick_ratio"].rolling(cfg.chop.window, min_periods=5).mean()
        h1_chop = (0.85 * feats["chop_score"] + 0.15 * 100.0 * scale01(wick_mean, 0.3, 0.7)).round(2)
        alignment_error = None
        try:
            align = align_h4_to_h1(feats["available_at"], h4_result, cfg.alignment)
        except AlignmentError as exc:  # fail safe: never attach possibly-future H4 information
            alignment_error = str(exc)
            align = pd.DataFrame({"h4_index": np.full(n, -1), "h4_context_status": "ALIGNMENT_ERROR"}, index=feats.index)
        a = {
            "open": feats["open"].to_numpy(float), "high": feats["high"].to_numpy(float),
            "low": feats["low"].to_numpy(float), "close": feats["close"].to_numpy(float),
            "atr": feats["atr"].to_numpy(float), "ts": list(feats["timestamp"]), "tf": tf,
            "chop": h1_chop.to_numpy(float), "eff": feats["directional_efficiency"].to_numpy(float),
            "atr_ratio": feats["atr_ratio"].to_numpy(float), "overlap_mean": feats["overlap_mean"].to_numpy(float),
            "overlap_prev": feats["overlap_prev"].to_numpy(float),
            "bullish_disp": disp["bullish_displacement_score"].to_numpy(float),
            "bearish_disp": disp["bearish_displacement_score"].to_numpy(float),
            "disp_eff": disp["disp_efficiency"].to_numpy(float),
        }
        st = structure
        state = {
            "tracker": SwingMemoryTracker(st.swings, cfg.structure.swing_history_size),
            "legs": LegEngine(feats, a["atr"], a["overlap_prev"], cfg.legs, cfg.data.pip_size),
            "analyzer": BreakoutAnalyzer(a, st.swings, zones_by_bar, cfg.breakout_context, cfg.levels.strong_level_score,
                                         cfg.structure.break_monitor_bars),
            "liq": LiquidityMap(a, st.swings, st.breaks, cfg.liquidity),
            "origin": OriginZoneTracker(a, {"bullish": a["bullish_disp"], "bearish": a["bearish_disp"]},
                                        feats["structure_state"].to_numpy(object), cfg.displacement.window, cfg.origin_zones),
            "expansion": ExpansionTracker(cfg.compression, cfg.displacement.moderate_score),
            "trackers": {s: SetupTracker(s, cfg) for s in SIDES},
            "breaks_at": {}, "known_breaks": [], "accepted_swings": [], "transitions": [], "reclaims": [],
        }
        for b in st.breaks:
            state["breaks_at"].setdefault(b.break_index, []).append(b)
        records = feats.to_dict("records")
        align_records = align.to_dict("records")
        rows, details, errors, qualified = [], [], [], []
        degraded = alignment_error is not None
        last_gap = None
        for c in range(n):
            flags = records[c].get("data_quality_flags") or []
            if any(f in GAP_FLAGS for f in flags):
                last_gap = c
            if degraded:
                reason = "H4_ALIGNMENT_ERROR" if alignment_error else "H1_CONTEXT_ERROR"
                rows.append(self._blocked_row(records[c], reason))
                details.append({"explanation": {s: {"state": "BLOCKED", "blockers": [reason]} for s in SIDES},
                                "error": alignment_error or "degraded"})
                continue
            try:
                if _fault_at is not None and c == _fault_at:
                    raise RuntimeError("injected fault (test)")
                row, det = self._bar(c, records[c], align_records[c], a, feats, disp, pats, rej, h1_chop, zones_by_bar,
                                     h4_result, state, last_gap, as_of, n, qualified)
            except Exception as exc:
                degraded = True
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("H1 setup evaluation failed at bar %d (%s); blocking from here on", c, msg)
                errors.append({"index": c, "error": msg, "traceback": traceback.format_exc(limit=4)})
                row = self._blocked_row(records[c], "H1_CONTEXT_ERROR")
                det = {"explanation": {s: {"state": "BLOCKED", "blockers": ["H1_CONTEXT_ERROR"]} for s in SIDES}, "error": msg}
            rows.append(row)
            details.append(det)
        frame = pd.DataFrame(rows, index=feats.index)
        trk = state["trackers"]
        setups = sorted(trk["long"].setups + trk["short"].setups, key=lambda s: (s.created_index, s.side))
        cf = sorted(trk["long"].counterfactual + trk["short"].counterfactual,
                    key=lambda r: (r.get("ended_at") or r.get("at"), r["side"]))
        return H1SetupResult(
            features=feats, frame=frame, details=details, alignment=align, structure=st, zones_by_bar=zones_by_bar,
            setups=setups, qualified_setups=qualified, counterfactual=cf, transitions=state["transitions"],
            reclaims=state["reclaims"], sweep_events=state["liq"].events, expansions=state["expansion"].events,
            errors=errors + ([{"index": 0, "error": alignment_error}] if alignment_error else []), report=report, config=cfg,
        )

    # ------------------------------------------------------------------
    def _blocked_row(self, row: dict, reason: str) -> dict:
        return {"timestamp": row["timestamp"], "available_at": row["available_at"], "h1_blockers": [reason],
                "actionable_setup_permission_long": False, "actionable_setup_permission_short": False,
                "long_setup_state": "NO_SETUP", "short_setup_state": "NO_SETUP", "reason_codes": [f"H1_BLOCKER_{reason}"],
                "alignment_state": "BLOCKED_BY_H4" if reason == "H4_ALIGNMENT_ERROR" else "BLOCKED", "h1_error": reason}

    # ------------------------------------------------------------------
    def _bar(self, c, row, al, a, feats, disp, pats, rej, h1_chop, zones_by_bar, h4, S, last_gap, as_of, n, qualified):
        cfg = self.cfg
        close, atr = a["close"][c], a["atr"][c]
        at = (a["ts"][c] + a["tf"]).isoformat()
        memory = S["tracker"].update(c)
        S["accepted_swings"].extend(S["tracker"].by_accept.get(c, ()))
        new_breaks = S["breaks_at"].get(c, [])
        S["known_breaks"].extend(new_breaks)
        k = int(al["h4_index"])
        h4_ok = al.get("h4_context_status") == "OK" and k >= 0
        perm = al.get("h4_directional_permission") if k >= 0 else None
        permitted = {"long": perm in ("ALLOW_LONG", "ALLOW_BOTH") and h4_ok,
                     "short": perm in ("ALLOW_SHORT", "ALLOW_BOTH") and h4_ok}
        h4_det = h4.context.details[k] if (k >= 0 and h4.context is not None) else {}

        # ---- H1 structure hierarchy & legs --------------------------------
        hc = cfg.hierarchy
        primary = classify_layer("h1_primary", memory[-hc.primary_swings:], close, atr, hc, protect=True)
        immediate = classify_layer("h1_immediate", memory[-hc.immediate_swings:], close, atr, hc, provisional=True)
        ref_dir = 1 if perm == "ALLOW_LONG" else (-1 if perm == "ALLOW_SHORT" else primary.direction)
        legs_eng = S["legs"]
        conf_legs = [legs_eng.confirmed_leg(memory[j - 1], memory[j]) for j in range(1, len(memory))]
        active = legs_eng.active_leg(memory[-1], c) if memory and np.isfinite(atr) else None
        all_legs = conf_legs + ([active] if active is not None else [])
        legs_eng.classify(all_legs, ref_dir)

        # ---- displacement / momentum -------------------------------------
        lo = max(0, c - 2)
        disp_recent = {"long": float(np.max(a["bullish_disp"][lo:c + 1])), "short": float(np.max(a["bearish_disp"][lo:c + 1]))}
        mom_net = float(row.get("momentum_score")) if row.get("momentum_score") == row.get("momentum_score") else 0.0
        bull_mom = feats["bullish_momentum_score"].to_numpy(float)
        bear_mom = feats["bearish_momentum_score"].to_numpy(float)

        # ---- breakouts, transitions, reclaims -----------------------------
        analyzer = S["analyzer"]
        analyzer.set_views(S["accepted_swings"][-64:], S["known_breaks"][-64:])
        monitor = cfg.structure.break_monitor_bars
        kb = S["known_breaks"]
        latest = kb[-1] if kb and c - kb[-1].break_index <= monitor else None
        bc = analyzer.evaluate(latest, c) if latest is not None else None
        for ev in new_breaks:
            counter = bear_mom if ev.direction == "bullish" else bull_mom
            te = make_transition(ev, c, conf_legs, counter, a["bullish_disp"][c], a["bearish_disp"][c], a,
                                 cfg.transition, len(S["transitions"]))
            if te is not None:
                S["transitions"].append(te)
        for ev in reversed(kb):
            if c - ev.break_index > monitor:
                break
            if any(t.index == c and t.to_state in (FAILED, INVALIDATED) for t in ev.transitions):
                S["reclaims"].append(make_reclaim(ev, c, a, a["bullish_disp"][c], a["bearish_disp"][c], len(S["reclaims"])))
        breaks_by_id = {b.event_id: b for b in kb[-64:]}
        trans = {"long": None, "short": None}
        for te in reversed(S["transitions"]):
            if c - te.index > cfg.transition.recent_bars:
                break
            side = "long" if te.direction == "bullish" else "short"
            if trans[side] is None and te.break_event_id in breaks_by_id:
                trans[side] = te.evaluate(breaks_by_id[te.break_event_id], c)
        recl = {"long": None, "short": None}
        for rc in reversed(S["reclaims"]):
            if c - rc.reclaim_index > cfg.reclaim.recent_bars:
                break
            side = "long" if rc.direction == "bullish" else "short"
            if recl[side] is None:
                recl[side] = rc.evaluate(c, a)

        # ---- liquidity -----------------------------------------------------
        liq = S["liq"]
        liq.step(c)
        sweeps = {"long": None, "short": None}
        for ev in reversed(liq.events):
            if c - ev.index > cfg.liquidity.sweep_memory_bars:
                break
            side = "long" if ev.implication == "bullish" else "short"
            if sweeps[side] is None and ev.state_at(c) != "BREAK_AND_ACCEPT":
                opp_after = any(b.break_index > ev.index and b.direction == ev.implication for b in kb[-32:])
                snap = ev.snapshot_at(c, liq.score(ev, c, opp_after))
                snap["extreme"] = float(a["low"][ev.index] if side == "long" else a["high"][ev.index])
                sweeps[side] = snap

        # ---- zones, confluence ---------------------------------------------
        zones = zones_by_bar[c]
        origin_active = S["origin"].step(c)
        levels = [{"lower": z.lower, "upper": z.upper, "source": "H1_ZONE", "timeframe": "H1", "strength": z.strength}
                  for z in zones]
        levels += [{"lower": z.lower, "upper": z.upper, "source": f"H1_{z.direction.upper()}_ORIGIN", "timeframe": "H1",
                    "strength": z.displacement_score} for z in origin_active]
        if k >= 0:
            levels += [{"lower": z.lower, "upper": z.upper, "source": "H4_ZONE", "timeframe": "H4", "strength": z.strength}
                       for z in h4.zones_by_bar[k]]
            for v in (h4_det.get("origin_zones") or {}).get("active") or []:
                levels.append({"lower": v["lower"], "upper": v["upper"], "source": f"H4_{v['direction'].upper()}_ORIGIN",
                               "timeframe": "H4", "strength": v["displacement_strength"]})
            for key in ("last_swing_high", "last_swing_low"):
                p = al.get(f"h4_{key}")
                if p is not None and p == p:
                    levels.append({"lower": p, "upper": p, "source": f"H4_{key.upper()}", "timeframe": "H4", "strength": 60.0})
        conf = confluence(close, atr, levels, cfg.confluence)

        # ---- rejection -------------------------------------------------------
        sup_levels = [(x["lower"], x["upper"]) for x in levels] + [(r.price, r.price) for r in liq.active.values()]
        rej_s = {"long": rejection_score("long", c, a, rej["bullish_rejection_base"].to_numpy(float), sup_levels, cfg.rejection),
                 "short": rejection_score("short", c, a, rej["bearish_rejection_base"].to_numpy(float), sup_levels, cfg.rejection)}

        # ---- compression / expansion ----------------------------------------
        comp_score, comp_comps = compression(memory, a, c, cfg.compression)
        exp = S["expansion"].step(c, comp_score, a["bullish_disp"][c], a["bearish_disp"][c], new_breaks, breaks_by_id,
                                  bar_range=float(a["high"][c] - a["low"][c]))
        latest_exp = S["expansion"].events[-1] if S["expansion"].events else None

        # ---- pullbacks -----------------------------------------------------
        pbs = {}
        for side in SIDES:
            imp = reference_impulse(conf_legs, side, cfg.pullback)
            pbs[side] = pullback(side, imp, c, a, bear_mom if side == "long" else bull_mom, cfg.pullback)

        # ---- location & room -------------------------------------------------
        pdz = premium_discount(memory, close, atr, cfg.location_context)
        h4_pct = al.get("h4_structural_range_percentile")
        rx = cfg.levels.range_extreme_lookback
        r0 = max(0, c - rx + 1)
        hi_i, lo_i = r0 + int(np.argmax(a["high"][r0:c + 1])), r0 + int(np.argmin(a["low"][r0:c + 1]))
        excl = active.start_index + 1 if active is not None else c + 1
        h4_room = (h4_det.get("room") or {}) if k >= 0 else {}
        loc, room, recent_bull_break, recent_bear_break = {}, {}, None, None
        for ev in reversed(kb):
            if c - ev.break_index > 24:
                break
            s_now = ev.state_at(c)
            if s_now in (FAILED, INVALIDATED):
                continue
            if ev.direction == "bullish" and recent_bull_break is None:
                recent_bull_break = ev
            if ev.direction == "bearish" and recent_bear_break is None:
                recent_bear_break = ev
        for side in SIDES:
            d = 1 if side == "long" else -1
            pct = pdz["structural_range_percentile"]
            h1_pos = ((100 - pct) / 100 if d > 0 else pct / 100) if pct is not None else 0.5
            h4_pos = ((100 - h4_pct) / 100 if d > 0 else h4_pct / 100) if h4_pct is not None and h4_pct == h4_pct else 0.5
            rb = recent_bull_break if d > 0 else recent_bear_break
            retest = 1.0 if (rb is not None and np.isfinite(atr) and 0 <= d * (close - rb.level) <= atr) else 0.0
            sw = sweeps[side]
            swing = next((s for s in reversed(memory) if s.kind == ("low" if d > 0 else "high")), None)
            near_swing = float(np.clip(1 - abs(close - swing.price) / atr, 0, 1)) if swing is not None and atr > 0 else 0.0
            comps = {"confluence": (conf["long_score"] if d > 0 else conf["short_score"]) / 100,
                     "h1_range_position": h1_pos, "h4_range_position": h4_pos, "break_retest_area": retest,
                     "liquidity_interaction": (sw["sweep_score"] / 100) if sw else 0.0, "recent_swing": near_swing}
            wts = {"confluence": 0.35, "h1_range_position": 0.15, "h4_range_position": 0.15, "break_retest_area": 0.15,
                   "liquidity_interaction": 0.10, "recent_swing": 0.10}
            loc[side] = {"score": round(100 * sum(comps[x] * w for x, w in wts.items()), 2),
                         "components": {x: round(v, 4) for x, v in comps.items()}}
            barrier = ((h4_room.get(side) or {}).get("nearest_barrier") or {}).get("price")
            room[side] = h1_room(side, close, atr, c, zones, memory, origin_active,
                                 a["high"][hi_i] if d > 0 else a["low"][lo_i], hi_i if d > 0 else lo_i, excl,
                                 cfg.levels.strong_level_score, cfg.room, barrier)

        # ---- blockers --------------------------------------------------------
        blockers = []
        bcfg = cfg.blockers
        if bcfg.block_invalid_data and row.get("data_quality_status") == "ERROR":
            blockers.append("INVALID_H1_DATA")
        if as_of is not None and c == n - 1 and \
                pd.Timestamp(as_of) - (a["ts"][c] + a["tf"]) > pd.Timedelta(hours=bcfg.stale_after_hours):
            blockers.append("STALE_H1_DATA")
        if not bool(row.get("warmup_complete")) or (k >= 0 and not bool(al.get("h4_warmup_complete"))):
            blockers.append("INSUFFICIENT_HISTORY")
        if bcfg.gap_block_bars and last_gap is not None and c - last_gap < bcfg.gap_block_bars:
            blockers.append("UNRESOLVED_DATA_GAP")
        shock = row.get("shock_severity") or 0.0
        if shock >= bcfg.shock_block_severity or (bcfg.block_extreme_volatility_regime and row.get("volatility_regime") == "extreme"):
            blockers.append("EXTREME_H1_VOLATILITY")
        if h1_chop.iloc[c] >= bcfg.severe_chop_score:
            blockers.append("SEVERE_H1_CHOP")
        status = al.get("h4_context_status")
        if status == "NONE":
            blockers.append("NO_H4_CONTEXT")
        elif status == "STALE":
            blockers.append("STALE_H4_CONTEXT")
        elif perm in (None, "BLOCK_ALL"):
            blockers.append("H4_BLOCK_ALL")

        # ---- alignment state -------------------------------------------------
        align_state = self._alignment_state(perm, h4_ok, primary, immediate, pbs, active.direction if active else None)

        # ---- per-side setup evaluation ------------------------------------
        vol_q = {"normal": 100.0, "low": 75.0, "high": 75.0, "very_low": 50.0, "extreme": 25.0}.get(row.get("volatility_regime"), 40.0)
        ext_dir = 1 if row.get("extension_direction") == "up" else (-1 if row.get("extension_direction") == "down" else 0)
        ext_score = float(row.get("extension_score")) if row.get("extension_score") == row.get("extension_score") else 0.0
        bc_d = bc.to_dict() if bc is not None else None
        out_side, expl = {}, {}
        for side in SIDES:
            d = 1 if side == "long" else -1
            opp = "short" if side == "long" else "long"
            t_score = trans[side]["transition_score"] if trans[side] else 0.0
            r_score = recl[side]["reclaim_score"] if recl[side] else 0.0
            trig_parts = {"transition": t_score, "reclaim": r_score, "rejection": rej_s[side][0],
                          "displacement": disp_recent[side]}
            src = max(trig_parts, key=lambda x: (trig_parts[x], x))
            exd = None
            if latest_exp is not None:
                exd = {"event_id": latest_exp.event_id, "direction": latest_exp.direction,
                       "bars_since": c - latest_exp.index, "state": latest_exp.state_at(c),
                       "structural_break": latest_exp.structural_break,
                       "expansion_quality": (exp["latest_expansion"] or {}).get("expansion_quality", 0.0)
                       if exp["latest_expansion"] and exp["latest_expansion"]["event_id"] == latest_exp.event_id else 0.0}
            comp_ext = None
            if latest_exp is not None:
                seg = a["low"][latest_exp.compression_start:latest_exp.index + 1] if d > 0 else \
                    a["high"][latest_exp.compression_start:latest_exp.index + 1]
                comp_ext = float(np.min(seg) if d > 0 else np.max(seg))
            s = {
                "d": d, "dir_name": "bullish" if d > 0 else "bearish", "atr": atr, "pullback": pbs[side],
                "trigger": {"score": trig_parts[src], "source": src, "parts": trig_parts},
                "location": loc[side]["score"], "room": room[side]["room_score"], "break_ctx": bc_d,
                "rejection": rej_s[side][0], "displacement": disp_recent[side], "transition": t_score, "reclaim": r_score,
                "sweep": sweeps[side], "sweep_extreme": sweeps[side]["extreme"] if sweeps[side] else None,
                "expansion": exd, "compression_extreme": comp_ext, "extension_score": ext_score, "extension_dir": ext_dir,
                "opposing_displacement": disp_recent[opp], "chop": float(h1_chop.iloc[c]) if h1_chop.iloc[c] == h1_chop.iloc[c] else 50.0,
                "primary_dir": primary.direction,
                "h4_side_score": float(al.get(f"h4_{side}_context_score") or 0.0) if k >= 0 else 0.0,
                "h4_permission_confidence": float(al.get("h4_permission_confidence") or 0.0) if k >= 0 else 0.0,
                "primary_aligned": primary.confidence if primary.direction == d else 0.0,
                "immediate_aligned": immediate.confidence if immediate.direction == d else 0.0,
                "momentum_net": mom_net,
                "sweep_score_same": sweeps[side]["sweep_score"] if sweeps[side] else 0.0,
                "sweep_score_opp": sweeps[opp]["sweep_score"] if sweeps[opp] else 0.0,
                "efficiency_pct": float(row.get("directional_efficiency_percentile") or 50.0)
                if row.get("directional_efficiency_percentile") == row.get("directional_efficiency_percentile") else 50.0,
                "vol_quality": vol_q,
            }
            fams = families_for_side(side, s, cfg)
            s["families"] = fams
            elig = [f for f in FAMILIES if fams[f]["eligible"]]
            chosen = max(elig, key=lambda f: (fams[f]["family_score"], -FAMILIES.index(f))) if elig else None
            conflict, hits = setup_conflict(side, s, chosen, cfg)
            qf = quality_families(s, conflict, cfg)
            per_family = {f: score_side(qf, fams[f]["family_score"], conflict, cfg) for f in FAMILIES}
            q_score, sscore, sconf = per_family[chosen] if chosen else score_side(qf, 0.0, conflict, cfg)
            side_blockers = list(blockers) + ([] if permitted[side] or "H4_BLOCK_ALL" in blockers
                                              or "NO_H4_CONTEXT" in blockers or "STALE_H4_CONTEXT" in blockers
                                              else ["H4_PERMISSION_CONFLICT"])
            actionable = permitted[side] and not blockers

            def payload(stp, score, side=side, s=s, fams=fams, qf=qf, conflict=conflict, sconf=None):
                pf = per_family[stp.family]
                return self._qualified(stp, score, pf[2], side, c, at, al, primary, immediate, pbs[side], loc[side],
                                       room[side], s, fams[stp.family], qf, conflict, hits, row, h1_chop.iloc[c])

            tctx = {
                "permitted": actionable, "blockers": [b for b in side_blockers if b != "H4_PERMISSION_CONFLICT"],
                "families": fams, "chosen": chosen, "setup_score": sscore, "confidence": sconf, "conflict": conflict,
                "room": room[side]["room_score"], "close": close, "atr": atr if np.isfinite(atr) else 0.0, "d": d,
                "h4_regime": al.get("h4_regime") if k >= 0 else None, "pullback_state": pbs[side]["pullback_state"],
                "break_state": bc.state if (bc is not None and bc.direction == s["dir_name"]) else None,
                "opposing_displacement": disp_recent[opp], "h4_permission": perm,
                "missing": fams[chosen]["missing"] if chosen else [],
                "family_setup_score": lambda f, pf=per_family: pf[f][1], "qualified_payload": payload,
            }
            # the tracker's own confidence check uses the active family's confidence
            tr = S["trackers"][side]
            if tr.active is not None:
                tctx["confidence"] = per_family[tr.active.family][2]
            res = tr.step(c, at, tctx)
            if "QUALIFIED" in res["events"]:
                qualified.append(res["setup"].qualified)
            codes = self._side_codes(side, perm, pbs[side], trans[side], disp_recent[side], sweeps[side], rej_s[side],
                                     conf, room[side], s["chop"], conflict, hits, res, recl[side], chosen, side_blockers)
            out_side[side] = {
                "family": chosen or "NO_VALID_SETUP", "setup_score": sscore, "confidence": sconf, "quality": q_score,
                "conflict": conflict, "state": res["setup_state"], "setup_id": res["setup_id"], "actionable": actionable,
                "events": res["events"], "codes": codes,
            }
            expl[side] = {
                "state": res["setup_state"], "setup_id": res["setup_id"], "setup_family": chosen or "NO_VALID_SETUP",
                "active_family": res["setup_family"], "actionable_setup_permission": actionable,
                "setup_score": sscore, "setup_confidence": sconf,
                "confidence_note": "evidence consistency, NOT a probability of success; never a sizing input",
                "quality_families": qf, "archetype_scores": {f: fams[f]["family_score"] for f in FAMILIES},
                "archetype_eligibility": {f: fams[f]["eligible"] for f in FAMILIES},
                "archetype_missing": {f: fams[f]["missing"] for f in FAMILIES}, "trigger": s["trigger"],
                "conflict": {"score": conflict, "hits": hits}, "blockers": side_blockers,
                "missing_requirements": res["missing_requirements"], "reason_codes": codes,
                "evidence": [x for x in codes if not x.startswith(("H1_BLOCKER_", "H1_CONFLICT_", "SETUP_"))],
            }

        g_codes = [f"H1_BLOCKER_{b}" for b in blockers]
        row_out = {
            "timestamp": row["timestamp"], "available_at": row["available_at"],
            "h4_index": k, "h4_timestamp": al.get("h4_timestamp"), "h4_available_at": al.get("h4_available_at"),
            "h4_context_status": status,
            "h4_context_age_hours": al.get("h4_context_age_hours"), "h4_permission": perm,
            "h4_permission_confidence": al.get("h4_permission_confidence"),
            "h4_long_context_score": al.get("h4_long_context_score"), "h4_short_context_score": al.get("h4_short_context_score"),
            "h4_primary_structure": al.get("h4_primary_structure"), "h4_regime": al.get("h4_regime"),
            "h1_primary_structure": primary.classification, "h1_primary_confidence": primary.confidence,
            "h1_immediate_structure": immediate.classification, "h1_immediate_confidence": immediate.confidence,
            "h1_structure_state": row.get("structure_state"), "alignment_state": align_state,
            "active_leg_direction": active.direction if active else None,
            "active_leg_classification": active.classification if active else None,
            "last_leg_classification": conf_legs[-1].classification if conf_legs else None,
            "bullish_displacement_score": float(a["bullish_disp"][c]), "bearish_displacement_score": float(a["bearish_disp"][c]),
            "momentum_net": mom_net, "momentum_strength": abs(mom_net), "momentum_acceleration": row.get("momentum_acceleration"),
            "atr": atr, "atr_pct": row.get("atr_pct"), "atr_percentile": row.get("atr_percentile"),
            "volatility_regime": str(row.get("volatility_regime")).upper(), "volatility_trend": row.get("volatility_trend"),
            "h1_vs_h4_atr_pct": (row.get("atr_pct") / al["h4_atr_pct"]) if k >= 0 and al.get("h4_atr_pct") else np.nan,
            "h1_chop_score": float(h1_chop.iloc[c]) if h1_chop.iloc[c] == h1_chop.iloc[c] else np.nan,
            "directional_efficiency": row.get("directional_efficiency"),
            "directional_efficiency_percentile": row.get("directional_efficiency_percentile"),
            "long_confluence_score": conf["long_score"], "short_confluence_score": conf["short_score"],
            "h4_h1_zone_confluence": conf["h4_h1_confluence"],
            "bullish_sweep_score": sweeps["long"]["sweep_score"] if sweeps["long"] else 0.0,
            "bearish_sweep_score": sweeps["short"]["sweep_score"] if sweeps["short"] else 0.0,
            "breakout_state": bc.state if bc else None, "breakout_direction": bc.direction if bc else None,
            "breakout_quality_score": bc.breakout_quality_score if bc else np.nan,
            "acceptance_state": bc.acceptance_state if bc else None,
            "bullish_transition_score": trans["long"]["transition_score"] if trans["long"] else 0.0,
            "bearish_transition_score": trans["short"]["transition_score"] if trans["short"] else 0.0,
            "bullish_reclaim_score": recl["long"]["reclaim_score"] if recl["long"] else 0.0,
            "bearish_reclaim_score": recl["short"]["reclaim_score"] if recl["short"] else 0.0,
            "bullish_rejection_score": rej_s["long"][0], "bearish_rejection_score": rej_s["short"][0],
            "compression_score": comp_score, "expansion_state": exp["expansion_state"],
            **{f"{side}_pullback_state": pbs[side]["pullback_state"] for side in SIDES},
            **{f"{side}_pullback_quality": pbs[side]["pullback_quality"] for side in SIDES},
            **{f"{side}_pullback_depth_pct": pbs[side]["depth_pct"] for side in SIDES},
            **{f"{side}_location_score": loc[side]["score"] for side in SIDES},
            "h1_long_room_score": room["long"]["room_score"], "h1_short_room_score": room["short"]["room_score"],
            **{f"{side}_setup_conflict_score": out_side[side]["conflict"] for side in SIDES},
            "long_setup_score": out_side["long"]["setup_score"], "short_setup_score": out_side["short"]["setup_score"],
            "long_setup_confidence": out_side["long"]["confidence"], "short_setup_confidence": out_side["short"]["confidence"],
            "long_setup_family": out_side["long"]["family"], "short_setup_family": out_side["short"]["family"],
            "long_setup_state": out_side["long"]["state"], "short_setup_state": out_side["short"]["state"],
            "long_setup_id": out_side["long"]["setup_id"], "short_setup_id": out_side["short"]["setup_id"],
            "actionable_setup_permission_long": out_side["long"]["actionable"],
            "actionable_setup_permission_short": out_side["short"]["actionable"],
            "setup_events": [f"{s_}:{e}" for s_ in SIDES for e in out_side[s_]["events"]],
            "h1_blockers": list(blockers),
            "reason_codes": list(dict.fromkeys(g_codes + out_side["long"]["codes"] + out_side["short"]["codes"])),
            **{c_: bool(pats[c_].iloc[c]) for c_ in pats.columns},
            "h1_error": None,
        }
        det = {
            "alignment": {kk: to_jsonable(v) for kk, v in al.items()},
            "hierarchy": {"primary": primary.to_dict(), "immediate": immediate.to_dict()},
            "legs": [lg.to_dict() for lg in all_legs[-6:]], "pullback": pbs, "transitions": trans, "reclaims": recl,
            "breakout": bc_d, "sweeps": sweeps, "rejection": {s_: rej_s[s_][1] for s_ in SIDES}, "confluence": conf,
            "compression": {"score": comp_score, "components": comp_comps}, "expansion": exp,
            "location": loc, "room": room, "premium_discount": pdz, "explanation": expl,
        }
        return row_out, det

    # ------------------------------------------------------------------
    @staticmethod
    def _alignment_state(perm, h4_ok, primary, immediate, pbs, active_dir=None) -> str:
        if not h4_ok or perm in (None, "BLOCK_ALL"):
            return "BLOCKED_BY_H4"
        pcls = primary.classification
        if perm == "ALLOW_BOTH":
            return "H1_TRANSITION" if pcls == "TRANSITIONAL" else ("H1_RANGE" if pcls in ("RANGING", "NEUTRAL") else "H4_TWO_WAY")
        d = 1 if perm == "ALLOW_LONG" else -1
        side = "long" if d > 0 else "short"
        name = "BULL" if d > 0 else "BEAR"
        pb = pbs[side]["pullback_state"]
        if pb in ("FAILED_PULLBACK_CONTEXT", "STRUCTURE_THREATENING_PULLBACK") and primary.direction == -d:
            return "H1_COUNTERTREND"
        correcting = active_dir == ("down" if d > 0 else "up")
        if (pb in PULLBACK_OK and correcting) or immediate.direction == -d:
            return f"H4_{name}_H1_PULLBACK"
        if primary.direction == d:
            return f"FULL_{name}ISH_ALIGNMENT"
        if pcls == "TRANSITIONAL":
            return "H1_TRANSITION"
        if pcls in ("RANGING", "NEUTRAL"):
            return "H1_RANGE"
        if primary.direction == -d:
            return "H1_COUNTERTREND"
        return "CONFLICTED"

    # ------------------------------------------------------------------
    @staticmethod
    def _side_codes(side, perm, pb, trans, disp_recent, sweep, rej, conf, room, chop, conflict, hits, res, recl,
                    chosen, blockers) -> list[str]:
        d = 1 if side == "long" else -1
        codes = []
        if perm in ("ALLOW_LONG", "ALLOW_BOTH") and d > 0:
            codes.append("H4_LONG_PERMISSION")
        if perm in ("ALLOW_SHORT", "ALLOW_BOTH") and d < 0:
            codes.append("H4_SHORT_PERMISSION")
        if pb["pullback_state"] in ("SHALLOW_PULLBACK", "HEALTHY_PULLBACK"):
            codes.append("H1_HEALTHY_PULLBACK")
        elif pb["pullback_state"] == "DEEP_PULLBACK":
            codes.append("H1_DEEP_PULLBACK")
        elif pb["pullback_state"] in ("STRUCTURE_THREATENING_PULLBACK", "FAILED_PULLBACK_CONTEXT"):
            codes.append("H1_PULLBACK_STRUCTURE_DAMAGED")
        if pb["counter_momentum_deterioration"] >= 0.5 and pb["pullback_state"] in PULLBACK_OK:
            codes.append("H1_COUNTER_MOMENTUM_DETERIORATING")
        if trans and trans["transition_score"] >= 50:
            codes.append("H1_BULLISH_TRANSITION" if d > 0 else "H1_BEARISH_TRANSITION")
        if recl and recl["reclaim_score"] >= 50:
            codes.append("H1_BULLISH_RECLAIM" if d > 0 else "H1_BEARISH_RECLAIM")
        if disp_recent >= 60:
            codes.append("H1_BULLISH_DISPLACEMENT" if d > 0 else "H1_BEARISH_DISPLACEMENT")
        if sweep and sweep["sweep_score"] >= 50:
            codes.append("H1_DOWNSIDE_SWEEP" if d > 0 else "H1_UPSIDE_SWEEP")
        if rej[0] >= 50 and rej[1].get("level_interaction"):
            codes.append("H1_SUPPORT_REACTION" if d > 0 else "H1_RESISTANCE_REACTION")
        near = conf["long"] if d > 0 else conf["short"]
        if near is not None and len(near["timeframes"]) == 2:
            codes.append("H4_H1_ZONE_CONFLUENCE")
        if room["room_score"] >= 40:
            codes.append("SUFFICIENT_UPSIDE_ROOM" if d > 0 else "SUFFICIENT_DOWNSIDE_ROOM")
        else:
            codes.append("INSUFFICIENT_UPSIDE_ROOM" if d > 0 else "INSUFFICIENT_DOWNSIDE_ROOM")
        if chop >= 60:
            codes.append("H1_HIGH_CHOP")
        if conflict >= 50:
            codes.append("H1_HIGH_CONFLICT")
        codes += [f"H1_CONFLICT_{h}" for h in hits]
        if chosen:
            codes.append(f"SETUP_FAMILY_{chosen}")
        for e in res["events"]:
            codes.append(f"SETUP_{e}")
        codes += [f"H1_BLOCKER_{b}" for b in blockers]
        return list(dict.fromkeys(codes))

    # ------------------------------------------------------------------
    @staticmethod
    def _qualified(stp, score, confidence, side, c, at, al, primary, immediate, pb, loc, room, s, fam, qf, conflict,
                   hits, row, chop) -> dict:
        return to_jsonable({
            "setup_id": stp.setup_id, "timestamp": at, "bar_index": c, "symbol": "GBPJPY",
            "direction": "LONG" if side == "long" else "SHORT", "setup_family": stp.family, "anchor": stp.anchor,
            "h4_permission": al.get("h4_directional_permission"), "h4_permission_confidence": al.get("h4_permission_confidence"),
            "h4_context_score": al.get(f"h4_{side}_context_score"), "h4_bar": al.get("h4_timestamp"),
            "h1_primary_structure": primary.classification, "h1_immediate_structure": immediate.classification,
            "pullback_state": pb["pullback_state"], "pullback_quality": pb["pullback_quality"],
            "location_score": loc["score"], "displacement_score": s["displacement"],
            "transition_score": s["transition"], "rejection_score": s["rejection"],
            "liquidity_context": {"sweep": s["sweep"], "same_side_sweep_score": s["sweep_score_same"]},
            "market_quality": qf["MARKET_QUALITY"], "chop_score": chop, "room_score": room["room_score"],
            "conflict_score": conflict, "conflict_hits": hits, "setup_score": score, "setup_confidence": confidence,
            "trigger": s["trigger"], "quality_families": qf, "invalidation_reference": stp.invalidation_reference,
            "invalidation_reference_note": "structural premise level for research - NOT a stop loss",
            "not_a_trade": "no entry price, stop loss, take profit, position size or order exists in Phase 1C",
        })
