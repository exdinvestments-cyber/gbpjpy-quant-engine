"""Phase 1D Entry Intelligence Engine (orchestration).

Consumes a Phase 1C ``H1SetupResult`` (plus the H4 result it was aligned to)
and decides - for every QUALIFIED setup - whether a precise, timely and
executable ENTRY CANDIDATE develops.  An entry candidate is NOT an order:
nothing here sizes a position, sets a stop loss or take profit, or talks to
any trading platform or broker account.

Decision timing (closed-H1 default)
-----------------------------------
* Confirmation is judged at the CLOSE of H1 bar ``c`` from completed bars only
  (no intrabar sequencing is assumed).
* The decision (accept / defer / reject / expire / invalidate) is taken at the
  first price observable AFTER that close - historically the OPEN of bar
  ``c+1`` - using only: information known at the close of ``c`` plus that one
  price (and the spread known at that moment).  The high, low, close and spread
  report of bar ``c+1`` are never used for that decision.
* Every transition is appended with its bar index and phase (OPEN / CLOSE), so
  ``state_at(i)`` replays history exactly (tested in tests/test_entry_lookahead.py).

Fail-safe: an exception while evaluating a bar invalidates every active
candidate and stops creating new ones from that bar on (``ENTRY_CONTEXT_ERROR``).
"""

from __future__ import annotations

import logging
import traceback
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..context.displacement import displacement_features
from ..features.structure import FAILED, INVALIDATED
from ..snapshot import to_jsonable
from .config import CONFIRMATION_FAMILIES, EntryConfig
from .confirmation import SHORT_NAME, confirmation_quality, evaluate_families
from .execution import (abnormality, assess_spread, deterioration, executable_reference, gap_info, news_status,
                        session_context, slippage, spread_code, stacked_barriers, to_pips)
from .interfaces import BarOpenQuotes, UnknownSlippage
from .lifecycle import EntryCandidate
from .scoring import CHASE_CLASSES, chase_risk, entry_conflict, entry_extension, entry_market_quality, entry_quality, freshness

logger = logging.getLogger(__name__)
SIDES = ("long", "short")
FAMILY_CODE = {
    "STRUCTURAL_BREAK_CONFIRMATION": "STRUCTURAL_CONFIRMATION",
    "DISPLACEMENT_CONFIRMATION": "DISPLACEMENT_CONFIRMED",
    "BREAK_RETEST_CONFIRMATION": "BREAK_RETEST_CONFIRMED",
    "SWEEP_RECLAIM_CONFIRMATION": "SWEEP_RECLAIM_CONFIRMED",
    "MOMENTUM_REACCELERATION_CONFIRMATION": "MOMENTUM_REACCELERATION",
    "COMPRESSION_EXPANSION_CONFIRMATION": "COMPRESSION_EXPANSION_CONFIRMED",
}
AUDIT_LABEL = {
    "STRUCTURAL_BREAK_CONFIRMATION": "{dir} STRUCTURAL BREAK",
    "DISPLACEMENT_CONFIRMATION": "DISPLACEMENT CONFIRMED",
    "BREAK_RETEST_CONFIRMATION": "BREAK-RETEST CONFIRMED",
    "SWEEP_RECLAIM_CONFIRMATION": "SWEEP + RECLAIM CONFIRMED",
    "MOMENTUM_REACCELERATION_CONFIRMATION": "MOMENTUM REACCELERATION",
    "COMPRESSION_EXPANSION_CONFIRMATION": "COMPRESSION -> EXPANSION CONFIRMED",
}
DECISION_CODE = {"ACCEPT_ENTRY_CANDIDATE": "ENTRY_CANDIDATE_ACCEPTED", "DEFER": "ENTRY_DEFERRED",
                 "REJECT": "ENTRY_REJECTED", "EXPIRE": "ENTRY_EXPIRED", "INVALIDATE": "ENTRY_INVALIDATED"}
CATEGORY_CODE = {"H4_PERMISSION_REVOKED": "H4_PERMISSION_REVOKED", "SETUP_INVALIDATED": "SETUP_INVALIDATED",
                 "SETUP_EXPIRED": "SETUP_EXPIRED", "NO_CONFIRMATION": "NO_CONFIRMATION", "WINDOW_EXPIRED": "ENTRY_WINDOW_EXPIRED",
                 "SUPERSEDED": "ENTRY_SUPERSEDED", "ENTRY_CONTEXT_ERROR": "ENTRY_CONTEXT_ERROR",
                 "CHASE": "CHASE_RISK_HIGH", "SPREAD": "SPREAD_TOO_HIGH", "EXTENSION": "ENTRY_OVEREXTENDED",
                 "ROOM": "INSUFFICIENT_REMAINING_ROOM", "STALENESS": "SIGNAL_STALE", "ABNORMAL": "EXECUTION_CONDITIONS_ABNORMAL",
                 "PRICE_DETERIORATION": "PRICE_DETERIORATION_EXCESSIVE", "GAP": "EXECUTION_GAP",
                 "CONFLICT": "ENTRY_CONFLICT_TOO_HIGH", "QUALITY": "ENTRY_QUALITY_TOO_LOW", "NEWS": "NEWS_EVENT_NEARBY"}


def end_code(cand) -> str | None:
    """Stable reason code for why a candidate ended (None while active or after acceptance)."""
    if cand.end_category is None:
        return None
    if cand.end_category == "STRUCTURAL_INVALIDATION":
        for code in ("OPPOSING_STRUCTURE_BREAK", "CONFIRMATION_LEVEL_LOST"):
            if (cand.end_reason or "").startswith(code):
                return code
        return "STRUCTURAL_INVALIDATION"
    return CATEGORY_CODE.get(cand.end_category, "ENTRY_REJECTED")


DECISION_STATE = {"ACCEPT_ENTRY_CANDIDATE": "ENTRY_CANDIDATE", "DEFER": "DEFERRED", "REJECT": "REJECTED",
                  "EXPIRE": "EXPIRED", "INVALIDATE": "INVALIDATED"}


def _f(v, default=np.nan) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if np.isfinite(x) else default


@dataclass
class EntryResult:
    frame: pd.DataFrame  # per-H1-bar entry intelligence (research storage)
    details: list
    candidates: list = field(default_factory=list)
    accepted: list = field(default_factory=list)
    counterfactual: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    config: EntryConfig | None = None

    def candidate(self, entry_candidate_id: str) -> EntryCandidate:
        return next(c for c in self.candidates if c.entry_candidate_id == entry_candidate_id)

    def explain_candidate(self, entry_candidate_id: str) -> dict:
        """Full audit trail of one candidate (accepted or not)."""
        return to_jsonable(self.candidate(entry_candidate_id).to_dict())

    def export(self, path) -> object:
        from ..logging_utils import export_frame

        return export_frame(self.frame, path)


class EntryIntelligenceEngine:
    def __init__(self, config: EntryConfig | None = None):
        self.cfg = config or EntryConfig()
        self.cfg.validate()

    # ------------------------------------------------------------------
    def run(self, h1_result, h4_result, quotes=None, news=None, slippage_model=None, intrabar=None,
            _fault_at: int | None = None) -> EntryResult:
        cfg = self.cfg
        feats, fr = h1_result.features, h1_result.frame
        hcfg = h1_result.config
        n = len(feats)
        pip = hcfg.data.pip_size
        if quotes is None:
            quotes = BarOpenQuotes(feats)
        if isinstance(quotes, BarOpenQuotes):
            quotes.check_grid(feats["timestamp"])
        slip = slippage_model or UnknownSlippage()
        disp = displacement_features(feats, hcfg.displacement)

        def col(frame, name, dtype=float):
            if name not in frame.columns:
                return np.full(n, np.nan if dtype is float else None, dtype=dtype if dtype is float else object)
            return frame[name].to_numpy(dtype) if dtype is float else frame[name].to_numpy(object)

        a = {
            "open": col(feats, "open"), "high": col(feats, "high"), "low": col(feats, "low"), "close": col(feats, "close"),
            "atr": col(feats, "atr"), "range_atr": col(feats, "range_atr"), "atr_ratio": col(feats, "atr_ratio"),
            "shock": col(feats, "shock_severity"), "ema_fast": col(feats, "ema_fast"),
            "imp_up": col(feats, "impulse_up_atr"), "imp_dn": col(feats, "impulse_down_atr"),
            "ext_pct": col(feats, "extension_percentile"), "ext_dir": col(feats, "extension_direction", object),
            "vol_regime": col(feats, "volatility_regime", object),
            "disp": {1: disp["bullish_displacement_score"].to_numpy(float), -1: disp["bearish_displacement_score"].to_numpy(float)},
            "multibar": {1: disp["bullish_displacement_multibar"].to_numpy(bool),
                         -1: disp["bearish_displacement_multibar"].to_numpy(bool)},
            "momentum_net": col(fr, "momentum_net"), "chop": col(fr, "h1_chop_score"),
            "eff_pct": col(fr, "directional_efficiency_percentile"),
        }
        spread_pips = np.array([to_pips(s, cfg.execution, pip) if np.isfinite(s) else np.nan for s in col(feats, "spread")],
                               dtype=float)
        ts = list(feats["timestamp"])
        avail = list(feats["available_at"])
        fr_rec = fr.to_dict("records")
        st = h1_result.structure
        breaks = sorted(st.breaks, key=lambda b: (b.break_index, b.event_id))
        break_idx = [b.break_index for b in breaks]
        breaks_by_id = {b.event_id: b for b in breaks}
        swings_acc = sorted(st.swings, key=lambda s: (s.accepted_index, s.swing_id))
        swing_acc_idx = [s.accepted_index for s in swings_acc]
        h4_ext = (h4_result.features["extension_score"].to_numpy(float), h4_result.features["extension_direction"].to_numpy(object))
        sweeps = list(h1_result.sweep_events)
        sweep_idx = [e.index for e in sweeps]
        exps = list(h1_result.expansions)
        exp_idx = [e.index for e in exps]
        recl = sorted(h1_result.reclaims, key=lambda r: r.reclaim_index)
        recl_idx = [r.reclaim_index for r in recl]
        setups_by_q = {}
        for s in h1_result.setups:
            if s.qualified_index is not None:
                setups_by_q.setdefault(s.qualified_index, []).append(s)
        setup_by_id = {s.setup_id: s for s in h1_result.setups}

        S = {"a": a, "fr": fr_rec, "ts": ts, "avail": avail, "spread_pips": spread_pips, "pip": pip, "hcfg": hcfg,
             "h1": h1_result, "h4": h4_result, "h4_ext": h4_ext, "quotes": quotes, "news": news, "slip": slip,
             "intrabar": intrabar, "breaks": breaks, "break_idx": break_idx, "breaks_by_id": breaks_by_id,
             "swings_acc": swings_acc, "swing_acc_idx": swing_acc_idx, "setup_by_id": setup_by_id, "n": n}
        active: dict[str, EntryCandidate | None] = {s: None for s in SIDES}
        candidates, accepted, cf, errors = [], [], [], []
        rows, details = [], []
        reopen_index = 0
        degraded = False
        for c in range(n):
            ev = {s: [] for s in SIDES}
            det = {"families": {}, "decisions": [], "closure": None}
            closure = c > 0 and (ts[c] - avail[c - 1]) > pd.Timedelta(hours=cfg.gap.closure_gap_hours)
            if closure:
                reopen_index = c
                det["closure"] = {"reopen_index": c, "gap_hours": round((ts[c] - avail[c - 1]).total_seconds() / 3600, 2)}
            guard = c < reopen_index + cfg.gap.reopen_revalidation_bars and reopen_index > 0
            for side in SIDES:
                if active[side] is not None and not active[side].is_active:
                    active[side] = None
            shown = dict(active)
            if degraded:
                rows.append(self._row(c, S, active, ev, {}, guard, closure, error="ENTRY_CONTEXT_ERROR"))
                details.append(det)
                continue
            try:
                if _fault_at is not None and c == _fault_at:
                    raise RuntimeError("injected fault (test)")
                # ---- OPEN phase: first executable price after the close of c-1 ----------------
                for side in SIDES:
                    cand = active[side]
                    if cand is not None and cand.awaiting_price and c >= 1:
                        rec = self._opportunity(cand, c, c - 1, quotes.quote_after(c - 1), S)
                        det["decisions"].append(rec)
                        ev[side].append(rec["decision"])
                        if rec["decision"] == "ACCEPT_ENTRY_CANDIDATE":
                            accepted.append(cand.accepted)
                        self._after_decision(cand, rec, cf)
                # ---- CLOSE phase ---------------------------------------------------------------
                lo = bisect_left(break_idx, c - cfg.confirmation.retest_max_bars - 1)
                ctx = {"a": a, "recent_breaks": breaks[lo:bisect_right(break_idx, c)], "swings": st.swings,
                       "zones_by_bar": h1_result.zones_by_bar, "importance_norm_atr": hcfg.breakout_context.importance_norm_atr,
                       "disp_window": hcfg.displacement.window,
                       "sweeps": sweeps[bisect_left(sweep_idx, c - cfg.confirmation.sweep_max_bars):bisect_right(sweep_idx, c)],
                       "reclaims": recl[bisect_left(recl_idx, c - cfg.confirmation.sweep_max_bars - 1):bisect_right(recl_idx, c)],
                       "expansions": exps[bisect_left(exp_idx, c - cfg.confirmation.expansion_max_bars):bisect_right(exp_idx, c)],
                       "breaks_by_id": breaks_by_id,
                       "sweep_score": {1: _f(fr_rec[c].get("bullish_sweep_score"), 0.0),
                                       -1: _f(fr_rec[c].get("bearish_sweep_score"), 0.0)},
                       "counter_det": {1: self._counter_det(h1_result, c, "long"), -1: self._counter_det(h1_result, c, "short")},
                       "h4_confidence": _f(fr_rec[c].get("h4_permission_confidence"), 0.0)}
                fams_side = {}
                for side in SIDES:
                    d = 1 if side == "long" else -1
                    cand = active[side]
                    if cand is not None and cand.is_active:
                        self._close_checks(cand, c, S, ev[side], cf, closure, reopen_index)
                        if not cand.is_active:
                            active[side] = None
                    q_eff = cand.qualified_index if (cand is not None and cand.is_active) else -1
                    fams_side[side] = evaluate_families(c, d, q_eff, ctx, cfg.confirmation)
                    cand = active[side]
                    if cand is not None and cand.is_active and cand.state in ("WAITING_FOR_CONFIRMATION", "CONFIRMING") \
                            and cand.confirmation is None:
                        self._confirm(cand, c, fams_side[side], S, ev[side], guard, cf)
                        if not cand.is_active:
                            active[side] = None
                    # new qualified setups start WAITING at their qualification bar
                    for stp in setups_by_q.get(c, []):
                        if stp.side != side:
                            continue
                        if active[side] is not None and active[side].is_active:
                            cur = active[side]
                            cur.end(c, "CLOSE", self._iso(avail[c]), "EXPIRED",
                                    f"superseded by newer qualified setup {stp.setup_id}", "SUPERSEDED")
                            self._log_cf(cur, cf)
                        cand = self._new_candidate(stp, c, S)
                        candidates.append(cand)
                        active[side] = cand
                        ev[side].append("WAITING_FOR_CONFIRMATION")
                det["families"] = {s: fams_side[s] for s in SIDES}
                row = self._row(c, S, {s: active[s] or shown[s] for s in SIDES}, ev, fams_side, guard, closure)
            except Exception as exc:  # fail safe
                degraded = True
                msg = f"{type(exc).__name__}: {exc}"
                logger.error("entry evaluation failed at bar %d (%s); invalidating active candidates", c, msg)
                errors.append({"index": c, "error": msg, "traceback": traceback.format_exc(limit=4)})
                for side in SIDES:
                    cand = active[side]
                    if cand is not None and cand.is_active:
                        cand.end(c, "CLOSE", self._iso(avail[c]), "INVALIDATED", f"fail-safe: {msg}", "ENTRY_CONTEXT_ERROR",
                                 "INVALIDATE")
                        self._log_cf(cand, cf)
                    active[side] = None
                row = self._row(c, S, shown, ev, {}, guard, closure, error="ENTRY_CONTEXT_ERROR")
                det["error"] = msg
            rows.append(row)
            details.append(det)
        # live mode: a quote source may already supply the first price after the final close
        if not degraded and n:
            for side in SIDES:
                cand = active[side]
                if cand is not None and cand.awaiting_price:
                    qt = quotes.quote_after(n - 1)
                    if qt is not None:
                        rec = self._opportunity(cand, n, n - 1, qt, S)
                        if rec["decision"] == "ACCEPT_ENTRY_CANDIDATE":
                            accepted.append(cand.accepted)
                        self._after_decision(cand, rec, cf)
        frame = pd.DataFrame(rows, index=feats.index)
        cf.sort(key=lambda r: (r["ended_index"], r["side"]))
        return EntryResult(frame=frame, details=details, candidates=candidates, accepted=accepted, counterfactual=cf,
                           errors=errors, config=cfg)

    # ------------------------------------------------------------------
    @staticmethod
    def _iso(t) -> str:
        return pd.Timestamp(t).isoformat()

    @staticmethod
    def _counter_det(h1, c, side) -> float:
        pb = (h1.details[c].get("pullback") or {}).get(side) or {}
        return _f(pb.get("counter_momentum_deterioration"), 0.0)

    def _new_candidate(self, stp, c, S) -> EntryCandidate:
        at = self._iso(S["avail"][c])
        cand = EntryCandidate(
            entry_candidate_id=f"E-{stp.setup_id}", setup_id=stp.setup_id, side=stp.side, setup_family=stp.family,
            qualified_index=c, qualified_at=at, qualified_close=float(S["a"]["close"][c]),
            policy=self.cfg.policy_for(stp.family), setup_payload=stp.qualified or {},
        )
        cand.note(c, "CLOSE", at, "SETUP QUALIFIED", {"setup_id": stp.setup_id, "setup_family": stp.family,
                                                        "setup_score": (stp.qualified or {}).get("setup_score")})
        cand.move(c, "CLOSE", at, "WAITING_FOR_CONFIRMATION", "Phase 1C setup QUALIFIED; waiting for entry confirmation")
        cand.note(c, "CLOSE", at, "WAITING FOR CONFIRMATION", {"allowed_confirmations": list(cand.policy["allowed"])})
        return cand

    # ------------------------------------------------------------------
    def _integrity(self, cand, c, S) -> tuple[str, str, str] | None:
        """Hard revalidation known at the close of ``c``: (to_state, reason, category) or None if intact."""
        d, side = cand.direction, cand.side
        row = S["fr"][c]
        stp = S["setup_by_id"][cand.setup_id]
        sst = stp.state_at(c)
        if sst == "INVALIDATED":
            return "INVALIDATED", f"SETUP_INVALIDATED: Phase 1C setup invalidated ({stp.end_reason})", "SETUP_INVALIDATED"
        if sst == "EXPIRED":
            return "EXPIRED", f"SETUP_EXPIRED: Phase 1C setup expired ({stp.end_reason})", "SETUP_EXPIRED"
        if sst != "QUALIFIED":
            return "INVALIDATED", f"SETUP_INVALIDATED: setup state {sst}", "SETUP_INVALIDATED"
        perm, status = row.get("h4_permission"), row.get("h4_context_status")
        allowed = ("ALLOW_LONG", "ALLOW_BOTH") if d > 0 else ("ALLOW_SHORT", "ALLOW_BOTH")
        if status != "OK" or perm not in allowed:
            what = "H4 BLOCK_ALL" if perm == "BLOCK_ALL" else (f"H4 context {status}" if status != "OK" else f"H4 permission {perm}")
            return "INVALIDATED", f"H4_PERMISSION_REVOKED: {what}", "H4_PERMISSION_REVOKED"
        blockers = row.get("h1_blockers") or []
        if len(blockers):
            return "INVALIDATED", "H1 hard blocker: " + ",".join(blockers), "SETUP_INVALIDATED"
        close = S["a"]["close"][c]
        opp = "bearish" if d > 0 else "bullish"
        lo, hi = bisect_left(S["break_idx"], c), bisect_right(S["break_idx"], c)
        if any(b.direction == opp for b in S["breaks"][lo:hi]):
            return "INVALIDATED", "OPPOSING_STRUCTURE_BREAK: H1 structure broke against the setup", "STRUCTURAL_INVALIDATION"
        ref = stp.invalidation_reference
        if ref is not None and d * (close - ref) < 0:
            return "INVALIDATED", "close beyond the setup's structural invalidation reference", "STRUCTURAL_INVALIDATION"
        if cand.confirmation is not None and cand.confirmation.get("reference_level") is not None and c > cand.confirmation_index:
            if d * (close - cand.confirmation["reference_level"]) < 0:
                return "INVALIDATED", "CONFIRMATION_LEVEL_LOST: close back through the confirmation reference", \
                    "STRUCTURAL_INVALIDATION"
        return None

    def _close_checks(self, cand, c, S, ev, cf, closure, reopen_index) -> None:
        cfg = self.cfg
        at = self._iso(S["avail"][c])
        if c == cand.qualified_index:
            return
        bad = self._integrity(cand, c, S)
        if bad is not None:
            to, reason, cat = bad
            cand.end(c, "CLOSE", at, to, reason, cat, "INVALIDATE" if to == "INVALIDATED" else "EXPIRE")
            ev.append(to)
            self._log_cf(cand, cf)
            return
        if closure and cand.state in ("WAITING_FOR_CONFIRMATION", "CONFIRMING") and cand.confirmation is None:
            cand.weekend_carryover = True
            cand.reconfirm_after = reopen_index + cfg.gap.reopen_revalidation_bars
            cand.note(c, "CLOSE", at, "WEEKEND/REOPEN REVALIDATION",
                      f"qualified before a market closure; confirmation must come from bars >= {cand.reconfirm_after}")
        if cand.state == "DEFERRED" and cand.reconfirm_after is not None:
            cand.confirmation_history.append(cand.confirmation)
            cand.confirmation, cand.confirmation_index = None, None
            cand.move(c, "CLOSE", at, "WAITING_FOR_CONFIRMATION",
                      "pre-closure confirmation discarded; a post-reopen confirmation is required")
            ev.append("WAITING_FOR_CONFIRMATION")
            return
        if cand.state == "ENTRY_CANDIDATE":
            d = cand.direction
            atr = S["a"]["atr"][c]
            ref = cand.accepted["executable_reference_price"]
            away = d * (S["a"]["close"][c] - ref) / atr if np.isfinite(atr) and atr > 0 else 0.0
            elapsed = c - cand.accepted_index + 1
            if away >= cfg.deterioration.reject_atr:
                cand.end(c, "CLOSE", at, "EXPIRED", f"ENTRY_WINDOW_EXPIRED: price moved {away:.2f} ATR away from the "
                                                    "executable reference", "WINDOW_EXPIRED")
                cand.window_status = "EXPIRED"
                ev.append("EXPIRED")
            elif elapsed >= cfg.lifecycle.candidate_valid_bars:
                cand.end(c, "CLOSE", at, "EXPIRED", f"ENTRY_WINDOW_EXPIRED: candidate availability ({elapsed} bars) elapsed",
                         "WINDOW_EXPIRED")
                cand.window_status = "EXPIRED"
                ev.append("EXPIRED")
            else:
                cand.window_status = "DETERIORATING" if away > cfg.deterioration.acceptable_atr else "AVAILABLE"
            return
        if cand.confirmation is None:
            d = cand.direction
            atr = S["a"]["atr"][c]
            ran = d * (S["a"]["close"][c] - cand.qualified_close) / atr if np.isfinite(atr) and atr > 0 else 0.0
            cand.ran_atr = ran  # evaluated after this bar's confirmation check
            if c - cand.qualified_index > cfg.lifecycle.max_wait_bars:
                cand.end(c, "CLOSE", at, "EXPIRED", f"NO_CONFIRMATION within {cfg.lifecycle.max_wait_bars} bars",
                         "NO_CONFIRMATION", "EXPIRE")
                ev.append("EXPIRED")
                self._log_cf(cand, cf)

    def _confirm(self, cand, c, fams, S, ev, guard, cf) -> None:
        cfg = self.cfg
        at = self._iso(S["avail"][c])
        pol = cand.policy
        scores = {f: fams[f]["score"] for f in CONFIRMATION_FAMILIES}
        eligible = [f for f in pol["allowed"] if fams[f]["detected"] and fams[f]["score"] >= pol["min_family_score"]]
        primary = max(eligible, key=lambda f: (scores[f], -CONFIRMATION_FAMILIES.index(f))) if eligible else None
        mq = 0.5 * (100.0 - _f(S["a"]["chop"][c], 50.0)) + 0.5 * _f(S["a"]["eff_pct"][c], 50.0)
        quality, qinfo = confirmation_quality(scores, primary, mq, cfg.confirmation.w_primary)
        blocked_reason = None
        if c <= cand.qualified_index:
            blocked_reason = "confirmation must come after qualification"
        elif guard:
            blocked_reason = "reopen revalidation: first post-closure bar(s) cannot confirm"
        elif cand.reconfirm_after is not None and c < cand.reconfirm_after:
            blocked_reason = "post-reopen confirmation required"
        confirmed = primary is not None and quality >= pol["min_confirmation_quality"] and blocked_reason is None
        if confirmed:
            d = cand.direction
            fam = fams[primary]
            cand.confirmation = {
                "family": primary, "score": fam["score"], "confirmation_quality_score": quality, "quality_detail": qinfo,
                "index": c, "at": at, "signal_price": float(S["a"]["close"][c]), "reference_level": fam["reference_level"],
                "evidence": fam["evidence"], "components": fam["components"], "family_scores": scores,
                "supporting_families": [f for f in CONFIRMATION_FAMILIES if fams[f]["detected"] and f != primary
                                        and scores[f] >= cfg.confirmation.confirming_score],
                "intrabar": self._intrabar(cand, c, fam, S),
            }
            cand.confirmation_index = c
            dirname = "BULLISH" if d > 0 else "BEARISH"
            for f in [primary] + cand.confirmation["supporting_families"]:
                cand.note(c, "CLOSE", at, AUDIT_LABEL[f].format(dir=dirname), {"score": scores[f]})
            prev = cand.state
            cand.move(c, "CLOSE", at, "CONFIRMING", f"confirmation complete: {primary} score {scores[primary]:.1f}, "
                                                    f"quality {quality:.1f}; awaiting first executable price")
            if prev != "CONFIRMING":
                ev.append("CONFIRMING")
            ev.append("CONFIRMED")
            return
        best_allowed = max((scores[f] for f in pol["allowed"]), default=0.0)
        if cand.state == "WAITING_FOR_CONFIRMATION" and best_allowed >= cfg.confirmation.confirming_score and blocked_reason is None:
            cand.move(c, "CLOSE", at, "CONFIRMING", f"partial confirmation evidence (best allowed family {best_allowed:.1f})")
            ev.append("CONFIRMING")
        elif cand.state == "CONFIRMING" and (best_allowed < cfg.confirmation.confirming_score or blocked_reason):
            cand.move(c, "CLOSE", at, "WAITING_FOR_CONFIRMATION", blocked_reason or "confirmation evidence faded")
            ev.append("WAITING_FOR_CONFIRMATION")
        ran = cand.ran_atr
        cand.last_scores = scores
        if ran > cfg.lifecycle.run_away_atr:
            cand.end(c, "CLOSE", at, "EXPIRED", f"price ran {ran:.2f} ATR in the setup direction without confirmation",
                     "NO_CONFIRMATION", "EXPIRE")
            ev.append("EXPIRED")
            self._log_cf(cand, cf)

    def _intrabar(self, cand, c, fam, S) -> dict:
        prov = S["intrabar"]
        if prov is None:
            return {"source": "none", "note": "closed-H1 decision model; intrabar sequence unknown and not assumed"}
        obs = prov.confirm(cand.side.upper(), fam["reference_level"], S["ts"][c], S["avail"][c], as_of=S["avail"][c])
        return {"source": obs.source, "confirmed": obs.confirmed, "detail": obs.detail,
                "note": "research evidence only - not used for Phase 1D decisions"}

    # ------------------------------------------------------------------
    def _opportunity(self, cand, j, p, quote, S) -> dict:
        """Evaluate one executable opportunity: the first price after the close of bar ``p`` (index ``j`` = p + 1)."""
        cfg = self.cfg
        a, pip, d, side = S["a"], S["pip"], cand.direction, cand.side
        conf = cand.confirmation
        rec = {"entry_candidate_id": cand.entry_candidate_id, "index": j, "based_on_close_of": p, "side": side}
        close_t = S["avail"][p]
        if quote is None:
            return {**rec, "decision": None, "note": "no executable price yet"}
        t = quote.time
        at = self._iso(t)
        rec["at"] = at
        checks, codes = [], []
        atr = a["atr"][p]
        # ---- market closure between the close and the first price ---------------------
        if (t - close_t) > pd.Timedelta(hours=cfg.gap.closure_gap_hours):
            cand.reconfirm_after = j + cfg.gap.reopen_revalidation_bars
            reason = ("WEEKEND_REOPEN_REVALIDATION: market closed between confirmation and the first executable price; "
                      "the pre-closure confirmation is not carried over")
            return self._decide(cand, rec, "DEFER", reason, "WEEKEND_REOPEN", ["WEEKEND_REOPEN_REVALIDATION"],
                                [("market_open_continuity", "FAILED", round((t - close_t).total_seconds() / 3600, 2))], {})
        # ---- freshness / window ------------------------------------------------------------
        bars_conf = p - cand.confirmation_index
        move_conf = d * (quote.price - conf["signal_price"]) / atr if atr > 0 else 0.0
        move_qual = d * (quote.price - cand.qualified_close) / atr if atr > 0 else 0.0
        hours = (t - S["avail"][cand.confirmation_index]).total_seconds() / 3600.0
        fresh = freshness(bars_conf, p - cand.qualified_index, max(move_conf, 0.0), max(move_qual, 0.0), hours, cfg.freshness)
        fresh_code = {"FRESH": "SIGNAL_FRESH", "AGING": "SIGNAL_AGING"}.get(fresh["freshness_class"], "SIGNAL_STALE")
        # ---- spread, executable reference, slippage ------------------------------------
        sp_pips = to_pips(quote.spread, cfg.execution, pip) if quote.spread is not None else None
        hist = S["spread_pips"][max(0, p - cfg.spread.history_bars + 1):p + 1]
        spread = assess_spread(sp_pips, hist, atr, pip, cfg.spread)
        xref = executable_reference(d, quote.price, sp_pips, cfg.execution, pip)
        ex = xref["executable_reference_price"]
        slip = slippage(S["slip"], d, t, sp_pips, atr, pip)
        gap = gap_info(d, a["close"][p], quote.price, atr, cfg.gap)
        detr = deterioration(d, conf["signal_price"], ex, quote.price, atr, pip, cfg.deterioration)
        # ---- room / barriers (H1 + H4, clustered) at the executable reference ------------
        k = int(_f(S["fr"][p].get("h4_index"), -1))
        h4_room = ((S["h4"].context.details[k].get("room") or {}).get(side) or {}) if (k >= 0 and S["h4"].context) else {}
        swings = self._swings_known(p, S)
        bars = stacked_barriers(d, ex, atr, S["h1"].zones_by_bar[p], swings, h4_room.get("barriers"), cfg.barriers, S["hcfg"].room)
        nearest = bars["nearest"]["distance_atr"] if bars["nearest"] else None
        # ---- chase / extension ---------------------------------------------------------------
        stp = S["setup_by_id"][cand.setup_id]
        pb = (S["h1"].details[cand.qualified_index].get("pullback") or {}).get(side) or {}
        imp_atr = None
        if _f(pb.get("depth_pct"), 0.0) > 0 and _f(pb.get("depth_atr"), 0.0) > 0:
            imp_atr = pb["depth_atr"] / (pb["depth_pct"] / 100.0)
        chase = chase_risk(d, ex, conf["reference_level"], stp.invalidation_reference, imp_atr, nearest, atr, cfg.chase)
        origin = next((s.price for s in reversed(swings) if s.kind == ("low" if d > 0 else "high")), None)
        h4x = (_f(S["h4_ext"][0][k]), S["h4_ext"][1][k]) if k >= 0 else (np.nan, None)
        ext = entry_extension(d, ex, a["ema_fast"][p], origin, a["imp_up"][p] if d > 0 else a["imp_dn"][p], a["ext_pct"][p],
                              a["ext_dir"][p], h4x[0], h4x[1], atr, cfg.extension)
        # ---- abnormality, market quality, news, session -----------------------------------
        ratio = spread["spread_median_ratio"]
        abn = abnormality(a["range_atr"][p], a["atr_ratio"][p], gap["gap_atr"], ratio, a["shock"][p], cfg.abnormal)
        mq = entry_market_quality(a["chop"][p], a["eff_pct"][p], a["vol_regime"][p], gap["gap_atr"])
        news = news_status(S["news"], t, cfg.news)
        sess = session_context(t, S["hcfg"].sessions.sessions)
        # ---- structural integrity (hard checks already passed at the close of p) --------
        row = S["fr"][p]
        prim = str(row.get("h1_primary_structure"))
        aligned = 1.0 if prim == ("BULLISH" if d > 0 else "BEARISH") else (0.0 if prim == ("BEARISH" if d > 0 else "BULLISH") else 0.5)
        holding = 1.0 if conf["reference_level"] is None or d * (a["close"][p] - conf["reference_level"]) > 0 else 0.0
        opp = "bearish" if d > 0 else "bullish"
        lo, hi = bisect_left(S["break_idx"], cand.qualified_index + 1), bisect_right(S["break_idx"], p)
        clean = 0.0 if any(b.direction == opp and b.state_at(p) not in (FAILED, INVALIDATED) for b in S["breaks"][lo:hi]) else 1.0
        integrity = 100.0 * (0.4 * aligned + 0.3 * holding + 0.3 * clean)
        # ---- conflict & quality -------------------------------------------------------------
        hits = []
        if d * _f(a["momentum_net"][p], 0.0) < 0:
            hits.append("MOMENTUM_DETERIORATING")
        if ext["entry_extension_score"] >= cfg.extension.extended_score:
            hits.append("PRICE_EXTENDED")
        if bars["remaining_room_score"] < 50.0:
            hits.append("ROOM_COLLAPSED")
        if spread["spread_status"] in ("ELEVATED", "HIGH", "EXTREME"):
            hits.append("POOR_SPREAD")
        if abn["execution_abnormality_score"] >= 50.0:
            hits.append("ABNORMAL_VOLATILITY")
        key = f"h4_{side}_context_score"
        q0, qp = S["fr"][cand.qualified_index], row
        if _f(qp.get(key), 0) < _f(q0.get(key), 0) - 10 or \
                _f(qp.get("h4_permission_confidence"), 0) < _f(q0.get("h4_permission_confidence"), 0) - 10:
            hits.append("H4_CONTEXT_WEAKENED")
        if bars["barrier_density_score"] >= 50.0 or bars["cluster_nearby"]:
            hits.append("OPPOSING_BARRIER_CLUSTER")
        conflict = entry_conflict(hits)
        spread_q = spread["spread_quality_score"] if spread["spread_quality_score"] is not None else cfg.scoring.unknown_spread_quality
        qfam = {
            "CONFIRMATION": conf["confirmation_quality_score"], "TIMING": 100.0 - chase["chase_risk_score"],
            "FRESHNESS": fresh["signal_freshness_score"], "PRICE_QUALITY": 100.0 - detr["price_deterioration_score"],
            "MARKET_QUALITY": mq["entry_market_quality_score"], "STRUCTURAL_INTEGRITY": integrity,
            "ROOM": bars["remaining_room_score"],
            "EXECUTION_CONDITIONS": 0.5 * spread_q + 0.5 * (100.0 - abn["execution_abnormality_score"]),
            "CONFLICT": 100.0 - conflict,
        }
        qfam = {kk: round(float(v), 2) for kk, v in qfam.items()}
        quality = entry_quality(qfam, conflict, cfg.scoring)
        metrics = {
            "quote": {"time": at, "chart_price": quote.price, "spread_source": quote.spread_source, "source": quote.source},
            "freshness": fresh, "spread": spread, "executable": xref, "slippage": slip, "gap": gap, "deterioration": detr,
            "barriers": bars, "chase": chase, "extension": ext, "abnormality": abn, "market_quality": mq, "news": news,
            "session": sess, "structural_integrity_score": round(integrity, 2),
            "conflict": {"entry_conflict_score": conflict, "hits": hits}, "quality_families": qfam, "entry_quality_score": quality,
        }
        # ---- checks & codes --------------------------------------------------------------------
        codes += [FAMILY_CODE[conf["family"]], fresh_code, f"CHASE_RISK_{chase['chase_class']}",
                  {"NOT_EXTENDED": "ENTRY_NOT_EXTENDED", "EXTENDED": "ENTRY_EXTENDED", "OVEREXTENDED": "ENTRY_OVEREXTENDED"}[ext["extension_state"]],
                  spread_code(spread["spread_status"]),
                  {"ACCEPTABLE": "PRICE_DETERIORATION_ACCEPTABLE", "DETERIORATING": "ENTRY_WINDOW_DETERIORATING",
                   "EXCESSIVE": "PRICE_DETERIORATION_EXCESSIVE"}[detr["deterioration_status"]],
                  "SUFFICIENT_REMAINING_ROOM" if bars["remaining_room_score"] >= cfg.barriers.min_remaining_room_score
                  else "INSUFFICIENT_REMAINING_ROOM",
                  "H4_PERMISSION_VALID", "SETUP_STILL_VALID",
                  "EXECUTION_CONDITIONS_ABNORMAL" if abn["status"] == "ABNORMAL" else "EXECUTION_CONDITIONS_NORMAL",
                  {"UNKNOWN": "NEWS_STATUS_UNKNOWN", "CLEAR": "NEWS_CLEAR"}.get(news["news_status"], "NEWS_EVENT_NEARBY")]
        if bars["cluster_nearby"]:
            codes.append("BARRIER_CLUSTER_NEARBY")
        if slip["slippage_status"] == "UNKNOWN":
            codes.append("SLIPPAGE_UNKNOWN")
        if gap["gap_class"] in ("LARGE", "EXTREME"):
            codes.append("EXECUTION_GAP")
        if cand.weekend_carryover:
            codes.append("WEEKEND_REOPEN_REVALIDATION")
        codes += [f"ENTRY_CONFLICT_{h}" for h in hits]
        checks += [
            ("h4_permission", "PASSED", S["fr"][p].get("h4_permission")),
            ("setup_validity", "PASSED", "QUALIFIED"),
            ("structural_integrity", "PASSED", round(integrity, 2)),
            ("freshness", "PASSED" if fresh["freshness_class"] in ("FRESH", "AGING") else "FAILED", fresh["freshness_class"]),
            ("spread", "FAILED" if spread["blocked"] else ("WARN" if spread["spread_status"] != "NORMAL" else "PASSED"),
             spread["spread_status"]),
            ("price_deterioration", "FAILED" if detr["deterioration_status"] == "EXCESSIVE" else
             ("WARN" if detr["deterioration_status"] == "DETERIORATING" else "PASSED"), detr["price_deterioration_atr"]),
            ("gap", "FAILED" if gap["gap_class"] == "EXTREME" else ("WARN" if gap["gap_class"] == "LARGE" else "PASSED"),
             gap["gap_atr"]),
            ("room", "PASSED" if bars["remaining_room_score"] >= cfg.barriers.min_remaining_room_score else "FAILED",
             bars["remaining_room_score"]),
            ("chase", "FAILED" if chase["rejects"] else "PASSED", chase["chase_class"]),
            ("extension", "FAILED" if ext["extension_state"] == "OVEREXTENDED" else "PASSED", ext["extension_state"]),
            ("abnormality", "FAILED" if abn["execution_abnormality_score"] >= cfg.abnormal.reject_score else
             ("WARN" if abn["execution_abnormality_score"] >= cfg.abnormal.defer_score else "PASSED"),
             abn["execution_abnormality_score"]),
            ("news", "FAILED" if news["blocked"] else ("WARN" if news["news_status"] != "CLEAR" else "PASSED"), news["news_status"]),
            ("entry_conflict", "FAILED" if conflict > cfg.scoring.max_entry_conflict else "PASSED", conflict),
            ("entry_quality", "FAILED" if quality < cfg.scoring.min_entry_quality else "PASSED", quality),
        ]
        # ---- decision ------------------------------------------------------------------------------
        if fresh["freshness_class"] == "EXPIRED" or bars_conf >= cfg.lifecycle.window_bars or hours > cfg.lifecycle.window_max_hours:
            why = (f"ENTRY_WINDOW_EXPIRED: {bars_conf} bars / {hours:.1f} h after confirmation"
                   if fresh["freshness_class"] != "EXPIRED" else "SIGNAL_STALE: freshness EXPIRED")
            return self._decide(cand, rec, "EXPIRE", why, "WINDOW_EXPIRED", codes + ["ENTRY_WINDOW_EXPIRED"], checks, metrics)
        rejects = []
        if spread["blocked"]:
            rejects.append(("SPREAD", f"SPREAD_TOO_HIGH ({spread['spread_status']})" if spread["spread_status"] != "UNKNOWN"
                            else "SPREAD_UNKNOWN and configuration blocks unknown spread"))
        if news["blocked"]:
            rejects.append(("NEWS", f"news status {news['news_status']} blocked by configuration"))
        if detr["deterioration_status"] == "EXCESSIVE":
            rejects.append(("PRICE_DETERIORATION", f"PRICE_DETERIORATION_EXCESSIVE ({detr['price_deterioration_atr']:.2f} ATR)"))
        if gap["gap_class"] == "EXTREME":
            rejects.append(("GAP", f"price discontinuity: gap {gap['gap_atr']:.2f} ATR"))
        if abn["execution_abnormality_score"] >= cfg.abnormal.reject_score:
            rejects.append(("ABNORMAL", f"EXECUTION_CONDITIONS_ABNORMAL ({abn['execution_abnormality_score']:.0f})"))
        if chase["rejects"]:
            rejects.append(("CHASE", f"CHASE_RISK_{chase['chase_class']} ({chase['chase_risk_score']:.1f})"))
        if ext["extension_state"] == "OVEREXTENDED":
            rejects.append(("EXTENSION", f"ENTRY_OVEREXTENDED ({ext['entry_extension_score']:.1f})"))
        if bars["remaining_room_score"] < cfg.barriers.min_remaining_room_score:
            rejects.append(("ROOM", f"INSUFFICIENT_REMAINING_ROOM ({bars['remaining_room_score']:.1f})"))
        if fresh["freshness_class"] == "STALE":
            rejects.append(("STALENESS", f"SIGNAL_STALE ({fresh['signal_freshness_score']:.1f})"))
        if conflict > cfg.scoring.max_entry_conflict:
            rejects.append(("CONFLICT", f"ENTRY_CONFLICT_TOO_HIGH ({conflict:.1f})"))
        if quality < cfg.scoring.min_entry_quality:
            rejects.append(("QUALITY", f"ENTRY_QUALITY_TOO_LOW ({quality:.1f})"))
        if rejects:
            extra = ["ENTRY_CONFLICT_TOO_HIGH"] if any(r[0] == "CONFLICT" for r in rejects) else []
            extra += ["ENTRY_QUALITY_TOO_LOW"] if any(r[0] == "QUALITY" for r in rejects) else []
            rec["all_rejection_reasons"] = [{"category": r[0], "reason": r[1]} for r in rejects]
            return self._decide(cand, rec, "REJECT", "; ".join(r[1] for r in rejects), rejects[0][0], codes + extra, checks, metrics)
        friday = S["avail"][cand.confirmation_index]
        if friday.dayofweek == 4 and friday.hour >= cfg.gap.friday_cutoff_hour_utc:
            return self._decide(cand, rec, "DEFER", "PRE_WEEKEND_CUTOFF: late-Friday confirmation is not accepted into the weekend",
                                "PRE_WEEKEND", codes + ["PRE_WEEKEND_CUTOFF"], checks, metrics)
        if gap["gap_class"] == "LARGE" and gap["gap_direction"] == "AGAINST_THESIS":
            return self._decide(cand, rec, "DEFER", f"gap of {gap['gap_atr']:.2f} ATR against the thesis: re-check structure at "
                                                    "the next close", "GAP", codes, checks, metrics)
        if abn["execution_abnormality_score"] >= cfg.abnormal.defer_score:
            return self._decide(cand, rec, "DEFER", f"abnormal execution conditions ({abn['execution_abnormality_score']:.0f})",
                                "ABNORMAL", codes, checks, metrics)
        return self._decide(cand, rec, "ACCEPT_ENTRY_CANDIDATE", "all entry checks passed", None, codes, checks, metrics)

    def _decide(self, cand, rec, decision, reason, category, codes, checks, metrics) -> dict:
        j, at = rec["index"], rec["at"]
        codes = list(dict.fromkeys(codes + [DECISION_CODE[decision]]))
        rec.update({"decision": decision, "reason": reason, "category": category, "reason_codes": codes,
                    "checks": [{"check": c[0], "result": c[1], "value": c[2]} for c in checks], "metrics": metrics})
        rec = to_jsonable(rec)
        cand.opportunities.append(rec)
        for c in checks:
            cand.note(j, "OPEN", at, f"{c[0].upper().replace('_', ' ')} CHECK {c[1]}", c[2])
        to = DECISION_STATE[decision]
        if decision in ("REJECT", "EXPIRE", "INVALIDATE"):
            cand.end(j, "OPEN", at, to, reason, category, decision)
        else:
            cand.move(j, "OPEN", at, to, reason)
            cand.note(j, "OPEN", at, f"ENTRY {'CANDIDATE ACCEPTED' if decision == 'ACCEPT_ENTRY_CANDIDATE' else 'DEFERRED'}", reason)
            if decision == "ACCEPT_ENTRY_CANDIDATE":
                cand.decision = decision
                cand.accepted_index = j
                cand.window_status = metrics["deterioration"]["window_status"]
                cand.accepted = self._payload(cand, rec, metrics, codes)
        return rec

    def _after_decision(self, cand, rec, cf) -> None:
        if rec.get("decision") in ("REJECT", "EXPIRE", "INVALIDATE"):
            self._log_cf(cand, cf)

    def _payload(self, cand, rec, m, codes) -> dict:
        conf = cand.confirmation
        sp = m["spread"]
        return to_jsonable({
            "entry_candidate_id": cand.entry_candidate_id, "setup_id": cand.setup_id, "timestamp": rec["at"],
            "decision_bar_index": rec["index"], "confirmation_timestamp": conf["at"], "symbol": "GBPJPY",
            "direction": "LONG" if cand.direction > 0 else "SHORT", "setup_family": cand.setup_family,
            "confirmation_family": conf["family"],
            "signal_price": conf["signal_price"], "signal_price_basis": m["executable"]["chart_price_basis"],
            "executable_reference_price": m["executable"]["executable_reference_price"],
            "execution_side": m["executable"]["execution_side"], "spread_assumed": m["executable"]["spread_assumed"],
            "spread_status": sp["spread_status"], "spread_pips": sp["spread_pips"],
            "slippage_status": m["slippage"]["slippage_status"], "slippage_estimate_pips": m["slippage"]["slippage_estimate_pips"],
            "price_deterioration": {k: m["deterioration"][k] for k in ("price_deterioration_pips", "price_deterioration_atr",
                                                                        "price_deterioration_score", "deterioration_status")},
            "entry_window_status": m["deterioration"]["window_status"],
            "gap_atr": m["gap"]["gap_atr"],
            "chase_risk_score": m["chase"]["chase_risk_score"], "chase_class": m["chase"]["chase_class"],
            "extension_score": m["extension"]["entry_extension_score"], "extension_state": m["extension"]["extension_state"],
            "freshness_score": m["freshness"]["signal_freshness_score"], "freshness_class": m["freshness"]["freshness_class"],
            "confirmation_quality_score": conf["confirmation_quality_score"],
            "entry_market_quality_score": m["market_quality"]["entry_market_quality_score"],
            "remaining_room_score": m["barriers"]["remaining_room_score"],
            "barrier_density_score": m["barriers"]["barrier_density_score"],
            "nearest_opposing_barrier": m["barriers"]["nearest"],
            "execution_abnormality_score": m["abnormality"]["execution_abnormality_score"],
            "entry_conflict_score": m["conflict"]["entry_conflict_score"], "entry_conflict_hits": m["conflict"]["hits"],
            "entry_quality_score": m["entry_quality_score"], "entry_quality_families": m["quality_families"],
            "h4_permission": cand.setup_payload.get("h4_permission"),
            "h1_setup_score": cand.setup_payload.get("setup_score"),
            "session_context": m["session"], "news_status": m["news"]["news_status"],
            "reason_codes": codes,
            "entry_quality_note": "descriptive quality, NOT a probability and never a position-sizing input",
            "not_an_order": "no position size, risk amount, stop loss, take profit or order ticket exists in Phase 1D",
        })

    # ------------------------------------------------------------------
    @staticmethod
    def _swings_known(p, S) -> list:
        hi = bisect_right(S["swing_acc_idx"], p)
        out = [s for s in S["swings_acc"][max(0, hi - 48):hi] if s.removed_index is None or s.removed_index > p]
        return out[-S["hcfg"].structure.swing_history_size:]

    def _log_cf(self, cand, cf) -> None:
        if cand.accepted is not None:
            return
        last = cand.opportunities[-1] if cand.opportunities else None
        cf.append(to_jsonable({
            "entry_candidate_id": cand.entry_candidate_id, "setup_id": cand.setup_id, "side": cand.side,
            "setup_family": cand.setup_family, "end_state": cand.state, "category": cand.end_category,
            "end_code": end_code(cand), "reason_codes": list(dict.fromkeys((last or {}).get("reason_codes", [])
                                                                          + [end_code(cand)])),
            "reason": cand.end_reason, "ended_index": cand.end_index,
            "ended_at": cand.transitions[-1][2] if cand.transitions else None,
            "confirmed": cand.confirmation is not None,
            "confirmation_family": (cand.confirmation or {}).get("family"),
            "signal_price": (cand.confirmation or {}).get("signal_price"),
            "hypothetical": ({"executable_reference_price": last["metrics"]["executable"]["executable_reference_price"],
                              "chase_risk_score": last["metrics"]["chase"]["chase_risk_score"],
                              "extension_score": last["metrics"]["extension"]["entry_extension_score"],
                              "spread_status": last["metrics"]["spread"]["spread_status"],
                              "remaining_room_score": last["metrics"]["barriers"]["remaining_room_score"],
                              "entry_quality_score": last["metrics"]["entry_quality_score"],
                              "all_rejection_reasons": last.get("all_rejection_reasons")}
                             if last and last.get("metrics") else None),
            "last_family_scores": getattr(cand, "last_scores", None),
            "note": "retained for later validation of the filters; filters must not be retro-fitted to individual outcomes",
        }))

    # ------------------------------------------------------------------
    def _row(self, c, S, active, ev, fams_side, guard, closure, error=None) -> dict:
        sp = S["spread_pips"][c]
        row = {"timestamp": S["ts"][c], "available_at": S["avail"][c], "reopen_bar": bool(closure),
               "reopen_guard": bool(guard), "spread_pips": float(sp) if np.isfinite(sp) else np.nan,
               "entry_error": error}
        codes = []
        for side in SIDES:
            cand = active[side]
            row[f"{side}_entry_state"] = cand.state if cand is not None else "NO_ENTRY"
            row[f"{side}_entry_candidate_id"] = cand.entry_candidate_id if cand is not None else None
            row[f"{side}_setup_id"] = cand.setup_id if cand is not None else None
            fams = fams_side.get(side) or {}
            for f in CONFIRMATION_FAMILIES:
                row[f"{side}_conf_{SHORT_NAME[f]}_score"] = fams[f]["score"] if f in fams else np.nan
            row[f"{side}_confirmation_family"] = cand.confirmation["family"] if (cand is not None and cand.confirmation) else None
            row[f"{side}_confirmation_quality"] = cand.confirmation["confirmation_quality_score"] \
                if (cand is not None and cand.confirmation) else np.nan
            row[f"{side}_window_status"] = cand.window_status if (cand is not None and cand.state == "ENTRY_CANDIDATE") else None
            row[f"{side}_entry_events"] = list(ev[side])
            if cand is not None and cand.end_index == c and end_code(cand):
                codes.append(end_code(cand))
            for e in ev[side]:
                codes.append({"WAITING_FOR_CONFIRMATION": "ENTRY_WAITING_FOR_CONFIRMATION", "CONFIRMING": "ENTRY_CONFIRMING",
                              "CONFIRMED": FAMILY_CODE.get((cand.confirmation or {}).get("family"), "ENTRY_CONFIRMING")
                              if cand is not None else "ENTRY_CONFIRMING",
                              "ACCEPT_ENTRY_CANDIDATE": "ENTRY_CANDIDATE_ACCEPTED", "DEFER": "ENTRY_DEFERRED",
                              "REJECT": "ENTRY_REJECTED", "EXPIRE": "ENTRY_EXPIRED", "INVALIDATE": "ENTRY_INVALIDATED",
                              "EXPIRED": "ENTRY_EXPIRED", "INVALIDATED": "ENTRY_INVALIDATED"}.get(e, "ENTRY_CONFIRMING"))
        if closure:
            codes.append("WEEKEND_REOPEN_REVALIDATION")
        if error:
            codes.append("ENTRY_CONTEXT_ERROR")
        row["reason_codes"] = list(dict.fromkeys(codes))
        return row


__all__ = ["CHASE_CLASSES", "EntryIntelligenceEngine", "EntryResult"]
