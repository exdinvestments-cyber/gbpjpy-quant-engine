"""H1 setup families, setup conflict, quality families, scores and the setup
lifecycle (NO_SETUP -> WATCHING -> DEVELOPING -> QUALIFIED, or INVALIDATED /
EXPIRED).

A QUALIFIED setup is NOT a trade: there is no entry price, stop, target, size
or order anywhere in this module.  ``invalidation_reference`` is the
structural price whose loss would invalidate the setup's premise; it is stored
for research and is not a stop-loss.

``setup_confidence`` measures how CONSISTENT the evidence families are (share
of weighted families at or above 50, reduced by conflict).  It is NOT a
probability of success and must not be used for position sizing.

All thresholds and weights are documented, untested baselines; none was tuned
to make scores high or to profit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

FAMILIES = ("TREND_PULLBACK_CONTINUATION", "BREAK_RETEST_CONTINUATION",
            "LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION", "COMPRESSION_EXPANSION_IN_H4_DIRECTION")
FAMILY_SHORT = {"TREND_PULLBACK_CONTINUATION": "TPC", "BREAK_RETEST_CONTINUATION": "BRC",
                "LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION": "LSR", "COMPRESSION_EXPANSION_IN_H4_DIRECTION": "CEX"}
PULLBACK_OK = ("SHALLOW_PULLBACK", "HEALTHY_PULLBACK", "DEEP_PULLBACK")
CONFLICT_WEIGHTS = {
    "H1_STRONGLY_OPPOSED": 0.35,
    "H1_HIGH_CHOP": 0.25,
    "TRIGGER_UNDER_BARRIER": 0.30,
    "SWEEP_WITHOUT_FOLLOW_THROUGH": 0.20,
    "HIGH_BREAKOUT_FAILURE_RISK": 0.20,
    "EXTREME_EXTENSION": 0.25,
    "H1_STRUCTURE_BROKEN_AGAINST": 0.30,
}


def families_for_side(side: str, s: dict, cfg) -> dict:
    """Evaluate the four archetypes for one side.  ``s`` holds the side's point-in-time inputs."""
    sc = cfg.setup
    pb, trig = s["pullback"], s["trigger"]
    out = {}
    # A. trend pullback continuation
    elig = pb["pullback_state"] in PULLBACK_OK and pb["impulse_leg"] is not None
    score = 100.0 * (0.25 * pb["pullback_quality"] / 100 + 0.20 * s["location"] / 100
                     + 0.15 * pb["counter_momentum_deterioration"] + 0.25 * trig["score"] / 100 + 0.15 * s["room"] / 100)
    out["TREND_PULLBACK_CONTINUATION"] = {
        "eligible": elig, "family_score": round(score, 2), "trigger_score": trig["score"], "trigger": trig["source"],
        "anchor": f"L{pb['impulse_leg']}" if pb["impulse_leg"] else None,
        # premise while developing: the impulse origin holds; frozen to the correction extreme at qualification
        "invalidation_reference": pb.get("impulse_origin"),
        "qualified_reference": pb.get("correction_extreme"),
        "missing": [] if elig else [f"pullback_state {pb['pullback_state']} not in {PULLBACK_OK}"],
    }
    # B. break + retest continuation
    bc = s["break_ctx"]
    ok_break = (bc is not None and bc["direction"] == s["dir_name"] and bc["bars_since_break"] <= sc.family_event_bars
                and bc["state"] in ("CONFIRMED", "ACCEPTED"))
    retest = ok_break and "successful_retest" in bc["acceptance_evidence"]
    acc = {"ACCEPTING": 1.0, "PARTIALLY_ACCEPTING": 0.6}.get(bc["acceptance_state"], 0.2) if bc else 0.0
    trig_b = max(s["rejection"], s["displacement"]) if retest else 0.0
    score = 100.0 * (0.30 * (bc["breakout_quality_score"] / 100 if bc else 0) + 0.20 * acc + 0.25 * trig_b / 100
                     + 0.10 * s["location"] / 100 + 0.15 * s["room"] / 100)
    miss = []
    if not ok_break:
        miss.append("no recent confirmed/accepted H1 break in this direction")
    elif not retest:
        miss.append("no successful retest of the broken level yet")
    out["BREAK_RETEST_CONTINUATION"] = {
        "eligible": bool(ok_break and retest), "family_score": round(score, 2), "trigger_score": round(trig_b, 2),
        "trigger": "retest_reaction", "anchor": f"B{bc['event_id']}" if ok_break else None,
        "invalidation_reference": (bc["level"] - s["d"] * 0.25 * s["atr"]) if ok_break else None, "missing": miss,
    }
    # C. liquidity sweep reversal in the H4 direction
    sw = s["sweep"]
    ok_sw = sw is not None and sw["bars_since"] <= sc.family_event_bars and sw["state"] != "BREAK_AND_ACCEPT"
    trig_c = max(s["transition"], s["displacement"], s["reclaim"]) if ok_sw else 0.0
    score = 100.0 * (0.35 * (sw["sweep_score"] / 100 if ok_sw else 0) + 0.35 * trig_c / 100
                     + 0.15 * s["location"] / 100 + 0.15 * s["room"] / 100)
    out["LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"] = {
        "eligible": bool(ok_sw), "family_score": round(score, 2), "trigger_score": round(trig_c, 2),
        "trigger": "post_sweep_transition_or_displacement", "anchor": f"S{sw['event_id']}" if ok_sw else None,
        "invalidation_reference": s.get("sweep_extreme") if ok_sw else None,
        "missing": [] if ok_sw else ["no recent rejected liquidity sweep implying this direction"],
    }
    # D. compression -> expansion in the H4 direction
    ex = s["expansion"]
    ok_ex = (ex is not None and ex["direction"] == s["dir_name"] and ex["bars_since"] <= sc.family_event_bars
             and ex["state"] in ("EXPANDING", "EXPANSION_ACCEPTED"))
    chase = s["extension_score"] >= sc.extension_chase_score and s["extension_dir"] == s["d"]
    trig_d = ex["expansion_quality"] if ok_ex else 0.0
    score = 100.0 * (0.30 * trig_d / 100 + 0.25 * s["displacement"] / 100 + 0.15 * (1.0 if ok_ex and ex["structural_break"] else 0)
                     + 0.15 * s["room"] / 100 + 0.15 * (0.0 if chase else 1.0))
    miss = [] if ok_ex else ["no recent expansion out of compression in this direction"]
    if chase:
        miss.append("expansion already excessively extended (chasing)")
    out["COMPRESSION_EXPANSION_IN_H4_DIRECTION"] = {
        "eligible": bool(ok_ex and not chase), "family_score": round(score, 2), "trigger_score": round(trig_d, 2),
        "trigger": "expansion_quality", "anchor": f"X{ex['event_id']}" if ok_ex else None,
        "invalidation_reference": s.get("compression_extreme") if ok_ex else None, "missing": miss,
    }
    return out


def setup_conflict(side: str, s: dict, chosen: str | None, cfg) -> tuple[float, list[str]]:
    hits = []
    if s["opposing_displacement"] >= 70:
        hits.append("H1_STRONGLY_OPPOSED")
    if s["chop"] >= 60:
        hits.append("H1_HIGH_CHOP")
    if s["trigger"]["score"] >= 50 and s["room"] < cfg.setup.min_room_score:
        hits.append("TRIGGER_UNDER_BARRIER")
    if chosen == "LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION" and s["families"][chosen]["trigger_score"] < 40:
        hits.append("SWEEP_WITHOUT_FOLLOW_THROUGH")
    bc = s["break_ctx"]
    if bc is not None and bc["direction"] == s["dir_name"] and bc["false_break_risk_score"] >= 60 and \
            bc["state"] in ("CANDIDATE", "CONFIRMED", "ACCEPTED"):
        hits.append("HIGH_BREAKOUT_FAILURE_RISK")
    if s["extension_dir"] == s["d"] and s["extension_score"] >= cfg.setup.extension_chase_score:
        hits.append("EXTREME_EXTENSION")
    if s["pullback"]["pullback_state"] in ("FAILED_PULLBACK_CONTEXT", "STRUCTURE_THREATENING_PULLBACK") and \
            s["primary_dir"] == -s["d"]:
        hits.append("H1_STRUCTURE_BROKEN_AGAINST")
    prod = 1.0
    for h in hits:
        prod *= 1.0 - CONFLICT_WEIGHTS[h]
    return round(100.0 * (1.0 - prod), 2), hits


def quality_families(s: dict, conflict: float, cfg) -> dict:
    d = s["d"]
    fam = {
        "H4_CONTEXT": 0.6 * s["h4_side_score"] + 0.4 * s["h4_permission_confidence"],
        "H1_STRUCTURE": 0.35 * s["primary_aligned"] + 0.65 * max(s["immediate_aligned"], s["transition"]),
        "PULLBACK_QUALITY": s["pullback"]["pullback_quality"],
        "LOCATION": s["location"],
        "DISPLACEMENT": s["displacement"],
        "MOMENTUM": float(np.clip(50.0 + d * s["momentum_net"] / 2.0, 0, 100)),
        "LIQUIDITY_CONTEXT": float(np.clip(50.0 + 0.5 * (s["sweep_score_same"] - s["sweep_score_opp"]), 0, 100)),
        "MARKET_QUALITY": 0.5 * (100.0 - s["chop"]) + 0.25 * s["efficiency_pct"] + 0.25 * s["vol_quality"],
        "ROOM_TO_MOVE": s["room"],
        "CONFLICT": 100.0 - conflict,
    }
    return {k: round(float(np.clip(v, 0, 100)), 2) for k, v in fam.items()}


FAMILY_WEIGHT_KEYS = {"H4_CONTEXT": "w_h4_context", "H1_STRUCTURE": "w_h1_structure", "PULLBACK_QUALITY": "w_pullback",
                      "LOCATION": "w_location", "DISPLACEMENT": "w_displacement", "MOMENTUM": "w_momentum",
                      "LIQUIDITY_CONTEXT": "w_liquidity", "MARKET_QUALITY": "w_market_quality",
                      "ROOM_TO_MOVE": "w_room", "CONFLICT": "w_conflict"}


def score_side(fams: dict, family_score: float, conflict: float, cfg) -> tuple[float, float, float]:
    sc = cfg.scoring
    w = {k: getattr(sc, v) for k, v in FAMILY_WEIGHT_KEYS.items()}
    tot = sum(w.values())
    q = sum(fams[k] * wt for k, wt in w.items()) / tot
    blended = sc.family_blend * family_score + (1 - sc.family_blend) * q
    setup_score = round(float(np.clip(blended * (1.0 - sc.conflict_penalty * conflict / 100.0), 0, 100)), 2)
    consistency = sum(wt for k, wt in w.items() if fams[k] >= 50.0) / tot
    confidence = round(100.0 * consistency * (1.0 - conflict / 100.0), 2)
    return round(q, 2), setup_score, confidence


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@dataclass
class Setup:
    setup_id: str
    side: str
    family: str
    anchor: str
    created_index: int
    created_at: str
    start_close: float
    h4_regime_at_start: str | None
    state: str = "WATCHING"
    transitions: list = field(default_factory=list)  # (index, at, from, to, reason)
    peak_score: float = 0.0
    last_score: float = 0.0
    last_missing: list = field(default_factory=list)
    last_blockers: list = field(default_factory=list)
    qualified_index: int | None = None
    qualified: dict | None = None
    invalidation_reference: float | None = None
    end_index: int | None = None
    end_reason: str | None = None

    def state_at(self, c: int) -> str:
        st = "NO_SETUP"
        for idx, _, _, to, _ in self.transitions:
            if idx <= c:
                st = to
        return st

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transitions"] = [{"index": t[0], "at": t[1], "from": t[2], "to": t[3], "reason": t[4]} for t in self.transitions]
        return d


class SetupTracker:
    """One active setup per side; stable IDs; an anchor event can create at most one setup."""

    def __init__(self, side: str, cfg):
        self.side = side
        self.cfg = cfg
        self.active: Setup | None = None
        self.used_anchors: set = set()
        self.setups: list[Setup] = []
        self.counterfactual: list[dict] = []
        self.gated_seen: set = set()

    def _move(self, st: Setup, c: int, at: str, to: str, reason: str) -> None:
        st.transitions.append((c, at, st.state, to, reason))
        st.state = to

    def _close(self, st: Setup, c: int, at: str, to: str, reason: str) -> None:
        self._move(st, c, at, to, reason)
        st.end_index, st.end_reason = c, reason
        if st.qualified_index is None and st.peak_score >= self.cfg.setup.counterfactual_min_score:
            self.counterfactual.append({
                "kind": "FAILED_QUALIFICATION", "setup_id": st.setup_id, "side": st.side, "family": st.family,
                "anchor": st.anchor, "created_at": st.created_at, "ended_at": at, "end_state": to,
                "invalidation_or_expiry_reason": reason, "peak_setup_score": st.peak_score,
                "last_setup_score": st.last_score, "missing_requirements": list(st.last_missing),
                "blockers": list(st.last_blockers),
            })
        self.active = None

    def step(self, c: int, at: str, ctx: dict) -> dict:
        """``ctx``: permitted, blockers, families, chosen, setup_score, confidence, conflict, room, close, atr,
        h4_regime, pullback_state, opposing_displacement, break_state_for_anchor, qualified_payload (callable)."""
        cfg = self.cfg.setup
        events = []
        st = self.active
        if st is not None:
            fam = ctx["families"][st.family]
            reason = None
            if not ctx["permitted"]:
                reason, to = "H4 permission no longer includes this direction", "INVALIDATED"
            elif ctx["blockers"]:
                reason, to = "hard blocker: " + ",".join(ctx["blockers"]), "INVALIDATED"
            elif st.invalidation_reference is not None and \
                    ctx["d"] * (ctx["close"] - st.invalidation_reference) < 0:
                reason, to = "close beyond the invalidation reference (setup structure failed)", "INVALIDATED"
            elif st.family == "TREND_PULLBACK_CONTINUATION" and ctx["pullback_state"] == "FAILED_PULLBACK_CONTEXT":
                reason, to = "H1 pullback broke the impulse origin", "INVALIDATED"
            elif st.family == "BREAK_RETEST_CONTINUATION" and ctx["break_state"] in ("FAILED", "INVALIDATED"):
                reason, to = "breakout failed", "INVALIDATED"
            elif ctx["opposing_displacement"] >= cfg.opposing_displacement_invalidate:
                reason, to = "opposing displacement dominates", "INVALIDATED"
            elif ctx["room"] < cfg.min_room_score / 3.0:
                if st.qualified_index is None:
                    reason, to = "room to move disappeared", "INVALIDATED"
                else:
                    reason, to = "premise consumed: price reached the nearest opposing barrier", "EXPIRED"
            elif st.qualified_index is not None and c - st.qualified_index > cfg.qualified_expiry_bars:
                reason, to = f"qualified setup stale after {cfg.qualified_expiry_bars} bars", "EXPIRED"
            elif st.qualified_index is None and c - st.created_index > cfg.watch_expiry_bars:
                reason, to = f"not qualified within {cfg.watch_expiry_bars} bars", "EXPIRED"
            elif st.qualified_index is None and ctx["atr"] > 0 and \
                    ctx["d"] * (ctx["close"] - st.start_close) / ctx["atr"] > cfg.max_travel_atr:
                reason, to = f"price travelled > {cfg.max_travel_atr} ATR in the setup direction without it", "EXPIRED"
            elif ctx["h4_regime"] != st.h4_regime_at_start and st.qualified_index is None:
                reason, to = f"H4 regime changed ({st.h4_regime_at_start} -> {ctx['h4_regime']})", "EXPIRED"
            elif st.qualified_index is None and (not fam["eligible"] or fam["anchor"] != st.anchor):
                reason, to = "structural premise no longer present or superseded by a newer event", "EXPIRED"
            if reason is not None:
                self._close(st, c, at, to, reason)
                events.append(to)
                st = None
        if st is None and ctx["permitted"] and not ctx["blockers"] and ctx["chosen"] is not None:
            fam = ctx["families"][ctx["chosen"]]
            key = (ctx["chosen"], fam["anchor"])
            if key not in self.used_anchors:
                self.used_anchors.add(key)
                sid = f"H1-{self.side.upper()}-{FAMILY_SHORT[ctx['chosen']]}-{fam['anchor']}-{c}"
                st = Setup(setup_id=sid, side=self.side, family=ctx["chosen"], anchor=fam["anchor"], created_index=c,
                           created_at=at, start_close=ctx["close"], h4_regime_at_start=ctx["h4_regime"],
                           invalidation_reference=fam["invalidation_reference"])
                st.transitions.append((c, at, "NO_SETUP", "WATCHING", f"{ctx['chosen']} premise present ({fam['anchor']})"))
                self.setups.append(st)
                self.active = st
                events.append("WATCHING")
        elif not ctx["permitted"] and ctx["chosen"] is not None:
            # research: a promising setup that H4 did not permit (never becomes a setup, logged once per anchor)
            fam = ctx["families"][ctx["chosen"]]
            key = (ctx["chosen"], fam["anchor"])
            if ctx["setup_score"] >= cfg.counterfactual_min_score and key not in self.gated_seen:
                self.gated_seen.add(key)
                self.counterfactual.append({
                    "kind": "GATED_BY_H4", "side": self.side, "family": ctx["chosen"], "anchor": fam["anchor"],
                    "at": at, "setup_score": ctx["setup_score"], "setup_confidence": ctx["confidence"],
                    "h4_permission": ctx["h4_permission"], "blockers": list(ctx["blockers"]),
                    "missing_requirements": ["H4 permission does not include this direction"] + list(ctx["missing"]),
                })
        missing = []
        if st is not None:
            fam = ctx["families"][st.family]
            score = ctx["family_setup_score"](st.family)
            st.last_score = score
            st.peak_score = max(st.peak_score, score)
            st.last_blockers = list(ctx["blockers"])
            if st.qualified_index is None:
                missing = qualification_gaps(fam, score, ctx, cfg)
                st.last_missing = missing
                if st.state == "WATCHING" and score >= cfg.developing_score:
                    self._move(st, c, at, "DEVELOPING", f"setup score {score:.1f} >= {cfg.developing_score}")
                    events.append("DEVELOPING")
                if not missing:
                    st.qualified_index = c
                    if fam.get("qualified_reference") is not None:
                        st.invalidation_reference = fam["qualified_reference"]
                    st.qualified = ctx["qualified_payload"](st, score)
                    self._move(st, c, at, "QUALIFIED", "all qualification requirements met")
                    events.append("QUALIFIED")
        return {"setup_id": st.setup_id if st else None, "setup_state": st.state if st else "NO_SETUP",
                "setup_family": st.family if st else None, "events": events, "missing_requirements": missing,
                "setup": st}


def qualification_gaps(fam: dict, score: float, ctx: dict, cfg) -> list[str]:
    gaps = []
    if not fam["eligible"]:
        gaps += fam["missing"] or ["family premise not present"]
    if score < cfg.qualify_score:
        gaps.append(f"setup_score {score:.1f} < {cfg.qualify_score}")
    if ctx["confidence"] < cfg.qualify_min_confidence:
        gaps.append(f"setup_confidence {ctx['confidence']:.1f} < {cfg.qualify_min_confidence}")
    if ctx["conflict"] > cfg.max_conflict:
        gaps.append(f"h1_setup_conflict_score {ctx['conflict']:.1f} > {cfg.max_conflict}")
    if ctx["room"] < cfg.min_room_score:
        gaps.append(f"room_score {ctx['room']:.1f} < {cfg.min_room_score}")
    if fam["trigger_score"] < cfg.min_trigger_score:
        gaps.append(f"trigger ({fam['trigger']}) {fam['trigger_score']:.1f} < {cfg.min_trigger_score}")
    if ctx["blockers"]:
        gaps.append("blockers: " + ",".join(ctx["blockers"]))
    if not ctx["permitted"]:
        gaps.append("H4 permission does not include this direction")
    return gaps


__all__ = ["FAMILIES", "Setup", "SetupTracker", "families_for_side", "quality_families", "score_side",
           "setup_conflict"]
