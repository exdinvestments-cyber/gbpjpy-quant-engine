"""Phase 1C: setup families, conflict, quality families, scores, lifecycle, invalidation, expiry, duplicate
protection, H4 permission gating, hard blockers, qualified output and counterfactual logging."""

from __future__ import annotations

import json

import pytest

from gbpjpy_engine.h1 import H1Config, explain_h1_bar, explain_h1_setup
from gbpjpy_engine.h1.setups import (FAMILIES, SetupTracker, families_for_side, quality_families, score_side,
                                     setup_conflict)
from gbpjpy_engine.reason_codes import ReasonCode

CFG = H1Config()
VALID = {c.value for c in ReasonCode}


def side_inputs(**over):
    pb = {"pullback_state": "HEALTHY_PULLBACK", "pullback_quality": 75.0, "counter_momentum_deterioration": 0.7,
          "impulse_leg": "10-11", "impulse_origin": 100.0, "correction_extreme": 101.5}
    s = {"d": 1, "dir_name": "bullish", "atr": 0.2, "pullback": pb, "trigger": {"score": 65.0, "source": "transition"},
         "location": 60.0, "room": 70.0, "break_ctx": None, "rejection": 40.0, "displacement": 55.0, "transition": 65.0,
         "reclaim": 0.0, "sweep": None, "sweep_extreme": None, "expansion": None, "compression_extreme": None,
         "extension_score": 40.0, "extension_dir": 1, "opposing_displacement": 5.0, "chop": 30.0, "primary_dir": 1,
         "h4_side_score": 70.0, "h4_permission_confidence": 75.0, "primary_aligned": 80.0, "immediate_aligned": 70.0,
         "momentum_net": 30.0, "sweep_score_same": 0.0, "sweep_score_opp": 0.0, "efficiency_pct": 60.0, "vol_quality": 100.0}
    for k, v in over.items():
        if k == "pullback":
            s["pullback"] = {**pb, **v}
        else:
            s[k] = v
    return s


# ------------------------------------------------------------------ families
def test_trend_pullback_family():
    f = families_for_side("long", side_inputs(), CFG)
    tpc = f["TREND_PULLBACK_CONTINUATION"]
    assert tpc["eligible"] and tpc["anchor"] == "L10-11" and tpc["invalidation_reference"] == 100.0
    assert tpc["qualified_reference"] == 101.5 and 0 <= tpc["family_score"] <= 100
    bad = families_for_side("long", side_inputs(pullback={"pullback_state": "FAILED_PULLBACK_CONTEXT"}), CFG)
    assert not bad["TREND_PULLBACK_CONTINUATION"]["eligible"] and bad["TREND_PULLBACK_CONTINUATION"]["missing"]


def _bc(state="CONFIRMED", evidence=("successful_retest",), bars=4, risk=20.0):
    return {"direction": "bullish", "bars_since_break": bars, "state": state, "acceptance_evidence": list(evidence),
            "acceptance_state": "ACCEPTING", "breakout_quality_score": 70.0, "event_id": 5, "level": 102.0,
            "false_break_risk_score": risk}


def test_break_retest_family_distinguishes_retest_from_failure():
    f = families_for_side("long", side_inputs(break_ctx=_bc(), rejection=60.0), CFG)["BREAK_RETEST_CONTINUATION"]
    assert f["eligible"] and f["anchor"] == "B5" and f["invalidation_reference"] == pytest.approx(102.0 - 0.25 * 0.2)
    no_retest = families_for_side("long", side_inputs(break_ctx=_bc(evidence=())), CFG)["BREAK_RETEST_CONTINUATION"]
    assert not no_retest["eligible"] and "retest" in no_retest["missing"][0]
    failed = families_for_side("long", side_inputs(break_ctx=_bc(state="FAILED")), CFG)["BREAK_RETEST_CONTINUATION"]
    assert not failed["eligible"]
    old = families_for_side("long", side_inputs(break_ctx=_bc(bars=40)), CFG)["BREAK_RETEST_CONTINUATION"]
    assert not old["eligible"]


def test_sweep_family():
    sw = {"bars_since": 3, "state": "SWEEP_AND_REJECT", "sweep_score": 70.0, "event_id": 9}
    f = families_for_side("long", side_inputs(sweep=sw, sweep_extreme=100.8), CFG)
    fam = f["LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"]
    assert fam["eligible"] and fam["anchor"] == "S9" and fam["invalidation_reference"] == 100.8
    accepted = families_for_side("long", side_inputs(sweep={**sw, "state": "BREAK_AND_ACCEPT"}), CFG)
    assert not accepted["LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"]["eligible"]


def test_compression_expansion_family_avoids_chasing():
    ex = {"event_id": 2, "direction": "bullish", "bars_since": 2, "state": "EXPANDING", "structural_break": True,
          "expansion_quality": 70.0}
    ok = families_for_side("long", side_inputs(expansion=ex, compression_extreme=100.5), CFG)
    assert ok["COMPRESSION_EXPANSION_IN_H4_DIRECTION"]["eligible"]
    chase = families_for_side("long", side_inputs(expansion=ex, extension_score=95.0), CFG)
    fam = chase["COMPRESSION_EXPANSION_IN_H4_DIRECTION"]
    assert not fam["eligible"] and any("chasing" in m for m in fam["missing"])
    wrong_dir = families_for_side("long", side_inputs(expansion={**ex, "direction": "bearish"}), CFG)
    assert not wrong_dir["COMPRESSION_EXPANSION_IN_H4_DIRECTION"]["eligible"]


# ------------------------------------------------------------------ conflict / quality / score
def test_setup_conflict_components():
    s = side_inputs(opposing_displacement=80.0, chop=65.0, room=10.0, extension_score=95.0,
                    break_ctx=_bc(risk=75.0), pullback={"pullback_state": "FAILED_PULLBACK_CONTEXT"}, primary_dir=-1)
    s["families"] = families_for_side("long", s, CFG)
    score, hits = setup_conflict("long", s, None, CFG)
    assert set(hits) == {"H1_STRONGLY_OPPOSED", "H1_HIGH_CHOP", "TRIGGER_UNDER_BARRIER", "EXTREME_EXTENSION",
                         "HIGH_BREAKOUT_FAILURE_RISK", "H1_STRUCTURE_BROKEN_AGAINST"}
    import numpy as np

    from gbpjpy_engine.h1.setups import CONFLICT_WEIGHTS

    assert score == pytest.approx(100 * (1 - np.prod([1 - CONFLICT_WEIGHTS[h] for h in hits])), abs=0.01)
    assert {f"H1_CONFLICT_{h}" for h in hits} <= VALID
    clean = side_inputs()
    clean["families"] = families_for_side("long", clean, CFG)
    assert setup_conflict("long", clean, "TREND_PULLBACK_CONTINUATION", CFG) == (0.0, [])


def test_quality_families_and_conflict_reduce_score():
    s = side_inputs()
    fams = quality_families(s, 0.0, CFG)
    assert set(fams) == {"H4_CONTEXT", "H1_STRUCTURE", "PULLBACK_QUALITY", "LOCATION", "DISPLACEMENT", "MOMENTUM",
                         "LIQUIDITY_CONTEXT", "MARKET_QUALITY", "ROOM_TO_MOVE", "CONFLICT"}
    q0, score0, conf0 = score_side(fams, 70.0, 0.0, CFG)
    q1, score1, conf1 = score_side(quality_families(s, 60.0, CFG), 70.0, 60.0, CFG)
    assert score1 < score0 and conf1 < conf0 and 0 <= conf1 <= 100
    # saturating one family cannot move the quality score by more than that family's weight
    sat = dict(fams, DISPLACEMENT=100.0)
    low = dict(fams, DISPLACEMENT=0.0)
    assert score_side(sat, 0, 0, CFG)[0] - score_side(low, 0, 0, CFG)[0] <= 100 * CFG.scoring.w_displacement / 1.0 + 1e-9


# ------------------------------------------------------------------ lifecycle
def fams_with(anchor="L1-2", eligible=True, trigger=70.0, ref=100.0, qref=101.0, family="TREND_PULLBACK_CONTINUATION"):
    out = {f: {"eligible": False, "family_score": 10.0, "trigger_score": 0.0, "trigger": "x", "anchor": None,
               "invalidation_reference": None, "missing": ["n/a"]} for f in FAMILIES}
    out[family] = {"eligible": eligible, "family_score": 70.0, "trigger_score": trigger, "trigger": "transition",
                   "anchor": anchor, "invalidation_reference": ref, "qualified_reference": qref, "missing": []}
    return out


def tctx(score=70.0, permitted=True, blockers=(), close=102.0, anchor="L1-2", eligible=True, trigger=70.0, room=70.0,
         conf=70.0, conflict=0.0, regime="BULL_TREND", opp=0.0, pb="HEALTHY_PULLBACK", atr=0.2):
    fams = fams_with(anchor=anchor, eligible=eligible, trigger=trigger)
    return {"permitted": permitted, "blockers": list(blockers), "families": fams,
            "chosen": "TREND_PULLBACK_CONTINUATION" if eligible else None, "setup_score": score, "confidence": conf,
            "conflict": conflict, "room": room, "close": close, "atr": atr, "d": 1, "h4_regime": regime,
            "pullback_state": pb, "break_state": None, "opposing_displacement": opp, "h4_permission": "ALLOW_LONG",
            "missing": [], "family_setup_score": lambda f, s=score: s,
            "qualified_payload": lambda st, sc: {"setup_id": st.setup_id, "setup_score": sc}}


def test_lifecycle_watching_developing_qualified_and_stable_id():
    tr = SetupTracker("long", CFG)
    r0 = tr.step(0, "t0", tctx(score=40.0, trigger=20.0))
    assert r0["setup_state"] == "WATCHING" and r0["setup_id"] == "H1-LONG-TPC-L1-2-0"
    r1 = tr.step(1, "t1", tctx(score=50.0, trigger=20.0))
    assert r1["setup_state"] == "DEVELOPING" and r1["setup_id"] == r0["setup_id"]
    assert any("trigger" in m for m in r1["missing_requirements"])
    r2 = tr.step(2, "t2", tctx(score=65.0))
    assert r2["setup_state"] == "QUALIFIED" and "QUALIFIED" in r2["events"]
    st = tr.setups[0]
    assert st.qualified == {"setup_id": "H1-LONG-TPC-L1-2-0", "setup_score": 65.0}
    assert st.invalidation_reference == 101.0  # frozen to the correction extreme at qualification
    assert [t[3] for t in st.transitions] == ["WATCHING", "DEVELOPING", "QUALIFIED"]
    assert st.state_at(1) == "DEVELOPING" and st.state_at(0) == "WATCHING"  # history kept
    r3 = tr.step(3, "t3", tctx(score=66.0))
    assert r3["setup_state"] == "QUALIFIED" and r3["events"] == []  # no duplicate qualification


@pytest.mark.parametrize("over,reason", [
    ({"permitted": False}, "H4 permission"),
    ({"blockers": ["SEVERE_H1_CHOP"]}, "hard blocker"),
    ({"close": 99.0}, "invalidation reference"),
    ({"pb": "FAILED_PULLBACK_CONTEXT"}, "impulse origin"),
    ({"opp": 80.0}, "opposing displacement"),
    ({"room": 5.0}, "room to move disappeared"),
])
def test_invalidation_reasons(over, reason):
    tr = SetupTracker("long", CFG)
    tr.step(0, "t0", tctx(score=40.0, trigger=10.0))
    r = tr.step(1, "t1", tctx(score=40.0, trigger=10.0, **over))
    st = tr.setups[0]
    assert st.state == "INVALIDATED" and reason in st.end_reason and "INVALIDATED" in r["events"]


def test_expiry_rules_and_duplicate_protection():
    sc = CFG.setup
    tr = SetupTracker("long", CFG)
    tr.step(0, "t0", tctx(score=40.0, trigger=10.0))
    for c in range(1, sc.watch_expiry_bars + 2):
        tr.step(c, f"t{c}", tctx(score=40.0, trigger=10.0))
    assert tr.setups[0].state == "EXPIRED" and "not qualified within" in tr.setups[0].end_reason
    # the same anchor can never create a second setup
    assert len(tr.setups) == 1 and tr.active is None
    tr.step(100, "t100", tctx(score=40.0, trigger=10.0))
    assert len(tr.setups) == 1
    # a new structural anchor creates a new, distinct setup id
    tr.step(101, "t101", tctx(score=40.0, trigger=10.0, anchor="L3-4"))
    assert len(tr.setups) == 2 and tr.setups[1].setup_id != tr.setups[0].setup_id
    # travel, regime change and supersession expire too
    for over, why in (({"close": 102.0 + 0.2 * 4}, "travelled"), ({"regime": "RANGE"}, "regime changed"),
                      ({"anchor": "L9-9"}, "superseded")):
        t = SetupTracker("long", CFG)
        t.step(0, "t0", tctx(score=40.0, trigger=10.0))
        t.step(1, "t1", tctx(score=40.0, trigger=10.0, **over))
        assert t.setups[0].state == "EXPIRED" and why in t.setups[0].end_reason


def test_qualified_setup_expires_when_stale():
    tr = SetupTracker("long", CFG)
    tr.step(0, "t0", tctx())
    assert tr.setups[0].state == "QUALIFIED"
    for c in range(1, CFG.setup.qualified_expiry_bars + 2):
        tr.step(c, f"t{c}", tctx())
    assert tr.setups[0].state == "EXPIRED" and "stale" in tr.setups[0].end_reason


def test_gating_and_counterfactual_logging():
    tr = SetupTracker("long", CFG)
    r = tr.step(0, "t0", tctx(permitted=False, score=70.0))
    assert r["setup_state"] == "NO_SETUP" and tr.setups == []
    assert tr.counterfactual[0]["kind"] == "GATED_BY_H4"
    assert "H4 permission does not include this direction" in tr.counterfactual[0]["missing_requirements"]
    tr.step(1, "t1", tctx(permitted=False, score=70.0))
    assert len(tr.counterfactual) == 1  # logged once per anchor
    tr2 = SetupTracker("long", CFG)
    tr2.step(0, "t0", tctx(score=55.0, trigger=20.0))
    tr2.step(1, "t1", tctx(score=55.0, trigger=20.0, close=99.0))
    cf = tr2.counterfactual[0]
    assert cf["kind"] == "FAILED_QUALIFICATION" and cf["peak_setup_score"] == 55.0
    assert cf["invalidation_or_expiry_reason"].startswith("close beyond") and cf["missing_requirements"]
    blocked = SetupTracker("long", CFG)
    assert blocked.step(0, "t0", tctx(blockers=["H4_BLOCK_ALL"]))["setup_state"] == "NO_SETUP"


# ------------------------------------------------------------------ engine-level gating & outputs
def test_h4_permission_gates_h1(h1_results):
    for name, res_t in h1_results.items():
        sc, h4, res = res_t
        f = res.frame
        long_ok = f["h4_permission"].isin(["ALLOW_LONG", "ALLOW_BOTH"]) & (f["h4_context_status"] == "OK")
        short_ok = f["h4_permission"].isin(["ALLOW_SHORT", "ALLOW_BOTH"]) & (f["h4_context_status"] == "OK")
        assert not (f["actionable_setup_permission_long"] & ~long_ok).any(), name
        assert not (f["actionable_setup_permission_short"] & ~short_ok).any(), name
        blocked = f["h1_blockers"].map(bool)
        assert not (f["actionable_setup_permission_long"] & blocked).any()
        for q in res.qualified_setups:
            i = q["bar_index"]
            assert f["h1_blockers"].iloc[i] == []
            want = "ALLOW_LONG" if q["direction"] == "LONG" else "ALLOW_SHORT"
            assert f["h4_permission"].iloc[i] in (want, "ALLOW_BOTH")
        block_all = (f["h4_permission"] == "BLOCK_ALL") & (f["h4_context_status"] == "OK")
        assert all("H4_BLOCK_ALL" in b for b in f.loc[block_all, "h1_blockers"])
        not_ok = f["h4_context_status"] != "OK"
        assert all(("STALE_H4_CONTEXT" in b) or ("NO_H4_CONTEXT" in b) for b in f.loc[not_ok, "h1_blockers"])


def test_scenario_setups(h1_results):
    sc, _, up = h1_results["h1_uptrend_pullbacks"]
    assert up.qualified_setups and all(q["direction"] == "LONG" for q in up.qualified_setups)
    assert all(s.side == "long" for s in up.setups)
    sc, _, dn = h1_results["h1_downtrend_pullbacks"]
    assert dn.qualified_setups and all(q["direction"] == "SHORT" for q in dn.qualified_setups)
    _, _, rg = h1_results["h1_range"]
    assert rg.qualified_setups == [] and rg.setups == []
    assert rg.counterfactual and all(c["kind"] == "GATED_BY_H4" for c in rg.counterfactual)


FORBIDDEN_KEYS = {"entry", "entry_price", "stop", "stop_loss", "take_profit", "target", "position_size", "size",
                  "lots", "volume", "order"}


def test_qualified_output_structure_and_audit(h1_results):
    _, _, res = h1_results["h1_uptrend_pullbacks"]
    q = res.qualified_setups[0]
    for k in ("setup_id", "timestamp", "direction", "setup_family", "h4_permission", "h4_permission_confidence",
              "h4_context_score", "h1_primary_structure", "h1_immediate_structure", "pullback_state", "pullback_quality",
              "location_score", "displacement_score", "transition_score", "rejection_score", "liquidity_context",
              "market_quality", "chop_score", "room_score", "conflict_score", "setup_score", "setup_confidence",
              "invalidation_reference"):
        assert k in q, k
    assert not FORBIDDEN_KEYS & set(q)
    json.dumps(q)
    st = next(s for s in res.setups if s.setup_id == q["setup_id"])
    assert st.qualified == q
    audit = res.explain_setup(q["setup_id"])
    codes = audit["qualification_explanation"]["reason_codes"]
    assert "SETUP_QUALIFIED" in codes and "H4_LONG_PERMISSION" in codes and set(codes) <= VALID
    txt = explain_h1_setup(res, q["setup_id"])
    assert "WHY IT QUALIFIED" in txt and "not a trade" in txt
    assert "H1 bar" in explain_h1_bar(res, q["bar_index"])


def test_every_bar_is_explained_with_valid_codes(h1_results):
    _, _, res = h1_results["h1_downtrend_pullbacks"]
    for i in range(0, len(res.frame), 23):
        assert set(res.frame["reason_codes"].iloc[i]) <= VALID
        for side in ("long", "short"):
            e = res.explain(i, side)
            assert "state" in e and "blockers" in e
            if e["state"] not in ("NO_SETUP", "BLOCKED", "QUALIFIED"):
                assert e["missing_requirements"]


def test_rejected_setups_explain_failure(h1_results):
    _, _, res = h1_results["h1_uptrend_pullbacks"]
    ended = [s for s in res.setups if s.state in ("INVALIDATED", "EXPIRED") and s.qualified_index is None]
    assert ended
    for s in ended:
        assert s.end_reason
        audit = res.explain_setup(s.setup_id)
        assert audit["end_explanation"]["reason_codes"]
    cf = [c for c in res.counterfactual if c["kind"] == "FAILED_QUALIFICATION"]
    for c in cf:
        assert c["missing_requirements"] and c["invalidation_or_expiry_reason"]


def test_hard_blockers_cannot_be_overridden(h1_results):
    for _, _, res in h1_results.values():
        f = res.frame
        blocked = f["h1_blockers"].map(bool)
        assert not f.loc[blocked, "actionable_setup_permission_long"].any()
        assert not f.loc[blocked, "actionable_setup_permission_short"].any()
        # no setup ever becomes QUALIFIED on a bar that carries a hard blocker
        assert not any("SETUP_QUALIFIED" in codes for codes in f.loc[blocked, "reason_codes"])
        assert all(f["h1_blockers"].iloc[q["bar_index"]] == [] for q in res.qualified_setups)


def test_fail_safe_exception_blocks(h1_results, h1_engine):
    sc, h4, _ = h1_results["h1_uptrend_pullbacks"]
    res = h1_engine.run(sc.h1.iloc[:1500].reset_index(drop=True), h4, _fault_at=1400)
    f = res.frame
    assert res.errors and res.errors[0]["index"] == 1400
    assert not f["actionable_setup_permission_long"].iloc[1400:].any()
    assert all(b == ["H1_CONTEXT_ERROR"] for b in f["h1_blockers"].iloc[1400:])
    assert all(q["bar_index"] < 1400 for q in res.qualified_setups)
