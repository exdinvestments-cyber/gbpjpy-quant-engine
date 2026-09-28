"""Phase 1B: displacement, breakout quality, acceptance vs rejection, failed breaks, false-break risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ctx_helpers import T0, swing
from gbpjpy_engine.config import BreakoutContextConfig, DisplacementConfig
from gbpjpy_engine.context.breakouts import BreakoutAnalyzer, IntrabarEvidence
from gbpjpy_engine.context.displacement import displacement_features
from gbpjpy_engine.features.structure import BreakEvent, BreakTransition

TF = pd.Timedelta(hours=4)


def _disp_frame(o, h, l, c, atr=0.5):
    n = len(c)
    o, h, l, c = map(lambda x: np.asarray(x, float), (o, h, l, c))
    rng = np.where(h > l, h - l, 1e-9)
    return pd.DataFrame({
        "open": o, "high": h, "low": l, "close": c, "atr": np.full(n, atr),
        "overlap_prev": np.full(n, 0.2), "close_location": (c - l) / rng,
        "bars_since_break": np.full(n, np.nan), "last_break_direction": [None] * n,
    })


def _flat(n=40):
    c = np.full(n, 100.0)
    return list(c), list(c + 0.05), list(c - 0.05), list(c)


def test_single_large_candle_is_not_strong_displacement():
    o, h, l, c = _flat()
    o[30], c[30], h[30], l[30] = 100.0, 102.0, 102.05, 99.95  # one 4-ATR candle, then flat
    for k in range(31, 40):
        o[k] = c[k] = 102.0
        h[k], l[k] = 102.05, 101.95
    d = displacement_features(_disp_frame(o, h, l, c), DisplacementConfig())
    assert d["bullish_displacement_score"].max() < DisplacementConfig().strong_score
    assert not d["bullish_displacement_multibar"].any()
    assert d["bullish_displacement_class"].iloc[30] in ("MODERATE", "WEAK")


def test_multi_candle_displacement_is_strong_and_symmetric():
    o, h, l, c = _flat()
    for k, lvl in zip(range(30, 33), (100.0, 100.6, 101.2)):
        o[k], c[k], h[k], l[k] = lvl, lvl + 0.6, lvl + 0.62, lvl - 0.02
    d = displacement_features(_disp_frame(o, h, l, c), DisplacementConfig())
    assert d["bullish_displacement_score"].iloc[32] >= DisplacementConfig().strong_score
    assert d["bullish_displacement_class"].iloc[32] in ("STRONG", "EXTREME")
    assert d["bearish_displacement_score"].iloc[32] == 0 or d["bearish_displacement_score"].iloc[32] < 10
    # mirror image -> bearish
    o2, h2, l2, c2 = [200 - x for x in o], [200 - x for x in l], [200 - x for x in h], [200 - x for x in c]
    d2 = displacement_features(_disp_frame(o2, h2, l2, c2), DisplacementConfig())
    assert d2["bearish_displacement_score"].iloc[32] == pytest.approx(d["bullish_displacement_score"].iloc[32], abs=1e-6)


def _event(b, close, level=100.0, direction="bullish", mag=None, atr=1.0):
    t = T0 + TF * b
    mag = (close - level) if mag is None else mag
    return BreakEvent(
        event_id=0, direction=direction, break_type="BOS", level=level, swing_id=0,
        level_occurred_at=T0, level_confirmed_at=T0 + TF * 3, break_index=b, break_time=t, available_at=t + TF,
        close=close, magnitude=mag, magnitude_atr=mag / atr, atr_at_break=atr, wick_beyond_atr=0.3,
        closed_beyond=True, structure_before="bullish",
    )


def _arrays(o, h, l, c, n_disp=None):
    n = len(c)
    return {"open": np.asarray(o, float), "high": np.asarray(h, float), "low": np.asarray(l, float),
            "close": np.asarray(c, float), "atr": np.ones(n), "bullish_disp": np.asarray(n_disp or [0.0] * n, float),
            "bearish_disp": np.zeros(n), "disp_eff": np.full(n, 0.7), "chop": np.full(n, 30.0), "eff": np.full(n, 0.5),
            "ts": [T0 + TF * i for i in range(n)], "tf": TF}


def _analyzer(arrays, n):
    an = BreakoutAnalyzer(arrays, [swing(0, "high", 0, 100.0, sig=2.5)], [[] for _ in range(n)],
                          BreakoutContextConfig(), 60.0, 60)
    an.set_views([], [])
    return an


def _tr(ev, idx, frm, to, why="t"):
    ev.transitions.append(BreakTransition(idx, T0 + TF * (idx + 1), frm, to, why))


def test_accepting_break_with_retest_and_follow_through():
    c = [99.0, 99.3, 99.5, 99.7, 99.8, 100.9, 101.2, 101.4, 101.1, 101.8, 102.2, 102.5, 102.9]
    o = [c[0]] + c[:-1]
    h = [x + 0.1 for x in c]
    l = [min(a_, b_) - 0.1 for a_, b_ in zip(o, c)]
    l[8] = 100.15  # retest of the broken level, closing back above it
    arr = _arrays(o, h, l, c, [0, 0, 0, 0, 0, 70, 65, 50, 20, 40, 45, 50, 55])
    ev = _event(5, 100.9)
    _tr(ev, 5, None, "CANDIDATE")
    _tr(ev, 7, "CANDIDATE", "CONFIRMED")
    _tr(ev, 11, "CONFIRMED", "ACCEPTED")
    an = _analyzer(arr, len(c))
    assert an.evaluate(ev, 5).acceptance_state == "UNRESOLVED"  # never decided on the break bar itself
    ctx = an.evaluate(ev, 12)
    assert ctx.acceptance_state == "ACCEPTING"
    assert {"multiple_closes_beyond_level", "successful_retest", "continued_displacement"} <= set(ctx.acceptance_evidence)
    assert ctx.state == "ACCEPTED" and ctx.accepted_at is not None and ctx.confirmed_at is not None
    assert ctx.max_excursion_atr == pytest.approx(2.9)
    assert ctx.failure_score == 0.0
    assert 0 <= ctx.breakout_quality_score <= 100
    # point-in-time: at bar 9 the lifecycle was only CONFIRMED and acceptance not yet at its later value
    early = an.evaluate(ev, 9)
    assert early.state == "CONFIRMED" and early.accepted_at is None


def test_rejected_break_scores_failure_and_false_break_risk():
    c = [99.0, 99.3, 99.5, 99.7, 99.8, 100.3, 99.5, 99.2]
    o = [c[0]] + c[:-1]
    h = [x + 0.1 for x in c]
    h[5] = 101.2  # large upper rejection wick on the break bar
    l = [min(a_, b_) - 0.1 for a_, b_ in zip(o, c)]
    arr = _arrays(o, h, l, c)
    arr["bearish_disp"][6] = 65.0
    ev = _event(5, 100.3)
    _tr(ev, 5, None, "CANDIDATE")
    _tr(ev, 6, "CANDIDATE", "INVALIDATED")
    an = _analyzer(arr, len(c))
    live = an.evaluate(ev, 5)
    assert live.false_break_risk_score >= 40
    assert {"weak_penetration", "rejection_wick", "weak_displacement"} <= set(live.false_break_risk_reasons)
    assert live.breakout_quality_class in ("WEAK_BREAK", "FALSE_BREAK_CANDIDATE")
    ctx = an.evaluate(ev, 6)
    assert ctx.acceptance_state == "REJECTING"
    assert {"close_back_inside", "opposing_displacement", "lifecycle_invalidated"} <= set(ctx.rejection_evidence)
    assert ctx.failure_score > 50
    assert ctx.false_break_risk_score == 100.0 and ctx.false_break_risk_reasons == ["break_already_invalidated"]


def test_intrabar_extension_point():
    c = [99.0, 99.3, 99.5, 99.7, 99.8, 100.6, 100.8]
    o = [c[0]] + c[:-1]
    arr = _arrays(o, [x + 0.1 for x in c], [x - 0.3 for x in c], c)
    ev = _event(5, 100.6)
    _tr(ev, 5, None, "CANDIDATE")
    an = _analyzer(arr, len(c))
    base = an.evaluate(ev, 6)
    assert base.intrabar_source == "none"
    ib = an.evaluate(ev, 6, IntrabarEvidence(traded_back_through_level=True, source="future-m15-provider"))
    assert "intrabar_trade_back_through_level" in ib.rejection_evidence and ib.intrabar_source == "future-m15-provider"
    assert ib.state == base.state  # completed-bar lifecycle unchanged by optional evidence


def test_false_breakout_scenario_context(scenario_results):
    sc, res = scenario_results["false_breakout"]
    k = sc.events["break_index"]
    cf = res.context.frame
    assert cf["breakout_state"].iloc[k] == "CANDIDATE"
    assert cf["acceptance_state"].iloc[k] == "UNRESOLVED"
    assert cf["breakout_state"].iloc[k + 1] == "INVALIDATED"
    assert cf["acceptance_state"].iloc[k + 1] == "REJECTING"
    assert cf["failure_score"].iloc[k + 1] > 0
    assert "BULLISH_FAILED_BREAK" in cf["context_reason_codes"].iloc[k + 1]
    rec = [r for r in res.context.failed_breaks if r["failed_index"] == k + 1]
    assert rec and rec[0]["attempted_direction"] == "bullish" and rec[0]["failure_score"] > 0
    for key in ("level", "max_excursion_atr", "time_beyond_level_bars", "close_behaviour_atr", "structural_consequence"):
        assert key in rec[0]


def test_late_failure_recognised_without_rewriting_history(rw_result):
    cf = rw_result.context.frame
    found = 0
    for ev in rw_result.structure.breaks:
        states = [t.to_state for t in ev.transitions]
        if states[:2] != ["CANDIDATE", "CONFIRMED"] or "FAILED" not in states:
            continue
        conf_i = ev.transitions[1].index
        fail_i = next(t.index for t in ev.transitions if t.to_state == "FAILED")
        if cf["breakout_event_id"].iloc[fail_i] != ev.event_id or cf["breakout_event_id"].iloc[conf_i] != ev.event_id:
            continue
        found += 1
        assert cf["breakout_state"].iloc[conf_i] == "CONFIRMED"  # what was known then
        assert cf["breakout_state"].iloc[fail_i] == "FAILED"
        det_then = rw_result.context.details[conf_i]["breakout"]
        assert det_then["failed_at"] is None
        assert rw_result.context.details[fail_i]["breakout"]["failed_at"] is not None
        assert any(r["event_id"] == ev.event_id and r["failed_index"] == fail_i for r in rw_result.context.failed_breaks)
    assert found >= 1


def test_breakout_quality_and_acceptance_on_uptrend(scenario_results):
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    q = cf["breakout_quality_score"].dropna()
    assert len(q) and q.between(0, 100).all()
    assert (cf["acceptance_state"] == "ACCEPTING").any()
    assert cf["context_reason_codes"].map(lambda c: "BULLISH_BREAK_ACCEPTED" in c).any()
    assert not cf["context_reason_codes"].map(lambda c: "BEARISH_BREAK_ACCEPTED" in c).any()
    assert set(cf["breakout_quality_class"].dropna()) <= {"HIGH_QUALITY_BREAK", "MODERATE_BREAK", "WEAK_BREAK", "FALSE_BREAK_CANDIDATE"}
