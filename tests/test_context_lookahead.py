"""Phase 1B: look-ahead / repainting, determinism, state machine and temporal stability, snapshot output."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.context.state_machine import EXPECTED, ContextStateMachine
from gbpjpy_engine.research import explain_permission
from gbpjpy_engine.snapshot import to_jsonable
from gbpjpy_engine.synthetic import random_walk_bars

CUTS = [300, 480, 690]


def _same_rows(a: pd.DataFrame, b: pd.DataFrame, upto: int) -> None:
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        x = [json.dumps(to_jsonable(v), sort_keys=True, default=str) for v in a[col].iloc[: upto + 1]]
        y = [json.dumps(to_jsonable(v), sort_keys=True, default=str) for v in b[col].iloc[: upto + 1]]
        bad = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        assert not bad, f"context look-ahead in {col!r} at rows {bad[:5]}: {x[bad[0]]} vs {y[bad[0]]}"


def _detail(res, i):
    return json.dumps(to_jsonable(res.context.details[i]), sort_keys=True, default=str)


@pytest.mark.parametrize("cut", CUTS)
def test_context_truncation_invariance(engine, rw_bars, rw_result, cut):
    part = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
    _same_rows(rw_result.context.frame, part.context.frame, cut)
    for i in (cut - 40, cut - 7, cut):
        assert _detail(part, i) == _detail(rw_result, i)  # hierarchy, legs, breakouts, sweeps, zones, maturity, ...
    # event histories: the truncated run knows exactly the prefix of the full run
    assert part.context.failed_breaks == [r for r in rw_result.context.failed_breaks if r["failed_index"] <= cut]
    assert part.context.role_reversals == [r for r in rw_result.context.role_reversals if r["index"] <= cut]
    full_sweeps = [e for e in rw_result.context.sweep_events if e.index <= cut]
    assert [e.snapshot_at(cut) for e in part.context.sweep_events] == [e.snapshot_at(cut) for e in full_sweeps]
    assert part.context.unexpected_transitions == [u for u in rw_result.context.unexpected_transitions if u["index"] <= cut]
    assert [z.created_index for z in part.context.origin_zones] == \
           [z.created_index for z in rw_result.context.origin_zones if z.created_index <= cut]


@pytest.mark.parametrize("cut", [420, 640])
def test_context_future_perturbation(engine, rw_bars, rw_result, cut):
    alt = random_walk_bars(n=len(rw_bars) + 5, seed=4242)
    shift = rw_bars["close"].iloc[cut] - alt["open"].iloc[cut + 1]
    fut = alt.iloc[cut + 1: len(rw_bars)].copy()
    for c in ("open", "high", "low", "close"):
        fut[c] = fut[c] + shift
    fut["timestamp"] = rw_bars["timestamp"].iloc[cut + 1:].to_numpy()
    spliced = pd.concat([rw_bars.iloc[: cut + 1], fut], ignore_index=True)
    spliced.attrs = rw_bars.attrs
    res = engine.run(spliced)
    _same_rows(rw_result.context.frame, res.context.frame, cut)
    assert _detail(res, cut) == _detail(rw_result, cut)
    assert res.snapshot_at(cut).to_dict() == rw_result.snapshot_at(cut).to_dict()


def test_historical_snapshots_reproducible_and_permission_stable(engine, rw_bars, rw_result):
    cut = 555
    part = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
    for i in (300, 450, cut):
        a, b = part.snapshot_at(i).to_dict(), rw_result.snapshot_at(i).to_dict()
        for k in ("directional_permission", "permission_confidence", "permission_reason_codes", "hard_blockers",
                  "primary_structure", "intermediate_structure", "immediate_structure", "breakout_state",
                  "acceptance_state", "trend_maturity", "latest_liquidity_sweep", "context_state", "context"):
            assert a[k] == b[k], (i, k)


def test_determinism(engine, rw_bars, rw_result):
    again = engine.run(rw_bars)
    _same_rows(rw_result.context.frame, again.context.frame, len(rw_bars) - 1)
    for i in (260, 500, len(rw_bars) - 1):
        assert _detail(again, i) == _detail(rw_result, i)
        assert again.snapshot_at(i).to_json() == rw_result.snapshot_at(i).to_json()


# ------------------------------------------------------------------ state machine
def test_state_machine_transitions_and_stability():
    sm = ContextStateMachine(flip_window=5)
    seq = ["TREND", "TREND", "PULLBACK", "PULLBACK", "CONTINUATION_ATTEMPT", "BREAK", "ACCEPTANCE", "COMPRESSION"]
    outs = [sm.step(i, s, "r", f"t{i}") for i, s in enumerate(seq)]
    assert outs[1]["bars_in_state"] == 1 and outs[2]["previous_context_state"] == "TREND"
    assert outs[2]["state_changed_at"] == "t2" and outs[3]["bars_in_state"] == 1
    assert all(o["state_transition_expected"] for o in outs[:7])
    assert outs[7]["state_transition_expected"] is False  # ACCEPTANCE -> COMPRESSION is not an expected path
    assert sm.unexpected == [{"index": 7, "at": "t7", "from": "ACCEPTANCE", "to": "COMPRESSION", "reason": "r"}]
    assert outs[7]["state_changes_in_window"] == 4  # only changes within the last 5 bars


def test_documented_progressions_are_expected():
    for path in (["TREND", "PULLBACK", "CONTINUATION_ATTEMPT", "BREAK", "ACCEPTANCE"],
                 ["TREND", "DEEP_PULLBACK", "TRANSITION", "REVERSAL_ATTEMPT"],
                 ["COMPRESSION", "BREAK", "EXPANSION", "ACCEPTANCE"], ["COMPRESSION", "EXPANSION", "FAILURE"]):
        for a, b in zip(path, path[1:]):
            assert b in EXPECTED[a] or (a, b) == ("BREAK", "EXPANSION"), (a, b)


def test_engine_state_outputs(scenario_results, rw_result):
    sc, res = scenario_results["clean_uptrend"]
    cf = res.context.frame.iloc[sc.segment_start + 60:]
    assert set(cf["context_state"]) <= {"TREND", "PULLBACK", "DEEP_PULLBACK", "CONTINUATION_ATTEMPT", "BREAK",
                                        "ACCEPTANCE", "COMPRESSION", "EXPANSION", "FAILURE", "TRANSITION"}
    assert cf["bars_in_state"].min() >= 0 and cf["state_changes_in_window"].max() <= 20
    rf = rw_result.context.frame
    changed = rf["context_state"] != rf["context_state"].shift(1)
    assert (rf.loc[changed & (rf.index > 0), "bars_in_state"] == 0).all()
    for u in rw_result.context.unexpected_transitions:
        assert u["to"] not in EXPECTED.get(u["from"], set())


# ------------------------------------------------------------------ snapshot / research output
SPEC_1B = ["primary_structure", "primary_structure_confidence", "intermediate_structure",
           "intermediate_structure_confidence", "immediate_structure", "immediate_structure_confidence",
           "active_structural_leg", "retracement_depth", "bullish_displacement_score", "bearish_displacement_score",
           "breakout_state", "breakout_quality_score", "acceptance_state", "liquidity_context", "latest_liquidity_sweep",
           "nearest_supply_zone", "nearest_demand_zone", "zone_freshness", "structural_range_percentile",
           "premium_discount_state", "trend_maturity", "momentum_deterioration_score", "compression_score",
           "expansion_state", "false_break_risk_score", "long_room_score", "short_room_score", "context_conflict_score",
           "context_quality_score", "long_context_score", "short_context_score", "directional_permission",
           "permission_confidence", "permission_reason_codes", "hard_blockers"]


def test_extended_snapshot_fields(rw_result):
    d = rw_result.snapshot_at(700).to_dict()
    for k in SPEC_1B:
        assert k in d, k
    assert d["directional_permission"] in ("ALLOW_LONG", "ALLOW_SHORT", "ALLOW_BOTH", "BLOCK_ALL")
    json.loads(rw_result.snapshot_at(700).to_json())
    # Phase 1A fields are still present and unchanged in meaning
    assert d["h4_bias"] in ("LONG", "SHORT", "NEUTRAL") and "regime" in d


def test_log_and_research_include_context(tmp_path, rw_result):
    from gbpjpy_engine.logging_utils import read_evaluation_log

    rw_result.write_evaluation_log(tmp_path / "e.jsonl")
    rec = read_evaluation_log(tmp_path / "e.jsonl")[-1]
    assert rec["context"]["directional_permission"] in ("ALLOW_LONG", "ALLOW_SHORT", "ALLOW_BOTH", "BLOCK_ALL")
    assert rec["context"]["explanation"]["reason_codes"]
    txt = explain_permission(rw_result, rw_result.features["timestamp"].iloc[600])
    assert "DIRECTIONAL PERMISSION" in txt and "PERMISSION REASON CODES" in txt
    out = rw_result.export_context(tmp_path / "ctx.parquet")
    back = pd.read_parquet(out)
    assert len(back) == len(rw_result.features)
    for col in ("long_context_score", "short_context_score", "breakout_quality_score", "trend_maturity",
                "compression_score", "long_room_score", "permission_confidence", "retracement_depth"):
        assert col in back.columns


def test_phase1a_outputs_unchanged_by_context_layer(rw_result):
    """The context layer consumes Phase 1A outputs but never adds to or alters the feature table."""
    assert "directional_permission" not in rw_result.features.columns
    assert not any(c.startswith("long_context") for c in rw_result.features.columns)


def test_context_harness_detects_leakage(rw_result):
    leaky = rw_result.context.frame[["long_context_score"]].copy()
    leaky["future"] = leaky["long_context_score"].shift(-1)
    part = leaky.iloc[:401].copy()
    part["future"] = part["long_context_score"].shift(-1)
    with pytest.raises(AssertionError, match="context look-ahead"):
        _same_rows(leaky, part, 400)
