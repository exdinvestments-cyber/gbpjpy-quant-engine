"""Break lifecycle: CANDIDATE -> CONFIRMED -> ACCEPTED / FAILED, CANDIDATE -> INVALIDATED.

Historical state is never rewritten: the per-bar frame and ``state_at`` keep
what was known at each bar, while the current state evolves.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gbpjpy_engine.config import StructureConfig
from gbpjpy_engine.features.structure import (
    ACCEPTED,
    CANDIDATE,
    CONFIRMED,
    FAILED,
    INVALIDATED,
)
from test_structure import legs, run

LEVEL = 106.05  # swing high created by legs(100, [104, 102, 106, 104])


def _break_path(after):
    closes, _ = legs(100.0, [104, 102, 106, 104])
    n0 = len(closes)
    return closes + [104.8, 105.6, 106.4] + list(after), n0 + 2  # break bar = close 106.40 (0.35 ATR beyond)


def _event(res, k):
    ev = [b for b in res.breaks if b.direction == "bullish" and b.level == pytest.approx(LEVEL)]
    assert len(ev) == 1
    assert ev[0].break_index == k
    return ev[0]


def test_confirmed_then_failed_later_without_rewriting_history():
    # hold above the level for the confirmation window, then collapse back through it
    closes, k = _break_path([106.5, 106.6, 106.3, 105.9, 105.6, 105.2])
    _, res = run(closes)
    ev = _event(res, k)
    f = res.frame
    assert [t.to_state for t in ev.transitions] == [CANDIDATE, CONFIRMED, FAILED]
    conf_idx = ev.transitions[1].index
    fail_idx = ev.transitions[2].index
    assert conf_idx == k + StructureConfig().break_confirm_bars
    # 106.3 (+0.25) and 105.9 (-0.15 ATR) are not failures; 105.6 (-0.45 ATR) is
    assert fail_idx == k + 5
    # point-in-time history is preserved
    assert f["last_break_status"].iloc[k] == CANDIDATE
    assert f["last_break_status"].iloc[conf_idx] == CONFIRMED
    assert (f["last_break_status"].iloc[conf_idx:fail_idx] == CONFIRMED).all()
    assert f["last_break_status"].iloc[fail_idx] == FAILED
    assert ev.state_at(fail_idx - 1) == CONFIRMED and ev.state_at(fail_idx) == FAILED
    assert ev.state_at(k - 1) is None
    # transition bookkeeping
    assert ev.state == FAILED and ev.previous_state == CONFIRMED
    assert "back through level" in ev.transition_reason
    assert ev.confirmed_at == ev.transitions[1].at and ev.failed_at == ev.transitions[2].at
    assert ev.failed_at > ev.confirmed_at > ev.break_time
    assert f["last_break_previous_status"].iloc[fail_idx] == CONFIRMED
    assert pd.isna(f["last_break_failed_at"].iloc[fail_idx - 1])
    assert f["last_break_failed_at"].iloc[fail_idx] == ev.failed_at
    snap_then = ev.snapshot_at(fail_idx - 1)
    assert snap_then["state"] == CONFIRMED and snap_then["failed_at"] is None
    assert ev.snapshot_at(fail_idx)["failed_at"] is not None


def test_candidate_invalidated_inside_confirmation_window():
    closes, k = _break_path([105.8, 105.5])
    _, res = run(closes)
    ev = _event(res, k)
    assert [t.to_state for t in ev.transitions] == [CANDIDATE, INVALIDATED]
    assert ev.transitions[1].index == k + 1
    assert ev.invalidated_at is not None and ev.confirmed_at is None
    assert res.frame["last_break_status"].iloc[k] == CANDIDATE
    assert res.frame["last_break_status"].iloc[k + 1] == INVALIDATED


def test_accepted_after_follow_through_then_failed():
    closes, k = _break_path([106.8, 107.1, 107.3, 107.4, 107.5, 107.4, 107.2, 106.5, 105.7])
    _, res = run(closes)
    ev = _event(res, k)
    states = [t.to_state for t in ev.transitions]
    assert states == [CANDIDATE, CONFIRMED, ACCEPTED, FAILED]
    acc = ev.transitions[2]
    assert acc.index == k + StructureConfig().break_accept_bars
    assert ev.max_follow_through_atr >= StructureConfig().break_accept_atr
    assert ev.transitions[3].from_state == ACCEPTED
    f = res.frame
    assert f["last_break_status"].iloc[acc.index] == ACCEPTED
    assert f["last_break_status"].iloc[ev.transitions[3].index] == FAILED


def test_no_acceptance_without_follow_through():
    closes, k = _break_path([106.5, 106.6, 106.55, 106.6, 106.5, 106.6, 106.55, 106.6])
    _, res = run(closes)
    ev = _event(res, k)
    assert [t.to_state for t in ev.transitions] == [CANDIDATE, CONFIRMED]


def test_monitoring_window_freezes_lifecycle():
    cfg = StructureConfig(break_accept_bars=3, break_monitor_bars=4)
    closes, k = _break_path([106.5, 106.6, 106.7, 106.8, 106.9, 105.0])
    _, res = run(closes, structure=cfg)
    ev = _event(res, k)
    assert FAILED not in [t.to_state for t in ev.transitions]
    assert ev.monitoring_ended_index == k + 4


def test_failed_break_no_longer_holds_transitional_state():
    # bullish structure, bearish CHoCH that is confirmed and later fails -> no permanent 'transitional' override
    closes, _ = legs(100.0, [104, 102, 106, 104, 108, 106, 110])
    closes = closes + [109.0, 108.0, 107.0, 106.0, 105.4, 105.3, 105.2, 106.0, 107.0, 108.0, 109.0]
    _, res = run(closes)
    choch = [b for b in res.breaks if b.break_type == "CHOCH" and b.direction == "bearish"]
    assert choch
    ev = choch[0]
    assert ev.state == FAILED
    f = res.frame
    fail_idx = ev.transitions[-1].index
    assert f["structure_state"].iloc[ev.break_index] == "transitional"
    assert f["structure_state"].iloc[fail_idx - 1] == "transitional"  # live CHoCH (CONFIRMED)
    # on the bar the CHoCH FAILS, the override lapses and the swing sequence (bullish) applies again
    assert f["swing_structure_state"].iloc[fail_idx] == "bullish"
    assert f["structure_state"].iloc[fail_idx] == "bullish"


def test_lifecycle_is_point_in_time_across_truncation(engine, rw_bars, rw_result):
    for cut in (350, 610):
        part = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
        full = {b.event_id: b for b in rw_result.structure.breaks}
        for b in part.structure.breaks:
            fb = full[b.event_id]
            assert b.state == fb.state_at(cut)
            assert b.snapshot_at(cut) == fb.snapshot_at(cut)
            assert [t.to_dict() for t in b.transitions] == [t.to_dict() for t in fb.transitions_known_at(cut)]


def test_all_lifecycle_states_occur_on_random_walk(rw_result):
    seen = {t.to_state for b in rw_result.structure.breaks for t in b.transitions}
    assert {CANDIDATE, CONFIRMED, ACCEPTED, FAILED, INVALIDATED} <= seen
    for b in rw_result.structure.breaks:
        for t0, t1 in zip(b.transitions, b.transitions[1:]):
            assert t1.index > t0.index and t1.from_state == t0.to_state
            assert (t0.to_state, t1.to_state) in {
                (CANDIDATE, CONFIRMED), (CANDIDATE, INVALIDATED), (CONFIRMED, ACCEPTED),
                (CONFIRMED, FAILED), (ACCEPTED, FAILED),
            }
