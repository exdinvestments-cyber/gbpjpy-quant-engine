"""Phase 1C critical look-ahead tests.

An H1 bar at time T must not be able to see future H1 candles, the UNFINISHED
H4 candle containing T, future swing confirmations, future breakout or sweep
outcomes, or future setup outcomes.  Changing future data must not rewrite
historical setup states.  Plus timeframe-boundary and deterministic-replay
tests.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.data.resample import resample_complete
from gbpjpy_engine.h1.alignment import align_h4_to_h1
from gbpjpy_engine.snapshot import to_jsonable


def pipeline(engine, h1_engine, h1, h4=None):
    h1 = h1.reset_index(drop=True)
    h4 = resample_complete(h1, 60, 240)[0] if h4 is None else h4
    h4r = engine.run(h4)
    return h4r, h1_engine.run(h1, h4r)


def js(v):
    return json.dumps(to_jsonable(v), sort_keys=True, default=str)


def same_rows(a: pd.DataFrame, b: pd.DataFrame, upto: int) -> None:
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        x = [js(v) for v in a[col].iloc[: upto + 1]]
        y = [js(v) for v in b[col].iloc[: upto + 1]]
        bad = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        assert not bad, f"H1 look-ahead in {col!r} at rows {bad[:5]}: {x[bad[0]]} vs {y[bad[0]]}"


def mid_bucket(res, near: int) -> int:
    """An H1 index close to ``near`` whose H4 candle is still forming (not its last hour)."""
    ts = res.frame["timestamp"]
    for i in range(near, near - 12, -1):
        if ts.iloc[i].hour % 4 in (0, 1):
            return i
    raise AssertionError


@pytest.fixture(scope="module")
def up(h1_results):
    return h1_results["h1_uptrend_pullbacks"]


def check_prefix(full, part, cut):
    same_rows(full.frame, part.frame, cut)
    same_rows(full.features, part.features, cut)
    for i in (cut - 30, cut - 3, cut):
        assert js(part.details[i]) == js(full.details[i]), i
    # setups: every setup known by the truncated run has exactly the history the full run had at the cut
    fs = {s.setup_id: s for s in full.setups}
    for s in part.setups:
        f = fs[s.setup_id]
        assert [t for t in f.transitions if t[0] <= cut] == s.transitions
        assert f.state_at(cut) == s.state
    assert {s.setup_id for s in part.setups} == {s.setup_id for s in full.setups if s.created_index <= cut}
    assert part.qualified_setups == [q for q in full.qualified_setups if q["bar_index"] <= cut]
    # swings, breaks and sweeps: nothing confirmed / resolved after the cut is visible
    assert {s.swing_id for s in part.structure.swings} == {s.swing_id for s in full.structure.swings if s.accepted_index <= cut}
    fb = {b.event_id: b for b in full.structure.breaks}
    for b in part.structure.breaks:
        assert [t.to_dict() for t in b.transitions] == [t.to_dict() for t in fb[b.event_id].transitions_known_at(cut)]
    fsw = [e for e in full.sweep_events if e.index <= cut]
    assert [e.snapshot_at(cut) for e in part.sweep_events] == [e.snapshot_at(cut) for e in fsw]
    assert [t.index for t in part.transitions] == [t.index for t in full.transitions if t.index <= cut]


def test_truncation_invariance_around_a_qualification(engine, h1_engine, up):
    sc, _, full = up
    q = full.qualified_setups[0]["bar_index"]
    for cut in (q, mid_bucket(full, q + 6)):
        _, part = pipeline(engine, h1_engine, sc.h1.iloc[: cut + 1])
        check_prefix(full, part, cut)


def test_unfinished_h4_candle_cannot_leak(engine, h1_engine, up):
    sc, h4_full, full = up
    cut = mid_bucket(full, full.qualified_setups[-1]["bar_index"])
    k_open = full.frame["timestamp"].iloc[cut].floor("4h")
    alt = sc.h1.copy()
    # rewrite the rest of the H4 candle that is still forming at `cut` (and everything after it)
    later = alt.index > cut
    for c in ("open", "high", "low", "close"):
        alt.loc[later, c] = alt.loc[later, c] + 3.0 + np.sin(np.arange(later.sum())) * 0.5
    alt.loc[later, "high"] = alt.loc[later, ["open", "high", "close"]].max(axis=1) + 0.01
    alt.loc[later, "low"] = alt.loc[later, ["open", "low", "close"]].min(axis=1) - 0.01
    h4_alt, res_alt = pipeline(engine, h1_engine, alt)
    # the H4 candle containing `cut` really is different in the two worlds ...
    a = h4_full.features.set_index("timestamp").loc[k_open, ["high", "close"]]
    b = h4_alt.features.set_index("timestamp").loc[k_open, ["high", "close"]]
    assert not np.allclose(a.to_numpy(float), b.to_numpy(float))
    # ... yet nothing the H1 engine reported up to `cut` changes
    same_rows(full.frame, res_alt.frame, cut)
    assert js(full.details[cut]) == js(res_alt.details[cut])
    assert full.frame["h4_available_at"].iloc[cut] <= full.frame["available_at"].iloc[cut]
    assert full.frame["h4_timestamp"].iloc[cut] < k_open


def test_future_changes_do_not_rewrite_setup_history(engine, h1_engine, up):
    sc, _, full = up
    q = full.qualified_setups[0]
    cut = q["bar_index"]
    alt = sc.h1.copy()
    later = alt.index > cut
    for c in ("open", "high", "low", "close"):
        alt.loc[later, c] = alt.loc[later, c] - 4.0  # a crash right after qualification
    _, res_alt = pipeline(engine, h1_engine, alt)
    st_full = next(s for s in full.setups if s.setup_id == q["setup_id"])
    st_alt = next(s for s in res_alt.setups if s.setup_id == q["setup_id"])
    assert st_alt.qualified == st_full.qualified == q
    assert st_alt.state_at(cut) == st_full.state_at(cut) == "QUALIFIED"
    assert st_alt.state in ("INVALIDATED", "EXPIRED")  # the future changed the ending, not the past
    same_rows(full.frame, res_alt.frame, cut)


def test_deterministic_replay(engine, h1_engine, up):
    sc, _, full = up
    _, again = pipeline(engine, h1_engine, sc.h1)
    same_rows(full.frame, again.frame, len(full.frame) - 1)
    assert [s.to_dict() for s in again.setups] == [s.to_dict() for s in full.setups]
    assert again.qualified_setups == full.qualified_setups
    assert js(again.counterfactual) == js(full.counterfactual)
    for i in range(0, len(full.frame), 211):
        assert js(again.details[i]) == js(full.details[i])


# ------------------------------------------------------------------ timeframe boundaries
def test_missing_h1_candles_fail_safe(engine, h1_engine, up):
    sc, _, full = up
    gap_at = full.qualified_setups[0]["bar_index"] - 40
    holed = sc.h1.drop(index=[gap_at, gap_at + 1]).reset_index(drop=True)
    h4, dropped = resample_complete(holed, 60, 240)
    assert dropped  # the damaged H4 bucket is not emitted as if complete
    h4r, res = pipeline(engine, h1_engine, holed, h4)
    f = res.frame
    assert "UNRESOLVED_DATA_GAP" in f["h1_blockers"].iloc[gap_at]
    assert not f["actionable_setup_permission_long"].iloc[gap_at]
    known = f["h4_index"] >= 0
    assert (f.loc[known, "h4_available_at"] <= f.loc[known, "available_at"]).all()


def test_missing_h4_candles_never_substitute_future_context(engine, h1_engine, up):
    sc, h4_full, full = up
    i4 = int(full.frame["h4_index"].iloc[full.qualified_setups[0]["bar_index"]])
    h4_holed = h4_full.features[["timestamp", "open", "high", "low", "close", "volume", "spread", "source"]]
    h4_holed = h4_holed.drop(index=[i4 - 3, i4 - 2, i4 - 1]).reset_index(drop=True)
    h4_holed.attrs = sc.h4.attrs
    h4r = engine.run(h4_holed)
    al = align_h4_to_h1(full.features["available_at"], h4r, h1_engine.cfg.alignment)
    h4c = h4r.features["available_at"].to_numpy()
    ok = al["h4_index"] >= 0
    assert (h4c[al.loc[ok, "h4_index"]] <= full.features.loc[ok, "available_at"].to_numpy()).all()
    assert (al["h4_context_status"] == "STALE").any()
    res = h1_engine.run(sc.h1, h4r)
    stale = res.frame["h4_context_status"] == "STALE"
    assert all("STALE_H4_CONTEXT" in b for b in res.frame.loc[stale, "h1_blockers"])


def test_dst_style_offset_h4_grid(engine, h1_engine, up):
    sc, _, _ = up
    h4_off, _ = resample_complete(sc.h1, 60, 240, offset="1h")  # broker H4 grid 01:00, 05:00, ... UTC
    h4r = engine.run(h4_off)
    res = h1_engine.run(sc.h1, h4r)
    f = res.frame
    known = f["h4_index"] >= 0
    assert (f.loc[known, "h4_available_at"] <= f.loc[known, "available_at"]).all()
    # an H1 bar closing at 02:00 cannot see the H4 candle 01:00-05:00
    midweek = (f["available_at"].dt.hour == 2) & f["available_at"].dt.dayofweek.isin([1, 2, 3])
    for i in np.flatnonzero(midweek.to_numpy())[5:15]:
        assert f["h4_available_at"].iloc[i] == f["available_at"].iloc[i] - pd.Timedelta(hours=1)
