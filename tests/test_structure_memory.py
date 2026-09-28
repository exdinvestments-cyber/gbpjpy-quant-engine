"""Rolling structural memory and explicit occurrence vs confirmation timestamps."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine import H4Config, H4MarketIntelligenceEngine
from gbpjpy_engine.config import StructureConfig, SwingConfig, config_from_dict
from test_structure import legs, run

TF = pd.Timedelta(hours=4)


def _long_uptrend():
    anchors = []
    level = 100.0
    for _ in range(14):
        anchors += [level + 4, level + 2]
        level += 2
    return legs(100.0, anchors)[0]


def test_multi_swing_history_retained_and_labelled():
    _, res = run(_long_uptrend())
    c = len(res.frame) - 1
    hist = res.history_at(c)
    assert len(hist) == StructureConfig().swing_history_size == 16
    assert all(hist[k].confirm_index <= hist[k + 1].confirm_index for k in range(len(hist) - 1))
    assert {s.label for s in hist} <= {"HH", "HL"}
    assert res.frame["n_swings"].iloc[c] == 16
    seq = res.frame["swing_sequence"].iloc[c].split(">")
    assert len(seq) == 16 and set(seq) == {"HH", "HL"}
    assert res.frame["hist_hh"].iloc[c] + res.frame["hist_hl"].iloc[c] == 16
    for s in hist:
        d = s.to_dict()
        for k in ("occurred_at", "confirmed_at", "price", "type", "classification", "significance_atr"):
            assert k in d
        assert s.significance_atr is not None and s.significance_atr >= SwingConfig().min_swing_atr


def test_swing_history_size_is_configurable():
    closes = _long_uptrend()
    _, small = run(closes, structure=StructureConfig(swing_history_size=8, quality_swings=8))
    _, big = run(closes, structure=StructureConfig(swing_history_size=24))
    c = len(closes) - 1
    assert len(small.history_at(c)) == 8
    assert len(big.history_at(c)) > 16
    # the most recent swings are identical regardless of memory size
    assert [s.to_dict() for s in small.history_at(c)] == [s.to_dict() for s in big.history_at(c)[-8:]]
    # classification of the latest structure is unchanged by memory size
    assert small.frame["structure_state"].iloc[c] == big.frame["structure_state"].iloc[c] == "bullish"


def test_swing_history_size_validation():
    with pytest.raises(ValueError):
        config_from_dict({"structure": {"swing_history_size": 3}})
    with pytest.raises(ValueError):
        config_from_dict({"structure": {"swing_history_size": 6, "quality_swings": 8}})


def test_history_is_point_in_time(rw_result):
    """history_at(c) never contains swings confirmed after c and is the window held at c."""
    st = rw_result.structure
    for c in (300, 500, 777):
        hist = st.history_at(c)
        assert len(hist) <= st.history_size
        assert all(s.accepted_index <= c for s in hist)
        assert all(s.confirmed_at <= rw_result.features["available_at"].iloc[c] for s in hist)
        assert rw_result.features["n_swings"].iloc[c] == len(hist)
        snap_hist = rw_result.snapshot_at(c).market_structure["swing_history"]
        assert [h["swing_id"] for h in snap_hist] == [s.swing_id for s in hist]


def test_occurrence_and_confirmation_timestamps_are_distinct(rw_result):
    R = SwingConfig().right_bars
    f = rw_result.features
    for s in rw_result.structure.swings:
        assert s.occurred_at == f["timestamp"].iloc[s.pivot_index]
        assert s.confirmed_at == f["available_at"].iloc[s.confirm_index]
        assert s.confirmation_lag_bars == R
        assert s.confirmed_at - s.occurred_at >= (R + 1) * TF  # equality except across weekends
    known = f["last_swing_high_confirmed_at"].notna()
    assert (f.loc[known, "last_swing_high_occurred_at"] < f.loc[known, "last_swing_high_confirmed_at"]).all()
    snap = rw_result.latest().market_structure
    assert snap["last_swing_high_occurred_at"] and snap["last_swing_high_confirmed_at"]
    assert all(h["occurred_at"] < h["confirmed_at"] for h in snap["swing_history"])


def test_confirmation_sensitivity_is_configurable(rw_bars):
    cfg5 = replace(H4Config(), swing=SwingConfig(left_bars=3, right_bars=5))
    res5 = H4MarketIntelligenceEngine(cfg5).run(rw_bars)
    assert res5.structure.swings
    assert all(s.confirmation_lag_bars == 5 for s in res5.structure.swings)
    assert all(s.confirm_index - s.pivot_index == 5 for s in res5.structure.swings)


def test_historical_swing_records_never_rewritten(engine, rw_bars, rw_result):
    """Swings known in a truncated run keep identical timestamps/price/label in the full run."""
    cut = 520
    part = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
    full = {s.swing_id: s for s in rw_result.structure.swings}
    assert part.structure.swings
    for s in part.structure.swings:
        f = full[s.swing_id]
        for k in ("occurred_at", "confirmed_at", "price", "label", "significance_atr", "pivot_index", "confirm_index"):
            assert getattr(f, k) == getattr(s, k)
        # superseding after the cut is recorded as a later event, never before it
        if f.removed_index is not None and s.removed_index is None:
            assert f.removed_index > cut
