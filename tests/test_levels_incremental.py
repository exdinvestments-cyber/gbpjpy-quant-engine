"""Incremental zone engine: equivalence with the reference rebuild, updates,
invalidation, ageing/expiry and point-in-time event history."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.config import LevelConfig, StructureConfig, SwingConfig
from gbpjpy_engine.features.levels import compute_levels, compute_levels_rebuild
from gbpjpy_engine.features.structure import compute_structure
from gbpjpy_engine.synthetic import random_walk_bars
from helpers import make_bars
from test_structure import legs

TF = pd.Timedelta(hours=4)
ZONE_KEYS = ("lower", "upper", "midpoint", "zone_type", "sources", "n_sources", "interactions", "touch_bars",
             "last_interaction_index", "bars_since_interaction", "age_bars", "rejections", "rejection_strength",
             "breaks", "distance", "distance_atr", "strength")


def _structure(bars, atr_value=0.4):
    atr = pd.Series(np.full(len(bars), atr_value))
    half = pd.Series(np.full(len(bars), 0.5))
    return atr, compute_structure(bars, atr, half, half, SwingConfig(), StructureConfig(), TF)


@pytest.mark.parametrize("seed,lookback", [(4, 300), (9, 120)])
def test_incremental_equals_rebuild_bar_by_bar(seed, lookback):
    bars = random_walk_bars(1500, seed=seed)
    atr, st = _structure(bars)
    cfg = LevelConfig(lookback_bars=lookback, invalidate_after_breaks=0)
    f_ref, z_ref = compute_levels_rebuild(bars, atr, st, cfg)
    f_inc, z_inc = compute_levels(bars, atr, st, cfg)
    pd.testing.assert_frame_equal(f_ref, f_inc)
    for c, (a, b) in enumerate(zip(z_ref, z_inc)):
        assert [tuple(getattr(z, k) for k in ZONE_KEYS) for z in a] == \
               [tuple(getattr(z, k) for k in ZONE_KEYS) for z in b], f"zone mismatch at bar {c}"


def _zone_path():
    # swing high at 106.05, pullback, second swing high at 106.25 (within 0.5 ATR tolerance)
    closes, _ = legs(100.0, [104, 102, 106.0, 104.0, 106.2, 103.5, 104.5])
    return make_bars(closes)


def test_new_source_updates_existing_zone_and_keeps_identity():
    bars = _zone_path()
    atr, st = _structure(bars, 1.0)
    _, zones, book = compute_levels(bars, atr, st, LevelConfig(invalidate_after_breaks=0), return_book=True)
    highs = [s for s in st.swings if s.kind == "high"]
    first, second = highs[1], highs[2]  # the two tops near 106
    z_before = next(z for z in zones[second.accepted_index - 1] if z.lower <= first.price <= z.upper)
    z_after = next(z for z in zones[second.accepted_index] if z.lower <= second.price <= z.upper)
    assert z_after.zone_id == z_before.zone_id
    assert z_after.created_index == z_before.created_index
    # the newly confirmed swing is added to the existing zone (the trailing range-high source at the
    # same price already defined the upper bound before the swing could be confirmed)
    assert z_after.n_sources == z_before.n_sources + 1
    assert z_after.upper >= second.price >= z_after.lower
    assert any(e.event == "updated" and e.zone_id == z_after.zone_id and e.index == second.accepted_index
               for e in book.events)
    assert "created" in {e.event for e in book.events}


def test_zone_invalidated_after_repeated_crossing():
    closes, _ = legs(103.0, [100.0, 102.0])  # swing low at 99.95
    closes = closes + [101.0, 99.0] * 6
    bars = make_bars(closes)
    atr, st = _structure(bars, 1.0)
    low = next(s for s in st.swings if s.kind == "low")
    cfg = LevelConfig(invalidate_after_breaks=4)
    _, zones, book = compute_levels(bars, atr, st, cfg, return_book=True)
    inv = [z for z in book.invalidated if z["lower"] <= low.price <= z["upper"]]
    assert inv, "zone that price keeps crossing must be invalidated"
    zid, idx = inv[0]["zone_id"], inv[0]["index"]
    assert "crossed the zone 4 times" in inv[0]["reason"]
    assert any(z.zone_id == zid for z in zones[idx - 1])
    for c in range(idx, len(zones)):
        assert all(z.zone_id != zid for z in zones[c])
    # disabled invalidation keeps the zone (with its break count)
    _, zones0, book0 = compute_levels(bars, atr, st, replace(cfg, invalidate_after_breaks=0), return_book=True)
    assert not book0.invalidated
    last = next(z for z in zones0[-1] if z.lower <= low.price <= z.upper)
    assert last.breaks >= 4


def test_zone_ageing_and_expiry():
    bars = random_walk_bars(900, seed=21)
    atr, st = _structure(bars)
    cfg = LevelConfig(lookback_bars=60)
    _, zones, book = compute_levels(bars, atr, st, cfg, return_book=True)
    prev = {}
    for c, zs in enumerate(zones):
        for z in zs:
            assert z.bars_since_created == c - z.created_index
            assert z.age_bars <= cfg.lookback_bars - 1 + 0  # sources older than the lookback have aged out
            if z.zone_id in prev and prev[z.zone_id][0] == c - 1:
                assert z.created_index == prev[z.zone_id][1]
            prev[z.zone_id] = (c, z.created_index)
    expired = [e for e in book.events if e.event == "expired"]
    assert expired
    for e in expired:
        assert all(z.zone_id != e.zone_id for zs in zones[e.index:] for z in zs)


def test_zone_events_are_point_in_time(engine, rw_bars, rw_result):
    cut = 480
    part = engine.run(rw_bars.iloc[: cut + 1].reset_index(drop=True))
    full_prefix = [e.to_dict() for e in rw_result.zone_events if e.index <= cut]
    assert [e.to_dict() for e in part.zone_events] == full_prefix
    assert part.invalidated_zones == [z for z in rw_result.invalidated_zones if z["index"] <= cut]
    assert [z.to_dict() for z in part.zones_by_bar[cut]] == [z.to_dict() for z in rw_result.zones_by_bar[cut]]


def test_engine_zone_outputs_consistent(rw_result):
    f = rw_result.features
    for c in (300, 600, 850):
        zs = rw_result.zones_by_bar[c]
        assert len(zs) <= rw_result.config.levels.max_zones
        assert len({z.zone_id for z in zs}) == len(zs)
        if not np.isnan(f["nearest_support"].iloc[c]):
            assert any(z.midpoint == f["nearest_support"].iloc[c] and z.zone_type == "support" for z in zs)
