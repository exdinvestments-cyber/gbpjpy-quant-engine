from __future__ import annotations

import numpy as np
import pandas as pd

from gbpjpy_engine.config import LevelConfig, StructureConfig, SwingConfig
from gbpjpy_engine.features.levels import _cluster, compute_levels
from gbpjpy_engine.features.structure import compute_structure
from helpers import make_bars


def test_cluster_uses_tolerance():
    pts = [(100.0, "a", 1), (100.2, "b", 2), (100.35, "c", 3), (101.5, "d", 4)]
    cl = _cluster(pts, 0.25)
    assert [len(c) for c in cl] == [3, 1]
    cl2 = _cluster(pts, 0.1)
    assert [len(c) for c in cl2] == [1, 1, 1, 1]


def _range_bars(n=220):
    # repeated tests of ~102 resistance and ~98 support
    t = np.arange(n)
    closes = 100 + 2.0 * np.sin(2 * np.pi * t / 20)
    return make_bars(closes)


def test_range_produces_support_and_resistance_zones():
    bars = _range_bars()
    atr = pd.Series(np.full(len(bars), 0.8))
    half = pd.Series(np.full(len(bars), 0.5))
    st = compute_structure(bars, atr, half, half, SwingConfig(), StructureConfig(), pd.Timedelta(hours=4))
    frame, zones = compute_levels(bars, atr, st, LevelConfig())
    c = len(bars) - 1
    z = zones[c]
    assert 2 <= len(z) <= LevelConfig().max_zones
    # repeated swing highs cluster into ONE resistance-type zone near 102
    top = [q for q in z if q.upper > 101.5]
    assert len(top) == 1
    assert top[0].interactions >= 5
    assert top[0].strength >= 60
    assert frame["nearest_support"].iloc[c] < bars["close"].iloc[c] or frame["distance_to_support"].iloc[c] == 0
    assert frame["nearest_resistance"].iloc[c] > frame["nearest_support"].iloc[c]
    assert frame["distance_to_resistance_atr"].iloc[c] >= 0
    assert frame["distance_to_support_atr"].iloc[c] >= 0


def test_zone_fields_present():
    bars = _range_bars(160)
    atr = pd.Series(np.full(len(bars), 0.8))
    half = pd.Series(np.full(len(bars), 0.5))
    st = compute_structure(bars, atr, half, half, SwingConfig(), StructureConfig(), pd.Timedelta(hours=4))
    _, zones = compute_levels(bars, atr, st, LevelConfig())
    d = zones[-1][0].to_dict()
    for k in ("upper", "lower", "midpoint", "zone_type", "interactions", "last_interaction_time", "age_bars",
              "rejection_strength", "breaks", "distance", "distance_atr", "strength"):
        assert k in d
    for zn in zones[-1]:
        assert zn.lower <= zn.midpoint <= zn.upper
        assert 0 <= zn.strength <= 100


def test_clustering_tolerance_is_volatility_aware():
    rng = np.random.default_rng(5)
    t = np.arange(220)
    bars = make_bars(100 + 2.0 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.15, 220))
    half = pd.Series(np.full(len(bars), 0.5))
    small_atr = pd.Series(np.full(len(bars), 0.05))
    big_atr = pd.Series(np.full(len(bars), 3.0))
    st = compute_structure(bars, pd.Series(np.full(len(bars), 0.8)), half, half, SwingConfig(), StructureConfig(), pd.Timedelta(hours=4))
    _, z_small = compute_levels(bars, small_atr, st, LevelConfig(max_zones=50))
    _, z_big = compute_levels(bars, big_atr, st, LevelConfig(max_zones=50))
    assert len(z_big[-1]) < len(z_small[-1])
