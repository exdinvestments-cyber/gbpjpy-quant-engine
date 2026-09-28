"""Phase 1B: liquidity references, sweeps vs continuation, origin zones, freshness, role reversal,
dynamic zone context."""

from __future__ import annotations

import numpy as np
import pytest

from ctx_helpers import price_arrays, swing, zone
from gbpjpy_engine.config import LiquidityConfig, OriginZoneConfig, RoleReversalConfig
from gbpjpy_engine.context.liquidity import BREAK_AND_ACCEPT, SWEEP_AND_REJECT, UNRESOLVED, LiquidityMap
from gbpjpy_engine.context.zones import OriginZoneTracker, RoleReversalTracker, dynamic_context


def _bars(closes, highs=None, lows=None):
    c = np.asarray(closes, float)
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) + 0.1 if highs is None else np.asarray(highs, float)
    l = np.minimum(o, c) - 0.1 if lows is None else np.asarray(lows, float)
    return price_arrays(o, h, l, c)


def _run_liq(arr, swings, upto=None):
    m = LiquidityMap(arr, swings, [], LiquidityConfig())
    for c in range(len(arr["close"]) if upto is None else upto + 1):
        m.step(c)
    return m


def _sweep_path(after):
    closes = [100.0] * 8 + after
    highs = list(np.array(closes) + 0.2)
    return closes, highs


def test_sweep_and_reject_evolves_without_premature_decision():
    closes, highs = _sweep_path([100.5, 100.4, 100.3, 100.2, 100.1])
    highs[8] = 101.3  # trades through the 101.0 swing high, closes back below
    arr = _bars(closes, highs)
    sw = [swing(0, "high", 2, 101.0, sig=2.0)]
    sw[0].accepted_index = 5
    m = _run_liq(arr, sw)
    ev = m.events[0]
    assert ev.side == "upside" and ev.implication == "bearish" and ev.index == 8
    assert ev.state_at(8) == UNRESOLVED and ev.state_at(10) == UNRESOLVED
    assert ev.state_at(11) == SWEEP_AND_REJECT
    assert ev.penetration_atr == pytest.approx(0.3) and ev.rejection_atr == pytest.approx(0.5)
    assert m.score(ev, 11, False) > 50
    assert m.score(ev, 11, True) > m.score(ev, 11, False)  # structural effect adds evidence
    # historical immutability: a later run knows more, but the state at bar 10 is unchanged
    early = _run_liq(arr, sw, upto=10)
    assert early.events[0].state_at(10) == ev.state_at(10) == UNRESOLVED


def test_break_and_accept_is_continuation_not_sweep():
    closes, highs = _sweep_path([100.9, 101.5, 101.8, 102.0])
    highs[8] = 101.4
    arr = _bars(closes, highs)
    sw = [swing(0, "high", 2, 101.0)]
    sw[0].accepted_index = 5
    m = _run_liq(arr, sw)
    ev = m.events[0]
    assert ev.state_at(8) == UNRESOLVED
    assert ev.state_at(9) == BREAK_AND_ACCEPT
    assert m.score(ev, 10, False) == 0.0


def test_sweep_later_reclaimed_keeps_history():
    closes, highs = _sweep_path([100.5, 100.4, 100.3, 100.2, 100.6, 101.4])
    highs[8] = 101.3
    arr = _bars(closes, highs)
    sw = [swing(0, "high", 2, 101.0)]
    sw[0].accepted_index = 5
    m = _run_liq(arr, sw)
    ev = m.events[0]
    assert ev.state_at(12) == SWEEP_AND_REJECT
    assert ev.state_at(13) == BREAK_AND_ACCEPT
    assert ev.transitions[-1][3] == "reference later reclaimed"


def test_downside_sweep_is_bullish():
    closes = [100.0] * 8 + [99.6, 99.7, 99.8, 99.9]
    lows = list(np.array(closes) - 0.2)
    lows[8] = 98.7
    arr = _bars(closes, lows=lows)
    sw = [swing(0, "low", 2, 99.0)]
    sw[0].accepted_index = 5
    ev = _run_liq(arr, sw).events[0]
    assert ev.side == "downside" and ev.implication == "bullish" and ev.state_at(11) == SWEEP_AND_REJECT


def test_equal_and_near_equal_clusters():
    arr = _bars([100.0] * 12)
    sw = [swing(0, "high", 1, 101.0), swing(1, "high", 3, 101.05), swing(2, "high", 5, 103.0),
          swing(3, "low", 2, 98.0), swing(4, "low", 4, 98.3)]
    for s in sw:
        s.accepted_index = 6
    m = _run_liq(arr, sw)
    up = m.clusters("upside", 1.0)
    types = {x["type"]: x for x in up}
    assert types["EQUAL_HIGHS"]["size"] == 2 and types["EQUAL_HIGHS"]["price"] == pytest.approx(101.025)
    assert "SWING_LIQUIDITY" in types and types["SWING_LIQUIDITY"].get("structural_extreme")
    dn = m.clusters("downside", 1.0)
    assert dn[0]["type"] == "NEAR_EQUAL_LOWS" and dn[0]["size"] == 2
    for x in up + dn:
        assert all(r.startswith("S") for r in x["ref_ids"])


def test_sweep_of_equal_highs_reports_cluster():
    closes = [100.0] * 10 + [100.6, 100.5, 100.4, 100.3]
    highs = list(np.array(closes) + 0.2)
    highs[10] = 101.4
    arr = _bars(closes, highs)
    sw = [swing(0, "high", 1, 101.0), swing(1, "high", 4, 101.05)]
    for s in sw:
        s.accepted_index = 7
    ev = _run_liq(arr, sw).events[0]
    assert ev.cluster_size == 2 and set(ev.ref_ids) == {"S0", "S1"}


# ---------------------------------------------------------------- origin zones
def _origin_setup():
    n = 40
    c = np.full(n, 100.0)
    c[10:13] = [100.8, 101.6, 102.4]
    c[13:20] = 102.6
    c[20] = 102.6
    c[21:30] = 103.0
    c[30:] = 99.0
    o = np.concatenate([[c[0]], c[:-1]])
    h, l = np.maximum(o, c) + 0.1, np.minimum(o, c) - 0.1
    l[20] = 100.05  # revisit into the origin zone
    arr = price_arrays(o, h, l, c)
    bull = np.zeros(n)
    bull[12:14] = 70.0
    return arr, bull


def test_origin_zone_creation_freshness_and_invalidation():
    arr, bull = _origin_setup()
    tr = OriginZoneTracker(arr, {"bullish": bull, "bearish": np.zeros(len(bull))},
                           np.array(["bullish"] * len(bull), dtype=object), 3, OriginZoneConfig())
    views = {}
    for c in range(len(bull)):
        active = tr.step(c)
        views[c] = [z.view(c, arr["close"][c], 1.0, OriginZoneConfig()) for z in active]
    assert len(tr.zones) == 1
    z = tr.zones[0]
    assert z.direction == "demand" and z.created_index == 12 and z.origin_indices == [8, 9]
    assert z.lower == pytest.approx(99.9) and z.upper == pytest.approx(100.1)
    assert views[12][0]["freshness"] == "FRESH"
    assert views[20][0]["freshness"] == "LIGHTLY_TESTED" and views[20][0]["revisits"] == 1
    assert views[25][0]["reaction_strength_atr"] > 2
    assert views[20][0]["penetration_depth"] == pytest.approx(0.25)
    assert z.invalidated_index == 30 and views[30] == []
    assert z.freshness(OriginZoneConfig()) == "INVALIDATED"
    for k in ("created_at", "origin_structure", "displacement_strength", "invalidation_status"):
        assert k in views[20][0]
    assert views[12][0]["lower"] == views[25][0]["lower"]  # boundaries never resized


def test_single_bar_spike_does_not_create_origin_zone():
    arr, bull = _origin_setup()
    tr = OriginZoneTracker(arr, {"bullish": np.zeros(len(bull)), "bearish": np.zeros(len(bull))},
                           np.array(["neutral"] * len(bull), dtype=object), 3, OriginZoneConfig())
    for c in range(len(bull)):
        tr.step(c)
    assert tr.zones == []


# ---------------------------------------------------------------- role reversal
def _role_run(closes, lows=None):
    arr = _bars(closes, lows=lows)
    z = zone(1, 100.0, 100.2, zone_type="resistance")
    tr = RoleReversalTracker(arr, RoleReversalConfig())
    for c in range(len(closes)):
        z.zone_type = "resistance" if arr["close"][c] < z.midpoint else "support"
        tr.step(c, [z])
    return tr


def test_role_reversal_requires_break_acceptance_retest_reaction():
    closes = [99.5, 99.6, 100.5, 100.7, 100.9, 100.4, 100.9, 101.0]
    lows = list(np.array(closes) - 0.1)
    lows[5] = 100.1  # retest into the zone, closing above
    tr = _role_run(closes, lows)
    t = tr.tracks[1]
    states = [x[2] for x in t.transitions]
    assert states == ["BROKEN", "ACCEPTED", "RETESTED", "FLIPPED"]
    assert t.original_role == "resistance" and t.new_role == "support" and 0 < t.confidence <= 100
    assert tr.flips and tr.flips[0]["type"] == "RESISTANCE_TO_SUPPORT" and tr.flips[0]["index"] == 6


def test_crossing_alone_does_not_flip_role():
    tr = _role_run([99.5, 99.6, 100.5, 100.7, 100.9, 101.2, 101.4, 101.6])
    assert tr.flips == []
    assert tr.tracks[1].state == "ACCEPTED"


def test_failed_flip_returns_to_original():
    tr = _role_run([99.5, 99.6, 100.5, 99.7, 99.6])
    states = [x[2] for x in tr.tracks[1].transitions]
    assert states == ["BROKEN", "FAILED_FLIP", "ORIGINAL"] and tr.flips == []


# ---------------------------------------------------------------- dynamic zone context
def test_dynamic_zone_context_does_not_resize():
    d = dynamic_context(100.0, 101.0, 2.0, atr_now=0.5, formation_atr=1.0)
    assert d == {"current_atr": 0.5, "current_width_atr": 2.0, "current_distance_atr": 4.0, "volatility_vs_formation": 0.5}
    assert dynamic_context(100.0, 101.0, 0.0, float("nan"), 1.0)["current_atr"] is None


def test_engine_zone_context_preserves_original_boundaries(rw_result):
    for c in (400, 700):
        zs = {z.zone_id: z for z in rw_result.zones_by_bar[c]}
        for zc in rw_result.context.details[c]["sr_zone_context"]:
            z = zs[zc["zone_id"]]
            assert (zc["lower"], zc["upper"]) == (z.lower, z.upper)
            assert zc["formation_atr"] == z.formation_atr and z.formation_atr > 0
            assert zc["current_width_atr"] == pytest.approx((z.upper - z.lower) / rw_result.features["atr"].iloc[c])
            assert zc["volatility_vs_formation"] == pytest.approx(rw_result.features["atr"].iloc[c] / z.formation_atr)
