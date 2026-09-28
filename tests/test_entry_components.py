"""Phase 1D unit tests: spread, bid/ask, slippage, gap, deterioration, abnormality, barriers, session, news,
freshness, chase, extension, market quality, conflict, quality, config."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine.config import RoomConfig, describe_config
from gbpjpy_engine.entry import (EmpiricalSlippage, EntryConfig, FixedSlippage, NewsEvent, SpreadDependentSlippage,
                                 UnknownSlippage, VolatilityDependentSlippage, entry_config_from_dict, load_entry_config)
from gbpjpy_engine.entry.config import ExecutionPriceConfig, NewsConfig, SpreadConfig
from gbpjpy_engine.entry.execution import (abnormality, assess_spread, deterioration, executable_reference, gap_info,
                                           news_status, session_context, slippage, stacked_barriers, to_pips)
from gbpjpy_engine.entry.scoring import (chase_risk, entry_conflict, entry_extension, entry_market_quality, entry_quality,
                                         freshness)

CFG = EntryConfig()
PIP = 0.01
ROOT = Path(gbpjpy_engine.__file__).parents[2]


# ------------------------------------------------------------------ spread
def test_spread_status_levels_and_unknown():
    hist = np.full(100, 2.0)
    atr = 0.30  # 30 pips
    assert assess_spread(1.8, hist, atr, PIP, CFG.spread)["spread_status"] == "NORMAL"
    assert assess_spread(4.2, hist, atr, PIP, CFG.spread)["spread_status"] == "ELEVATED"  # distribution ratio 2.1
    assert assess_spread(7.0, hist, atr, PIP, CFG.spread)["spread_status"] == "HIGH"
    ext = assess_spread(15.0, hist, atr, PIP, CFG.spread)
    assert ext["spread_status"] == "EXTREME" and ext["blocked"]
    unk = assess_spread(None, hist, atr, PIP, CFG.spread)
    assert unk["spread_status"] == "UNKNOWN" and unk["spread_pips"] is None and unk["spread_quality_score"] is None
    assert not unk["blocked"]  # UNKNOWN is visible but not blocked by default
    assert assess_spread(None, hist, atr, PIP, replace(CFG.spread, block_unknown=True))["blocked"]


def test_spread_relative_to_atr_and_quality_monotonic():
    hist = np.full(100, 2.0)
    wide_atr = assess_spread(2.0, hist, 0.40, PIP, CFG.spread)
    tight_atr = assess_spread(2.0, hist, 0.10, PIP, CFG.spread)  # 2 pips = 20% of a 10-pip ATR
    assert wide_atr["spread_status"] == "NORMAL" and tight_atr["spread_status"] == "HIGH"
    q = [assess_spread(s, hist, 0.3, PIP, CFG.spread)["spread_quality_score"] for s in (1.5, 3.0, 6.0, 9.0)]
    assert q == sorted(q, reverse=True)


def test_spread_blocker_is_configurable():
    hist = np.full(100, 2.0)
    assert assess_spread(7.0, hist, 0.3, PIP, CFG.spread)["blocked"]
    relaxed = replace(CFG.spread, block_statuses=("EXTREME",))
    assert not assess_spread(7.0, hist, 0.3, PIP, relaxed)["blocked"]


def test_spread_units_are_declared_not_guessed():
    assert to_pips(25.0, ExecutionPriceConfig(spread_unit="points"), PIP) == pytest.approx(2.5)
    assert to_pips(0.025, ExecutionPriceConfig(spread_unit="price"), PIP) == pytest.approx(2.5)
    assert to_pips(2.5, ExecutionPriceConfig(spread_unit="pips"), PIP) == 2.5
    assert to_pips(np.nan, ExecutionPriceConfig(), PIP) is None


# ------------------------------------------------------------------ bid / ask
def test_long_uses_ask_short_uses_bid_on_bid_charts():
    e = CFG.execution  # bid-based chart prices
    lng = executable_reference(1, 190.000, 2.0, e, PIP)
    sht = executable_reference(-1, 190.000, 2.0, e, PIP)
    assert lng["execution_side"] == "ASK" and lng["executable_reference_price"] == pytest.approx(190.02)
    assert sht["execution_side"] == "BID" and sht["executable_reference_price"] == pytest.approx(190.00)
    mid = executable_reference(1, 190.000, 2.0, replace(e, price_basis="mid"), PIP)
    assert mid["executable_reference_price"] == pytest.approx(190.01)
    ask = executable_reference(-1, 190.000, 2.0, replace(e, price_basis="ask"), PIP)
    assert ask["executable_reference_price"] == pytest.approx(189.98)


def test_unknown_spread_is_never_zero():
    r = executable_reference(1, 190.0, None, CFG.execution, PIP)
    assert r["spread_assumed"] and r["spread_used_pips"] == CFG.execution.unknown_spread_assumption_pips > 0
    assert r["executable_reference_price"] > 190.0
    with pytest.raises(ValueError):
        replace(CFG, execution=replace(CFG.execution, unknown_spread_assumption_pips=0.0)).validate()


# ------------------------------------------------------------------ slippage interface
def test_slippage_default_unknown_and_models_only_use_supplied_data():
    t = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    assert slippage(UnknownSlippage(), 1, t, 2.0, 0.3, PIP)["slippage_status"] == "UNKNOWN"
    assert slippage(FixedSlippage(0.5), 1, t, 2.0, 0.3, PIP)["slippage_estimate_pips"] == 0.5
    assert slippage(SpreadDependentSlippage(0.5), 1, t, None, 0.3, PIP)["slippage_status"] == "UNKNOWN"
    assert slippage(SpreadDependentSlippage(0.5), 1, t, 2.0, 0.3, PIP)["slippage_estimate_pips"] == 1.0
    assert slippage(VolatilityDependentSlippage(0.02), -1, t, 2.0, 0.3, PIP)["slippage_estimate_pips"] == pytest.approx(0.6)
    assert slippage(EmpiricalSlippage(()), 1, t, 2.0, 0.3, PIP)["slippage_status"] == "UNKNOWN"
    assert slippage(EmpiricalSlippage((0.1, 0.3, 0.5)), 1, t, 2.0, 0.3, PIP)["slippage_estimate_pips"] == pytest.approx(0.3)


# ------------------------------------------------------------------ gap / deterioration / abnormality
def test_gap_classes_and_direction():
    g = CFG.gap
    assert gap_info(1, 190.0, 190.01, 0.3, g)["gap_class"] == "NONE"
    assert gap_info(1, 190.0, 190.09, 0.3, g)["gap_class"] == "SMALL"
    big = gap_info(1, 190.0, 190.20, 0.3, g)
    assert big["gap_class"] == "LARGE" and big["gap_direction"] == "WITH_TRADE"
    assert gap_info(1, 190.0, 189.80, 0.3, g)["gap_direction"] == "AGAINST_THESIS"
    assert gap_info(-1, 190.0, 189.60, 0.3, g)["gap_class"] == "EXTREME"


def test_price_deterioration_and_window_status():
    d = CFG.deterioration
    ok = deterioration(1, 190.0, 190.02, 190.0, 0.3, PIP, d)
    assert ok["deterioration_status"] == "ACCEPTABLE" and ok["window_status"] == "AVAILABLE"
    assert ok["price_deterioration_pips"] == pytest.approx(2.0)
    mid = deterioration(1, 190.0, 190.12, 190.10, 0.3, PIP, d)
    assert mid["deterioration_status"] == "DETERIORATING" and mid["window_status"] == "DETERIORATING"
    bad = deterioration(1, 190.0, 190.25, 190.23, 0.3, PIP, d)
    assert bad["deterioration_status"] == "EXCESSIVE" and bad["window_status"] == "EXPIRED"
    assert bad["price_deterioration_score"] == 100.0
    better = deterioration(-1, 190.0, 190.05, 190.05, 0.3, PIP, d)  # short filled higher = better
    assert better["price_deterioration_pips"] < 0 and better["deterioration_status"] == "ACCEPTABLE"


def test_abnormal_movement_filter():
    a = CFG.abnormal
    calm = abnormality(1.0, 1.0, 0.0, 1.0, 0.0, a)
    assert calm["execution_abnormality_score"] == 0.0 and calm["status"] == "NORMAL"
    giant = abnormality(4.5, 1.0, 0.0, 1.0, 0.0, a)
    assert giant["execution_abnormality_score"] == 100.0 and "extreme_candle" in giant["drivers"]
    assert abnormality(1.0, 1.0, 0.0, 6.0, 0.0, a)["drivers"] == ["spread_explosion"]
    assert abnormality(1.0, 2.5, 0.0, 1.0, 0.0, a)["status"] == "ABNORMAL"


# ------------------------------------------------------------------ barriers
def _zone(lo, hi, strength, kind="resistance"):
    return SimpleNamespace(lower=lo, upper=hi, strength=strength, zone_type=kind)


def _swing(price, kind, sig=2.0):
    return SimpleNamespace(price=price, kind=kind, significance_atr=sig)


def test_barrier_stacking_clusters_h1_h4_once_and_keeps_several():
    ref, atr = 190.00, 0.30
    h1z = [_zone(190.60, 190.70, 70.0), _zone(191.50, 191.60, 80.0), _zone(189.0, 189.1, 90.0, "support")]
    h1s = [_swing(190.68, "high"), _swing(189.5, "low")]
    h4 = [{"kind": "strong_resistance_zone", "price": 190.72, "strength": 65.0},
          {"kind": "confirmed_swing_high", "price": 192.40, "strength": None}]
    out = stacked_barriers(1, ref, atr, h1z, h1s, h4, CFG.barriers, RoomConfig())
    cl = out["clusters"]
    assert len(cl) == 3  # 190.6-190.72 merged (H1 zone + H1 swing + H4 zone), 191.5, 192.4
    assert cl[0]["timeframes"] == ["H1", "H4"] and cl[0]["n_members"] == 3
    assert cl[0]["distance_atr"] == pytest.approx(2.0)
    assert all(c["price"] > ref for c in cl)  # only opposing (above) barriers for a long
    assert out["cluster_nearby"] is False and out["barrier_density_score"] > 0
    near = stacked_barriers(1, 190.45, atr, h1z, h1s, h4, CFG.barriers, RoomConfig())
    assert near["cluster_nearby"] and near["remaining_room_score"] < out["remaining_room_score"]
    assert near["barrier_density_score"] > out["barrier_density_score"]
    free = stacked_barriers(1, 195.0, atr, h1z, h1s, h4, CFG.barriers, RoomConfig())
    assert free["remaining_room_score"] == 100.0 and free["note"] == "NO_BARRIER_FOUND"


def test_weak_h1_zones_are_not_barriers():
    out = stacked_barriers(1, 190.0, 0.3, [_zone(190.3, 190.4, 20.0)], [], [], CFG.barriers, RoomConfig())
    assert out["nearest"] is None


# ------------------------------------------------------------------ session / news
def test_session_context_dst_aware():
    summer = session_context(pd.Timestamp("2023-07-12 13:30", tz="UTC"), gbpjpy_engine.config.SessionConfig().sessions)
    winter = session_context(pd.Timestamp("2023-01-11 13:30", tz="UTC"), gbpjpy_engine.config.SessionConfig().sessions)
    assert "london_new_york_overlap" in summer["session_overlap"]  # NY opens 12:00 UTC in summer
    assert "new_york" in summer["active_sessions"] and "new_york" in winter["active_sessions"]
    early = session_context(pd.Timestamp("2023-01-11 12:30", tz="UTC"), gbpjpy_engine.config.SessionConfig().sessions)
    assert "new_york" not in early["active_sessions"]  # NY opens 13:00 UTC in winter
    night = session_context(pd.Timestamp("2023-01-11 22:30", tz="UTC"), gbpjpy_engine.config.SessionConfig().sessions)
    assert night["session_label"] == "off_peak"


class _Calendar:
    def __init__(self, events):
        self._events = events

    def events(self, start, end, known_at):
        return list(self._events)


def test_news_unknown_without_provider_and_not_blocking_by_default():
    t = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    out = news_status(None, t, CFG.news)
    assert out["news_status"] == "UNKNOWN" and out["events"] == [] and not out["blocked"]
    assert news_status(None, t, replace(CFG.news, block_unknown=True))["blocked"]


def test_news_provider_events_and_no_future_schedule_leak():
    t = pd.Timestamp("2024-01-02 10:00", tz="UTC")
    boe = NewsEvent(t + pd.Timedelta(minutes=20), "GBP", "HIGH", "BoE rate decision", "central_bank")
    late = NewsEvent(t + pd.Timedelta(minutes=10), "JPY", "HIGH", "surprise", known_since=t + pd.Timedelta(hours=1))
    low = NewsEvent(t + pd.Timedelta(minutes=5), "GBP", "LOW", "minor")
    usd = NewsEvent(t + pd.Timedelta(minutes=5), "USD", "HIGH", "NFP")
    out = news_status(_Calendar([boe, late, low, usd]), t, CFG.news)
    assert out["news_status"] == "EVENT_IMMINENT" and [e["name"] for e in out["events"]] == ["BoE rate decision"]
    assert not out["blocked"]
    assert news_status(_Calendar([boe]), t, replace(CFG.news, block_on_event=True))["blocked"]
    assert news_status(_Calendar([late]), t, CFG.news)["news_status"] == "CLEAR"


# ------------------------------------------------------------------ timing / quality scores
def test_freshness_classes():
    f = CFG.freshness
    assert freshness(0, 1, 0.0, 0.0, 0.0, f)["freshness_class"] == "FRESH"
    assert freshness(1, 6, 0.4, 1.0, 1.5, f)["freshness_class"] == "AGING"
    assert freshness(2, 10, 0.8, 2.0, 3.0, f)["freshness_class"] == "STALE"
    assert freshness(3, 12, 1.0, 2.5, 4.0, f)["freshness_class"] == "EXPIRED"
    s = [freshness(k, 1, 0.0, 0.0, float(k), f)["signal_freshness_score"] for k in range(4)]
    assert s == sorted(s, reverse=True)


def test_chase_prevention_scores_and_classes():
    c = CFG.chase
    low = chase_risk(1, 190.05, 190.0, 189.8, 4.0, 5.0, 0.3, c)
    assert low["chase_class"] == "LOW" and not low["rejects"]
    ran = chase_risk(1, 191.3, 190.4, 189.8, 3.0, 0.5, 0.3, c)
    assert ran["chase_class"] in ("HIGH", "EXTREME") and ran["rejects"]
    assert ran["recent_impulse_travelled_pct"] > 100
    assert chase_risk(1, 191.3, 190.4, 189.8, 3.0, 0.5, 0.3, replace(c, reject_at="EXTREME"))["rejects"] == \
        (ran["chase_class"] == "EXTREME")


def test_entry_extension_states():
    x = CFG.extension
    calm = entry_extension(1, 190.1, 190.0, 189.7, 1.0, 40.0, "up", 30.0, "up", 0.3, x)
    assert calm["extension_state"] == "NOT_EXTENDED"
    hot = entry_extension(1, 191.0, 190.0, 189.0, 7.0, 99.0, "up", 95.0, "up", 0.3, x)
    assert hot["extension_state"] == "OVEREXTENDED"
    # extension against the trade direction does not count
    assert entry_extension(-1, 191.0, 190.0, 189.0, 0.0, 99.0, "up", 95.0, "up", 0.3, x)["entry_extension_score"] == 0.0


def test_market_quality_conflict_and_quality():
    good = entry_market_quality(30.0, 70.0, "normal", 0.0)["entry_market_quality_score"]
    bad = entry_market_quality(75.0, 20.0, "extreme", 1.2)["entry_market_quality_score"]
    assert good > bad
    assert entry_conflict([]) == 0.0
    assert entry_conflict(["ROOM_COLLAPSED"]) == 30.0
    assert entry_conflict(["ROOM_COLLAPSED", "POOR_SPREAD"]) == pytest.approx(100 * (1 - 0.7 * 0.8))
    fams = {k: 80.0 for k in ("CONFIRMATION", "TIMING", "FRESHNESS", "PRICE_QUALITY", "MARKET_QUALITY",
                              "STRUCTURAL_INTEGRITY", "ROOM", "EXECUTION_CONDITIONS", "CONFLICT")}
    assert entry_quality(fams, 0.0, CFG.scoring) == 80.0
    assert entry_quality(fams, 40.0, CFG.scoring) == pytest.approx(80.0 * 0.8)


# ------------------------------------------------------------------ config
def test_entry_config_documented_yaml_and_validation():
    rows = describe_config(EntryConfig())
    assert all(r["doc"] and len(r["doc"]) > 10 for r in rows)
    assert load_entry_config(ROOT / "config" / "entry_default.yaml").to_dict() == EntryConfig().to_dict()
    assert load_entry_config().config_hash() == EntryConfig().config_hash()
    with pytest.raises(KeyError):
        entry_config_from_dict({"spread": {"nope": 1}})
    with pytest.raises(ValueError):
        entry_config_from_dict({"lifecycle": {"max_active_per_side": 2}})
    with pytest.raises(ValueError):
        entry_config_from_dict({"policy": {"policies": [["TREND_PULLBACK_CONTINUATION", ["STRUCTURAL_BREAK_CONFIRMATION"], 55, 45]]}})
    assert entry_config_from_dict({"spread": {"block_statuses": ["EXTREME"]}}).spread.block_statuses == ("EXTREME",)


def test_family_specific_policies_exist_for_every_setup_family():
    from gbpjpy_engine.h1.setups import FAMILIES

    pols = {f: CFG.policy_for(f) for f in FAMILIES}
    assert len({p["allowed"] for p in pols.values()}) == len(FAMILIES)  # not one identical rule for all families
    assert "SWEEP_RECLAIM_CONFIRMATION" in pols["LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"]["allowed"]
    assert "SWEEP_RECLAIM_CONFIRMATION" not in pols["TREND_PULLBACK_CONTINUATION"]["allowed"]
    assert "BREAK_RETEST_CONFIRMATION" in pols["BREAK_RETEST_CONTINUATION"]["allowed"]
    assert "COMPRESSION_EXPANSION_CONFIRMATION" in pols["COMPRESSION_EXPANSION_IN_H4_DIRECTION"]["allowed"]
    assert isinstance(NewsConfig().currencies, tuple) and set(NewsConfig().currencies) == {"GBP", "JPY"}
    assert SpreadConfig().block_unknown is False
