"""Synthetic regime tests: ENGINEERING checks that descriptive states are logical.

These are not evidence of profitability.  Each check inspects the scenario
segment after a settling period (60 bars) so indicators reflect the scenario.
"""

from __future__ import annotations

import pytest

from gbpjpy_engine.synthetic import generate_scenario

SETTLE = 60
RANGE_FAMILY = {"RANGE", "HIGH_VOLATILITY_RANGE", "LOW_VOLATILITY_COMPRESSION"}
BULL = {"STRONG_BULL_TREND", "BULL_TREND", "WEAK_BULL_TREND"}
BEAR = {"STRONG_BEAR_TREND", "BEAR_TREND", "WEAK_BEAR_TREND"}


def seg(scenario_results, name):
    sc, res = scenario_results[name]
    return sc, res, res.features.iloc[sc.segment_start + SETTLE :]


def test_clean_uptrend(scenario_results):
    _, _, f = seg(scenario_results, "clean_uptrend")
    assert (f["structure_state"] == "bullish").mean() > 0.9
    assert f["regime"].isin(BULL).mean() > 0.9
    assert (f["regime"] == "STRONG_BULL_TREND").mean() > 0.7
    assert (f["h4_bias"] == "LONG").mean() > 0.8
    assert (f["h4_bias"] == "SHORT").sum() == 0
    assert f["chop_score"].mean() < 30
    assert f["market_quality"].isin(["clean_trend", "trend_with_noise"]).mean() > 0.9
    assert f["structure_quality_score"].mean() > 65
    assert f["directional_efficiency"].mean() > 0.35


def test_clean_downtrend(scenario_results):
    _, _, f = seg(scenario_results, "clean_downtrend")
    assert (f["structure_state"] == "bearish").mean() > 0.9
    assert f["regime"].isin(BEAR).mean() > 0.9
    assert (f["h4_bias"] == "SHORT").mean() > 0.8
    assert (f["h4_bias"] == "LONG").sum() == 0


def test_sideways_range(scenario_results):
    _, _, f = seg(scenario_results, "sideways_range")
    assert f["regime"].isin(RANGE_FAMILY | {"TRANSITION", "UNCLEAR"}).mean() > 0.7
    assert not f["regime"].str.startswith("STRONG").any()
    assert (f["h4_bias"] == "NEUTRAL").mean() > 0.9
    assert f["chop_score"].mean() > 50
    assert f["directional_efficiency"].mean() < 0.3


def test_volatile_range(scenario_results):
    _, _, f = seg(scenario_results, "volatile_range")
    assert (f["regime"] == "HIGH_VOLATILITY_RANGE").mean() > 0.4
    assert f["regime"].isin(RANGE_FAMILY).mean() > 0.6
    assert f["volatility_regime"].isin(["high", "extreme"]).mean() > 0.6
    assert (f["h4_bias"] == "NEUTRAL").mean() > 0.9
    assert f["chop_score"].mean() > 60


def test_low_volatility_compression(scenario_results):
    sc, res, f = seg(scenario_results, "low_volatility_compression")
    late = res.features.iloc[sc.segment_start + 150 :]
    assert (late["regime"] == "LOW_VOLATILITY_COMPRESSION").mean() > 0.6
    assert (late["volatility_regime"] == "very_low").mean() > 0.9
    assert (f["h4_bias"] == "NEUTRAL").mean() > 0.9
    assert (late["volatility_trend"] != "expanding").all()


def test_trend_reversal(scenario_results):
    sc, res, _ = seg(scenario_results, "trend_reversal")
    f = res.features
    k = sc.events["reversal_index"]
    pre = f.iloc[k - 80 : k]
    post = f.iloc[k + 60 :]
    assert (pre["h4_bias"] == "LONG").mean() > 0.8
    assert (post["h4_bias"] == "SHORT").mean() > 0.7
    assert (post["h4_bias"] == "LONG").sum() == 0
    # the structure must pass through a non-bullish state before turning bearish
    window = f.iloc[k : k + 60]["structure_state"]
    first_bear = window[window == "bearish"].index.min()
    assert first_bear is not None
    assert (window.loc[:first_bear] != "bullish").any()
    # a bearish structural break occurs after the reversal
    assert any(b.direction == "bearish" and k <= b.break_index <= k + 80 for b in res.structure.breaks)


def test_false_breakout(scenario_results):
    sc, res, _ = seg(scenario_results, "false_breakout")
    k = sc.events["break_index"]
    brk = [b for b in res.structure.breaks if b.break_index == k and b.direction == "bullish"]
    assert brk, "the fake breakout bar must be registered as a structural break"
    b = brk[0]
    assert b.status == "INVALIDATED"
    assert b.status_history[-1][0] == k + 1
    f = res.features
    assert f["last_break_status"].iloc[k] == "CANDIDATE"  # not known to be false at the time
    assert f["last_break_status"].iloc[k + 1] == "INVALIDATED"
    assert "BREAK_REJECTED" in f["reason_codes"].iloc[k + 1]
    # no bullish bias results from the failed breakout
    assert (f["h4_bias"].iloc[k : k + 10] != "LONG").all()


def test_volatility_shock(scenario_results):
    sc, res, _ = seg(scenario_results, "volatility_shock")
    f = res.features
    k = sc.events["shock_index"]
    assert bool(f["volatility_shock"].iloc[k])
    assert f["shock_severity"].iloc[k] >= 90
    assert f["regime"].iloc[k] == "VOLATILITY_SHOCK"
    assert "VOLATILITY_SHOCK" in f["reason_codes"].iloc[k]
    before = f.iloc[k - 100 : k]
    assert before["volatility_shock"].mean() < 0.05
    assert f["candle_class"].iloc[k] == "strong_bearish_close"
    assert bool(f["abnormal_expansion"].iloc[k])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scenarios_robust_to_seed(engine, seed):
    up = generate_scenario("clean_uptrend", seed=seed)
    f = engine.run(up.bars).features.iloc[up.segment_start + SETTLE :]
    assert (f["h4_bias"] == "LONG").mean() > 0.8
    rg = generate_scenario("volatile_range", seed=seed)
    g = engine.run(rg.bars).features.iloc[rg.segment_start + SETTLE :]
    assert (g["h4_bias"] == "NEUTRAL").mean() > 0.9
    assert g["regime"].isin(RANGE_FAMILY).mean() > 0.6
