"""Phase 1H cost models and historical execution simulator (hand-computed expectations)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.validation import (CommissionModel, CostScenario, MissingSpread, SlippageModel, SpreadModel, SwapModel,
                                      simulate_trade, standard_scenarios)
from gbpjpy_engine.validation.simulator import ExitPlan
from validation_helpers import T0, bars, proposal

SC = standard_scenarios("PER_BAR")
ZERO, BASE, STRESS = SC["ZERO_COST_DIAGNOSTIC"], SC["BASELINE"], SC["STRESSED"]


# --------------------------------------------------------------------------- cost models
def test_spread_models_never_silently_zero():
    assert SpreadModel("HISTORICAL").spread_pips(2.5) == (2.5, "HISTORICAL")
    with pytest.raises(MissingSpread):
        SpreadModel("HISTORICAL").spread_pips(np.nan)
    assert SpreadModel("HISTORICAL", assumed_pips=3.0).spread_pips(None) == (3.0, "ASSUMED_FALLBACK")
    assert SpreadModel("ASSUMED", 3.0, 1.5, 1.0).spread_pips(9.9) == (5.5, "ASSUMED")
    with pytest.raises(ValueError):
        CostScenario("X", "BASELINE", spread=SpreadModel("ZERO_DIAGNOSTIC"))
    with pytest.raises(ValueError):
        CostScenario("X", "STRESSED", slippage=SlippageModel("ZERO_SLIPPAGE_DIAGNOSTIC_ONLY"))
    assert "DIAGNOSTIC UPPER BOUND" in ZERO.describe()["presentation"]


def test_standard_scenarios_label_assumptions():
    s = standard_scenarios("NONE")
    assert s["BASELINE"].spread.kind == "ASSUMED" and s["BASELINE"].spread.assumed_pips == 3.0
    assert s["BASELINE"].commission.kind == "SCENARIO" and "UNKNOWN" in s["BASELINE"].commission.source
    assert standard_scenarios("PER_BAR", known_commission_pips=0.7)["BASELINE"].commission.kind == "KNOWN"
    assert s["STRESSED"].realism == "STRESSED" and s["STRESSED"].swap.kind == "SCENARIO"


def test_commission_kinds():
    assert CommissionModel("UNKNOWN").pips() == (None, "UNKNOWN")
    assert CommissionModel("KNOWN", 0.7, "broker").pips() == (0.7, "KNOWN")
    with pytest.raises(ValueError):
        CommissionModel("ZERO_DECLARED").pips()  # a zero commission must cite its source
    assert CommissionModel("ZERO_DECLARED", source="raw-spread account terms").pips() == (0.0, "ZERO_DECLARED")


def test_slippage_models_are_adverse_and_deterministic():
    assert SlippageModel("ZERO_SLIPPAGE_DIAGNOSTIC_ONLY").side_pips(2, 30, "k") == 0.0
    assert SlippageModel("FIXED", 0.7).side_pips(2, 30, "k") == 0.7
    assert SlippageModel("SPREAD_DEPENDENT", spread_fraction=0.5).side_pips(3, 30, "k") == 1.5
    assert SlippageModel("VOLATILITY_DEPENDENT", atr_fraction=0.02).side_pips(3, 40, "k") == 0.8
    e = SlippageModel("EMPIRICAL", samples_pips=(0.0, 0.5, 1.0, 3.0), seed=4)
    assert e.side_pips(2, 30, "a") == e.side_pips(2, 30, "a") and e.side_pips(2, 30, "a") in (0.0, 0.5, 1.0, 3.0)
    assert SlippageModel("FIXED", -1.0).side_pips(2, 30, "k") == 0.0  # never favourable in research costs


def test_swap_nights_and_triple_day():
    sw = SwapModel("SCENARIO", -0.5, 0.3)
    rolls, charged = sw.nights(T0, T0 + pd.Timedelta(days=3))  # Mon 00:00 -> Thu 00:00
    assert (rolls, charged) == (3, 5)  # Mon, Tue, Wed(x3)
    assert sw.pips(1, T0, T0 + pd.Timedelta(days=3))[0] == pytest.approx(-2.5)
    assert sw.pips(-1, T0, T0 + pd.Timedelta(days=3))[0] == pytest.approx(1.5)
    assert SwapModel("UNKNOWN").pips(1, T0, T0 + pd.Timedelta(days=3))[:2] == (None, "UNKNOWN")
    assert sw.nights(T0, T0 + pd.Timedelta(hours=5)) == (0, 0)


# --------------------------------------------------------------------------- simulator
def win_bars():
    return bars([(190.00, 190.10, 189.95, 190.05), (190.05, 191.10, 190.00, 191.00), (191.0, 191.1, 190.9, 191.0)])


def test_long_target_bid_ask_and_costs():
    p = proposal()
    z = simulate_trade(win_bars(), 0, p, ZERO)
    b = simulate_trade(win_bars(), 0, p, BASE)
    assert z.exit_reason == b.exit_reason == "TARGET" and z.exit_fill == 191.02
    assert z.entry_fill == 190.00 and b.entry_fill == pytest.approx(190.025)  # ASK (+2 pips) + 0.5 pip slippage
    assert b.net_pips == pytest.approx(99.5 - 0.5) and b.net_R == pytest.approx(99.0 / 50)
    assert b.gross_pips == pytest.approx(102.0) and z.net_R == pytest.approx(z.gross_R)
    assert b.spread_status == "HISTORICAL" and b.commission_status == "SCENARIO" and b.swap_status == "UNKNOWN"
    assert b.mfe_pips == pytest.approx((191.02 - 190.025) / 0.01) and b.mae_pips == pytest.approx((190.025 - 189.95) / 0.01)
    assert b.holding_bars == 2 and b.capture_ratio == pytest.approx(b.net_R / b.mfe_R, abs=1e-3)


def test_short_exits_on_the_ask():
    p = proposal("SHORT", entry=190.00, stop=190.50, target=189.00)
    rows = [(190.00, 190.05, 189.90, 189.95), (189.95, 190.47, 189.80, 190.0)]  # BID high 190.47 -> ASK 190.49 < stop
    t = simulate_trade(bars(rows), 0, p, BASE)
    assert t.exit_reason == "END_OF_DATA" and "OPEN_AT_END" in t.flags
    rows2 = [(190.00, 190.05, 189.90, 189.95), (189.95, 190.49, 189.80, 190.0)]  # ASK 190.51 >= stop
    t2 = simulate_trade(bars(rows2), 0, p, BASE)
    assert t2.exit_reason == "STOP" and t2.entry_fill == pytest.approx(189.995)  # BID - 0.5 pip
    assert t2.net_R < -1.0  # spread + slippage + commission make a stop worse than -1R


def test_gap_through_stop_loses_more_than_one_r_and_gap_through_target_is_not_a_gift():
    p = proposal()
    t = simulate_trade(bars([(190.0, 190.1, 189.9, 190.0), (189.30, 189.4, 189.2, 189.3)]), 0, p, BASE)
    assert t.exit_reason == "GAP_STOP" and "GAP_THROUGH_STOP" in t.flags and t.net_R < -1.3
    g = simulate_trade(bars([(190.0, 190.1, 189.9, 190.0), (191.50, 191.6, 191.4, 191.5)]), 0, p, BASE)
    assert g.exit_reason == "TARGET" and g.exit_fill == 191.02


def test_ambiguous_bar_is_never_resolved_favourably():
    rows = [(190.0, 190.1, 189.9, 190.0), (190.0, 191.2, 189.3, 190.0)]
    p = proposal()
    c = simulate_trade(bars(rows), 0, p, BASE, ambiguity_policy="CONSERVATIVE")
    a = simulate_trade(bars(rows), 0, p, BASE, ambiguity_policy="AMBIGUOUS")
    u = simulate_trade(bars(rows), 0, p, BASE, ambiguity_policy="LOWER_TIMEFRAME_REQUIRED")
    assert c.exit_reason == a.exit_reason == u.exit_reason == "STOP"
    assert a.ambiguity == "AMBIGUOUS" and u.ambiguity == "UNRESOLVED" and c.ambiguity == "AMBIGUOUS_CONSERVATIVE"


def test_lower_timeframe_resolves_sequencing_only():
    rows = [(190.0, 190.1, 189.9, 190.0), (190.0, 191.2, 189.3, 190.0)]
    t1 = T0 + pd.Timedelta(hours=1)
    m15_target_first = bars([(190.0, 191.1, 189.95, 191.0), (191.0, 191.0, 189.3, 189.4), (189.4, 189.5, 189.35, 189.4),
                             (189.4, 190.0, 189.35, 190.0)], t1, freq="15min")
    m15_stop_first = bars([(190.0, 190.05, 189.3, 189.4), (189.4, 191.2, 189.35, 191.1), (191.1, 191.1, 190.9, 191.0),
                           (191.0, 191.0, 190.0, 190.0)], t1, freq="15min")
    p = proposal()
    a = simulate_trade(bars(rows), 0, p, BASE, ambiguity_policy="LOWER_TIMEFRAME_REQUIRED", ltf=m15_target_first)
    b = simulate_trade(bars(rows), 0, p, BASE, ambiguity_policy="LOWER_TIMEFRAME_REQUIRED", ltf=m15_stop_first)
    assert a.exit_reason == "TARGET" and a.ambiguity == "RESOLVED_LTF"
    assert b.exit_reason == "STOP" and b.ambiguity == "RESOLVED_LTF"
    assert a.entry_fill == b.entry_fill  # the lower timeframe never changes the decision or the entry


@pytest.mark.parametrize("ptype,entry", [("BID", 190.025), ("ASK", 190.005), ("MID", 190.015), ("UNKNOWN", 190.025)])
def test_price_representation_is_explicit(ptype, entry):
    t = simulate_trade(win_bars(), 0, proposal(), BASE, ptype)
    assert t.entry_fill == pytest.approx(entry)
    assert ("PRICE_TYPE_UNKNOWN" in t.flags) == (ptype == "UNKNOWN")


def test_unknown_commission_leaves_net_undefined_and_swap_scenario_applies():
    sc = CostScenario("X", "BASELINE", SpreadModel("HISTORICAL"), CommissionModel("UNKNOWN"), SlippageModel("FIXED", 0.5))
    t = simulate_trade(win_bars(), 0, proposal(), sc)
    assert t.net_R is None and "COMMISSION_UNKNOWN" in t.flags
    long_hold = bars([(190.0, 190.1, 189.9, 190.0)] * 60)
    s2 = simulate_trade(long_hold, 0, proposal(), STRESS)
    assert s2.swap_nights >= 2 and s2.swap_pips < 0 and s2.exit_reason == "END_OF_DATA"


def test_max_holding_exit_plan_and_partial_architecture():
    t = simulate_trade(bars([(190.0, 190.1, 189.9, 190.0)] * 10), 0, proposal(), BASE, exit_plan=ExitPlan(max_holding_bars=3))
    assert t.exit_reason == "MAX_HOLDING" and t.holding_bars == 4 and "TIME_CAPPED" in t.flags
    assert ExitPlan().partial_fractions == ()  # partial exits exist architecturally but are not active by default
