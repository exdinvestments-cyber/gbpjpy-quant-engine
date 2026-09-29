"""Phase 1H statistics: R metrics, expectancy, drawdown, confidence intervals, breakdowns, concentration, Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.risk.research import RiskOfRuinInputs
from gbpjpy_engine.validation import metrics as mt
from gbpjpy_engine.validation import montecarlo as mc


def trades(r, start="2021-01-04", step_days=3, one_r=50.0, direction=None, family=None):
    t = pd.date_range(start, periods=len(r), freq=f"{step_days}D", tz="UTC")
    return pd.DataFrame({"net_R": r, "gross_R": np.asarray(r) + 0.05, "entry_time": t.astype(str),
                         "exit_time": (t + pd.Timedelta(hours=10)).astype(str), "holding_hours": 10.0, "one_R_pips": one_r,
                         "direction": direction or ["LONG", "SHORT"] * (len(r) // 2) + ["LONG"] * (len(r) % 2),
                         "setup_family": family or "TREND_PULLBACK_CONTINUATION", "mae_R": 0.5, "mfe_R": 1.0})


def test_r_distribution_and_expectancy_two_ways():
    r = [2.0, -1.0, -1.0, 3.0, 0.0, -1.2]
    d = mt.r_distribution(r)
    assert d["n"] == 6 and d["total_R"] == pytest.approx(1.8) and d["tail"]["worse_than_minus_1R"] == 1
    e = mt.expectancy(r)
    assert e["wins"] == 2 and e["losses"] == 3 and e["break_even"] == 1
    assert e["expectancy_R_direct"] == pytest.approx(0.3) and e["expectancy_R_formula"] == pytest.approx(0.3)
    assert mt.profit_factor(r) == pytest.approx(5.0 / 3.2) and mt.profit_factor([1.0, 2.0]) is None


def test_drawdown_duration_recovery_and_streaks():
    r = [1, -1, -1, -1, 2, 2, -1]
    d = mt.drawdown(r, pd.date_range("2024-01-01", periods=7, freq="D", tz="UTC"))
    assert d["max_drawdown_R"] == 3.0 and d["trades_to_recover"] == 2 and d["max_drawdown_duration_days"] == 3.0
    assert mt.streaks(r) == {"max_consecutive_wins": 2, "max_consecutive_losses": 3}
    eps = mt.drawdown_episodes(r)
    assert eps[0]["depth_R"] == 3.0 and eps[-1]["recovered_at"] is None


def test_bootstrap_is_seeded_and_block_bootstrap_supported():
    r = np.random.default_rng(0).normal(0.1, 1.0, 120)
    a = mt.bootstrap(r, lambda x: float(x.mean()), 500, seed=7)
    b = mt.bootstrap(r, lambda x: float(x.mean()), 500, seed=7)
    c = mt.bootstrap(r, lambda x: float(x.mean()), 500, seed=8)
    assert a == b and a != c and a["ci_low"] < a["estimate"] < a["ci_high"] and 0 <= a["p_le_zero"] <= 1
    blk = mt.bootstrap(r, lambda x: float(x.mean()), 500, seed=7, block=5)
    assert blk["method"] == "moving-block bootstrap" and blk["seed"] == 7
    ci = mt.confidence_intervals(r, seed=1, n_boot=300)
    assert set(ci) >= {"win_rate_wilson_95", "win_rate", "mean_R", "expectancy_R", "profit_factor", "sample"}
    lo, hi = mt.wilson(30, 100)
    assert lo < 0.3 < hi


def test_small_sample_warning_levels():
    assert mt.sample_warning(10)["level"] == "INSUFFICIENT_SAMPLE"
    assert mt.sample_warning(50)["level"] == "SMALL_SAMPLE"
    assert mt.sample_warning(150)["level"] == "MODERATE_SAMPLE"
    assert "NOT evidence" in mt.sample_warning(400)["message"]
    assert mt.bootstrap([1.0], lambda x: x.mean())["status"] == "NOT_ESTIMABLE"


def test_summary_and_calendar_breakdowns_keep_losing_periods():
    r = [1.0] * 40 + [-1.0] * 40 + [0.5] * 40
    df = trades(r, start="2021-01-04", step_days=9)
    s = mt.summarise(df)
    assert s["trades"] == 120 and s["expectancy_R"] == pytest.approx(np.mean(r)) and s["sample"]["level"] == "MODERATE_SAMPLE"
    cal = mt.calendar_breakdowns(df)
    assert any(v["total_R"] < 0 for v in cal["year"].values())  # the losing year is reported, not hidden
    assert len(cal["quarter"]) >= 8 and len(cal["month"]) >= 24
    ra = mt.risk_adjusted(df)
    assert ra["status"] == "OK" and "annualisation" in ra and ra["sharpe_per_trade"] is not None


def test_concentration_and_remove_best_trades():
    r = [10.0, 8.0] + [-0.2] * 30
    c = mt.concentration(trades(r))
    assert c["best_trade_share"] == pytest.approx(10 / 12) and c["flag"].startswith("OUTLIER_DEPENDENT")
    assert c["remove_best"]["without_best_1"]["total_R"] == pytest.approx(sum(r) - 10)
    assert c["remove_best"]["without_best_10"]["trades"] == len(r) - 10
    small = mt.concentration(trades([1.0, -1.0]))
    assert small["remove_best"]["without_best_5"]["status"].startswith("NOT_MEANINGFUL")


def test_distribution_stability_frequency_and_exposure():
    df = trades([1.0, -1.0] * 30)
    st = mt.distribution_stability(df)
    assert len(st["segments"]) == 3 and all(0 <= k["ks_statistic"] <= 1 for k in st["ks"])
    f = mt.trade_frequency(df, df["entry_time"].iloc[0], df["exit_time"].iloc[-1])
    assert f["per_week_mean"] == pytest.approx(60 / ((pd.Timestamp(df["exit_time"].iloc[-1]) - pd.Timestamp(df["entry_time"].iloc[0])).days / 7), rel=0.05)
    x = mt.exposure_fraction(["2024-01-01 00:00", "2024-01-01 05:00"], ["2024-01-01 10:00", "2024-01-01 12:00"],
                             "2024-01-01", "2024-01-02")
    assert x == pytest.approx(0.5)


# --------------------------------------------------------------------------- Monte Carlo
R = np.array([2.0, -1.0, -1.0, 1.5, -1.0, 2.5, -1.0, -1.0, 3.0, -1.0] * 5)


def test_sequence_monte_carlo_is_seeded_and_covers_streaks_and_drawdowns():
    a = mc.sequence_monte_carlo(R, 300, seed=3)
    assert a == mc.sequence_monte_carlo(R, 300, seed=3) and a["seed"] == 3
    for k in ("ending_R", "max_drawdown_R", "max_losing_streak", "longest_underwater_trades", "fraction_time_underwater"):
        assert a[k]["p05"] <= a[k]["median"] <= a[k]["p95"]
    perm = mc.sequence_monte_carlo(R, 200, seed=1, method="PERMUTATION")
    assert perm["ending_R"]["min"] == pytest.approx(R.sum()) == pytest.approx(perm["ending_R"]["max"])  # same trades
    assert mc.sequence_monte_carlo(R, 200, 1, "BLOCK", 5)["block"] == 5
    assert "regime" in a["caveat"] and mc.sequence_monte_carlo([1, 2], 10)["status"].startswith("NOT_MEANINGFUL")


def test_cost_missed_worse_fill_and_tail_stress():
    df = trades(list(R))
    cs = mc.cost_stress(df)
    assert cs["grid"][0]["expectancy_R"] > cs["grid"][-1]["expectancy_R"]
    be = cs["breakeven_extra_round_turn_pips"]
    assert np.mean(R - be / 50.0) == pytest.approx(0.0, abs=1e-9)
    assert mc.random_cost_stress(df, 200, 5) == mc.random_cost_stress(df, 200, 5)
    m = mc.missed_trades(R, 0.25, 300, 2)
    assert m["fraction_removed"] == 0.25 and m["seed"] == 2
    w = mc.worse_fills(df, 2, 2, 300, 1)
    assert w["expectancy_R"]["max"] < R.mean()  # worse fills only ever hurt
    t = mc.tail_stress(R, 0.2, (0.5, 2.0), 300, 1)
    assert t["worst_trade_R"]["min"] < -1.5 and t["expectancy_R"]["max"] <= R.mean()


def test_risk_of_ruin_is_never_reported_as_zero_and_uses_the_phase1f_interface():
    inp = RiskOfRuinInputs(win_loss_distribution_R=tuple(R), ruin_threshold_percent=50.0)
    rr = mc.risk_of_ruin(inp, 0.25, 100, 500, seed=4)
    assert rr["probability_of_ruin"]["observed"] == 0 and "<" in rr["probability_of_ruin"]["reported"]
    bad = RiskOfRuinInputs(win_loss_distribution_R=(-1.0,) * 20, ruin_threshold_percent=20.0)
    rb = mc.risk_of_ruin(bad, 2.0, 50, 200, seed=4)
    assert rb["probability_of_ruin"]["observed"] == 1.0
    assert mc.risk_of_ruin(RiskOfRuinInputs(), 1.0)["status"].startswith("NOT_ESTIMABLE")
    blk = mc.risk_of_ruin(inp, 1.0, 100, 200, seed=1, block=5)
    assert "block" in blk["method"]


def test_compounding_is_separate_and_policy_comparison_selects_nothing():
    df = trades(list(R))
    c1, c2 = mc.compounding(df, 0.5), mc.compounding(df, 1.0)
    assert c2["max_drawdown_pct"] > c1["max_drawdown_pct"] and "judge the edge from R" in c1["note"]
    pc = mc.risk_policy_comparison(R, (0.25, 0.5, 1.0), 100, 200, seed=2)
    assert set(pc["policies"]) == {"0.25%", "0.5%", "1.0%"} and pc["selection_rule"].startswith("NONE")
    cap = mc.capital_scale([30, 60, 90], 1.0, max_volume=50)
    assert cap["rows"][-1]["share_trades_above_max_volume"] > 0 and "linearly" in cap["note"]
