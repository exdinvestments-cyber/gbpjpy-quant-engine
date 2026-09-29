"""Phase 1H integration: replay, ledgers, costs, account simulation, diagnostics, robustness, runner.

The market data here is the SYNTHETIC Phase 1D fixture - it exercises the machinery only and says nothing about
real GBPJPY behaviour.  Every run through the runner is labelled SYNTHETIC_TEST_ONLY / NOT_VALIDATED."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.config import describe_config
from gbpjpy_engine.validation import (DataStore, DatasetProvenance, ImportSpec, StrategyConfigs, account_simulation,
                                      check_language, cost_scenario_ledgers, first_real_data_run, objective_analysis,
                                      run_pipeline, run_validation, signal_ledger, standard_scenarios, strategy_trades, warmup)
from gbpjpy_engine.validation import diagnostics as dg
from gbpjpy_engine.validation import robustness as rb
from gbpjpy_engine.validation.objective import objective_is_isolated

SCEN = standard_scenarios("PER_BAR")


@pytest.fixture(scope="session")
def syn_h1():
    from gbpjpy_engine.synthetic_entry import generate_entry_scenario

    return generate_entry_scenario("entry_uptrend").h1


@pytest.fixture(scope="session")
def pipe(syn_h1):
    return run_pipeline(syn_h1)


@pytest.fixture(scope="session")
def ledgers(pipe):
    return cost_scenario_ledgers(pipe, SCEN, "BID")


@pytest.fixture(scope="session")
def syn_store(tmp_path_factory, syn_h1):
    d = tmp_path_factory.mktemp("store")
    f = d / "synthetic_h1.csv"
    syn_h1.drop(columns=["source"]).to_csv(f, index=False)
    store = DataStore(d / "store")
    prov = DatasetProvenance(provider="synthetic_fixture", symbol="GBPJPY", timeframe="H1", source_timezone="UTC",
                             price_type="BID", volume_type="TICK_VOLUME", spread_availability="PER_BAR", spread_unit="PIPS",
                             is_synthetic=True, retrieved_at="2024-01-01T00:00:00+00:00")
    p = store.import_raw(f, ImportSpec(prov))
    return store, p.dataset_id


@pytest.fixture(scope="session")
def validation_run(syn_store, tmp_path_factory):
    store, ds = syn_store
    return run_validation(store, ds, workdir=tmp_path_factory.mktemp("res"), development=0.5, validation=0.45,
                          allow_synthetic_for_testing=True, n_null_runs=5, now="2024-01-01T00:00:00+00:00")


# --------------------------------------------------------------------------- replay, warm-up, ledgers
def test_warmup_is_determined_and_nothing_earlier_is_scored(pipe, ledgers):
    w = warmup(pipe)
    assert w["complete"] and w["index"] >= max(w["components"].values())
    assert all(t.entry_index >= w["index"] for t in ledgers["BASELINE"].trades)
    early = run_pipeline(pipe.h1_bars.iloc[:300].reset_index(drop=True))
    assert not warmup(early)["complete"] and strategy_trades(early, SCEN["BASELINE"], "BID").trades == []


def test_signal_ledger_includes_rejected_opportunities(pipe):
    led = signal_ledger(pipe)
    assert {"H1_SETUP", "H4_GATED"} <= set(led["stage"])
    for c in ("h4_regime", "h4_permission", "setup_family", "setup_score", "entry_decision", "entry_quality_score",
              "trade_proposal_id", "trade_decision", "risk_decision", "execution_decision", "reason_codes"):
        assert c in led.columns
    assert (led["setup_state"].isin(["INVALIDATED", "EXPIRED", "GATED_BY_H4"])).sum() > 0
    assert led["trade_decision"].eq("PROPOSE_TRADE").sum() == len(pipe.trade.proposals)
    assert led["trade_quality_score"].notna().sum() == len(pipe.trade.proposals)


def test_trade_ledger_records_everything(ledgers):
    t = ledgers["BASELINE"].trades[0].to_dict()
    for k in ("trade_id", "direction", "setup_family", "entry_time", "exit_time", "entry_fill", "stop", "target",
              "spread_entry_pips", "commission_pips", "slippage_entry_pips", "swap_pips", "gross_pips", "net_pips", "gross_R",
              "net_R", "mae_pips", "mfe_pips", "mae_R", "mfe_R", "mae_atr", "holding_hours", "exit_reason"):
        assert k in t


def test_critical_cost_test_identical_signals_different_net(pipe, ledgers):
    props = hashlib.sha256(json.dumps(pipe.trade.proposals, sort_keys=True, default=str).encode()).hexdigest()
    z, b, s = ledgers["ZERO_COST_DIAGNOSTIC"], ledgers["BASELINE"], ledgers["STRESSED"]
    assert z.path == b.path == s.path and len(b.trades) > 0
    assert [t.entry_time for t in z.trades] == [t.entry_time for t in b.trades] == [t.entry_time for t in s.trades]
    for tz, tb, ts in zip(z.trades, b.trades, s.trades):
        assert (tz.stop, tz.target) == (tb.stop, tb.target) == (ts.stop, ts.target)
        assert tz.net_R >= tb.net_R >= ts.net_R and tz.net_R == pytest.approx(tz.gross_R)
    assert sum(t.net_R for t in z.trades) > sum(t.net_R for t in s.trades)
    assert z.scenario["presentation"].startswith("DIAGNOSTIC UPPER BOUND")
    again = cost_scenario_ledgers(pipe, SCEN, "BID")
    assert hashlib.sha256(json.dumps(pipe.trade.proposals, sort_keys=True, default=str).encode()).hexdigest() == props
    assert [t.to_dict() for t in again["STRESSED"].trades] == [t.to_dict() for t in s.trades]


def test_account_simulation_goes_through_phase1f_and_phase1g(pipe, ledgers):
    run = account_simulation(pipe, ledgers["BASELINE"])
    d = list(run.decisions.values())
    assert d and all(x["risk_decision"] == "RISK_APPROVED" for x in d)
    assert all(x["execution_decision"] == "FILLED" and x["protection"] == "PROTECTED" for x in d)
    assert run.final_balance != run.starting_balance and len(run.closed) == len(d)
    blocked = account_simulation(pipe, ledgers["BASELINE"], max_notional_multiple=None)
    assert all(x["risk_rejection"] == "MARGIN_UNKNOWN" for x in blocked.decisions.values())  # fail closed


# --------------------------------------------------------------------------- diagnostics
def test_critical_family_test_reports_all_four_families(pipe, ledgers):
    cov = dg.family_coverage(pipe, dg.enrich(pipe, ledgers["BASELINE"].trades))
    assert set(dg.FAMILIES) <= set(cov)
    for f in dg.FAMILIES:
        assert cov[f]["coverage_status"] in ("NEVER_OBSERVED", "NEVER_QUALIFIED", "EXTREMELY_RARE", "RARE", "OBSERVED")
        assert {"setups_created", "qualified", "entry_candidates", "proposals", "trades"} <= set(cov[f])


def _scored(score, r):
    n = len(score)
    return pd.DataFrame({"setup_score": score, "net_R": r, "mae_R": 0.5, "mfe_R": 1.0,
                         "entry_quality_score": np.nan, "setup_confidence": np.nan, "trade_construction_quality_score": np.nan,
                         "entry_time": pd.date_range("2022-01-03", periods=n, freq="D", tz="UTC").astype(str)})


def test_critical_score_test_reports_whatever_the_relationship_is():
    rng = np.random.default_rng(0)
    s = np.linspace(40, 90, 120)
    good = dg.score_calibration(_scored(s, (s - 60) / 10 + rng.normal(0, .1, 120)))["setup_score"]
    neg = dg.score_calibration(_scored(s, -(s - 60) / 10 + rng.normal(0, .1, 120)))["setup_score"]
    noise = dg.score_calibration(_scored(s, rng.normal(0, 1, 120)))["setup_score"]
    assert good["verdict"] == "MONOTONIC_INCREASING" and len(good["bins"]) == 10
    assert neg["verdict"] == "NEGATIVE_RELATIONSHIP" and neg["calibration_warning"]
    assert noise["verdict"] in ("STATISTICALLY_UNCLEAR", "POSITIVE_BUT_NON_MONOTONIC", "NEGATIVE_RELATIONSHIP")
    small = dg.score_calibration(_scored(s[:8], rng.normal(0, 1, 8)))["setup_score"]
    assert small["status"] == "NOT_MEANINGFUL_SAMPLE_TOO_SMALL"
    assert dg.score_calibration(_scored(s, s))["entry_quality_score"]["status"] == "NOT_AVAILABLE"


def test_descriptive_diagnostics_run_on_the_ledger(pipe, ledgers):
    df = dg.enrich(pipe, ledgers["BASELINE"].trades)
    for c in ("h4_regime", "h1_volatility_regime", "h1_atr_percentile", "h4_atr_percentile", "session_label", "day_of_week",
              "news_status", "stop_distance_atr", "target_reachability_score", "barrier_count"):
        assert c in df.columns
    att = dg.attribution(df)
    assert {"direction", "setup_family", "h4_regime", "session", "day_of_week", "hour_utc_bucket"} <= set(att)
    assert "hypotheses" in dg.mae_mfe(df) and "target_hit_rate" in dg.stop_target_analysis(df)
    cf = dg.counterfactuals(pipe, SCEN["BASELINE"])
    assert cf["label"].startswith("POST-DECISION DIAGNOSTIC") and cf["filters"]
    st = dg.structural_limitations(pipe, df)
    assert {"one_active_setup_limit", "h1_swing_lag", "swing_horizon", "multiple_breaks", "barriers", "weekend_context_age",
            "news"} <= set(st)
    assert st["news"]["status"] == "NEWS_DATA_UNAVAILABLE" and st["h1_swing_lag"]["lag_bars_median"] >= 3


def test_simple_and_null_baselines(pipe, ledgers):
    w = warmup(pipe)["index"]
    props = rb.simple_baseline_proposals(pipe)
    assert props and all(p["setup_family"] == "SIMPLE_BASELINE" for p in props)
    df = rb.simulate_proposals(pipe, props, SCEN["BASELINE"], warmup_index=w)
    assert len(df) > 0 and (df["entry_index"] >= w).all()
    strat = dg.enrich(pipe, ledgers["BASELINE"].trades)
    a = rb.null_baselines(pipe, strat, SCEN["BASELINE"], 4, seed=3, warmup_index=w)
    assert a == rb.null_baselines(pipe, strat, SCEN["BASELINE"], 4, seed=3, warmup_index=w)
    assert {"random_direction", "random_timing_constrained_context"} <= set(a)


# --------------------------------------------------------------------------- robustness (no auto-tuning)
def test_perturbations_never_modify_the_baseline_configuration():
    base = StrategyConfigs()
    snap = json.dumps([describe_config(getattr(base, n)) for n in ("h4", "h1", "entry", "trade")], default=str)
    p = rb.perturb(base, ("h1", "setup", "qualify_score"), 1.1)
    assert p.h1.setup.qualify_score == pytest.approx(66.0) and base.h1.setup.qualify_score == 60.0
    w = rb.reweight(base, "h1_setup_scoring", "w_liquidity", 1.2)
    ws = [getattr(w.h1.scoring, n) for n in rb.WEIGHT_GROUPS["h1_setup_scoring"][2]]
    assert sum(ws) == pytest.approx(sum(getattr(base.h1.scoring, n) for n in rb.WEIGHT_GROUPS["h1_setup_scoring"][2]))
    a = rb.ablate(base, "without_liquidity")
    assert a.h1.scoring.w_liquidity == 0.0
    assert json.dumps([describe_config(getattr(StrategyConfigs(), n)) for n in ("h4", "h1", "entry", "trade")],
                      default=str) == snap
    with pytest.raises(ValueError):
        rb.threshold_alternatives(lambda c: pd.DataFrame(), base, {("h1", "setup", "qualify_score"): tuple(range(25))})


def _fake_run(expect_by_value):
    def fn(cfg):
        v = round(cfg.h1.setup.qualify_score, 3)
        e = expect_by_value(v)
        return pd.DataFrame({"net_R": [e + 1.0, e - 1.0] * 20, "entry_time": pd.date_range("2022-01-03", periods=40,
                                                                                          tz="UTC").astype(str),
                             "exit_time": pd.date_range("2022-01-04", periods=40, tz="UTC").astype(str), "holding_hours": 5.0})
    return fn


def test_critical_parameter_fragility_is_flagged():
    base = StrategyConfigs()
    p = (("h1", "setup", "qualify_score"),)
    stable = rb.sensitivity(_fake_run(lambda v: 0.3), base, p)
    cliff = rb.sensitivity(_fake_run(lambda v: 0.3 if v == 60.0 else -0.1), base, p)
    fragile = rb.sensitivity(_fake_run(lambda v: 0.3 if 54 <= v <= 66 else -0.2), base, p)
    none = rb.sensitivity(_fake_run(lambda v: -0.1), base, p)
    key = "h1.setup.qualify_score"
    assert stable["parameters"][key]["flag"] == "STABLE_REGION"
    assert cliff["parameters"][key]["flag"] == "CLIFF_EDGE" and cliff["fragile_share"] == 1.0
    assert fragile["parameters"][key]["flag"] == "FRAGILE"
    assert none["parameters"][key]["flag"] == "NO_BASELINE_EDGE"
    assert [r["factor"] for r in stable["parameters"][key]["rows"]] == list(rb.DEFAULT_FACTORS)


def test_real_sensitivity_and_ablation_run_on_the_engine(pipe):
    def run(cfg):
        o = run_pipeline(pipe.h1_bars, configs=cfg)
        return dg.enrich(o, cost_scenario_ledgers(o, {"BASELINE": SCEN["BASELINE"]}, "BID")["BASELINE"].trades)

    res = rb.sensitivity(run, StrategyConfigs(), (("trade", "stop", "buffer_atr"),), factors=(0.9, 1.0))
    rows = res["parameters"]["trade.stop.buffer_atr"]["rows"]
    assert [r["factor"] for r in rows] == [0.9, 1.0] and rows[1]["trades"] == res["baseline"]["trades"]


def test_overfitting_risk_diagnostic():
    lo = rb.overfitting_risk(20, 2, 500, 0.0, 0.2, 0.18, 0.1, 0.3)
    hi = rb.overfitting_risk(650, 60, 12, 0.8, 0.5, -0.1, 1.2, 1.0)
    assert lo["level"] == "LOW" and hi["level"] == "HIGH" and "not a formal probability" in hi["note"]
    assert set(hi["warnings"]) >= {"experiments", "small_sample", "is_oos_degradation"}


# --------------------------------------------------------------------------- runner, firewall, reproducibility
def test_runner_refuses_synthetic_and_reports_real_data_required(syn_store):
    store, ds = syn_store
    assert first_real_data_run(store.root)["status"] == "REAL_DATA_REQUIRED"
    r = run_validation(store, ds)
    assert r["status"] == "REAL_DATA_REQUIRED" and r["reason"] == "dataset is synthetic"


def test_validation_result_object_is_complete_and_labelled(validation_run):
    r = validation_run["result"]
    d = r.to_dict()
    for k in ("experiment_id", "strategy_version", "config_version", "dataset_id", "date_range", "trade_count", "gross_R",
              "net_R", "expectancy_R", "profit_factor", "win_rate", "max_drawdown_R", "drawdown_duration_days", "costs",
              "long_metrics", "short_metrics", "setup_family_metrics", "regime_metrics", "walk_forward_metrics",
              "sensitivity_metrics", "monte_carlo_summary", "data_quality_warnings", "research_warnings", "validation_status"):
        assert k in d
    assert r.validation_status == "NOT_VALIDATED" and r.analysis_label == "SYNTHETIC_TEST_ONLY"
    assert any("SYNTHETIC TEST DATA" in w for w in r.research_warnings)
    assert any("event awareness" in w for w in r.research_warnings)
    assert r.details["holdout"] is None and r.details["objective"]["status"] == "NOT_EVALUATED_STRATEGY_NOT_VALIDATED"
    assert r.details["gates"]["decision"] == "NOT_ELIGIBLE"
    w = validation_run["written"]
    assert {"trade_ledger", "r_curve", "result", "signal_ledger", "configuration_freeze"} <= set(w)
    check_language(open(str(w["result"]).replace("result.json", "report.txt")).read())
    perf = validation_run["performance"]
    assert perf["h1_bars_processed"] > 0 and perf["bars_per_second"] > 0 and "bottleneck" in perf


def test_holdout_never_touched_by_default(validation_run, syn_store):
    from gbpjpy_engine.validation import HoldoutLock

    w = validation_run["written"]["result"]
    lock = HoldoutLock(__import__("pathlib").Path(w).parents[1] / "holdout_lock.json")
    assert lock.state == "LOCKED" and lock.first_accessed_at is None


def test_determinism_same_inputs_same_results(validation_run, syn_store, tmp_path):
    store, ds = syn_store
    again = run_validation(store, ds, workdir=tmp_path, development=0.5, validation=0.45, allow_synthetic_for_testing=True,
                           n_null_runs=5, now="2024-01-01T00:00:00+00:00")

    def clean(x):
        d = json.loads(x.to_json())
        d["details"].pop("performance")
        return d

    assert clean(again["result"]) == clean(validation_run["result"])


def test_critical_risk_firewall_objectives_change_nothing(pipe, ledgers):
    before = (json.dumps(pipe.trade.proposals, sort_keys=True, default=str),
              json.dumps(account_simulation(pipe, ledgers["BASELINE"]).decisions, sort_keys=True, default=str))
    for target in (500_000, 5_000_000, 1):
        objective_analysis("HOLDOUT_TESTED", target, 0.2, (0.1, 0.3), 50)
    after = (json.dumps(pipe.trade.proposals, sort_keys=True, default=str),
             json.dumps(account_simulation(pipe, ledgers["BASELINE"]).decisions, sort_keys=True, default=str))
    assert before == after and objective_is_isolated()
    pat = re.compile(r"(profit|annual|monthly|desired).*(target|goal|objective|return)|target_?(profit|equity|return)")
    cfg = StrategyConfigs()
    for n in ("h4", "h1", "entry", "trade"):
        assert not [r["name"] for r in describe_config(getattr(cfg, n)) if pat.search(r["name"])]
    import dataclasses

    for n in ("risk", "execution"):
        names = [f.name for sec in dataclasses.fields(getattr(cfg, n)) for f in dataclasses.fields(getattr(getattr(cfg, n), sec.name))]
        assert not [x for x in names if pat.search(x)]
