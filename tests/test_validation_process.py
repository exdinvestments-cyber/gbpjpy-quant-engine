"""Phase 1H research process: splits, holdout lock, walk-forward, manifest, budget, hypotheses, freeze, gates."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from gbpjpy_engine.validation import (UNLOCK_CONFIRMATION, ChronologicalSplit, ExperimentManifest, ExperimentRecord, Hypothesis,
                                      HypothesisRegister, HoldoutLock, HoldoutLocked, ManifestTampered, MarketingLanguage,
                                      PromotionGates, ResearchBudget, ResearchBudgetExhausted, SplitGuard, StrategyConfigs,
                                      check_language, configuration_freeze, evaluate_gates, objective_analysis,
                                      strategy_snapshot, validation_status, walk_forward_report, walk_forward_windows)
from validation_helpers import random_walk_h1


def guard(tmp_path, n=2000):
    bars = random_walk_h1(n)
    split = ChronologicalSplit.by_fraction(bars["timestamp"], 0.6, 0.2, embargo_hours=48)
    return bars, split, SplitGuard(bars, split, HoldoutLock(tmp_path / "lock.json"))


# --------------------------------------------------------------------------- splits
def test_chronological_split_is_ordered_with_embargo():
    bars = random_walk_h1(1000)
    s = ChronologicalSplit.by_fraction(bars["timestamp"], 0.6, 0.2, embargo_hours=24)
    d, v, h = s.bounds("DEVELOPMENT"), s.bounds("VALIDATION"), s.bounds("FINAL_HOLDOUT")
    assert d[0] < d[1] < v[0] < v[1] < h[0] < h[1] and v[0] - d[1] == pd.Timedelta(hours=24)
    segs = [s.segment_of(t) for t in bars["timestamp"]]
    order = [x for x in segs if x]
    assert order == sorted(order, key=["DEVELOPMENT", "VALIDATION", "FINAL_HOLDOUT"].index)  # never shuffled
    assert None in segs  # embargo bars belong to no segment
    with pytest.raises(ValueError):
        ChronologicalSplit.by_fraction(bars["timestamp"], 0.7, 0.3)


def test_critical_holdout_cannot_leak_into_development_exploration_sensitivity_or_walk_forward(tmp_path):
    bars, split, g = guard(tmp_path)
    hold_start = split.bounds("FINAL_HOLDOUT")[0]
    for seg, purpose in (("DEVELOPMENT", "EXPLORATORY"), ("DEVELOPMENT", "PARAMETER_EXPLORATION"),
                         ("DEVELOPMENT", "SENSITIVITY_TUNING"), ("VALIDATION", "WALK_FORWARD_CALIBRATION"),
                         ("VALIDATION", "PRE_DECLARED_TEST")):
        df, info = g.data(seg, purpose)
        assert pd.to_datetime(df["timestamp"], utc=True).max() + pd.Timedelta(hours=1) <= split.bounds(seg)[1]
        assert (pd.to_datetime(df["timestamp"], utc=True) < hold_start).all()
    for purpose in ("FINAL_EVALUATION", "PARAMETER_EXPLORATION", "SENSITIVITY_TUNING", "WALK_FORWARD_CALIBRATION"):
        with pytest.raises(HoldoutLocked):
            g.data("FINAL_HOLDOUT", purpose)
    assert g.lock.first_accessed_at is None and g.lock.state == "LOCKED"
    windows = walk_forward_windows(split.start, split.validation_end, 1, 2)
    assert all(pd.Timestamp(w.test_end) <= pd.Timestamp(split.validation_end) for w in windows)


def test_unlock_requires_evidence_and_first_access_is_recorded(tmp_path):
    bars, split, g = guard(tmp_path)
    with pytest.raises(HoldoutLocked):
        g.lock.unlock("SNAP-1", "cfg", [], "VAL-1", "ops", UNLOCK_CONFIRMATION)  # no hypotheses
    with pytest.raises(HoldoutLocked):
        g.lock.unlock("SNAP-1", "cfg", ["H1"], "VAL-1", "ops", "yes")
    g.lock.unlock("SNAP-1", "cfg", ["H1"], "VAL-1", "ops", UNLOCK_CONFIRMATION, at="2024-06-01T00:00:00+00:00")
    for purpose in ("PARAMETER_EXPLORATION", "SENSITIVITY_TUNING", "WALK_FORWARD_CALIBRATION"):
        with pytest.raises(HoldoutLocked):
            g.data("FINAL_HOLDOUT", purpose)  # still forbidden after unlock
    df, info = g.data("FINAL_HOLDOUT", "FINAL_EVALUATION", "SNAP-1", at="2024-06-02T00:00:00+00:00")
    assert info["analysis_label"] == "HOLDOUT_RESULT" and len(df) == len(bars)
    _, again = g.data("FINAL_HOLDOUT", "FINAL_EVALUATION", "SNAP-1")
    assert again["analysis_label"] == "POST_HOC"
    reopened = HoldoutLock(tmp_path / "lock.json")  # persisted across processes
    assert reopened.first_accessed_at == "2024-06-02T00:00:00+00:00" and reopened.state == "UNLOCKED"
    assert [a["label"] for a in json.loads((tmp_path / "lock.json").read_text())["accesses"]] == ["HOLDOUT_RESULT", "POST_HOC"]


# --------------------------------------------------------------------------- walk-forward
def test_walk_forward_windows_and_aggregation():
    a = walk_forward_windows("2020-01-01", "2022-01-01", 6, 12, "ANCHORED")
    r = walk_forward_windows("2020-01-01", "2022-01-01", 6, 12, "ROLLING")
    assert len(a) == 2 and all(w.train_start == a[0].train_start for w in a) and r[1].train_start > r[0].train_start
    assert all(pd.Timestamp(w.train_end) <= pd.Timestamp(w.test_start) for w in a)
    t = pd.DataFrame({"entry_time": ["2021-02-01T00:00:00+00:00", "2021-03-01T00:00:00+00:00", "2021-08-01T00:00:00+00:00"],
                      "exit_time": ["2021-02-02T00:00:00+00:00", "2021-03-02T00:00:00+00:00", "2021-08-02T00:00:00+00:00"],
                      "net_R": [2.0, -1.0, 5.0], "holding_hours": 5.0})
    rep = walk_forward_report(t, walk_forward_windows("2020-01-01T00:00:00+00:00", "2022-01-01T00:00:00+00:00", 6, 12))
    assert rep["n_windows"] == 2 and rep["profitable_windows"] == 2 and [w["trades"] for w in rep["windows"]] == [2, 1]
    assert rep["best_window_share_of_positive_R"] == pytest.approx(5 / 6)


# --------------------------------------------------------------------------- manifest, budget, hypotheses
def rec(i, seg="DEVELOPMENT", label="EXPLORATORY_ANALYSIS", exploratory=True):
    return ExperimentRecord(f"EXP-{i}", "h", {"h1.setup.qualify_score": 55}, "why", "DS", seg, label, "SNAP", "CFG", 1,
                            ["expectancy"], {"expectancy_R": 0.1}, exploratory=exploratory)


def test_manifest_is_append_only_hash_chained_and_keeps_failures(tmp_path):
    m = ExperimentManifest(tmp_path / "m.jsonl")
    m.append(rec(1))
    failed = rec(2)
    failed.status = "FAILED"
    m.append(failed)
    assert [r["status"] for r in m.records()] == ["COMPLETED", "FAILED"] and m.verify()
    with pytest.raises(ValueError):
        m.append(rec(1))  # ids never reused
    with pytest.raises(ValueError):
        m.append(rec(3, label="WHATEVER"))
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    (tmp_path / "m.jsonl").write_text(lines[1] + "\n")  # someone deletes the first experiment
    with pytest.raises(ManifestTampered):
        m.verify()
    assert not any(hasattr(m, n) for n in ("delete", "remove", "edit"))


def test_research_budget_forces_new_untouched_data(tmp_path):
    m = ExperimentManifest(tmp_path / "m.jsonl")
    b = ResearchBudget(max_exploratory_experiments=2)
    for i in range(2):
        assert b.check(m, "DEVELOPMENT")["remaining"] == 2 - i
        m.append(rec(i))
    m.append(rec(9, exploratory=False, label="PRE_DECLARED_TEST"))  # pre-declared tests do not consume the budget
    with pytest.raises(ResearchBudgetExhausted):
        b.check(m, "DEVELOPMENT")
    assert b.check(m, "VALIDATION")["used"] == 0


def test_hypothesis_register_separates_discovery_from_confirmation(tmp_path):
    reg = HypothesisRegister(tmp_path / "h.jsonl")
    h = Hypothesis("H-1", "losers often reach 1R first", "MAE/MFE table", "exit management", "partial at 1R",
                   "DS/DEVELOPMENT", "an untouched later period")
    reg.add(h)
    with pytest.raises(ValueError):
        reg.add(h)
    assert reg.all()[0]["untouched_data_required"] == "an untouched later period" and reg.all()[0]["status"] == "OPEN"


# --------------------------------------------------------------------------- freeze
def test_configuration_freeze_labels_every_parameter():
    c = StrategyConfigs()
    f = configuration_freeze(c)
    assert f["n_parameters"] > 600 and f["label_counts"]["EMPIRICALLY_VALIDATED"] == 0
    ext = {(r["section"], r["name"]) for r in f["parameters"] if r["label"] == "EXTERNAL_CONSTRAINT"}
    assert ("data", "pip_size") in ext and ("data", "symbol") in ext
    assert all(r["label"] == "UNFITTED_BASELINE" for r in f["parameters"] if r["name"].startswith("w_"))
    kinds = {r["kind"] for r in f["parameters"]}
    assert {"weight", "lookback", "threshold", "atr_multiplier", "buffer"} <= kinds
    c2 = replace(c, h1=replace(c.h1, setup=replace(c.h1.setup, qualify_score=61.0)))
    assert configuration_freeze(c2)["configuration_hash"] != f["configuration_hash"]


def test_strategy_snapshot_identifies_the_tested_strategy():
    s1 = strategy_snapshot(StrategyConfigs(), {"spread": "HISTORICAL"}, now="2024-01-01T00:00:00+00:00")
    s2 = strategy_snapshot(StrategyConfigs(), {"spread": "HISTORICAL"}, now="2025-01-01T00:00:00+00:00")
    assert s1["snapshot_id"] == s2["snapshot_id"] and s1["created_at"] != s2["created_at"]
    for k in ("package_version", "git_commit", "strategy_source_sha256", "configuration_hash", "risk_policy_hash",
              "execution_policy_hash", "environment"):
        assert k in s1
    assert strategy_snapshot(StrategyConfigs(), {"spread": "ASSUMED"})["snapshot_id"] != s1["snapshot_id"]


# --------------------------------------------------------------------------- statuses, gates, objective, language
def test_validation_status_is_evidence_based_not_pnl_based():
    assert validation_status({"real_data": False, "in_sample": True}) == "NOT_VALIDATED"
    ev = {"real_data": True, "data_quality_ok": True}
    assert validation_status(ev) == "DATA_VALIDATED"
    assert validation_status(dict(ev, in_sample=True)) == "IN_SAMPLE_ONLY"
    assert validation_status(dict(ev, in_sample=True, out_of_sample=True)) == "OUT_OF_SAMPLE_TESTED"
    assert validation_status(dict(ev, walk_forward=True)) == "WALK_FORWARD_TESTED"
    assert validation_status(dict(ev, holdout=True)) == "HOLDOUT_TESTED"
    assert validation_status(dict(ev, holdout=True, gates_passed=True)) == "PAPER_TRADING_REQUIRED"
    assert validation_status(dict(ev, holdout=True, failed=True)) == "FAILED_VALIDATION"


def test_promotion_gates_fail_visibly():
    good = {"oos_trades": 150, "oos_expectancy_R": 0.2, "oos_prob_expectancy_le_zero": 0.02, "stressed_cost_expectancy_R": 0.05,
            "max_drawdown_R": 10, "walk_forward_profitable_share": 0.7, "fragile_parameter_share": 0.1,
            "best5_share_of_profit": 0.3, "execution_stress_prob_le_zero": 0.1, "data_years": 8}
    assert evaluate_gates(PromotionGates(), good)["all_passed"]
    bad = dict(good, oos_trades=40, fragile_parameter_share=None)
    r = evaluate_gates(PromotionGates(), bad)
    assert not r["all_passed"] and r["decision"] == "NOT_ELIGIBLE" and set(r["failed"]) == {"adequate sample",
                                                                                            "parameter stability"}
    assert any(g["status"] == "UNAVAILABLE" for g in r["gates"])


def test_objective_is_not_analysed_before_validation_and_says_when_evidence_is_missing():
    assert objective_analysis("OUT_OF_SAMPLE_TESTED")["status"] == "NOT_EVALUATED_STRATEGY_NOT_VALIDATED"
    assert objective_analysis("HOLDOUT_TESTED", expectancy_R=-0.1, trades_per_year=50)["status"] == \
        "EVIDENCE_DOES_NOT_SUPPORT_OBJECTIVE"
    ok = objective_analysis("HOLDOUT_TESTED", 500_000, 0.2, (0.05, 0.35), 60)
    r05 = next(r for r in ok["rows"] if r["risk_percent"] == 0.5)
    assert r05["required_starting_capital_gbp"] == pytest.approx(500_000 / (0.005 * 0.2 * 60))
    assert r05["capital_range_from_expectancy_ci_gbp"][1] > r05["required_starting_capital_gbp"]


def test_reports_reject_marketing_language():
    check_language("Observed historical out-of-sample expectancy under these assumptions was 0.1 R.")
    for bad in ("proven profitable", "guaranteed returns", "a money-making machine", "institutional-grade returns",
                "safe profits"):
        with pytest.raises(MarketingLanguage):
            check_language(f"This is {bad}.")
