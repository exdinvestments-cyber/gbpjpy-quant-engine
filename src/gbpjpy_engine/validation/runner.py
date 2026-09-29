"""Phase 1H orchestration: first real-data run and the complete validation pipeline.

``first_real_data_run`` refuses to fabricate anything: when no REAL GBPJPY H1
dataset (with provenance) exists in the store it returns REAL_DATA_REQUIRED and
the import specification.  Synthetic datasets are never substituted; they can
only be run with ``allow_synthetic_for_testing=True`` and are then labelled
SYNTHETIC_TEST_ONLY with validation status NOT_VALIDATED.

The final holdout is never evaluated by default.  Evaluating it requires an
unlocked ``HoldoutLock`` and ``evaluate_holdout=True``; the first access is
labelled HOLDOUT_RESULT and every later one POST_HOC_ANALYSIS.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..risk.research import RiskOfRuinInputs
from . import diagnostics as dg
from . import metrics as mt
from . import montecarlo as mc
from .costs import standard_scenarios
from .datastore import DataStore, require_known_provenance
from .experiments import ExperimentManifest, ExperimentRecord, HypothesisRegister, ResearchBudget
from .gates import PromotionGates, evaluate_gates, validation_status
from .objective import objective_analysis
from .provenance import describe_accepted_formats
from .quality import quality_gate, research_quality_report
from .replay import StrategyConfigs, account_simulation, cost_scenario_ledgers, run_pipeline, signal_ledger, warmup
from .report import BacktestResult, export_dashboard, render_text
from .resampling import UTC_GRID, compare_h4, resample_h1_to_h4, verify_aggregation
from .robustness import null_baselines, overfitting_risk, simple_baseline_proposals, simulate_proposals
from .snapshot import configuration_freeze, strategy_snapshot
from .splits import ChronologicalSplit, HoldoutLock, SplitGuard, walk_forward_report, walk_forward_windows


def real_data_required(store_root) -> dict:
    return {"status": "REAL_DATA_REQUIRED", "store": str(store_root),
            "message": "no real GBPJPY H1 dataset with provenance exists in the research store; no validation result has "
                       "been produced and no synthetic data has been substituted",
            "import_specification": describe_accepted_formats(),
            "how_to_import": [
                "export GBPJPY H1 (and optionally H4, M15/M5/M1) history from the broker/terminal or a data vendor",
                "declare provenance: provider, source timezone (the terminal's server timezone for platform exports), price "
                "type (platform exports are normally BID), volume type, spread availability and unit (platform exports "
                "report spread in POINTS)",
                "python -m gbpjpy_engine validate import --file <path> --timeframe H1 --provider broker_export "
                "--source-tz <IANA tz> --price-type BID --volume-type TICK_VOLUME --spread PER_BAR --spread-unit POINTS "
                "--format TERMINAL_TAB_EXPORT --retrieved-at <ISO time>",
                "python -m gbpjpy_engine validate run",
            ]}


def first_real_data_run(store_root, **kw) -> dict:
    store = DataStore(store_root)
    h1 = [p for p in store.list_datasets(real_only=True) if p.timeframe == "H1" and p.symbol.upper() == "GBPJPY"]
    if not h1:
        return real_data_required(store_root)
    return run_validation(store, h1[0].dataset_id, **kw)


def _peak_rss_mb():
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)  # Linux reports KiB
    except Exception:  # noqa: BLE001 - platform without resource
        return None


def _years(df) -> float:
    t = pd.to_datetime(df["timestamp"], utc=True)
    return (t.max() - t.min()).days / 365.25 if len(t) else 0.0


def run_validation(store: DataStore, h1_dataset_id: str, h4_dataset_id: str | None = None, ltf_dataset_id: str | None = None,
                   workdir=None, seed: int = 20240101, configs: StrategyConfigs | None = None, development: float = 0.6,
                   validation: float = 0.2, ambiguity_policy: str = "CONSERVATIVE", run_sensitivity: bool = False,
                   sensitivity_fn=None, n_null_runs: int = 20, evaluate_holdout: bool = False,
                   allow_synthetic_for_testing: bool = False, known_commission_pips: float | None = None,
                   starting_balance: str = "10000", now=None) -> dict:
    t_start = time.perf_counter()
    cfg = configs or StrategyConfigs()
    work = Path(workdir or (store.root / "results"))
    work.mkdir(parents=True, exist_ok=True)
    manifest = ExperimentManifest(work / "experiment_manifest.jsonl")
    HypothesisRegister(work / "hypotheses.jsonl")
    lock = HoldoutLock(work / "holdout_lock.json")
    warnings: list[str] = []

    # ---- data, provenance, quality -------------------------------------------------------
    h1 = store.load_normalised(h1_dataset_id)
    prov = require_known_provenance(h1)
    if prov.is_synthetic and not allow_synthetic_for_testing:
        return dict(real_data_required(store.root), refused_dataset=h1_dataset_id, reason="dataset is synthetic")
    label_data = "SYNTHETIC_TEST_ONLY" if prov.is_synthetic else "REAL_DATA"
    if prov.is_synthetic:
        warnings.append("SYNTHETIC TEST DATA - these numbers say nothing about real GBPJPY behaviour")
    q1 = quality_gate(research_quality_report(h1, "H1", known_outages=prov.known_outages,
                                              spread_expected=prov.spread_availability in ("PER_BAR", "SAMPLED")))
    h4_derived, h4_rep = resample_h1_to_h4(h1, UTC_GRID)
    agg = verify_aggregation(h1, h4_derived) if len(h4_derived) < 20000 else {"exact": None, "note": "skipped (size)"}
    h4_cmp = None
    if h4_dataset_id:
        h4p = store.load_normalised(h4_dataset_id)
        h4_cmp = compare_h4(h4p, h4_derived)
        if h4_cmp["verdict"] != "CONSISTENT":
            warnings.append("provider H4 differs from H1-derived H4 - derived H4 is used; grids are never mixed")
    ltf = store.load_normalised(ltf_dataset_id) if ltf_dataset_id else None
    if ltf is None and ambiguity_policy == "LOWER_TIMEFRAME_REQUIRED":
        warnings.append("LOWER_TIMEFRAME_REQUIRED without lower-timeframe data: ambiguous bars are UNRESOLVED (conservative)")
    if prov.price_type == "UNKNOWN":
        warnings.append("price type UNKNOWN: bars treated as BID and every trade flagged PRICE_TYPE_UNKNOWN")
    if prov.price_type == "MID":
        warnings.append("MID prices: executable bid/ask derived with the modelled spread")
    if prov.spread_availability not in ("PER_BAR", "SAMPLED"):
        warnings.append("no historical spread: an explicit ASSUMED spread is used and stress-tested")

    # ---- freeze -----------------------------------------------------------------------------
    scen = standard_scenarios(prov.spread_availability, known_commission_pips)
    snap = strategy_snapshot(cfg, {k: v.describe() for k, v in scen.items()}, now)
    frz = configuration_freeze(cfg)
    exp_id = manifest.next_id("BASELINE")

    # ---- chronological split; holdout bars never enter development/validation ----------------
    split = ChronologicalSplit.by_fraction(h1["timestamp"], development, validation)
    guard = SplitGuard(h1, split, lock)
    dv_bars, dv_info = guard.data("VALIDATION", "PRE_DECLARED_TEST", snap["snapshot_id"])
    out = run_pipeline(dv_bars, configs=cfg)
    w = warmup(out)
    if not w["complete"]:
        warnings.append("warm-up never completed: no trades are scored")
    ledgers = cost_scenario_ledgers(out, scen, prov.price_type, ambiguity_policy, ltf, set(q1.blackout))
    base = ledgers["BASELINE"]
    trades = dg.enrich(out, base.trades)
    seg = trades["entry_time"].map(split.segment_of) if len(trades) else pd.Series(dtype=object)
    ins = trades[seg == "DEVELOPMENT"] if len(trades) else trades
    oos = trades[seg == "VALIDATION"] if len(trades) else trades
    embargoed = int((seg.isna()).sum()) if len(trades) else 0

    # ---- statistics -----------------------------------------------------------------------------
    s_all, s_in, s_oos = mt.summarise(trades), mt.summarise(ins), mt.summarise(oos)
    ci_oos = mt.confidence_intervals(oos["net_R"] if len(oos) else [], seed, block=None)
    ci_oos_block = mt.confidence_intervals(oos["net_R"] if len(oos) else [], seed, block=5) if len(oos) >= 30 else {}
    wf_windows = walk_forward_windows(split.start, split.validation_end, 6, 12, "ANCHORED")
    wf = walk_forward_report(trades, wf_windows) if len(trades) else {"windows": [], "n_windows": len(wf_windows)}
    by_scen = {k: {"trades": len(v.trades), "summary": mt.summarise(dg.enrich(out, v.trades)) if v.trades else {"trades": 0},
                   "scenario": v.scenario} for k, v in ledgers.items()}
    r_oos = oos["net_R"].to_numpy(float) if len(oos) else np.array([])
    r_all = trades["net_R"].to_numpy(float) if len(trades) else np.array([])
    mcs = {"sequence_bootstrap": mc.sequence_monte_carlo(r_all, 2000, seed),
           "sequence_block": mc.sequence_monte_carlo(r_all, 2000, seed, "BLOCK", 5),
           "sequence_permutation": mc.sequence_monte_carlo(r_all, 2000, seed, "PERMUTATION"),
           "cost_stress": mc.cost_stress(trades), "random_cost_stress": mc.random_cost_stress(trades, 1000, seed),
           "missed_trades_10pct": mc.missed_trades(r_all, 0.10, 1000, seed),
           "missed_trades_25pct": mc.missed_trades(r_all, 0.25, 1000, seed),
           "worse_fills": mc.worse_fills(trades, 2.0, 2.0, 1000, seed), "tail_stress": mc.tail_stress(r_all, 0.02, (0.5, 2.0), 1000, seed),
           "risk_of_ruin_0_5pct": mc.risk_of_ruin(RiskOfRuinInputs(win_loss_distribution_R=tuple(r_all),
                                                                   tail_loss_distribution_R=tuple(r_all[r_all < -1.0]),
                                                                   ruin_threshold_percent=30.0), 0.5, 250, 5000, seed),
           "risk_policy_comparison": mc.risk_policy_comparison(r_all, (0.25, 0.5, 1.0), 250, 2000, seed),
           "compounding_0_5pct": {k: v for k, v in mc.compounding(trades, 0.5).items() if k != "curve"},
           "capital_scale": mc.capital_scale(trades["one_R_pips"] if len(trades) else [], 0.5)}
    diag = {"family_coverage": dg.family_coverage(out, trades), "score_calibration": dg.score_calibration(trades, seed=seed),
            "attribution": dg.attribution(trades), "long_short": dg.long_short(trades), "mae_mfe": dg.mae_mfe(trades),
            "stop_target": dg.stop_target_analysis(trades), "counterfactuals": dg.counterfactuals(out, scen["BASELINE"],
                                                                                                   prov.price_type),
            "structural": dg.structural_limitations(out, trades), "calendar": mt.calendar_breakdowns(trades),
            "concentration": mt.concentration(trades), "stability": mt.distribution_stability(trades),
            "risk_adjusted": mt.risk_adjusted(trades) if len(trades) else {},
            "frequency": mt.trade_frequency(trades, split.start, split.validation_end),
            "time_in_market": mt.exposure_fraction(trades["entry_time"], trades["exit_time"], split.start, split.validation_end)
            if len(trades) else 0.0,
            "drawdown_episodes": mt.drawdown_episodes(r_all),
            "ambiguous_trades": int((trades["ambiguity"] != "NONE").sum()) if len(trades) else 0}
    if len(trades) and (trades["ambiguity"] != "NONE").any():
        clean = trades[trades["ambiguity"] == "NONE"]
        diag["excluding_ambiguous"] = mt.summarise(clean) if len(clean) else {"trades": 0}
    base_df = simulate_proposals(out, simple_baseline_proposals(out), scen["BASELINE"], prov.price_type,
                                 warmup_index=w["index"] or 0)
    baselines = {"simple_trend_pullback": mt.summarise(base_df) if len(base_df) else {"trades": 0},
                 "null": null_baselines(out, trades, scen["BASELINE"], n_null_runs, seed, prov.price_type, w["index"] or 0)
                 if len(trades) else {"status": "NO_STRATEGY_TRADES"}}
    sens = {}
    if run_sensitivity and sensitivity_fn is not None:
        from .robustness import ablation, sensitivity

        ResearchBudget().check(manifest, "DEVELOPMENT")
        sens = {"parameters": sensitivity(sensitivity_fn, cfg), "ablation": ablation(sensitivity_fn, cfg)}
    elif run_sensitivity:
        dev_bars, _ = guard.data("DEVELOPMENT", "SENSITIVITY_TUNING", snap["snapshot_id"])

        def fn(c):
            o = run_pipeline(dev_bars, configs=c)
            return dg.enrich(o, cost_scenario_ledgers(o, {"BASELINE": scen["BASELINE"]}, prov.price_type)["BASELINE"].trades)

        from .robustness import ablation, sensitivity

        ResearchBudget().check(manifest, "DEVELOPMENT")
        sens = {"parameters": sensitivity(fn, cfg), "ablation": ablation(fn, cfg)}
    else:
        sens = {"status": "NOT_RUN", "reason": "sensitivity is run explicitly (costly: one full replay per variant)"}
    account = account_simulation(out, base, starting_balance, "GBP", price_type=prov.price_type)

    # ---- overfitting, gates, status --------------------------------------------------------------
    fam_R = trades.groupby("setup_family")["net_R"].sum() if len(trades) else pd.Series(dtype=float)
    top_fam = float(fam_R.max() / fam_R[fam_R > 0].sum()) if len(fam_R) and (fam_R > 0).any() else None
    of = overfitting_risk(frz["n_tunable"], manifest.count(), len(trades),
                          (sens.get("parameters") or {}).get("fragile_share") if isinstance(sens, dict) else None,
                          s_in.get("expectancy_R"), s_oos.get("expectancy_R"),
                          diag["concentration"].get("best_5_share") if diag["concentration"] else None, top_fam)
    gate_metrics = {"oos_trades": s_oos.get("trades"), "oos_expectancy_R": s_oos.get("expectancy_R"),
                    "oos_prob_expectancy_le_zero": (ci_oos.get("expectancy_R") or {}).get("p_le_zero"),
                    "stressed_cost_expectancy_R": ((by_scen.get("STRESSED") or {}).get("summary") or {}).get("expectancy_R"),
                    "max_drawdown_R": s_all.get("max_drawdown_R"),
                    "walk_forward_profitable_share": (wf["profitable_windows"] / max(1, wf["n_windows"] - wf["empty_windows"]))
                    if wf.get("windows") else None,
                    "fragile_parameter_share": (sens.get("parameters") or {}).get("fragile_share") if isinstance(sens, dict) else None,
                    "best5_share_of_profit": diag["concentration"].get("best_5_share") if diag["concentration"] else None,
                    "execution_stress_prob_le_zero": mcs["worse_fills"].get("probability_expectancy_le_0"),
                    "data_years": _years(h1)}
    gates = evaluate_gates(PromotionGates(), gate_metrics)
    failed = bool(len(oos) and s_oos.get("expectancy_R") is not None and s_oos["expectancy_R"] <= 0)
    holdout_res = None
    if evaluate_holdout:
        full, info = guard.data("FINAL_HOLDOUT", "FINAL_EVALUATION", snap["snapshot_id"])
        o2 = run_pipeline(full, configs=cfg)
        l2 = cost_scenario_ledgers(o2, {"BASELINE": scen["BASELINE"]}, prov.price_type, ambiguity_policy, ltf, set(q1.blackout))
        t2 = dg.enrich(o2, l2["BASELINE"].trades)
        t2 = t2[t2["entry_time"].map(split.segment_of) == "FINAL_HOLDOUT"] if len(t2) else t2
        holdout_res = {"label": info["analysis_label"], "summary": mt.summarise(t2)}
    status = "NOT_VALIDATED" if prov.is_synthetic else validation_status(
        {"real_data": True, "data_quality_ok": q1.status != "BLOCKED", "in_sample": len(ins) > 0, "out_of_sample": len(oos) > 0,
         "walk_forward": bool(wf.get("n_windows")), "holdout": holdout_res is not None and holdout_res["label"] == "HOLDOUT_RESULT",
         "gates_passed": gates["all_passed"], "failed": failed})
    if s_all.get("sample", {}).get("level") in ("INSUFFICIENT_SAMPLE", "SMALL_SAMPLE"):
        warnings.append(s_all["sample"]["message"])
    for f, v in diag["family_coverage"].items():
        if v["coverage_status"] in ("NEVER_OBSERVED", "NEVER_QUALIFIED", "EXTREMELY_RARE"):
            warnings.append(f"setup family {f}: {v['coverage_status']}")
    warnings.append(diag["structural"]["news"]["disclosure"])
    warnings += [f"overfitting diagnostic factor high: {x}" for x in of["warnings"]]
    if diag["concentration"] and diag["concentration"].get("flag"):
        warnings.append(diag["concentration"]["flag"])

    peak = _peak_rss_mb()
    runtime = time.perf_counter() - t_start
    perf = {"h1_bars_processed": int(len(dv_bars)), "trades_generated": len(trades), "pipeline_seconds": round(out.runtime_seconds, 2),
            "total_seconds": round(runtime, 2), "bars_per_second": round(len(dv_bars) / out.runtime_seconds, 1)
            if out.runtime_seconds else None, "peak_process_memory_mb": peak,
            "bottleneck": "Phase 1A-1E feature/structure computation (pure Python loops); sensitivity multiplies it per variant"}
    result = BacktestResult(
        experiment_id=exp_id, strategy_version=snap["package_version"], strategy_snapshot_id=snap["snapshot_id"],
        config_version=frz["configuration_hash"], dataset_id=h1_dataset_id,
        date_range=(str(h1["timestamp"].min()), str(dv_info["last_bar"])), seed=seed,
        analysis_label="PRE_DECLARED_TEST" if not prov.is_synthetic else "SYNTHETIC_TEST_ONLY",
        trade_count=int(len(trades)), gross_R=float(trades["gross_R"].sum()) if len(trades) else None,
        net_R=float(r_all.sum()) if len(r_all) else None, expectancy_R=s_all.get("expectancy_R"),
        profit_factor=s_all.get("profit_factor"), win_rate=s_all.get("win_rate"), max_drawdown_R=s_all.get("max_drawdown_R"),
        drawdown_duration_days=s_all.get("max_drawdown_duration_days"),
        costs={k: v["scenario"] for k, v in by_scen.items()} | {"by_scenario": {k: v["summary"] for k, v in by_scen.items()}},
        long_metrics=diag["long_short"].get("LONG", {}), short_metrics=diag["long_short"].get("SHORT", {}),
        setup_family_metrics=diag["attribution"].get("setup_family", {}),
        regime_metrics={k: diag["attribution"].get(k, {}) for k in ("h4_regime", "h1_volatility_regime", "h4_volatility_regime")},
        walk_forward_metrics=wf, sensitivity_metrics=sens,
        monte_carlo_summary={k: v for k, v in mcs.items()},
        data_quality_warnings=[f"{k}: {v}" for k, v in q1.counts.items()], research_warnings=warnings,
        validation_status=status,
        details={"data_label": label_data, "provenance": prov.to_dict(), "quality": q1.summary(), "h4_definition": h4_rep,
                 "h4_aggregation_check": agg, "h4_provider_comparison": h4_cmp, "warmup": w, "split": split.to_dict(),
                 "segments": {"in_sample": s_in, "out_of_sample": s_oos, "embargoed_trades": embargoed},
                 "confidence_intervals_oos": ci_oos, "confidence_intervals_oos_block": ci_oos_block,
                 "diagnostics": diag, "baselines": baselines, "overfitting": of, "gates": gates, "holdout": holdout_res,
                 "account_simulation": account.to_dict(), "snapshot": snap,
                 "config_label_counts": frz["label_counts"], "performance": perf,
                 "objective": objective_analysis(status),
                 "excluded_proposals": base.excluded, "signal_ledger_rows": int(len(signal_ledger(out, account.decisions)))}
    )
    manifest.append(ExperimentRecord(exp_id, "frozen baseline, run once, unchanged", {}, "Phase 1H first run", h1_dataset_id,
                                     "DEVELOPMENT+VALIDATION", result.analysis_label if not prov.is_synthetic else
                                     "EXPLORATORY_ANALYSIS", snap["snapshot_id"], frz["configuration_hash"], seed,
                                     ["summary", "walk_forward", "costs", "monte_carlo", "diagnostics"],
                                     {"trades": result.trade_count, "expectancy_R": result.expectancy_R, "status": status},
                                     exploratory=False))
    parts = {"result": result.to_dict(), "signal_ledger": signal_ledger(out, account.decisions).to_dict("records"),
             "configuration_freeze": frz}
    written = export_dashboard(work / exp_id, trades, parts)
    (work / exp_id / "report.txt").write_text(render_text(result))
    return {"status": "COMPLETED", "result": result, "written": written, "performance": perf}
