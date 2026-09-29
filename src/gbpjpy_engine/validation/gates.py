"""Validation statuses and promotion gates (Phase 1H).

A strategy is never called validated because its total historical P&L is
positive.  ``validation_status`` is derived from which evidence exists;
``promotion_gates`` are configurable research thresholds (CONFIGURATION, NOT
VALIDATED EDGE) that must ALL pass before paper/demo trading is even
considered.  Failures are listed, never hidden.  Nothing here modifies a
strategy parameter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

VALIDATION_STATUSES = ("NOT_VALIDATED", "DATA_VALIDATED", "IN_SAMPLE_ONLY", "OUT_OF_SAMPLE_TESTED", "WALK_FORWARD_TESTED",
                       "HOLDOUT_TESTED", "PAPER_TRADING_REQUIRED", "FAILED_VALIDATION")


def validation_status(evidence: dict) -> str:
    """``evidence`` keys: real_data, data_quality_ok, in_sample, out_of_sample, walk_forward, holdout, gates_passed,
    failed (bool each)."""
    if not evidence.get("real_data") or not evidence.get("data_quality_ok"):
        return "NOT_VALIDATED"
    if evidence.get("failed"):
        return "FAILED_VALIDATION"
    if evidence.get("holdout") and evidence.get("gates_passed"):
        return "PAPER_TRADING_REQUIRED"
    if evidence.get("holdout"):
        return "HOLDOUT_TESTED"
    if evidence.get("walk_forward"):
        return "WALK_FORWARD_TESTED"
    if evidence.get("out_of_sample"):
        return "OUT_OF_SAMPLE_TESTED"
    if evidence.get("in_sample"):
        return "IN_SAMPLE_ONLY"
    return "DATA_VALIDATED"


@dataclass(frozen=True)
class PromotionGates:
    min_trades_out_of_sample: int = 100
    min_oos_expectancy_R: float = 0.05
    max_prob_expectancy_le_zero: float = 0.10  # bootstrap
    min_stressed_cost_expectancy_R: float = 0.0
    max_drawdown_R: float = 20.0
    min_profitable_walk_forward_share: float = 0.6
    max_fragile_parameter_share: float = 0.34
    max_best5_share_of_profit: float = 0.5
    max_execution_stress_prob_le_zero: float = 0.25
    min_data_years: float = 5.0

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_gates(g: PromotionGates, m: dict) -> dict:
    """``m`` metric keys (None = not available -> the gate FAILS as UNAVAILABLE)."""
    checks = [
        ("adequate sample", m.get("oos_trades"), lambda v: v >= g.min_trades_out_of_sample, f">= {g.min_trades_out_of_sample}"),
        ("positive out-of-sample expectancy", m.get("oos_expectancy_R"), lambda v: v >= g.min_oos_expectancy_R,
         f">= {g.min_oos_expectancy_R} R"),
        ("expectancy uncertainty", m.get("oos_prob_expectancy_le_zero"), lambda v: v <= g.max_prob_expectancy_le_zero,
         f"P(E<=0) <= {g.max_prob_expectancy_le_zero}"),
        ("cost robustness", m.get("stressed_cost_expectancy_R"), lambda v: v > g.min_stressed_cost_expectancy_R,
         f"> {g.min_stressed_cost_expectancy_R} R under stressed costs"),
        ("reasonable drawdown", m.get("max_drawdown_R"), lambda v: v <= g.max_drawdown_R, f"<= {g.max_drawdown_R} R"),
        ("walk-forward consistency", m.get("walk_forward_profitable_share"),
         lambda v: v >= g.min_profitable_walk_forward_share, f">= {g.min_profitable_walk_forward_share}"),
        ("parameter stability", m.get("fragile_parameter_share"), lambda v: v <= g.max_fragile_parameter_share,
         f"<= {g.max_fragile_parameter_share}"),
        ("no catastrophic concentration", m.get("best5_share_of_profit"), lambda v: v <= g.max_best5_share_of_profit,
         f"<= {g.max_best5_share_of_profit}"),
        ("execution robustness", m.get("execution_stress_prob_le_zero"), lambda v: v <= g.max_execution_stress_prob_le_zero,
         f"<= {g.max_execution_stress_prob_le_zero}"),
        ("data coverage", m.get("data_years"), lambda v: v >= g.min_data_years, f">= {g.min_data_years} years"),
    ]
    rows = []
    for name, val, ok, req in checks:
        status = "UNAVAILABLE" if val is None else ("PASS" if ok(val) else "FAIL")
        rows.append({"gate": name, "value": val, "requirement": req, "status": status})
    passed = all(r["status"] == "PASS" for r in rows)
    return {"gates": rows, "all_passed": passed, "failed": [r["gate"] for r in rows if r["status"] != "PASS"],
            "decision": "ELIGIBLE_FOR_PAPER_TRADING_REVIEW" if passed else "NOT_ELIGIBLE",
            "note": "gates are research thresholds, not claims of optimality; no parameter is tuned to pass them"}
