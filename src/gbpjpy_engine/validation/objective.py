"""The GBP 500,000 annual ambition - analysed ONLY after validation, never an input (Phase 1H).

This module has no path into any strategy, risk or execution decision: it
imports nothing from those layers and receives only already-computed results.
``test_validation_objective_firewall`` proves that changing the objective leaves
signals, entries, stops, targets, risk policy and trade acceptance identical.
"""

from __future__ import annotations

ANALYSABLE_STATUSES = ("HOLDOUT_TESTED", "PAPER_TRADING_REQUIRED")


def objective_analysis(validation_status: str, annual_target_gbp: float = 500_000.0, expectancy_R: float | None = None,
                       expectancy_ci: tuple | None = None, trades_per_year: float | None = None,
                       risk_percents=(0.25, 0.5, 1.0), max_drawdown_R_p95: float | None = None) -> dict:
    """Mathematics of the ambition under fixed-fractional risk, if and only if the strategy has been validated."""
    if validation_status not in ANALYSABLE_STATUSES:
        return {"status": "NOT_EVALUATED_STRATEGY_NOT_VALIDATED", "validation_status": validation_status,
                "statement": "the annual objective is not analysed before out-of-sample and holdout validation; it never "
                             "influences strategy or risk decisions"}
    if not expectancy_R or not trades_per_year or expectancy_R <= 0:
        return {"status": "EVIDENCE_DOES_NOT_SUPPORT_OBJECTIVE", "expectancy_R": expectancy_R,
                "statement": "validated expectancy is not positive - no capital level makes the objective supportable"}
    rows = []
    for rp in risk_percents:
        annual_R = expectancy_R * trades_per_year
        # simple (non-compounded) annual return fraction = risk% x R per year
        ret = rp / 100 * annual_R
        lo = hi = None
        if expectancy_ci and expectancy_ci[0] is not None:
            lo = rp / 100 * expectancy_ci[0] * trades_per_year
            hi = rp / 100 * expectancy_ci[1] * trades_per_year
        rows.append({"risk_percent": rp, "expected_annual_return_fraction": ret,
                     "required_starting_capital_gbp": annual_target_gbp / ret if ret > 0 else None,
                     "capital_range_from_expectancy_ci_gbp": [annual_target_gbp / hi if hi and hi > 0 else None,
                                                              annual_target_gbp / lo if lo and lo > 0 else None],
                     "drawdown_p95_fraction": (rp / 100 * max_drawdown_R_p95) if max_drawdown_R_p95 else None})
    return {"status": "ANALYSED_AFTER_VALIDATION", "annual_target_gbp": annual_target_gbp, "expectancy_R": expectancy_R,
            "trades_per_year": trades_per_year, "rows": rows,
            "caveats": ["uncertainty in expectancy dominates the required capital", "capacity/slippage limits at large size are "
                        "not linear", "historical frequency may not persist", "the objective never changes risk policy"],
            "compatibility": "the objective is only as credible as the lower confidence bound of expectancy"}


def objective_is_isolated() -> bool:
    """Structural check used by tests: this module imports no strategy/risk/execution engine."""
    import ast
    import inspect
    import sys

    tree = ast.parse(inspect.getsource(sys.modules[__name__]))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    return not any(m and any(x in m for x in ("entry", "trade", "risk", "execution", "h1", "context", "engine")) for m in mods)

