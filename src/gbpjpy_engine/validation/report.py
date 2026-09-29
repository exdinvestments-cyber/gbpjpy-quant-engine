"""Canonical backtest result, dashboard datasets and research report text (Phase 1H).

Reports use neutral research language ("observed", "historical", "simulated",
"out-of-sample", "under these assumptions").  ``check_language`` rejects
marketing claims.  Every section carries its analysis label:
PRE_DECLARED_TEST, EXPLORATORY_ANALYSIS, POST_HOC_ANALYSIS or HOLDOUT_RESULT.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

BANNED_PHRASES = (r"proven(ly)? profitable", r"\bguaranteed?\b", r"\brisk[- ]free\b", r"\bsafe (strategy|returns?|profits?)\b",
                  r"money[- ]making machine", r"institutional[- ]grade returns", r"can(no|')t lose", r"sure[- ]fire",
                  r"holy grail")


class MarketingLanguage(ValueError):
    pass


def check_language(text: str) -> str:
    hits = [p for p in BANNED_PHRASES if re.search(p, text, flags=re.IGNORECASE)]
    if hits:
        raise MarketingLanguage(f"report contains prohibited claims: {hits}")
    return text


@dataclass
class BacktestResult:
    experiment_id: str
    strategy_version: str
    strategy_snapshot_id: str
    config_version: str
    dataset_id: str
    date_range: tuple
    seed: int
    analysis_label: str
    trade_count: int
    gross_R: float | None
    net_R: float | None
    expectancy_R: float | None
    profit_factor: float | None
    win_rate: float | None
    max_drawdown_R: float | None
    drawdown_duration_days: float | None
    costs: dict = field(default_factory=dict)
    long_metrics: dict = field(default_factory=dict)
    short_metrics: dict = field(default_factory=dict)
    setup_family_metrics: dict = field(default_factory=dict)
    regime_metrics: dict = field(default_factory=dict)
    walk_forward_metrics: dict = field(default_factory=dict)
    sensitivity_metrics: dict = field(default_factory=dict)
    monte_carlo_summary: dict = field(default_factory=dict)
    data_quality_warnings: list = field(default_factory=list)
    research_warnings: list = field(default_factory=list)
    validation_status: str = "NOT_VALIDATED"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, default=_default, indent=1)


def _default(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def export_dashboard(outdir, trades: pd.DataFrame, parts: dict) -> dict:
    """Machine-readable datasets for a future dashboard (CSV + JSON; no UI)."""
    d = Path(outdir)
    d.mkdir(parents=True, exist_ok=True)
    written = {}
    if len(trades):
        t = trades.sort_values("entry_time").reset_index(drop=True)
        eq = t["net_R"].cumsum()
        curve = pd.DataFrame({"exit_time": t["exit_time"], "cum_net_R": eq, "cum_gross_R": t["gross_R"].cumsum(),
                              "drawdown_R": eq - eq.cummax().clip(lower=0)})
        for name, df in (("trade_ledger", t.drop(columns=["flags"], errors="ignore")), ("r_curve", curve)):
            p = d / f"{name}.csv"
            df.to_csv(p, index=False)
            written[name] = str(p)
    for name, obj in parts.items():
        p = d / f"{name}.json"
        p.write_text(json.dumps(obj, sort_keys=True, default=_default, indent=1))
        written[name] = str(p)
    return written


def render_text(result: BacktestResult) -> str:
    r = result
    lines = [f"PHASE 1H RESEARCH REPORT - {r.analysis_label}",
             f"experiment {r.experiment_id}  snapshot {r.strategy_snapshot_id}  config {r.config_version[:16]}  dataset "
             f"{r.dataset_id}  seed {r.seed}",
             f"period {r.date_range[0]} -> {r.date_range[1]}  validation status: {r.validation_status}",
             "", "All figures are historical simulations under the stated cost assumptions; they are not forecasts.",
             f"trades {r.trade_count}  expectancy {r.expectancy_R}  R  profit factor {r.profit_factor}  win rate {r.win_rate}",
             f"net R {r.net_R}  gross R {r.gross_R}  max drawdown {r.max_drawdown_R} R ({r.drawdown_duration_days} days)",
             "", "COSTS", json.dumps(r.costs, default=_default, sort_keys=True)[:2000],
             "", "RESEARCH WARNINGS"] + [f"  - {w}" for w in r.research_warnings] + \
            ["", "DATA-QUALITY WARNINGS"] + [f"  - {w}" for w in r.data_quality_warnings[:50]]
    return check_language("\n".join(lines))
