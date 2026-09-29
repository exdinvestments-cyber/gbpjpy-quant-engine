"""Phase 1H: real-data validation and research infrastructure.

Purpose: try to FALSIFY the strategy on real GBPJPY history.  This package
reads strategy outputs and never writes back to any strategy, risk or
execution parameter (no auto-tuning).  It performs no network access and
places no order.  Without a real, provenance-documented dataset it reports
REAL_DATA_REQUIRED; synthetic data is never substituted.
"""

from .costs import (CommissionModel, CostScenario, MissingSpread, SlippageModel, SpreadModel, SwapModel,
                    standard_scenarios)
from .datastore import DataStore, RawDataModified, require_known_provenance
from .experiments import (ExperimentManifest, ExperimentRecord, Hypothesis, HypothesisRegister, ManifestTampered,
                          ResearchBudget, ResearchBudgetExhausted)
from .gates import VALIDATION_STATUSES, PromotionGates, evaluate_gates, validation_status
from .objective import objective_analysis
from .provenance import (PRICE_TYPES, DatasetProvenance, FileProvider, HistoricalDataProvider, ImportSpec, ProvenanceError,
                         describe_accepted_formats)
from .quality import DataQualityBlocked, QualityPolicy, quality_gate, research_quality_report
from .replay import (PipelineOutputs, StrategyConfigs, account_simulation, cost_scenario_ledgers, run_pipeline, signal_ledger,
                     strategy_trades, warmup)
from .report import BacktestResult, MarketingLanguage, check_language, export_dashboard
from .resampling import UTC_GRID, H4BarDefinition, compare_h4, resample_h1_to_h4, verify_aggregation
from .runner import first_real_data_run, real_data_required, run_validation
from .simulator import AMBIGUITY_POLICIES, ExitPlan, SimulatedTrade, simulate_trade
from .snapshot import configuration_freeze, strategy_snapshot
from .splits import (UNLOCK_CONFIRMATION, ChronologicalSplit, HoldoutLock, HoldoutLocked, SplitGuard, walk_forward_report,
                     walk_forward_windows)

__all__ = [n for n in dir() if not n.startswith("_")]
