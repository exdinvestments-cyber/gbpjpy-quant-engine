"""GBPJPY quant engine - Phase 1A: H4 Market Intelligence Engine.

Descriptive market-context analysis only.  This package contains NO order
placement, NO broker connectivity, NO position sizing and NO entry logic.
"""

__version__ = "0.3.0-phase1c"

from .config import H4Config, describe_config, load_config  # noqa: E402
from .engine import H4AnalysisResult, H4MarketIntelligenceEngine  # noqa: E402

__all__ = ["H4Config", "H4AnalysisResult", "H4MarketIntelligenceEngine", "describe_config", "load_config", "__version__"]
