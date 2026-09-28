"""Phase 1D: Entry Intelligence Engine (analysis only).

Consumes a Phase 1C ``H1SetupResult`` and decides whether a QUALIFIED H1 setup
has developed into a precise, timely and executable ENTRY CANDIDATE.  An entry
candidate is NOT an order: no position size, risk amount, stop loss, take
profit, order or broker connection exists anywhere in this package.
"""

from .config import CONFIRMATION_FAMILIES, EntryConfig, entry_config_from_dict, load_entry_config
from .engine import EntryIntelligenceEngine, EntryResult
from .interfaces import (BarOpenQuotes, EmpiricalSlippage, ExecutableQuote, FixedSlippage, IntrabarConfirmation,
                         IntrabarConfirmationProvider, LowerTimeframeProvider, NewsEvent, NewsProvider, QuoteSource,
                         SlippageModel, SpreadDependentSlippage, UnknownSlippage, VolatilityDependentSlippage)
from .lifecycle import EntryCandidate
from .report import explain_entry_candidate

__all__ = ["BarOpenQuotes", "CONFIRMATION_FAMILIES", "EmpiricalSlippage", "EntryCandidate", "EntryConfig",
           "EntryIntelligenceEngine", "EntryResult", "ExecutableQuote", "FixedSlippage", "IntrabarConfirmation",
           "IntrabarConfirmationProvider", "LowerTimeframeProvider", "NewsEvent", "NewsProvider", "QuoteSource",
           "SlippageModel", "SpreadDependentSlippage", "UnknownSlippage", "VolatilityDependentSlippage",
           "entry_config_from_dict", "explain_entry_candidate", "load_entry_config"]
