"""Phase 1E: Trade Construction Engine (analysis only).

Turns ACCEPTED Phase 1D entry candidates into PROPOSED TRADES: structural
invalidation stop first, structural targets second, R relationships last.  A
proposed trade is NOT an order: no volume, lot size, monetary or percentage
risk, leverage, order or broker connection exists anywhere in this package.
"""

from .config import STOP_TYPES, TradeConfig, load_trade_config, trade_config_from_dict
from .construct import TRADE_STATES, TradeConstruction, TradeContext, construct
from .engine import TradeConstructionEngine, TradeResult
from .interfaces import (UNKNOWN_CONSTRAINTS, BreakEvenPolicy, BreakEvenSpec, BrokerConstraintSource, BrokerStopConstraints,
                         CostEstimate, DisabledPolicy, ExitLeg, ExitPlan, FixedCommission, ManagementAction,
                         ManagementContext, TimeExitPolicy, TimeExitSpec, TrailingPolicy, TrailingSpec,
                         TransactionCostModel, UnknownCosts)
from .report import explain_trade
from .symbol import GBPJPY, SymbolSpec, SymbolSpecSource

__all__ = ["BreakEvenPolicy", "BreakEvenSpec", "BrokerConstraintSource", "BrokerStopConstraints", "CostEstimate",
           "DisabledPolicy", "ExitLeg", "ExitPlan", "FixedCommission", "GBPJPY", "ManagementAction", "ManagementContext",
           "STOP_TYPES", "SymbolSpec", "SymbolSpecSource", "TRADE_STATES", "TimeExitPolicy", "TimeExitSpec", "TradeConfig",
           "TradeConstruction", "TradeConstructionEngine", "TradeContext", "TradeResult", "TrailingPolicy", "TrailingSpec",
           "TransactionCostModel", "UNKNOWN_CONSTRAINTS", "UnknownCosts", "construct", "explain_trade",
           "load_trade_config", "trade_config_from_dict"]
