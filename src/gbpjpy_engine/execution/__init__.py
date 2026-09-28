"""Phase 1G: Execution Safety (RISK-APPROVED TRADE -> ORDER INTENT -> platform-neutral order lifecycle).

Execution may only ACCEPT, DEFER, REJECT, CANCEL, RECONCILE or HALT.  It never
improves a price, moves a stop, extends a target or raises a volume.  There is
NO platform connection and NO network code here: the only ``ExecutionPort``
implementation is the deterministic, testing-only ``PaperBroker``.  Future
MT4 / MT5 adapters implement the port outside the strategy core.
"""

from .config import ExecutionPolicy, execution_policy_from_dict, load_execution_policy
from .engine import ExecutionEngine, GateDependencies, IllegalTransition
from .model import (ACCOUNT_MODES, CONNECTION_STATES, EXECUTION_STATES, LIFECYCLE, ORDER_TYPES, REJECTION_REASONS,
                    SESSION_STATES, TERMINAL, TRANSIENT_REJECTIONS, BrokerCapabilities, BrokerOrderView, BrokerPositionView,
                    CancelRequest, CloseRequest, ConnectionStatus, FillEvent, MarketSnapshot, ModificationRequest, OrderIntent,
                    OwnershipTag, PortResponse, Position, StrategyIdentity)
from .paper import FaultPlan, PaperBroker
from .port import Clock, ExecutionPort, SimClock, SymbolMapper, SymbolMapping, canonical_rejection
from .report import explain_execution
from .store import CorruptedExecutionState, ExecutionState, InMemoryExecutionStore, JsonFileExecutionStore

__all__ = ["ACCOUNT_MODES", "CONNECTION_STATES", "EXECUTION_STATES", "LIFECYCLE", "ORDER_TYPES", "REJECTION_REASONS",
           "SESSION_STATES", "TERMINAL", "TRANSIENT_REJECTIONS", "BrokerCapabilities", "BrokerOrderView", "BrokerPositionView",
           "CancelRequest", "Clock", "CloseRequest", "ConnectionStatus", "CorruptedExecutionState", "ExecutionEngine",
           "ExecutionPolicy", "ExecutionPort", "ExecutionState", "FaultPlan", "FillEvent", "GateDependencies",
           "IllegalTransition", "InMemoryExecutionStore", "JsonFileExecutionStore", "MarketSnapshot", "ModificationRequest",
           "OrderIntent", "OwnershipTag", "PaperBroker", "PortResponse", "Position", "SimClock", "StrategyIdentity",
           "SymbolMapper", "SymbolMapping", "canonical_rejection", "execution_policy_from_dict", "explain_execution",
           "load_execution_policy"]
