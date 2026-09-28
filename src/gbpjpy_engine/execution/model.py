"""Canonical execution objects (Phase 1G) - platform neutral.

Nothing here talks to any platform.  Naming note: the protected Phase 1A
safety test forbids certain order-function identifiers in the package, so the
canonical port verbs are ``request_submission`` / ``request_modification`` /
``request_cancellation`` / ``request_close`` and protective levels are
``stop_price`` / ``target_price``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import pandas as pd

ORDER_TYPES = ("MARKET", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP")
LIFECYCLE = ("CREATED", "VALIDATING", "SUBMIT_READY", "SUBMISSION_REQUESTED", "ACKNOWLEDGED", "PARTIALLY_FILLED",
             "FILLED", "REJECTED", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL_FAILED", "EXPIRED",
             "UNKNOWN", "RECONCILIATION_REQUIRED")
TERMINAL = ("FILLED", "REJECTED", "CANCELLED", "EXPIRED")
CONNECTION_STATES = ("CONNECTED", "DEGRADED", "DISCONNECTED", "UNKNOWN")
SESSION_STATES = ("OPEN", "CLOSED", "HALTED", "UNKNOWN")
EXECUTION_STATES = ("EXECUTION_ENABLED", "EXECUTION_PAUSED", "EXECUTION_HALTED")
ACCOUNT_MODES = ("HEDGING", "NETTING")
REJECTION_REASONS = ("INVALID_VOLUME", "INVALID_PRICE", "INVALID_STOPS", "MARKET_CLOSED", "INSUFFICIENT_MARGIN",
                     "TRADE_DISABLED", "PRICE_CHANGED", "OFF_QUOTES", "SYMBOL_DISABLED", "REQUOTE",
                     "TRANSIENT_INFRASTRUCTURE", "UNKNOWN_BROKER_REJECTION")
TRANSIENT_REJECTIONS = ("PRICE_CHANGED", "OFF_QUOTES", "REQUOTE", "TRANSIENT_INFRASTRUCTURE")


@dataclass(frozen=True)
class StrategyIdentity:
    """Adapter-neutral strategy identifier (an adapter may map ``numeric_id`` to a platform magic number)."""

    name: str = "GBPJPY_H4H1_CORE"
    instance: str = "default"
    numeric_id: int = 4411  # configuration value; the core never interprets it


@dataclass(frozen=True)
class OwnershipTag:
    strategy: str
    instance: str
    numeric_id: int
    setup_id: str
    entry_candidate_id: str
    trade_proposal_id: str
    risk_approval_id: str
    order_intent_id: str
    client_order_key: str  # stable idempotency key; adapters carry it in a comment / external mapping

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class OrderIntent:
    """Immutable canonical intent (Phase A).  Reference price is NOT a guaranteed fill price."""

    order_intent_id: str
    risk_approval_id: str
    trade_proposal_id: str
    entry_candidate_id: str
    setup_id: str
    created_at: pd.Timestamp
    expires_at: pd.Timestamp
    symbol: str
    direction: str  # LONG | SHORT
    order_type: str  # MARKET | BUY_LIMIT | SELL_LIMIT | BUY_STOP | SELL_STOP
    reference_entry: float  # Phase 1E/1F executable reference (ASK for long, BID for short)
    pending_price: float | None  # pending orders only
    stop_price: float  # structural stop (never tightened / widened by execution)
    target_price: float
    approved_volume: Decimal  # Phase 1F ceiling - execution may only keep or reduce it
    approved_risk_amount: Decimal  # Phase 1F permitted monetary risk (account currency)
    account_currency: str
    extra_risk_pips: Decimal  # Phase 1F slippage/commission allowances already in the risk figure
    minimum_net_r: float
    cost_extra_pips: float  # Phase 1E cost assumptions used for estimated net R
    atr_price: float  # H1 ATR at proposal time (for deterioration in ATR)
    max_spread_pips: float
    max_deterioration_pips: float
    max_slippage_pips: float | None
    nearest_barrier_price: float | None  # first opposing structure on the path (room recheck)
    ownership: OwnershipTag
    strategy_metadata: tuple = ()  # frozen (key, value) pairs
    risk_metadata: tuple = ()
    reason_codes: tuple = ()
    fill_note: str = "reference price is not a guaranteed fill; the actual fill is recorded separately"

    @property
    def d(self) -> int:
        return 1 if self.direction == "LONG" else -1

    def to_dict(self) -> dict:
        out = asdict(self)
        for k in ("created_at", "expires_at"):
            out[k] = getattr(self, k).isoformat()
        out["approved_volume"] = str(self.approved_volume)
        out["approved_risk_amount"] = str(self.approved_risk_amount)
        out["extra_risk_pips"] = str(self.extra_risk_pips)
        out["strategy_metadata"] = dict(self.strategy_metadata)
        out["risk_metadata"] = dict(self.risk_metadata)
        out["reason_codes"] = list(self.reason_codes)
        return out

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str  # canonical
    bid: float
    ask: float
    timestamp: pd.Timestamp  # quote time (provider clock mapped to UTC)
    session_status: str = "UNKNOWN"
    source: str = "unknown"


@dataclass(frozen=True)
class ConnectionStatus:
    state: str  # CONNECTED | DEGRADED | DISCONNECTED | UNKNOWN
    timestamp: pd.Timestamp
    detail: str = ""


@dataclass(frozen=True)
class BrokerCapabilities:
    """Reported by the adapter - the core never guesses."""

    account_mode: str  # HEDGING | NETTING
    supports_market: bool
    supports_pending: bool
    atomic_protection: bool  # entry with stop/target in one request
    min_volume: Decimal
    max_volume: Decimal
    volume_step: Decimal
    stop_level_points: Decimal | None
    freeze_level_points: Decimal | None
    filling_modes: tuple = ()
    reported_at: pd.Timestamp | None = None
    source: str = "adapter"


@dataclass(frozen=True)
class PortResponse:
    status: str  # ACK | REJECTED | TIMEOUT | DISCONNECTED | ERROR
    broker_ref: str | None = None
    rejection_reason: str | None = None  # canonical (REJECTION_REASONS)
    broker_time: pd.Timestamp | None = None
    detail: str = ""


@dataclass(frozen=True)
class FillEvent:
    fill_id: str  # unique per fill (duplicates are ignored)
    client_order_key: str
    broker_ref: str | None
    volume: Decimal
    price: float
    timestamp: pd.Timestamp
    commission: Decimal = Decimal(0)
    position_id: str | None = None


@dataclass(frozen=True)
class BrokerOrderView:
    broker_ref: str
    symbol: str  # canonical (after reverse mapping)
    direction: str
    order_type: str
    volume: Decimal
    price: float | None
    state: str  # WORKING | FILLED | CANCELLED | REJECTED
    client_order_key: str | None  # None = not ours / unknown ownership
    strategy_numeric_id: int | None = None


@dataclass(frozen=True)
class BrokerPositionView:
    position_id: str
    symbol: str
    direction: str
    volume: Decimal
    average_price: float
    stop_price: float | None
    target_price: float | None
    client_order_key: str | None
    strategy_numeric_id: int | None = None
    opened_at: pd.Timestamp | None = None


@dataclass(frozen=True)
class CloseRequest:
    position_id: str
    volume: Decimal  # full or partial
    reason: str
    timestamp: pd.Timestamp
    idempotency_key: str
    emergency: bool = False


@dataclass(frozen=True)
class ModificationRequest:
    position_id: str
    field: str  # STOP | TARGET
    old_price: float | None
    new_price: float
    reason: str
    timestamp: pd.Timestamp
    idempotency_key: str
    risk_policy_approval: str | None = None  # an explicit future risk-policy authorisation id


@dataclass(frozen=True)
class CancelRequest:
    order_intent_id: str
    broker_ref: str | None
    reason: str
    timestamp: pd.Timestamp
    idempotency_key: str


@dataclass
class Position:
    """Canonical position record owned by this strategy."""

    position_id: str
    symbol: str
    direction: str
    volume: Decimal
    average_entry: float
    stop_price: float | None
    target_price: float | None
    opened_at: str
    unrealised_pnl: Decimal = Decimal(0)
    remaining_risk: Decimal | None = None
    execution_status: str = "OPEN"
    protection_status: str = "PENDING"  # PENDING | PROTECTED | UNPROTECTED_POSITION
    source_order_intent_ids: list = field(default_factory=list)
