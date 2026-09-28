"""Canonical, provider-neutral account state and account events (Phase 1F).

Future adapters convert platform account/position data into these objects.
The account currency is whatever the account reports - nothing assumes GBP.
Deposits, withdrawals and corrections are explicit ``BalanceAdjustment``
events so they are never mistaken for trading profit or loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

import pandas as pd

from .money import D, NumericError

PROFILES = ("PERSONAL", "EXTERNAL_RULESET")


@dataclass(frozen=True)
class OpenPosition:
    position_id: str
    symbol: str
    direction: str  # LONG | SHORT
    volume: Decimal
    entry_price: float
    stop_price: float | None  # None = no protective stop (treated as unbounded risk)
    opened_at: pd.Timestamp
    current_price: float | None = None
    floating_pnl: Decimal = Decimal(0)  # account currency
    trade_proposal_id: str | None = None
    entry_candidate_id: str | None = None
    setup_id: str | None = None


@dataclass(frozen=True)
class AccountState:
    account_id: str
    currency: str
    balance: Decimal
    equity: Decimal
    free_margin: Decimal | None
    used_margin: Decimal | None
    margin_level: Decimal | None  # percent; None when no margin is used / unknown
    floating_pnl: Decimal
    timestamp: pd.Timestamp
    open_positions: tuple = ()
    pending_risk: Decimal = Decimal(0)  # risk the provider already reports as pending (outside this engine)
    broker_leverage: float | None = None  # informational; affects margin feasibility only
    realised_daily_pnl: Decimal | None = None  # provider-reported, informational
    realised_weekly_pnl: Decimal | None = None
    realised_monthly_pnl: Decimal | None = None
    peak_equity: Decimal | None = None  # provider-reported, informational (the engine keeps its own)
    profile: str = "PERSONAL"
    source: str = "unknown"

    def issues(self) -> list[str]:
        out = []
        if not (isinstance(self.currency, str) and len(self.currency) == 3 and self.currency.isalpha() and self.currency.isupper()):
            out.append("invalid account currency")
        if self.timestamp is None or pd.Timestamp(self.timestamp).tzinfo is None:
            out.append("account timestamp must be timezone-aware")
        try:
            if D(self.balance) <= 0:
                out.append("balance must be positive")
            if D(self.equity) <= 0:
                out.append("equity must be positive")
            D(self.floating_pnl)
            D(self.pending_risk)
            for name in ("free_margin", "used_margin", "margin_level"):
                v = getattr(self, name)
                if v is not None and D(v) < 0:
                    out.append(f"{name} must not be negative")
            for p in self.open_positions:
                if D(p.volume) <= 0:
                    out.append(f"position {p.position_id} has non-positive volume")
                if p.direction not in ("LONG", "SHORT"):
                    out.append(f"position {p.position_id} has invalid direction")
        except NumericError as exc:
            out.append(f"non-finite account value: {exc}")
        if self.profile not in PROFILES:
            out.append(f"unknown account profile {self.profile!r}")
        return out


@dataclass(frozen=True)
class BalanceAdjustment:
    adjustment_id: str
    kind: str  # DEPOSIT | WITHDRAWAL | CORRECTION
    amount: Decimal  # positive for deposits, negative for withdrawals / negative corrections
    timestamp: pd.Timestamp
    note: str = ""


@dataclass(frozen=True)
class ClosedTradeResult:
    """Reported by a future execution engine; the only way trading P&L enters the risk ledger."""

    result_id: str
    position_id: str
    trade_proposal_id: str | None
    realised_pnl: Decimal  # account currency, before fees/swap
    fees: Decimal
    swap: Decimal
    close_timestamp: pd.Timestamp
    close_reason: str  # STOP | TARGET | MANUAL | TIME | PARTIAL | OTHER
    closed_volume: Decimal | None = None
    partial: bool = False

    @property
    def net(self) -> Decimal:
        return D(self.realised_pnl) + D(self.fees) + D(self.swap)


@dataclass(frozen=True)
class StopChange:
    position_id: str
    old_stop: float | None
    new_stop: float
    timestamp: pd.Timestamp
    reason: str = ""


@runtime_checkable
class AccountStateSource(Protocol):
    def account_state(self, as_of: pd.Timestamp) -> AccountState:
        ...


@runtime_checkable
class ClosedResultSink(Protocol):
    def record_closed_result(self, result: ClosedTradeResult) -> dict:
        ...


def positions_by(account: AccountState, symbol: str, direction: str | None = None) -> list:
    return [p for p in account.open_positions if p.symbol == symbol and (direction is None or p.direction == direction)]


__all__ = ["AccountState", "AccountStateSource", "BalanceAdjustment", "ClosedResultSink", "ClosedTradeResult", "OpenPosition",
           "PROFILES", "StopChange", "positions_by", "field"]
