"""External risk rulesets, consistency architecture and portfolio-exposure
extension point (Phase 1F).

``ExternalRiskPolicy`` lets the same GBPJPY strategy run inside a different
risk envelope (e.g. a future evaluation-account ruleset) without changing any
strategy logic.  ``GenericRuleset`` is a configurable, provider-neutral
implementation - no specific firm is encoded.  Consistency rules are
REPORTED, never used to distort trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Protocol, runtime_checkable

from .money import D, ZERO


@dataclass(frozen=True)
class ExternalContext:
    as_of: object
    currency: str
    balance: Decimal
    equity: Decimal
    hwm_equity: Decimal
    today_loss: Decimal  # realised + floating loss today (>= 0)
    open_positions: int
    hours_to_weekend_close: float
    news_status: str
    daily_pnl_history: dict = field(default_factory=dict)  # trading-day -> realised net P&L


@dataclass(frozen=True)
class ExternalPolicyResult:
    status: str  # OK | BLOCK | BREACH | UNKNOWN
    remaining_risk: Decimal | None = None
    max_volume: Decimal | None = None
    reasons: tuple = ()
    consistency: dict = field(default_factory=dict)


@runtime_checkable
class ExternalRiskPolicy(Protocol):
    name: str

    def evaluate(self, ctx: ExternalContext) -> ExternalPolicyResult:
        ...


@dataclass(frozen=True)
class GenericRuleset:
    name: str = "generic_external_ruleset"
    initial_balance: Decimal = Decimal(0)
    daily_loss_limit: Decimal | None = None  # account currency
    max_drawdown: Decimal | None = None  # account currency
    drawdown_mode: str = "STATIC"  # STATIC (from initial balance) | TRAILING (from equity high-water mark)
    max_volume: Decimal | None = None
    max_open_positions: int | None = None
    weekend_holding: str = "ALLOW"  # ALLOW | DISALLOW
    weekend_cutoff_hours: float = 4.0
    news_restricted: bool = False
    consistency_max_day_share: float | None = None  # e.g. 0.3 = no day > 30% of total profit (REPORTED only)

    def evaluate(self, ctx: ExternalContext) -> ExternalPolicyResult:
        reasons, remaining = [], []
        if self.daily_loss_limit is not None:
            left = D(self.daily_loss_limit) - D(ctx.today_loss)
            if left <= 0:
                return ExternalPolicyResult("BREACH", ZERO, reasons=("external daily loss limit reached",))
            remaining.append(left)
        if self.max_drawdown is not None:
            anchor = D(self.initial_balance) if self.drawdown_mode == "STATIC" else D(ctx.hwm_equity)
            floor = anchor - D(self.max_drawdown)
            left = D(ctx.equity) - floor
            if left <= 0:
                return ExternalPolicyResult("BREACH", ZERO, reasons=(f"external {self.drawdown_mode.lower()} drawdown breached",))
            remaining.append(left)
        if self.max_open_positions is not None and ctx.open_positions >= self.max_open_positions:
            return ExternalPolicyResult("BLOCK", reasons=("external maximum open positions reached",))
        if self.weekend_holding == "DISALLOW" and ctx.hours_to_weekend_close <= self.weekend_cutoff_hours:
            return ExternalPolicyResult("BLOCK", reasons=("external ruleset disallows weekend holding",))
        if self.news_restricted and ctx.news_status in ("EVENT_IMMINENT", "EVENT_RECENT"):
            return ExternalPolicyResult("BLOCK", reasons=("external ruleset restricts trading around news",))
        cons = {}
        if self.consistency_max_day_share is not None:
            prof = {k: v for k, v in ctx.daily_pnl_history.items() if D(v) > 0}
            tot = sum((D(v) for v in prof.values()), ZERO)
            share = max((D(v) / tot for v in prof.values()), default=ZERO) if tot > 0 else ZERO
            cons = {"max_day_share": str(share), "limit": self.consistency_max_day_share,
                    "status": "WITHIN" if share <= D(self.consistency_max_day_share) else "OUTSIDE",
                    "note": "reported for the ruleset owner; trades are never distorted to satisfy it"}
        return ExternalPolicyResult("OK", min(remaining) if remaining else None, self.max_volume, tuple(reasons), cons)


@runtime_checkable
class ExposureModel(Protocol):
    """Future portfolio extension (multiple GBP / JPY positions, correlated FX). Not needed for GBPJPY-only."""

    def correlated_risk(self, symbol: str, direction: str, open_positions, reservations) -> Decimal:
        ...


class SingleSymbolExposure:
    """Current behaviour: only GBPJPY exists, so correlated exposure equals same-symbol exposure (handled by the
    symbol caps); this placeholder returns zero ADDITIONAL correlated risk."""

    def correlated_risk(self, symbol, direction, open_positions, reservations) -> Decimal:
        return ZERO
