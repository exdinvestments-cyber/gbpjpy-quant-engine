"""Provider-neutral margin estimation (Phase 1F).

Margin is a FEASIBILITY check only.  Leverage never changes the strategy's
risk appetite: a 1:500 account and a 1:30 account receive the same permitted
risk and volume whenever both can carry the margin (tested).

``NotionalLeverageMarginModel``: margin = volume x contract_size (GBP notional)
converted to the account currency at ``as_of`` / account leverage.  Without a
known leverage the estimate is UNKNOWN.  Adapters may supply the provider's own
requirement through ``MarginModel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from .fx import ConversionUnavailable, factor
from .money import D


@dataclass(frozen=True)
class MarginEstimate:
    amount: Decimal | None  # account currency; None = unknown
    status: str  # KNOWN | UNKNOWN
    method: str
    rates: tuple = ()
    note: str = ""


@runtime_checkable
class MarginModel(Protocol):
    def estimate(self, contract, volume, account, as_of, rates, max_age) -> MarginEstimate:
        ...


class NotionalLeverageMarginModel:
    def estimate(self, contract, volume, account, as_of, rates, max_age) -> MarginEstimate:
        lev = account.broker_leverage
        if lev is None or not lev or lev <= 0:
            return MarginEstimate(None, "UNKNOWN", "notional/leverage", note="account leverage unknown")
        try:
            f, used = factor(contract.base_currency, account.currency, as_of, rates, max_age)
        except ConversionUnavailable as exc:
            return MarginEstimate(None, "UNKNOWN", "notional/leverage", note=str(exc))
        amt = D(volume) * D(contract.contract_size) * f / D(lev)
        return MarginEstimate(amt, "KNOWN", "notional/leverage", tuple(r.to_dict() for r in used))


@dataclass(frozen=True)
class FixedMarginPerVolume:
    """Adapter/researcher-supplied margin per 1.0 volume in the account currency."""

    per_volume: Decimal

    def estimate(self, contract, volume, account, as_of, rates, max_age) -> MarginEstimate:
        return MarginEstimate(D(self.per_volume) * D(volume), "KNOWN", "fixed_per_volume")


class UnknownMargin:
    def estimate(self, contract, volume, account, as_of, rates, max_age) -> MarginEstimate:
        return MarginEstimate(None, "UNKNOWN", "none", note="no margin model")
