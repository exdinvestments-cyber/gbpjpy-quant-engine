"""Canonical contract specification (Phase 1F).

Broker adapters populate these values from the real symbol metadata; nothing
here assumes every provider uses identical contract sizes or volume rules.
The default is a documented canonical GBPJPY description, not a claim about
any particular provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

from ..trade.symbol import SymbolSpec
from .money import D, NumericError


@dataclass(frozen=True)
class ContractSpec:
    symbol: str = "GBPJPY"
    base_currency: str = "GBP"
    quote_currency: str = "JPY"
    contract_size: Decimal = Decimal("100000")  # base-currency units per 1.0 volume
    digits: int = 3
    point: Decimal = Decimal("0.001")
    pip_size: Decimal = Decimal("0.01")
    min_volume: Decimal = Decimal("0.01")
    max_volume: Decimal = Decimal("100")
    volume_step: Decimal = Decimal("0.01")
    min_stop_distance_points: Decimal | None = None  # None = unknown
    margin_mode: str = "UNKNOWN"
    source: str = "canonical_default"

    def issues(self) -> list[str]:
        out = []
        try:
            for name in ("contract_size", "point", "pip_size", "min_volume", "max_volume", "volume_step"):
                if D(getattr(self, name)) <= 0:
                    out.append(f"{name} must be positive")
        except NumericError as exc:
            return [f"non-numeric contract field: {exc}"]
        if not out:
            if D(self.min_volume) > D(self.max_volume):
                out.append("min_volume > max_volume")
            if (D(self.min_volume) / D(self.volume_step)) % 1 != 0:
                out.append("min_volume is not a whole number of volume steps")
            if abs(float(D(self.point)) - 10 ** (-self.digits)) > 1e-12:
                out.append("point must equal 10**-digits")
            if (D(self.pip_size) / D(self.point)) % 1 != 0:
                out.append("pip_size must be a whole number of points")
        for c in (self.base_currency, self.quote_currency):
            if not (isinstance(c, str) and len(c) == 3 and c.isalpha() and c.isupper()):
                out.append(f"invalid currency code {c!r}")
        return out

    def matches(self, sym: SymbolSpec) -> bool:
        return (sym.symbol == self.symbol and sym.digits == self.digits
                and math.isclose(sym.pip_size, float(self.pip_size)) and math.isclose(sym.point, float(self.point)))

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Decimal) else v) for k, v in self.__dict__.items()}


GBPJPY_CONTRACT = ContractSpec()


@runtime_checkable
class ContractSpecSource(Protocol):
    def contract_spec(self, symbol: str) -> ContractSpec:
        """The provider's real contract metadata mapped to the canonical ContractSpec (adapter-side)."""
        ...
