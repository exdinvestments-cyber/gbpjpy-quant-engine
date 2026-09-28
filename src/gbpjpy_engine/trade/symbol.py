"""Canonical symbol metadata and GBPJPY price / pip mathematics (Phase 1E).

Every price-distance conversion in the trade-construction layer goes through
``SymbolSpec`` - nothing else in ``trade/`` assumes a pip size, point size or
price precision.  A future MT4/MT5 adapter supplies the broker's real symbol
specification through ``SymbolSpecSource``; the core only consumes the
canonical ``SymbolSpec``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, Decimal
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SymbolSpec:
    symbol: str = "GBPJPY"
    digits: int = 3  # quoted decimals (3-digit JPY quotes)
    point: float = 0.001  # smallest price increment
    pip_size: float = 0.01  # one pip for a JPY-quoted pair
    source: str = "canonical_default"

    def validate(self) -> None:
        for name in ("point", "pip_size"):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive finite number")
        if self.digits < 0 or abs(10 ** (-self.digits) - self.point) > 1e-12:
            raise ValueError("point must equal 10**-digits")
        ratio = self.pip_size / self.point
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("pip_size must be a whole number of points")

    @property
    def points_per_pip(self) -> int:
        return int(round(self.pip_size / self.point))

    # ---- conversions --------------------------------------------------------
    def to_pips(self, price_distance: float) -> float:
        return price_distance / self.pip_size

    def to_points(self, price_distance: float) -> float:
        return price_distance / self.point

    def pips_to_price(self, pips: float) -> float:
        return pips * self.pip_size

    def points_to_price(self, points: float) -> float:
        return points * self.point

    def normalize(self, price: float, mode: str = "nearest") -> float:
        """Round a price to the symbol precision. ``mode``: nearest | down | up (exact decimal arithmetic)."""
        if not math.isfinite(price):
            raise ValueError("cannot normalise a non-finite price")
        rounding = {"nearest": ROUND_HALF_EVEN, "down": ROUND_FLOOR, "up": ROUND_CEILING}[mode]
        q = Decimal(1).scaleb(-self.digits)
        return float(Decimal(repr(price)).quantize(q, rounding=rounding))

    def normalize_away(self, price: float, direction: int, reference_side: str) -> float:
        """Round so that a level moves AWAY from the entry for stops (``reference_side='stop'``: never tighter) or
        TOWARD the entry for targets (``'target'``: never farther)."""
        if reference_side == "stop":
            return self.normalize(price, "down" if direction > 0 else "up")
        return self.normalize(price, "down" if direction > 0 else "up")

    def to_dict(self) -> dict:
        return {"symbol": self.symbol, "digits": self.digits, "point": self.point, "pip_size": self.pip_size,
                "points_per_pip": self.points_per_pip, "source": self.source}


GBPJPY = SymbolSpec()


@runtime_checkable
class SymbolSpecSource(Protocol):
    def symbol_spec(self, symbol: str) -> SymbolSpec:
        """The broker's actual symbol specification mapped to the canonical SymbolSpec (adapter-side)."""
        ...


__all__ = ["GBPJPY", "SymbolSpec", "SymbolSpecSource"]
