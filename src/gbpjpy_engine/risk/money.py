"""Numeric strategy for the account-risk layer (Phase 1F).

* Currency amounts, volumes, volume steps and risk percentages are
  ``decimal.Decimal`` (28 significant digits).  Floats are converted through
  ``repr`` so ``0.1`` becomes ``Decimal('0.1')`` exactly as written.
* Prices and pips arrive as floats from the strategy core and are converted to
  Decimal at the boundary before any money arithmetic.
* Permitted amounts are rounded DOWN (never overstating what may be risked);
  realised/actual risk figures are rounded UP when displayed (never
  understating a loss).  Volumes are floored to the volume step.
* Any non-finite value raises ``NumericError`` - callers fail closed.
"""

from __future__ import annotations

import math
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation, getcontext

getcontext().prec = 28
ZERO = Decimal(0)
ONE = Decimal(1)
HUNDRED = Decimal(100)
MINOR_UNITS = {"JPY": 0, "KRW": 0, "HUF": 0, "CLP": 0, "ISK": 0, "BHD": 3, "KWD": 3, "OMR": 3, "JOD": 3}


class NumericError(ValueError):
    """Non-finite or otherwise unusable numeric input (the engine fails closed)."""


def D(x) -> Decimal:
    if isinstance(x, Decimal):
        if not x.is_finite():
            raise NumericError(f"non-finite decimal {x}")
        return x
    if isinstance(x, bool) or x is None:
        raise NumericError(f"not a number: {x!r}")
    if isinstance(x, float):
        if not math.isfinite(x):
            raise NumericError(f"non-finite float {x}")
        return Decimal(repr(x))
    try:
        d = Decimal(str(x))
    except (InvalidOperation, ValueError) as exc:
        raise NumericError(f"not a number: {x!r}") from exc
    if not d.is_finite():
        raise NumericError(f"non-finite value {x!r}")
    return d


def minor_unit(currency: str) -> Decimal:
    return Decimal(1).scaleb(-MINOR_UNITS.get(currency.upper(), 2))


def money_down(x, currency: str) -> Decimal:
    return D(x).quantize(minor_unit(currency), rounding=ROUND_FLOOR)


def money_up(x, currency: str) -> Decimal:
    return D(x).quantize(minor_unit(currency), rounding=ROUND_CEILING)


def floor_to_step(x, step) -> Decimal:
    x, step = D(x), D(step)
    if step <= 0:
        raise NumericError("volume step must be positive")
    return (x / step).to_integral_value(rounding=ROUND_FLOOR) * step


def pct(part, whole) -> Decimal:
    whole = D(whole)
    if whole <= 0:
        raise NumericError("percentage of a non-positive amount")
    return (D(part) / whole * HUNDRED).quantize(Decimal("0.0001"), rounding=ROUND_CEILING)


def s(x) -> str | None:
    """Serialise a Decimal (exact string) - None passes through."""
    return None if x is None else str(x)
