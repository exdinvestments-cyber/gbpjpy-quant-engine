"""Timestamp-aware currency conversion (Phase 1F).

``RateProvider.rate(base, quote, as_of)`` returns the latest known quote
``1 base = rate quote`` observed at or before ``as_of`` (never a later one).
``convert`` tries the direct pair, its inverse, then one hop through a pivot
currency.  Every rate used is recorded.  Missing or stale rates raise
``ConversionUnavailable`` - the engine fails closed and never guesses.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

import pandas as pd

from .money import ONE, D


class ConversionUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class ConversionRate:
    base: str
    quote: str
    rate: Decimal  # 1 base = rate quote
    timestamp: pd.Timestamp
    source: str = "unknown"

    def to_dict(self) -> dict:
        return {"pair": f"{self.base}{self.quote}", "rate": str(self.rate), "timestamp": self.timestamp.isoformat(),
                "source": self.source}


@runtime_checkable
class RateProvider(Protocol):
    def rate(self, base: str, quote: str, as_of: pd.Timestamp) -> ConversionRate | None:
        ...


class StaticRates:
    """Research / test provider: timestamped quotes; ``rate`` never returns a quote newer than ``as_of``."""

    def __init__(self, quotes=()):
        self._q: dict = {}
        for q in quotes:
            self.add(q)

    def add(self, q: ConversionRate) -> None:
        if q.timestamp.tzinfo is None:
            raise ValueError("conversion rates must be timezone-aware")
        lst = self._q.setdefault((q.base, q.quote), [])
        lst.append(q)
        lst.sort(key=lambda r: r.timestamp)

    def rate(self, base, quote, as_of):
        lst = self._q.get((base, quote))
        if not lst:
            return None
        i = bisect_right([r.timestamp for r in lst], pd.Timestamp(as_of))
        return lst[i - 1] if i else None


def rates_from_bars(bars: pd.DataFrame, base: str = "GBP", quote: str = "JPY") -> StaticRates:
    """GBPJPY closes as timestamped rates, each available only at its bar CLOSE (look-ahead safe)."""
    tf = bars["timestamp"].diff().median() if len(bars) > 1 else pd.Timedelta(hours=1)
    avail = bars["available_at"] if "available_at" in bars.columns else bars["timestamp"] + tf
    return StaticRates(ConversionRate(base, quote, D(float(c)), pd.Timestamp(t), "bar_close")
                       for c, t in zip(bars["close"], avail))


def _fresh(r: ConversionRate | None, as_of, max_age: pd.Timedelta) -> ConversionRate | None:
    if r is None:
        return None
    if r.timestamp > as_of:
        raise ConversionUnavailable(f"rate {r.base}{r.quote} is from the future ({r.timestamp} > {as_of})")
    if as_of - r.timestamp > max_age:
        raise ConversionUnavailable(f"rate {r.base}{r.quote} is stale ({as_of - r.timestamp} > {max_age})")
    if D(r.rate) <= 0:
        raise ConversionUnavailable(f"rate {r.base}{r.quote} is not positive")
    return r


def factor(src: str, dst: str, as_of, provider, max_age: pd.Timedelta, pivots=("GBP", "USD", "EUR", "JPY")) -> tuple[Decimal, list]:
    """Multiplier converting an amount in ``src`` into ``dst`` at ``as_of`` and the rates used."""
    as_of = pd.Timestamp(as_of)
    if src == dst:
        return ONE, []
    if provider is None:
        raise ConversionUnavailable(f"no rate provider for {src}->{dst}")

    def one(a, b):
        r = _fresh(provider.rate(a, b, as_of), as_of, max_age)
        if r is not None:
            return D(r.rate), [r]
        r = _fresh(provider.rate(b, a, as_of), as_of, max_age)
        if r is not None:
            return ONE / D(r.rate), [r]
        return None

    got = one(src, dst)
    if got:
        return got
    for p in pivots:
        if p in (src, dst):
            continue
        a, b = one(src, p), one(p, dst)
        if a and b:
            return a[0] * b[0], a[1] + b[1]
    raise ConversionUnavailable(f"no conversion path {src}->{dst} at {as_of}")
