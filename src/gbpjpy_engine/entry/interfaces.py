"""Broker-neutral execution-research interfaces for Phase 1D.

Nothing here connects to a broker, platform or data vendor.  Future MT4/MT5
(or any other) adapters implement these protocols OUTSIDE the strategy core
and hand over plain values in UTC.

* ``QuoteSource``    - the first price observable after a given moment (plus
                        the spread known at that moment).  The historical
                        default is ``BarOpenQuotes``: the next H1 bar's open,
                        with the spread reported by the last COMPLETED bar.
* ``SlippageModel``  - research hook.  The default ``UnknownSlippage`` reports
                        that live slippage is UNKNOWN; the other models only
                        apply numbers a researcher supplies explicitly.
* ``NewsProvider``   - scheduled economic-calendar events.  No provider exists
                        yet, so news_status is UNKNOWN.  Events carry
                        ``known_since`` so a schedule published later can
                        never leak backwards; outcomes (actual/forecast
                        values) are deliberately not part of the contract.
* ``LowerTimeframeProvider`` / ``IntrabarConfirmationProvider`` - optional
                        M15/M5/M1/tick evidence for future execution research.
                        Phase 1D never depends on them (completed H1 bars are
                        the default decision model) and records their evidence
                        for research only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutableQuote:
    """First price observable at/after a decision moment, on the chart price basis."""

    time: pd.Timestamp  # when this price was first observable (UTC)
    price: float  # chart-basis price (bid/mid/ask as declared by ExecutionPriceConfig.price_basis)
    spread: float | None  # spread KNOWN at ``time`` in the configured spread units; None = unknown
    spread_source: str  # e.g. "last_completed_bar_report", "quote"
    source: str = "bar_open"


@runtime_checkable
class QuoteSource(Protocol):
    def quote_after(self, index: int) -> ExecutableQuote | None:
        """First executable price after the close of H1 bar ``index`` (None: not yet observable)."""
        ...


class BarOpenQuotes:
    """Historical OHLC default: the next bar's OPEN is the first price available after a bar close.

    The spread attached to that price is the spread reported by the bar that has just CLOSED (``index``) -
    the latest spread information that existed at that moment.  The next bar's own spread report
    summarises the whole next hour and would be future information, so it is never used.
    Limitations (documented in docs/PHASE_1D_ENTRY_INTELLIGENCE.md): the true first tradable quote,
    latency, queue position and intrabar spread are unknown in OHLC data.
    """

    def __init__(self, bars: pd.DataFrame):
        self.ts = list(pd.to_datetime(bars["timestamp"]))
        self.open = bars["open"].to_numpy(float)
        self.spread = bars["spread"].to_numpy(float) if "spread" in bars.columns else np.full(len(bars), np.nan)
        self.n = len(bars)

    def check_grid(self, timestamps) -> None:
        ts = list(pd.to_datetime(pd.Series(list(timestamps))))
        if ts != self.ts[: len(ts)] or len(ts) > self.n:
            raise ValueError("quote bars must share the H1 timestamp grid of the setup result")

    def quote_after(self, index: int) -> ExecutableQuote | None:
        j = index + 1
        if j >= self.n:
            return None
        s = self.spread[index]
        return ExecutableQuote(time=self.ts[j], price=float(self.open[j]),
                               spread=float(s) if np.isfinite(s) else None,
                               spread_source="last_completed_bar_report", source="next_bar_open")


# ---------------------------------------------------------------------------
# Slippage (research hook - never invented)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SlippageContext:
    direction: str  # "LONG" | "SHORT"
    time: pd.Timestamp
    spread_pips: float | None
    atr_pips: float | None


@dataclass(frozen=True)
class SlippageEstimate:
    pips: float | None  # adverse slippage in pips; None = unknown
    status: str  # "UNKNOWN" | "MODELLED"
    model: str
    note: str = ""


@runtime_checkable
class SlippageModel(Protocol):
    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        ...


class UnknownSlippage:
    """Default: live slippage is unknown and is reported as such (never assumed to be zero)."""

    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        return SlippageEstimate(None, "UNKNOWN", "none", "actual live slippage is unknown; no model configured")


@dataclass(frozen=True)
class FixedSlippage:
    pips: float

    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        return SlippageEstimate(float(self.pips), "MODELLED", "fixed", "researcher-supplied constant")


@dataclass(frozen=True)
class SpreadDependentSlippage:
    spread_fraction: float

    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        if ctx.spread_pips is None:
            return SlippageEstimate(None, "UNKNOWN", "spread_dependent", "spread unknown")
        return SlippageEstimate(self.spread_fraction * ctx.spread_pips, "MODELLED", "spread_dependent")


@dataclass(frozen=True)
class VolatilityDependentSlippage:
    atr_fraction: float

    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        if ctx.atr_pips is None:
            return SlippageEstimate(None, "UNKNOWN", "volatility_dependent", "ATR unknown")
        return SlippageEstimate(self.atr_fraction * ctx.atr_pips, "MODELLED", "volatility_dependent")


@dataclass(frozen=True)
class EmpiricalSlippage:
    """Quantile of OBSERVED broker slippage samples (pips) supplied by the researcher."""

    samples: Sequence[float] = field(default_factory=tuple)
    quantile: float = 0.5

    def estimate(self, ctx: SlippageContext) -> SlippageEstimate:
        if len(self.samples) == 0:
            return SlippageEstimate(None, "UNKNOWN", "empirical", "no observed samples supplied")
        return SlippageEstimate(float(np.quantile(np.asarray(self.samples, float), self.quantile)), "MODELLED", "empirical")


# ---------------------------------------------------------------------------
# News (economic calendar) - interface only, no events are ever invented
# ---------------------------------------------------------------------------
IMPORTANCE = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass(frozen=True)
class NewsEvent:
    time: pd.Timestamp  # scheduled event time (UTC)
    currency: str  # "GBP", "JPY", or "GLOBAL"
    importance: str  # LOW / MEDIUM / HIGH
    name: str
    category: str = ""
    known_since: pd.Timestamp | None = None  # when the schedule entry became known (None = long before)


@runtime_checkable
class NewsProvider(Protocol):
    def events(self, start: pd.Timestamp, end: pd.Timestamp, known_at: pd.Timestamp) -> list[NewsEvent]:
        """Scheduled events in [start, end] whose schedule was known at ``known_at``."""
        ...


# ---------------------------------------------------------------------------
# Lower-timeframe / intrabar execution research (optional, never required)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IntrabarConfirmation:
    confirmed: bool | None
    first_touch_time: pd.Timestamp | None = None
    detail: dict = field(default_factory=dict)
    source: str = "none"


@runtime_checkable
class LowerTimeframeProvider(Protocol):
    def closed_bars(self, timeframe: str, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> pd.DataFrame:
        """Canonical CLOSED bars (M15/M5/M1) or ticks ('TICK') in [start, end) with close <= as_of."""
        ...


@runtime_checkable
class IntrabarConfirmationProvider(Protocol):
    def confirm(self, direction: str, level: float | None, start: pd.Timestamp, end: pd.Timestamp,
                as_of: pd.Timestamp) -> IntrabarConfirmation:
        """Lower-timeframe evidence about how a completed H1 confirmation bar unfolded (as_of = its close)."""
        ...


__all__ = ["BarOpenQuotes", "EmpiricalSlippage", "ExecutableQuote", "FixedSlippage", "IntrabarConfirmation",
           "IntrabarConfirmationProvider", "LowerTimeframeProvider", "NewsEvent", "NewsProvider", "QuoteSource",
           "SlippageContext", "SlippageEstimate", "SlippageModel", "SpreadDependentSlippage", "UnknownSlippage",
           "VolatilityDependentSlippage"]
