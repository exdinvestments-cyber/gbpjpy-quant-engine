"""The strict platform-neutral execution port (Phase 1G) - INTERFACE ONLY.

Future MT4 / MT5 adapters implement ``ExecutionPort`` outside the strategy
core.  The core never sees platform tickets, error numbers, symbol suffixes or
magic-number semantics: adapters translate them into the canonical objects in
``execution.model``.  There is no network implementation in this package; the
only implementation is the deterministic ``PaperBroker`` used for tests.

The mapping from the specification's verb names to these canonical verbs is
documented in ``docs/PHASE_1G_EXECUTION_SAFETY.md``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ExecutionPort(Protocol):
    def get_connection_status(self, as_of: pd.Timestamp):
        ...

    def get_market_snapshot(self, symbol: str, as_of: pd.Timestamp):
        """Latest canonical quote at or before ``as_of`` (never a later one)."""
        ...

    def get_symbol_specification(self, symbol: str, as_of: pd.Timestamp):
        ...

    def get_capabilities(self, as_of: pd.Timestamp):
        ...

    def get_account_snapshot(self, as_of: pd.Timestamp):
        ...

    def get_open_positions(self, as_of: pd.Timestamp) -> list:
        ...

    def get_pending_orders(self, as_of: pd.Timestamp) -> list:
        ...

    def get_fills(self, client_order_key: str, as_of: pd.Timestamp) -> list:
        ...

    def request_submission(self, intent, volume, ownership, as_of: pd.Timestamp, attach_protection: bool):
        """Send ONE order for ``intent`` (the caller guarantees idempotency). Returns a PortResponse."""
        ...

    def request_protection(self, position_id: str, stop_price: float, target_price: float | None, key: str,
                           as_of: pd.Timestamp):
        ...

    def request_modification(self, request, as_of: pd.Timestamp):
        ...

    def request_cancellation(self, request, as_of: pd.Timestamp):
        ...

    def request_close(self, request, as_of: pd.Timestamp):
        ...

    def get_order_status(self, client_order_key: str, as_of: pd.Timestamp):
        ...


@runtime_checkable
class Clock(Protocol):
    def now(self) -> pd.Timestamp:
        """Strategy/event time (UTC)."""
        ...

    def monotonic(self) -> float:
        """Monotonic seconds for timeout / latency measurement (never wall-clock arithmetic)."""
        ...


class SimClock:
    """Deterministic clock for tests and research replay."""

    def __init__(self, start: pd.Timestamp):
        self._t = pd.Timestamp(start)
        self._m = 0.0

    def now(self) -> pd.Timestamp:
        return self._t

    def monotonic(self) -> float:
        return self._m

    def advance(self, seconds: float) -> pd.Timestamp:
        self._t = self._t + pd.Timedelta(seconds=seconds)
        self._m += seconds
        return self._t


@runtime_checkable
class SymbolMapper(Protocol):
    def to_broker(self, canonical: str) -> str:
        ...

    def to_canonical(self, broker_symbol: str) -> str:
        ...


class SymbolMapping:
    """Adapter-level canonical <-> provider symbol mapping (no hard-coded suffixes)."""

    def __init__(self, mapping: dict):
        self._fwd = dict(mapping)
        self._rev = {v: k for k, v in mapping.items()}
        if len(self._rev) != len(self._fwd):
            raise ValueError("symbol mapping must be one-to-one")

    def to_broker(self, canonical: str) -> str:
        if canonical not in self._fwd:
            raise KeyError(f"no provider symbol configured for {canonical}")
        return self._fwd[canonical]

    def to_canonical(self, broker_symbol: str) -> str:
        if broker_symbol not in self._rev:
            raise KeyError(f"unknown provider symbol {broker_symbol}")
        return self._rev[broker_symbol]


def canonical_rejection(adapter_code, table: dict) -> str:
    """Adapters translate platform codes into canonical reasons through their own table."""
    return table.get(adapter_code, "UNKNOWN_BROKER_REJECTION")
