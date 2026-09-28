"""Canonical GBPJPY H4 bar representation.

Conventions (applied identically to historical files and, later, live data):

* ``timestamp`` is the bar OPEN time, timezone-aware, normalised to UTC.
* ``close_time`` = ``timestamp`` + timeframe.  A bar's information (and every
  feature computed from it) is only available at ``close_time``.
* Prices are floats in JPY.  ``volume`` and ``spread`` are optional; when
  absent they are NaN and flagged by validation - never invented.
* ``volume`` is BROKER TICK VOLUME (number of quote updates on one broker's
  feed).  Spot FX is decentralised: tick volume is not exchange-traded volume
  and differs between brokers.  Volume and spread are preserved for later
  execution / market-quality modules and are not used by Phase 1A features.
* Future MT4/MT5 adapters must convert broker server timestamps to this
  canonical UTC form before data reaches the strategy core (see
  ``data.interfaces``).
* Nothing in this module sorts, de-duplicates or repairs data.  That is left
  to explicit, logged validation decisions.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "spread", "source"]
PRICE_COLUMNS = ["open", "high", "low", "close"]

_DEFAULT_ALIASES = {
    "time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "tick_volume": "volume",
    "tickvol": "volume",
    "vol": "volume",
}


class DataIntegrityError(ValueError):
    """Raised when data cannot be safely used (never silently repaired)."""

    def __init__(self, message: str, report=None):
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class Bar:
    """One canonical H4 bar - the shared unit for historical and live feeds."""

    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float = float("nan")
    spread: float = float("nan")
    source: str = "unknown"

    def to_record(self) -> dict:
        return asdict(self)


def bars_to_frame(bars: Iterable[Bar]) -> pd.DataFrame:
    df = pd.DataFrame([b.to_record() for b in bars], columns=CANONICAL_COLUMNS)
    return to_canonical(df, source=None)


def to_canonical(
    raw: pd.DataFrame,
    source: str | None = "unknown",
    assume_timezone: str | None = None,
    column_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Convert a raw OHLCV frame into the canonical representation.

    Parameters
    ----------
    raw:
        Input frame.  Column names are matched case-insensitively.
    source:
        Data source label.  If ``None`` an existing ``source`` column is kept.
    assume_timezone:
        IANA timezone used ONLY when timestamps are naive (e.g. broker server
        time "Europe/Athens" or "Etc/GMT-2").  Naive timestamps without an
        explicit timezone are rejected - guessing would be a silent repair.
    column_map:
        Optional explicit renaming applied before alias matching.
    """
    df = raw.copy()
    if column_map:
        df = df.rename(columns=dict(column_map))
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in _DEFAULT_ALIASES.items() if k in df.columns and v not in df.columns})

    missing = [c for c in ["timestamp", *PRICE_COLUMNS] if c not in df.columns]
    if missing:
        raise DataIntegrityError(f"missing required columns: {missing}")

    ts = pd.to_datetime(df["timestamp"], utc=False)
    if getattr(ts.dt, "tz", None) is None:
        if assume_timezone is None:
            raise DataIntegrityError(
                "timestamps are timezone-naive; pass assume_timezone explicitly (e.g. broker server tz). "
                "The engine never guesses a timezone."
            )
        ts = ts.dt.tz_localize(assume_timezone, ambiguous="raise", nonexistent="raise")
        logger.info("localised naive timestamps as %s and converted to UTC", assume_timezone)
    ts = ts.dt.tz_convert("UTC")
    # normalise resolution so results are identical regardless of input precision
    df["timestamp"] = ts.astype("datetime64[ns, UTC]")

    for c in PRICE_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in ("volume", "spread"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
        else:
            df[c] = np.nan
    if source is not None or "source" not in df.columns:
        df["source"] = source if source is not None else "unknown"
    df["source"] = df["source"].astype(str)

    df = df[CANONICAL_COLUMNS].reset_index(drop=True)
    df.attrs["columns_present"] = {c: c in [str(x).strip().lower() for x in raw.columns] for c in ("volume", "spread")}
    return df


def load_csv(
    path: str | Path,
    source: str | None = None,
    assume_timezone: str | None = None,
    column_map: Mapping[str, str] | None = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Load a CSV of H4 bars into canonical form (no sorting or repair)."""
    path = Path(path)
    raw = pd.read_csv(path, **read_csv_kwargs)
    return to_canonical(raw, source=source or path.name, assume_timezone=assume_timezone, column_map=column_map)


def exclude_unclosed_bars(df: pd.DataFrame, as_of: pd.Timestamp, timeframe_minutes: int = 240) -> pd.DataFrame:
    """Drop bars whose close_time is after ``as_of`` (i.e. still forming).

    Used for live operation so an unfinished H4 candle is never evaluated as
    though it were closed.  The number of excluded bars is logged.
    """
    as_of = pd.Timestamp(as_of)
    if as_of.tzinfo is None:
        raise DataIntegrityError("as_of must be timezone-aware")
    close_time = df["timestamp"] + pd.Timedelta(minutes=timeframe_minutes)
    keep = close_time <= as_of.tz_convert("UTC")
    dropped = int((~keep).sum())
    if dropped:
        logger.warning("excluded %d unclosed bar(s) with close_time > %s", dropped, as_of)
    return df.loc[keep].reset_index(drop=True)
