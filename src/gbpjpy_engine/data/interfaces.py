"""Broker-neutral data boundary.

Target architecture::

            GBPJPY Strategy Core  (this package - platform agnostic)
                      |
          Broker-neutral interfaces (this module)
               /                     \\
        MT4 adapter             MT5 adapter        <- NOT implemented (future phases)

Rules for every future adapter (MT4, MT5, file replay, other brokers):

1. Deliver only CLOSED H4 bars in the canonical frame (``data.model``):
   ``timestamp`` = bar OPEN time, timezone-aware, converted to UTC.
2. Broker server timestamps (often EET/EEST, i.e. UTC+2/+3 with DST) must be
   converted to UTC by the adapter using the broker's real server timezone
   (``to_canonical(..., assume_timezone=...)``).  The core never guesses a
   timezone and never silently corrects one; DST grid shifts that survive
   conversion are flagged (``TIMEZONE_ALIGNMENT_SHIFT``), not repaired.
3. ``volume`` is BROKER TICK VOLUME (count of price updates on that broker's
   feed).  Spot FX has no centralised exchange, so it is NOT traded volume
   and is not comparable across brokers.  ``spread`` is whatever the source
   reports (units documented by the adapter).  Both are carried through for
   later execution / market-quality modules; Phase 1A features do not use them.
4. Adapters contain all platform-specific code.  Nothing in the strategy core
   may import a platform SDK.

Only a structural interface is defined here - there is no implementation, no
connection code and no order functionality.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from .model import CANONICAL_COLUMNS, DataIntegrityError


@runtime_checkable
class BarSource(Protocol):
    """Anything that can supply closed canonical GBPJPY H4 bars."""

    def closed_bars(self, as_of: pd.Timestamp) -> pd.DataFrame:
        """Return canonical bars whose close time is <= ``as_of`` (tz-aware)."""
        ...


def assert_canonical(df: pd.DataFrame) -> None:
    """Raise DataIntegrityError unless ``df`` is in canonical form (columns + tz-aware UTC)."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise DataIntegrityError(f"bars are not canonical; missing {missing} (use data.to_canonical)")
    tz = getattr(df["timestamp"].dt, "tz", None) if len(df) else getattr(df["timestamp"].dtype, "tz", None)
    if tz is None:
        raise DataIntegrityError("canonical timestamps must be timezone-aware (UTC)")
