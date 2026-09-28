"""Aggregate canonical bars to a higher timeframe (e.g. H1 -> H4).

Only COMPLETE buckets are emitted: a bucket needs every constituent lower-
timeframe bar (e.g. four H1 bars for one H4 bar).  Incomplete buckets are
dropped and reported - never filled or repaired - so a partially observed
H4 candle can never masquerade as a completed one.  Bucket boundaries are
anchored on the UTC grid with an optional ``offset`` (brokers whose H4 grid
opens at e.g. 21:00 UTC use offset 1h).  Volume is summed; spread is the
mean of available values.
"""

from __future__ import annotations

import logging

import pandas as pd

from .model import CANONICAL_COLUMNS, DataIntegrityError

logger = logging.getLogger(__name__)


def resample_complete(
    bars: pd.DataFrame,
    from_minutes: int = 60,
    to_minutes: int = 240,
    offset: pd.Timedelta | str | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Return (aggregated canonical bars, list of dropped incomplete bucket opens)."""
    if to_minutes % from_minutes:
        raise DataIntegrityError("target timeframe must be a multiple of the source timeframe")
    if len(bars) == 0:
        return bars.iloc[0:0][CANONICAL_COLUMNS].copy(), []
    need = to_minutes // from_minutes
    ts = bars["timestamp"]
    off = pd.Timedelta(offset) if offset is not None else pd.Timedelta(0)
    bucket = (ts - off).dt.floor(f"{to_minutes}min") + off
    g = bars.assign(_b=bucket.to_numpy()).groupby("_b", sort=True)
    agg = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", lambda v: v.sum(min_count=1)), spread=("spread", "mean"),
                source=("source", "first"), n=("close", "size"))
    complete = agg["n"] == need
    dropped = list(agg.index[~complete])
    if dropped:
        logger.info("dropped %d incomplete %d-minute bucket(s)", len(dropped), to_minutes)
    out = agg.loc[complete].drop(columns="n").reset_index().rename(columns={"_b": "timestamp"})
    out["timestamp"] = out["timestamp"].astype("datetime64[ns, UTC]")
    out["source"] = out["source"].astype(str) + f":agg{to_minutes}"
    out["volume"] = out["volume"].astype(float)
    out["spread"] = out["spread"].astype(float)
    out = out[CANONICAL_COLUMNS]
    out.attrs["columns_present"] = bars.attrs.get("columns_present", {"volume": True, "spread": True})
    return out, dropped


def h1_to_h4(h1: pd.DataFrame, offset=None) -> pd.DataFrame:
    return resample_complete(h1, 60, 240, offset)[0]


__all__ = ["resample_complete", "h1_to_h4"]
