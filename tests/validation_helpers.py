"""Helpers for Phase 1H tests.  Every dataset built here is SYNTHETIC and used only to test the machinery."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbpjpy_engine.validation import DatasetProvenance, ImportSpec

T0 = pd.Timestamp("2024-01-08 00:00", tz="UTC")  # Monday


def bars(rows, start=T0, freq="1h", spread=2.0) -> pd.DataFrame:
    """rows = [(o, h, l, c), ...] -> canonical frame (BID prices, spread in pips)."""
    ts = pd.date_range(start, periods=len(rows), freq=freq)
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "timestamp", ts)
    df["volume"] = 100.0
    df["spread"] = spread
    df["source"] = "test"
    df["timestamp"] = df["timestamp"].astype("datetime64[ns, UTC]")
    return df


def flat_bars(n, price=190.0, start=T0, rng=0.05, spread=2.0) -> pd.DataFrame:
    return bars([(price, price + rng, price - rng, price)] * n, start, spread=spread)


def proposal(direction="LONG", ts=T0, entry=190.02, stop=189.52, target=191.02, pid="P-1", family="TREND_PULLBACK_CONTINUATION"):
    one = abs(entry - stop) / 0.01
    return {"trade_proposal_id": pid, "timestamp": pd.Timestamp(ts).isoformat(), "direction": direction, "setup_family": family,
            "executable_reference_price": entry, "proposed_stop_price": stop, "primary_target": {"price": target},
            "risk_unit": {"one_R_pips": round(one, 2)}, "volatility": {"h1_atr_pips": 30.0}}


def prov(**kw) -> DatasetProvenance:
    base = dict(provider="test_vendor", symbol="GBPJPY", timeframe="H1", source_timezone="UTC", price_type="BID",
                volume_type="TICK_VOLUME", spread_availability="PER_BAR", spread_unit="PIPS", retrieved_at="2024-06-01T00:00:00+00:00")
    base.update(kw)
    return DatasetProvenance(**base)


def spec(**kw) -> ImportSpec:
    fmt = kw.pop("fmt", "CSV")
    return ImportSpec(prov(**kw), fmt=fmt)


def random_walk_h1(n=600, seed=3, start=T0, spread=2.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = []
    t = start
    while len(ts) < n:
        if t.dayofweek < 5:
            ts.append(t)
        t += pd.Timedelta(hours=1)
    c = 190 + np.cumsum(rng.normal(0, 0.08, n))
    o = np.concatenate([[190.0], c[:-1]])
    h = np.maximum(o, c) + rng.uniform(0, 0.05, n)
    lo = np.minimum(o, c) - rng.uniform(0, 0.05, n)
    df = pd.DataFrame({"timestamp": pd.DatetimeIndex(ts).astype("datetime64[ns, UTC]"), "open": o, "high": h, "low": lo,
                       "close": c, "volume": 100.0, "spread": spread, "source": "test"})
    return df
