"""Helpers for Phase 1B context tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from gbpjpy_engine.context.hierarchy import LayerResult
from gbpjpy_engine.features.structure import Swing

T0 = pd.Timestamp("2024-01-01", tz="UTC")


def swing(sid, kind, idx, price, label=None, atr=1.0, sig=1.5):
    t = T0 + pd.Timedelta(hours=4 * idx)
    return Swing(swing_id=sid, kind=kind, pivot_index=idx, occurred_at=t, price=float(price), confirm_index=idx + 3,
                 confirmed_at=t + pd.Timedelta(hours=16), atr_at_confirmation=atr, label=label, significance_atr=sig,
                 accepted_index=idx + 3)


def zigzag(prices, start_kind="low", step=6, atr=1.0):
    """Alternating swings with labels computed like the structure engine."""
    out, kind = [], start_kind
    for k, p in enumerate(prices):
        prev = next((s for s in reversed(out) if s.kind == kind), None)
        if prev is None:
            lab = None
        elif kind == "high":
            lab = "HH" if p > prev.price + 0.1 else ("LH" if p < prev.price - 0.1 else "EH")
        else:
            lab = "HL" if p > prev.price + 0.1 else ("LL" if p < prev.price - 0.1 else "EL")
        out.append(swing(k, kind, k * step, p, lab, atr))
        kind = "high" if kind == "low" else "low"
    return out


def layer(cls="NEUTRAL", conf=0.0, broken_from=None):
    r = LayerResult(layer="x", classification=cls, confidence=conf)
    r.broken_from = broken_from
    return r


def zone(zid, lower, upper, strength=70.0, zone_type="resistance", age=50, distance=0.0):
    return SimpleNamespace(zone_id=zid, lower=lower, upper=upper, strength=strength, zone_type=zone_type,
                           age_bars=age, distance=distance, formation_atr=1.0, midpoint=(lower + upper) / 2)


def price_arrays(o, h, l, c, atr=1.0):
    n = len(c)
    return {"open": np.asarray(o, float), "high": np.asarray(h, float), "low": np.asarray(l, float),
            "close": np.asarray(c, float), "atr": np.full(n, atr), "ts": [T0 + pd.Timedelta(hours=4 * i) for i in range(n)],
            "tf": pd.Timedelta(hours=4)}
