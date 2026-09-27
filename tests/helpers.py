"""Test helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbpjpy_engine.data.model import to_canonical
from gbpjpy_engine.synthetic import h4_weekday_grid


def make_bars(closes, highs=None, lows=None, opens=None, start="2024-01-01 00:00", volume=True, spread=True):
    """Build canonical bars from explicit arrays (weekday H4 grid)."""
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.asarray(opens, dtype=float) if opens is not None else np.concatenate([[closes[0]], closes[:-1]])
    highs = np.asarray(highs, dtype=float) if highs is not None else np.maximum(opens, closes) + 0.05
    lows = np.asarray(lows, dtype=float) if lows is not None else np.minimum(opens, closes) - 0.05
    raw = {"timestamp": h4_weekday_grid(n, start), "open": opens, "high": highs, "low": lows, "close": closes}
    if volume:
        raw["volume"] = np.full(n, 1000.0)
    if spread:
        raw["spread"] = np.full(n, 2.0)
    return to_canonical(pd.DataFrame(raw), source="test")
