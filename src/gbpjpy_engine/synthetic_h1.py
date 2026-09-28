"""Deterministic synthetic GBPJPY-like H1 datasets for ENGINEERING tests of
Phase 1C.  H4 bars are derived from the same H1 path with
``data.resample.resample_complete`` (complete buckets only), so both
timeframes describe one consistent market.  Not market simulations; not
evidence of any edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.resample import resample_complete
from .synthetic import _bars_from_path, _compose

H1_SCENARIOS = ("h1_uptrend_pullbacks", "h1_downtrend_pullbacks", "h1_range")


@dataclass
class H1Scenario:
    name: str
    h1: pd.DataFrame
    h4: pd.DataFrame
    segment_start_h1: int
    events: dict = field(default_factory=dict)


def h1_weekday_grid(n: int, start: str = "2023-01-02 00:00") -> pd.DatetimeIndex:
    t = pd.date_range(start, periods=int(n * 1.5) + 200, freq="1h", tz="UTC")
    return t[t.dayofweek < 5][:n]


def generate_h1_scenario(name: str, seed: int = 5, lead_in: int = 1200, length: int = 720,
                         start_price: float = 185.0) -> H1Scenario:
    if name not in H1_SCENARIOS:
        raise KeyError(f"unknown H1 scenario {name!r}; choose from {H1_SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = lead_in + length
    drift = np.zeros(n)
    amp = np.full(n, 0.9)
    period = np.full(n, 60.0)  # lead_in is a whole number of periods -> no phase jump
    noise = np.full(n, 0.05)
    wick = np.full(n, 0.05)
    seg = slice(lead_in, n)
    if name == "h1_uptrend_pullbacks":
        drift[seg], amp[seg], period[seg] = 0.03, 0.8, 48.0
    elif name == "h1_downtrend_pullbacks":
        drift[seg], amp[seg], period[seg] = -0.03, 0.8, 48.0
    else:
        amp[seg], period[seg], noise[seg] = 0.9, 40.0, 0.08
    path = _compose(rng, start_price, drift, amp, period, noise)
    times = h1_weekday_grid(len(path))
    h1 = _bars_from_path(rng, path, wick, times, name)
    h4, _ = resample_complete(h1, 60, 240)
    return H1Scenario(name=name, h1=h1, h4=h4, segment_start_h1=lead_in)
