"""Deterministic synthetic GBPJPY-like H1 datasets for ENGINEERING tests of
Phase 1D (entry intelligence).

The Phase 1C scenarios (``synthetic_h1``) have an H1 ATR of ~15 pips, below
typical GBPJPY H1 volatility, so their 1.5-3.5 pip synthetic spreads are
10-23 % of ATR and the entry engine (correctly) classifies them as HIGH.  The
Phase 1D scenarios use the same generator at a volatility closer to real
GBPJPY H1 behaviour (ATR ~25-30 pips) and are otherwise built the same way:
H4 bars are resampled from the same H1 path (complete buckets only).

Not market simulations and not evidence of any edge.  The Phase 1C scenarios
are left untouched.
"""

from __future__ import annotations

import numpy as np

from .data.resample import resample_complete
from .synthetic import _bars_from_path, _compose
from .synthetic_h1 import H1Scenario, h1_weekday_grid

ENTRY_SCENARIOS = ("entry_uptrend", "entry_downtrend")


def generate_entry_scenario(name: str, seed: int = 5, lead_in: int = 1200, length: int = 720, vol_scale: float = 1.8,
                            start_price: float = 185.0) -> H1Scenario:
    if name not in ENTRY_SCENARIOS:
        raise KeyError(f"unknown entry scenario {name!r}; choose from {ENTRY_SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = lead_in + length
    k = vol_scale
    drift = np.zeros(n)
    amp = np.full(n, 0.9 * k)
    period = np.full(n, 60.0)
    noise = np.full(n, 0.05 * k)
    wick = np.full(n, 0.05 * k)
    seg = slice(lead_in, n)
    drift[seg] = (0.03 if name == "entry_uptrend" else -0.03) * k
    amp[seg], period[seg] = 0.8 * k, 48.0
    path = _compose(rng, start_price, drift, amp, period, noise)
    h1 = _bars_from_path(rng, path, wick, h1_weekday_grid(len(path)), name)
    h4, _ = resample_complete(h1, 60, 240)
    return H1Scenario(name=name, h1=h1, h4=h4, segment_start_h1=lead_in)
