"""Deterministic synthetic GBPJPY-like H4 datasets for ENGINEERING tests.

These scenarios check that the engine produces logically appropriate
descriptive states.  They are not market simulations and are not evidence of
any trading edge.

Every scenario = a ``lead_in`` of neutral, normal-volatility ranging bars (so
EMA200 and percentile histories are warm) followed by the scenario segment.
Timestamps are on a UTC H4 grid (00,04,...,20) Monday-Friday.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data.model import to_canonical

SCENARIOS = (
    "clean_uptrend",
    "clean_downtrend",
    "sideways_range",
    "volatile_range",
    "low_volatility_compression",
    "trend_reversal",
    "false_breakout",
    "volatility_shock",
)


@dataclass
class SyntheticScenario:
    name: str
    bars: pd.DataFrame
    segment_start: int
    events: dict = field(default_factory=dict)
    description: str = ""


def h4_weekday_grid(n: int, start: str = "2023-01-02 00:00") -> pd.DatetimeIndex:
    """n H4 bar-open timestamps (UTC) skipping Saturdays and Sundays."""
    out = []
    t = pd.Timestamp(start, tz="UTC")
    step = pd.Timedelta(hours=4)
    while len(out) < n:
        if t.dayofweek < 5:
            out.append(t)
        t += step
    return pd.DatetimeIndex(out)


def _ou(rng, sd: np.ndarray, phi: float = 0.7) -> np.ndarray:
    """Stationary AR(1) noise: keeps ranges bounded instead of drifting like a random walk."""
    x = np.zeros(len(sd))
    eps = rng.normal(0.0, 1.0, len(sd)) * sd
    for t in range(1, len(sd)):
        x[t] = phi * x[t - 1] + eps[t]
    return x


def _compose(rng, start_price, drift, amp, period, noise_sd, jumps=None) -> np.ndarray:
    """path[t] = start + cumulative drift (+jumps) + amp*sin(phase) + AR(1) noise, t = 0..N."""
    n = len(drift)
    level = np.concatenate([[0.0], np.cumsum(drift)])
    if jumps:
        for k, j in jumps.items():
            level[k + 1:] += j
    phase = np.concatenate([[0.0], np.cumsum(2 * np.pi / np.asarray(period, dtype=float))])
    amp_full = np.concatenate([[amp[0]], amp])
    wave = amp_full * np.sin(phase)
    noise = _ou(rng, np.concatenate([[0.0], noise_sd]))
    return start_price + level + wave + noise


def _bars_from_path(rng, path: np.ndarray, wick_sd, times: pd.DatetimeIndex, name: str, extra_wicks=None) -> pd.DataFrame:
    n = len(path) - 1
    o = path[:-1]
    c = path[1:]
    wick_sd = np.broadcast_to(np.asarray(wick_sd, dtype=float), (n,))
    up = np.abs(rng.normal(0.0, 1.0, n)) * wick_sd
    dn = np.abs(rng.normal(0.0, 1.0, n)) * wick_sd
    if extra_wicks:
        for i, (u, d) in extra_wicks.items():
            up[i] += u
            dn[i] += d
    h = np.maximum(o, c) + up
    l = np.minimum(o, c) - dn
    vol = rng.integers(2000, 9000, n).astype(float)
    spread = np.round(rng.uniform(1.5, 3.5, n), 1)
    raw = pd.DataFrame(
        {"timestamp": times[:n], "open": o, "high": h, "low": l, "close": c, "volume": vol, "spread": spread}
    )
    return to_canonical(raw, source=f"synthetic:{name}")


def generate_scenario(name: str, seed: int = 7, lead_in: int = 324, length: int = 300, start_price: float = 185.0) -> SyntheticScenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = lead_in + length
    drift = np.zeros(n)
    amp = np.full(n, 1.2)
    period = np.full(n, 36.0)  # lead_in is a whole number of lead-in periods -> no phase jump
    noise = np.full(n, 0.20)
    wick = np.full(n, 0.12)
    seg = slice(lead_in, n)
    events: dict = {}
    jumps: dict = {}
    extra_wicks: dict = {}

    if name in ("clean_uptrend", "clean_downtrend"):
        drift[seg] = 0.10 if name == "clean_uptrend" else -0.10
        amp[seg], period[seg], noise[seg], wick[seg] = 1.0, 24.0, 0.08, 0.07
        desc = "persistent drift with regular shallow pullbacks"
    elif name == "sideways_range":
        amp[seg], period[seg], noise[seg], wick[seg] = 1.0, 20.0, 0.22, 0.12
        desc = "no drift, oscillation plus stationary noise"
    elif name == "volatile_range":
        amp[seg], period[seg], noise[seg], wick[seg] = 2.4, 12.0, 0.80, 0.40
        desc = "no drift, large oscillations and noise (volatility well above lead-in)"
    elif name == "low_volatility_compression":
        amp[seg] = np.linspace(0.8, 0.06, length)
        period[seg] = 20.0
        noise[seg] = np.linspace(0.16, 0.02, length)
        wick[seg] = np.linspace(0.10, 0.012, length)
        desc = "no drift, oscillation and noise shrinking steadily"
    elif name == "trend_reversal":
        half = length // 2
        drift[lead_in:lead_in + half] = 0.10
        drift[lead_in + half:] = -0.11
        amp[seg], period[seg], noise[seg], wick[seg] = 1.0, 24.0, 0.08, 0.07
        events["reversal_index"] = lead_in + half
        desc = "clean uptrend followed by clean downtrend"
    elif name == "false_breakout":
        amp[seg], period[seg], noise[seg], wick[seg] = 1.2, 24.0, 0.05, 0.06
        events["break_index"] = lead_in + 24 * 8 + 6  # at a wave crest
        desc = "range with one close well above the range high, immediately followed by a close back inside"
    elif name == "volatility_shock":
        amp[seg], period[seg], noise[seg], wick[seg] = 1.0, 24.0, 0.12, 0.10
        k = lead_in + 200
        jumps[k] = -3.2
        extra_wicks[k] = (0.1, 0.9)
        events["shock_index"] = k
        desc = "normal range then a single abnormally large bearish displacement bar"

    path = _compose(rng, start_price, drift, amp, period, noise, jumps)
    if name == "false_breakout":
        k = events["break_index"]
        crest = start_price + 1.2  # wave crest level of the range
        path[k + 1] = crest + 1.0  # close of bar k: well beyond the range high
        path[k + 2] = crest - 0.6  # close of bar k+1: back inside
    times = h4_weekday_grid(len(path))
    bars = _bars_from_path(rng, path, wick, times, name, extra_wicks)
    return SyntheticScenario(name=name, bars=bars, segment_start=lead_in, events=events, description=desc)


def random_walk_bars(n: int = 900, seed: int = 11, start_price: float = 185.0) -> pd.DataFrame:
    """Generic noisy random-walk bars (used by the anti-look-ahead tests)."""
    rng = np.random.default_rng(seed)
    regime_drift = np.repeat(rng.normal(0.0, 0.06, n // 60 + 1), 60)[:n]
    sd = np.repeat(rng.uniform(0.15, 0.5, n // 80 + 1), 80)[:n]
    inc = regime_drift + rng.normal(0.0, 1.0, n) * sd
    path = start_price + np.concatenate([[0.0], np.cumsum(inc)])
    return _bars_from_path(rng, path, sd * 0.5, h4_weekday_grid(n + 1), "random_walk")
