"""Post-construction outcome storage (MAE / MFE in R) - research only.

This module is NEVER used by trade construction (tested: the engine and
construction modules do not import it).  It evaluates a proposal against bars
that START AFTER the proposal time, so excursions can be measured for later
validation without ever influencing how a historical proposal was built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeOutcomeRecord:
    trade_proposal_id: str
    bars_evaluated: int
    mae_R: float | None  # maximum adverse excursion, in R (>= 0)
    mfe_R: float | None  # maximum favourable excursion, in R (>= 0)
    note: str = "computed AFTER construction from later bars; never an input to proposal generation"

    def to_dict(self) -> dict:
        return asdict(self)


def compute_excursions(proposal: dict, later_bars: pd.DataFrame) -> TradeOutcomeRecord:
    t0 = pd.Timestamp(proposal["timestamp"])
    if len(later_bars) and pd.Timestamp(later_bars["timestamp"].iloc[0]) < t0:
        raise ValueError("outcome bars must start at/after the proposal time (no mixing with construction data)")
    d = 1 if proposal["direction"] == "LONG" else -1
    entry = proposal["executable_reference_price"]
    one_r = proposal["risk_unit"]["one_R_price"]
    if not len(later_bars) or one_r <= 0:
        return TradeOutcomeRecord(proposal["trade_proposal_id"], 0, None, None)
    hi, lo = later_bars["high"].to_numpy(float), later_bars["low"].to_numpy(float)
    fav = (np.max(hi) - entry) if d > 0 else (entry - np.min(lo))
    adv = (entry - np.min(lo)) if d > 0 else (np.max(hi) - entry)
    return TradeOutcomeRecord(proposal["trade_proposal_id"], len(later_bars), round(max(adv, 0.0) / one_r, 4),
                              round(max(fav, 0.0) / one_r, 4))
