"""Explicit research cost models (Phase 1H): spread, commission, slippage, swap.

Every model carries a STATUS so every result states its cost assumptions:

* spread: HISTORICAL (timestamp-appropriate, from the data) / ASSUMED (declared
  research assumption because the data has no spread) / ZERO_DIAGNOSTIC.
  A missing historical spread is never silently read as zero.
* commission: KNOWN / ZERO_DECLARED (genuinely zero, with a stated source) /
  UNKNOWN / SCENARIO.
* slippage: ZERO_SLIPPAGE_DIAGNOSTIC_ONLY / FIXED / SPREAD_DEPENDENT /
  VOLATILITY_DEPENDENT / EMPIRICAL (resampled from supplied observations,
  deterministic per trade and seed).  Slippage is always ADVERSE or zero.
* swap: UNKNOWN / KNOWN / SCENARIO with long/short rates per night, a triple
  rollover weekday and the rollover hour.

Scenario realism: DIAGNOSTIC (upper bound only - never presented as realistic),
BASELINE, STRESSED.  Values here are research assumptions, NOT VALIDATED EDGE.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd

SPREAD_KINDS = ("HISTORICAL", "ASSUMED", "ZERO_DIAGNOSTIC")
COMMISSION_KINDS = ("KNOWN", "ZERO_DECLARED", "UNKNOWN", "SCENARIO")
SLIPPAGE_KINDS = ("ZERO_SLIPPAGE_DIAGNOSTIC_ONLY", "FIXED", "SPREAD_DEPENDENT", "VOLATILITY_DEPENDENT", "EMPIRICAL")
SWAP_KINDS = ("UNKNOWN", "KNOWN", "SCENARIO")
REALISM = ("DIAGNOSTIC", "BASELINE", "STRESSED")


class MissingSpread(ValueError):
    """Historical spread requested but the bar has none (never replaced by zero)."""


@dataclass(frozen=True)
class SpreadModel:
    kind: str = "HISTORICAL"
    assumed_pips: float | None = None  # ASSUMED value, or the explicit fallback for missing HISTORICAL bars
    multiplier: float = 1.0
    add_pips: float = 0.0

    def spread_pips(self, bar_spread_pips) -> tuple[float, str]:
        if self.kind == "ZERO_DIAGNOSTIC":
            return 0.0, "ZERO_DIAGNOSTIC"
        if self.kind == "ASSUMED":
            if self.assumed_pips is None:
                raise ValueError("ASSUMED spread needs an explicit value")
            return self.assumed_pips * self.multiplier + self.add_pips, "ASSUMED"
        v = None if bar_spread_pips is None else float(bar_spread_pips)
        if v is not None and np.isfinite(v):
            return v * self.multiplier + self.add_pips, "HISTORICAL"
        if self.assumed_pips is not None:
            return self.assumed_pips * self.multiplier + self.add_pips, "ASSUMED_FALLBACK"
        raise MissingSpread("bar has no spread and no explicit fallback assumption was declared")


@dataclass(frozen=True)
class CommissionModel:
    kind: str = "UNKNOWN"
    pips_round_turn: float | None = None  # expressed in pips of price so it converts directly to R
    source: str = ""

    def pips(self) -> tuple[float | None, str]:
        if self.kind == "ZERO_DECLARED":
            if not self.source:
                raise ValueError("zero commission must cite its source")
            return 0.0, "ZERO_DECLARED"
        if self.kind in ("KNOWN", "SCENARIO"):
            if self.pips_round_turn is None:
                raise ValueError(f"{self.kind} commission needs a value")
            return float(self.pips_round_turn), self.kind
        return None, "UNKNOWN"


@dataclass(frozen=True)
class SlippageModel:
    kind: str = "FIXED"
    pips: float = 0.5  # FIXED per side
    spread_fraction: float = 0.25  # SPREAD_DEPENDENT
    atr_fraction: float = 0.02  # VOLATILITY_DEPENDENT
    samples_pips: tuple = ()  # EMPIRICAL adverse slippage observations (>= 0)
    stop_extra_pips: float = 0.0  # additional adverse slippage on stop exits
    seed: int = 0

    def side_pips(self, spread_pips: float, atr_pips: float, key: str) -> float:
        k = self.kind
        if k == "ZERO_SLIPPAGE_DIAGNOSTIC_ONLY":
            return 0.0
        if k == "FIXED":
            v = self.pips
        elif k == "SPREAD_DEPENDENT":
            v = self.spread_fraction * spread_pips
        elif k == "VOLATILITY_DEPENDENT":
            v = self.atr_fraction * atr_pips
        elif k == "EMPIRICAL":
            if not self.samples_pips:
                raise ValueError("EMPIRICAL slippage needs observations")
            h = int(hashlib.sha256(f"{self.seed}|{key}".encode()).hexdigest()[:12], 16)
            v = float(sorted(self.samples_pips)[h % len(self.samples_pips)])
        else:
            raise ValueError(k)
        return max(0.0, float(v))  # never favourable in research costs


@dataclass(frozen=True)
class SwapModel:
    kind: str = "UNKNOWN"
    long_pips_per_night: float | None = None  # positive = credit, negative = cost (in pips of price)
    short_pips_per_night: float | None = None
    triple_weekday: int = 2  # Wednesday rollover charges three nights (typical FX convention; declare per provider)
    rollover_hour_utc: int = 21

    def nights(self, entry: pd.Timestamp, exit_: pd.Timestamp) -> tuple[int, int]:
        """(rollovers crossed, charged nights incl. triple) between entry and exit."""
        if exit_ <= entry:
            return 0, 0
        first = entry.normalize() + pd.Timedelta(hours=self.rollover_hour_utc)
        if first <= entry:
            first += pd.Timedelta(days=1)
        rolls = pd.date_range(first, exit_, freq="D")
        rolls = rolls[rolls.dayofweek < 5]
        charged = sum(3 if r.dayofweek == self.triple_weekday else 1 for r in rolls)
        return len(rolls), charged

    def pips(self, direction: int, entry: pd.Timestamp, exit_: pd.Timestamp) -> tuple[float | None, str, int]:
        _, charged = self.nights(entry, exit_)
        if self.kind == "UNKNOWN":
            return None, "UNKNOWN", charged
        rate = self.long_pips_per_night if direction > 0 else self.short_pips_per_night
        if rate is None:
            return None, "UNKNOWN", charged
        return float(rate) * charged, self.kind, charged


@dataclass(frozen=True)
class CostScenario:
    name: str
    realism: str
    spread: SpreadModel = field(default_factory=SpreadModel)
    commission: CommissionModel = field(default_factory=CommissionModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    swap: SwapModel = field(default_factory=SwapModel)
    note: str = ""

    def __post_init__(self):
        if self.realism not in REALISM:
            raise ValueError(f"realism must be one of {REALISM}")
        zeroish = (self.spread.kind == "ZERO_DIAGNOSTIC" or self.slippage.kind == "ZERO_SLIPPAGE_DIAGNOSTIC_ONLY")
        if zeroish and self.realism != "DIAGNOSTIC":
            raise ValueError("zero spread / zero slippage is permitted ONLY in a DIAGNOSTIC scenario")

    def describe(self) -> dict:
        d = asdict(self)
        d["presentation"] = ("DIAGNOSTIC UPPER BOUND - not realistic live performance" if self.realism == "DIAGNOSTIC"
                             else self.realism)
        return d

    def stressed(self, name: str, spread_mult=1.5, spread_add=1.0, slip_mult=2.0, stop_extra=2.0,
                 commission_add=0.5) -> "CostScenario":
        c = self.commission
        c2 = replace(c, kind="SCENARIO", pips_round_turn=(c.pips_round_turn or 0.0) + commission_add)
        s = self.slippage
        s2 = replace(s, pips=s.pips * slip_mult, spread_fraction=s.spread_fraction * slip_mult,
                     atr_fraction=s.atr_fraction * slip_mult, stop_extra_pips=s.stop_extra_pips + stop_extra)
        return replace(self, name=name, realism="STRESSED",
                       spread=replace(self.spread, multiplier=self.spread.multiplier * spread_mult,
                                      add_pips=self.spread.add_pips + spread_add),
                       commission=c2, slippage=s2)


def standard_scenarios(spread_availability: str, known_commission_pips: float | None = None,
                       unknown_spread_assumption_pips: float = 3.0, assumed_commission_pips: float = 0.5,
                       assumed_slippage_pips: float = 0.5) -> dict:
    """ZERO_DIAGNOSTIC, BASELINE and STRESSED scenarios for one dataset.

    The default assumptions reuse the values already declared by earlier phases (Phase 1D unknown-spread assumption,
    Phase 1E commission / slippage assumptions) and are labelled as assumptions in every result."""
    if spread_availability in ("PER_BAR", "SAMPLED"):
        spread = SpreadModel("HISTORICAL", assumed_pips=None)
    else:
        spread = SpreadModel("ASSUMED", assumed_pips=unknown_spread_assumption_pips)
    commission = (CommissionModel("KNOWN", known_commission_pips, "provider-declared") if known_commission_pips is not None
                  else CommissionModel("SCENARIO", assumed_commission_pips, "Phase 1E assumption (commission UNKNOWN)"))
    base = CostScenario("BASELINE", "BASELINE", spread, commission, SlippageModel("FIXED", assumed_slippage_pips),
                        SwapModel("UNKNOWN"), note="swap UNKNOWN: excluded and disclosed")
    zero = CostScenario("ZERO_COST_DIAGNOSTIC", "DIAGNOSTIC", SpreadModel("ZERO_DIAGNOSTIC"),
                        CommissionModel("ZERO_DECLARED", 0.0, "diagnostic only"), SlippageModel("ZERO_SLIPPAGE_DIAGNOSTIC_ONLY"),
                        SwapModel("UNKNOWN"), note="upper-bound diagnostic; never realistic")
    stressed = base.stressed("STRESSED")
    stressed = replace(stressed, swap=SwapModel("SCENARIO", -1.0, -1.0), note="adverse swap scenario 1 pip/night both sides")
    return {"ZERO_COST_DIAGNOSTIC": zero, "BASELINE": base, "STRESSED": stressed}
