"""Broker-neutral trade-construction interfaces and management-policy
architecture (Phase 1E).

Nothing here connects to a broker or places anything.  Future MT4/MT5
adapters implement ``BrokerConstraintSource`` / ``TransactionCostModel``
outside the strategy core.  When no adapter exists the constraint status is
UNKNOWN and unknown costs are represented explicitly (never as zero).

Management policies (partial exits, break-even, trailing, time exits) are
REPRESENTATIONS for later research.  They are all disabled by default; none is
assumed to improve expectancy and none is optimised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Broker stop-level constraints
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BrokerStopConstraints:
    status: str  # "KNOWN" | "UNKNOWN"
    min_stop_distance_points: float | None = None  # broker "stops level"
    freeze_level_points: float | None = None
    source: str = "none"


UNKNOWN_CONSTRAINTS = BrokerStopConstraints("UNKNOWN", None, None, "no broker adapter")


@runtime_checkable
class BrokerConstraintSource(Protocol):
    def stop_constraints(self, symbol: str, time: pd.Timestamp) -> BrokerStopConstraints:
        ...


# ---------------------------------------------------------------------------
# Transaction costs (commission etc.) - slippage uses the Phase 1D SlippageModel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CostEstimate:
    commission_pips_round_turn: float | None  # pip-equivalent (account-independent); None = unknown
    status: str  # "KNOWN" | "UNKNOWN"
    source: str = "none"


@runtime_checkable
class TransactionCostModel(Protocol):
    def estimate(self, symbol: str, time: pd.Timestamp, direction: str) -> CostEstimate:
        ...


class UnknownCosts:
    def estimate(self, symbol, time, direction) -> CostEstimate:
        return CostEstimate(None, "UNKNOWN", "no cost model")


@dataclass(frozen=True)
class FixedCommission:
    pips_round_turn: float

    def estimate(self, symbol, time, direction) -> CostEstimate:
        return CostEstimate(float(self.pips_round_turn), "KNOWN", "researcher-supplied")


# ---------------------------------------------------------------------------
# Exit / management architecture (representations only; disabled by default)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExitLeg:
    target_label: str  # "T1", "T2", ... or "RUNNER"
    fraction: float


@dataclass(frozen=True)
class ExitPlan:
    kind: str  # SINGLE_TARGET | MULTI_TARGET
    legs: tuple = ()
    note: str = "research representation; fractions are NOT optimised and partial exits are not assumed to help"

    def validate(self) -> None:
        tot = sum(leg.fraction for leg in self.legs)
        if not self.legs or abs(tot - 1.0) > 1e-9 or any(leg.fraction <= 0 for leg in self.legs):
            raise ValueError("exit plan fractions must be positive and sum to 1")

    @staticmethod
    def single(primary_label: str) -> "ExitPlan":
        p = ExitPlan("SINGLE_TARGET", (ExitLeg(primary_label, 1.0),))
        p.validate()
        return p

    @staticmethod
    def split(labels_and_fractions) -> "ExitPlan":
        p = ExitPlan("MULTI_TARGET", tuple(ExitLeg(lbl, float(f)) for lbl, f in labels_and_fractions))
        p.validate()
        return p

    def to_dict(self) -> dict:
        return {"kind": self.kind, "legs": [{"target": leg.target_label, "fraction": leg.fraction} for leg in self.legs],
                "note": self.note}


@dataclass(frozen=True)
class ManagementContext:
    """What a future trade manager would know at a completed bar (supplied later; unused in Phase 1E)."""

    bars_in_trade: int
    r_progress: float  # open result in R at the bar close
    target_reached: tuple = ()
    new_structure: bool = False
    session: str | None = None


@dataclass(frozen=True)
class ManagementAction:
    action: str  # "NONE" | "MOVE_STOP" | "EXIT"
    new_stop_price: float | None = None
    reason: str = ""


@runtime_checkable
class BreakEvenPolicy(Protocol):
    enabled: bool

    def evaluate(self, ctx: ManagementContext) -> ManagementAction:
        ...


@runtime_checkable
class TrailingPolicy(Protocol):
    enabled: bool

    def evaluate(self, ctx: ManagementContext) -> ManagementAction:
        ...


@runtime_checkable
class TimeExitPolicy(Protocol):
    enabled: bool

    def evaluate(self, ctx: ManagementContext) -> ManagementAction:
        ...


@dataclass(frozen=True)
class DisabledPolicy:
    """Default for break-even, trailing and time exits: does nothing."""

    kind: str
    enabled: bool = False

    def evaluate(self, ctx: ManagementContext) -> ManagementAction:
        return ManagementAction("NONE", None, f"{self.kind} disabled")


@dataclass(frozen=True)
class BreakEvenSpec:
    trigger: str  # R_ACHIEVED | STRUCTURAL_PROGRESS | TARGET_REACHED | NEW_H1_STRUCTURE
    trigger_r: float | None = None
    enabled: bool = False


@dataclass(frozen=True)
class TrailingSpec:
    method: str  # STRUCTURE | ATR | SWING | FIXED_R
    params: dict = field(default_factory=dict)
    enabled: bool = False


@dataclass(frozen=True)
class TimeExitSpec:
    max_bars: int | None = None
    no_progress_bars: int | None = None
    exit_on_session_transition: str | None = None
    enabled: bool = False


BREAK_EVEN_TRIGGERS = ("R_ACHIEVED", "STRUCTURAL_PROGRESS", "TARGET_REACHED", "NEW_H1_STRUCTURE")
TRAILING_METHODS = ("STRUCTURE", "ATR", "SWING", "FIXED_R")


def default_management(cfg, primary_label: str | None) -> dict:
    plan = ExitPlan.single(primary_label) if primary_label else None
    return {
        "exit_plan": plan.to_dict() if plan else None,
        "break_even": {"enabled": cfg.break_even_enabled, "available_triggers": list(BREAK_EVEN_TRIGGERS)},
        "trailing": {"enabled": cfg.trailing_enabled, "available_methods": list(TRAILING_METHODS)},
        "time_exit": {"enabled": cfg.time_exit_enabled,
                      "available": ["max_bars", "no_progress_bars", "exit_on_session_transition"]},
        "note": "architecture only - no management rule is active or assumed to add edge",
    }


__all__ = ["BREAK_EVEN_TRIGGERS", "BreakEvenPolicy", "BreakEvenSpec", "BrokerConstraintSource", "BrokerStopConstraints",
           "CostEstimate", "DisabledPolicy", "ExitLeg", "ExitPlan", "FixedCommission", "ManagementAction",
           "ManagementContext", "TRAILING_METHODS", "TimeExitPolicy", "TimeExitSpec", "TrailingPolicy", "TrailingSpec",
           "TransactionCostModel", "UNKNOWN_CONSTRAINTS", "UnknownCosts", "default_management"]
