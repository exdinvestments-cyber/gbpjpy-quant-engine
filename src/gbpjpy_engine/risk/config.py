"""Phase 1F account-risk policy - one central place for every value.

EVERY VALUE HERE IS CONFIGURATION, NOT VALIDATED EDGE.  The baselines are
deliberately conservative engineering defaults; none was chosen from
(synthetic) profitability and none is claimed to be optimal.

There is intentionally no field for any profit objective (annual, monthly or
daily targets, amounts to recover, account-size goals): unknown keys are
rejected by the loader, and a source test forbids such inputs in the risk
package.  Quality scores are not risk inputs either.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..config import _p

NOTE = " [CONFIGURATION - NOT VALIDATED EDGE]"
BASES = ("BALANCE", "EQUITY", "LOWER_OF_BALANCE_OR_EQUITY")
WEEKEND = ("ALLOW", "DISALLOW", "CLOSE_BEFORE_WEEKEND", "UNKNOWN_EXTERNAL_POLICY")
NEWS = ("ALLOW", "BLOCK_BEFORE", "BLOCK_AFTER", "REDUCE_RISK", "UNKNOWN")


def _d(v, doc):
    return _p(v, doc + NOTE)


@dataclass(frozen=True)
class SizingPolicy:
    base_risk_percent: float = _d(0.5, "Risk per trade as % of the sizing capital (baseline).")
    max_single_trade_risk_percent: float = _d(1.0, "HARD ceiling per trade (% of sizing capital). Nothing can exceed it.")
    basis: str = _d("LOWER_OF_BALANCE_OR_EQUITY", "Sizing capital: BALANCE, EQUITY or LOWER_OF_BALANCE_OR_EQUITY "
                                                 "(default: the lower of the two - floating profit is never sized on, "
                                                 "floating loss always is).")
    stop_slippage_allowance_pips: float = _d(1.0, "Risk-side slippage allowance added to the stop distance when no "
                                                  "better slippage data exists (an assumption, not observed slippage).")
    entry_slippage_allowance_pips: float = _d(0.5, "Entry slippage allowance added to the risk distance.")
    assumed_commission_pips_round_turn: float = _d(0.5, "Commission assumption (pip-equivalent) when no commission "
                                                        "model exists - never zero.")
    rounding_tolerance: str = _d("0.00000001", "Absolute numeric tolerance (account currency) for actual <= permitted.")


@dataclass(frozen=True)
class LimitPolicy:
    daily_loss_percent: float = _d(2.0, "Daily loss budget (% of the day's reference capital) incl. open and reserved risk.")
    weekly_loss_percent: float = _d(4.0, "Weekly loss budget (% of the week's reference capital).")
    monthly_loss_percent: float = _d(8.0, "Monthly loss budget (% of the month's reference capital).")
    monthly_enabled: bool = _d(True, "Enable the monthly loss budget.")
    max_aggregate_risk_percent: float = _d(1.5, "Maximum open + reserved + new risk across the account (%).")
    max_symbol_risk_percent: float = _d(1.5, "Maximum GBPJPY open + reserved + new risk (%).")
    max_symbol_positions: int = _d(2, "Maximum simultaneous GBPJPY positions + reservations.")
    max_same_direction_positions: int = _d(1, "Maximum simultaneous GBPJPY positions + reservations in one direction.")
    allow_opposite_direction: bool = _d(False, "Allow a new trade opposite to an open GBPJPY position (hedging).")
    pyramiding_enabled: bool = _d(False, "Adding to an existing same-direction position. OFF: requires separate "
                                         "empirical validation. Adding to a LOSING position is always prohibited.")
    aggregate_policy: str = _d("REDUCE", "When aggregate/symbol caps bind: REDUCE the new exposure to what remains, "
                                         "or REJECT. Never exceed.")


@dataclass(frozen=True)
class ConsecutiveLossPolicy:
    reduce_after: int = _d(3, "After this many consecutive closed losses permitted risk is multiplied by reduce_factor "
                              "(0 disables). Never an increase.")
    reduce_factor: float = _d(0.5, "Multiplier (<= 1) applied after reduce_after consecutive losses.")
    pause_after: int = _d(5, "After this many consecutive closed losses new trades are PAUSED (0 disables).")
    pause_until: str = _d("NEXT_TRADING_DAY", "PAUSE lasts until NEXT_TRADING_DAY or MANUAL_RESET (no psychological "
                                              "timing assumed).")


@dataclass(frozen=True)
class DrawdownPolicy:
    tiers: tuple = _d((("NORMAL", 0.0, 1.0), ("CAUTION", 5.0, 0.75), ("DEFENSIVE", 10.0, 0.5), ("HALT", 15.0, 0.0)),
                      "(state, equity drawdown % from the high-water mark at/above which it applies, risk multiplier <= 1).")
    halt_on_critical: bool = _d(True, "Entering the HALT tier sets the global state to HALTED (manual reset required).")


@dataclass(frozen=True)
class MarginPolicy:
    require_margin_known: bool = _d(True, "Reject when required margin cannot be estimated reliably.")
    min_free_margin_after_percent_of_equity: float = _d(50.0, "Free margin after the trade must stay >= this % of equity.")
    min_post_trade_margin_level_percent: float = _d(300.0, "Margin level (equity / used margin) after the trade must be >= this.")
    margin_policy: str = _d("REJECT", "When margin binds: REJECT, or REDUCE volume to what margin safely allows.")


@dataclass(frozen=True)
class DataPolicy:
    account_max_age_seconds: float = _d(300.0, "Account snapshots older than this are stale (rejected).")
    rate_max_age_seconds: float = _d(7200.0, "Conversion rates older than this are stale (rejected).")
    timezone: str = _d("America/New_York", "Timezone of the trading-day roll (never the machine clock).")
    rollover_hour: int = _d(17, "Local hour at which the trading day rolls.")
    halt_on_invalid_account: bool = _d(True, "An invalid account snapshot HALTS new risk approvals.")
    halt_on_invalid_symbol: bool = _d(True, "An invalid contract specification HALTS new risk approvals.")
    halt_on_daily_limit: bool = _d(False, "Reaching the daily budget also HALTS (otherwise it only blocks new trades that day).")
    halt_on_external_breach: bool = _d(True, "An external ruleset breach HALTS new risk approvals.")


@dataclass(frozen=True)
class EventPolicy:
    weekend: str = _d("ALLOW", "ALLOW, DISALLOW, CLOSE_BEFORE_WEEKEND or UNKNOWN_EXTERNAL_POLICY (the last three block "
                               "new approvals inside weekend_cutoff_hours before the Friday roll).")
    weekend_cutoff_hours: float = _d(4.0, "Hours before the Friday roll inside which blocking weekend policies apply.")
    news: str = _d("ALLOW", "ALLOW, BLOCK_BEFORE, BLOCK_AFTER, REDUCE_RISK or UNKNOWN (Phase 1D news status).")
    news_reduce_factor: float = _d(0.5, "Risk multiplier (<= 1) for REDUCE_RISK around relevant GBP/JPY events.")
    block_unknown_news: bool = _d(False, "Block when news status is UNKNOWN (no provider exists yet).")


@dataclass(frozen=True)
class RiskPolicy:
    sizing: SizingPolicy = field(default_factory=SizingPolicy)
    limits: LimitPolicy = field(default_factory=LimitPolicy)
    consecutive: ConsecutiveLossPolicy = field(default_factory=ConsecutiveLossPolicy)
    drawdown: DrawdownPolicy = field(default_factory=DrawdownPolicy)
    margin: MarginPolicy = field(default_factory=MarginPolicy)
    data: DataPolicy = field(default_factory=DataPolicy)
    events: EventPolicy = field(default_factory=EventPolicy)

    def to_dict(self) -> dict:
        return asdict(self)

    def policy_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def validate(self) -> None:
        s, lim = self.sizing, self.limits
        if not (0 < s.base_risk_percent <= s.max_single_trade_risk_percent <= 5.0):
            raise ValueError("require 0 < base_risk_percent <= max_single_trade_risk_percent <= 5")
        if s.basis not in BASES:
            raise ValueError(f"sizing.basis must be one of {BASES}")
        if min(s.stop_slippage_allowance_pips, s.entry_slippage_allowance_pips) < 0 or s.assumed_commission_pips_round_turn <= 0:
            raise ValueError("slippage allowances must be >= 0 and the commission assumption > 0 (never zero)")
        for v in (lim.daily_loss_percent, lim.weekly_loss_percent, lim.monthly_loss_percent, lim.max_aggregate_risk_percent,
                  lim.max_symbol_risk_percent):
            if v <= 0:
                raise ValueError("loss budgets and caps must be positive")
        if lim.aggregate_policy not in ("REDUCE", "REJECT") or self.margin.margin_policy not in ("REDUCE", "REJECT"):
            raise ValueError("aggregate/margin policies must be REDUCE or REJECT")
        c = self.consecutive
        if not (0 <= c.reduce_factor <= 1) or c.pause_until not in ("NEXT_TRADING_DAY", "MANUAL_RESET"):
            raise ValueError("consecutive-loss policy: reduce_factor in [0, 1]; pause_until NEXT_TRADING_DAY|MANUAL_RESET")
        prev = -1.0
        for name, th, mult in self.drawdown.tiers:
            if not (0 <= mult <= 1) or th < prev:
                raise ValueError("drawdown tiers must have ascending thresholds and multipliers in [0, 1] (never > 1)")
            prev = th
        e = self.events
        if e.weekend not in WEEKEND or e.news not in NEWS or not (0 <= e.news_reduce_factor <= 1):
            raise ValueError("invalid event policy")


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def risk_policy_from_dict(values) -> RiskPolicy:
    base = RiskPolicy()
    unknown = set(values or {}) - {f.name for f in fields(RiskPolicy)}
    if unknown:
        raise KeyError(f"unknown risk policy sections: {sorted(unknown)}")
    sections = {}
    for name, v in (values or {}).items():
        cur = getattr(base, name)
        bad = set(v) - {f.name for f in fields(cur)}
        if bad:
            raise KeyError(f"unknown risk policy keys for {name}: {sorted(bad)}")
        sections[name] = replace(cur, **{k: _tuples(x) for k, x in v.items()})
    pol = replace(base, **sections)
    pol.validate()
    return pol


def load_risk_policy(path=None) -> RiskPolicy:
    if path is None:
        pol = RiskPolicy()
        pol.validate()
        return pol
    from pathlib import Path

    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return risk_policy_from_dict(data)
