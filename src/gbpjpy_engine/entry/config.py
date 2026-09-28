"""Phase 1D entry-intelligence configuration.

Every parameter is documented.  All values are untested engineering baselines
chosen for defensibility; none was fitted to profit or tuned to make more
candidates pass.  An entry candidate is not an order, and nothing here sizes
a position or sets a stop loss or take profit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..config import _p

CONFIRMATION_FAMILIES = (
    "STRUCTURAL_BREAK_CONFIRMATION",
    "DISPLACEMENT_CONFIRMATION",
    "BREAK_RETEST_CONFIRMATION",
    "SWEEP_RECLAIM_CONFIRMATION",
    "MOMENTUM_REACCELERATION_CONFIRMATION",
    "COMPRESSION_EXPANSION_CONFIRMATION",
)


@dataclass(frozen=True)
class ConfirmationConfig:
    confirming_score: float = _p(35.0, "An allowed confirmation family scoring at/above this (but below its policy minimum) "
                                       "puts the candidate in CONFIRMING (partial evidence).")
    structural_min_state: str = _p("CANDIDATE", "Minimum Phase 1A break lifecycle state (CANDIDATE/CONFIRMED/ACCEPTED) "
                                                "for a structural-break confirmation. CANDIDATE = judged at the break close.")
    structural_lookback_bars: int = _p(3, "A structural break at most this many bars old (and after qualification) can confirm.")
    giant_candle_atr: float = _p(2.5, "Range (prior ATR) at/above which a candle without multi-bar support is an isolated "
                                      "giant candle; its displacement confirmation score is multiplied by 0.6.")
    retest_max_bars: int = _p(24, "A break older than this cannot be the subject of a break-retest confirmation.")
    retest_tolerance_atr: float = _p(0.3, "A low (long) / high (short) within this many ATR of the broken level is a retest.")
    retest_max_penetration_atr: float = _p(0.5, "Penetration back through the level beyond this many ATR = breakout failure.")
    sweep_max_bars: int = _p(12, "A liquidity sweep older than this cannot be reclaimed into a confirmation.")
    reacceleration_lookback: int = _p(5, "Bars used for body progression / structural progress in momentum reacceleration.")
    expansion_max_bars: int = _p(6, "An expansion-from-compression event older than this cannot confirm.")
    extreme_expansion_atr: float = _p(3.0, "Expansion bar range (prior ATR) at/above which the expansion is treated as a chase "
                                           "and its confirmation score is capped at 40.")
    w_primary: float = _p(0.6, "confirmation_quality = w_primary x best qualifying family + (1 - w_primary) x support from "
                               "the OTHER evidence groups (the primary family's own group is excluded: no double counting).")


@dataclass(frozen=True)
class PolicyConfig:
    """Family-specific confirmation policies: which confirmation families may confirm which setup family."""

    policies: tuple = _p(
        (
            ("TREND_PULLBACK_CONTINUATION",
             ("STRUCTURAL_BREAK_CONFIRMATION", "DISPLACEMENT_CONFIRMATION", "MOMENTUM_REACCELERATION_CONFIRMATION"),
             55.0, 45.0),
            ("BREAK_RETEST_CONTINUATION",
             ("BREAK_RETEST_CONFIRMATION", "DISPLACEMENT_CONFIRMATION", "MOMENTUM_REACCELERATION_CONFIRMATION"),
             55.0, 45.0),
            ("LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION",
             ("SWEEP_RECLAIM_CONFIRMATION", "STRUCTURAL_BREAK_CONFIRMATION"),
             55.0, 45.0),
            ("COMPRESSION_EXPANSION_IN_H4_DIRECTION",
             ("COMPRESSION_EXPANSION_CONFIRMATION", "STRUCTURAL_BREAK_CONFIRMATION", "DISPLACEMENT_CONFIRMATION"),
             55.0, 45.0),
        ),
        "Per setup family: (setup_family, allowed confirmation families, minimum family score, minimum "
        "confirmation_quality_score). Untested baselines - NOT optimised.",
    )


@dataclass(frozen=True)
class LifecycleConfig:
    max_wait_bars: int = _p(8, "A qualified setup without confirmation after this many H1 bars EXPIRES.")
    run_away_atr: float = _p(2.0, "Price moving this many ATR in the setup direction (from the qualification close) without "
                                  "confirmation EXPIRES the candidate (the move left without it).")
    window_bars: int = _p(2, "Executable opportunities (next bar opens) after confirmation before the entry window EXPIRES.")
    window_max_hours: float = _p(3.0, "Wall-clock hours after the confirmation close after which the window EXPIRES.")
    candidate_valid_bars: int = _p(2, "After acceptance, bars for which an ENTRY_CANDIDATE stays available before EXPIRING.")
    max_active_per_side: int = _p(1, "Concurrent tracked candidates per side. Phase 1C allows one active setup per side, so "
                                     "this must be 1 for now; the tracker is keyed by candidate id to allow more later.")


@dataclass(frozen=True)
class FreshnessConfig:
    stale_confirmation_bars: int = _p(3, "Bars since confirmation mapping the confirmation-age component to fully stale.")
    stale_qualification_bars: int = _p(12, "Bars since qualification mapping the qualification-age component to fully stale.")
    stale_move_since_confirmation_atr: float = _p(1.0, "Directional move since the confirmation close (ATR) that is fully stale.")
    stale_move_since_qualification_atr: float = _p(2.5, "Directional move since the qualification close (ATR) that is fully stale.")
    stale_hours: float = _p(4.0, "Wall-clock hours since the confirmation close that are fully stale.")
    fresh_score: float = _p(70.0, "signal_freshness_score at/above which the signal is FRESH.")
    aging_score: float = _p(45.0, "Score at/above which the signal is AGING (below: STALE).")
    stale_score: float = _p(25.0, "Score below which the signal is EXPIRED.")


@dataclass(frozen=True)
class ChaseConfig:
    moderate_score: float = _p(30.0, "chase_risk_score at/above which chase risk is MODERATE (below: LOW).")
    high_score: float = _p(50.0, "chase_risk_score at/above which chase risk is HIGH.")
    extreme_score: float = _p(70.0, "chase_risk_score at/above which chase risk is EXTREME.")
    reject_at: str = _p("HIGH", "Chase class (MODERATE/HIGH/EXTREME) at/above which the candidate is REJECTED.")


@dataclass(frozen=True)
class EntryExtensionConfig:
    extended_score: float = _p(50.0, "entry_extension_score at/above which the entry is EXTENDED.")
    overextended_score: float = _p(75.0, "entry_extension_score at/above which the entry is OVEREXTENDED (REJECTED).")


@dataclass(frozen=True)
class ExecutionPriceConfig:
    price_basis: str = _p("bid", "What the OHLC prices represent: 'bid' (typical retail FX charts), 'mid' or 'ask'. "
                                 "LONG executes on the ASK, SHORT on the BID; the reference is adjusted accordingly.")
    spread_unit: str = _p("pips", "Units of the canonical 'spread' column as declared by the data adapter: "
                                  "'pips', 'points' (see points_per_pip) or 'price'.")
    points_per_pip: float = _p(10.0, "Broker points per pip when spread_unit == 'points' (3-digit JPY quotes: 10).")
    unknown_spread_assumption_pips: float = _p(
        3.0, "When spread is UNKNOWN the executable reference is computed with this CONSERVATIVE assumed spread and "
             "flagged spread_assumed=true. Zero spread is never assumed.")


@dataclass(frozen=True)
class SpreadConfig:
    normal_max_pips: float = _p(4.0, "Spread (pips) at/below which the absolute test is NORMAL.")
    elevated_max_pips: float = _p(6.0, "Spread at/below which the absolute test is ELEVATED (above: HIGH).")
    high_max_pips: float = _p(10.0, "Spread at/below which the absolute test is HIGH (above: EXTREME).")
    normal_max_atr_fraction: float = _p(0.10, "Spread / H1 ATR at/below which the relative test is NORMAL.")
    elevated_max_atr_fraction: float = _p(0.15, "Spread / ATR at/below which the relative test is ELEVATED.")
    high_max_atr_fraction: float = _p(0.25, "Spread / ATR at/below which the relative test is HIGH (above: EXTREME).")
    history_bars: int = _p(200, "Trailing completed bars forming the recent spread distribution.")
    history_min: int = _p(20, "Minimum known spreads before the distribution test is used.")
    normal_max_ratio: float = _p(1.5, "Spread / trailing median at/below which the distribution test is NORMAL.")
    elevated_max_ratio: float = _p(2.5, "Spread / median at/below which the distribution test is ELEVATED.")
    high_max_ratio: float = _p(4.0, "Spread / median at/below which the distribution test is HIGH (above: EXTREME).")
    block_statuses: tuple = _p(("HIGH", "EXTREME"), "Spread statuses that REJECT a candidate (SPREAD_TOO_HIGH).")
    block_unknown: bool = _p(False, "REJECT when spread is UNKNOWN (off by default; UNKNOWN always stays visible).")


@dataclass(frozen=True)
class GapConfig:
    small_gap_atr: float = _p(0.10, "Gap (next open vs confirmation-bar close, ATR) below which there is no gap.")
    defer_gap_atr: float = _p(0.50, "Gap against the thesis at/above which the decision is DEFERRED to re-check structure.")
    reject_gap_atr: float = _p(1.00, "Gap in either direction at/above which the candidate is REJECTED (price discontinuity).")
    closure_gap_hours: float = _p(2.0, "Time between a bar close and the next available price above which the market was "
                                       "closed (weekend / holiday / feed outage): a pre-closure confirmation is not carried over.")
    reopen_revalidation_bars: int = _p(1, "After a closure, this many completed bars must pass before a confirmation counts.")
    friday_cutoff_hour_utc: int = _p(20, "Confirmations whose bar closes on Friday at/after this UTC hour are DEFERRED "
                                         "(never accepted into the weekend).")


@dataclass(frozen=True)
class DeteriorationConfig:
    acceptable_atr: float = _p(0.25, "Adverse executable-vs-signal difference (ATR) at/below which the window is AVAILABLE.")
    reject_atr: float = _p(0.60, "Adverse difference at/above which the candidate is REJECTED (window EXPIRED by price).")


@dataclass(frozen=True)
class BarrierConfig:
    max_barriers: int = _p(5, "Nearest opposing barrier clusters retained (H1 + H4 combined).")
    cluster_tolerance_atr: float = _p(0.5, "Barriers within this many H1 ATR are one cluster (counted once).")
    min_zone_strength: float = _p(50.0, "Minimum level_strength_score for an H1 zone to count as a barrier.")
    density_window_atr: float = _p(3.0, "Clusters within this distance contribute to barrier_density_score.")
    density_norm: float = _p(1.5, "Weighted cluster sum mapping to barrier_density_score 100.")
    nearby_atr: float = _p(1.0, "A multi-member or multi-timeframe cluster within this distance raises BARRIER_CLUSTER_NEARBY.")
    min_remaining_room_score: float = _p(30.0, "remaining room score below which the candidate is REJECTED.")


@dataclass(frozen=True)
class AbnormalConfig:
    range_atr_start: float = _p(2.0, "Bar range (prior ATR) where the extreme-candle component starts.")
    range_atr_full: float = _p(4.0, "Bar range where the extreme-candle component is 1.")
    atr_ratio_start: float = _p(1.3, "Short/long ATR ratio where the volatility-expansion component starts.")
    atr_ratio_full: float = _p(2.2, "ATR ratio where the volatility-expansion component is 1.")
    gap_atr_start: float = _p(0.5, "Gap (ATR) where the gap component starts.")
    gap_atr_full: float = _p(2.0, "Gap where the gap component is 1.")
    spread_ratio_start: float = _p(2.0, "Spread / median where the spread-explosion component starts.")
    spread_ratio_full: float = _p(5.0, "Spread / median where the spread-explosion component is 1.")
    defer_score: float = _p(60.0, "execution_abnormality_score at/above which the decision is DEFERRED.")
    reject_score: float = _p(85.0, "Score at/above which the candidate is REJECTED.")


@dataclass(frozen=True)
class NewsConfig:
    currencies: tuple = _p(("GBP", "JPY"), "Currencies whose scheduled events are relevant (a provider may add 'GLOBAL').")
    min_importance: str = _p("HIGH", "Minimum event importance considered (LOW/MEDIUM/HIGH).")
    pre_event_minutes: float = _p(30.0, "An event starting within this many minutes is EVENT_IMMINENT.")
    post_event_minutes: float = _p(30.0, "An event that started within this many minutes is EVENT_RECENT.")
    block_on_event: bool = _p(False, "REJECT when an EVENT_IMMINENT/EVENT_RECENT status exists (off: no validated rule yet).")
    block_unknown: bool = _p(False, "REJECT when news_status is UNKNOWN (off by default; UNKNOWN is always recorded).")


@dataclass(frozen=True)
class EntryScoringConfig:
    w_confirmation: float = _p(0.20, "Entry-quality family weight: CONFIRMATION.")
    w_timing: float = _p(0.15, "Family weight: TIMING (100 - chase risk).")
    w_freshness: float = _p(0.10, "Family weight: FRESHNESS.")
    w_price_quality: float = _p(0.10, "Family weight: PRICE_QUALITY (100 - price deterioration).")
    w_market_quality: float = _p(0.10, "Family weight: MARKET_QUALITY (execution-time chop/efficiency/volatility/gap).")
    w_structural_integrity: float = _p(0.10, "Family weight: STRUCTURAL_INTEGRITY.")
    w_room: float = _p(0.10, "Family weight: ROOM (remaining room at the executable reference).")
    w_execution_conditions: float = _p(0.10, "Family weight: EXECUTION_CONDITIONS (spread quality, abnormality).")
    w_conflict: float = _p(0.05, "Family weight: CONFLICT (100 - entry conflict).")
    conflict_penalty: float = _p(0.5, "entry_quality_score *= 1 - conflict_penalty x entry_conflict/100.")
    min_entry_quality: float = _p(50.0, "entry_quality_score below which the candidate is REJECTED.")
    max_entry_conflict: float = _p(60.0, "entry_conflict_score above which the candidate is REJECTED.")
    unknown_spread_quality: float = _p(40.0, "Spread-quality value used in scores when spread is UNKNOWN (never 100).")


@dataclass(frozen=True)
class EntryConfig:
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    freshness: FreshnessConfig = field(default_factory=FreshnessConfig)
    chase: ChaseConfig = field(default_factory=ChaseConfig)
    extension: EntryExtensionConfig = field(default_factory=EntryExtensionConfig)
    execution: ExecutionPriceConfig = field(default_factory=ExecutionPriceConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    gap: GapConfig = field(default_factory=GapConfig)
    deterioration: DeteriorationConfig = field(default_factory=DeteriorationConfig)
    barriers: BarrierConfig = field(default_factory=BarrierConfig)
    abnormal: AbnormalConfig = field(default_factory=AbnormalConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    scoring: EntryScoringConfig = field(default_factory=EntryScoringConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def policy_for(self, setup_family: str) -> dict:
        for fam, allowed, min_score, min_quality in self.policy.policies:
            if fam == setup_family:
                return {"setup_family": fam, "allowed": tuple(allowed), "min_family_score": float(min_score),
                        "min_confirmation_quality": float(min_quality)}
        raise KeyError(f"no confirmation policy for setup family {setup_family!r}")

    def validate(self) -> None:
        if self.execution.price_basis not in ("bid", "mid", "ask"):
            raise ValueError("execution.price_basis must be bid, mid or ask")
        if self.execution.spread_unit not in ("pips", "points", "price"):
            raise ValueError("execution.spread_unit must be pips, points or price")
        if self.execution.unknown_spread_assumption_pips <= 0:
            raise ValueError("unknown_spread_assumption_pips must be > 0 (zero spread is never assumed)")
        if self.lifecycle.max_active_per_side != 1:
            raise ValueError("max_active_per_side must be 1 while Phase 1C tracks one active setup per side")
        if self.confirmation.structural_min_state not in ("CANDIDATE", "CONFIRMED", "ACCEPTED"):
            raise ValueError("structural_min_state must be CANDIDATE, CONFIRMED or ACCEPTED")
        if self.chase.reject_at not in ("MODERATE", "HIGH", "EXTREME"):
            raise ValueError("chase.reject_at must be MODERATE, HIGH or EXTREME")
        if not (self.chase.moderate_score <= self.chase.high_score <= self.chase.extreme_score):
            raise ValueError("chase thresholds must be ordered")
        if not (self.deterioration.acceptable_atr < self.deterioration.reject_atr):
            raise ValueError("deterioration.acceptable_atr must be < reject_atr")
        if not (self.gap.small_gap_atr <= self.gap.defer_gap_atr <= self.gap.reject_gap_atr):
            raise ValueError("gap thresholds must be ordered")
        seen = set()
        for fam, allowed, _, _ in self.policy.policies:
            if not set(allowed) <= set(CONFIRMATION_FAMILIES) or not allowed:
                raise ValueError(f"policy for {fam} names unknown confirmation families")
            seen.add(fam)
        from ..h1.setups import FAMILIES

        if seen != set(FAMILIES):
            raise ValueError("every Phase 1C setup family needs exactly one confirmation policy")
        w = [getattr(self.scoring, f.name) for f in fields(self.scoring) if f.name.startswith("w_")]
        if min(w) < 0 or sum(w) <= 0:
            raise ValueError("entry scoring weights must be non-negative and not all zero")


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def entry_config_from_dict(values) -> EntryConfig:
    """Build an EntryConfig from a (partial) dict; unspecified keys keep the defaults."""
    base = EntryConfig()
    known = {f.name for f in fields(EntryConfig)}
    unknown = set(values or {}) - known
    if unknown:
        raise KeyError(f"unknown entry config sections: {sorted(unknown)}")
    sections = {}
    for name, v in (values or {}).items():
        current = getattr(base, name)
        bad = set(v) - {f.name for f in fields(current)}
        if bad:
            raise KeyError(f"unknown config keys for {name}: {sorted(bad)}")
        sections[name] = replace(current, **{k: _tuples(val) for k, val in v.items()})
    cfg = replace(base, **sections)
    cfg.validate()
    return cfg


def load_entry_config(path=None) -> EntryConfig:
    """Load an entry config from YAML/JSON; missing keys keep the defaults."""
    if path is None:
        cfg = EntryConfig()
        cfg.validate()
        return cfg
    from pathlib import Path

    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return entry_config_from_dict(data)
