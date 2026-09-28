"""Central configuration for the GBPJPY H4 Market Intelligence Engine.

Every parameter carries a ``doc`` string in its dataclass field metadata that
explains its purpose.  ``describe_config()`` renders all of them, and the
same text is reproduced in ``docs/PHASE_1A_H4_MARKET_INTELLIGENCE.md``.

These values are *baseline defaults chosen for defensibility*, not optimised
values.  No parameter in this file has been fitted to historical performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping


def _p(default: Any, doc: str) -> Any:
    """Declare a documented parameter."""
    if isinstance(default, (list, dict)):
        raise TypeError("use tuples for sequence defaults to keep config immutable")
    return field(default=default, metadata={"doc": doc})


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DataConfig:
    symbol: str = _p("GBPJPY", "Instrument identifier. The engine is dedicated to GBPJPY.")
    timeframe_minutes: int = _p(240, "Bar duration in minutes (H4 = 240). Bar timestamps are bar OPEN times.")
    pip_size: float = _p(0.01, "Price value of one pip for a JPY-quoted pair.")
    max_weekend_gap_hours: float = _p(
        72.0,
        "Largest gap between consecutive bar opens that is treated as a normal weekend closure "
        "(only when the gap spans a Saturday). Longer gaps are flagged as missing bars.",
    )
    abnormal_gap_range_mult: float = _p(
        2.0,
        "An open-vs-previous-close gap larger than this multiple of the trailing median bar range "
        "is flagged ABNORMAL_PRICE_GAP (flagged, never repaired).",
    )
    gap_reference_window: int = _p(50, "Trailing bars used for the median-range reference in gap detection.")


@dataclass(frozen=True)
class EngineConfig:
    warmup_bars: int = _p(
        250,
        "Bars required before outputs are considered fully formed (EMA200 + percentile history). "
        "Before this, regime=UNCLEAR and bias=NEUTRAL with reason INSUFFICIENT_HISTORY.",
    )
    fail_on_data_errors: bool = _p(
        True,
        "Raise DataIntegrityError when ERROR-severity data issues exist (duplicates, out-of-order, "
        "invalid OHLC, non-positive prices, naive timestamps). Data is never silently repaired.",
    )


@dataclass(frozen=True)
class SwingConfig:
    left_bars: int = _p(3, "Bars to the LEFT of a pivot that must be strictly lower (high pivot) / higher (low pivot).")
    right_bars: int = _p(
        3,
        "Bars to the RIGHT required to confirm a pivot. A pivot at bar i only becomes known at the "
        "CLOSE of bar i+right_bars; that is its availability timestamp.",
    )
    min_swing_atr: float = _p(
        1.0,
        "Minimum distance (in ATR at confirmation time) from the previous opposite swing for a new "
        "swing to be accepted. Filters candle noise out of the structural swing sequence.",
    )
    equal_level_atr: float = _p(
        0.10,
        "Swings within this many ATR of the previous same-type swing are labelled equal (EH/EL) "
        "rather than higher/lower.",
    )


@dataclass(frozen=True)
class StructureConfig:
    break_min_atr: float = _p(
        0.25,
        "A structural break requires the candle CLOSE to be beyond the swing level by at least this "
        "many ATR. Wick-only or tiny closes are not structural breaks. Such a break starts as CANDIDATE.",
    )
    break_confirm_bars: int = _p(
        2,
        "Bars after a break during which every close must hold beyond the level for the break to move "
        "CANDIDATE -> CONFIRMED. Any close back through the level in this window moves it to INVALIDATED.",
    )
    break_accept_bars: int = _p(
        6,
        "Minimum bars since the break before a CONFIRMED break can become ACCEPTED (requires follow-through, "
        "see break_accept_atr).",
    )
    break_accept_atr: float = _p(
        1.0,
        "Follow-through required for ACCEPTED: the best close beyond the level must reach this many ATR "
        "(ATR at the break bar).",
    )
    break_fail_atr: float = _p(
        0.25,
        "A CONFIRMED or ACCEPTED break becomes FAILED when a later close is back through the level by at least "
        "this many ATR (ATR of that later bar). Earlier recorded states are never rewritten.",
    )
    break_monitor_bars: int = _p(
        60,
        "Bars after the break during which the lifecycle is monitored. After this the last state is frozen; "
        "a much later return through the level is treated as new structure, not as failure of this break.",
    )
    swing_history_size: int = _p(
        16,
        "Rolling number of confirmed structural swings retained in working memory and exposed per bar "
        "(swing_history). Must be >= 4 and >= quality_swings.",
    )
    transition_memory_bars: int = _p(
        30,
        "How long (bars) a counter-structure break (CHoCH) that is neither INVALIDATED nor FAILED keeps the "
        "structure state 'transitional' unless the swing sequence re-establishes a clear direction first.",
    )
    quality_swings: int = _p(8, "Number of most recent confirmed swings used for structure-quality measures.")
    quality_window_bars: int = _p(60, "Trailing bars used for break-conflict, reversal-frequency and persistence measures.")


@dataclass(frozen=True)
class TrendConfig:
    ema_fast: int = _p(20, "Fast EMA period.")
    ema_mid: int = _p(50, "Medium EMA period; also the trend baseline for crossings and flatness.")
    ema_slow: int = _p(200, "Slow EMA period (long-term context).")
    slope_lookback: int = _p(5, "Bars over which EMA slopes are measured.")
    slope_norm_atr_per_bar: float = _p(0.08, "EMA slope (ATR per bar) that maps to ~76% of maximum slope contribution (tanh scaling).")
    separation_norm_atr: float = _p(1.5, "EMA20-EMA50 separation (ATR) that maps to ~76% of maximum separation contribution.")
    distance_norm_atr: float = _p(2.0, "Price-to-EMA50 distance (ATR) that maps to ~76% of maximum distance contribution.")
    persistence_window: int = _p(30, "Bars over which EMA20-vs-EMA50 alignment persistence is averaged.")
    trend_threshold: float = _p(30.0, "Absolute trend score (0-100) required for a bullish/bearish trend state.")
    strong_threshold: float = _p(60.0, "Absolute trend score required for a strong trend state.")


@dataclass(frozen=True)
class AdxConfig:
    period: int = _p(14, "Wilder ADX / DI period.")
    slope_lookback: int = _p(3, "Bars over which ADX slope is measured.")
    persistence_window: int = _p(20, "Bars over which DI dominance persistence is averaged.")
    weak_adx: float = _p(18.0, "ADX below this is treated as non-directional context.")
    strong_adx: float = _p(25.0, "ADX above this is treated as a directional environment (context only, never a signal).")


@dataclass(frozen=True)
class VolatilityConfig:
    atr_period: int = _p(14, "Primary ATR period (Wilder).")
    atr_short_period: int = _p(5, "Short ATR for expansion/contraction measurement.")
    atr_long_period: int = _p(50, "Long ATR for expansion/contraction measurement.")
    percentile_lookback: int = _p(500, "Trailing bars (~4 months of H4) for ATR% percentile ranking.")
    percentile_min_periods: int = _p(100, "Minimum history before an ATR percentile is reported.")
    very_low_pct: float = _p(10.0, "ATR percentile at/below which volatility is 'very_low'.")
    low_pct: float = _p(30.0, "ATR percentile at/below which volatility is 'low'.")
    high_pct: float = _p(70.0, "ATR percentile at/above which volatility is 'high'.")
    extreme_pct: float = _p(92.0, "ATR percentile at/above which volatility is 'extreme'.")
    expansion_ratio: float = _p(1.20, "Short/long ATR ratio at/above which volatility is 'expanding'.")
    contraction_ratio: float = _p(0.80, "Short/long ATR ratio at/below which volatility is 'contracting'.")


@dataclass(frozen=True)
class ShockConfig:
    range_atr_start: float = _p(2.0, "Bar range / previous ATR at which range-shock severity starts above 0.")
    range_atr_extreme: float = _p(5.0, "Bar range / previous ATR at which range-shock severity reaches 100.")
    gap_atr_start: float = _p(0.5, "|open - previous close| / previous ATR at which gap severity starts.")
    gap_atr_extreme: float = _p(3.0, "Gap / previous ATR at which gap severity reaches 100.")
    wick_atr_start: float = _p(1.2, "Largest wick / previous ATR at which wick severity starts.")
    wick_atr_extreme: float = _p(3.5, "Largest wick / previous ATR at which wick severity reaches 100.")
    displacement_atr_start: float = _p(1.5, "|close - previous close| / previous ATR at which displacement severity starts.")
    displacement_atr_extreme: float = _p(4.0, "Close-to-close displacement / previous ATR at which severity reaches 100.")
    expansion_ratio_start: float = _p(1.4, "Short/long ATR ratio at which ATR-expansion severity starts.")
    expansion_ratio_extreme: float = _p(2.6, "Short/long ATR ratio at which ATR-expansion severity reaches 100.")
    shock_threshold: float = _p(50.0, "shock_severity at/above which volatility_shock=True.")


@dataclass(frozen=True)
class MomentumConfig:
    roc_periods: tuple = _p((3, 6, 12), "Return lookbacks (bars) for ATR-and-sqrt(n)-normalised rate of change.")
    roc_norm: float = _p(1.0, "Normalised ROC that maps to ~76% of maximum ROC contribution (tanh scaling).")
    velocity_window: int = _p(6, "Bars over which impulse velocity (ATR per bar) is measured.")
    velocity_norm: float = _p(0.25, "Velocity (ATR/bar) mapping to ~76% of maximum velocity contribution.")
    body_window: int = _p(6, "Bars over which signed body/range ratios are averaged.")
    consecutive_cap: int = _p(5, "Consecutive same-direction closes counted up to this cap.")
    history_lookback: int = _p(500, "Trailing bars for the percentile of current momentum vs historical momentum.")
    accel_lookback: int = _p(3, "Bars over which net momentum change is measured for acceleration state.")
    accel_threshold: float = _p(10.0, "Net-momentum change (points) required to call accelerating/decelerating.")
    flat_threshold: float = _p(10.0, "|net momentum| below which momentum is 'flat'.")


@dataclass(frozen=True)
class CandleConfig:
    strong_body_ratio: float = _p(0.60, "Body/range at/above which a close may be 'strong directional'.")
    strong_close_location: float = _p(0.75, "Close location (0=low,1=high) required for a strong bullish close (mirrored for bearish).")
    rejection_wick_ratio: float = _p(0.50, "Single wick/range at/above which a candle may be a 'rejection'.")
    rejection_max_body: float = _p(0.35, "Maximum body/range for a rejection candle.")
    indecision_body_ratio: float = _p(0.20, "Body/range at/below which a candle is 'indecision'.")
    abnormal_range_atr: float = _p(2.0, "Range / previous ATR at/above which a candle is 'abnormal expansion'.")


@dataclass(frozen=True)
class ChopConfig:
    window: int = _p(20, "Trailing bars for overlap, baseline-crossing and displacement measures.")
    choppiness_period: int = _p(14, "Choppiness Index period.")
    flat_slope_atr_per_bar: float = _p(0.06, "EMA50 slope (ATR/bar) at/above which the baseline is no longer considered flat.")
    compressed_separation_atr: float = _p(1.0, "EMA20-EMA50 separation (ATR) at/above which EMAs are not compressed.")
    severe_threshold: float = _p(72.0, "chop_score at/above which market quality is 'severe_chop'.")
    range_threshold: float = _p(55.0, "chop_score at/above which market quality is 'range'.")
    clean_threshold: float = _p(38.0, "chop_score below which (with trend strength) quality is 'clean_trend'.")
    clean_trend_strength: float = _p(50.0, "Trend strength required for 'clean_trend'.")
    noisy_trend_strength: float = _p(30.0, "Trend strength required for 'trend_with_noise'.")


@dataclass(frozen=True)
class EfficiencyConfig:
    lookback: int = _p(20, "Bars for directional efficiency: |net displacement| / sum(|bar-to-bar moves|).")
    percentile_lookback: int = _p(500, "Trailing bars for the efficiency percentile.")
    percentile_min_periods: int = _p(100, "Minimum history before an efficiency percentile is reported.")
    low_efficiency: float = _p(0.20, "Efficiency below this raises LOW_DIRECTIONAL_EFFICIENCY.")
    high_efficiency: float = _p(0.45, "Efficiency at/above this raises HIGH_DIRECTIONAL_EFFICIENCY.")


@dataclass(frozen=True)
class LevelConfig:
    lookback_bars: int = _p(300, "Only swings/breaks/touches within this many trailing bars contribute to zones.")
    cluster_atr_mult: float = _p(0.5, "Level prices within this many ATR (current ATR) are clustered into one zone.")
    min_half_width_atr: float = _p(0.10, "Minimum half-width of a zone in ATR.")
    range_extreme_lookback: int = _p(60, "Recent range high/low lookback used as a level source.")
    max_zones: int = _p(12, "Maximum zones retained (strongest first) to avoid meaningless line clutter.")
    near_level_atr: float = _p(0.75, "A zone within this many ATR of price counts as 'near'.")
    strong_level_score: float = _p(60.0, "level_strength_score at/above which a zone is 'strong'.")
    invalidate_after_breaks: int = _p(
        4,
        "A zone is INVALIDATED once closes have crossed from one side of it to the other this many times "
        "within the lookback (it no longer behaves as a level). 0 disables break-based invalidation.",
    )


@dataclass(frozen=True)
class RoundNumberConfig:
    intervals: tuple = _p((1.0, 0.5), "JPY round-number intervals (1.00 = 100 pips, 0.50 = 50 pips). Context only.")
    near_pips: float = _p(15.0, "Distance (pips) to the primary round number that raises NEAR_ROUND_NUMBER.")


@dataclass(frozen=True)
class RangeLocationConfig:
    short: int = _p(20, "Short lookback (bars) for range-location percentile (~3.3 trading days).")
    medium: int = _p(60, "Medium lookback (bars) (~2 trading weeks).")
    long: int = _p(180, "Long lookback (bars) (~6 trading weeks).")


@dataclass(frozen=True)
class ExtensionConfig:
    impulse_window: int = _p(10, "Bars used to measure the recent impulse magnitude.")
    percentile_lookback: int = _p(500, "Trailing bars for the extension percentile.")
    percentile_min_periods: int = _p(100, "Minimum history before an extension percentile is used.")
    absolute_norm_atr: float = _p(3.0, "Composite EMA distance (ATR) that maps to an absolute sub-score of 100.")
    extended_score: float = _p(65.0, "extension_score at/above which state is 'extended'.")
    extreme_score: float = _p(85.0, "extension_score at/above which state is 'extremely_extended'.")


@dataclass(frozen=True)
class SessionConfig:
    sessions: tuple = _p(
        (
            ("asia", "Asia/Tokyo", "09:00", "18:00"),
            ("london", "Europe/London", "08:00", "17:00"),
            ("new_york", "America/New_York", "08:00", "17:00"),
        ),
        "Session definitions as (name, IANA timezone, local start, local end). DST is resolved per date via zoneinfo.",
    )
    min_overlap_minutes: int = _p(60, "Minutes a bar must overlap a session for that session to be counted as active.")


@dataclass(frozen=True)
class RegimeConfig:
    adx_trend_min: float = _p(20.0, "ADX required (with aligned structure/trend) for STRONG trend regimes.")
    strong_max_chop: float = _p(45.0, "Maximum chop_score for STRONG trend regimes.")
    strong_min_quality: float = _p(50.0, "Minimum structure_quality_score for STRONG trend regimes.")
    trend_max_chop: float = _p(55.0, "Maximum chop_score for (non-weak) trend regimes.")
    weak_max_chop: float = _p(65.0, "Maximum chop_score for WEAK trend regimes.")
    range_min_chop: float = _p(55.0, "Minimum chop_score for range regimes.")
    high_vol_range_pct: float = _p(70.0, "ATR percentile at/above which a range is HIGH_VOLATILITY_RANGE.")
    compression_pct: float = _p(20.0, "ATR percentile at/below which LOW_VOLATILITY_COMPRESSION may be declared.")
    compression_ratio: float = _p(0.90, "Short/long ATR ratio at/below which volatility is compressing.")


@dataclass(frozen=True)
class BiasConfig:
    structure_weight: float = _p(0.40, "Family weight: market structure.")
    trend_weight: float = _p(0.35, "Family weight: trend (EMA state + ADX/DI; correlated, so one family).")
    momentum_weight: float = _p(0.25, "Family weight: momentum.")
    chop_penalty: float = _p(0.40, "Market-quality dampener: evidence *= 1 - chop_penalty * chop_score/100 (both sides).")
    extreme_vol_factor: float = _p(0.85, "Volatility dampener applied to both sides when volatility is extreme.")
    shock_factor: float = _p(0.70, "Volatility dampener applied to both sides during a volatility shock.")
    extended_penalty: float = _p(0.10, "Location penalty on the side of an 'extended' move.")
    extreme_extension_penalty: float = _p(0.20, "Location penalty on the side of an 'extremely_extended' move.")
    opposing_level_penalty: float = _p(0.10, "Location penalty when price is near a strong opposing zone.")
    min_evidence: float = _p(45.0, "Dominant evidence score required before a directional bias is permitted.")
    min_separation: float = _p(20.0, "Required |bullish - bearish| evidence separation for a directional bias.")
    conflict_level: float = _p(35.0, "If BOTH evidence scores are at/above this, bias is NEUTRAL with BIAS_CONFLICT.")


# ---------------------------------------------------------------------------
# Phase 1B - H4 context and directional permission
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HierarchyConfig:
    primary_swings: int = _p(16, "Confirmed swings (most recent) interpreted for PRIMARY structure (<= swing_history_size).")
    intermediate_swings: int = _p(8, "Confirmed swings interpreted for INTERMEDIATE structure.")
    immediate_swings: int = _p(4, "Confirmed swings interpreted for IMMEDIATE structure (plus the current close).")
    consistency_threshold: float = _p(
        0.60, "Recency-weighted share of labels (HH/HL or LH/LL) required for a BULLISH/BEARISH layer."
    )
    recency_weight_ratio: float = _p(
        2.0,
        "Linear recency weighting: the newest swing in a window weighs this many times the oldest (weights rise "
        "linearly in between). 1.0 = equal weights. Documented, not exponential, not fitted.",
    )
    range_band_atr: float = _p(1.5, "Swing highs and swing lows each spanning <= this many ATR classify the window as RANGING.")
    protected_break_atr: float = _p(
        0.25, "A close beyond the protected swing (the low/high that launched the latest HH/LL) by this many ATR makes the layer TRANSITIONAL."
    )
    provisional_break_atr: float = _p(
        0.10, "IMMEDIATE layer only: a close beyond the latest swing high/low by this many ATR counts as a provisional HH/LL."
    )


@dataclass(frozen=True)
class LegConfig:
    strong_impulse_score: float = _p(70.0, "impulse_score at/above which a with-structure leg is STRONG_IMPULSE.")
    normal_impulse_score: float = _p(55.0, "impulse_score at/above which a with-structure leg is NORMAL_IMPULSE (below: WEAK_IMPULSE).")
    impulsive_score: float = _p(55.0, "impulse_score at/above which a leg's nature is IMPULSIVE.")
    corrective_score: float = _p(40.0, "impulse_score below which a leg's nature is CORRECTIVE (in between: MIXED).")
    healthy_correction_ratio: float = _p(
        0.60, "Counter-structure leg length / preceding leg length at/below which a correction is HEALTHY (descriptive band)."
    )
    choppy_efficiency: float = _p(0.30, "Correction directional efficiency below which it is CHOPPY_CORRECTION.")
    choppy_overlap: float = _p(0.60, "Mean candle overlap above which a correction is CHOPPY_CORRECTION.")


@dataclass(frozen=True)
class DisplacementConfig:
    window: int = _p(3, "Bars in the multi-candle displacement window (a single candle alone is never enough).")
    percentile_lookback: int = _p(500, "Trailing bars for the displacement-magnitude percentile.")
    min_multi_bar_atr: float = _p(1.5, "Net multi-bar move (ATR before the window) required for STRONG/EXTREME displacement.")
    min_directional_closes: int = _p(2, "Directional closes in the window required for STRONG/EXTREME displacement.")
    min_efficiency: float = _p(0.5, "Window directional efficiency required for STRONG/EXTREME displacement.")
    weak_score: float = _p(20.0, "Score at/above which displacement is WEAK (below: NONE).")
    moderate_score: float = _p(40.0, "Score at/above which displacement is MODERATE.")
    strong_score: float = _p(60.0, "Score at/above which displacement is STRONG.")
    extreme_score: float = _p(80.0, "Score at/above which displacement is EXTREME.")
    recent_bars: int = _p(6, "Bars over which the most recent displacement is considered 'recent' context.")


@dataclass(frozen=True)
class BreakoutContextConfig:
    importance_norm_atr: float = _p(3.0, "Swing significance (ATR) mapping to full level-importance.")
    retest_tolerance_atr: float = _p(0.30, "A bar trading back within this many ATR of a broken level (and closing beyond it) is a retest.")
    acceptance_min_closes: int = _p(3, "Closes beyond the level after the break that count as multi-close acceptance evidence.")
    high_quality_score: float = _p(70.0, "breakout_quality_score at/above which a break is HIGH_QUALITY_BREAK.")
    moderate_quality_score: float = _p(50.0, "breakout_quality_score at/above which a break is MODERATE_BREAK.")
    weak_quality_score: float = _p(30.0, "breakout_quality_score at/above which a break is WEAK_BREAK (below: FALSE_BREAK_CANDIDATE).")
    high_false_break_risk: float = _p(60.0, "false_break_risk_score at/above which HIGH_FALSE_BREAK_RISK is raised.")
    opposing_zone_atr: float = _p(1.0, "A strong opposing zone within this many ATR beyond a broken level adds false-break risk.")


@dataclass(frozen=True)
class LiquidityConfig:
    lookback_bars: int = _p(200, "Swings (by pivot bar) within this many bars can be liquidity references.")
    equal_tol_atr: float = _p(0.15, "Swing extremes within this many ATR form an EQUAL_HIGHS/EQUAL_LOWS cluster.")
    near_equal_tol_atr: float = _p(0.35, "Swing extremes within this many ATR (but not equal) form a NEAR_EQUAL cluster.")
    min_penetration_atr: float = _p(0.02, "Minimum trade beyond a reference (ATR) to count as a penetration.")
    max_sweep_penetration_atr: float = _p(2.0, "Penetration (ATR) at which a move is no longer sweep-like (score component reaches 0).")
    accept_close_atr: float = _p(0.25, "A close beyond the reference by this many ATR resolves the event as BREAK_AND_ACCEPT.")
    resolve_bars: int = _p(3, "Bars after penetration with every close back inside before SWEEP_AND_REJECT is declared.")
    monitor_bars: int = _p(12, "Bars after penetration during which a sweep event can still change state; then frozen.")
    sweep_memory_bars: int = _p(12, "A sweep contributes to bullish/bearish sweep scores for this many bars after penetration.")


@dataclass(frozen=True)
class OriginZoneConfig:
    min_displacement_score: float = _p(60.0, "Displacement score (STRONG) that qualifies an origin zone.")
    origin_bars: int = _p(2, "Bars immediately preceding the displacement window that form the origin zone.")
    max_width_atr: float = _p(2.5, "Origin zones wider than this many ATR at creation are not created (not a compact origin).")
    max_age_bars: int = _p(300, "Origin zones older than this are retired (ageing).")
    lightly_tested: int = _p(1, "Interactions at/below which a zone is LIGHTLY_TESTED (0 = FRESH).")
    tested: int = _p(3, "Interactions at/below which a zone is TESTED (above: HEAVILY_TESTED).")


@dataclass(frozen=True)
class RoleReversalConfig:
    acceptance_closes: int = _p(2, "Consecutive closes beyond a crossed zone required before a role reversal can be considered.")
    retest_window_bars: int = _p(30, "Bars after acceptance within which a retest must occur.")
    reaction_atr: float = _p(0.5, "Move away from the zone after a retest (ATR) confirming the new role.")


@dataclass(frozen=True)
class LocationContextConfig:
    min_range_atr: float = _p(1.5, "Structural range (latest swing high - latest swing low) must be at least this many ATR to be valid.")
    deep_discount_pct: float = _p(20.0, "Range percentile below which location is DEEP_DISCOUNT.")
    discount_pct: float = _p(45.0, "Range percentile below which location is DISCOUNT.")
    premium_pct: float = _p(55.0, "Range percentile above which location is PREMIUM (45-55 = EQUILIBRIUM).")
    deep_premium_pct: float = _p(80.0, "Range percentile above which location is DEEP_PREMIUM.")
    shallow_retracement_pct: float = _p(30.0, "Retracement below this % of the active impulse is SHALLOW (descriptive band, not Fibonacci).")
    normal_retracement_pct: float = _p(55.0, "Retracement below this % is NORMAL.")
    deep_retracement_pct: float = _p(80.0, "Retracement below this % is DEEP (at/above: VERY_DEEP).")


@dataclass(frozen=True)
class MaturityConfig:
    impulses_full: int = _p(5, "Number of with-trend impulses mapping to full maturity contribution.")
    distance_full_atr: float = _p(15.0, "Cumulative with-trend distance (ATR) mapping to full maturity contribution.")
    duration_full_bars: int = _p(150, "Trend duration (bars) mapping to full maturity contribution.")
    developing_score: float = _p(0.35, "Maturity score at/above which a trend is DEVELOPING (below and >1 impulse: EARLY).")
    mature_score: float = _p(0.60, "Maturity score at/above which a trend is MATURE.")
    extended_score: float = _p(0.80, "Maturity score at/above which a trend is EXTENDED.")
    exhaustion_deterioration: float = _p(60.0, "Momentum deterioration at/above which MATURE/EXTENDED becomes EXHAUSTION_RISK.")
    deterioration_high: float = _p(60.0, "momentum_deterioration_score at/above which MOMENTUM_DETERIORATING is raised.")


@dataclass(frozen=True)
class CompressionConfig:
    amplitude_swings: int = _p(6, "Most recent swings whose amplitude trend is measured.")
    range_short: int = _p(10, "Short window (bars) for mean candle range.")
    range_long: int = _p(50, "Long window (bars) for mean candle range.")
    compression_threshold: float = _p(60.0, "compression_score at/above which compression is present.")
    min_compression_bars: int = _p(4, "Consecutive compression bars forming a compression episode.")
    expansion_window_bars: int = _p(3, "Bars after a compression episode ends within which a displacement/break counts as expansion.")
    expansion_track_bars: int = _p(20, "Bars an expansion event is tracked for acceptance/failure.")
    expansion_range_ratio: float = _p(
        1.5, "An expansion bar's range must be at least this multiple of the mean bar range during the compression episode."
    )


@dataclass(frozen=True)
class RoomConfig:
    min_room_atr: float = _p(0.5, "Distance (ATR) to the nearest opposing barrier mapping to room score 0.")
    full_room_atr: float = _p(4.0, "Distance (ATR) to the nearest opposing barrier mapping to room score 100.")
    sufficient_room_score: float = _p(40.0, "Room score at/above which SUFFICIENT_*_ROOM is raised (below: INSUFFICIENT).")


@dataclass(frozen=True)
class ContextScoringConfig:
    q_structure: float = _p(0.20, "Context-quality family weight: STRUCTURE.")
    q_displacement: float = _p(0.10, "Context-quality family weight: DISPLACEMENT.")
    q_momentum: float = _p(0.10, "Context-quality family weight: MOMENTUM.")
    q_volatility: float = _p(0.15, "Context-quality family weight: VOLATILITY.")
    q_location: float = _p(0.10, "Context-quality family weight: LOCATION.")
    q_liquidity: float = _p(0.10, "Context-quality family weight: LIQUIDITY_CONTEXT.")
    q_room: float = _p(0.10, "Context-quality family weight: ROOM_TO_MOVE.")
    q_market_quality: float = _p(0.15, "Context-quality family weight: MARKET_QUALITY.")
    d_structure: float = _p(0.35, "Directional family weight: structure (primary/intermediate/immediate hierarchy).")
    d_continuation: float = _p(0.15, "Directional family weight: continuation behaviour (pullback health, accepted breaks).")
    d_displacement: float = _p(0.15, "Directional family weight: recent displacement.")
    d_level_behaviour: float = _p(0.10, "Directional family weight: support/resistance, role reversal and origin-zone behaviour.")
    d_liquidity: float = _p(0.10, "Directional family weight: rejected liquidity sweeps.")
    d_room: float = _p(0.15, "Directional family weight: room to move.")
    penalty_opposing_level: float = _p(0.30, "Directional score multiplier reduction when room in that direction is insufficient.")
    penalty_extension: float = _p(0.20, "Reduction when extremely extended in that direction.")
    penalty_failed_break: float = _p(0.30, "Reduction after a recent FAILED break in that direction.")
    penalty_transition: float = _p(0.20, "Reduction when structure is transitioning against that direction.")
    penalty_severe_chop: float = _p(0.40, "Reduction under severe chop.")
    penalty_conflict: float = _p(0.50, "Reduction = penalty_conflict x context_conflict_score/100.")
    high_conflict: float = _p(50.0, "context_conflict_score at/above which HIGH_CONTEXT_CONFLICT is raised.")
    low_conflict: float = _p(20.0, "context_conflict_score below which LOW_CONTEXT_CONFLICT is raised.")
    low_quality: float = _p(40.0, "context_quality_score below which LOW_CONTEXT_QUALITY is raised.")


@dataclass(frozen=True)
class PermissionConfig:
    min_context_score: float = _p(55.0, "long/short_context_score required before that direction may be permitted.")
    transition_min_context_score: float = _p(65.0, "Higher requirement when the regime is TRANSITION or UNCLEAR.")
    min_room_score: float = _p(35.0, "Room score required in the permitted direction.")
    max_conflict: float = _p(55.0, "context_conflict_score above which no direction is permitted.")
    min_quality: float = _p(45.0, "context_quality_score required for any permission.")
    both_separation: float = _p(
        15.0, "If both directions qualify and their scores differ by at least this much, only the stronger is permitted."
    )
    allow_both_regimes: tuple = _p(
        ("RANGE", "HIGH_VOLATILITY_RANGE"),
        "Regimes in which ALLOW_BOTH is possible (two-way range context). Elsewhere two-way qualification -> BLOCK_ALL.",
    )


@dataclass(frozen=True)
class BlockerConfig:
    block_invalid_data: bool = _p(True, "INVALID_DATA blocker: bar carries ERROR-level data issues.")
    block_insufficient_history: bool = _p(True, "INSUFFICIENT_HISTORY blocker: warm-up incomplete.")
    gap_block_bars: int = _p(6, "UNRESOLVED_DATA_GAP blocker for this many bars after a MISSING_BARS/abnormal-gap flag (0 disables).")
    shock_block_severity: float = _p(85.0, "EXTREME_VOLATILITY_SHOCK blocker when shock_severity on the bar is at/above this.")
    block_severe_chop: bool = _p(True, "SEVERE_CHOP blocker when market quality is severe_chop.")
    block_unclassifiable_structure: bool = _p(True, "UNCLASSIFIABLE_STRUCTURE blocker when primary AND intermediate are UNCLEAR.")
    stale_after_hours: float = _p(
        12.0, "STALE_DATA blocker (latest bar only, when an as_of time is supplied): as_of - bar close exceeds this."
    )


@dataclass(frozen=True)
class ContextStateConfig:
    flip_window_bars: int = _p(20, "Trailing bars over which context-state changes are counted (temporal stability).")


@dataclass(frozen=True)
class H4Config:
    data: DataConfig = field(default_factory=DataConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    swing: SwingConfig = field(default_factory=SwingConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    adx: AdxConfig = field(default_factory=AdxConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    shock: ShockConfig = field(default_factory=ShockConfig)
    momentum: MomentumConfig = field(default_factory=MomentumConfig)
    candle: CandleConfig = field(default_factory=CandleConfig)
    chop: ChopConfig = field(default_factory=ChopConfig)
    efficiency: EfficiencyConfig = field(default_factory=EfficiencyConfig)
    levels: LevelConfig = field(default_factory=LevelConfig)
    round_numbers: RoundNumberConfig = field(default_factory=RoundNumberConfig)
    range_location: RangeLocationConfig = field(default_factory=RangeLocationConfig)
    extension: ExtensionConfig = field(default_factory=ExtensionConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    legs: LegConfig = field(default_factory=LegConfig)
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    breakout_context: BreakoutContextConfig = field(default_factory=BreakoutContextConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    origin_zones: OriginZoneConfig = field(default_factory=OriginZoneConfig)
    role_reversal: RoleReversalConfig = field(default_factory=RoleReversalConfig)
    location_context: LocationContextConfig = field(default_factory=LocationContextConfig)
    maturity: MaturityConfig = field(default_factory=MaturityConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    context_scoring: ContextScoringConfig = field(default_factory=ContextScoringConfig)
    permission: PermissionConfig = field(default_factory=PermissionConfig)
    blockers: BlockerConfig = field(default_factory=BlockerConfig)
    context_state: ContextStateConfig = field(default_factory=ContextStateConfig)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        """Stable short hash of the configuration for reproducibility records."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def validate(self) -> None:
        s = self.swing
        if s.left_bars < 1 or s.right_bars < 1:
            raise ValueError("swing.left_bars and swing.right_bars must be >= 1")
        t = self.trend
        if not (t.ema_fast < t.ema_mid < t.ema_slow):
            raise ValueError("trend EMA periods must satisfy ema_fast < ema_mid < ema_slow")
        if t.trend_threshold >= t.strong_threshold:
            raise ValueError("trend.trend_threshold must be < trend.strong_threshold")
        v = self.volatility
        if not (0 <= v.very_low_pct < v.low_pct < v.high_pct < v.extreme_pct <= 100):
            raise ValueError("volatility percentile thresholds must be increasing within 0..100")
        st = self.structure
        if st.swing_history_size < 4 or st.swing_history_size < st.quality_swings:
            raise ValueError("structure.swing_history_size must be >= 4 and >= structure.quality_swings")
        if not (1 <= st.break_confirm_bars <= st.break_accept_bars <= st.break_monitor_bars):
            raise ValueError("require 1 <= break_confirm_bars <= break_accept_bars <= break_monitor_bars")
        hc = self.hierarchy
        if not (2 <= hc.immediate_swings <= hc.intermediate_swings <= hc.primary_swings <= st.swing_history_size):
            raise ValueError("require 2 <= immediate <= intermediate <= primary swings <= structure.swing_history_size")
        b = self.bias
        if min(b.structure_weight, b.trend_weight, b.momentum_weight) < 0:
            raise ValueError("bias family weights must be non-negative")
        if b.structure_weight + b.trend_weight + b.momentum_weight <= 0:
            raise ValueError("at least one bias family weight must be positive")
        if self.data.timeframe_minutes != 240:
            raise ValueError("Phase 1A engine is H4-only (timeframe_minutes must be 240)")


def _build(cls: type, values: Mapping[str, Any] | None) -> Any:
    values = dict(values or {})
    kwargs = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    for name, f in known.items():
        if name not in values:
            continue
        v = values[name]
        ftype = f.default_factory if f.default_factory is not MISSING else None  # type: ignore[misc]
        if ftype is not None and is_dataclass(ftype):
            kwargs[name] = _build(ftype, v)
        elif isinstance(v, list):
            kwargs[name] = tuple(tuple(x) if isinstance(x, list) else x for x in v)
        else:
            kwargs[name] = v
    return cls(**kwargs)


def config_from_dict(values: Mapping[str, Any] | None) -> H4Config:
    cfg = _build(H4Config, values)
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None) -> H4Config:
    """Load config from a YAML/JSON file; missing keys fall back to defaults."""
    if path is None:
        cfg = H4Config()
        cfg.validate()
        return cfg
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return config_from_dict(data)


def describe_config(cfg: H4Config | None = None) -> list[dict]:
    """Return [{section, name, value, doc}] for every parameter."""
    cfg = cfg or H4Config()
    rows = []
    for section in fields(cfg):
        sec_obj = getattr(cfg, section.name)
        for f in fields(sec_obj):
            rows.append(
                {
                    "section": section.name,
                    "name": f.name,
                    "value": getattr(sec_obj, f.name),
                    "doc": f.metadata.get("doc", ""),
                }
            )
    return rows
