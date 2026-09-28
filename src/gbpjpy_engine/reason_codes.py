"""Machine-readable reason codes attached to every H4 evaluation.

Codes are stable string identifiers intended for later trade auditing.
Each has a short human description in ``REASON_DESCRIPTIONS``.
"""

from __future__ import annotations

from enum import Enum


class ReasonCode(str, Enum):
    # structure
    STRUCTURE_BULLISH = "STRUCTURE_BULLISH"
    STRUCTURE_BEARISH = "STRUCTURE_BEARISH"
    STRUCTURE_NEUTRAL = "STRUCTURE_NEUTRAL"
    STRUCTURE_TRANSITIONAL = "STRUCTURE_TRANSITIONAL"
    STRUCTURE_UNCLEAR = "STRUCTURE_UNCLEAR"
    BOS_BULLISH = "BOS_BULLISH"
    BOS_BEARISH = "BOS_BEARISH"
    CHOCH_BULLISH = "CHOCH_BULLISH"
    CHOCH_BEARISH = "CHOCH_BEARISH"
    BREAK_REJECTED = "BREAK_REJECTED"
    BREAK_FAILED = "BREAK_FAILED"
    HIGH_STRUCTURE_QUALITY = "HIGH_STRUCTURE_QUALITY"
    LOW_STRUCTURE_QUALITY = "LOW_STRUCTURE_QUALITY"
    # trend
    TREND_ALIGNED_BULLISH = "TREND_ALIGNED_BULLISH"
    TREND_ALIGNED_BEARISH = "TREND_ALIGNED_BEARISH"
    TREND_STRONG_BULLISH = "TREND_STRONG_BULLISH"
    TREND_STRONG_BEARISH = "TREND_STRONG_BEARISH"
    TREND_NEUTRAL = "TREND_NEUTRAL"
    EMA_SLOPE_POSITIVE = "EMA_SLOPE_POSITIVE"
    EMA_SLOPE_NEGATIVE = "EMA_SLOPE_NEGATIVE"
    STRUCTURE_TREND_CONFLICT = "STRUCTURE_TREND_CONFLICT"
    # ADX
    ADX_STRONG = "ADX_STRONG"
    ADX_WEAK = "ADX_WEAK"
    DI_BULLISH = "DI_BULLISH"
    DI_BEARISH = "DI_BEARISH"
    # momentum
    MOMENTUM_BULLISH = "MOMENTUM_BULLISH"
    MOMENTUM_BEARISH = "MOMENTUM_BEARISH"
    MOMENTUM_ACCELERATING = "MOMENTUM_ACCELERATING"
    MOMENTUM_DECELERATING = "MOMENTUM_DECELERATING"
    # market quality
    LOW_CHOP = "LOW_CHOP"
    HIGH_CHOP = "HIGH_CHOP"
    SEVERE_CHOP = "SEVERE_CHOP"
    LOW_DIRECTIONAL_EFFICIENCY = "LOW_DIRECTIONAL_EFFICIENCY"
    HIGH_DIRECTIONAL_EFFICIENCY = "HIGH_DIRECTIONAL_EFFICIENCY"
    # volatility
    VOLATILITY_VERY_LOW = "VOLATILITY_VERY_LOW"
    VOLATILITY_LOW = "VOLATILITY_LOW"
    VOLATILITY_NORMAL = "VOLATILITY_NORMAL"
    VOLATILITY_HIGH = "VOLATILITY_HIGH"
    VOLATILITY_EXTREME = "VOLATILITY_EXTREME"
    VOLATILITY_COMPRESSION = "VOLATILITY_COMPRESSION"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    # location
    NEAR_STRONG_SUPPORT = "NEAR_STRONG_SUPPORT"
    NEAR_STRONG_RESISTANCE = "NEAR_STRONG_RESISTANCE"
    NEAR_ROUND_NUMBER = "NEAR_ROUND_NUMBER"
    EXTENDED_UP = "EXTENDED_UP"
    EXTENDED_DOWN = "EXTENDED_DOWN"
    OVEREXTENDED_UP = "OVEREXTENDED_UP"
    OVEREXTENDED_DOWN = "OVEREXTENDED_DOWN"
    RANGE_HIGH_LOCATION = "RANGE_HIGH_LOCATION"
    RANGE_LOW_LOCATION = "RANGE_LOW_LOCATION"
    # bias
    BIAS_LONG = "BIAS_LONG"
    BIAS_SHORT = "BIAS_SHORT"
    BIAS_NEUTRAL = "BIAS_NEUTRAL"
    BIAS_CONFLICT = "BIAS_CONFLICT"
    BIAS_INSUFFICIENT_EVIDENCE = "BIAS_INSUFFICIENT_EVIDENCE"
    BIAS_INSUFFICIENT_SEPARATION = "BIAS_INSUFFICIENT_SEPARATION"
    BIAS_REGIME_NON_DIRECTIONAL = "BIAS_REGIME_NON_DIRECTIONAL"
    BIAS_REGIME_CONTRADICTS = "BIAS_REGIME_CONTRADICTS"
    # data / engine
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    DATA_QUALITY_WARNING = "DATA_QUALITY_WARNING"
    # ---- Phase 1B: context and directional permission ----
    PRIMARY_STRUCTURE_BULLISH = "PRIMARY_STRUCTURE_BULLISH"
    PRIMARY_STRUCTURE_BEARISH = "PRIMARY_STRUCTURE_BEARISH"
    PRIMARY_STRUCTURE_RANGING = "PRIMARY_STRUCTURE_RANGING"
    PRIMARY_STRUCTURE_TRANSITIONAL = "PRIMARY_STRUCTURE_TRANSITIONAL"
    PRIMARY_STRUCTURE_NEUTRAL = "PRIMARY_STRUCTURE_NEUTRAL"
    PRIMARY_STRUCTURE_UNCLEAR = "PRIMARY_STRUCTURE_UNCLEAR"
    INTERMEDIATE_STRUCTURE_BULLISH = "INTERMEDIATE_STRUCTURE_BULLISH"
    INTERMEDIATE_STRUCTURE_BEARISH = "INTERMEDIATE_STRUCTURE_BEARISH"
    IMMEDIATE_STRUCTURE_BULLISH = "IMMEDIATE_STRUCTURE_BULLISH"
    IMMEDIATE_STRUCTURE_BEARISH = "IMMEDIATE_STRUCTURE_BEARISH"
    PROTECTED_LOW_BROKEN = "PROTECTED_LOW_BROKEN"
    PROTECTED_HIGH_BROKEN = "PROTECTED_HIGH_BROKEN"
    HEALTHY_PULLBACK = "HEALTHY_PULLBACK"
    DEEP_PULLBACK = "DEEP_PULLBACK"
    CHOPPY_PULLBACK = "CHOPPY_PULLBACK"
    STRUCTURAL_TRANSITION = "STRUCTURAL_TRANSITION"
    STRONG_BULLISH_DISPLACEMENT = "STRONG_BULLISH_DISPLACEMENT"
    STRONG_BEARISH_DISPLACEMENT = "STRONG_BEARISH_DISPLACEMENT"
    BULLISH_BREAK_ACCEPTED = "BULLISH_BREAK_ACCEPTED"
    BEARISH_BREAK_ACCEPTED = "BEARISH_BREAK_ACCEPTED"
    BULLISH_FAILED_BREAK = "BULLISH_FAILED_BREAK"
    BEARISH_FAILED_BREAK = "BEARISH_FAILED_BREAK"
    HIGH_QUALITY_BREAK = "HIGH_QUALITY_BREAK"
    BREAK_REJECTING = "BREAK_REJECTING"
    DOWNSIDE_LIQUIDITY_SWEEP = "DOWNSIDE_LIQUIDITY_SWEEP"
    UPSIDE_LIQUIDITY_SWEEP = "UPSIDE_LIQUIDITY_SWEEP"
    LIQUIDITY_BREAK_ACCEPTED = "LIQUIDITY_BREAK_ACCEPTED"
    BULLISH_ROLE_REVERSAL = "BULLISH_ROLE_REVERSAL"
    BEARISH_ROLE_REVERSAL = "BEARISH_ROLE_REVERSAL"
    STRONG_SUPPORT_NEARBY = "STRONG_SUPPORT_NEARBY"
    STRONG_RESISTANCE_NEARBY = "STRONG_RESISTANCE_NEARBY"
    DEMAND_ZONE_REACTION = "DEMAND_ZONE_REACTION"
    SUPPLY_ZONE_REACTION = "SUPPLY_ZONE_REACTION"
    SUFFICIENT_LONG_ROOM = "SUFFICIENT_LONG_ROOM"
    INSUFFICIENT_LONG_ROOM = "INSUFFICIENT_LONG_ROOM"
    SUFFICIENT_SHORT_ROOM = "SUFFICIENT_SHORT_ROOM"
    INSUFFICIENT_SHORT_ROOM = "INSUFFICIENT_SHORT_ROOM"
    TREND_EARLY = "TREND_EARLY"
    TREND_EXTENDED = "TREND_EXTENDED"
    EXHAUSTION_RISK = "EXHAUSTION_RISK"
    MOMENTUM_DETERIORATING = "MOMENTUM_DETERIORATING"
    COMPRESSION_PRESENT = "COMPRESSION_PRESENT"
    EXPANSION_FROM_COMPRESSION = "EXPANSION_FROM_COMPRESSION"
    HIGH_FALSE_BREAK_RISK = "HIGH_FALSE_BREAK_RISK"
    HIGH_CONTEXT_CONFLICT = "HIGH_CONTEXT_CONFLICT"
    LOW_CONTEXT_CONFLICT = "LOW_CONTEXT_CONFLICT"
    LOW_CONTEXT_QUALITY = "LOW_CONTEXT_QUALITY"
    CONFLICT_PRIMARY_VS_DISPLACEMENT = "CONFLICT_PRIMARY_VS_DISPLACEMENT"
    CONFLICT_TREND_VS_OPPOSING_LEVEL = "CONFLICT_TREND_VS_OPPOSING_LEVEL"
    CONFLICT_STRUCTURE_VS_FAILED_BREAK = "CONFLICT_STRUCTURE_VS_FAILED_BREAK"
    CONFLICT_TREND_VS_EXTENSION = "CONFLICT_TREND_VS_EXTENSION"
    CONFLICT_STRUCTURE_VS_SHOCK = "CONFLICT_STRUCTURE_VS_SHOCK"
    CONFLICT_HIERARCHY_DISAGREEMENT = "CONFLICT_HIERARCHY_DISAGREEMENT"
    CONFLICT_STRUCTURE_VS_TREND_ENGINE = "CONFLICT_STRUCTURE_VS_TREND_ENGINE"
    CONFLICT_OPPOSING_SWEEP = "CONFLICT_OPPOSING_SWEEP"
    CONFLICT_FALSE_BREAK_RISK = "CONFLICT_FALSE_BREAK_RISK"
    LONG_PERMISSION_GRANTED = "LONG_PERMISSION_GRANTED"
    SHORT_PERMISSION_GRANTED = "SHORT_PERMISSION_GRANTED"
    BOTH_DIRECTIONS_ALLOWED = "BOTH_DIRECTIONS_ALLOWED"
    ALL_DIRECTIONS_BLOCKED = "ALL_DIRECTIONS_BLOCKED"
    LONG_CONTEXT_INSUFFICIENT = "LONG_CONTEXT_INSUFFICIENT"
    SHORT_CONTEXT_INSUFFICIENT = "SHORT_CONTEXT_INSUFFICIENT"
    CONTEXT_QUALITY_TOO_LOW = "CONTEXT_QUALITY_TOO_LOW"
    CONTEXT_CONFLICT_TOO_HIGH = "CONTEXT_CONFLICT_TOO_HIGH"
    TWO_WAY_CONTEXT_OUTSIDE_RANGE = "TWO_WAY_CONTEXT_OUTSIDE_RANGE"
    BLOCKER_INVALID_DATA = "BLOCKER_INVALID_DATA"
    BLOCKER_STALE_DATA = "BLOCKER_STALE_DATA"
    BLOCKER_INSUFFICIENT_HISTORY = "BLOCKER_INSUFFICIENT_HISTORY"
    BLOCKER_UNRESOLVED_DATA_GAP = "BLOCKER_UNRESOLVED_DATA_GAP"
    BLOCKER_EXTREME_VOLATILITY_SHOCK = "BLOCKER_EXTREME_VOLATILITY_SHOCK"
    BLOCKER_SEVERE_CHOP = "BLOCKER_SEVERE_CHOP"
    BLOCKER_UNCLASSIFIABLE_STRUCTURE = "BLOCKER_UNCLASSIFIABLE_STRUCTURE"
    BLOCKER_CONTEXT_ERROR = "BLOCKER_CONTEXT_ERROR"
    # ---- Phase 1C: H1 setup intelligence ----
    H4_LONG_PERMISSION = "H4_LONG_PERMISSION"
    H4_SHORT_PERMISSION = "H4_SHORT_PERMISSION"
    H1_HEALTHY_PULLBACK = "H1_HEALTHY_PULLBACK"
    H1_DEEP_PULLBACK = "H1_DEEP_PULLBACK"
    H1_PULLBACK_STRUCTURE_DAMAGED = "H1_PULLBACK_STRUCTURE_DAMAGED"
    H1_COUNTER_MOMENTUM_DETERIORATING = "H1_COUNTER_MOMENTUM_DETERIORATING"
    H1_BULLISH_TRANSITION = "H1_BULLISH_TRANSITION"
    H1_BEARISH_TRANSITION = "H1_BEARISH_TRANSITION"
    H1_BULLISH_RECLAIM = "H1_BULLISH_RECLAIM"
    H1_BEARISH_RECLAIM = "H1_BEARISH_RECLAIM"
    H1_BULLISH_DISPLACEMENT = "H1_BULLISH_DISPLACEMENT"
    H1_BEARISH_DISPLACEMENT = "H1_BEARISH_DISPLACEMENT"
    H1_DOWNSIDE_SWEEP = "H1_DOWNSIDE_SWEEP"
    H1_UPSIDE_SWEEP = "H1_UPSIDE_SWEEP"
    H1_SUPPORT_REACTION = "H1_SUPPORT_REACTION"
    H1_RESISTANCE_REACTION = "H1_RESISTANCE_REACTION"
    H4_H1_ZONE_CONFLUENCE = "H4_H1_ZONE_CONFLUENCE"
    SUFFICIENT_UPSIDE_ROOM = "SUFFICIENT_UPSIDE_ROOM"
    SUFFICIENT_DOWNSIDE_ROOM = "SUFFICIENT_DOWNSIDE_ROOM"
    INSUFFICIENT_UPSIDE_ROOM = "INSUFFICIENT_UPSIDE_ROOM"
    INSUFFICIENT_DOWNSIDE_ROOM = "INSUFFICIENT_DOWNSIDE_ROOM"
    H1_HIGH_CHOP = "H1_HIGH_CHOP"
    H1_HIGH_CONFLICT = "H1_HIGH_CONFLICT"
    SETUP_WATCHING = "SETUP_WATCHING"
    SETUP_DEVELOPING = "SETUP_DEVELOPING"
    SETUP_QUALIFIED = "SETUP_QUALIFIED"
    SETUP_INVALIDATED = "SETUP_INVALIDATED"
    SETUP_EXPIRED = "SETUP_EXPIRED"
    SETUP_FAMILY_TREND_PULLBACK_CONTINUATION = "SETUP_FAMILY_TREND_PULLBACK_CONTINUATION"
    SETUP_FAMILY_BREAK_RETEST_CONTINUATION = "SETUP_FAMILY_BREAK_RETEST_CONTINUATION"
    SETUP_FAMILY_LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION = "SETUP_FAMILY_LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION"
    SETUP_FAMILY_COMPRESSION_EXPANSION_IN_H4_DIRECTION = "SETUP_FAMILY_COMPRESSION_EXPANSION_IN_H4_DIRECTION"
    H1_CONFLICT_H1_STRONGLY_OPPOSED = "H1_CONFLICT_H1_STRONGLY_OPPOSED"
    H1_CONFLICT_H1_HIGH_CHOP = "H1_CONFLICT_H1_HIGH_CHOP"
    H1_CONFLICT_TRIGGER_UNDER_BARRIER = "H1_CONFLICT_TRIGGER_UNDER_BARRIER"
    H1_CONFLICT_SWEEP_WITHOUT_FOLLOW_THROUGH = "H1_CONFLICT_SWEEP_WITHOUT_FOLLOW_THROUGH"
    H1_CONFLICT_HIGH_BREAKOUT_FAILURE_RISK = "H1_CONFLICT_HIGH_BREAKOUT_FAILURE_RISK"
    H1_CONFLICT_EXTREME_EXTENSION = "H1_CONFLICT_EXTREME_EXTENSION"
    H1_CONFLICT_H1_STRUCTURE_BROKEN_AGAINST = "H1_CONFLICT_H1_STRUCTURE_BROKEN_AGAINST"
    H1_BLOCKER_INVALID_H1_DATA = "H1_BLOCKER_INVALID_H1_DATA"
    H1_BLOCKER_STALE_H1_DATA = "H1_BLOCKER_STALE_H1_DATA"
    H1_BLOCKER_INSUFFICIENT_HISTORY = "H1_BLOCKER_INSUFFICIENT_HISTORY"
    H1_BLOCKER_EXTREME_H1_VOLATILITY = "H1_BLOCKER_EXTREME_H1_VOLATILITY"
    H1_BLOCKER_SEVERE_H1_CHOP = "H1_BLOCKER_SEVERE_H1_CHOP"
    H1_BLOCKER_H4_BLOCK_ALL = "H1_BLOCKER_H4_BLOCK_ALL"
    H1_BLOCKER_H4_PERMISSION_CONFLICT = "H1_BLOCKER_H4_PERMISSION_CONFLICT"
    H1_BLOCKER_UNRESOLVED_DATA_GAP = "H1_BLOCKER_UNRESOLVED_DATA_GAP"
    H1_BLOCKER_NO_H4_CONTEXT = "H1_BLOCKER_NO_H4_CONTEXT"
    H1_BLOCKER_STALE_H4_CONTEXT = "H1_BLOCKER_STALE_H4_CONTEXT"
    H1_BLOCKER_H4_ALIGNMENT_ERROR = "H1_BLOCKER_H4_ALIGNMENT_ERROR"
    H1_BLOCKER_H1_CONTEXT_ERROR = "H1_BLOCKER_H1_CONTEXT_ERROR"
    # Phase 1D - entry intelligence
    STRUCTURAL_CONFIRMATION = "STRUCTURAL_CONFIRMATION"
    DISPLACEMENT_CONFIRMED = "DISPLACEMENT_CONFIRMED"
    BREAK_RETEST_CONFIRMED = "BREAK_RETEST_CONFIRMED"
    SWEEP_RECLAIM_CONFIRMED = "SWEEP_RECLAIM_CONFIRMED"
    MOMENTUM_REACCELERATION = "MOMENTUM_REACCELERATION"
    COMPRESSION_EXPANSION_CONFIRMED = "COMPRESSION_EXPANSION_CONFIRMED"
    SIGNAL_FRESH = "SIGNAL_FRESH"
    SIGNAL_AGING = "SIGNAL_AGING"
    SIGNAL_STALE = "SIGNAL_STALE"
    CHASE_RISK_LOW = "CHASE_RISK_LOW"
    CHASE_RISK_MODERATE = "CHASE_RISK_MODERATE"
    CHASE_RISK_HIGH = "CHASE_RISK_HIGH"
    CHASE_RISK_EXTREME = "CHASE_RISK_EXTREME"
    ENTRY_NOT_EXTENDED = "ENTRY_NOT_EXTENDED"
    ENTRY_EXTENDED = "ENTRY_EXTENDED"
    ENTRY_OVEREXTENDED = "ENTRY_OVEREXTENDED"
    SPREAD_ACCEPTABLE = "SPREAD_ACCEPTABLE"
    SPREAD_ELEVATED = "SPREAD_ELEVATED"
    SPREAD_TOO_HIGH = "SPREAD_TOO_HIGH"
    SPREAD_UNKNOWN = "SPREAD_UNKNOWN"
    PRICE_DETERIORATION_ACCEPTABLE = "PRICE_DETERIORATION_ACCEPTABLE"
    PRICE_DETERIORATION_EXCESSIVE = "PRICE_DETERIORATION_EXCESSIVE"
    ENTRY_WINDOW_DETERIORATING = "ENTRY_WINDOW_DETERIORATING"
    ENTRY_WINDOW_EXPIRED = "ENTRY_WINDOW_EXPIRED"
    SUFFICIENT_REMAINING_ROOM = "SUFFICIENT_REMAINING_ROOM"
    INSUFFICIENT_REMAINING_ROOM = "INSUFFICIENT_REMAINING_ROOM"
    BARRIER_CLUSTER_NEARBY = "BARRIER_CLUSTER_NEARBY"
    H4_PERMISSION_VALID = "H4_PERMISSION_VALID"
    H4_PERMISSION_REVOKED = "H4_PERMISSION_REVOKED"
    SETUP_STILL_VALID = "SETUP_STILL_VALID"
    EXECUTION_CONDITIONS_NORMAL = "EXECUTION_CONDITIONS_NORMAL"
    EXECUTION_CONDITIONS_ABNORMAL = "EXECUTION_CONDITIONS_ABNORMAL"
    EXECUTION_GAP = "EXECUTION_GAP"
    NEWS_STATUS_UNKNOWN = "NEWS_STATUS_UNKNOWN"
    NEWS_CLEAR = "NEWS_CLEAR"
    NEWS_EVENT_NEARBY = "NEWS_EVENT_NEARBY"
    SLIPPAGE_UNKNOWN = "SLIPPAGE_UNKNOWN"
    WEEKEND_REOPEN_REVALIDATION = "WEEKEND_REOPEN_REVALIDATION"
    PRE_WEEKEND_CUTOFF = "PRE_WEEKEND_CUTOFF"
    ENTRY_CONFLICT_MOMENTUM_DETERIORATING = "ENTRY_CONFLICT_MOMENTUM_DETERIORATING"
    ENTRY_CONFLICT_PRICE_EXTENDED = "ENTRY_CONFLICT_PRICE_EXTENDED"
    ENTRY_CONFLICT_ROOM_COLLAPSED = "ENTRY_CONFLICT_ROOM_COLLAPSED"
    ENTRY_CONFLICT_POOR_SPREAD = "ENTRY_CONFLICT_POOR_SPREAD"
    ENTRY_CONFLICT_ABNORMAL_VOLATILITY = "ENTRY_CONFLICT_ABNORMAL_VOLATILITY"
    ENTRY_CONFLICT_H4_CONTEXT_WEAKENED = "ENTRY_CONFLICT_H4_CONTEXT_WEAKENED"
    ENTRY_CONFLICT_OPPOSING_BARRIER_CLUSTER = "ENTRY_CONFLICT_OPPOSING_BARRIER_CLUSTER"
    ENTRY_CONFLICT_TOO_HIGH = "ENTRY_CONFLICT_TOO_HIGH"
    ENTRY_QUALITY_TOO_LOW = "ENTRY_QUALITY_TOO_LOW"
    ENTRY_WAITING_FOR_CONFIRMATION = "ENTRY_WAITING_FOR_CONFIRMATION"
    ENTRY_CONFIRMING = "ENTRY_CONFIRMING"
    ENTRY_CANDIDATE_ACCEPTED = "ENTRY_CANDIDATE_ACCEPTED"
    ENTRY_DEFERRED = "ENTRY_DEFERRED"
    ENTRY_REJECTED = "ENTRY_REJECTED"
    ENTRY_EXPIRED = "ENTRY_EXPIRED"
    ENTRY_INVALIDATED = "ENTRY_INVALIDATED"
    ENTRY_SUPERSEDED = "ENTRY_SUPERSEDED"
    ENTRY_CONTEXT_ERROR = "ENTRY_CONTEXT_ERROR"
    NO_CONFIRMATION = "NO_CONFIRMATION"
    OPPOSING_STRUCTURE_BREAK = "OPPOSING_STRUCTURE_BREAK"
    CONFIRMATION_LEVEL_LOST = "CONFIRMATION_LEVEL_LOST"
    STRUCTURAL_INVALIDATION = "STRUCTURAL_INVALIDATION"
    # Phase 1E - trade construction
    STRUCTURAL_STOP_VALID = "STRUCTURAL_STOP_VALID"
    SWING_STOP_VALID = "SWING_STOP_VALID"
    ZONE_STOP_VALID = "ZONE_STOP_VALID"
    RECLAIM_FAILURE_STOP_VALID = "RECLAIM_FAILURE_STOP_VALID"
    BREAK_RETEST_STOP_VALID = "BREAK_RETEST_STOP_VALID"
    VOLATILITY_ADJUSTED_STOP_VALID = "VOLATILITY_ADJUSTED_STOP_VALID"
    STOP_BUFFER_APPLIED = "STOP_BUFFER_APPLIED"
    STOP_INSIDE_NOISE = "STOP_INSIDE_NOISE"
    STOP_DISTANCE_EXCESSIVE = "STOP_DISTANCE_EXCESSIVE"
    NO_STRUCTURAL_STOP = "NO_STRUCTURAL_STOP"
    INVALID_STOP_GEOMETRY = "INVALID_STOP_GEOMETRY"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    ZERO_RISK_DISTANCE = "ZERO_RISK_DISTANCE"
    STOP_ON_WRONG_SIDE_OF_ENTRY = "STOP_ON_WRONG_SIDE_OF_ENTRY"
    STOP_BELOW_MINIMUM_DISTANCE = "STOP_BELOW_MINIMUM_DISTANCE"
    STOP_INSIDE_BROKER_STOP_LEVEL = "STOP_INSIDE_BROKER_STOP_LEVEL"
    STOP_NOT_AT_SYMBOL_PRECISION = "STOP_NOT_AT_SYMBOL_PRECISION"
    NUMERIC_INVALID = "NUMERIC_INVALID"
    STRUCTURAL_TARGET_FOUND = "STRUCTURAL_TARGET_FOUND"
    H1_TARGET_FOUND = "H1_TARGET_FOUND"
    H4_TARGET_FOUND = "H4_TARGET_FOUND"
    TARGET_PATH_CLEAR = "TARGET_PATH_CLEAR"
    TARGET_PATH_BARRIERS_PRESENT = "TARGET_PATH_BARRIERS_PRESENT"
    TARGET_PATH_CONGESTED = "TARGET_PATH_CONGESTED"
    TARGET_UNREALISTIC = "TARGET_UNREALISTIC"
    ASYMMETRY_ACCEPTABLE = "ASYMMETRY_ACCEPTABLE"
    ASYMMETRY_INSUFFICIENT = "ASYMMETRY_INSUFFICIENT"
    COSTS_KNOWN = "COSTS_KNOWN"
    COSTS_UNKNOWN = "COSTS_UNKNOWN"
    COSTS_DEGRADE_RR = "COSTS_DEGRADE_RR"
    TRADE_PROPOSED = "TRADE_PROPOSED"
    TRADE_REJECTED = "TRADE_REJECTED"
    TRADE_INVALIDATED = "TRADE_INVALIDATED"
    TRADE_EXPIRED = "TRADE_EXPIRED"
    TRADE_CONFLICT_EXCELLENT_ENTRY_HUGE_STOP = "TRADE_CONFLICT_EXCELLENT_ENTRY_HUGE_STOP"
    TRADE_CONFLICT_GOOD_RR_UNREALISTIC_TARGET = "TRADE_CONFLICT_GOOD_RR_UNREALISTIC_TARGET"
    TRADE_CONFLICT_TARGET_DENSE_BARRIERS = "TRADE_CONFLICT_TARGET_DENSE_BARRIERS"
    TRADE_CONFLICT_STOP_IN_ORDINARY_NOISE = "TRADE_CONFLICT_STOP_IN_ORDINARY_NOISE"
    TRADE_CONFLICT_COSTS_DESTROY_ASYMMETRY = "TRADE_CONFLICT_COSTS_DESTROY_ASYMMETRY"
    TRADE_CONFLICT_H4_CONTEXT_WEAKENING = "TRADE_CONFLICT_H4_CONTEXT_WEAKENING"
    TRADE_CONFLICT_TOO_HIGH = "TRADE_CONFLICT_TOO_HIGH"
    TRADE_QUALITY_TOO_LOW = "TRADE_QUALITY_TOO_LOW"
    THESIS_FAILED_BEFORE_ENTRY = "THESIS_FAILED_BEFORE_ENTRY"
    NEWS_UNKNOWN = "NEWS_UNKNOWN"
    ENTRY_CANDIDATE_NOT_VALID = "ENTRY_CANDIDATE_NOT_VALID"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


REASON_DESCRIPTIONS: dict[str, str] = {
    "STRUCTURE_BULLISH": "confirmed swing sequence is Higher High / Higher Low",
    "STRUCTURE_BEARISH": "confirmed swing sequence is Lower High / Lower Low",
    "STRUCTURE_NEUTRAL": "confirmed swing sequence has no clear direction (contracting/equal swings)",
    "STRUCTURE_TRANSITIONAL": "structure is transitioning (conflicting swings or active change of character)",
    "STRUCTURE_UNCLEAR": "not enough confirmed swings to classify structure",
    "BOS_BULLISH": "recent bullish break of structure (with prevailing structure)",
    "BOS_BEARISH": "recent bearish break of structure (with prevailing structure)",
    "CHOCH_BULLISH": "recent bullish change of character (break against prevailing bearish structure)",
    "CHOCH_BEARISH": "recent bearish change of character (break against prevailing bullish structure)",
    "BREAK_REJECTED": "most recent structural break was INVALIDATED (closed back through the level before confirmation)",
    "BREAK_FAILED": "most recent structural break FAILED after confirmation (price closed decisively back through the level)",
    "HIGH_STRUCTURE_QUALITY": "structure is clean and persistent",
    "LOW_STRUCTURE_QUALITY": "structure is noisy, overlapping or unstable",
    "TREND_ALIGNED_BULLISH": "EMA trend engine is bullish",
    "TREND_ALIGNED_BEARISH": "EMA trend engine is bearish",
    "TREND_STRONG_BULLISH": "EMA trend engine is strongly bullish",
    "TREND_STRONG_BEARISH": "EMA trend engine is strongly bearish",
    "TREND_NEUTRAL": "EMA trend engine is neutral",
    "EMA_SLOPE_POSITIVE": "trend baseline slopes upward",
    "EMA_SLOPE_NEGATIVE": "trend baseline slopes downward",
    "STRUCTURE_TREND_CONFLICT": "swing structure and EMA trend point in opposite directions",
    "ADX_STRONG": "ADX indicates a directional environment",
    "ADX_WEAK": "ADX indicates a weak / non-directional environment",
    "DI_BULLISH": "+DI dominates -DI",
    "DI_BEARISH": "-DI dominates +DI",
    "MOMENTUM_BULLISH": "net momentum is bullish",
    "MOMENTUM_BEARISH": "net momentum is bearish",
    "MOMENTUM_ACCELERATING": "momentum is accelerating in its direction",
    "MOMENTUM_DECELERATING": "momentum is decelerating",
    "LOW_CHOP": "low chop / clean price action",
    "HIGH_CHOP": "high chop / ranging price action",
    "SEVERE_CHOP": "severe chop",
    "LOW_DIRECTIONAL_EFFICIENCY": "net displacement is small relative to distance travelled",
    "HIGH_DIRECTIONAL_EFFICIENCY": "price is moving efficiently in one direction",
    "VOLATILITY_VERY_LOW": "ATR percentile very low",
    "VOLATILITY_LOW": "ATR percentile low",
    "VOLATILITY_NORMAL": "ATR percentile normal",
    "VOLATILITY_HIGH": "ATR percentile high",
    "VOLATILITY_EXTREME": "ATR percentile extreme",
    "VOLATILITY_COMPRESSION": "short-term ATR contracting relative to long-term ATR",
    "VOLATILITY_EXPANSION": "short-term ATR expanding relative to long-term ATR",
    "VOLATILITY_SHOCK": "abnormal price behaviour detected on this bar",
    "NEAR_STRONG_SUPPORT": "price is near a strong H4 support zone",
    "NEAR_STRONG_RESISTANCE": "price is near a strong H4 resistance zone",
    "NEAR_ROUND_NUMBER": "price is near a JPY psychological round number",
    "EXTENDED_UP": "price is extended above its trend baseline",
    "EXTENDED_DOWN": "price is extended below its trend baseline",
    "OVEREXTENDED_UP": "price is extremely extended above its trend baseline",
    "OVEREXTENDED_DOWN": "price is extremely extended below its trend baseline",
    "RANGE_HIGH_LOCATION": "price is in the top of its medium-term H4 range",
    "RANGE_LOW_LOCATION": "price is in the bottom of its medium-term H4 range",
    "BIAS_LONG": "strategic H4 bias LONG (context only, not an entry signal)",
    "BIAS_SHORT": "strategic H4 bias SHORT (context only, not an entry signal)",
    "BIAS_NEUTRAL": "strategic H4 bias NEUTRAL",
    "BIAS_CONFLICT": "bullish and bearish evidence materially conflict",
    "BIAS_INSUFFICIENT_EVIDENCE": "dominant evidence below minimum",
    "BIAS_INSUFFICIENT_SEPARATION": "bullish/bearish evidence not separated enough",
    "BIAS_REGIME_NON_DIRECTIONAL": "regime is non-directional",
    "BIAS_REGIME_CONTRADICTS": "evidence direction contradicts the regime",
    "INSUFFICIENT_HISTORY": "warm-up period: not enough history for reliable features",
    "DATA_QUALITY_WARNING": "input bar carries data-quality warnings",
    "PRIMARY_STRUCTURE_BULLISH": "primary (full swing history) structure is bullish",
    "PRIMARY_STRUCTURE_BEARISH": "primary (full swing history) structure is bearish",
    "PRIMARY_STRUCTURE_RANGING": "primary structure is ranging",
    "PRIMARY_STRUCTURE_TRANSITIONAL": "primary structure is transitioning",
    "PRIMARY_STRUCTURE_NEUTRAL": "primary structure is neutral / mixed",
    "PRIMARY_STRUCTURE_UNCLEAR": "primary structure cannot be classified (insufficient confirmed swings)",
    "INTERMEDIATE_STRUCTURE_BULLISH": "intermediate structure is bullish",
    "INTERMEDIATE_STRUCTURE_BEARISH": "intermediate structure is bearish",
    "IMMEDIATE_STRUCTURE_BULLISH": "immediate structure is bullish",
    "IMMEDIATE_STRUCTURE_BEARISH": "immediate structure is bearish",
    "PROTECTED_LOW_BROKEN": "close below the protected swing low of a bullish structure",
    "PROTECTED_HIGH_BROKEN": "close above the protected swing high of a bearish structure",
    "HEALTHY_PULLBACK": "counter-structure correction is healthy (moderate depth, not choppy)",
    "DEEP_PULLBACK": "counter-structure correction is deep",
    "CHOPPY_PULLBACK": "counter-structure correction is choppy / overlapping",
    "STRUCTURAL_TRANSITION": "structural layers indicate a transition",
    "STRONG_BULLISH_DISPLACEMENT": "strong multi-candle bullish displacement (recent)",
    "STRONG_BEARISH_DISPLACEMENT": "strong multi-candle bearish displacement (recent)",
    "BULLISH_BREAK_ACCEPTED": "recent bullish structural break is being accepted",
    "BEARISH_BREAK_ACCEPTED": "recent bearish structural break is being accepted",
    "BULLISH_FAILED_BREAK": "a recent bullish structural break FAILED or was invalidated",
    "BEARISH_FAILED_BREAK": "a recent bearish structural break FAILED or was invalidated",
    "HIGH_QUALITY_BREAK": "most recent structural break is high quality",
    "BREAK_REJECTING": "price is rejecting the most recent structural break",
    "DOWNSIDE_LIQUIDITY_SWEEP": "potential downside liquidity reference swept and rejected (bullish context)",
    "UPSIDE_LIQUIDITY_SWEEP": "potential upside liquidity reference swept and rejected (bearish context)",
    "LIQUIDITY_BREAK_ACCEPTED": "a liquidity reference was broken and the move accepted (continuation, not a sweep)",
    "BULLISH_ROLE_REVERSAL": "former resistance zone confirmed as support (break, acceptance, retest, reaction)",
    "BEARISH_ROLE_REVERSAL": "former support zone confirmed as resistance",
    "STRONG_SUPPORT_NEARBY": "strong support zone close below price",
    "STRONG_RESISTANCE_NEARBY": "strong resistance zone close above price",
    "DEMAND_ZONE_REACTION": "price reacted from a bullish displacement-origin zone",
    "SUPPLY_ZONE_REACTION": "price reacted from a bearish displacement-origin zone",
    "SUFFICIENT_LONG_ROOM": "sufficient structural room above price",
    "INSUFFICIENT_LONG_ROOM": "insufficient structural room above price",
    "SUFFICIENT_SHORT_ROOM": "sufficient structural room below price",
    "INSUFFICIENT_SHORT_ROOM": "insufficient structural room below price",
    "TREND_EARLY": "current directional move is early",
    "TREND_EXTENDED": "current directional move is extended",
    "EXHAUSTION_RISK": "extended/mature move with deteriorating momentum",
    "MOMENTUM_DETERIORATING": "directional quality is deteriorating (smaller impulses, deeper corrections, slower)",
    "COMPRESSION_PRESENT": "structural compression present (no directional implication)",
    "EXPANSION_FROM_COMPRESSION": "directional expansion out of a compression episode",
    "HIGH_FALSE_BREAK_RISK": "most recent break carries high false-break risk",
    "HIGH_CONTEXT_CONFLICT": "context evidence materially conflicts",
    "LOW_CONTEXT_CONFLICT": "context evidence is coherent",
    "LOW_CONTEXT_QUALITY": "context quality is low",
    "CONFLICT_PRIMARY_VS_DISPLACEMENT": "primary structure opposed by strong recent displacement",
    "CONFLICT_TREND_VS_OPPOSING_LEVEL": "directional structure faces a strong opposing level immediately ahead",
    "CONFLICT_STRUCTURE_VS_FAILED_BREAK": "structure opposed by a failed break in its own direction",
    "CONFLICT_TREND_VS_EXTENSION": "directional structure is extremely extended",
    "CONFLICT_STRUCTURE_VS_SHOCK": "strong structure during a volatility shock",
    "CONFLICT_HIERARCHY_DISAGREEMENT": "primary and intermediate structure disagree",
    "CONFLICT_STRUCTURE_VS_TREND_ENGINE": "swing structure and EMA trend engine disagree",
    "CONFLICT_OPPOSING_SWEEP": "recent liquidity sweep opposes the structural direction",
    "CONFLICT_FALSE_BREAK_RISK": "latest break in the structural direction carries high false-break risk",
    "LONG_PERMISSION_GRANTED": "H1 may SEARCH for long setups (context permission, not a trade signal)",
    "SHORT_PERMISSION_GRANTED": "H1 may SEARCH for short setups (context permission, not a trade signal)",
    "BOTH_DIRECTIONS_ALLOWED": "two-way context: H1 may search both directions",
    "ALL_DIRECTIONS_BLOCKED": "no direction permitted",
    "LONG_CONTEXT_INSUFFICIENT": "long context score, room or quality below requirement",
    "SHORT_CONTEXT_INSUFFICIENT": "short context score, room or quality below requirement",
    "CONTEXT_QUALITY_TOO_LOW": "context quality below the permission requirement",
    "CONTEXT_CONFLICT_TOO_HIGH": "context conflict above the permission limit",
    "TWO_WAY_CONTEXT_OUTSIDE_RANGE": "both directions qualified outside a range regime (treated as conflicting)",
    "BLOCKER_INVALID_DATA": "hard blocker: invalid input data",
    "BLOCKER_STALE_DATA": "hard blocker: data is stale",
    "BLOCKER_INSUFFICIENT_HISTORY": "hard blocker: insufficient history",
    "BLOCKER_UNRESOLVED_DATA_GAP": "hard blocker: recent unresolved data gap",
    "BLOCKER_EXTREME_VOLATILITY_SHOCK": "hard blocker: extreme volatility shock",
    "BLOCKER_SEVERE_CHOP": "hard blocker: severe chop",
    "BLOCKER_UNCLASSIFIABLE_STRUCTURE": "hard blocker: structure cannot be classified",
    "BLOCKER_CONTEXT_ERROR": "hard blocker: context computation failed (fail-safe)",
    "H4_LONG_PERMISSION": "H4 permits the H1 engine to search long",
    "H4_SHORT_PERMISSION": "H4 permits the H1 engine to search short",
    "H1_HEALTHY_PULLBACK": "H1 correction against the permitted direction is shallow/healthy",
    "H1_DEEP_PULLBACK": "H1 correction is deep",
    "H1_PULLBACK_STRUCTURE_DAMAGED": "H1 correction threatens or broke the impulse origin",
    "H1_COUNTER_MOMENTUM_DETERIORATING": "counter-direction H1 momentum is fading during the correction",
    "H1_BULLISH_TRANSITION": "measurable H1 structural transition upward (break out of bearish/neutral structure after a counter move)",
    "H1_BEARISH_TRANSITION": "measurable H1 structural transition downward",
    "H1_BULLISH_RECLAIM": "a lost H1 level was reclaimed upward",
    "H1_BEARISH_RECLAIM": "a reclaimed H1 level was lost downward",
    "H1_BULLISH_DISPLACEMENT": "strong multi-candle bullish H1 displacement",
    "H1_BEARISH_DISPLACEMENT": "strong multi-candle bearish H1 displacement",
    "H1_DOWNSIDE_SWEEP": "potential H1 downside liquidity reference swept and rejected",
    "H1_UPSIDE_SWEEP": "potential H1 upside liquidity reference swept and rejected",
    "H1_SUPPORT_REACTION": "quantified bullish rejection at an H1/H4 level",
    "H1_RESISTANCE_REACTION": "quantified bearish rejection at an H1/H4 level",
    "H4_H1_ZONE_CONFLUENCE": "price interacting with a clustered H4 + H1 level (counted once)",
    "SUFFICIENT_UPSIDE_ROOM": "sufficient H1/H4 room above price",
    "SUFFICIENT_DOWNSIDE_ROOM": "sufficient H1/H4 room below price",
    "INSUFFICIENT_UPSIDE_ROOM": "insufficient H1/H4 room above price",
    "INSUFFICIENT_DOWNSIDE_ROOM": "insufficient H1/H4 room below price",
    "H1_HIGH_CHOP": "H1 chop is high",
    "H1_HIGH_CONFLICT": "H1 setup evidence materially conflicts",
    "SETUP_WATCHING": "a setup premise appeared; setup is being watched",
    "SETUP_DEVELOPING": "setup score reached the developing threshold",
    "SETUP_QUALIFIED": "setup met every qualification requirement (NOT a trade)",
    "SETUP_INVALIDATED": "setup invalidated (reason recorded)",
    "SETUP_EXPIRED": "setup expired (reason recorded)",
    "SETUP_FAMILY_TREND_PULLBACK_CONTINUATION": "best eligible archetype: trend pullback continuation",
    "SETUP_FAMILY_BREAK_RETEST_CONTINUATION": "best eligible archetype: break + retest continuation",
    "SETUP_FAMILY_LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION": "best eligible archetype: liquidity sweep reversal in the H4 direction",
    "SETUP_FAMILY_COMPRESSION_EXPANSION_IN_H4_DIRECTION": "best eligible archetype: compression -> expansion in the H4 direction",
    "H1_CONFLICT_H1_STRONGLY_OPPOSED": "setup conflict: strong opposing H1 displacement",
    "H1_CONFLICT_H1_HIGH_CHOP": "setup conflict: high H1 chop",
    "H1_CONFLICT_TRIGGER_UNDER_BARRIER": "setup conflict: trigger directly under/over an opposing barrier",
    "H1_CONFLICT_SWEEP_WITHOUT_FOLLOW_THROUGH": "setup conflict: sweep without follow-through",
    "H1_CONFLICT_HIGH_BREAKOUT_FAILURE_RISK": "setup conflict: breakout with high failure risk",
    "H1_CONFLICT_EXTREME_EXTENSION": "setup conflict: setup after extreme extension",
    "H1_CONFLICT_H1_STRUCTURE_BROKEN_AGAINST": "setup conflict: H1 structure broken against the setup direction",
    "H1_BLOCKER_INVALID_H1_DATA": "H1 blocker: invalid H1 data",
    "H1_BLOCKER_STALE_H1_DATA": "H1 blocker: stale H1 data",
    "H1_BLOCKER_INSUFFICIENT_HISTORY": "H1 blocker: insufficient H1 or H4 history",
    "H1_BLOCKER_EXTREME_H1_VOLATILITY": "H1 blocker: extreme H1 volatility",
    "H1_BLOCKER_SEVERE_H1_CHOP": "H1 blocker: severe H1 chop",
    "H1_BLOCKER_H4_BLOCK_ALL": "H1 blocker: H4 permission is BLOCK_ALL",
    "H1_BLOCKER_H4_PERMISSION_CONFLICT": "H1 blocker (side): H4 does not permit this direction",
    "H1_BLOCKER_UNRESOLVED_DATA_GAP": "H1 blocker: recent unresolved data gap",
    "H1_BLOCKER_NO_H4_CONTEXT": "H1 blocker: no completed H4 bar available yet",
    "H1_BLOCKER_STALE_H4_CONTEXT": "H1 blocker: latest completed H4 context is too old (missing H4 bars / gap)",
    "H1_BLOCKER_H4_ALIGNMENT_ERROR": "H1 blocker: H4/H1 alignment could not be proven point-in-time safe",
    "H1_BLOCKER_H1_CONTEXT_ERROR": "H1 blocker: H1 evaluation failed (fail-safe)",
    "STRUCTURAL_CONFIRMATION": "entry confirmed by a meaningful H1 structural break in the permitted direction",
    "DISPLACEMENT_CONFIRMED": "entry confirmed by multi-candle directional displacement",
    "BREAK_RETEST_CONFIRMED": "entry confirmed by a controlled retest of a broken level and a directional response",
    "SWEEP_RECLAIM_CONFIRMED": "entry confirmed by a liquidity sweep, reclaim and directional follow-through",
    "MOMENTUM_REACCELERATION": "entry confirmed by fading counter momentum and resumed permitted-direction momentum",
    "COMPRESSION_EXPANSION_CONFIRMED": "entry confirmed by measurable directional expansion out of compression",
    "SIGNAL_FRESH": "entry signal is fresh",
    "SIGNAL_AGING": "entry signal is aging",
    "SIGNAL_STALE": "entry signal is stale (too old or price already moved)",
    "CHASE_RISK_LOW": "low chase risk at the executable reference",
    "CHASE_RISK_MODERATE": "moderate chase risk at the executable reference",
    "CHASE_RISK_HIGH": "high chase risk: price already travelled too far from the setup/confirmation",
    "CHASE_RISK_EXTREME": "extreme chase risk",
    "ENTRY_NOT_EXTENDED": "entry would not occur after excessive directional extension",
    "ENTRY_EXTENDED": "entry would occur after notable directional extension",
    "ENTRY_OVEREXTENDED": "entry would occur after excessive directional extension (poor timing)",
    "SPREAD_ACCEPTABLE": "spread known and normal",
    "SPREAD_ELEVATED": "spread elevated",
    "SPREAD_TOO_HIGH": "spread demonstrably unacceptable (HIGH/EXTREME)",
    "SPREAD_UNKNOWN": "spread unavailable - not assumed to be zero",
    "PRICE_DETERIORATION_ACCEPTABLE": "executable reference close to the signal price",
    "PRICE_DETERIORATION_EXCESSIVE": "executable reference materially worse than the signal price",
    "ENTRY_WINDOW_DETERIORATING": "entry window still open but the executable price is deteriorating",
    "ENTRY_WINDOW_EXPIRED": "entry window elapsed (time, bars or price movement)",
    "SUFFICIENT_REMAINING_ROOM": "sufficient room to the stacked H1/H4 opposing barriers at the executable reference",
    "INSUFFICIENT_REMAINING_ROOM": "insufficient room to the nearest opposing barrier cluster at the executable reference",
    "BARRIER_CLUSTER_NEARBY": "a multi-member or multi-timeframe opposing barrier cluster is nearby",
    "H4_PERMISSION_VALID": "H4 permission still includes the candidate direction",
    "H4_PERMISSION_REVOKED": "H4 permission changed (BLOCK_ALL, reversal or stale context) - candidate invalidated",
    "SETUP_STILL_VALID": "the Phase 1C setup is still QUALIFIED at decision time",
    "EXECUTION_CONDITIONS_NORMAL": "no abnormal execution conditions detected",
    "EXECUTION_CONDITIONS_ABNORMAL": "abnormal execution conditions (extreme candle, volatility, gap or spread explosion)",
    "EXECUTION_GAP": "material gap between the confirmation close and the first executable price",
    "NEWS_STATUS_UNKNOWN": "no economic-calendar provider: news status UNKNOWN (no events invented)",
    "NEWS_CLEAR": "no relevant scheduled event near the execution time",
    "NEWS_EVENT_NEARBY": "relevant scheduled GBP/JPY event near the execution time",
    "SLIPPAGE_UNKNOWN": "actual live slippage is unknown (no model configured)",
    "WEEKEND_REOPEN_REVALIDATION": "market closure/reopen: pre-closure evidence is not carried over without revalidation",
    "PRE_WEEKEND_CUTOFF": "late-Friday confirmation deferred (never accepted into the weekend)",
    "ENTRY_CONFLICT_MOMENTUM_DETERIORATING": "entry conflict: confirmation exists but H1 momentum opposes it",
    "ENTRY_CONFLICT_PRICE_EXTENDED": "entry conflict: confirmation exists but price is extended",
    "ENTRY_CONFLICT_ROOM_COLLAPSED": "entry conflict: remaining room has collapsed",
    "ENTRY_CONFLICT_POOR_SPREAD": "entry conflict: spread is poor",
    "ENTRY_CONFLICT_ABNORMAL_VOLATILITY": "entry conflict: abnormal volatility at execution time",
    "ENTRY_CONFLICT_H4_CONTEXT_WEAKENED": "entry conflict: H4 context weakened since qualification",
    "ENTRY_CONFLICT_OPPOSING_BARRIER_CLUSTER": "entry conflict: dense opposing barrier cluster nearby",
    "ENTRY_CONFLICT_TOO_HIGH": "entry conflict above the configured limit",
    "ENTRY_QUALITY_TOO_LOW": "entry quality below the configured minimum",
    "ENTRY_WAITING_FOR_CONFIRMATION": "qualified setup waiting for entry confirmation",
    "ENTRY_CONFIRMING": "entry confirmation developing / complete and awaiting the first executable price",
    "ENTRY_CANDIDATE_ACCEPTED": "executable entry candidate accepted (NOT an order)",
    "ENTRY_DEFERRED": "entry decision deferred to the next executable opportunity",
    "ENTRY_REJECTED": "entry candidate rejected (reason recorded, kept as counterfactual)",
    "ENTRY_EXPIRED": "entry candidate expired",
    "ENTRY_INVALIDATED": "entry candidate invalidated",
    "ENTRY_SUPERSEDED": "entry candidate superseded by a newer qualified setup",
    "ENTRY_CONTEXT_ERROR": "entry evaluation failed (fail-safe: candidates invalidated)",
    "NO_CONFIRMATION": "qualified setup never produced a valid entry confirmation",
    "OPPOSING_STRUCTURE_BREAK": "H1 structure broke against the candidate direction",
    "CONFIRMATION_LEVEL_LOST": "price closed back through the confirmation reference",
    "STRUCTURAL_INVALIDATION": "structural premise of the candidate failed",
    "STRUCTURAL_STOP_VALID": "stop at the setup's structural invalidation level (buffered)",
    "SWING_STOP_VALID": "stop beyond the most recent confirmed H1 swing (buffered)",
    "ZONE_STOP_VALID": "stop beyond the far edge of the nearest H1 zone (buffered)",
    "RECLAIM_FAILURE_STOP_VALID": "stop beyond the swept liquidity extreme (reclaim failure)",
    "BREAK_RETEST_STOP_VALID": "stop beyond the retest structure of the broken level",
    "VOLATILITY_ADJUSTED_STOP_VALID": "preferred reference sat inside noise: stop beyond the next structure (never closer)",
    "STOP_BUFFER_APPLIED": "volatility-aware noise buffer (and trigger-side spread) applied beyond the structural level",
    "STOP_INSIDE_NOISE": "stop sits inside ordinary observed GBPJPY noise",
    "STOP_DISTANCE_EXCESSIVE": "structural stop requires an excessive distance for current volatility",
    "NO_STRUCTURAL_STOP": "no structural invalidation reference on the correct side of the entry",
    "INVALID_STOP_GEOMETRY": "stop failed sanity checks (geometry, precision, minimum distance)",
    "NON_FINITE_VALUE": "non-finite value in stop/target mathematics",
    "ZERO_RISK_DISTANCE": "stop equals the entry (zero risk distance)",
    "STOP_ON_WRONG_SIDE_OF_ENTRY": "stop is not below a long / above a short entry",
    "STOP_BELOW_MINIMUM_DISTANCE": "stop closer than the configured minimum distance",
    "STOP_INSIDE_BROKER_STOP_LEVEL": "stop closer than the known broker stop level",
    "STOP_NOT_AT_SYMBOL_PRECISION": "stop price not at symbol precision",
    "NUMERIC_INVALID": "numeric failure (NaN, infinity, invalid geometry) - rejected safely",
    "STRUCTURAL_TARGET_FOUND": "at least one structural target exists beyond the entry",
    "H1_TARGET_FOUND": "a target built from H1 structure",
    "H4_TARGET_FOUND": "a target built from H4 structure",
    "TARGET_PATH_CLEAR": "no barrier between the entry and the primary target",
    "TARGET_PATH_BARRIERS_PRESENT": "barriers exist between the entry and the primary target (not congested)",
    "TARGET_PATH_CONGESTED": "path to every realistic target is congested by barriers",
    "TARGET_UNREALISTIC": "no structural target is realistically reachable",
    "ASYMMETRY_ACCEPTABLE": "primary target provides the configured minimum estimated net R",
    "ASYMMETRY_INSUFFICIENT": "natural structure does not provide the configured minimum R (not manufactured)",
    "COSTS_KNOWN": "all transaction-cost components known",
    "COSTS_UNKNOWN": "some transaction costs unknown: conservative assumptions applied and flagged",
    "COSTS_DEGRADE_RR": "gross R passes but costs push estimated net R below the minimum",
    "TRADE_PROPOSED": "proposed trade (NOT an order)",
    "TRADE_REJECTED": "trade construction rejected (retained for research)",
    "TRADE_INVALIDATED": "trade construction invalidated",
    "TRADE_EXPIRED": "trade construction expired",
    "TRADE_CONFLICT_EXCELLENT_ENTRY_HUGE_STOP": "trade conflict: excellent entry but a wide stop",
    "TRADE_CONFLICT_GOOD_RR_UNREALISTIC_TARGET": "trade conflict: good R relies on a poorly reachable target",
    "TRADE_CONFLICT_TARGET_DENSE_BARRIERS": "trade conflict: dense barriers before the target",
    "TRADE_CONFLICT_STOP_IN_ORDINARY_NOISE": "trade conflict: stop close to ordinary noise",
    "TRADE_CONFLICT_COSTS_DESTROY_ASYMMETRY": "trade conflict: costs destroy the asymmetry",
    "TRADE_CONFLICT_H4_CONTEXT_WEAKENING": "trade conflict: H4 context weakened since qualification",
    "TRADE_CONFLICT_TOO_HIGH": "trade construction conflict above the limit",
    "TRADE_QUALITY_TOO_LOW": "trade construction quality below the minimum",
    "THESIS_FAILED_BEFORE_ENTRY": "thesis re-check failed immediately before finalising",
    "NEWS_UNKNOWN": "news status unknown - no safety is assumed",
    "ENTRY_CANDIDATE_NOT_VALID": "the entry candidate was not valid at the proposal time",
}


def describe(code: str | ReasonCode) -> str:
    key = code.value if isinstance(code, ReasonCode) else str(code)
    return REASON_DESCRIPTIONS.get(key, key)


def codes(*items: ReasonCode) -> list[str]:
    return [c.value for c in items]
