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
}


def describe(code: str | ReasonCode) -> str:
    key = code.value if isinstance(code, ReasonCode) else str(code)
    return REASON_DESCRIPTIONS.get(key, key)


def codes(*items: ReasonCode) -> list[str]:
    return [c.value for c in items]
