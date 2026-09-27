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
    "BREAK_REJECTED": "most recent structural break was rejected (closed back through the level)",
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
}


def describe(code: str | ReasonCode) -> str:
    key = code.value if isinstance(code, ReasonCode) else str(code)
    return REASON_DESCRIPTIONS.get(key, key)


def codes(*items: ReasonCode) -> list[str]:
    return [c.value for c in items]
