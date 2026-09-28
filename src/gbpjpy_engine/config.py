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
