"""Phase 1C H1 configuration.

Reuses the Phase 1A/1B section types unchanged (same meaning, same documented
defaults) with ``data.timeframe_minutes = 60``, and adds H1-specific sections.
Every H1-specific parameter is documented; all values are untested baselines,
not fitted to profit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..config import (
    AdxConfig, BreakoutContextConfig, CandleConfig, ChopConfig, CompressionConfig, DataConfig, DisplacementConfig,
    EfficiencyConfig, EngineConfig, ExtensionConfig, HierarchyConfig, LegConfig, LevelConfig, LiquidityConfig,
    LocationContextConfig,
    MomentumConfig, OriginZoneConfig, RangeLocationConfig, RoomConfig, RoundNumberConfig, SessionConfig,
    ShockConfig, StructureConfig, SwingConfig, TrendConfig, VolatilityConfig, _p,
)


@dataclass(frozen=True)
class H1AlignmentConfig:
    max_context_age_hours: float = _p(
        8.0,
        "If (H1 bar close - latest completed H4 close) exceeds this, the attached H4 context is STALE and H1 is blocked. "
        "Normal ages are 0-3h; larger values mean missing H4 bars or a session gap.",
    )


@dataclass(frozen=True)
class H1PullbackConfig:
    min_impulse_atr: float = _p(1.5, "Minimum size (H1 ATR) of the reference impulse a pullback is measured against.")
    impulse_lookback_legs: int = _p(6, "Most recent confirmed H1 legs searched for the reference impulse.")
    no_pullback_pct: float = _p(10.0, "Retracement below this % of the reference impulse = NO_PULLBACK.")
    shallow_pct: float = _p(30.0, "Retracement below this % = SHALLOW_PULLBACK (descriptive band).")
    healthy_pct: float = _p(70.0, "Retracement below this % = HEALTHY_PULLBACK (descriptive band).")
    deep_pct: float = _p(100.0, "Retracement below this % = DEEP_PULLBACK; at/above = STRUCTURE_THREATENING.")
    failed_buffer_atr: float = _p(
        0.25, "A close beyond the impulse origin by this many ATR = FAILED_PULLBACK_CONTEXT (structure broken)."
    )
    ideal_depth_low: float = _p(35.0, "Lower edge of the depth band given full depth credit in pullback quality (untested).")
    ideal_depth_high: float = _p(65.0, "Upper edge of the depth band given full depth credit in pullback quality (untested).")
    max_velocity_atr: float = _p(0.8, "Correction velocity (ATR/bar) at which the controlled-velocity component reaches 0.")


@dataclass(frozen=True)
class H1TransitionConfig:
    recent_bars: int = _p(12, "A structural transition stays 'recent' context for this many H1 bars.")
    min_counter_leg_atr: float = _p(1.0, "A prior counter-direction move of at least this many ATR must precede a transition.")
    stall_bars: int = _p(3, "Bars without a new counter-direction extreme that count as 'counter structure stopped progressing'.")


@dataclass(frozen=True)
class H1ReclaimConfig:
    recent_bars: int = _p(12, "A reclaim stays 'recent' context for this many H1 bars.")


@dataclass(frozen=True)
class H1RejectionConfig:
    level_tolerance_atr: float = _p(0.3, "A wick within this many ATR of an H1/H4 level or liquidity reference counts as a level interaction.")
    significance_lookback: int = _p(200, "Trailing bars for the historical percentile of rejection-wick size.")
    recent_bars: int = _p(3, "Rejections from the last N bars (incl. current) are considered, with follow-through measured to now.")


@dataclass(frozen=True)
class H1ConfluenceConfig:
    cluster_tolerance_atr: float = _p(0.5, "H4 and H1 levels within this many H1 ATR are merged into ONE confluence cluster.")
    near_atr: float = _p(1.0, "A confluence cluster within this many H1 ATR of the close counts as 'interacting'.")


@dataclass(frozen=True)
class SetupConfig:
    family_event_bars: int = _p(12, "Breaks, sweeps and expansions older than this many H1 bars cannot anchor a new setup.")
    developing_score: float = _p(45.0, "Setup score at/above which a WATCHING setup becomes DEVELOPING.")
    qualify_score: float = _p(60.0, "Setup score required for QUALIFIED (untested baseline).")
    qualify_min_confidence: float = _p(50.0, "Setup confidence required for QUALIFIED.")
    max_conflict: float = _p(50.0, "h1_setup_conflict_score above which a setup cannot qualify.")
    min_room_score: float = _p(30.0, "H1 room score in the setup direction required to qualify (and below which room 'disappears').")
    min_trigger_score: float = _p(
        50.0, "Family trigger evidence (transition / reaction / rejection / expansion) required to qualify."
    )
    watch_expiry_bars: int = _p(24, "A setup that has not qualified within this many bars of first being watched EXPIRES.")
    qualified_expiry_bars: int = _p(12, "A QUALIFIED setup is considered stale and EXPIRES after this many bars.")
    max_travel_atr: float = _p(
        3.0, "A setup EXPIRES if price has travelled this many ATR in the setup direction since it started (move left without it)."
    )
    opposing_displacement_invalidate: float = _p(70.0, "Opposing H1 displacement at/above this INVALIDATES an active setup.")
    counterfactual_min_score: float = _p(
        45.0, "Setups that never qualified but reached this peak score are kept in the counterfactual log."
    )
    extension_chase_score: float = _p(
        85.0, "H1 extension_score at/above which (in the setup direction) the setup is treated as chasing (conflict + penalty)."
    )


@dataclass(frozen=True)
class H1ScoringConfig:
    w_h4_context: float = _p(0.15, "Setup-quality family weight: H4_CONTEXT (H4 context score and permission confidence).")
    w_h1_structure: float = _p(0.15, "Family weight: H1_STRUCTURE (hierarchy + transition).")
    w_pullback: float = _p(0.10, "Family weight: PULLBACK_QUALITY.")
    w_location: float = _p(0.10, "Family weight: LOCATION.")
    w_displacement: float = _p(0.10, "Family weight: DISPLACEMENT.")
    w_momentum: float = _p(0.10, "Family weight: MOMENTUM.")
    w_liquidity: float = _p(0.05, "Family weight: LIQUIDITY_CONTEXT.")
    w_market_quality: float = _p(0.10, "Family weight: MARKET_QUALITY (chop, efficiency, volatility).")
    w_room: float = _p(0.10, "Family weight: ROOM_TO_MOVE.")
    w_conflict: float = _p(0.05, "Family weight: CONFLICT (100 - conflict).")
    family_blend: float = _p(
        0.5, "setup_score = family_blend x setup-family (archetype) score + (1 - family_blend) x quality-family score."
    )
    conflict_penalty: float = _p(0.5, "setup_score *= 1 - conflict_penalty x conflict/100.")


@dataclass(frozen=True)
class H1BlockerConfig:
    block_invalid_data: bool = _p(True, "INVALID_H1_DATA blocker when the H1 bar carries ERROR-level data issues.")
    stale_after_hours: float = _p(3.0, "STALE_H1_DATA (latest bar only, when as_of is supplied): as_of - H1 close exceeds this.")
    gap_block_bars: int = _p(3, "UNRESOLVED_DATA_GAP for this many H1 bars after a missing-bar / gap / grid-shift flag (0 disables).")
    shock_block_severity: float = _p(85.0, "EXTREME_H1_VOLATILITY when H1 shock_severity is at/above this.")
    block_extreme_volatility_regime: bool = _p(True, "EXTREME_H1_VOLATILITY also when the H1 volatility regime is 'extreme'.")
    severe_chop_score: float = _p(72.0, "SEVERE_H1_CHOP when h1_chop_score is at/above this.")


def _h1_data() -> DataConfig:
    return DataConfig(timeframe_minutes=60)


def _h1_hierarchy() -> HierarchyConfig:
    return HierarchyConfig(primary_swings=12, intermediate_swings=8, immediate_swings=4)


@dataclass(frozen=True)
class H1Config:
    # ---- reused Phase 1A sections (H1 bars) ----
    data: DataConfig = field(default_factory=_h1_data)
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
    # ---- reused Phase 1B sections ----
    hierarchy: HierarchyConfig = field(default_factory=_h1_hierarchy)
    legs: LegConfig = field(default_factory=LegConfig)
    displacement: DisplacementConfig = field(default_factory=DisplacementConfig)
    breakout_context: BreakoutContextConfig = field(default_factory=BreakoutContextConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    origin_zones: OriginZoneConfig = field(default_factory=OriginZoneConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    room: RoomConfig = field(default_factory=RoomConfig)
    location_context: LocationContextConfig = field(default_factory=LocationContextConfig)
    # ---- Phase 1C sections ----
    alignment: H1AlignmentConfig = field(default_factory=H1AlignmentConfig)
    pullback: H1PullbackConfig = field(default_factory=H1PullbackConfig)
    transition: H1TransitionConfig = field(default_factory=H1TransitionConfig)
    reclaim: H1ReclaimConfig = field(default_factory=H1ReclaimConfig)
    rejection: H1RejectionConfig = field(default_factory=H1RejectionConfig)
    confluence: H1ConfluenceConfig = field(default_factory=H1ConfluenceConfig)
    setup: SetupConfig = field(default_factory=SetupConfig)
    scoring: H1ScoringConfig = field(default_factory=H1ScoringConfig)
    blockers: H1BlockerConfig = field(default_factory=H1BlockerConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def validate(self) -> None:
        if self.data.timeframe_minutes != 60:
            raise ValueError("H1Config requires data.timeframe_minutes == 60")
        h = self.hierarchy
        if not (2 <= h.immediate_swings <= h.intermediate_swings <= h.primary_swings <= self.structure.swing_history_size):
            raise ValueError("H1 hierarchy windows must satisfy immediate <= intermediate <= primary <= swing_history_size")
        s = self.setup
        if not (0 < s.developing_score <= s.qualify_score <= 100):
            raise ValueError("require 0 < developing_score <= qualify_score <= 100")
        w = [getattr(self.scoring, f.name) for f in fields(self.scoring) if f.name.startswith("w_")]
        if min(w) < 0 or sum(w) <= 0:
            raise ValueError("H1 scoring family weights must be non-negative and not all zero")


def h1_config_from_dict(values) -> H1Config:
    """Build an H1Config from a (partial) dict; unspecified keys keep the H1 defaults."""
    base = H1Config()
    known = {f.name for f in fields(H1Config)}
    unknown = set(values or {}) - known
    if unknown:
        raise KeyError(f"unknown H1 config sections: {sorted(unknown)}")
    sections = {}
    for name, v in (values or {}).items():
        current = getattr(base, name)
        section_fields = {f.name for f in fields(current)}
        bad = set(v) - section_fields
        if bad:
            raise KeyError(f"unknown config keys for {name}: {sorted(bad)}")
        conv = {k: tuple(tuple(x) if isinstance(x, list) else x for x in val) if isinstance(val, list) else val
                for k, val in v.items()}
        sections[name] = replace(current, **conv)
    cfg = replace(base, **sections)
    cfg.validate()
    return cfg


def load_h1_config(path=None) -> H1Config:
    """Load an H1 config from YAML/JSON; missing keys keep the H1 defaults."""
    if path is None:
        cfg = H1Config()
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
    return h1_config_from_dict(data)
