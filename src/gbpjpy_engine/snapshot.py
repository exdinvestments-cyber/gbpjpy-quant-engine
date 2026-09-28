"""Structured per-bar H4 snapshot consumed by later phases."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


def to_jsonable(v: Any) -> Any:
    """Convert numpy/pandas scalars to plain JSON-safe Python values (NaN -> None)."""
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [to_jsonable(x) for x in v]
    if isinstance(v, (pd.Timestamp,)):
        return None if pd.isna(v) else v.isoformat()
    if v is pd.NaT:
        return None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    return v


@dataclass
class H4Snapshot:
    timestamp: str
    available_at: str
    symbol: str
    close: float
    market_structure: dict
    structure_quality_score: float | None
    structure_quality_components: dict
    trend_state: str
    trend_strength: float | None
    trend_score: float | None
    adx: float | None
    plus_di: float | None
    minus_di: float | None
    adx_detail: dict
    atr: float | None
    atr_percentile: float | None
    volatility_regime: str
    volatility_detail: dict
    volatility_shock: bool
    shock_severity: float | None
    momentum_score: float | None
    momentum_state: str
    momentum_detail: dict
    candle: dict
    chop_score: float | None
    market_quality: str
    directional_efficiency: float | None
    directional_efficiency_percentile: float | None
    nearest_support: float | None
    nearest_resistance: float | None
    support_strength: float | None
    resistance_strength: float | None
    distance_to_support_atr: float | None
    distance_to_resistance_atr: float | None
    zones: list
    round_number: dict
    range_location_short: float | None
    range_location_medium: float | None
    range_location_long: float | None
    extension_score: float | None
    extension_state: str
    extension_direction: str
    session_context: dict
    time_context: dict
    regime: str
    regime_evidence: list
    regime_conflicting_evidence: list
    bullish_evidence_score: float
    bearish_evidence_score: float
    evidence_families: dict
    evidence_modifiers: dict
    h4_bias: str
    bias_confidence: float
    data_quality_status: str
    data_quality_flags: list
    warmup_complete: bool
    reason_codes: list
    market_data: dict = field(default_factory=dict)
    # ---- Phase 1B: H4 context & directional permission (additive) ----
    primary_structure: str | None = None
    primary_structure_confidence: float | None = None
    intermediate_structure: str | None = None
    intermediate_structure_confidence: float | None = None
    immediate_structure: str | None = None
    immediate_structure_confidence: float | None = None
    active_structural_leg: dict | None = None
    retracement_depth: float | None = None
    bullish_displacement_score: float | None = None
    bearish_displacement_score: float | None = None
    breakout_state: str | None = None
    breakout_quality_score: float | None = None
    acceptance_state: str | None = None
    liquidity_context: dict | None = None
    latest_liquidity_sweep: dict | None = None
    nearest_supply_zone: dict | None = None
    nearest_demand_zone: dict | None = None
    zone_freshness: dict | None = None
    structural_range_percentile: float | None = None
    premium_discount_state: str | None = None
    trend_maturity: str | None = None
    momentum_deterioration_score: float | None = None
    compression_score: float | None = None
    expansion_state: str | None = None
    false_break_risk_score: float | None = None
    long_room_score: float | None = None
    short_room_score: float | None = None
    context_conflict_score: float | None = None
    context_quality_score: float | None = None
    long_context_score: float | None = None
    short_context_score: float | None = None
    directional_permission: str | None = None
    permission_confidence: float | None = None
    permission_reason_codes: list = field(default_factory=list)
    hard_blockers: list = field(default_factory=list)
    context_state: dict | None = None
    context: dict | None = None
    warnings: list = field(default_factory=list)
    engine_version: str = ""
    config_hash: str = ""

    def to_dict(self) -> dict:
        return to_jsonable(asdict(self))

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def _g(row: Mapping[str, Any], k: str, default=None):
    v = row.get(k, default)
    return to_jsonable(v)


def build_snapshot(
    row: Mapping[str, Any],
    zones: list,
    regime_res,
    bias_res,
    reason_codes: list[str],
    warnings: list[str],
    symbol: str,
    engine_version: str,
    config_hash: str,
    swing_history: list | None = None,
    context_row: Mapping[str, Any] | None = None,
    context_detail: Mapping[str, Any] | None = None,
) -> H4Snapshot:
    ms = {
        "state": _g(row, "structure_state"),
        "swing_state": _g(row, "swing_structure_state"),
        "defined": _g(row, "structure_defined"),
        "last_swing_high": _g(row, "last_swing_high"),
        "last_swing_high_occurred_at": _g(row, "last_swing_high_occurred_at"),
        "last_swing_high_confirmed_at": _g(row, "last_swing_high_confirmed_at"),
        "last_high_label": _g(row, "last_high_label"),
        "last_swing_low": _g(row, "last_swing_low"),
        "last_swing_low_occurred_at": _g(row, "last_swing_low_occurred_at"),
        "last_swing_low_confirmed_at": _g(row, "last_swing_low_confirmed_at"),
        "last_low_label": _g(row, "last_low_label"),
        "swing_sequence": _g(row, "swing_sequence"),
        "history_label_counts": {k: _g(row, f"hist_{k}") for k in ("hh", "hl", "lh", "ll")},
        "swing_history": [to_jsonable(s.to_dict()) for s in (swing_history or [])],
        "last_break": {
            "type": _g(row, "last_break_type"),
            "direction": _g(row, "last_break_direction"),
            "level": _g(row, "last_break_level"),
            "time": _g(row, "last_break_time"),
            "magnitude": _g(row, "last_break_magnitude"),
            "magnitude_atr": _g(row, "last_break_atr"),
            "status": _g(row, "last_break_status"),
            "previous_status": _g(row, "last_break_previous_status"),
            "transition_reason": _g(row, "last_break_transition_reason"),
            "confirmed_at": _g(row, "last_break_confirmed_at"),
            "failed_at": _g(row, "last_break_failed_at"),
            "structure_before": _g(row, "last_break_structure_before"),
            "structure_after": _g(row, "last_break_structure_after"),
            "bars_since": _g(row, "bars_since_break"),
        },
    }
    sq = {k[3:]: _g(row, k) for k in row.keys() if str(k).startswith("sq_")}
    rn_keys = [k for k in row.keys() if str(k).startswith(("nearest_round_number", "round_number_distance"))]
    snap = H4Snapshot(
        timestamp=_g(row, "timestamp"),
        available_at=_g(row, "available_at"),
        symbol=symbol,
        close=_g(row, "close"),
        market_structure=ms,
        structure_quality_score=_g(row, "structure_quality_score"),
        structure_quality_components=sq,
        trend_state=_g(row, "trend_state"),
        trend_strength=_g(row, "trend_strength"),
        trend_score=_g(row, "trend_score"),
        adx=_g(row, "adx"),
        plus_di=_g(row, "plus_di"),
        minus_di=_g(row, "minus_di"),
        adx_detail={k: _g(row, k) for k in ("adx_slope", "di_spread", "directional_persistence", "adx_state")},
        atr=_g(row, "atr"),
        atr_percentile=_g(row, "atr_percentile"),
        volatility_regime=_g(row, "volatility_regime"),
        volatility_detail={
            k: _g(row, k)
            for k in ("atr_pct", "atr_short", "atr_long", "atr_ratio", "volatility_trend", "shock_driver",
                      "shock_range", "shock_gap", "shock_wick", "shock_displacement", "shock_atr_expansion")
        },
        volatility_shock=bool(row.get("volatility_shock", False)),
        shock_severity=_g(row, "shock_severity"),
        momentum_score=_g(row, "momentum_score"),
        momentum_state=_g(row, "momentum_state"),
        momentum_detail={
            k: _g(row, k)
            for k in ("bullish_momentum_score", "bearish_momentum_score", "momentum_percentile",
                      "momentum_acceleration", "momentum_change", "impulse_velocity_atr")
        },
        candle={
            k: _g(row, k)
            for k in ("candle_range", "candle_body", "body_range_pct", "upper_wick", "lower_wick", "upper_wick_pct",
                      "lower_wick_pct", "range_atr", "close_location", "candle_class", "abnormal_expansion")
        },
        chop_score=_g(row, "chop_score"),
        market_quality=_g(row, "market_quality"),
        directional_efficiency=_g(row, "directional_efficiency"),
        directional_efficiency_percentile=_g(row, "directional_efficiency_percentile"),
        nearest_support=_g(row, "nearest_support"),
        nearest_resistance=_g(row, "nearest_resistance"),
        support_strength=_g(row, "support_strength"),
        resistance_strength=_g(row, "resistance_strength"),
        distance_to_support_atr=_g(row, "distance_to_support_atr"),
        distance_to_resistance_atr=_g(row, "distance_to_resistance_atr"),
        zones=[to_jsonable(z.to_dict()) for z in zones],
        round_number={k: _g(row, k) for k in rn_keys},
        range_location_short=_g(row, "range_location_short"),
        range_location_medium=_g(row, "range_location_medium"),
        range_location_long=_g(row, "range_location_long"),
        extension_score=_g(row, "extension_score"),
        extension_state=_g(row, "extension_state"),
        extension_direction=_g(row, "extension_direction"),
        session_context={
            k: _g(row, k)
            for k in row.keys()
            if str(k).startswith("session_") or k in ("primary_session", "active_sessions")
        },
        time_context={k: _g(row, k) for k in ("day_of_week", "day_name", "hour_utc", "month", "quarter", "year")},
        regime=regime_res.regime.value,
        regime_evidence=list(regime_res.evidence),
        regime_conflicting_evidence=list(regime_res.conflicting_evidence),
        bullish_evidence_score=bias_res.bullish_evidence_score,
        bearish_evidence_score=bias_res.bearish_evidence_score,
        evidence_families=bias_res.family_scores,
        evidence_modifiers=bias_res.modifiers,
        h4_bias=bias_res.bias,
        bias_confidence=bias_res.bias_confidence,
        data_quality_status=_g(row, "data_quality_status"),
        data_quality_flags=list(row.get("data_quality_flags") or []),
        warmup_complete=bool(row.get("warmup_complete", False)),
        reason_codes=list(reason_codes),
        # carried for later execution / market-quality modules; NOT used by Phase 1A features
        market_data={k: _g(row, k) for k in ("volume", "spread", "source")},
        warnings=list(warnings),
        engine_version=engine_version,
        config_hash=config_hash,
    )
    if context_row is not None:
        _attach_context(snap, context_row, context_detail or {})
    return snap


def _attach_context(snap: H4Snapshot, cr: Mapping[str, Any], cd: Mapping[str, Any]) -> None:
    g = lambda k: to_jsonable(cr.get(k))  # noqa: E731
    for k in ("primary_structure", "primary_structure_confidence", "intermediate_structure",
              "intermediate_structure_confidence", "immediate_structure", "immediate_structure_confidence",
              "bullish_displacement_score", "bearish_displacement_score", "breakout_state", "breakout_quality_score",
              "acceptance_state", "structural_range_percentile", "premium_discount_state", "trend_maturity",
              "momentum_deterioration_score", "compression_score", "expansion_state", "false_break_risk_score",
              "long_room_score", "short_room_score", "context_conflict_score", "context_quality_score",
              "long_context_score", "short_context_score", "directional_permission", "permission_confidence"):
        setattr(snap, k, g(k))
    snap.retracement_depth = g("retracement_depth")
    snap.permission_reason_codes = list(cr.get("permission_reason_codes") or [])
    snap.hard_blockers = list(cr.get("hard_blockers") or [])
    snap.active_structural_leg = to_jsonable(cd.get("active_leg"))
    liq = cd.get("liquidity") or {}
    snap.liquidity_context = to_jsonable({k: v for k, v in liq.items() if k != "latest_sweep"}) if liq else None
    snap.latest_liquidity_sweep = to_jsonable(liq.get("latest_sweep")) if liq else None
    oz = cd.get("origin_zones") or {}
    snap.nearest_supply_zone = to_jsonable(oz.get("nearest_supply"))
    snap.nearest_demand_zone = to_jsonable(oz.get("nearest_demand"))
    snap.zone_freshness = {"supply": g("supply_zone_freshness"), "demand": g("demand_zone_freshness")}
    snap.context_state = {k: g(k) for k in ("context_state", "previous_context_state", "bars_in_state",
                                            "state_changed_at", "state_transition_expected", "state_changes_in_window")}
    snap.context = to_jsonable({**cd, "context_reason_codes": list(cr.get("context_reason_codes") or []),
                                "context_error": cr.get("context_error")})
