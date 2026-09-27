"""Transparent H4 regime classifier.

The classifier is an ordered rule list evaluated on one bar's feature row.
It returns the regime plus the supporting evidence and conflicting evidence
(as reason codes) so a human can audit exactly why a label was chosen.

Rule order (first match wins):
  0. warm-up incomplete                              -> UNCLEAR
  1. volatility shock on this bar                    -> VOLATILITY_SHOCK
  2. structure & trend aligned, strong trend, ADX >= adx_trend_min,
     chop <= strong_max_chop, quality >= strong_min_quality -> STRONG_*_TREND
  3. structure & trend aligned, chop < trend_max_chop -> *_TREND
  4. low ATR percentile + contracting ATR, trend not strong -> LOW_VOLATILITY_COMPRESSION
  5. chop >= range_min_chop and trend not strong     -> HIGH_VOLATILITY_RANGE / RANGE
  6. one directional family, none opposing, chop < weak_max_chop -> WEAK_*_TREND
  7. structure/trend conflict, transitional structure or recent CHoCH -> TRANSITION
  8. otherwise                                       -> RANGE if choppy-ish, else UNCLEAR
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..config import H4Config
from ..reason_codes import ReasonCode as RC


class Regime(str, Enum):
    STRONG_BULL_TREND = "STRONG_BULL_TREND"
    BULL_TREND = "BULL_TREND"
    WEAK_BULL_TREND = "WEAK_BULL_TREND"
    STRONG_BEAR_TREND = "STRONG_BEAR_TREND"
    BEAR_TREND = "BEAR_TREND"
    WEAK_BEAR_TREND = "WEAK_BEAR_TREND"
    RANGE = "RANGE"
    HIGH_VOLATILITY_RANGE = "HIGH_VOLATILITY_RANGE"
    LOW_VOLATILITY_COMPRESSION = "LOW_VOLATILITY_COMPRESSION"
    TRANSITION = "TRANSITION"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    UNCLEAR = "UNCLEAR"


BULLISH_REGIMES = {Regime.STRONG_BULL_TREND, Regime.BULL_TREND, Regime.WEAK_BULL_TREND}
BEARISH_REGIMES = {Regime.STRONG_BEAR_TREND, Regime.BEAR_TREND, Regime.WEAK_BEAR_TREND}
NON_DIRECTIONAL_REGIMES = {
    Regime.RANGE,
    Regime.HIGH_VOLATILITY_RANGE,
    Regime.LOW_VOLATILITY_COMPRESSION,
    Regime.TRANSITION,
    Regime.UNCLEAR,
}


@dataclass
class RegimeResult:
    regime: Regime
    evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    rule: str = ""

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "rule": self.rule,
            "evidence": list(self.evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
        }


def _num(row: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    v = row.get(key, default)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v


def _dir_from_trend_state(s: str | None) -> int:
    if s in ("bullish", "strong_bullish"):
        return 1
    if s in ("bearish", "strong_bearish"):
        return -1
    return 0


def descriptive_codes(row: Mapping[str, Any], cfg: H4Config) -> list[str]:
    """Reason codes describing the bar's feature state (independent of regime choice)."""
    out: list[RC] = []
    st = row.get("structure_state")
    out.append(
        {
            "bullish": RC.STRUCTURE_BULLISH,
            "bearish": RC.STRUCTURE_BEARISH,
            "transitional": RC.STRUCTURE_TRANSITIONAL,
        }.get(st, RC.STRUCTURE_NEUTRAL if row.get("structure_defined") else RC.STRUCTURE_UNCLEAR)
    )
    bsb = _num(row, "bars_since_break")
    if not math.isnan(bsb) and bsb <= cfg.structure.transition_memory_bars:
        status = row.get("last_break_status")
        if status == "rejected":
            out.append(RC.BREAK_REJECTED)
        else:
            bt, bd = row.get("last_break_type"), row.get("last_break_direction")
            if bt == "CHOCH":
                out.append(RC.CHOCH_BULLISH if bd == "bullish" else RC.CHOCH_BEARISH)
            elif bt == "BOS":
                out.append(RC.BOS_BULLISH if bd == "bullish" else RC.BOS_BEARISH)
    q = _num(row, "structure_quality_score")
    if q >= 65:
        out.append(RC.HIGH_STRUCTURE_QUALITY)
    elif q < 35:
        out.append(RC.LOW_STRUCTURE_QUALITY)

    ts = row.get("trend_state")
    out += {
        "strong_bullish": [RC.TREND_STRONG_BULLISH, RC.TREND_ALIGNED_BULLISH],
        "bullish": [RC.TREND_ALIGNED_BULLISH],
        "strong_bearish": [RC.TREND_STRONG_BEARISH, RC.TREND_ALIGNED_BEARISH],
        "bearish": [RC.TREND_ALIGNED_BEARISH],
        "neutral": [RC.TREND_NEUTRAL],
    }.get(ts, [])
    slope = _num(row, "ema_mid_slope_atr")
    if slope >= cfg.chop.flat_slope_atr_per_bar:
        out.append(RC.EMA_SLOPE_POSITIVE)
    elif slope <= -cfg.chop.flat_slope_atr_per_bar:
        out.append(RC.EMA_SLOPE_NEGATIVE)
    sdir = 1 if st == "bullish" else (-1 if st == "bearish" else 0)
    if sdir and _dir_from_trend_state(ts) == -sdir:
        out.append(RC.STRUCTURE_TREND_CONFLICT)

    adx = _num(row, "adx")
    if adx >= cfg.adx.strong_adx:
        out.append(RC.ADX_STRONG)
    elif adx < cfg.adx.weak_adx:
        out.append(RC.ADX_WEAK)
    di = _num(row, "di_spread")
    if di >= 5:
        out.append(RC.DI_BULLISH)
    elif di <= -5:
        out.append(RC.DI_BEARISH)

    mom = _num(row, "momentum_score")
    if mom >= 20:
        out.append(RC.MOMENTUM_BULLISH)
    elif mom <= -20:
        out.append(RC.MOMENTUM_BEARISH)
    acc = row.get("momentum_acceleration")
    if acc == "accelerating":
        out.append(RC.MOMENTUM_ACCELERATING)
    elif acc == "decelerating":
        out.append(RC.MOMENTUM_DECELERATING)

    chop = _num(row, "chop_score")
    if chop >= cfg.chop.severe_threshold:
        out += [RC.SEVERE_CHOP, RC.HIGH_CHOP]
    elif chop >= cfg.chop.range_threshold:
        out.append(RC.HIGH_CHOP)
    elif chop < cfg.chop.clean_threshold:
        out.append(RC.LOW_CHOP)
    eff = _num(row, "directional_efficiency")
    if eff < cfg.efficiency.low_efficiency:
        out.append(RC.LOW_DIRECTIONAL_EFFICIENCY)
    elif eff >= cfg.efficiency.high_efficiency:
        out.append(RC.HIGH_DIRECTIONAL_EFFICIENCY)

    vr = row.get("volatility_regime")
    out += {
        "very_low": [RC.VOLATILITY_VERY_LOW],
        "low": [RC.VOLATILITY_LOW],
        "normal": [RC.VOLATILITY_NORMAL],
        "high": [RC.VOLATILITY_HIGH],
        "extreme": [RC.VOLATILITY_EXTREME],
    }.get(vr, [])
    vt = row.get("volatility_trend")
    if vt == "contracting":
        out.append(RC.VOLATILITY_COMPRESSION)
    elif vt == "expanding":
        out.append(RC.VOLATILITY_EXPANSION)
    if bool(row.get("volatility_shock")):
        out.append(RC.VOLATILITY_SHOCK)

    near = cfg.levels.near_level_atr
    strong = cfg.levels.strong_level_score
    if _num(row, "distance_to_support_atr") <= near and _num(row, "support_strength") >= strong:
        out.append(RC.NEAR_STRONG_SUPPORT)
    if _num(row, "distance_to_resistance_atr") <= near and _num(row, "resistance_strength") >= strong:
        out.append(RC.NEAR_STRONG_RESISTANCE)
    if abs(_num(row, "round_number_distance_pips", 1e9)) <= cfg.round_numbers.near_pips:
        out.append(RC.NEAR_ROUND_NUMBER)

    es, ed = row.get("extension_state"), row.get("extension_direction")
    if es == "extremely_extended":
        out.append(RC.OVEREXTENDED_UP if ed == "up" else RC.OVEREXTENDED_DOWN)
    elif es == "extended":
        out.append(RC.EXTENDED_UP if ed == "up" else RC.EXTENDED_DOWN)
    rl = _num(row, "range_location_medium")
    if rl >= 85:
        out.append(RC.RANGE_HIGH_LOCATION)
    elif rl <= 15:
        out.append(RC.RANGE_LOW_LOCATION)
    return [c.value for c in out]


_BULL_CONFLICTS = {
    "STRUCTURE_BEARISH", "TREND_ALIGNED_BEARISH", "MOMENTUM_BEARISH", "DI_BEARISH", "CHOCH_BEARISH",
    "HIGH_CHOP", "SEVERE_CHOP", "LOW_DIRECTIONAL_EFFICIENCY", "OVEREXTENDED_UP", "EXTENDED_UP",
    "NEAR_STRONG_RESISTANCE", "VOLATILITY_EXTREME", "STRUCTURE_TREND_CONFLICT", "LOW_STRUCTURE_QUALITY",
    "MOMENTUM_DECELERATING", "BREAK_REJECTED", "ADX_WEAK",
}
_BEAR_CONFLICTS = {
    "STRUCTURE_BULLISH", "TREND_ALIGNED_BULLISH", "MOMENTUM_BULLISH", "DI_BULLISH", "CHOCH_BULLISH",
    "HIGH_CHOP", "SEVERE_CHOP", "LOW_DIRECTIONAL_EFFICIENCY", "OVEREXTENDED_DOWN", "EXTENDED_DOWN",
    "NEAR_STRONG_SUPPORT", "VOLATILITY_EXTREME", "STRUCTURE_TREND_CONFLICT", "LOW_STRUCTURE_QUALITY",
    "MOMENTUM_DECELERATING", "BREAK_REJECTED", "ADX_WEAK",
}
_RANGE_CONFLICTS = {
    "TREND_STRONG_BULLISH", "TREND_STRONG_BEARISH", "HIGH_DIRECTIONAL_EFFICIENCY", "ADX_STRONG",
    "STRUCTURE_BULLISH", "STRUCTURE_BEARISH", "LOW_CHOP",
}


def _split(regime: Regime, codes: list[str], row: Mapping[str, Any] | None = None) -> tuple[list[str], list[str]]:
    mom = _num(row or {}, "momentum_score", 0.0)
    if regime in BULLISH_REGIMES:
        conflicts = set(_BULL_CONFLICTS)
        # acceleration/deceleration codes are direction-agnostic: resolve with the momentum sign
        conflicts.discard("MOMENTUM_DECELERATING")
        if mom < 0:
            conflicts.add("MOMENTUM_ACCELERATING")
        else:
            conflicts.add("MOMENTUM_DECELERATING")
    elif regime in BEARISH_REGIMES:
        conflicts = set(_BEAR_CONFLICTS)
        conflicts.discard("MOMENTUM_DECELERATING")
        if mom > 0:
            conflicts.add("MOMENTUM_ACCELERATING")
        else:
            conflicts.add("MOMENTUM_DECELERATING")
    elif regime in (Regime.RANGE, Regime.HIGH_VOLATILITY_RANGE, Regime.LOW_VOLATILITY_COMPRESSION):
        conflicts = _RANGE_CONFLICTS
    else:
        conflicts = set()
    ev = [c for c in codes if c not in conflicts]
    co = [c for c in codes if c in conflicts]
    return ev, co


def classify_regime(row: Mapping[str, Any], cfg: H4Config, warmup_complete: bool = True) -> RegimeResult:
    rc = cfg.regime
    codes = descriptive_codes(row, cfg)
    if not warmup_complete:
        return RegimeResult(Regime.UNCLEAR, [RC.INSUFFICIENT_HISTORY.value], [], rule="warmup")

    if bool(row.get("volatility_shock")):
        ev, co = _split(Regime.VOLATILITY_SHOCK, codes, row)
        return RegimeResult(Regime.VOLATILITY_SHOCK, ev, co, rule="shock")

    st = row.get("structure_state")
    sdir = 1 if st == "bullish" else (-1 if st == "bearish" else 0)
    tstate = row.get("trend_state")
    tdir = _dir_from_trend_state(tstate)
    strong_trend = tstate in ("strong_bullish", "strong_bearish")
    chop = _num(row, "chop_score", 50.0)
    adx = _num(row, "adx", 0.0)
    quality = _num(row, "structure_quality_score", 0.0)
    vol_pct = _num(row, "atr_percentile", 50.0)
    atr_ratio = _num(row, "atr_ratio", 1.0)

    def result(regime: Regime, rule: str) -> RegimeResult:
        ev, co = _split(regime, codes, row)
        return RegimeResult(regime, ev, co, rule=rule)

    aligned = sdir != 0 and sdir == tdir
    if aligned and strong_trend and adx >= rc.adx_trend_min and chop <= rc.strong_max_chop and quality >= rc.strong_min_quality:
        return result(Regime.STRONG_BULL_TREND if sdir > 0 else Regime.STRONG_BEAR_TREND, "aligned_strong")
    if aligned and chop < rc.trend_max_chop:
        return result(Regime.BULL_TREND if sdir > 0 else Regime.BEAR_TREND, "aligned")
    if vol_pct <= rc.compression_pct and atr_ratio <= rc.compression_ratio and not strong_trend:
        return result(Regime.LOW_VOLATILITY_COMPRESSION, "compression")
    if chop >= rc.range_min_chop and not strong_trend:
        if vol_pct >= rc.high_vol_range_pct:
            return result(Regime.HIGH_VOLATILITY_RANGE, "range_high_vol")
        return result(Regime.RANGE, "range")
    one_sided = (sdir != 0 and tdir in (0, sdir)) or (tdir != 0 and sdir == 0 and st != "transitional")
    if one_sided and chop < rc.weak_max_chop:
        d = sdir or tdir
        return result(Regime.WEAK_BULL_TREND if d > 0 else Regime.WEAK_BEAR_TREND, "one_sided")
    conflict = sdir != 0 and tdir == -sdir
    if conflict or st == "transitional" or "CHOCH_BULLISH" in codes or "CHOCH_BEARISH" in codes:
        return result(Regime.TRANSITION, "transition")
    if chop >= rc.range_min_chop - 10:
        return result(Regime.RANGE, "range_fallback")
    return result(Regime.UNCLEAR, "fallback")
