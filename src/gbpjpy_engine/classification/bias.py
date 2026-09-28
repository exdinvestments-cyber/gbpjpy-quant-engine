"""Strategic H4 directional bias (LONG / SHORT / NEUTRAL).

This is CONTEXT/PERMISSION for later phases, not an entry signal.

Confluence without double counting
----------------------------------
Features are grouped into families.  Each directional family produces ONE
bullish and ONE bearish sub-score in [0, 1]; correlated features inside a
family are blended, never counted separately:

  structure  (weight 0.40)  swing state x structure quality, recent live BOS/CHoCH
                            (INVALIDATED / FAILED breaks carry no evidence)
  trend      (weight 0.35)  trend_score (EMA order, slopes, price position,
                            separation, persistence) blended 80/20 with DI
                            spread scaled by ADX.  ADX/DI and EMAs are both
                            trend-persistence measures -> one family.
  momentum   (weight 0.25)  bullish/bearish momentum scores (ROC, velocity,
                            body ratios, consecutive closes -> one family)

Non-directional families act as DAMPENERS (they can only reduce evidence):

  market quality  chop_score reduces both sides
  volatility      extreme volatility / shock reduces both sides
  location        extension and proximity to strong opposing zones reduce the
                  side that is stretched into the obstacle

Decision
--------
bias is directional only if ALL hold:
  * regime is not non-directional (RANGE*, COMPRESSION, TRANSITION, UNCLEAR)
  * evidence direction agrees with a directional regime
  * NOT (both sides >= conflict_level)          -> else BIAS_CONFLICT
  * dominant >= min_evidence                     -> else BIAS_INSUFFICIENT_EVIDENCE
  * |bull - bear| >= min_separation              -> else BIAS_INSUFFICIENT_SEPARATION

bias_confidence = 100 * sqrt(dominant/100 * separation/100) for a directional
bias, and 0 for NEUTRAL (it measures confidence in the directional permission).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..config import H4Config
from ..reason_codes import ReasonCode as RC
from .regime import BEARISH_REGIMES, BULLISH_REGIMES, NON_DIRECTIONAL_REGIMES, Regime, _num


@dataclass
class BiasResult:
    bias: str
    bias_confidence: float
    bullish_evidence_score: float
    bearish_evidence_score: float
    family_scores: dict = field(default_factory=dict)
    modifiers: dict = field(default_factory=dict)
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bias": self.bias,
            "bias_confidence": self.bias_confidence,
            "bullish_evidence_score": self.bullish_evidence_score,
            "bearish_evidence_score": self.bearish_evidence_score,
            "family_scores": self.family_scores,
            "modifiers": self.modifiers,
            "reason_codes": self.reason_codes,
        }


def _clip01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    return max(0.0, min(1.0, x))


def family_scores(row: Mapping[str, Any], cfg: H4Config) -> dict[str, dict[str, float]]:
    # structure family
    st = row.get("structure_state")
    q = _clip01(_num(row, "structure_quality_score", 0.0) / 100.0)
    s_bull = s_bear = 0.0
    if st == "bullish":
        s_bull = 0.5 + 0.5 * q
    elif st == "bearish":
        s_bear = 0.5 + 0.5 * q
    bsb = _num(row, "bars_since_break")
    recent_break = not math.isnan(bsb) and bsb <= cfg.structure.transition_memory_bars and row.get("last_break_status") in ("CANDIDATE", "CONFIRMED", "ACCEPTED")
    if recent_break:
        bd = row.get("last_break_direction")
        if st == "transitional":
            # a live change of character gives modest evidence in its direction
            if bd == "bullish":
                s_bull = max(s_bull, 0.35)
            elif bd == "bearish":
                s_bear = max(s_bear, 0.35)
        elif row.get("last_break_type") == "BOS" and row.get("last_break_status") in ("CONFIRMED", "ACCEPTED"):
            if bd == "bullish" and st == "bullish":
                s_bull = min(1.0, s_bull + 0.1)
            elif bd == "bearish" and st == "bearish":
                s_bear = min(1.0, s_bear + 0.1)

    # trend family (EMA trend score + DI spread weighted by ADX)
    t = _num(row, "trend_score", 0.0)
    t = 0.0 if math.isnan(t) else t / 100.0
    di = _num(row, "di_spread", 0.0)
    adx = _num(row, "adx", 0.0)
    di_c = 0.0 if math.isnan(di) else max(-1.0, min(1.0, di / 25.0)) * _clip01((adx - 15.0) / 15.0)
    tf = 0.8 * t + 0.2 * di_c
    t_bull, t_bear = max(tf, 0.0), max(-tf, 0.0)

    # momentum family
    m_bull = _clip01(_num(row, "bullish_momentum_score", 0.0) / 100.0)
    m_bear = _clip01(_num(row, "bearish_momentum_score", 0.0) / 100.0)
    return {
        "structure": {"bullish": round(s_bull, 4), "bearish": round(s_bear, 4)},
        "trend": {"bullish": round(t_bull, 4), "bearish": round(t_bear, 4)},
        "momentum": {"bullish": round(m_bull, 4), "bearish": round(m_bear, 4)},
    }


def derive_bias(row: Mapping[str, Any], regime: Regime, cfg: H4Config, warmup_complete: bool = True) -> BiasResult:
    b = cfg.bias
    fam = family_scores(row, cfg)
    weights = {"structure": b.structure_weight, "trend": b.trend_weight, "momentum": b.momentum_weight}
    wsum = sum(weights.values())
    raw_bull = 100.0 * sum(fam[k]["bullish"] * w for k, w in weights.items()) / wsum
    raw_bear = 100.0 * sum(fam[k]["bearish"] * w for k, w in weights.items()) / wsum

    # dampeners
    chop = _num(row, "chop_score", 50.0)
    chop = 50.0 if math.isnan(chop) else chop
    quality_f = 1.0 - b.chop_penalty * chop / 100.0
    vol_f = 1.0
    if row.get("volatility_regime") == "extreme":
        vol_f *= b.extreme_vol_factor
    if bool(row.get("volatility_shock")):
        vol_f *= b.shock_factor
    loc_bull = loc_bear = 1.0
    es, ed = row.get("extension_state"), row.get("extension_direction")
    pen = b.extreme_extension_penalty if es == "extremely_extended" else (b.extended_penalty if es == "extended" else 0.0)
    if ed == "up":
        loc_bull -= pen
    elif ed == "down":
        loc_bear -= pen
    near, strong = cfg.levels.near_level_atr, cfg.levels.strong_level_score
    if _num(row, "distance_to_resistance_atr") <= near and _num(row, "resistance_strength") >= strong:
        loc_bull -= b.opposing_level_penalty
    if _num(row, "distance_to_support_atr") <= near and _num(row, "support_strength") >= strong:
        loc_bear -= b.opposing_level_penalty

    bull = round(raw_bull * quality_f * vol_f * loc_bull, 2)
    bear = round(raw_bear * quality_f * vol_f * loc_bear, 2)
    modifiers = {
        "market_quality_factor": round(quality_f, 4),
        "volatility_factor": round(vol_f, 4),
        "location_factor_bullish": round(loc_bull, 4),
        "location_factor_bearish": round(loc_bear, 4),
        "raw_bullish": round(raw_bull, 2),
        "raw_bearish": round(raw_bear, 2),
    }

    def neutral(*codes: RC) -> BiasResult:
        return BiasResult("NEUTRAL", 0.0, bull, bear, fam, modifiers, [RC.BIAS_NEUTRAL.value, *[c.value for c in codes]])

    if not warmup_complete:
        return neutral(RC.INSUFFICIENT_HISTORY)
    dominant = max(bull, bear)
    sep = abs(bull - bear)
    direction = "LONG" if bull > bear else "SHORT"
    if regime in NON_DIRECTIONAL_REGIMES:
        extra = [RC.BIAS_CONFLICT] if min(bull, bear) >= b.conflict_level else []
        return neutral(RC.BIAS_REGIME_NON_DIRECTIONAL, *extra)
    if (regime in BULLISH_REGIMES and direction == "SHORT") or (regime in BEARISH_REGIMES and direction == "LONG"):
        return neutral(RC.BIAS_REGIME_CONTRADICTS, RC.BIAS_CONFLICT)
    if min(bull, bear) >= b.conflict_level:
        return neutral(RC.BIAS_CONFLICT)
    if dominant < b.min_evidence:
        return neutral(RC.BIAS_INSUFFICIENT_EVIDENCE)
    if sep < b.min_separation:
        return neutral(RC.BIAS_INSUFFICIENT_SEPARATION)
    conf = round(100.0 * math.sqrt((dominant / 100.0) * (sep / 100.0)), 2)
    code = RC.BIAS_LONG if direction == "LONG" else RC.BIAS_SHORT
    return BiasResult(direction, conf, bull, bear, fam, modifiers, [code.value])
