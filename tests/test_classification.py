from __future__ import annotations

import pytest

from gbpjpy_engine.classification import Regime, classify_regime, derive_bias, family_scores
from gbpjpy_engine.config import H4Config
from gbpjpy_engine.reason_codes import ReasonCode

CFG = H4Config()


def bull_row(**over):
    row = {
        "structure_state": "bullish", "structure_defined": True, "structure_quality_score": 80.0,
        "trend_state": "strong_bullish", "trend_score": 85.0, "trend_strength": 85.0, "ema_mid_slope_atr": 0.2,
        "adx": 32.0, "di_spread": 15.0, "chop_score": 20.0, "directional_efficiency": 0.55,
        "bullish_momentum_score": 55.0, "bearish_momentum_score": 5.0, "momentum_score": 50.0,
        "momentum_acceleration": "steady", "volatility_regime": "normal", "volatility_trend": "stable",
        "atr_percentile": 50.0, "atr_ratio": 1.0, "volatility_shock": False, "extension_state": "normal",
        "extension_direction": "up", "range_location_medium": 70.0, "bars_since_break": 5.0,
        "last_break_type": "BOS", "last_break_direction": "bullish", "last_break_status": "confirmed",
    }
    row.update(over)
    return row


def bear_row(**over):
    row = bull_row(
        structure_state="bearish", trend_state="strong_bearish", trend_score=-85.0, ema_mid_slope_atr=-0.2,
        di_spread=-15.0, bullish_momentum_score=5.0, bearish_momentum_score=55.0, momentum_score=-50.0,
        extension_direction="down", last_break_direction="bearish",
    )
    row.update(over)
    return row


def test_strong_bull_regime_with_evidence():
    r = classify_regime(bull_row(), CFG)
    assert r.regime == Regime.STRONG_BULL_TREND
    for code in ("STRUCTURE_BULLISH", "TREND_ALIGNED_BULLISH", "EMA_SLOPE_POSITIVE", "ADX_STRONG", "LOW_CHOP"):
        assert code in r.evidence
    assert r.rule == "aligned_strong"


def test_bull_regime_lists_conflicting_evidence():
    row = bull_row(extension_state="extended", distance_to_resistance_atr=0.3, resistance_strength=75.0)
    r = classify_regime(row, CFG)
    assert r.regime in (Regime.STRONG_BULL_TREND, Regime.BULL_TREND)
    assert "EXTENDED_UP" in r.conflicting_evidence
    assert "NEAR_STRONG_RESISTANCE" in r.conflicting_evidence


def test_strong_bear_regime():
    assert classify_regime(bear_row(), CFG).regime == Regime.STRONG_BEAR_TREND


def test_bull_trend_not_strong_when_adx_low():
    r = classify_regime(bull_row(adx=15.0, trend_state="bullish", trend_score=45.0), CFG)
    assert r.regime == Regime.BULL_TREND


def test_weak_trend_when_only_one_family_directional():
    r = classify_regime(bull_row(structure_state="neutral", trend_state="bullish", trend_score=40, chop_score=45), CFG)
    assert r.regime == Regime.WEAK_BULL_TREND


def test_range_and_high_vol_range_and_compression():
    rng = dict(structure_state="neutral", trend_state="neutral", trend_score=5.0, chop_score=70.0, adx=12.0)
    assert classify_regime(bull_row(**rng), CFG).regime == Regime.RANGE
    assert classify_regime(bull_row(**rng, atr_percentile=85.0), CFG).regime == Regime.HIGH_VOLATILITY_RANGE
    comp = classify_regime(bull_row(**rng, atr_percentile=8.0, atr_ratio=0.7), CFG)
    assert comp.regime == Regime.LOW_VOLATILITY_COMPRESSION


def test_transition_on_conflict():
    r = classify_regime(bull_row(trend_state="bearish", trend_score=-40.0, chop_score=45.0), CFG)
    assert r.regime == Regime.TRANSITION
    assert "STRUCTURE_TREND_CONFLICT" in r.evidence


def test_shock_overrides():
    r = classify_regime(bull_row(volatility_shock=True), CFG)
    assert r.regime == Regime.VOLATILITY_SHOCK
    assert "VOLATILITY_SHOCK" in r.evidence


def test_warmup_is_unclear():
    r = classify_regime(bull_row(), CFG, warmup_complete=False)
    assert r.regime == Regime.UNCLEAR
    assert r.evidence == ["INSUFFICIENT_HISTORY"]


def test_bias_long_and_short():
    b = derive_bias(bull_row(), Regime.STRONG_BULL_TREND, CFG)
    assert b.bias == "LONG"
    assert b.bullish_evidence_score > b.bearish_evidence_score + CFG.bias.min_separation
    assert 0 < b.bias_confidence <= 100
    s = derive_bias(bear_row(), Regime.STRONG_BEAR_TREND, CFG)
    assert s.bias == "SHORT"


def test_bias_neutral_in_non_directional_regime():
    b = derive_bias(bull_row(), Regime.RANGE, CFG)
    assert b.bias == "NEUTRAL"
    assert "BIAS_REGIME_NON_DIRECTIONAL" in b.reason_codes
    assert b.bias_confidence == 0


def test_bias_neutral_on_conflict():
    # clean bullish structure vs bearish EMA trend and bearish momentum: both sides carry real evidence
    row = bull_row(structure_state="bullish", structure_quality_score=80.0, trend_score=-80.0, di_spread=-25.0,
                   adx=30.0, bullish_momentum_score=0.0, bearish_momentum_score=90.0, chop_score=0.0,
                   last_break_type=None)
    b = derive_bias(row, Regime.WEAK_BEAR_TREND, CFG)
    assert b.bullish_evidence_score >= CFG.bias.conflict_level
    assert b.bearish_evidence_score >= CFG.bias.conflict_level
    assert b.bias == "NEUTRAL"
    assert "BIAS_CONFLICT" in b.reason_codes


def test_bias_requires_minimum_evidence():
    row = bull_row(structure_state="neutral", trend_score=35.0, bullish_momentum_score=20.0, chop_score=40.0)
    b = derive_bias(row, Regime.WEAK_BULL_TREND, CFG)
    assert b.bias == "NEUTRAL"
    assert "BIAS_INSUFFICIENT_EVIDENCE" in b.reason_codes


def test_bias_contradicting_regime_is_neutral():
    b = derive_bias(bear_row(), Regime.BULL_TREND, CFG)
    assert b.bias == "NEUTRAL"
    assert "BIAS_REGIME_CONTRADICTS" in b.reason_codes


def test_family_weighting_prevents_double_counting():
    """Saturating every correlated trend input cannot push evidence beyond the trend family weight."""
    only_trend = bull_row(structure_state="neutral", bullish_momentum_score=0.0, bearish_momentum_score=0.0,
                          trend_score=100.0, di_spread=100.0, adx=60.0, chop_score=0.0, last_break_type=None)
    b = derive_bias(only_trend, Regime.WEAK_BULL_TREND, CFG)
    w = CFG.bias
    max_trend_contrib = 100.0 * w.trend_weight / (w.structure_weight + w.trend_weight + w.momentum_weight)
    assert b.bullish_evidence_score == pytest.approx(max_trend_contrib, abs=0.01)
    assert b.bias == "NEUTRAL"  # one family alone is not enough


def test_dampeners_reduce_evidence():
    base = derive_bias(bull_row(), Regime.STRONG_BULL_TREND, CFG).bullish_evidence_score
    choppy = derive_bias(bull_row(chop_score=80.0), Regime.STRONG_BULL_TREND, CFG).bullish_evidence_score
    extended = derive_bias(bull_row(extension_state="extremely_extended"), Regime.STRONG_BULL_TREND, CFG).bullish_evidence_score
    shock = derive_bias(bull_row(volatility_shock=True), Regime.VOLATILITY_SHOCK, CFG).bullish_evidence_score
    assert choppy < base and extended < base and shock < base


def test_family_scores_structure():
    fam = family_scores(bull_row(), CFG)
    assert set(fam) == {"structure", "trend", "momentum"}
    assert fam["structure"]["bullish"] > 0 and fam["structure"]["bearish"] == 0


def test_all_emitted_codes_are_registered():
    valid = {c.value for c in ReasonCode}
    for row, reg in ((bull_row(), Regime.STRONG_BULL_TREND), (bear_row(), Regime.STRONG_BEAR_TREND)):
        r = classify_regime(row, CFG)
        b = derive_bias(row, reg, CFG)
        assert set(r.evidence + r.conflicting_evidence + b.reason_codes) <= valid
