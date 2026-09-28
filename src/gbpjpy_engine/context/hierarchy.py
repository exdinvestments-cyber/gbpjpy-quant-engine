"""Multi-level H4 structure from the full confirmed swing history.

Three nested windows of the rolling swing memory are interpreted with the same
objective rules:

  PRIMARY       the most recent ``primary_swings`` swings (default 16)
  INTERMEDIATE  the most recent ``intermediate_swings`` swings (default 8)
  IMMEDIATE     the most recent ``immediate_swings`` swings (default 4) plus the
                current close (a close beyond the latest swing high/low counts
                as a provisional HH/LL - the close is known, so this is
                point-in-time safe and makes the layer responsive without
                inventing swings)

Per window we measure recency-weighted label consistency (linear weights, the
newest swing weighing ``recency_weight_ratio`` x the oldest), net progression
of highs and lows in ATR, swing-amplitude expansion/contraction, mean swing
duration, correction depth and label violations.

Classification: RANGING (highs and lows each within ``range_band_atr``),
BULLISH / BEARISH (consistency >= threshold and highs AND lows progressing),
TRANSITIONAL (older half one direction, newer half the other, or HH+LL
expansion, or a close through the protected swing), NEUTRAL (mixed), UNCLEAR
(fewer than two highs and two lows).  A few bearish candles therefore cannot
flip a bullish PRIMARY layer: only confirmed swings or a close through the
protected swing can.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import HierarchyConfig
from ..features.indicators import scale01
from ..features.structure import Swing

BULL = ("HH", "HL")
BEAR = ("LH", "LL")
DIRECTION = {"BULLISH": 1, "BEARISH": -1}


@dataclass
class LayerResult:
    layer: str
    classification: str
    confidence: float
    swing_ids: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    weights: list = field(default_factory=list)
    bull_consistency: float = 0.0
    bear_consistency: float = 0.0
    high_progression_atr: float = 0.0
    low_progression_atr: float = 0.0
    amplitude_trend: float = 0.0
    avg_swing_duration: float = 0.0
    avg_correction_ratio: float | None = None
    violations: int = 0
    protected_level: float | None = None
    protected_broken: bool = False
    broken_from: str | None = None  # "BULLISH"/"BEARISH" when a protected swing was broken
    provisional: str | None = None
    notes: list = field(default_factory=list)

    @property
    def direction(self) -> int:
        return DIRECTION.get(self.classification, 0)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["direction"] = self.direction
        return d


def _weights(m: int, ratio: float) -> list[float]:
    if m <= 1:
        return [1.0] * m
    return [1.0 + (ratio - 1.0) * i / (m - 1) for i in range(m)]


def classify_layer(
    layer: str,
    window: list[Swing],
    close: float,
    atr: float,
    cfg: HierarchyConfig,
    provisional: bool = False,
    protect: bool = False,
) -> LayerResult:
    res = LayerResult(layer=layer, classification="UNCLEAR", confidence=0.0)
    highs = [s for s in window if s.kind == "high"]
    lows = [s for s in window if s.kind == "low"]
    res.swing_ids = [s.swing_id for s in window]
    labels = [s.label for s in window]
    if provisional and highs and lows and atr > 0:
        tol = cfg.provisional_break_atr * atr
        if close > highs[-1].price + tol:
            labels.append("HH")
            res.provisional = "HH"
        elif close < lows[-1].price - tol:
            labels.append("LL")
            res.provisional = "LL"
    res.labels = labels
    if len(highs) < 2 or len(lows) < 2 or not np.isfinite(atr) or atr <= 0:
        res.notes.append("fewer than two confirmed highs and lows")
        return res
    w = _weights(len(labels), cfg.recency_weight_ratio)
    res.weights = [round(x, 4) for x in w]
    tot = sum(wi for wi, lb in zip(w, labels) if lb is not None) or 1.0
    bull = sum(wi for wi, lb in zip(w, labels) if lb in BULL) / tot
    bear = sum(wi for wi, lb in zip(w, labels) if lb in BEAR) / tot
    res.bull_consistency, res.bear_consistency = round(bull, 4), round(bear, 4)
    hp = (highs[-1].price - highs[0].price) / atr
    lp = (lows[-1].price - lows[0].price) / atr
    if res.provisional == "HH":
        hp = (close - highs[0].price) / atr
    elif res.provisional == "LL":
        lp = (close - lows[0].price) / atr
    res.high_progression_atr, res.low_progression_atr = round(hp, 4), round(lp, 4)
    amps = [abs(window[k].price - window[k - 1].price) / atr for k in range(1, len(window))]
    if len(amps) >= 4:
        half = len(amps) // 2
        first, second = float(np.mean(amps[:half])), float(np.mean(amps[half:]))
        res.amplitude_trend = round((second - first) / first, 4) if first > 0 else 0.0
    res.avg_swing_duration = round(float(np.mean(np.diff([s.pivot_index for s in window]))), 2)
    high_span = (max(s.price for s in highs) - min(s.price for s in highs)) / atr
    low_span = (max(s.price for s in lows) - min(s.price for s in lows)) / atr
    dominant = 1 if bull > bear else (-1 if bear > bull else 0)
    res.violations = sum(1 for lb in labels if lb in (BEAR if dominant > 0 else BULL)) if dominant else 0
    if dominant and len(amps) >= 2:
        ratios = []
        for k in range(1, len(window) - 1):
            leg_in = window[k].price - window[k - 1].price
            leg_out = window[k + 1].price - window[k].price
            if np.sign(leg_in) == dominant and abs(leg_in) > 0:
                ratios.append(abs(leg_out) / abs(leg_in))
        res.avg_correction_ratio = round(float(np.mean(ratios)), 4) if ratios else None

    thr = cfg.consistency_threshold
    recent = [lb for lb in labels[-4:] if lb]
    older = [lb for lb in labels[:-4] if lb]
    older_dir = np.sign(sum(lb in BULL for lb in older) - sum(lb in BEAR for lb in older)) if older else 0
    recent_dir = np.sign(sum(lb in BULL for lb in recent) - sum(lb in BEAR for lb in recent))
    if high_span <= cfg.range_band_atr and low_span <= cfg.range_band_atr and highs[-1].price > lows[-1].price:
        cls = "RANGING"
        res.confidence = round(100.0 * (1.0 - max(high_span, low_span) / cfg.range_band_atr) * 0.5 + 50.0 * (1.0 - abs(bull - bear)), 2)
    elif bull >= thr and hp > 0 and lp > 0:
        cls = "BULLISH"
        res.confidence = round(100.0 * (0.6 * bull + 0.4 * float(scale01(min(hp, lp), 0.0, 2.0))), 2)
    elif bear >= thr and hp < 0 and lp < 0:
        cls = "BEARISH"
        res.confidence = round(100.0 * (0.6 * bear + 0.4 * float(scale01(-max(hp, lp), 0.0, 2.0))), 2)
    elif (older_dir and recent_dir and older_dir != recent_dir) or (hp > 0 > lp and "LL" in recent and "HH" in recent):
        cls = "TRANSITIONAL"
        share = sum(lb in (BULL if recent_dir > 0 else BEAR) for lb in recent) / max(len(recent), 1)
        res.confidence = round(100.0 * share * 0.8, 2)
    else:
        cls = "NEUTRAL"
        res.confidence = round(100.0 * (1.0 - max(bull, bear)), 2)
    res.classification = cls

    if protect and cls in ("BULLISH", "BEARISH"):
        # protected swing: the opposite swing that launched the latest extreme
        if cls == "BULLISH":
            last_high = highs[-1]
            prot = next((s for s in reversed(window) if s.kind == "low" and s.pivot_index < last_high.pivot_index), None)
            if prot is not None:
                res.protected_level = float(prot.price)
                if close < prot.price - cfg.protected_break_atr * atr:
                    res.protected_broken = True
        else:
            last_low = lows[-1]
            prot = next((s for s in reversed(window) if s.kind == "high" and s.pivot_index < last_low.pivot_index), None)
            if prot is not None:
                res.protected_level = float(prot.price)
                if close > prot.price + cfg.protected_break_atr * atr:
                    res.protected_broken = True
        if res.protected_broken:
            res.notes.append(f"close through protected {'low' if cls == 'BULLISH' else 'high'} {res.protected_level:.3f}")
            res.broken_from = cls
            res.classification = "TRANSITIONAL"
            res.confidence = round(res.confidence * 0.6, 2)
    return res


def classify_hierarchy(memory: list[Swing], close: float, atr: float, cfg: HierarchyConfig) -> dict[str, LayerResult]:
    return {
        "primary": classify_layer("primary", memory[-cfg.primary_swings:], close, atr, cfg, protect=True),
        "intermediate": classify_layer("intermediate", memory[-cfg.intermediate_swings:], close, atr, cfg, protect=True),
        "immediate": classify_layer("immediate", memory[-cfg.immediate_swings:], close, atr, cfg, provisional=True),
    }
