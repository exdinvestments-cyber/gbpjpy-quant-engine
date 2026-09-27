"""Human-readable debug / research view of a single H4 evaluation."""

from __future__ import annotations

from .engine import H4AnalysisResult
from .reason_codes import describe
from .snapshot import H4Snapshot


def _f(v, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def explain(snap: H4Snapshot, max_zones: int = 6) -> str:
    ms = snap.market_structure
    lb = ms["last_break"]
    vd = snap.volatility_detail
    md = snap.momentum_detail
    lines = [
        f"GBPJPY H4 bar opened {snap.timestamp}  (evaluation available at {snap.available_at})",
        f"close {_f(snap.close, 3)}   data quality: {snap.data_quality_status} {snap.data_quality_flags or ''}"
        f"   warm-up complete: {snap.warmup_complete}",
        "",
        f"REGIME          {snap.regime}",
        "  evidence:     " + ", ".join(snap.regime_evidence),
        "  conflicting:  " + (", ".join(snap.regime_conflicting_evidence) or "none"),
        "",
        f"H4 BIAS         {snap.h4_bias}   confidence {_f(snap.bias_confidence)}"
        "   (strategic context - NOT an entry signal)",
        f"  bullish evidence {_f(snap.bullish_evidence_score)}  |  bearish evidence {_f(snap.bearish_evidence_score)}",
        "  families (bull/bear): "
        + "; ".join(f"{k} {v['bullish']:.2f}/{v['bearish']:.2f}" for k, v in snap.evidence_families.items()),
        "  modifiers: " + ", ".join(f"{k}={v}" for k, v in snap.evidence_modifiers.items()),
        "",
        f"STRUCTURE       {ms['state']} (swing sequence: {ms['swing_state']}; labels high={ms['last_high_label']} "
        f"low={ms['last_low_label']})   quality {_f(snap.structure_quality_score)}",
        f"  last swing high {_f(ms['last_swing_high'], 3)} (pivot {ms['last_swing_high_time']}, confirmed {ms['last_swing_high_confirmed_at']})",
        f"  last swing low  {_f(ms['last_swing_low'], 3)} (pivot {ms['last_swing_low_time']}, confirmed {ms['last_swing_low_confirmed_at']})",
        f"  last break: {lb['type']} {lb['direction']} level {_f(lb['level'], 3)} at {lb['time']} "
        f"magnitude {_f(lb['magnitude_atr'])} ATR, status {lb['status']}, "
        f"{lb['structure_before']} -> {lb['structure_after']}, {_f(lb['bars_since'], 0)} bars ago",
        "",
        f"TREND           {snap.trend_state}  score {_f(snap.trend_score)}  strength {_f(snap.trend_strength)}",
        f"  ADX {_f(snap.adx)}  +DI {_f(snap.plus_di)}  -DI {_f(snap.minus_di)}  "
        f"slope {_f(snap.adx_detail['adx_slope'])}  DI persistence {_f(snap.adx_detail['directional_persistence'])}",
        f"VOLATILITY      {snap.volatility_regime}  ATR {_f(snap.atr, 3)} ({_f(vd['atr_pct'], 3)}%)  "
        f"percentile {_f(snap.atr_percentile, 1)}  short/long {_f(vd['atr_ratio'])} ({vd['volatility_trend']})",
        f"  shock {snap.volatility_shock}  severity {_f(snap.shock_severity, 1)}  driver {vd['shock_driver']}",
        f"MOMENTUM        {snap.momentum_state}  net {_f(snap.momentum_score)}  "
        f"(bull {_f(md['bullish_momentum_score'])} / bear {_f(md['bearish_momentum_score'])})",
        f"CHOP            {_f(snap.chop_score)}  market quality {snap.market_quality}  "
        f"efficiency {_f(snap.directional_efficiency)} (pctile {_f(snap.directional_efficiency_percentile, 1)})",
        f"EXTENSION       {snap.extension_state} {snap.extension_direction}  score {_f(snap.extension_score)}",
        f"RANGE LOCATION  short {_f(snap.range_location_short, 1)}  medium {_f(snap.range_location_medium, 1)}  "
        f"long {_f(snap.range_location_long, 1)}",
        f"ROUND NUMBER    {snap.round_number}",
        f"CANDLE          {snap.candle.get('candle_class')}  range {_f(snap.candle.get('range_atr'))} ATR  "
        f"body {_f(snap.candle.get('body_range_pct'), 0)}%  close loc {_f(snap.candle.get('close_location'))}",
        f"SESSION         primary {snap.session_context.get('primary_session')}  active {snap.session_context.get('active_sessions')}"
        f"  overlap {snap.session_context.get('session_overlap')}",
        "",
        "LEVELS (nearest first)",
        f"  support    {_f(snap.nearest_support, 3)} strength {_f(snap.support_strength)} "
        f"distance {_f(snap.distance_to_support_atr)} ATR",
        f"  resistance {_f(snap.nearest_resistance, 3)} strength {_f(snap.resistance_strength)} "
        f"distance {_f(snap.distance_to_resistance_atr)} ATR",
    ]
    zones = sorted(snap.zones, key=lambda z: z["distance_atr"])[:max_zones]
    for z in zones:
        lines.append(
            f"    [{z['lower']:.3f} - {z['upper']:.3f}] {z['zone_type']:<10} strength {z['strength']:5.1f} "
            f"touches {z['interactions']} rejections {z['rejections']} breaks {z['breaks']} "
            f"dist {z['distance_atr']:.2f} ATR sources {','.join(z['sources'])}"
        )
    lines += ["", "REASON CODES"]
    lines += [f"  {c:<32} {describe(c)}" for c in snap.reason_codes]
    if snap.warnings:
        lines += ["", "WARNINGS"] + [f"  {w}" for w in snap.warnings]
    return "\n".join(lines)


def inspect(result: H4AnalysisResult, timestamp=None) -> str:
    """Explain the evaluation at ``timestamp`` (bar open time) or the latest bar."""
    snap = result.latest() if timestamp is None else result.snapshot(timestamp)
    return explain(snap)
