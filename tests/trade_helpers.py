"""Constructed trade contexts for Phase 1E unit / edge-case tests (all values known at the proposal time)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from gbpjpy_engine.trade import GBPJPY, TradeConfig, TradeConstruction, TradeContext, construct


def lvl(price, strength=80.0, tf="H1", kind="h1_resistance_zone"):
    return {"price": price, "type": kind, "timeframe": tf, "strength": strength}


def mirror(x, pivot=190.0):
    return round(2 * pivot - x, 6)


def make_ctx(direction=1, **over) -> TradeContext:
    """A clean LONG (or mirrored SHORT) context: entry 190.000, ATR 30 pips, spread 2 pips (bid charts)."""
    m = (lambda x: x) if direction > 0 else mirror
    ctx = TradeContext(
        direction=direction, setup_family="TREND_PULLBACK_CONTINUATION", confirmation_family="MOMENTUM_REACCELERATION_CONFIRMATION",
        entry_price=190.020 if direction > 0 else 190.000, price_basis="bid", spread_pips=2.0, spread_assumed=False,
        atr=0.30, pip_size=0.01, last_close=m(190.0),
        stop_refs={"STRUCTURAL_INVALIDATION": {"price": m(189.40), "reason": "pullback extreme"},
                   "SWING_INVALIDATION": {"price": m(189.55), "reason": "last swing"},
                   "ZONE_INVALIDATION": {"price": m(189.30), "reason": "zone edge"}},
        structural_levels=[{"price": m(189.40), "reason": "pullback extreme"}, {"price": m(189.55), "reason": "last swing"},
                           {"price": m(189.30), "reason": "zone edge"}, {"price": m(188.90), "reason": "older swing"}],
        levels=[lvl(m(191.80), 80.0, "H1"), lvl(m(191.85), 70.0, "H4", "h4_strong_resistance_zone"), lvl(m(192.90), 60.0, "H4", "h4_swing_high")],
        adverse_wicks=list(np.full(50, 0.04)), adverse_excursions=list(np.full(40, 0.25)), ranges5=list(np.full(200, 0.6)),
        recent_extremes=list(np.full(50, m(189.95))), chop=40.0, momentum_net=20.0 * direction, h4_side_score=70.0,
        h4_side_score_at_qualification=70.0, h4_permission_confidence=70.0, h1_primary_alignment=1.0, vol_regime_h1="normal",
        vol_regime_h4="normal", entry_quality=70.0, setup_score=68.0, slippage_pips=None, slippage_status="UNKNOWN",
        meta={"symbol": "GBPJPY", "timestamp": "2024-01-02T10:00:00+00:00", "h4_permission": "ALLOW_LONG" if direction > 0
              else "ALLOW_SHORT", "news_status": "UNKNOWN", "session_context": {"session_label": "london"}},
    )
    return replace(ctx, **over)


def build(ctx, cfg=None, **kw):
    rec = TradeConstruction("P-test", "E-test", "S-test", "LONG" if ctx.direction > 0 else "SHORT", 10, "2024-01-02T10:00:00+00:00")
    return construct(ctx, rec, cfg or TradeConfig(), GBPJPY, **kw)
