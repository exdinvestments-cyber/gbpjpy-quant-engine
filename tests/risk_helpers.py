"""Shared constructors for Phase 1F risk tests (explicit, timestamped inputs only)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as Dec

import pandas as pd

from gbpjpy_engine.risk import (AccountRiskEngine, AccountState, ConversionRate, InMemoryRiskStateStore, OpenPosition,
                                RiskPolicy, StaticRates)
from trade_helpers import build, make_ctx

T0 = pd.Timestamp("2024-01-03 10:00", tz="UTC")  # Wednesday
RATE_TS = T0 - pd.Timedelta(minutes=30)


def rates(gbpjpy="190", usdjpy="150", eurjpy="160", gbpusd=None, ts=RATE_TS, extra=(), span_hours=24 * 8) -> StaticRates:
    """Constant test quotes published hourly from ``ts`` for ``span_hours`` (span 0 = a single quote)."""
    q = []
    for h in range(span_hours + 1):
        t = ts + pd.Timedelta(hours=h)
        q += [ConversionRate("GBP", "JPY", Dec(gbpjpy), t, "test"), ConversionRate("USD", "JPY", Dec(usdjpy), t, "test"),
              ConversionRate("EUR", "JPY", Dec(eurjpy), t, "test")]
        if gbpusd:
            q.append(ConversionRate("GBP", "USD", Dec(gbpusd), t, "test"))
    return StaticRates(list(q) + list(extra))


def account(balance="10000", equity=None, currency="GBP", ts=T0, lev=30, free=None, used="0", positions=(), floating="0",
            profile="PERSONAL") -> AccountState:
    eq = Dec(equity if equity is not None else balance)
    return AccountState("ACC-1", currency, Dec(balance), eq, Dec(free) if free is not None else eq, Dec(used), None,
                        Dec(floating), ts, tuple(positions), profile=profile, **{"broker_leverage": lev})


def position(pid="POS-1", direction="LONG", volume="0.10", entry=190.0, stop=189.5, floating="0", **kw) -> OpenPosition:
    return OpenPosition(pid, "GBPJPY", direction, Dec(volume), entry, stop, T0 - pd.Timedelta(hours=1),
                        floating_pnl=Dec(floating), **kw)


def proposal(direction=1, pid="P-1", ts=T0, **ctx_over) -> dict:
    p = dict(build(make_ctx(direction, **ctx_over)).proposal)
    p["trade_proposal_id"] = pid
    p["entry_candidate_id"] = pid.replace("P-", "E-")
    p["setup_id"] = pid.replace("P-", "S-")
    p["timestamp"] = ts.isoformat()
    return p


def engine(policy=None, rate_provider=None, **kw) -> AccountRiskEngine:
    return AccountRiskEngine(policy or RiskPolicy(), store=kw.pop("store", None) or InMemoryRiskStateStore(),
                             rates=rate_provider if rate_provider is not None else rates(), **kw)


def policy(**sections) -> RiskPolicy:
    base = RiskPolicy()
    return replace(base, **{k: replace(getattr(base, k), **v) for k, v in sections.items()})
