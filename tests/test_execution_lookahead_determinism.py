"""Phase 1G look-ahead safety, determinism and performance."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pandas as pd

from exec_helpers import H, World
from gbpjpy_engine.execution import FaultPlan
from risk_helpers import T0


def ready(**kw):
    w = World(**kw)
    w.ready()
    return w


def test_future_quotes_are_invisible():
    w = ready()
    w.port.add_quote(191.00, 191.02, w.clock.now() + pd.Timedelta(seconds=5))  # a future spike
    g = w.gate()
    assert g["outcome"] == "SUBMIT_READY" and g["executable_price"] == 190.02
    w.advance(5)
    assert "PRICE_DETERIORATED" in w.gate()["reason_codes"]


def test_future_upstream_invalidation_does_not_leak_backwards():
    w = ready()
    later = w.clock.now() + H
    setup = SimpleNamespace(transitions=[(1, T0 - H, "W", "QUALIFIED", ""), (2, later, "QUALIFIED", "INVALIDATED", "")])
    assert w.gate(setup=setup)["outcome"] == "SUBMIT_READY"


def test_future_h4_or_account_information_is_not_used():
    w = ready()
    from risk_helpers import account

    assert "ACCOUNT_STALE" in w.gate(account=account(ts=w.clock.now() + pd.Timedelta(seconds=30)))["reason_codes"]


def test_conversion_rates_after_now_are_not_used():
    w = ready()
    from risk_helpers import rates

    future_only = rates(ts=w.clock.now() + H, span_hours=2)
    assert "CONVERSION_STALE" in w.gate(rates=future_only)["reason_codes"]


def test_events_and_fills_are_not_visible_before_they_happen():
    w = ready(faults=FaultPlan(event_delay_seconds=3, hidden_fills=False))
    w.submit()
    assert w.engine.process_events(w.port) == []
    assert w.port.get_fills(w.intent.ownership.client_order_key, w.clock.now() - pd.Timedelta(seconds=1)) == []


def _scenario():
    w = ready(faults=FaultPlan(rejections=["PRICE_CHANGED"], fill_mode="PARTIAL", slippage_pips=0.5, duplicate_events=True))
    w.submit()
    w.advance(1)
    w.quote(190.01, 190.03)
    w.submit()
    w.engine.process_events(w.port)
    w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    w.engine.reconcile(w.port)
    return w.engine.state()


def test_identical_inputs_give_identical_event_logs_and_state():
    a, b = _scenario(), _scenario()
    assert a.events == b.events and a.orders == b.orders and a.fills == b.fills and a.positions == b.positions
    assert a.orders["OI-RA-P-1"]["state"] == "FILLED"


def test_gate_performance_benchmark():
    w = ready()
    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        w.gate()
    per = (time.perf_counter() - t0) / n
    assert per < 0.25, per  # generous bound for CI; the measured figure is reported in the phase report
