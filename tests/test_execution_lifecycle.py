"""Phase 1G lifecycle: rejections, bounded retries, breaker, fills/VWAP/slippage, cancellation,
modification and close contracts, halts and resets."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from exec_helpers import World, exec_policy
from gbpjpy_engine.execution import (CloseRequest, FaultPlan, FillEvent, IllegalTransition, ModificationRequest, SimClock)
from gbpjpy_engine.execution.engine import LEGAL
from gbpjpy_engine.risk import RESET_CONFIRMATION, ResetAuthorisation


def ready(**kw):
    w = World(**kw)
    w.ready()
    return w


# --------------------------------------------------------------------------- rejections and retries
@pytest.mark.parametrize("reason", ["INVALID_VOLUME", "INVALID_STOPS", "MARKET_CLOSED", "INSUFFICIENT_MARGIN", "TRADE_DISABLED",
                                    "SYMBOL_DISABLED", "UNKNOWN_BROKER_REJECTION"])
def test_permanent_rejection_is_terminal_and_releases_the_reservation(reason):
    w = ready(faults=FaultPlan(rejections=[reason]))
    res = w.submit()
    assert res["outcome"] == "BROKER_REJECTED" and res["rejection_reason"] == reason and not res["retry_scheduled"]
    assert w.order()["state"] == "REJECTED"
    assert w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"
    w.advance(10)
    assert w.submit()["outcome"] == "REJECT"  # never re-sent


def test_transient_rejection_retries_with_deterministic_bounded_backoff():
    pol = exec_policy(breaker={"max_consecutive_rejections": 10})
    w = ready(policy=pol, faults=FaultPlan(rejections=["TRANSIENT_INFRASTRUCTURE", "TRANSIENT_INFRASTRUCTURE"]))
    t0 = w.clock.now()
    r1 = w.submit()
    assert r1["retry_scheduled"] and w.order()["state"] == "VALIDATING"
    assert pd.Timestamp(w.order()["next_allowed_at"]) == t0 + pd.Timedelta(seconds=1)
    blocked = w.submit()
    assert blocked["outcome"] == "DEFER" and "RETRY_BACKOFF" in blocked["reason_codes"]
    w.advance(1)
    w.quote(190.00, 190.02)
    r2 = w.submit()
    assert r2["retry_scheduled"] and pd.Timestamp(w.order()["next_allowed_at"]) == w.clock.now() + pd.Timedelta(seconds=2)
    w.advance(2)
    w.quote(190.00, 190.02)
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    assert w.order()["attempts"] == 3


def test_retries_are_bounded():
    pol = exec_policy(breaker={"max_consecutive_rejections": 10})
    w = ready(policy=pol, faults=FaultPlan(rejections=["TRANSIENT_INFRASTRUCTURE"] * 5))
    for _ in range(5):
        w.submit()
        w.advance(10)
        w.quote(190.00, 190.02)
    assert w.order()["state"] == "REJECTED" and w.order()["attempts"] == 3


def test_requotes_are_bounded_and_never_chase():
    pol = exec_policy(breaker={"max_consecutive_rejections": 10}, retry={"max_attempts": 5})
    w = ready(policy=pol, faults=FaultPlan(rejections=["REQUOTE"] * 5))
    for _ in range(4):
        w.submit()
        w.advance(10)
        w.quote(190.00, 190.02)
    assert w.order()["state"] == "REJECTED" and w.order()["requotes"] == 3


def test_consecutive_rejections_halt_execution():
    w = ready(faults=FaultPlan(rejections=["PRICE_CHANGED"] * 3))
    for _ in range(3):
        w.submit()
        w.advance(10)
        w.quote(190.00, 190.02)
    st = w.engine.state()
    assert st.execution_state == "EXECUTION_HALTED" and "consecutive rejections" in st.halt_reason
    assert w.submit()["outcome"] in ("HALT", "REJECT")


def test_repeated_disconnects_pause_execution():
    w = ready(policy=exec_policy(breaker={"max_disconnects": 1}),
              faults=FaultPlan(submit_failures=["DISCONNECTED"], received_despite_failure=False))
    w.submit()
    assert w.engine.state().execution_state == "EXECUTION_PAUSED"


def test_manual_halt_and_authorised_reset_only():
    w = ready()
    w.engine.halt("maintenance", "ops")
    assert w.submit()["outcome"] == "HALT" and w.port.received == []
    assert w.engine.reset(ResetAuthorisation("ops", "done", "yes please"))["status"] == "REFUSED"
    assert w.engine.reset(ResetAuthorisation("ops", "done", RESET_CONFIRMATION))["status"] == "RESET"
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    assert any(e["type"] == "AUTHORISED_RESET" for e in w.engine.state().events)


def test_pause_defers():
    w = ready()
    w.engine.pause("slow feed")
    assert "EXECUTION_PAUSED" in w.submit()["reason_codes"]


def test_illegal_transitions_raise():
    w = ready()
    st = w.engine.state()
    with pytest.raises(IllegalTransition):
        w.engine._move(st, w.intent.order_intent_id, "FILLED", "skip")
    assert "SUBMISSION_REQUESTED" not in LEGAL["CREATED"] and "FILLED" not in LEGAL
    assert "VALIDATING" not in LEGAL["UNKNOWN"]  # an unknown outcome is never re-validated without reconciliation


# --------------------------------------------------------------------------- fills
def test_partial_fills_vwap_and_full_fill():
    w = ready(faults=FaultPlan(fill_mode="PARTIAL"))
    w.submit()
    out = w.engine.process_events(w.port)
    assert [o["status"] for o in out if "status" in o][-2:] == ["PARTIALLY_FILLED", "FILLED"]
    o = w.order()
    assert o["filled_volume"] == "0.13" and len(o["fills"]) == 2 and Decimal(o["vwap"]) == Decimal("190.02")
    assert w.risk.state().reservations["RA-P-1"]["state"] == "CONVERTED"


def test_pending_partial_fills_compute_vwap_and_keep_favourable_slippage():
    w = World()
    w.ready(order_type="BUY_LIMIT", pending_price=189.90)
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    key = w.intent.ownership.client_order_key
    w.port.fill_pending(key, 189.90, w.clock.now(), volume=Decimal("0.05"))
    w.port.fill_pending(key, 189.88, w.clock.now(), volume=Decimal("0.08"))
    w.engine.process_events(w.port)
    o = w.order()
    exp = (Decimal("0.05") * Decimal("189.90") + Decimal("0.08") * Decimal("189.88")) / Decimal("0.13")
    assert o["state"] == "FILLED" and Decimal(o["vwap"]) == exp
    slips = [Decimal(w.engine.state().fills[f]["ledger"]["slippage_pips"]) for f in o["fills"]]
    assert slips == [Decimal(0), Decimal(-2)]  # favourable slippage recorded (negative), never discarded


def test_adverse_slippage_recorded_and_excessive_slippage_raises_incident():
    w = ready(faults=FaultPlan(slippage_pips=1.5))
    w.submit()
    w.engine.process_events(w.port)
    f = next(iter(w.engine.state().fills.values()))
    assert Decimal(f["ledger"]["slippage_pips"]) == Decimal("1.5") and f["price"] == 190.035
    assert Decimal(f["ledger"]["slippage_money"]) > 0 and f["ledger"]["swap"] is None
    w2 = ready(faults=FaultPlan(slippage_pips=12.0))
    w2.submit()
    w2.engine.process_events(w2.port)
    assert any(i["kind"] == "EXCESSIVE_SLIPPAGE" for i in w2.engine.state().incidents)


def test_partial_then_stop_and_cancel_remainder_policy():
    w = ready(policy=exec_policy(fills={"partial_fill_policy": "CANCEL_REMAINDER"}), faults=FaultPlan(fill_mode="PARTIAL_THEN_STOP"))
    w.submit()
    w.engine.process_events(w.port)
    assert w.order()["state"] == "PARTIALLY_FILLED" and w.order()["filled_volume"] == "0.06"
    reqs = w.engine.state().control_actions
    assert reqs[f"CANCEL-REMAINDER-{w.intent.order_intent_id}"]["remaining"] == "0.07"


def test_duplicate_fill_events_are_ignored():
    w = ready(faults=FaultPlan(duplicate_events=True))
    w.submit()
    out = w.engine.process_events(w.port)
    assert sum(o.get("status") == "DUPLICATE_IGNORED" for o in out) == 1
    assert w.order()["filled_volume"] == "0.13"


def test_orphan_fill_is_a_critical_incident():
    w = ready()
    f = FillEvent("X-1", "not-ours", None, Decimal("0.1"), 190.0, w.clock.now())
    assert w.engine.on_fill(f)["status"] == "ORPHAN"
    st = w.engine.state()
    assert any(i["kind"] == "ORPHAN_FILL" for i in st.incidents)
    assert w.gate()["outcome"] == "DEFER"  # unresolved critical issue blocks new submissions


def test_overfill_halts():
    w = ready()
    w.submit()
    w.engine.process_events(w.port)
    key = w.intent.ownership.client_order_key
    w.engine.on_fill(FillEvent("EXTRA", key, "PAPER-1", Decimal("0.05"), 190.02, w.clock.now()))
    st = w.engine.state()
    assert st.execution_state == "EXECUTION_HALTED" and any(i["kind"] == "OVERFILL" for i in st.incidents)


# --------------------------------------------------------------------------- cancellation / expiry
def pending_world(**kw):
    w = World(**kw)
    w.ready(order_type="BUY_LIMIT", pending_price=189.90)
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    return w


def test_cancellation_lifecycle_releases_reservation():
    w = pending_world()
    res = w.engine.request_cancel(w.intent.order_intent_id, w.port, "setup invalidated")
    assert res["status"] == "CANCELLED"
    states = [h["to"] for h in w.order()["history"]]
    assert states[-3:] == ["CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED", "CANCELLED"]
    assert w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


def test_cancel_fill_race_is_resolved_by_the_fill():
    w = pending_world(faults=FaultPlan(fill_before_cancel=True))
    res = w.engine.request_cancel(w.intent.order_intent_id, w.port, "invalidated")
    assert res["status"] == "CANCEL_FAILED"
    w.engine.process_events(w.port)
    assert w.order()["state"] == "FILLED"


def test_cancel_unknown_outcome_requires_reconciliation():
    w = pending_world(faults=FaultPlan(cancel_response="TIMEOUT"))
    assert w.engine.request_cancel(w.intent.order_intent_id, w.port, "x")["status"] == "RECONCILIATION_REQUIRED"


def test_cancel_before_submission_and_not_cancellable_states():
    w = ready()
    assert w.engine.request_cancel(w.intent.order_intent_id, w.port, "no longer wanted")["status"] == "CANCELLED"
    w2 = ready()
    w2.submit()
    w2.engine.process_events(w2.port)
    assert w2.engine.request_cancel(w2.intent.order_intent_id, w2.port, "x")["status"] == "NOT_CANCELLABLE"


def test_expired_pending_order_is_cancelled_not_left_working():
    w = pending_world()
    w.advance(31 * 60)
    done = w.engine.expire_stale_intents(w.port)
    assert done == [w.intent.order_intent_id] and w.order()["state"] == "CANCELLED"
    assert w.port.get_order_status(w.intent.ownership.client_order_key, w.clock.now()) == "CANCELLED"


def test_expired_unsent_intent_expires():
    w = ready()
    w.advance(31 * 60)
    w.engine.expire_stale_intents(w.port)
    assert w.order()["state"] == "EXPIRED" and w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


# --------------------------------------------------------------------------- modification / close contracts
def filled_world(**kw):
    w = ready(**kw)
    w.submit()
    w.engine.process_events(w.port)
    w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    pid = w.order()["position_id"]
    return w, pid


def mod(pid, field, old, new, key, t, approval=None):
    return ModificationRequest(pid, field, old, new, "test", t, key, approval)


def test_stop_can_never_be_widened_or_target_extended_silently():
    w, pid = filled_world()
    t = w.clock.now()
    assert w.engine.request_modification(mod(pid, "STOP", 189.36, 189.20, "m1", t), w.port)["status"] == "REJECTED_WOULD_WIDEN_RISK"
    assert w.engine.request_modification(mod(pid, "TARGET", 191.785, 192.5, "m2", t), w.port)["status"] == \
        "REJECTED_TARGET_EXTENSION"
    ok = w.engine.request_modification(mod(pid, "STOP", 189.36, 189.50, "m3", t), w.port)
    assert ok["status"] == "ACCEPTED_CONTRACT" and ok["port_status"] == "ACK"
    wrong = w.engine.request_modification(mod(pid, "STOP", 189.36, 190.10, "m4", t), w.port)
    assert wrong["status"] == "REJECTED_INVALID_GEOMETRY"  # above the current BID for a long
    replay = w.engine.request_modification(mod(pid, "STOP", 189.36, 189.20, "m1", t), w.port)
    assert replay["idempotent_replay"] and replay["status"] == "REJECTED_WOULD_WIDEN_RISK"
    assert len(w.port.modification_log) == 1


def test_close_contracts_full_partial_emergency_and_foreign():
    w, pid = filled_world()
    t = w.clock.now()
    assert w.engine.request_close(CloseRequest(pid, Decimal("0.05"), "partial", t, "c1"))["status"] == "PARTIAL_CLOSE_CONTRACT"
    assert w.engine.request_close(CloseRequest(pid, Decimal("0.50"), "too much", t, "c2"))["status"] == "REJECTED_INVALID_VOLUME"
    assert w.engine.request_close(CloseRequest("FOREIGN", Decimal("0.1"), "x", t, "c3"))["status"] == \
        "REJECTED_UNKNOWN_OR_FOREIGN_POSITION"
    em = w.engine.request_close(CloseRequest(pid, Decimal("0.13"), "emergency", t, "c4", emergency=True))
    assert em["kind"] == "EMERGENCY_CLOSE" and em["status"] == "FULL_CLOSE_CONTRACT"
    assert w.engine.request_close(CloseRequest(pid, Decimal("0.13"), "emergency", t, "c4", emergency=True))["idempotent_replay"]
    assert w.port.close_log == []  # nothing sent without an explicit port


# --------------------------------------------------------------------------- latency, metrics, audit
def test_latency_measured_with_the_monotonic_clock():
    w = ready()
    real = w.port.request_submission

    def slow(*a, **k):
        w.clock.advance(2.0)
        return real(*a, **k)

    w.port.request_submission = slow
    w.submit()
    lat = w.engine.latency(w.intent.order_intent_id)
    assert lat["submit_to_ack"] == 2.0 and lat["decision_to_submit"] == 1.0


def test_late_ack_without_reference_is_treated_as_unknown():
    w = ready()
    from gbpjpy_engine.execution import PortResponse

    def late(*a, **k):
        w.clock.advance(15.0)
        return PortResponse("ACK", broker_ref=None)

    w.port.request_submission = late
    assert w.submit()["outcome"] == "UNKNOWN"


def test_metrics_and_audit_are_complete():
    w, pid = filled_world(faults=FaultPlan(slippage_pips=0.5))
    m = w.engine.metrics()
    assert m["filled"] == 1 and m["mean_slippage_pips"] == 0.5 and "never used to optimise" in m["note"]
    a = w.engine.audit(w.intent.order_intent_id)
    for k in ("why_strategy_qualified", "why_risk_approved", "execution_conditions", "requested_price", "adapter_acknowledgement",
              "fills", "realised", "protection", "errors_and_retries", "lifecycle", "final_state"):
        assert k in a
    from gbpjpy_engine.execution import explain_execution

    text = explain_execution(a)
    assert "FINAL GATE" in text and "PROTECTION PROTECTED" in text and "not a guaranteed fill" in text


def test_health_reports_degraded_monitoring_with_open_positions():
    w, pid = filled_world()
    w.port.set_connection("DISCONNECTED", w.clock.now())
    h = w.engine.health(w.port)
    assert h["monitoring"] == "DEGRADED_MONITORING" and h["open_positions"] == 1


def test_sim_clock_is_monotonic():
    c = SimClock(pd.Timestamp("2024-01-01", tz="UTC"))
    c.advance(1.5)
    assert c.monotonic() == 1.5 and c.now() == pd.Timestamp("2024-01-01 00:00:01.5", tz="UTC")
