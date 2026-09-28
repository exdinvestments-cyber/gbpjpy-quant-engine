"""Phase 1G critical tests (spec 83-87): duplicate order, unprotected position, adverse price
direction, spread spike and a risk halt immediately before submission."""

from __future__ import annotations

from decimal import Decimal

import pandas as pd
import pytest

from exec_helpers import World, exec_policy
from gbpjpy_engine.execution import FaultPlan, JsonFileExecutionStore


# --------------------------------------------------------------------------- 83. duplicate order
@pytest.mark.parametrize("failure", ["DISCONNECTED", "TIMEOUT", "ERROR", "RAISE"])
def test_duplicate_order_received_then_lost_ack_then_restart_never_resubmits(tmp_path, failure):
    """Submit -> the adapter receives it -> the connection drops before the ACK -> timeout -> restart.
    The engine must NOT resubmit; it must reconcile first and adopt the existing order."""
    store = JsonFileExecutionStore(tmp_path / "exec.json")
    w = World(faults=FaultPlan(submit_failures=[failure], received_despite_failure=True), store=store)
    w.ready()
    res = w.submit()
    assert res["outcome"] == "UNKNOWN" and "SUBMISSION_TIMEOUT" in res["reason_codes"]
    assert w.order()["state"] == "RECONCILIATION_REQUIRED"
    assert len(w.port.received) == 1
    # still the same process: a retry is refused - an unknown outcome is never read as "not sent"
    w.advance(30)
    again = w.submit()
    assert again["outcome"] == "REJECT" and "DUPLICATE_INTENT_BLOCKED" in again["reason_codes"]
    assert len(w.port.received) == 1
    # restart: a new process over the same persisted state
    w.restart()
    assert w.order()["state"] == "RECONCILIATION_REQUIRED"
    blocked = w.submit()
    assert blocked["outcome"] != "SUBMIT_READY" and len(w.port.received) == 1
    rec = w.engine.reconcile(w.port, startup=True)
    assert w.intent.order_intent_id in rec["adopted"] and rec["startup_done"]
    assert w.order()["state"] == "FILLED" and w.order()["filled_volume"] == "0.13"
    w.engine.process_events(w.port)  # late ACK / FILL events arrive after reconciliation: ignored as duplicates
    assert w.order()["state"] == "FILLED"
    assert w.submit()["outcome"] == "REJECT"
    assert len(w.port.received) == 1  # exactly one order ever reached the adapter
    assert len([p for p in w.port.get_open_positions(w.clock.now())]) == 1


def test_duplicate_order_crash_between_write_ahead_and_send_reconciles_first(tmp_path):
    """The process dies after persisting SUBMISSION_REQUESTED: on restart the outcome is UNKNOWN."""

    class Crash(BaseException):
        pass

    store = JsonFileExecutionStore(tmp_path / "exec.json")
    w = World(store=store)
    w.ready()
    real = w.port.request_submission

    def dies(*a, **k):
        real(*a, **k)  # the adapter received it ...
        raise Crash()  # ... and the process died before the reply was handled

    w.port.request_submission = dies
    with pytest.raises(Crash):
        w.submit()
    assert w.engine.state().orders[w.intent.order_intent_id]["state"] == "SUBMISSION_REQUESTED"
    w.port.request_submission = real
    w.restart()
    assert w.order()["state"] == "RECONCILIATION_REQUIRED"
    assert w.submit()["outcome"] == "REJECT" and len(w.port.received) == 1
    w.engine.reconcile(w.port, startup=True)
    assert w.order()["state"] == "FILLED" and len(w.port.received) == 1


def test_lost_order_is_resent_only_after_reconciliation_proves_absence_with_same_key():
    w = World(faults=FaultPlan(submit_failures=["TIMEOUT"], received_despite_failure=False))
    w.ready()
    assert w.submit()["outcome"] == "UNKNOWN" and len(w.port.received) == 0
    assert w.submit()["outcome"] == "REJECT"  # not before reconciliation
    rec = w.engine.reconcile(w.port)
    assert rec["critical"] == 0 and w.order()["state"] == "VALIDATING"
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    assert len(w.port.received) == 1 and w.port.received[0]["key"] == w.intent.ownership.client_order_key
    assert w.order()["attempts"] == 2


# --------------------------------------------------------------------------- 84. unprotected position
@pytest.mark.parametrize("kind", ["reject_protection", "wrong_stop", "atomic_dropped"])
def test_unprotected_position_is_critical_halts_and_records_emergency_close_intent(kind):
    faults = {"reject_protection": FaultPlan(reject_protection=True),
              "wrong_stop": FaultPlan(wrong_stop_offset_pips=5.0),
              "atomic_dropped": FaultPlan(ignore_atomic_protection=True)}[kind]
    w = World(faults=faults, atomic=(kind == "atomic_dropped"))
    w.ready()
    assert w.submit()["outcome"] == "ACKNOWLEDGED"
    w.engine.process_events(w.port)
    assert w.order()["state"] == "FILLED"
    res = w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    assert res["status"] == "UNPROTECTED_POSITION" and "UNPROTECTED_POSITION" in res["reason_codes"]
    st = w.engine.state()
    assert st.execution_state == "EXECUTION_HALTED"
    assert any(i["kind"] == "UNPROTECTED_POSITION" and i["severity"] == "CRITICAL" and i["open"] for i in st.incidents)
    pos = next(iter(st.positions.values()))
    assert pos["protection_status"] == "UNPROTECTED_POSITION"
    em = [r for r in st.control_actions.values() if r["kind"] == "EMERGENCY_CLOSE"]
    assert len(em) == 1 and em[0]["status"] == "INTENT_RECORDED_NOT_SENT" and em[0]["volume"] == "0.13"
    assert w.port.close_log == []  # nothing is closed in Phase 1G
    # idempotent: re-checking does not create a second emergency intent
    w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    assert len([r for r in w.engine.state().control_actions.values() if r["kind"] == "EMERGENCY_CLOSE"]) == 1
    # and every further submission is halted
    w2 = World(pid="P-2", store=w.store)
    w2.engine = w.engine
    w2.clock = w.clock
    w2.ready()
    assert w2.submit()["outcome"] in ("HALT", "DEFER") and w2.port.received == []


def test_protected_position_passes_verification():
    w = World(atomic=False)
    w.ready()
    w.submit()
    w.engine.process_events(w.port)
    res = w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    assert res["status"] == "PROTECTED" and w.engine.state().execution_state == "EXECUTION_ENABLED"
    assert len(w.port.protection_log) == 1


def test_missing_target_only_is_recorded_without_halting():
    w = World(faults=FaultPlan(drop_target=True))
    w.ready()
    w.submit()
    w.engine.process_events(w.port)
    res = w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    assert res["status"] == "PROTECTED_STOP_ONLY" and res["problems"]
    assert w.engine.state().execution_state == "EXECUTION_ENABLED"


# --------------------------------------------------------------------------- 85. adverse price direction
def test_long_adverse_move_beyond_limit_is_rejected_not_chased():
    w = World()
    w.ready()
    w.quote(190.08, 190.12)  # ASK +10 pips
    res = w.submit()
    assert res["outcome"] == "REJECT" and "PRICE_DETERIORATED" in res["reason_codes"]
    assert res["info"]["directional_deterioration"] == "ADVERSE" and w.port.received == []
    assert w.order()["state"] == "REJECTED"
    assert w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


def test_short_adverse_is_a_lower_bid():
    w = World(direction=-1)
    w.ready()
    ref = w.intent.reference_entry
    w.quote(round(ref - 0.10, 3), round(ref - 0.08, 3))
    res = w.submit()
    assert res["outcome"] == "REJECT" and "PRICE_DETERIORATED" in res["reason_codes"]
    assert w.port.received == []


def test_favourable_move_is_not_deterioration_and_is_not_improved_upon():
    w = World()
    w.ready()
    w.quote(189.90, 189.92)  # ASK 10 pips better
    g = w.gate()
    assert g["outcome"] == "SUBMIT_READY" and g["info"]["directional_deterioration"] == "FAVOURABLE"
    assert g["executable_price"] == 189.92
    assert Decimal(g["validated_volume"]) <= w.intent.approved_volume  # never more volume after a better price


def test_small_adverse_move_reduces_volume_rounded_down_within_the_phase1f_cap():
    w = World()
    w.ready()
    w.quote(190.07, 190.09)  # ASK +7 pips, still inside the deterioration limit
    g = w.gate()
    assert g["outcome"] == "SUBMIT_READY" and "VOLUME_REDUCED" in g["reason_codes"]
    assert g["validated_volume"] == "0.12"
    assert Decimal(g["info"]["actual_risk_at_submission"]) <= w.intent.approved_risk_amount


def test_adverse_move_without_volume_reduction_policy_rejects():
    w = World(policy=exec_policy(intent={"allow_volume_reduction": False}))
    w.ready()
    w.quote(190.07, 190.09)
    res = w.gate()
    assert res["outcome"] == "REJECT" and "RISK_EXCEEDED_AFTER_PRICE_MOVE" in res["reason_codes"]


# --------------------------------------------------------------------------- 86. spread spike
def test_spread_spike_defers_then_recovers():
    w = World()
    w.ready()
    w.quote(189.95, 190.02)  # 7-pip spread, ASK unchanged
    res = w.submit()
    assert res["outcome"] == "DEFER" and "SPREAD_TOO_HIGH" in res["reason_codes"] and w.port.received == []
    assert w.order()["state"] == "VALIDATING"
    w.advance(2)
    w.quote(190.00, 190.02)
    assert w.submit()["outcome"] == "ACKNOWLEDGED" and len(w.port.received) == 1


def test_spread_spike_rejects_when_policy_says_reject():
    w = World(policy=exec_policy(intent={"spread_action": "REJECT"}))
    w.ready()
    w.quote(189.95, 190.02)
    res = w.submit()
    assert res["outcome"] == "REJECT" and w.order()["state"] == "REJECTED" and w.port.received == []


def test_spread_relative_to_atr_is_also_enforced():
    w = World(policy=exec_policy(intent={"max_spread_atr_fraction": 0.05}))  # 2 pips / 30-pip ATR = 0.067
    w.ready()
    assert "SPREAD_TOO_HIGH" in w.gate()["reason_codes"]


# --------------------------------------------------------------------------- 87. risk halt immediately before submission
def test_risk_halt_immediately_before_submission_blocks_even_with_stale_dependencies():
    w = World()
    w.ready()
    deps_before = w.deps()  # built while risk was ENABLED
    assert w.gate()["outcome"] == "SUBMIT_READY"
    w.risk.halt("operator stop", "ops", w.clock.now())
    res = w.engine.submit(w.intent.order_intent_id, w.port, deps_before)
    assert res["outcome"] == "HALT" and "RISK_HALTED" in res["reason_codes"]
    assert w.port.received == [] and w.order()["state"] != "SUBMISSION_REQUESTED"


def test_risk_pause_defers_submission():
    w = World()
    w.ready()
    w.risk.pause("cooling off", "ops", w.clock.now())
    res = w.submit()
    assert res["outcome"] == "DEFER" and "RISK_PAUSED" in res["reason_codes"] and w.port.received == []


def test_risk_state_unavailable_fails_closed():
    w = World()
    w.engine.risk_engine = None
    w.ready()
    res = w.submit(risk_state=None)
    assert res["outcome"] == "HALT" and "RISK_STATE_UNAVAILABLE" in res["reason_codes"]


def test_released_risk_reservation_blocks_submission():
    w = World()
    w.ready()
    w.risk.release_reservation("RA-P-1", "upstream cancelled", w.clock.now())
    res = w.submit()
    assert res["outcome"] == "REJECT" and "RISK_APPROVAL_INVALID" in res["reason_codes"]
    assert pd.Timestamp(w.clock.now()) > pd.Timestamp(w.intent.created_at) - pd.Timedelta(seconds=1)
