"""Phase 1G reconciliation, restart recovery, persistence, out-of-order events and races."""

from __future__ import annotations

import json
import threading
from decimal import Decimal

import pandas as pd

from exec_helpers import World
from gbpjpy_engine.execution import ExecutionEngine, FaultPlan, InMemoryExecutionStore, JsonFileExecutionStore


def ready(**kw):
    w = World(**kw)
    w.ready()
    return w


def filled(**kw):
    w = ready(**kw)
    w.submit()
    w.engine.process_events(w.port)
    w.engine.ensure_protection(w.intent.order_intent_id, w.port)
    return w


def test_foreign_orders_and_positions_are_recorded_and_untouched():
    w = World()
    w.port.add_foreign_position("MANUAL-1", "SHORT", "0.5", 191.0, numeric_id=None)
    w.port.add_foreign_position("OTHER-EA", "LONG", "0.2", 189.0, numeric_id=77)
    w.port.add_foreign_order("F-ORD", "LONG", "0.1", 188.0, numeric_id=77)
    rec = w.engine.reconcile(w.port, startup=True)
    assert rec["critical"] == 0 and rec["startup_done"] and len(rec["foreign"]) == 3
    assert w.port.close_log == [] and w.port.modification_log == []
    assert len(w.port.get_open_positions(w.clock.now())) == 2


def test_orphan_order_and_unexpected_position_with_our_identity_block_startup():
    w = World()
    w.port.add_foreign_position("GHOST", "LONG", "0.1", 190.0, numeric_id=4411)  # our id, no known key
    w.port.add_foreign_order("GHOST-ORD", "LONG", "0.1", 189.0, numeric_id=4411, key="unknown-key")
    rec = w.engine.reconcile(w.port, startup=True)
    kinds = {i["kind"] for i in rec["issues"]}
    assert {"UNEXPECTED_BROKER_POSITION", "ORPHAN_BROKER_ORDER"} <= kinds and not rec["startup_done"]
    w.intent = w.engine.create_intent(w.approval, w.proposal)
    assert w.gate()["outcome"] == "DEFER"


def test_missing_position_volume_and_stop_mismatches():
    w = filled()
    pid = w.order()["position_id"]
    w.port.positions[pid]["volume"] = Decimal("0.10")
    w.port.positions[pid]["stop"] = 189.20
    rec = w.engine.reconcile(w.port)
    kinds = {i["kind"] for i in rec["issues"]}
    assert {"VOLUME_MISMATCH", "STOP_MISMATCH"} <= kinds and rec["critical"] >= 1
    # internal state is NOT silently overwritten
    assert w.engine.state().positions[pid]["volume"] == "0.13"
    w.port.positions[pid]["open"] = False
    rec2 = w.engine.reconcile(w.port)
    assert any(i["kind"] == "MISSING_BROKER_POSITION" for i in rec2["issues"])
    assert w.engine.state().execution_state == "EXECUTION_HALTED"  # repeated critical mismatches


def test_price_and_target_mismatch_are_non_critical_records():
    w = filled()
    pid = w.order()["position_id"]
    w.port.positions[pid]["price"] = 190.05
    w.port.positions[pid]["target"] = 191.5
    rec = w.engine.reconcile(w.port)
    kinds = {i["kind"]: i["critical"] for i in rec["issues"]}
    assert kinds == {"PRICE_MISMATCH": False, "TARGET_MISMATCH": False} and rec["critical"] == 0


def test_hidden_fills_are_discovered_by_reconciliation():
    w = ready(faults=FaultPlan(hidden_fills=True))
    w.submit()
    w.engine.process_events(w.port)
    assert w.order()["state"] == "ACKNOWLEDGED"
    w.engine.reconcile(w.port)
    assert w.order()["state"] == "FILLED" and w.order()["filled_volume"] == "0.13"


def test_unresolved_submission_status_is_critical():
    w = ready(faults=FaultPlan(submit_failures=["TIMEOUT"], received_despite_failure=False))
    w.submit()
    w.port.get_order_status = lambda key, as_of: "UNKNOWN"
    rec = w.engine.reconcile(w.port)
    assert any(i["kind"] == "SUBMISSION_STATUS_UNRESOLVED" for i in rec["issues"])
    assert w.order()["state"] == "RECONCILIATION_REQUIRED" and w.port.received == []


def test_expired_while_unknown_and_absent_becomes_expired():
    w = ready(faults=FaultPlan(submit_failures=["TIMEOUT"], received_despite_failure=False))
    w.submit()
    w.advance(31 * 60)
    w.engine.reconcile(w.port)
    assert w.order()["state"] == "EXPIRED" and w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


def test_resolve_issue_is_audited():
    w = ready()
    f_key = "stray"
    from gbpjpy_engine.execution import FillEvent

    w.engine.on_fill(FillEvent("S-1", f_key, None, Decimal("0.1"), 190.0, w.clock.now()))
    inc = w.engine.state().incidents[0]
    assert w.engine.resolve_issue(inc["id"], "ops", "manual position closed by operator")["resolved"]
    assert any(e["type"] == "ISSUE_RESOLVED" for e in w.engine.state().events)


# --------------------------------------------------------------------------- out-of-order / duplicate events
def test_fill_before_ack_after_a_timeout():
    w = ready(faults=FaultPlan(submit_failures=["TIMEOUT"], fill_before_ack=True))
    assert w.submit()["outcome"] == "UNKNOWN"
    out = w.engine.process_events(w.port)
    assert out[0]["status"] == "FILLED" and out[1]["status"] == "IGNORED"
    assert w.order()["state"] == "FILLED"


def test_delayed_events_are_not_seen_before_their_time():
    w = ready(faults=FaultPlan(event_delay_seconds=5))
    w.submit()
    assert w.engine.process_events(w.port) == []
    w.advance(5)
    assert len(w.engine.process_events(w.port)) == 2


def test_late_ack_after_reconciliation_adopted_order_is_ignored():
    w = World(faults=FaultPlan(submit_failures=["DISCONNECTED"], fill_mode="NONE"))
    w.ready(order_type="BUY_LIMIT", pending_price=189.90)
    w.submit()
    rec = w.engine.reconcile(w.port)
    assert w.intent.order_intent_id in rec["adopted"] and w.order()["state"] == "ACKNOWLEDGED"
    out = w.engine.process_events(w.port)
    assert out == [{"status": "IGNORED", "state": "ACKNOWLEDGED"}]


# --------------------------------------------------------------------------- restart, persistence
def test_file_store_restart_preserves_everything(tmp_path):
    store = JsonFileExecutionStore(tmp_path / "exec.json")
    w = filled(store=store)
    before = w.engine.state()
    w.restart()
    after = w.engine.state()
    assert after.orders == before.orders and after.positions == before.positions and after.fills == before.fills
    assert after.events[: len(before.events)] == before.events  # append-only
    assert after.events[-1]["type"] == "PROCESS_START" and not after.reconciliation["startup_done"]
    assert w.engine.reconcile(w.port, startup=True)["startup_done"]


def test_corrupted_state_fails_closed(tmp_path):
    path = tmp_path / "exec.json"
    store = JsonFileExecutionStore(path)
    w = filled(store=store)
    blob = json.loads(path.read_text())
    blob["state"]["orders"][w.intent.order_intent_id]["state"] = "CREATED"  # tampered, checksum now wrong
    path.write_text(json.dumps(blob))
    eng = ExecutionEngine(store=store, clock=w.clock)
    st = eng.state()
    assert st.execution_state == "EXECUTION_HALTED" and "corrupted" in st.halt_reason
    assert list(tmp_path.glob("exec.json.corrupt*"))


def test_in_memory_checksum_detects_corruption():
    s = InMemoryExecutionStore()
    w = filled(store=s)
    s.set_raw(s.raw().replace('"FILLED"', '"CREATED"'))
    w.restart()
    assert w.engine.state().execution_state == "EXECUTION_HALTED"


def test_event_store_is_append_only_with_monotonic_sequence():
    w = filled()
    ev = w.engine.state().events
    assert [e["seq"] for e in ev] == list(range(1, len(ev) + 1))
    assert not any(hasattr(w.engine, n) for n in ("delete_event", "edit_event", "rewrite_events"))


# --------------------------------------------------------------------------- races
def test_concurrent_submissions_of_one_intent_send_exactly_once():
    w = ready()
    results = []

    def go():
        results.append(w.submit()["outcome"])

    ts = [threading.Thread(target=go) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert results.count("ACKNOWLEDGED") == 1 and len(w.port.received) == 1


def test_two_processes_on_one_file_store_send_exactly_once(tmp_path):
    path = tmp_path / "exec.json"
    w = ready(store=JsonFileExecutionStore(path))
    other = ExecutionEngine(store=JsonFileExecutionStore(path), clock=w.clock, risk_engine=w.risk)
    other.reconcile(w.port, startup=True)
    out = []

    def a():
        out.append(w.submit()["outcome"])

    def b():
        out.append(other.submit(w.intent.order_intent_id, w.port, w.deps())["outcome"])

    ts = [threading.Thread(target=f) for f in (a, b)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out.count("ACKNOWLEDGED") == 1 and len(w.port.received) == 1


def test_restart_during_partial_fill_then_disconnect():
    w = ready(faults=FaultPlan(fill_mode="PARTIAL_THEN_STOP", submit_failures=["DISCONNECTED"]))
    w.submit()
    w.restart()
    rec = w.engine.reconcile(w.port, startup=True)
    assert w.order()["state"] == "PARTIALLY_FILLED" and w.order()["filled_volume"] == "0.06"
    assert rec["startup_done"] and len(w.port.received) == 1
    assert w.submit()["outcome"] == "REJECT"
    assert pd.Timestamp(w.engine.state().reconciliation["last_run"]) == w.clock.now()
