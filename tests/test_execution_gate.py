"""Phase 1G final submission gate: every check can only DEFER, REJECT or HALT - never improve anything."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pandas as pd
import pytest

from exec_helpers import H, World, exec_policy
from gbpjpy_engine.execution import BrokerCapabilities, MarketSnapshot, OrderIntent
from risk_helpers import T0, account
from risk_helpers import rates as mk_rates


def ready(**kw):
    w = World(**kw)
    w.ready()
    return w


def test_intent_is_immutable_and_idempotent():
    w = ready()
    with pytest.raises(Exception):
        w.intent.stop_price = 1.0  # type: ignore[misc]
    again = w.engine.create_intent(w.approval, w.proposal)
    assert again == w.intent and again.fingerprint() == w.intent.fingerprint()
    assert len(w.engine.state().intents) == 1
    assert w.intent.order_intent_id == "OI-RA-P-1" and len(w.intent.ownership.client_order_key) == 20
    assert w.intent.approved_volume == Decimal("0.13") and w.intent.reference_entry == 190.02
    assert w.intent.expires_at - w.intent.created_at == pd.Timedelta(minutes=30)
    assert isinstance(w.engine.intent(w.intent.order_intent_id), OrderIntent)


def test_only_risk_approved_decisions_become_intents():
    w = World()
    with pytest.raises(ValueError):
        w.engine.create_intent({"decision": "RISK_REJECTED", "approved_trade": None}, w.proposal)
    with pytest.raises(ValueError):
        w.engine.create_intent(w.approval, dict(w.proposal, trade_proposal_id="OTHER"))
    with pytest.raises(ValueError):
        w.engine.create_intent(w.approval, w.proposal, order_type="SELL_LIMIT", pending_price=190.5)  # wrong direction
    with pytest.raises(ValueError):
        w.engine.create_intent(w.approval, w.proposal, order_type="MARKET", pending_price=190.0)
    with pytest.raises(ValueError):
        w.engine.create_intent(w.approval, w.proposal, order_type="ICEBERG")


def test_explicit_clock_is_required():
    from gbpjpy_engine.execution import ExecutionEngine

    with pytest.raises(ValueError):
        ExecutionEngine()


def test_startup_reconciliation_required_before_any_submission():
    w = World()
    w.intent = w.engine.create_intent(w.approval, w.proposal)
    res = w.gate()
    assert res["outcome"] == "DEFER" and "RECONCILIATION_REQUIRED" in res["reason_codes"]


def test_stale_quote_defers_and_repeated_stale_quotes_pause():
    w = ready()
    w.advance(11)
    res = w.gate()
    assert res["outcome"] == "DEFER" and "QUOTE_STALE" in res["reason_codes"] and res["info"]["quote_age_seconds"] == 12.0
    for _ in range(4):
        w.gate()
    assert w.engine.state().execution_state == "EXECUTION_PAUSED"
    assert "EXECUTION_PAUSED" in w.gate()["reason_codes"]


def test_quote_freshness_resets_on_a_fresh_quote():
    w = ready()
    w.advance(11)
    w.gate()
    w.quote(190.00, 190.02)
    assert w.gate()["outcome"] == "SUBMIT_READY"
    assert w.engine.state().counters["stale_quotes"] == 0


@pytest.mark.parametrize("session", ["CLOSED", "HALTED", "UNKNOWN"])
def test_market_session_must_be_open(session):
    w = ready()
    w.quote(190.00, 190.02, session=session)
    res = w.gate()
    assert res["outcome"] == "DEFER" and "MARKET_CLOSED" in res["reason_codes"]


def test_unknown_session_allowed_only_by_explicit_policy():
    w = ready(policy=exec_policy(breaker={"allow_unknown_session": True}))
    w.quote(190.00, 190.02, session="UNKNOWN")
    assert w.gate()["outcome"] == "SUBMIT_READY"


@pytest.mark.parametrize("state", ["DISCONNECTED", "UNKNOWN", "DEGRADED"])
def test_connection_must_be_valid(state):
    w = ready()
    w.port.set_connection(state, w.clock.now())
    res = w.gate()
    assert res["outcome"] == "DEFER" and "CONNECTION_LOST" in res["reason_codes"]


def test_capabilities_must_be_fresh():
    w = ready()
    w.port.report_capabilities(T0 - 2 * H)
    assert "CAPABILITIES_STALE" in w.gate()["reason_codes"]


def test_account_snapshot_must_be_fresh_and_not_from_the_future():
    w = ready()
    assert "ACCOUNT_STALE" in w.gate(account=account(ts=T0 - pd.Timedelta(minutes=5)))["reason_codes"]
    assert "ACCOUNT_STALE" in w.gate(account=account(ts=w.clock.now() + H))["reason_codes"]
    assert "ACCOUNT_STALE" in w.gate(account=None)["reason_codes"]


def test_conversion_rates_must_be_available():
    w = ready()
    from gbpjpy_engine.risk import StaticRates

    res = w.gate(rates=StaticRates())
    assert res["outcome"] == "DEFER" and "CONVERSION_STALE" in res["reason_codes"]
    old = mk_rates(ts=T0 - pd.Timedelta(hours=10), span_hours=0)
    assert "CONVERSION_STALE" in w.gate(rates=old)["reason_codes"]


@pytest.mark.parametrize("which", ["construction", "entry_candidate", "setup"])
def test_upstream_invalidation_rejects(which):
    w = ready()
    t = w.clock.now() - pd.Timedelta(seconds=1)
    bad = {"construction": SimpleNamespace(transitions=[(1, T0, "X", "PROPOSED", ""), (2, t, "PROPOSED", "INVALIDATED", "")]),
           "entry_candidate": SimpleNamespace(transitions=[(1, "OPEN", T0, "C", "ENTRY_CANDIDATE", ""),
                                                           (2, "CLOSE", t, "ENTRY_CANDIDATE", "INVALIDATED", "")]),
           "setup": SimpleNamespace(transitions=[(1, T0, "W", "QUALIFIED", ""), (2, t, "QUALIFIED", "EXPIRED", "")])}[which]
    res = w.gate(**{which: bad})
    assert res["outcome"] == "REJECT" and "UPSTREAM_INVALIDATED" in res["reason_codes"]
    assert w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


def test_missing_upstream_state_fails_closed():
    w = ready()
    res = w.gate(setup=None)
    assert res["outcome"] == "DEFER" and "DEPENDENCY_UNAVAILABLE" in res["reason_codes"]


@pytest.mark.parametrize("perm", ["ALLOW_SHORT", "BLOCK_ALL", None])
def test_h4_permission_revocation_rejects(perm):
    w = ready()
    res = w.gate(h4_permission_now=perm)
    assert res["outcome"] == "REJECT" and "H4_PERMISSION_REVOKED" in res["reason_codes"]


def test_intent_expiry_and_aging():
    w = ready()
    w.advance(16 * 60)
    w.quote(190.00, 190.02)
    g = w.gate(account=account(ts=w.clock.now()))
    assert g["info"]["validity"] == "AGING"
    w.advance(15 * 60)
    res = w.gate()
    assert res["outcome"] == "REJECT" and "INTENT_EXPIRED" in res["reason_codes"] and w.order()["state"] == "EXPIRED"
    assert w.risk.state().reservations["RA-P-1"]["state"] == "RELEASED"


def test_order_type_support_is_checked():
    w = World()
    w.port.report_capabilities(T0, supports_pending=False)
    w.ready(order_type="BUY_LIMIT", pending_price=189.90)
    assert "ORDER_TYPE_UNSUPPORTED" in w.gate()["reason_codes"]


@pytest.mark.parametrize("otype,price,ok", [("BUY_LIMIT", 189.90, True), ("BUY_LIMIT", 190.10, False),
                                            ("BUY_STOP", 190.06, True), ("BUY_STOP", 189.90, False)])
def test_pending_order_geometry(otype, price, ok):
    w = World()
    w.ready(order_type=otype, pending_price=price)
    res = w.gate()
    if ok:
        assert res["outcome"] == "SUBMIT_READY" and res["executable_price"] == price
    else:
        assert "INVALID_ORDER_GEOMETRY" in res["reason_codes"]


def test_short_pending_geometry_and_bid_side():
    w = World(direction=-1)
    ref = w.proposal["executable_reference_price"]
    w.ready(order_type="SELL_LIMIT", pending_price=round(ref + 0.05, 3))
    assert w.gate()["outcome"] == "SUBMIT_READY"
    w2 = World(direction=-1)
    w2.ready()
    g = w2.gate()
    assert g["executable_price"] == w2.port.get_market_snapshot("GBPJPY", w2.clock.now()).bid  # SHORT sells the BID


def test_stop_level_from_capabilities():
    w = World()
    w.port.report_capabilities(T0, stop_level_points=Decimal(700))  # 70 pips > 66-pip stop
    w.ready()
    assert "INVALID_STOPS" in w.gate()["reason_codes"]


def test_stop_geometry_when_price_crosses_the_stop():
    w = World(policy=exec_policy(intent={"max_deterioration_pips": 500, "max_deterioration_atr": 50}))
    w.ready()
    w.quote(189.30, 189.32)  # ASK below the stop
    assert "INVALID_STOPS" in w.gate()["reason_codes"]


def test_rr_recheck_uses_the_executable_price():
    w = World(policy=exec_policy(intent={"max_deterioration_pips": 200, "max_deterioration_atr": 10}))
    w.ready()
    w.quote(191.00, 191.02)  # 100 pips worse: net R collapses
    res = w.gate()
    assert res["outcome"] == "REJECT" and ("RR_NO_LONGER_VALID" in res["reason_codes"] or "VOLUME_INVALID" in res["reason_codes"]
                                           or "RISK_EXCEEDED_AFTER_PRICE_MOVE" in res["reason_codes"])


def test_room_recheck_against_first_opposing_structure():
    w = World()
    w.proposal["primary_target"] = dict(w.proposal["primary_target"],
                                        path={"nearest_barrier": {"level": 190.10}})
    w.ready()
    res = w.gate()
    assert res["outcome"] == "REJECT" and "ROOM_NO_LONGER_VALID" in res["reason_codes"]


def test_volume_below_minimum_after_reduction_rejects():
    w = World()
    w.port.report_capabilities(T0, min_volume=Decimal("0.13"))
    w.ready()
    w.quote(190.07, 190.09)
    res = w.gate()
    assert res["outcome"] == "REJECT" and "VOLUME_INVALID" in res["reason_codes"]


def test_volume_never_increases_and_respects_capability_max():
    w = World()
    w.port.report_capabilities(T0, max_volume=Decimal("0.05"))
    w.ready()
    g = w.gate()
    assert g["validated_volume"] == "0.05"


def test_netting_account_with_existing_symbol_exposure_rejects():
    w = World(account_mode="NETTING")
    w.port.add_foreign_position("X-1", "SHORT", "0.2", 191.0, numeric_id=999)
    w.ready()
    assert "NETTING_POSITION_CONFLICT" in w.gate()["reason_codes"]


def test_hedging_account_leaves_foreign_positions_alone():
    w = World(account_mode="HEDGING")
    w.port.add_foreign_position("X-1", "SHORT", "0.2", 191.0, numeric_id=999)
    w.ready()
    assert w.gate()["outcome"] == "SUBMIT_READY"


def test_future_or_invalid_quote_is_refused():
    w = ready()

    class FuturePort(type(w.port)):
        def get_market_snapshot(self, symbol, as_of):
            return MarketSnapshot("GBPJPY", 190.0, 190.02, pd.Timestamp(as_of) + H, "OPEN")

    fp = FuturePort()
    fp.report_capabilities(T0)
    assert "QUOTE_STALE" in w.engine.gate(w.intent.order_intent_id, fp, w.deps())["reason_codes"]
    w.quote(190.05, 190.02)  # crossed quote
    assert "QUOTE_STALE" in w.gate()["reason_codes"]


def test_symbol_mismatch_rejected():
    w = ready()

    class Other(type(w.port)):
        def get_market_snapshot(self, symbol, as_of):
            return MarketSnapshot("EURJPY", 160.0, 160.02, pd.Timestamp(as_of), "OPEN")

    op = Other()
    op.report_capabilities(T0)
    assert "SYMBOL_MISMATCH" in w.engine.gate(w.intent.order_intent_id, op, w.deps())["reason_codes"]


def test_spread_percentile_is_recorded_not_optimised():
    w = ready()
    g = w.gate(spread_history_pips=(1.0, 1.5, 2.0, 2.5, 3.0))
    assert g["info"]["spread_percentile"] == 60.0 and g["outcome"] == "SUBMIT_READY"


def test_same_setup_cannot_have_two_live_intents():
    w = ready()
    w.submit()
    other = replace(w.intent, order_intent_id="OI-X")
    st = w.engine.state()
    d = other.to_dict()
    st.intents["OI-X"] = d
    st.orders["OI-X"] = dict(st.orders[w.intent.order_intent_id], state="CREATED", history=[], client_order_key="k2")
    w.engine.store.save(st)
    res = w.engine.gate("OI-X", w.port, w.deps())
    assert "DUPLICATE_INTENT_BLOCKED" in res["reason_codes"]


def test_unknown_intent():
    w = ready()
    assert w.engine.gate("NOPE", w.port, w.deps())["reason_codes"] == ["INTENT_UNKNOWN"]


def test_gate_is_read_only_on_prices_and_levels():
    w = ready()
    before = w.intent.to_dict()
    for q in [(190.0, 190.02), (189.95, 190.02), (190.06, 190.08)]:
        w.quote(*q)
        w.gate()
    assert w.engine.intent(w.intent.order_intent_id).to_dict() == before


def test_capabilities_object_is_platform_neutral():
    caps = BrokerCapabilities("HEDGING", True, True, True, Decimal("0.01"), Decimal("50"), Decimal("0.01"), None, None)
    assert caps.stop_level_points is None and caps.source == "adapter"
