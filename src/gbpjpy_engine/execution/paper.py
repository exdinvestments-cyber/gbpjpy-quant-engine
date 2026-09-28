"""Deterministic PAPER execution port (Phase 1G) - TESTING ONLY, NOT REALISM.

``PaperBroker`` implements ``ExecutionPort`` in memory so the execution-safety
engine can be exercised end to end.  It does not model real liquidity, queue
position, last-look, requote logic or fill probability, and it must never be
used to estimate live execution quality.  It never manufactures a better fill:
a market order fills at the current ASK (long) / BID (short) plus the
configured ADVERSE slippage (a negative value is explicitly requested
favourable slippage).  No network, no platform, no real order.

Faults are injected through ``FaultPlan`` - every fault is deterministic and
explicitly requested: submission timeout / disconnect / error / exception
(with or without the adapter having received the order), rejections and
requotes, partial fills, fills before acknowledgement, duplicated events,
protection rejection, wrong protective prices, a dropped target, cancel/fill
races and hidden fills (fills that only reconciliation can discover).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from ..risk.money import D, floor_to_step
from ..trade.symbol import GBPJPY
from .model import (BrokerCapabilities, BrokerOrderView, BrokerPositionView, ConnectionStatus, FillEvent, MarketSnapshot,
                    PortResponse)


@dataclass
class FaultPlan:
    """Deterministic fault injection. Lists are consumed one entry per submission attempt."""

    submit_failures: list = field(default_factory=list)  # entries: TIMEOUT | DISCONNECTED | ERROR | RAISE | None
    received_despite_failure: bool = True  # the adapter DID receive the order even though the response was lost
    rejections: list = field(default_factory=list)  # canonical rejection reasons, consumed per attempt (None = accept)
    fill_mode: str = "FULL"  # FULL | PARTIAL | PARTIAL_THEN_STOP | NONE
    partial_fractions: tuple = (Decimal("0.5"), Decimal("0.5"))
    slippage_pips: float = 0.0  # positive = adverse; negative only when favourable slippage is explicitly requested
    fill_before_ack: bool = False  # deliver FILL events before the ACK event
    duplicate_events: bool = False  # deliver every event twice
    hidden_fills: bool = False  # fills are NOT delivered as events (only visible through get_fills / positions)
    event_delay_seconds: float = 0.0
    reject_protection: bool = False  # separate protection request rejected
    ignore_atomic_protection: bool = False  # the adapter drops the stop/target sent with the entry
    wrong_stop_offset_pips: float = 0.0  # adapter confirms a different stop price
    drop_target: bool = False
    cancel_response: str = "ACK"  # ACK | REJECTED | TIMEOUT
    fill_before_cancel: bool = False  # race: the order fills just before the cancel arrives
    commission_per_volume: Decimal = Decimal(0)


class PaperBroker:
    """In-memory deterministic implementation of ``ExecutionPort`` (fault injection for tests)."""

    def __init__(self, symbol: str = "GBPJPY", account_mode: str = "HEDGING", atomic_protection: bool = True,
                 faults: FaultPlan | None = None, capabilities: BrokerCapabilities | None = None, account=None,
                 numeric_id: int | None = None, spec=None):
        self.symbol = symbol
        self.spec = spec or GBPJPY
        self.faults = faults or FaultPlan()
        self.capabilities = capabilities or BrokerCapabilities(
            account_mode=account_mode, supports_market=True, supports_pending=True, atomic_protection=atomic_protection,
            min_volume=Decimal("0.01"), max_volume=Decimal("50"), volume_step=Decimal("0.01"), stop_level_points=Decimal(0),
            freeze_level_points=Decimal(0), filling_modes=("IOC",), reported_at=None, source="paper")
        self.account = account
        self.numeric_id = numeric_id
        self.quotes: list[MarketSnapshot] = []
        self.connection: list[tuple] = []  # (timestamp, state)
        self.orders: dict = {}  # client_order_key -> order record
        self.positions: dict = {}  # position_id -> position record
        self.fills: dict = {}  # client_order_key -> [FillEvent]
        self.events: list = []  # (available_at, seq, event)
        self.delivered: set = set()
        self.received: list = []  # every submission that reached the adapter (for duplicate-order tests)
        self.close_log: list = []
        self.modification_log: list = []
        self.protection_log: list = []
        self._seq = 0
        self._attempt = 0

    # ------------------------------------------------------------------ scripted market / connection
    def add_quote(self, bid: float, ask: float, timestamp, session: str = "OPEN") -> None:
        self.quotes.append(MarketSnapshot(self.symbol, float(bid), float(ask), pd.Timestamp(timestamp), session, "paper"))
        self.quotes.sort(key=lambda q: q.timestamp)

    def set_connection(self, state: str, timestamp) -> None:
        self.connection.append((pd.Timestamp(timestamp), state))
        self.connection.sort(key=lambda c: c[0])

    def report_capabilities(self, timestamp, **changes) -> None:
        from dataclasses import replace

        self.capabilities = replace(self.capabilities, reported_at=pd.Timestamp(timestamp), **changes)

    def add_foreign_position(self, position_id: str, direction: str, volume, price: float, numeric_id: int | None = None,
                             key: str | None = None, stop: float | None = None) -> None:
        self.positions[position_id] = {"position_id": position_id, "symbol": self.symbol, "direction": direction,
                                       "volume": D(volume), "price": float(price), "stop": stop, "target": None, "key": key,
                                       "numeric_id": numeric_id, "opened_at": None, "open": True}

    def add_foreign_order(self, broker_ref: str, direction: str, volume, price: float, numeric_id: int | None = None,
                          key: str | None = None) -> None:
        self.orders[key or broker_ref] = {"broker_ref": broker_ref, "key": key, "direction": direction, "order_type": "BUY_LIMIT",
                                          "volume": D(volume), "price": price, "state": "WORKING", "numeric_id": numeric_id,
                                          "filled": D(0), "stop": None, "target": None, "foreign": True}

    # ------------------------------------------------------------------ port reads
    def _conn(self, as_of) -> str:
        st = "CONNECTED"
        for t, s in self.connection:
            if t <= as_of:
                st = s
        return st

    def get_connection_status(self, as_of):
        return ConnectionStatus(self._conn(pd.Timestamp(as_of)), pd.Timestamp(as_of), "paper")

    def get_market_snapshot(self, symbol, as_of):
        past = [q for q in self.quotes if q.timestamp <= pd.Timestamp(as_of) and q.symbol == symbol]
        return past[-1] if past else None

    def get_symbol_specification(self, symbol, as_of):
        return self.spec

    def get_capabilities(self, as_of):
        return self.capabilities

    def get_account_snapshot(self, as_of):
        return self.account

    def get_open_positions(self, as_of) -> list:
        return [BrokerPositionView(p["position_id"], p["symbol"], p["direction"], p["volume"], p["price"], p["stop"], p["target"],
                                   p["key"], p["numeric_id"], p["opened_at"]) for p in self.positions.values() if p["open"]]

    def get_pending_orders(self, as_of) -> list:
        return [BrokerOrderView(o["broker_ref"], self.symbol, o["direction"], o["order_type"], o["volume"] - o["filled"], o["price"],
                                "WORKING", o["key"], o["numeric_id"]) for o in self.orders.values() if o["state"] == "WORKING"]

    def get_fills(self, client_order_key, as_of) -> list:
        return [f for f in self.fills.get(client_order_key, []) if f.timestamp <= pd.Timestamp(as_of)]

    def get_order_status(self, client_order_key, as_of):
        o = self.orders.get(client_order_key)
        return o["state"] if o else "NOT_FOUND"

    def poll_events(self, as_of) -> list:
        out = []
        for at, seq, ev in sorted(self.events, key=lambda e: (e[0], e[1])):
            if at <= pd.Timestamp(as_of) and seq not in self.delivered:
                self.delivered.add(seq)
                out.append(ev)
        return out

    # ------------------------------------------------------------------ internals
    def _emit(self, as_of, ev) -> None:
        at = pd.Timestamp(as_of) + pd.Timedelta(seconds=self.faults.event_delay_seconds)
        for _ in range(2 if self.faults.duplicate_events else 1):
            self._seq += 1
            self.events.append((at, self._seq, ev))

    def _fill(self, key, volume, price, as_of) -> FillEvent:
        o = self.orders[key]
        n = len(self.fills.get(key, [])) + 1
        pid = f"PPOS-{key}"
        f = FillEvent(f"PF-{key}-{n}", key, o["broker_ref"], D(volume), self.spec.normalize(price), pd.Timestamp(as_of),
                      D(volume) * self.faults.commission_per_volume, pid)
        self.fills.setdefault(key, []).append(f)
        o["filled"] += D(volume)
        if o["filled"] >= o["volume"]:
            o["state"] = "FILLED"
        p = self.positions.get(pid)
        if p is None:
            p = {"position_id": pid, "symbol": self.symbol, "direction": o["direction"], "volume": D(0), "price": 0.0,
                 "stop": None, "target": None, "key": key, "numeric_id": o["numeric_id"], "opened_at": pd.Timestamp(as_of),
                 "open": True}
            if o["attach"] and not self.faults.ignore_atomic_protection:
                p["stop"] = self._confirmed_stop(o["stop"], o["direction"])
                p["target"] = None if self.faults.drop_target else o["target"]
            self.positions[pid] = p
        new_v = p["volume"] + D(volume)
        p["price"] = float((p["volume"] * D(p["price"]) + D(volume) * D(f.price)) / new_v)
        p["volume"] = new_v
        return f

    def _confirmed_stop(self, stop, direction):
        if stop is None:
            return None
        off = self.faults.wrong_stop_offset_pips * float(self.spec.pip_size)
        d = 1 if direction == "LONG" else -1
        return self.spec.normalize(stop - d * off) if off else stop

    # ------------------------------------------------------------------ port requests
    def request_submission(self, intent, volume, ownership, as_of, attach_protection):
        as_of = pd.Timestamp(as_of)
        self._attempt += 1
        if self._conn(as_of) == "DISCONNECTED":
            return PortResponse("DISCONNECTED", detail="not connected - request never left")
        rej = self.faults.rejections.pop(0) if self.faults.rejections else None
        if rej:
            return PortResponse("REJECTED", rejection_reason=rej, broker_time=as_of)
        failure = self.faults.submit_failures.pop(0) if self.faults.submit_failures else None
        key = ownership.client_order_key
        if failure and not self.faults.received_despite_failure:
            if failure == "RAISE":
                raise ConnectionError("paper transport failure (injected)")
            return PortResponse(failure, detail="injected failure - the adapter never received the order")
        # the adapter receives the order
        self.received.append({"key": key, "volume": D(volume), "at": as_of, "intent": intent.order_intent_id})
        ref = f"PAPER-{len(self.received)}"
        d = 1 if intent.direction == "LONG" else -1
        self.orders[key] = {"broker_ref": ref, "key": key, "direction": intent.direction, "order_type": intent.order_type,
                            "volume": D(volume), "price": intent.pending_price, "state": "WORKING",
                            "numeric_id": ownership.numeric_id, "filled": D(0), "stop": intent.stop_price,
                            "target": intent.target_price, "attach": bool(attach_protection), "foreign": False}
        ack = ("ACK", key, ref)
        fills = []
        if intent.order_type == "MARKET" and self.faults.fill_mode != "NONE":
            q = self.get_market_snapshot(self.symbol, as_of)
            px = (q.ask if d > 0 else q.bid) + d * self.faults.slippage_pips * float(self.spec.pip_size)
            if self.faults.fill_mode == "FULL":
                parts = [D(volume)]
            else:
                a = floor_to_step(D(volume) * D(self.faults.partial_fractions[0]), self.capabilities.volume_step)
                parts = [a] if self.faults.fill_mode == "PARTIAL_THEN_STOP" else [a, D(volume) - a]
            fills = [self._fill(key, v, px, as_of) for v in parts if v > 0]
        if self.faults.fill_before_ack:
            for f in fills:
                self._emit(as_of, ("FILL", f)) if not self.faults.hidden_fills else None
            self._emit(as_of, ack)
        else:
            self._emit(as_of, ack)
            for f in fills:
                self._emit(as_of, ("FILL", f)) if not self.faults.hidden_fills else None
        if failure == "RAISE":
            raise ConnectionError("paper transport failure after the order was received (injected)")
        if failure:
            return PortResponse(failure, detail="injected failure - the adapter DID receive the order")
        return PortResponse("ACK", broker_ref=ref, broker_time=as_of)

    def fill_pending(self, client_order_key: str, price: float, as_of, volume=None) -> FillEvent:
        """Test helper: trigger a working pending order (price chosen by the test, never improved)."""
        o = self.orders[client_order_key]
        f = self._fill(client_order_key, volume if volume is not None else o["volume"] - o["filled"], price, as_of)
        if not self.faults.hidden_fills:
            self._emit(as_of, ("FILL", f))
        return f

    def request_protection(self, position_id, stop_price, target_price, key, as_of):
        self.protection_log.append({"position_id": position_id, "key": key})
        if self._conn(pd.Timestamp(as_of)) == "DISCONNECTED":
            return PortResponse("DISCONNECTED")
        p = self.positions.get(position_id)
        if p is None:
            return PortResponse("REJECTED", rejection_reason="INVALID_STOPS", detail="unknown position")
        if self.faults.reject_protection:
            return PortResponse("REJECTED", rejection_reason="INVALID_STOPS")
        p["stop"] = self._confirmed_stop(stop_price, p["direction"])
        p["target"] = None if self.faults.drop_target else target_price
        return PortResponse("ACK")

    def request_modification(self, request, as_of):
        self.modification_log.append(request)
        p = self.positions.get(request.position_id)
        if p is None:
            return PortResponse("REJECTED", rejection_reason="INVALID_STOPS")
        p["stop" if request.field == "STOP" else "target"] = request.new_price
        return PortResponse("ACK")

    def request_cancellation(self, request, as_of):
        key = next((k for k, o in self.orders.items() if o["broker_ref"] == request.broker_ref), None)
        if key is None:
            return PortResponse("REJECTED", rejection_reason="UNKNOWN_BROKER_REJECTION", detail="unknown order")
        o = self.orders[key]
        if self.faults.fill_before_cancel and o["state"] == "WORKING":
            q = self.get_market_snapshot(self.symbol, as_of)
            d = 1 if o["direction"] == "LONG" else -1
            px = o["price"] if o["price"] is not None else (q.ask if d > 0 else q.bid)
            f = self._fill(key, o["volume"] - o["filled"], px, as_of)
            if not self.faults.hidden_fills:
                self._emit(as_of, ("FILL", f))
            return PortResponse("REJECTED", rejection_reason="INVALID_PRICE", detail="order already filled")
        if self.faults.cancel_response != "ACK":
            return PortResponse(self.faults.cancel_response, rejection_reason="TRADE_DISABLED")
        if o["state"] == "WORKING":
            o["state"] = "CANCELLED"
        return PortResponse("ACK")

    def request_close(self, request, as_of):
        self.close_log.append(request)
        p = self.positions.get(request.position_id)
        if p is None or not p["open"]:
            return PortResponse("REJECTED", rejection_reason="INVALID_VOLUME")
        p["volume"] -= D(request.volume)
        if p["volume"] <= 0:
            p["open"] = False
        return PortResponse("ACK")
