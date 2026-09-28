"""Phase 1G execution-safety engine: RISK-APPROVED TRADE -> ORDER INTENT ->
platform-neutral order lifecycle.  NO real platform connection exists.

Responsibilities stay separate: strategy decides whether a trade deserves to
exist, Phase 1F decides whether the account can afford it, execution decides
whether it is still safe and technically valid to send.  Execution may only
ACCEPT, DEFER, REJECT, CANCEL, RECONCILE or HALT: it never improves a price,
tightens or widens a stop, extends a target or raises a volume.

Two-phase safety
----------------
Phase A ``create_intent``: validate a Phase 1F approval into an immutable
OrderIntent (stable id ``OI-{risk_approval_id}`` and idempotency key).
Phase B ``submit``: re-run the final submission gate at the current time and
only then send ONE request through the ``ExecutionPort``.

Exactly-once intent
-------------------
SUBMISSION_REQUESTED is persisted BEFORE the port is called (write-ahead).  A
timeout, disconnect or error is never read as a rejection: the order becomes
UNKNOWN -> RECONCILIATION_REQUIRED and nothing is re-sent until
reconciliation proves the order absent (no order, no position, and the port's
order status NOT_FOUND).  Any re-send reuses the same client order key.
After a restart every in-flight submission becomes RECONCILIATION_REQUIRED and
no new submission is allowed before startup reconciliation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal

import pandas as pd

from ..risk.engine import RESET_CONFIRMATION
from ..risk.fx import ConversionUnavailable, factor
from ..risk.instruments import GBPJPY_CONTRACT
from ..risk.money import D, floor_to_step
from ..trade.symbol import GBPJPY
from .config import ExecutionPolicy
from .model import (TERMINAL, TRANSIENT_REJECTIONS, FillEvent, OrderIntent, OwnershipTag, StrategyIdentity)
from .store import CorruptedExecutionState, ExecutionState, InMemoryExecutionStore

ACTIVE_SENT = ("SUBMISSION_REQUESTED", "ACKNOWLEDGED", "PARTIALLY_FILLED", "UNKNOWN", "RECONCILIATION_REQUIRED",
               "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED")
LEGAL = {
    "CREATED": {"VALIDATING", "EXPIRED", "CANCELLED"},
    "VALIDATING": {"VALIDATING", "SUBMIT_READY", "REJECTED", "EXPIRED", "CANCELLED"},
    "SUBMIT_READY": {"VALIDATING", "SUBMISSION_REQUESTED", "REJECTED", "EXPIRED", "CANCELLED"},
    "SUBMISSION_REQUESTED": {"ACKNOWLEDGED", "REJECTED", "VALIDATING", "UNKNOWN", "RECONCILIATION_REQUIRED",
                             "PARTIALLY_FILLED", "FILLED"},
    "UNKNOWN": {"RECONCILIATION_REQUIRED", "PARTIALLY_FILLED", "FILLED"},
    "RECONCILIATION_REQUIRED": {"ACKNOWLEDGED", "PARTIALLY_FILLED", "FILLED", "VALIDATING", "EXPIRED", "REJECTED"},
    "ACKNOWLEDGED": {"PARTIALLY_FILLED", "FILLED", "CANCEL_REQUESTED", "RECONCILIATION_REQUIRED", "EXPIRED"},
    "PARTIALLY_FILLED": {"PARTIALLY_FILLED", "FILLED", "CANCEL_REQUESTED", "RECONCILIATION_REQUIRED"},
    "CANCEL_REQUESTED": {"CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL_FAILED", "FILLED", "PARTIALLY_FILLED", "UNKNOWN",
                         "RECONCILIATION_REQUIRED"},
    "CANCEL_ACKNOWLEDGED": {"CANCELLED", "FILLED", "PARTIALLY_FILLED", "RECONCILIATION_REQUIRED"},
    "CANCEL_FAILED": {"CANCEL_REQUESTED", "FILLED", "PARTIALLY_FILLED", "RECONCILIATION_REQUIRED"},
}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class GateDependencies:
    """Everything the final gate must re-verify.  Missing required items fail closed."""

    risk_state: object | None  # Phase 1F RiskState (global state + reservations)
    account: object | None  # Phase 1F AccountState (fresh)
    construction: object | None = None  # Phase 1E TradeConstruction
    entry_candidate: object | None = None  # Phase 1D EntryCandidate
    setup: object | None = None  # Phase 1C Setup
    h4_permission_now: str | None = None
    rates: object | None = None  # Phase 1F RateProvider
    spread_history_pips: tuple = ()


def _iso(t) -> str:
    return pd.Timestamp(t).isoformat()


def _state_at_time(transitions, now, at_pos: int, to_pos: int, initial: str) -> str:
    st = initial
    for t in transitions:
        try:
            if pd.Timestamp(t[at_pos]) <= now:
                st = t[to_pos]
        except (ValueError, TypeError):
            continue
    return st


class ExecutionEngine:
    def __init__(self, policy: ExecutionPolicy | None = None, store=None, clock=None, identity: StrategyIdentity | None = None,
                 contract=None, symbol_spec=None, risk_engine=None):
        self.policy = policy or ExecutionPolicy()
        self.policy.validate()
        self.store = store or InMemoryExecutionStore()
        if clock is None:
            raise ValueError("an explicit Clock is required (no wall-clock assumptions)")
        self.clock = clock
        self.identity = identity or StrategyIdentity()
        self.contract = contract or GBPJPY_CONTRACT
        self.spec = symbol_spec or GBPJPY
        self.risk_engine = risk_engine
        with self.store.transaction():
            st = self._load()
            recovered = [k for k, o in st.orders.items() if o["state"] in ("SUBMISSION_REQUESTED", "UNKNOWN")]
            for k in recovered:
                self._move(st, k, "RECONCILIATION_REQUIRED" if st.orders[k]["state"] == "UNKNOWN" else "UNKNOWN",
                           "process start: in-flight submission outcome unknown - never assumed failed")
                if st.orders[k]["state"] == "UNKNOWN":
                    self._move(st, k, "RECONCILIATION_REQUIRED", "reconcile before any further action")
            st.reconciliation["startup_done"] = False
            self._event(st, "PROCESS_START", None, {"recovered_in_flight": recovered})
            self.store.save(st)

    # ------------------------------------------------------------------ state plumbing
    def _load(self) -> ExecutionState:
        try:
            return self.store.load()
        except CorruptedExecutionState as exc:
            if hasattr(self.store, "quarantine"):
                self.store.quarantine()
            st = ExecutionState(execution_state="EXECUTION_HALTED", halt_reason=f"corrupted execution state: {exc}",
                                halt_source="AUTO")
            self._event(st, "HALT_TRIGGERED", None, {"reason": st.halt_reason})
            return st

    def _event(self, st, kind, intent_id, payload) -> None:
        st.events.append({"seq": len(st.events) + 1, "type": kind, "at": _iso(self.clock.now()),
                          "mono": self.clock.monotonic(), "intent_id": intent_id, "payload": payload})

    def _move(self, st, intent_id, to, reason) -> None:
        o = st.orders[intent_id]
        frm = o["state"]
        if to not in LEGAL.get(frm, set()):
            raise IllegalTransition(f"{intent_id}: {frm} -> {to}")
        o["history"].append({"at": _iso(self.clock.now()), "from": frm, "to": to, "reason": reason})
        o["state"] = to
        self._event(st, "STATE_" + to, intent_id, {"from": frm, "reason": reason})

    def _halt(self, st, reason, source="AUTO") -> None:
        if st.execution_state != "EXECUTION_HALTED":
            st.execution_state, st.halt_reason, st.halt_source = "EXECUTION_HALTED", reason, source
            self._event(st, "HALT_TRIGGERED", None, {"reason": reason, "source": source})

    def _incident(self, st, severity, kind, intent_id, detail) -> dict:
        inc = {"id": f"INC-{len(st.incidents) + 1}", "severity": severity, "kind": kind, "intent_id": intent_id,
               "detail": detail, "at": _iso(self.clock.now()), "open": True}
        st.incidents.append(inc)
        self._event(st, "INCIDENT", intent_id, inc)
        return inc

    def _beat(self, st, key) -> None:
        st.heartbeat[key] = _iso(self.clock.now())

    def state(self) -> ExecutionState:
        with self.store.transaction():
            return self._load()

    # ================================================================== Phase A: intent
    def create_intent(self, approval: dict, proposal: dict, order_type: str = "MARKET", pending_price: float | None = None) -> OrderIntent:
        if approval.get("decision") != "RISK_APPROVED" or not approval.get("approved_trade"):
            raise ValueError("only a Phase 1F RISK_APPROVED decision can become an order intent")
        a = approval["approved_trade"]
        if proposal.get("trade_proposal_id") != a["trade_proposal_id"]:
            raise ValueError("proposal does not belong to this risk approval")
        if order_type not in ("MARKET", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP"):
            raise ValueError(f"unknown order type {order_type}")
        if order_type == "MARKET" and pending_price is not None:
            raise ValueError("a market order has no pending price")
        if order_type != "MARKET":
            want = ("BUY_LIMIT", "BUY_STOP") if a["direction"] == "LONG" else ("SELL_LIMIT", "SELL_STOP")
            if order_type not in want or pending_price is None:
                raise ValueError("pending order type/direction mismatch or missing price")
        now = self.clock.now()
        oid = f"OI-{a['risk_approval_id']}"
        key = hashlib.sha256(f"{self.identity.name}|{self.identity.instance}|{oid}".encode()).hexdigest()[:20]
        own = OwnershipTag(self.identity.name, self.identity.instance, self.identity.numeric_id, a["setup_id"],
                           a["entry_candidate_id"], a["trade_proposal_id"], a["risk_approval_id"], oid, key)
        pt = proposal.get("primary_target") or {}
        path = pt.get("path") or {}
        barrier = (path.get("nearest_barrier") or {}).get("level")
        pol = self.policy.intent
        n = lambda x: self.spec.normalize(float(x)) if x is not None else None  # noqa: E731
        intent = OrderIntent(
            order_intent_id=oid, risk_approval_id=a["risk_approval_id"], trade_proposal_id=a["trade_proposal_id"],
            entry_candidate_id=a["entry_candidate_id"], setup_id=a["setup_id"], created_at=now,
            expires_at=now + pd.Timedelta(minutes=pol.validity_minutes), symbol=a["symbol"], direction=a["direction"],
            order_type=order_type, reference_entry=n(a["entry_reference"]), pending_price=n(pending_price),
            stop_price=n(a["stop"]), target_price=n(a["primary_target"]), approved_volume=D(a["approved_volume"]),
            approved_risk_amount=D(a["risk_amount"]), account_currency=a["account_currency"],
            extra_risk_pips=D(a["risk_distance_pips_incl_allowances"]) - D(a["stop_distance_pips"]),
            minimum_net_r=float(proposal.get("minimum_net_R_required") or 0.0),
            cost_extra_pips=float((proposal.get("cost_assumptions") or {}).get("total_extra_pips") or 0.0),
            atr_price=float((proposal.get("volatility") or {}).get("h1_atr_price") or 0.0),
            max_spread_pips=pol.max_spread_pips, max_deterioration_pips=pol.max_deterioration_pips,
            max_slippage_pips=pol.max_slippage_pips, nearest_barrier_price=barrier, ownership=own,
            strategy_metadata=tuple(sorted((("setup_family", proposal.get("setup_family")), ("confirmation_family", proposal.get("confirmation_family")),
                               ("trade_quality", proposal.get("trade_construction_quality_score")),
                               ("proposal_reason_codes", tuple(proposal.get("reason_codes") or ()))), key=lambda kv: kv[0])),
            risk_metadata=tuple(sorted((("binding_constraint", a["binding_constraint"]), ("actual_risk_percent", a["actual_risk_percent"]),
                           ("drawdown_state", a["drawdown_state"]), ("risk_reason_codes", tuple(a["reason_codes"]))), key=lambda kv: kv[0])),
            reason_codes=("INTENT_CREATED",),
        )
        if intent.atr_price <= 0:
            raise ValueError("intent needs the proposal ATR for deterioration checks")
        with self.store.transaction():
            st = self._load()
            if oid in st.intents:  # idempotent: the same approval always yields the same intent
                return self._intent_from(st.intents[oid])
            st.intents[oid] = intent.to_dict()
            st.orders[oid] = {"state": "CREATED", "history": [], "client_order_key": key, "attempts": 0, "requotes": 0,
                              "broker_ref": None, "validated_volume": None, "filled_volume": "0", "fills": [],
                              "vwap": None, "last_gate": None, "next_allowed_at": None, "latency": {
                                  "decision_at": approval["timestamp"], "created_at": _iso(now)}}
            self._event(st, "INTENT_CREATED", oid, {"fingerprint": intent.fingerprint(), "order_type": order_type})
            self.store.save(st)
        return intent

    @staticmethod
    def _intent_from(d: dict) -> OrderIntent:
        own = OwnershipTag(**d["ownership"])
        kw = dict(d)
        kw.update(ownership=own, created_at=pd.Timestamp(d["created_at"]), expires_at=pd.Timestamp(d["expires_at"]),
                  approved_volume=Decimal(d["approved_volume"]), approved_risk_amount=Decimal(d["approved_risk_amount"]),
                  extra_risk_pips=Decimal(d["extra_risk_pips"]),
                  strategy_metadata=tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in d["strategy_metadata"].items()),
                  risk_metadata=tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in d["risk_metadata"].items()),
                  reason_codes=tuple(d["reason_codes"]))
        return OrderIntent(**kw)

    def intent(self, intent_id: str) -> OrderIntent:
        return self._intent_from(self.state().intents[intent_id])

    # ================================================================== Phase B: final gate
    def gate(self, intent_id: str, port, deps: GateDependencies) -> dict:
        with self.store.transaction():
            st = self._load()
            res = self._gate(st, intent_id, port, deps)
            self.store.save(st)
            return res

    def _gate(self, st, oid, port, deps) -> dict:
        now = self.clock.now()
        pol, fr = self.policy, self.policy.freshness
        codes: list[str] = []
        info: dict = {"at": _iso(now)}
        if oid not in st.intents:
            return {"outcome": "REJECT", "reason": "unknown order intent", "reason_codes": ["INTENT_UNKNOWN"]}
        it = self._intent_from(st.intents[oid])
        o = st.orders[oid]
        d = it.d

        def out(outcome, reason, *extra, move=None):
            res = {"outcome": outcome, "intent_id": oid, "reason": reason, "reason_codes": list(dict.fromkeys(codes + list(extra))),
                   "info": info, "validated_volume": None, "executable_price": info.get("executable_price")}
            if move and o["state"] in LEGAL and move in LEGAL[o["state"]]:
                self._move(st, oid, move, reason)
                if move in ("REJECTED", "EXPIRED"):  # nothing was sent: the Phase 1F reservation is returned
                    self._release(oid, st, reason)
            o["last_gate"] = {k: res[k] for k in ("outcome", "reason", "reason_codes")} | {"at": _iso(now)}
            st.counters["gates_" + outcome.lower()] = st.counters.get("gates_" + outcome.lower(), 0) + 1
            self._event(st, "GATE_" + outcome, oid, {"reason": reason, "codes": res["reason_codes"]})
            return res

        # ---- global states (Phase 1F risk halt always wins) -------------------------------
        # With a live risk engine its CURRENT state is read here, so a halt raised after the
        # caller built ``deps`` still blocks this submission.
        if self.risk_engine is not None:
            try:
                deps = replace(deps, risk_state=self.risk_engine.state())
            except Exception:  # noqa: BLE001 - unreadable risk state fails closed
                deps = replace(deps, risk_state=None)
        if deps.risk_state is None:
            return out("HALT", "Phase 1F risk state unavailable - fail closed", "RISK_STATE_UNAVAILABLE")
        if deps.risk_state.global_state == "HALTED":
            return out("HALT", f"Phase 1F risk HALTED ({deps.risk_state.halt_reason})", "RISK_HALTED")
        if deps.risk_state.global_state == "PAUSED":
            return out("DEFER", "Phase 1F risk PAUSED", "RISK_PAUSED")
        if st.execution_state == "EXECUTION_HALTED":
            return out("HALT", f"execution HALTED ({st.halt_reason})", "EXECUTION_HALTED")
        if st.execution_state == "EXECUTION_PAUSED":
            return out("DEFER", "execution PAUSED", "EXECUTION_PAUSED")
        if o["state"] in ACTIVE_SENT or o["state"] in TERMINAL:
            return out("REJECT", f"intent already {o['state']} - never sent twice", "DUPLICATE_INTENT_BLOCKED")
        others = [k for k, v in st.orders.items() if k != oid and v["state"] in ACTIVE_SENT + ("FILLED",)
                  and st.intents[k]["setup_id"] == it.setup_id]
        if others:
            return out("REJECT", "another intent for the same setup is already live", "DUPLICATE_INTENT_BLOCKED")
        if not st.reconciliation.get("startup_done"):
            return out("DEFER", "startup reconciliation has not completed", "RECONCILIATION_REQUIRED")
        crit = [i for i in st.reconciliation.get("open_issues", []) if i["critical"]] + \
            [i for i in st.incidents if i["open"] and i["severity"] == "CRITICAL"]
        if crit:
            return out("DEFER", "unresolved critical reconciliation issue / execution incident", "RECONCILIATION_REQUIRED")
        if o["state"] == "CREATED":
            self._move(st, oid, "VALIDATING", "final gate started")
        # ---- intent validity window --------------------------------------------------------
        if now >= it.expires_at:
            return out("REJECT", "order intent expired", "INTENT_EXPIRED", move="EXPIRED")
        life = (it.expires_at - it.created_at).total_seconds()
        aging = (now - it.created_at).total_seconds() >= pol.intent.aging_fraction * life
        info["validity"] = "AGING" if aging else "VALID"
        codes.append("INTENT_VALID")
        if o.get("next_allowed_at") and now < pd.Timestamp(o["next_allowed_at"]):
            return out("DEFER", "bounded backoff in progress", "RETRY_BACKOFF")
        # ---- upstream revalidation (risk, proposal, entry, setup, H4) --------------------
        res_ = (deps.risk_state.reservations or {}).get(it.risk_approval_id)
        if res_ is None or res_["state"] != "RESERVED":
            return out("REJECT", "Phase 1F risk reservation is not active", "RISK_APPROVAL_INVALID", move="REJECTED")
        for obj, name, pos, want in ((deps.construction, "trade proposal", (1, 3), "PROPOSED"),
                                     (deps.entry_candidate, "entry candidate", (2, 4), "ENTRY_CANDIDATE"),
                                     (deps.setup, "H1 setup", (1, 3), "QUALIFIED")):
            if obj is None:
                return out("DEFER", f"{name} state unavailable - fail closed", "DEPENDENCY_UNAVAILABLE")
            s_now = _state_at_time(obj.transitions, now, pos[0], pos[1], "NONE")
            if s_now != want:
                return out("REJECT", f"{name} is {s_now} (not {want})", "UPSTREAM_INVALIDATED", move="REJECTED")
        ok_perm = ("ALLOW_LONG", "ALLOW_BOTH") if d > 0 else ("ALLOW_SHORT", "ALLOW_BOTH")
        if deps.h4_permission_now not in ok_perm:
            return out("REJECT", f"H4 permission now {deps.h4_permission_now}", "H4_PERMISSION_REVOKED", move="REJECTED")
        codes.append("UPSTREAM_VALID")
        # ---- connection, capabilities, market status --------------------------------------
        conn = port.get_connection_status(now)
        self._beat(st, "last_connection_check")
        if conn.state == "DEGRADED" and not pol.breaker.allow_degraded_connection or conn.state in ("DISCONNECTED", "UNKNOWN"):
            return out("DEFER", f"connection {conn.state}", "CONNECTION_LOST")
        codes.append("CONNECTION_VALID")
        caps = port.get_capabilities(now)
        if caps is None or caps.reported_at is None or (now - caps.reported_at).total_seconds() > fr.capabilities_max_age_seconds:
            return out("DEFER", "capabilities/symbol data stale or missing", "CAPABILITIES_STALE")
        if (it.order_type == "MARKET" and not caps.supports_market) or (it.order_type != "MARKET" and not caps.supports_pending):
            return out("REJECT", "order type not supported by the adapter", "ORDER_TYPE_UNSUPPORTED", move="REJECTED")
        if caps.account_mode == "NETTING" and any(p.symbol == it.symbol for p in port.get_open_positions(now)):
            return out("REJECT", "netting account already holds GBPJPY - a new order would net against it",
                       "NETTING_POSITION_CONFLICT", move="REJECTED")
        snap = port.get_market_snapshot(it.symbol, now)
        if snap is None:
            return out("DEFER", "no market snapshot", "QUOTE_STALE")
        if snap.symbol != it.symbol:
            return out("REJECT", "snapshot symbol mismatch", "SYMBOL_MISMATCH", move="REJECTED")
        age = (now - snap.timestamp).total_seconds()
        info["quote_age_seconds"] = age
        if snap.timestamp > now:
            return out("DEFER", "quote from the future refused", "QUOTE_STALE")
        if age > fr.quote_max_age_seconds:
            st.counters["stale_quotes"] += 1
            if st.counters["stale_quotes"] >= pol.breaker.max_stale_quote_events:
                st.execution_state = "EXECUTION_PAUSED"
                self._event(st, "HALT_TRIGGERED", oid, {"reason": "repeated stale quotes", "state": "EXECUTION_PAUSED"})
            return out("DEFER", f"STALE_QUOTE ({age:.1f}s)", "QUOTE_STALE")
        st.counters["stale_quotes"] = 0
        self._beat(st, "last_quote")
        codes.append("QUOTE_FRESH")
        sess = snap.session_status
        if sess != "OPEN" and not (sess == "UNKNOWN" and pol.breaker.allow_unknown_session):
            return out("DEFER", f"trading session {sess}", "MARKET_CLOSED")
        codes.append("MARKET_OPEN")
        acct = deps.account
        if acct is None or (now - pd.Timestamp(acct.timestamp)).total_seconds() > fr.account_max_age_seconds \
                or pd.Timestamp(acct.timestamp) > now:
            return out("DEFER", "account snapshot stale or missing", "ACCOUNT_STALE")
        self._beat(st, "last_account")
        # ---- bid/ask executable price, deterioration, spread -----------------------------
        bid, ask = self.spec.normalize(snap.bid), self.spec.normalize(snap.ask)
        if not (0 < bid <= ask):
            return out("DEFER", "invalid quote", "QUOTE_STALE")
        exe = ask if d > 0 else bid  # LONG buys the ASK, SHORT sells the BID - never a mid price
        if it.order_type != "MARKET":
            exe = it.pending_price
            geom_ok = {"BUY_LIMIT": exe < ask, "SELL_LIMIT": exe > bid, "BUY_STOP": exe > ask, "SELL_STOP": exe < bid}[it.order_type]
            if not geom_ok:
                return out("REJECT", f"{it.order_type} at {exe} is on the wrong side of the market", "INVALID_ORDER_GEOMETRY",
                           move="REJECTED")
        info["executable_price"] = exe
        pip = float(self.spec.pip_size)
        det = d * (exe - it.reference_entry)
        info["price_deterioration_pips"] = round(det / pip, 2)
        info["price_deterioration_atr"] = round(det / it.atr_price, 4)
        info["directional_deterioration"] = "ADVERSE" if det > 0 else ("FAVOURABLE" if det < 0 else "NONE")
        if det / pip > it.max_deterioration_pips or det / it.atr_price > pol.intent.max_deterioration_atr:
            act = pol.intent.deterioration_action
            return out(act, f"price deteriorated {det / pip:.1f} pips", "PRICE_DETERIORATED",
                       move="REJECTED" if act == "REJECT" else None)
        codes.append("PRICE_VALID")
        spread_p = (ask - bid) / pip
        info["spread_pips"] = round(spread_p, 2)
        info["spread_atr_fraction"] = round((ask - bid) / it.atr_price, 4)
        if deps.spread_history_pips:
            h = sorted(float(x) for x in deps.spread_history_pips)
            info["spread_percentile"] = round(100.0 * sum(x <= spread_p for x in h) / len(h), 1)
        if spread_p > it.max_spread_pips or (ask - bid) / it.atr_price > pol.intent.max_spread_atr_fraction:
            act = pol.intent.spread_action
            return out(act, f"spread {spread_p:.1f} pips too high at submission", "SPREAD_TOO_HIGH",
                       move="REJECTED" if act == "REJECT" else None)
        codes.append("SPREAD_VALID")
        # ---- stop / target geometry ---------------------------------------------------------
        S, T = it.stop_price, it.target_price
        if not ((d > 0 and S < exe < T) or (d < 0 and T < exe < S)):
            return out("REJECT", "stop/target geometry invalid at the executable price", "INVALID_STOPS", move="REJECTED")
        if caps.stop_level_points is not None:
            lvl = float(caps.stop_level_points) * float(self.spec.point)
            if abs(exe - S) < lvl or abs(T - exe) < lvl:
                return out("REJECT", "stop/target inside the adapter-reported stop level", "INVALID_STOPS", move="REJECTED")
        codes.append("STOPS_VALID")
        # ---- risk recheck at the current price (Phase 1F cap is authoritative) ----------
        try:
            fac, used = factor(self.contract.quote_currency, it.account_currency, now, deps.rates,
                               pd.Timedelta(seconds=fr.rate_max_age_seconds))
        except ConversionUnavailable as exc:
            return out("DEFER", f"conversion unavailable: {exc}", "CONVERSION_STALE")
        per_vol = (D(abs(exe - S)) + it.extra_risk_pips * D(self.contract.pip_size)) * D(self.contract.contract_size) * fac
        vol = it.approved_volume
        risk = vol * per_vol
        info["risk_at_approved_volume"] = str(risk)
        if risk > it.approved_risk_amount:
            if not pol.intent.allow_volume_reduction:
                return out("REJECT", "risk would exceed the Phase 1F cap after the price move", "RISK_EXCEEDED_AFTER_PRICE_MOVE",
                           move="REJECTED")
            vol = floor_to_step(it.approved_risk_amount / per_vol, self.contract.volume_step)
            codes.append("VOLUME_REDUCED")
            info["risk_recheck"] = "VOLUME_REDUCTION_REQUIRED"
        else:
            info["risk_recheck"] = "RISK_STILL_VALID"
        # ---- volume validation vs capabilities (never upward) -----------------------------
        vol = min(vol, D(caps.max_volume))
        vol = floor_to_step(vol, caps.volume_step)
        if vol < D(caps.min_volume) or vol <= 0:
            return out("REJECT", f"volume {vol} below the minimum after revalidation", "VOLUME_INVALID",
                       "RISK_EXCEEDED_AFTER_PRICE_MOVE" if info.get("risk_recheck") == "VOLUME_REDUCTION_REQUIRED" else "VOLUME_INVALID",
                       move="REJECTED")
        assert vol <= it.approved_volume and vol * per_vol <= it.approved_risk_amount
        codes += ["RISK_REVALIDATED", "VOLUME_VALID"]
        info["actual_risk_at_submission"] = str(vol * per_vol)
        # ---- honest R:R and room ---------------------------------------------------------------
        risk_d, rew_d = abs(exe - S), abs(T - exe)
        cost = it.cost_extra_pips * pip
        gross, net = rew_d / risk_d, (rew_d - cost) / (risk_d + cost)
        info.update({"gross_R": round(gross, 4), "estimated_net_R": round(net, 4)})
        if net < it.minimum_net_r:
            return out("REJECT", f"net R {net:.2f} < {it.minimum_net_r} at the executable price", "RR_NO_LONGER_VALID",
                       move="REJECTED")
        codes.append("RR_VALID")
        barrier = it.nearest_barrier_price if it.nearest_barrier_price is not None else T
        room = d * (barrier - exe) / it.atr_price
        info["room_atr"] = round(room, 4)
        if room < pol.intent.min_room_atr:
            return out("REJECT", f"room {room:.2f} ATR to the first opposing structure", "ROOM_NO_LONGER_VALID", move="REJECTED")
        codes.append("ROOM_VALID")
        res = out("SUBMIT_READY", "all final gates passed", "SUBMIT_READY",
                  move="SUBMIT_READY" if o["state"] != "SUBMIT_READY" else None)
        res["validated_volume"] = str(vol)
        o["validated_volume"] = str(vol)
        o["last_gate"]["validated_volume"] = str(vol)
        o["last_gate"]["executable_price"] = exe
        o["last_gate"]["spread_pips"] = info["spread_pips"]
        o["last_gate"]["conversion_factor"] = str(fac)
        return res

    # ================================================================== submission
    def submit(self, intent_id: str, port, deps: GateDependencies) -> dict:
        with self.store.transaction():
            st = self._load()
            g = self._gate(st, intent_id, port, deps)
            if g["outcome"] != "SUBMIT_READY":
                self.store.save(st)
                return g
            o = st.orders[intent_id]
            if o["attempts"] >= self.policy.retry.max_attempts:
                self._move(st, intent_id, "REJECTED", "bounded retries exhausted")
                self.store.save(st)
                return {**g, "outcome": "REJECT", "reason_codes": g["reason_codes"] + ["RETRIES_EXHAUSTED"]}
            it = self._intent_from(st.intents[intent_id])
            caps = port.get_capabilities(self.clock.now())
            o["attempts"] += 1
            o["latency"]["submit_requested_at"] = _iso(self.clock.now())
            o["latency"]["submit_mono"] = self.clock.monotonic()
            self._move(st, intent_id, "SUBMISSION_REQUESTED", f"attempt {o['attempts']} with key {o['client_order_key']}")
            self._event(st, "SUBMISSION_REQUESTED", intent_id, {"volume": o["validated_volume"], "attempt": o["attempts"],
                                                                 "executable_price": o["last_gate"]["executable_price"]})
            self.store.save(st)  # write-ahead: the request is recorded BEFORE anything is sent
            t0 = self.clock.monotonic()
            try:
                resp = port.request_submission(it, D(o["validated_volume"]), it.ownership, self.clock.now(),
                                               bool(caps.atomic_protection))
            except Exception as exc:  # the port failed mid-call: outcome unknown
                resp = None
                err = f"{type(exc).__name__}: {exc}"
            st = self._load()
            o = st.orders[intent_id]
            elapsed = self.clock.monotonic() - t0
            if o["state"] != "SUBMISSION_REQUESTED":  # fills/acks already processed re-entrantly
                self.store.save(st)
                return {"outcome": "SUBMITTED", "intent_id": intent_id, "state": o["state"], "reason_codes": ["SUBMITTED"]}
            if resp is None or resp.status in ("TIMEOUT", "DISCONNECTED", "ERROR") or \
                    (resp.status == "ACK" and elapsed > self.policy.retry.submission_timeout_seconds and resp.broker_ref is None):
                why = (resp.status if resp else err) if resp is None or resp.status != "ACK" else "late acknowledgement"
                if resp is not None and resp.status == "DISCONNECTED":
                    st.counters["disconnects"] += 1
                    if st.counters["disconnects"] >= self.policy.breaker.max_disconnects:
                        st.execution_state = "EXECUTION_PAUSED"
                self._move(st, intent_id, "UNKNOWN", f"submission outcome unknown ({why}) - NOT treated as rejected")
                self._move(st, intent_id, "RECONCILIATION_REQUIRED", "reconcile before any re-send")
                self._event(st, "SUBMISSION_TIMEOUT", intent_id, {"why": why, "elapsed": elapsed})
                self.store.save(st)
                return {"outcome": "UNKNOWN", "intent_id": intent_id, "reason_codes": ["SUBMISSION_TIMEOUT", "RECONCILIATION_REQUIRED"]}
            if resp.status == "ACK":
                o["broker_ref"] = resp.broker_ref
                o["latency"]["ack_at"] = _iso(self.clock.now())
                o["latency"]["submit_to_ack_seconds"] = elapsed
                self._beat(st, "last_success")
                st.counters["consecutive_rejections"] = 0
                self._move(st, intent_id, "ACKNOWLEDGED", f"acknowledged as {resp.broker_ref}")
                self._event(st, "ACKNOWLEDGED", intent_id, {"broker_ref": resp.broker_ref})
                self.store.save(st)
                return {"outcome": "ACKNOWLEDGED", "intent_id": intent_id, "broker_ref": resp.broker_ref, "reason_codes": ["ACKNOWLEDGED"]}
            reason = resp.rejection_reason or "UNKNOWN_BROKER_REJECTION"
            st.counters["consecutive_rejections"] += 1
            st.counters["broker_rejections"] = st.counters.get("broker_rejections", 0) + 1
            self._event(st, "REJECTION", intent_id, {"reason": reason, "detail": resp.detail})
            requote = reason in ("PRICE_CHANGED", "REQUOTE", "OFF_QUOTES")
            if requote:
                o["requotes"] += 1
                st.counters["requotes"] = st.counters.get("requotes", 0) + 1
            transient = reason in TRANSIENT_REJECTIONS and o["attempts"] < self.policy.retry.max_attempts and \
                (not requote or o["requotes"] <= self.policy.retry.max_requotes)
            if st.counters["consecutive_rejections"] >= self.policy.breaker.max_consecutive_rejections:
                self._halt(st, f"{st.counters['consecutive_rejections']} consecutive rejections")
            if transient and st.execution_state != "EXECUTION_HALTED":
                r = self.policy.retry
                delay = min(r.backoff_base_seconds * r.backoff_factor ** (o["attempts"] - 1), r.backoff_max_seconds)
                o["next_allowed_at"] = _iso(self.clock.now() + pd.Timedelta(seconds=delay))
                self._move(st, intent_id, "VALIDATING", f"transient rejection {reason}: bounded retry after {delay}s "
                                                        "and a complete re-validation")
            else:
                self._move(st, intent_id, "REJECTED", f"rejected: {reason}")
                self._release(intent_id, st, f"rejected: {reason}")
            self.store.save(st)
            return {"outcome": "BROKER_REJECTED", "intent_id": intent_id, "rejection_reason": reason, "retry_scheduled": transient,
                    "reason_codes": ["BROKER_REJECTED"]}

    def _release(self, oid, st, why) -> None:
        if self.risk_engine is not None:
            self.risk_engine.release_reservation(st.intents[oid]["risk_approval_id"], f"execution: {why}", self.clock.now())

    # ================================================================== broker events
    def process_events(self, port) -> list:
        """Pull and apply events the port has made available up to now (duplicates / disorder tolerated)."""
        out = []
        for ev in port.poll_events(self.clock.now()) if hasattr(port, "poll_events") else []:
            if ev[0] == "ACK":
                out.append(self.on_ack(ev[1], ev[2]))
            elif ev[0] == "FILL":
                out.append(self.on_fill(ev[1]))
        return out

    def _by_key(self, st, key):
        return next((k for k, o in st.orders.items() if o["client_order_key"] == key), None)

    def on_ack(self, client_key: str, broker_ref: str) -> dict:
        with self.store.transaction():
            st = self._load()
            oid = self._by_key(st, client_key)
            if oid is None:
                self._incident(st, "CRITICAL", "ORPHAN_ACK", None, {"client_order_key": client_key})
                self.store.save(st)
                return {"status": "ORPHAN"}
            o = st.orders[oid]
            if o["state"] in ("SUBMISSION_REQUESTED", "UNKNOWN", "RECONCILIATION_REQUIRED"):
                o["broker_ref"] = broker_ref
                o["latency"].setdefault("ack_at", _iso(self.clock.now()))
                self._move(st, oid, "ACKNOWLEDGED", "late acknowledgement")
                res = {"status": "ACKNOWLEDGED"}
            else:
                self._event(st, "DUPLICATE_OR_LATE_ACK_IGNORED", oid, {"state": o["state"], "broker_ref": broker_ref})
                res = {"status": "IGNORED", "state": o["state"]}
            self.store.save(st)
            return res

    def on_fill(self, fill: FillEvent) -> dict:
        with self.store.transaction():
            st = self._load()
            if fill.fill_id in st.fills:
                self._event(st, "DUPLICATE_FILL_IGNORED", st.fills[fill.fill_id]["intent_id"], {"fill_id": fill.fill_id})
                self.store.save(st)
                return {"status": "DUPLICATE_IGNORED"}
            oid = self._by_key(st, fill.client_order_key)
            if oid is None:
                inc = self._incident(st, "CRITICAL", "ORPHAN_FILL", None, {"fill_id": fill.fill_id, "key": fill.client_order_key})
                st.reconciliation["open_issues"].append({"id": inc["id"], "kind": "UNEXPECTED_BROKER_POSITION", "critical": True,
                                                         "detail": {"fill_id": fill.fill_id}})
                self.store.save(st)
                return {"status": "ORPHAN"}
            it = self._intent_from(st.intents[oid])
            o = st.orders[oid]
            allowed = D(o["validated_volume"] or it.approved_volume)
            filled = D(o["filled_volume"]) + D(fill.volume)
            st.fills[fill.fill_id] = {"intent_id": oid, "volume": str(fill.volume), "price": fill.price,
                                      "at": _iso(fill.timestamp), "commission": str(fill.commission), "broker_ref": fill.broker_ref}
            o["fills"].append(fill.fill_id)
            if filled > allowed:
                self._incident(st, "CRITICAL", "OVERFILL", oid, {"filled": str(filled), "allowed": str(allowed)})
                self._halt(st, "fill volume exceeded the approved exposure")
            prev_v, prev_p = D(o["filled_volume"]), D(o["vwap"]) if o["vwap"] else D(0)
            vwap = (prev_v * prev_p + D(fill.volume) * D(fill.price)) / filled
            o["filled_volume"], o["vwap"] = str(filled), str(vwap)
            o["latency"].setdefault("first_fill_at", _iso(fill.timestamp))
            o["latency"]["last_fill_at"] = _iso(fill.timestamp)
            pip = D(self.spec.pip_size)
            slip = D(it.d) * (D(fill.price) - D(it.pending_price if it.pending_price is not None else it.reference_entry)) / pip
            fac = D(o["last_gate"]["conversion_factor"]) if o.get("last_gate") and o["last_gate"].get("conversion_factor") else None
            ledger = {"fill_id": fill.fill_id, "spread_at_submission_pips": (o.get("last_gate") or {}).get("spread_pips"),
                      "commission": str(fill.commission), "slippage_pips": str(slip),
                      "slippage_money": str(slip * pip * D(self.contract.contract_size) * D(fill.volume) * fac) if fac else None,
                      "swap": None, "other_charges": None, "sign_note": "positive = adverse; favourable slippage kept negative"}
            st.fills[fill.fill_id]["ledger"] = ledger
            if slip > D(self.policy.fills.excessive_slippage_pips):
                self._incident(st, "HIGH", "EXCESSIVE_SLIPPAGE", oid, {"slippage_pips": str(slip)})
            state_before = o["state"]
            to = "FILLED" if filled >= allowed else "PARTIALLY_FILLED"
            if state_before in ("CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
                self._event(st, "CANCEL_FILL_CROSSED", oid, {"fill_id": fill.fill_id})
            if to in LEGAL.get(state_before, set()):
                self._move(st, oid, to, f"fill {fill.fill_id} {fill.volume} @ {fill.price}")
            elif state_before == "PARTIALLY_FILLED" and to == "PARTIALLY_FILLED":
                pass
            else:
                inc = self._incident(st, "HIGH", "IMPOSSIBLE_STATE", oid, {"state": state_before, "fill_id": fill.fill_id})
                if state_before not in ("FILLED",):
                    st.reconciliation["open_issues"].append({"id": inc["id"], "kind": "STATE_MISMATCH", "critical": True,
                                                             "intent_id": oid})
            self._event(st, "FILL" if to == "FILLED" else "PARTIAL_FILL", oid, {"fill_id": fill.fill_id, "volume": str(fill.volume),
                                                                                "price": fill.price, "slippage_pips": str(slip)})
            pid = fill.position_id or f"POS-{it.ownership.client_order_key}"
            pos = st.positions.get(pid) or {"position_id": pid, "symbol": it.symbol, "direction": it.direction, "volume": "0",
                                            "average_entry": None, "stop_price": None, "target_price": None,
                                            "opened_at": _iso(fill.timestamp), "unrealised_pnl": "0", "remaining_risk": None,
                                            "execution_status": "OPEN", "protection_status": "PENDING",
                                            "source_order_intent_ids": [oid], "client_order_key": it.ownership.client_order_key}
            pos["volume"], pos["average_entry"] = str(filled), float(vwap)
            if fac:
                pos["remaining_risk"] = str(abs(vwap - D(it.stop_price)) * D(self.contract.contract_size) * filled * fac)
            st.positions[pid] = pos
            o["position_id"] = pid
            o["realised"] = {"average_fill": float(vwap), "entry_deterioration_pips": str(D(it.d) * (vwap - D(it.reference_entry)) / pip),
                             "actual_initial_risk": pos["remaining_risk"],
                             "actual_gross_R": float(abs(D(it.target_price) - vwap) / abs(vwap - D(it.stop_price)))}
            if self.risk_engine is not None and state_before in ("SUBMISSION_REQUESTED", "ACKNOWLEDGED", "UNKNOWN",
                                                                 "RECONCILIATION_REQUIRED") and filled == D(fill.volume):
                self.risk_engine.convert_reservation(it.risk_approval_id, pid, self.clock.now())
            if to == "PARTIALLY_FILLED" and self.policy.fills.partial_fill_policy == "CANCEL_REMAINDER":
                st.control_actions[f"CANCEL-REMAINDER-{oid}"] = {"kind": "CANCEL_REMAINDER", "intent_id": oid, "status": "RECORDED",
                                                          "remaining": str(allowed - filled)}
                self._event(st, "CANCEL_REQUESTED", oid, {"remainder": str(allowed - filled)})
            self.store.save(st)
            return {"status": to, "filled_volume": str(filled), "vwap": str(vwap), "position_id": pid,
                    "reason_codes": ["FULL_FILL" if to == "FILLED" else "PARTIAL_FILL"]}

    # ================================================================== protection
    def ensure_protection(self, intent_id: str, port) -> dict:
        with self.store.transaction():
            st = self._load()
            o = st.orders[intent_id]
            it = self._intent_from(st.intents[intent_id])
            pid = o.get("position_id")
            if pid is None:
                return {"status": "NO_POSITION"}
            pos = st.positions[pid]
            caps = port.get_capabilities(self.clock.now())
            key = f"PROT-{it.ownership.client_order_key}"
            if not caps.atomic_protection:
                resp = port.request_protection(pid, it.stop_price, it.target_price, key, self.clock.now())
                self._event(st, "PROTECTION_REQUESTED", intent_id, {"status": resp.status, "reason": resp.rejection_reason})
            views = {p.position_id: p for p in port.get_open_positions(self.clock.now())}
            v = views.get(pid)
            tol = self.policy.protection.price_tolerance_points * float(self.spec.point) + 1e-9
            problems = []
            if v is None:
                problems.append("position not visible at the adapter")
            else:
                if v.symbol != it.symbol:
                    problems.append("symbol mismatch")
                if v.direction != it.direction:
                    problems.append("direction mismatch")
                if D(v.volume) != D(pos["volume"]):
                    problems.append("volume mismatch")
                if self.policy.protection.require_stop and (v.stop_price is None or abs(v.stop_price - it.stop_price) > tol):
                    problems.append(f"stop not confirmed at {it.stop_price} (adapter shows {v.stop_price})")
                if self.policy.protection.require_target and (v.target_price is None or abs(v.target_price - it.target_price) > tol):
                    problems.append(f"target not confirmed at {it.target_price} (adapter shows {v.target_price})")
            if problems and any("stop" in p or "position" in p or "direction" in p or "volume" in p or "symbol" in p for p in problems):
                pos["protection_status"] = "UNPROTECTED_POSITION"
                inc = self._incident(st, "CRITICAL", "UNPROTECTED_POSITION", intent_id, {"position_id": pid, "problems": problems})
                self._halt(st, f"unprotected position {pid}")
                st.counters["protection_failures"] = st.counters.get("protection_failures", 0) + 1
                if self.policy.protection.unprotected_action == "EMERGENCY_CLOSE_INTENT":
                    k = f"EMERGENCY-CLOSE-{pid}"
                    st.control_actions.setdefault(k, {"kind": "EMERGENCY_CLOSE", "position_id": pid, "volume": pos["volume"],
                                               "reason": "UNPROTECTED_POSITION", "incident": inc["id"], "at": _iso(self.clock.now()),
                                               "idempotency_key": k, "status": "INTENT_RECORDED_NOT_SENT"})
                    self._event(st, "EMERGENCY_CLOSE_INTENT", intent_id, {"position_id": pid, "key": k})
                self.store.save(st)
                return {"status": "UNPROTECTED_POSITION", "problems": problems, "reason_codes": ["UNPROTECTED_POSITION", "EXECUTION_HALTED"]}
            pos["protection_status"] = "PROTECTED" if not problems else "PROTECTED_STOP_ONLY"
            pos["stop_price"], pos["target_price"] = it.stop_price, (it.target_price if not problems else v.target_price)
            self._event(st, "PROTECTION_ATTACHED", intent_id, {"position_id": pid, "status": pos["protection_status"],
                                                               "problems": problems})
            self.store.save(st)
            return {"status": pos["protection_status"], "reason_codes": ["PROTECTION_VALID"], "problems": problems}

    # ================================================================== reconciliation
    def reconcile(self, port, startup: bool = False) -> dict:
        """Broker views are the execution source of truth; mismatches are recorded, never silently overwritten."""
        now = self.clock.now()
        with self.store.transaction():
            st = self._load()
            orders = port.get_pending_orders(now)
            positions = port.get_open_positions(now)
            issues, foreign, adopted = [], [], []
            ours_num = self.identity.numeric_id
            known_keys = {o["client_order_key"]: k for k, o in st.orders.items()}
            b_orders = {o.client_order_key: o for o in orders if o.client_order_key}
            b_pos = {p.client_order_key: p for p in positions if p.client_order_key}
            for o in orders:
                if o.client_order_key is None and o.strategy_numeric_id != ours_num:
                    foreign.append({"kind": "order", "broker_ref": o.broker_ref})
                elif o.client_order_key not in known_keys:
                    issues.append({"kind": "ORPHAN_BROKER_ORDER", "critical": True, "detail": {"broker_ref": o.broker_ref}})
            for p in positions:
                if p.client_order_key is None and p.strategy_numeric_id != ours_num:
                    foreign.append({"kind": "position", "position_id": p.position_id})
                elif p.client_order_key not in known_keys:
                    issues.append({"kind": "UNEXPECTED_BROKER_POSITION", "critical": True, "detail": {"position_id": p.position_id}})
            for key, oid in known_keys.items():
                o = st.orders[oid]
                it = st.intents[oid]
                if o["state"] in ("SUBMISSION_REQUESTED", "UNKNOWN", "RECONCILIATION_REQUIRED"):
                    if o["state"] == "SUBMISSION_REQUESTED":
                        self._move(st, oid, "UNKNOWN", "reconciliation of an unacknowledged submission")
                    if o["state"] == "UNKNOWN":
                        self._move(st, oid, "RECONCILIATION_REQUIRED", "reconciling")
                    fills = port.get_fills(key, now)
                    if key in b_pos or fills:
                        o["broker_ref"] = o["broker_ref"] or (b_orders[key].broker_ref if key in b_orders else None)
                        adopted.append(oid)
                        self._event(st, "RECONCILED_FOUND", oid, {"fills": len(fills)})
                        self.store.save(st)
                        for f in fills:
                            self.on_fill(f)
                        st = self._load()
                    elif key in b_orders:
                        o["broker_ref"] = b_orders[key].broker_ref
                        self._move(st, oid, "ACKNOWLEDGED", "reconciliation: the order exists at the adapter")
                        adopted.append(oid)
                    else:
                        status = port.get_order_status(key, now)
                        if status == "NOT_FOUND":
                            nxt = "EXPIRED" if now >= pd.Timestamp(it["expires_at"]) else "VALIDATING"
                            self._move(st, oid, nxt, "reconciliation proved the order absent (never received)")
                            if nxt == "EXPIRED":
                                self._release(oid, st, "expired after reconciliation")
                        else:
                            issues.append({"kind": "SUBMISSION_STATUS_UNRESOLVED", "critical": True, "intent_id": oid,
                                           "detail": {"status": status}})
                elif o["state"] in ("ACKNOWLEDGED", "PARTIALLY_FILLED", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"):
                    fills = port.get_fills(key, now)
                    unseen = [f for f in fills if f.fill_id not in st.fills]
                    if unseen:  # fills the event stream missed: adopt them (duplicates are ignored by fill id)
                        self._event(st, "RECONCILED_MISSED_FILLS", oid, {"fills": len(unseen)})
                        self.store.save(st)
                        for f in unseen:
                            self.on_fill(f)
                        st = self._load()
                    elif key not in b_orders and key not in b_pos and not fills:
                        issues.append({"kind": "MISSING_BROKER_ORDER", "critical": True, "intent_id": oid})
            for pid, pos in st.positions.items():
                if pos.get("execution_status") != "OPEN":
                    continue
                v = b_pos.get(pos.get("client_order_key"))
                if v is None:
                    issues.append({"kind": "MISSING_BROKER_POSITION", "critical": True, "position_id": pid})
                    continue
                if D(v.volume) != D(pos["volume"]):
                    issues.append({"kind": "VOLUME_MISMATCH", "critical": True, "position_id": pid,
                                   "detail": {"internal": pos["volume"], "adapter": str(v.volume)}})
                if v.direction != pos["direction"]:
                    issues.append({"kind": "STATE_MISMATCH", "critical": True, "position_id": pid, "detail": "direction"})
                if pos["average_entry"] is not None and abs(v.average_price - pos["average_entry"]) > float(self.spec.point):
                    issues.append({"kind": "PRICE_MISMATCH", "critical": False, "position_id": pid,
                                   "detail": {"internal": pos["average_entry"], "adapter": v.average_price}})
                if pos["protection_status"] == "PROTECTED" and v.stop_price != pos["stop_price"]:
                    issues.append({"kind": "STOP_MISMATCH", "critical": True, "position_id": pid,
                                   "detail": {"internal": pos["stop_price"], "adapter": v.stop_price}})
                if pos["protection_status"] == "PROTECTED" and v.target_price != pos["target_price"]:
                    issues.append({"kind": "TARGET_MISMATCH", "critical": False, "position_id": pid,
                                   "detail": {"internal": pos["target_price"], "adapter": v.target_price}})
            for i in issues:
                i["id"] = f"REC-{len(st.events) + 1}-{i['kind']}"
                self._event(st, "RECONCILIATION_MISMATCH", i.get("intent_id"), i)
            crit = [i for i in issues if i["critical"]]
            st.reconciliation["open_issues"] = issues
            st.reconciliation["last_run"] = _iso(now)
            st.reconciliation["foreign"] = foreign
            if crit:
                st.counters["mismatches"] += 1
                if st.counters["mismatches"] >= self.policy.breaker.max_reconciliation_mismatches:
                    self._halt(st, "repeated critical reconciliation mismatches")
            if startup:
                st.reconciliation["startup_done"] = not crit
            self._beat(st, "last_reconciliation")
            self._event(st, "RECONCILIATION_RUN", None, {"startup": startup, "issues": len(issues), "critical": len(crit),
                                                         "foreign": len(foreign), "adopted": adopted})
            self.store.save(st)
            return {"issues": issues, "critical": len(crit), "foreign": foreign, "adopted": adopted,
                    "startup_done": st.reconciliation["startup_done"],
                    "reason_codes": ["RECONCILIATION_REQUIRED"] if crit else ["RECONCILIATION_CLEAN"]}

    def resolve_issue(self, issue_id: str, operator: str, resolution: str) -> dict:
        with self.store.transaction():
            st = self._load()
            before = len(st.reconciliation["open_issues"])
            st.reconciliation["open_issues"] = [i for i in st.reconciliation["open_issues"] if i.get("id") != issue_id]
            for inc in st.incidents:
                if inc["id"] == issue_id:
                    inc["open"] = False
            self._event(st, "ISSUE_RESOLVED", None, {"id": issue_id, "operator": operator, "resolution": resolution})
            self.store.save(st)
            return {"resolved": before != len(st.reconciliation["open_issues"]) or any(i["id"] == issue_id for i in st.incidents)}

    # ================================================================== cancellation / expiry
    def request_cancel(self, intent_id: str, port, reason: str) -> dict:
        with self.store.transaction():
            st = self._load()
            o = st.orders[intent_id]
            if o["state"] in ("CREATED", "VALIDATING", "SUBMIT_READY"):
                self._move(st, intent_id, "CANCELLED", f"cancelled before submission: {reason}")
                self._release(intent_id, st, "cancelled before submission")
                self.store.save(st)
                return {"status": "CANCELLED"}
            if o["state"] not in ("ACKNOWLEDGED", "PARTIALLY_FILLED", "CANCEL_FAILED"):
                return {"status": "NOT_CANCELLABLE", "state": o["state"]}
            key = f"CANCEL-{o['client_order_key']}"
            self._move(st, intent_id, "CANCEL_REQUESTED", reason)
            self.store.save(st)
            from .model import CancelRequest
            resp = port.request_cancellation(CancelRequest(intent_id, o["broker_ref"], reason, self.clock.now(), key), self.clock.now())
            st = self._load()
            o = st.orders[intent_id]
            if o["state"] != "CANCEL_REQUESTED":
                self.store.save(st)
                return {"status": o["state"]}
            if resp.status == "ACK":
                self._move(st, intent_id, "CANCEL_ACKNOWLEDGED", "cancel acknowledged - awaiting confirmation")
                status = port.get_order_status(o["client_order_key"], self.clock.now())
                if status == "CANCELLED":
                    self._move(st, intent_id, "CANCELLED", "adapter confirms the order is cancelled")
                    if D(o["filled_volume"]) == 0:
                        self._release(intent_id, st, "cancelled without any fill")
            elif resp.status == "REJECTED":
                self._move(st, intent_id, "CANCEL_FAILED", resp.rejection_reason or "cancel rejected")
            else:
                self._move(st, intent_id, "RECONCILIATION_REQUIRED", f"cancel outcome unknown ({resp.status})")
            self.store.save(st)
            return {"status": st.orders[intent_id]["state"]}

    def expire_stale_intents(self, port) -> list:
        """Expired strategy intent must not leave a working order behind."""
        now, done = self.clock.now(), []
        for oid, o in list(self.state().orders.items()):
            it = self.state().intents[oid]
            if now < pd.Timestamp(it["expires_at"]):
                continue
            if o["state"] in ("CREATED", "VALIDATING", "SUBMIT_READY"):
                with self.store.transaction():
                    st = self._load()
                    self._move(st, oid, "EXPIRED", "validity window elapsed")
                    self._release(oid, st, "intent expired")
                    self.store.save(st)
                done.append(oid)
            elif o["state"] == "ACKNOWLEDGED":
                done.append(oid)
                self.request_cancel(oid, port, "pending intent expired")
        return done

    # ================================================================== modification / close contracts
    def request_modification(self, req, port=None) -> dict:
        """Contract only: risk-valid, idempotent, auditable.  A stop can never be widened (nor a target extended)
        without an explicit future risk-policy approval; trailing logic is not active."""
        with self.store.transaction():
            st = self._load()
            if req.idempotency_key in st.control_actions:
                return dict(st.control_actions[req.idempotency_key], idempotent_replay=True)
            pos = st.positions.get(req.position_id)
            rec = {"kind": f"MODIFY_{req.field}", "position_id": req.position_id, "new_price": req.new_price,
                   "old_price": req.old_price, "reason": req.reason, "at": _iso(req.timestamp)}
            if pos is None:
                rec["status"] = "REJECTED_UNKNOWN_POSITION"
            else:
                d = 1 if pos["direction"] == "LONG" else -1
                ref = pos["average_entry"]  # without a port: the stop must stay on the loss side of the entry
                snap = port.get_market_snapshot(pos["symbol"], self.clock.now()) if port is not None else None
                if snap is not None:
                    ref = snap.bid if d > 0 else snap.ask  # the price the position would exit at
                if req.field == "STOP":
                    widens = pos["stop_price"] is not None and d * (pos["stop_price"] - req.new_price) > 0
                    wrong = d * (ref - req.new_price) <= 0
                    if widens and req.risk_policy_approval is None:
                        rec["status"] = "REJECTED_WOULD_WIDEN_RISK"
                    elif wrong:
                        rec["status"] = "REJECTED_INVALID_GEOMETRY"
                    else:
                        rec["status"] = "ACCEPTED_CONTRACT"
                        rec["note"] = "moving a stop never guarantees less realised risk (gaps/slippage remain)"
                elif req.field == "TARGET":
                    extends = pos["target_price"] is not None and d * (req.new_price - pos["target_price"]) > 0
                    rec["status"] = "REJECTED_TARGET_EXTENSION" if extends and req.risk_policy_approval is None else "ACCEPTED_CONTRACT"
                else:
                    rec["status"] = "REJECTED_UNKNOWN_FIELD"
            if rec["status"] == "ACCEPTED_CONTRACT" and port is not None:
                resp = port.request_modification(req, self.clock.now())
                rec["port_status"] = resp.status
            st.control_actions[req.idempotency_key] = rec
            self._event(st, rec["kind"], None, rec)
            self.store.save(st)
            return rec

    def request_close(self, req, port=None) -> dict:
        """Contract for full/partial/emergency close (idempotent). Nothing is sent without an explicit port."""
        with self.store.transaction():
            st = self._load()
            if req.idempotency_key in st.control_actions:
                return dict(st.control_actions[req.idempotency_key], idempotent_replay=True)
            pos = st.positions.get(req.position_id)
            rec = {"kind": "EMERGENCY_CLOSE" if req.emergency else "CLOSE", "position_id": req.position_id,
                   "volume": str(req.volume), "reason": req.reason, "at": _iso(req.timestamp), "idempotency_key": req.idempotency_key}
            if pos is None:
                rec["status"] = "REJECTED_UNKNOWN_OR_FOREIGN_POSITION"
            elif D(req.volume) <= 0 or D(req.volume) > D(pos["volume"]):
                rec["status"] = "REJECTED_INVALID_VOLUME"
            else:
                rec["status"] = "PARTIAL_CLOSE_CONTRACT" if D(req.volume) < D(pos["volume"]) else "FULL_CLOSE_CONTRACT"
                if port is not None:
                    rec["port_status"] = port.request_close(req, self.clock.now()).status
            st.control_actions[req.idempotency_key] = rec
            self._event(st, rec["kind"], None, rec)
            self.store.save(st)
            return rec

    # ================================================================== halts, health, metrics, audit
    def halt(self, reason: str, operator: str) -> dict:
        with self.store.transaction():
            st = self._load()
            self._halt(st, f"manual: {reason} (by {operator})", "MANUAL")
            self.store.save(st)
            return {"execution_state": st.execution_state}

    def pause(self, reason: str) -> dict:
        with self.store.transaction():
            st = self._load()
            if st.execution_state == "EXECUTION_ENABLED":
                st.execution_state = "EXECUTION_PAUSED"
                self._event(st, "EXECUTION_PAUSED", None, {"reason": reason})
            self.store.save(st)
            return {"execution_state": st.execution_state}

    def reset(self, auth) -> dict:
        if getattr(auth, "confirmation", None) != RESET_CONFIRMATION or not auth.operator or not auth.reason:
            return {"status": "REFUSED"}
        with self.store.transaction():
            st = self._load()
            prev = st.execution_state
            st.execution_state, st.halt_reason, st.halt_source = "EXECUTION_ENABLED", None, None
            st.counters.update({"consecutive_rejections": 0, "stale_quotes": 0, "disconnects": 0, "mismatches": 0})
            self._event(st, "AUTHORISED_RESET", None, {"from": prev, "operator": auth.operator, "reason": auth.reason})
            self.store.save(st)
            return {"status": "RESET", "from": prev}

    def health(self, port=None) -> dict:
        st = self.state()
        conn = port.get_connection_status(self.clock.now()).state if port is not None else "UNKNOWN"
        open_pos = [p for p in st.positions.values() if p.get("execution_status") == "OPEN"]
        monitoring = "DEGRADED_MONITORING" if open_pos and conn != "CONNECTED" else "OK"
        return {"execution_state": st.execution_state, "connection": conn, "monitoring": monitoring,
                "heartbeat": dict(st.heartbeat), "open_positions": len(open_pos),
                "note": "a disconnected terminal does not remove an attached server-side stop, but monitoring is degraded"}

    def metrics(self) -> dict:
        st = self.state()
        orders = list(st.orders.values())
        n = max(len(orders), 1)
        slips = [float(f["ledger"]["slippage_pips"]) for f in st.fills.values() if f.get("ledger")]
        lat = [o["latency"].get("submit_to_ack_seconds") for o in orders if o["latency"].get("submit_to_ack_seconds") is not None]
        return {"intents": len(orders), "filled": sum(o["state"] == "FILLED" for o in orders),
                "rejected": sum(o["state"] == "REJECTED" for o in orders),
                "acceptance_rate": sum(o["state"] in ("FILLED", "PARTIALLY_FILLED", "ACKNOWLEDGED") for o in orders) / n,
                "partial_fill_rate": sum(len(o["fills"]) > 1 for o in orders) / n,
                "broker_rejections": st.counters.get("broker_rejections", 0), "requotes": st.counters.get("requotes", 0),
                "reconciliation_incidents": sum(e["type"] == "RECONCILIATION_MISMATCH" for e in st.events),
                "protection_failures": st.counters.get("protection_failures", 0),
                "mean_slippage_pips": (sum(slips) / len(slips)) if slips else None,
                "mean_submit_to_ack_seconds": (sum(lat) / len(lat)) if lat else None,
                "spread_at_submission_pips": [o["last_gate"].get("spread_pips") for o in orders if o.get("last_gate")],
                "note": "measurement only - never used to optimise the strategy"}

    def latency(self, intent_id: str) -> dict:
        o = self.state().orders[intent_id]["latency"]

        def sec(a, b):
            return (pd.Timestamp(o[b]) - pd.Timestamp(o[a])).total_seconds() if o.get(a) and o.get(b) else None

        return {"decision_to_submit": sec("decision_at", "submit_requested_at"), "submit_to_ack": o.get("submit_to_ack_seconds"),
                "ack_to_fill": sec("ack_at", "first_fill_at"), "total": sec("decision_at", "last_fill_at")}

    def audit(self, intent_id: str) -> dict:
        st = self.state()
        it, o = st.intents[intent_id], st.orders[intent_id]
        return {"why_strategy_qualified": it["strategy_metadata"], "why_risk_approved": it["risk_metadata"],
                "execution_conditions": o.get("last_gate"), "requested_price": it["reference_entry"],
                "adapter_acknowledgement": o.get("broker_ref"),
                "fills": [st.fills[f] for f in o["fills"]], "realised": o.get("realised"),
                "protection": (st.positions.get(o.get("position_id")) or {}).get("protection_status"),
                "errors_and_retries": [e for e in st.events if e["intent_id"] == intent_id and e["type"] in
                                       ("REJECTION", "SUBMISSION_TIMEOUT", "INCIDENT", "GATE_DEFER", "GATE_REJECT")],
                "lifecycle": o["history"], "final_state": o["state"],
                "events": [e for e in st.events if e["intent_id"] == intent_id]}
