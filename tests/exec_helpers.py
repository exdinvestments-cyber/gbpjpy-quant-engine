"""Shared world for Phase 1G execution tests: a real Phase 1F approval, explicit upstream
states, a deterministic clock and the testing-only paper port."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from gbpjpy_engine.execution import (ExecutionEngine, ExecutionPolicy, FaultPlan, GateDependencies, InMemoryExecutionStore,
                                     PaperBroker, SimClock)
from risk_helpers import T0, account, engine, proposal, rates

H = pd.Timedelta(hours=1)


def upstream(t=T0):
    """Phase 1C/1D/1E objects reduced to their transition histories (all valid at ``t``)."""
    setup = SimpleNamespace(transitions=[(0, t - 3 * H, "NO_SETUP", "WATCHING", "w"), (1, t - H, "WATCHING", "QUALIFIED", "q")])
    entry = SimpleNamespace(transitions=[(1, "OPEN", t, "WAITING_FOR_CONFIRMATION", "ENTRY_CANDIDATE", "accept")])
    cons = SimpleNamespace(transitions=[(1, t, "EVALUATING_RISK_REWARD", "PROPOSED", "proposed")])
    return setup, entry, cons


def exec_policy(**sections) -> ExecutionPolicy:
    base = ExecutionPolicy()
    return replace(base, **{k: replace(getattr(base, k), **v) for k, v in sections.items()})


class World:
    """One approved LONG (or SHORT) trade, a paper port with a fresh quote and a clock just after the decision."""

    def __init__(self, direction=1, policy=None, faults=None, store=None, account_mode="HEDGING", atomic=True, pid="P-1",
                 bid=None, ask=None, clock_offset=1.0, spread_pips=2.0, risk_store=None, approve=True, **port_kw):
        self.risk = engine(store=risk_store) if risk_store is not None else engine()
        self.proposal = proposal(direction, pid=pid)
        self.approval = self.risk.approve(self.proposal, account(), T0) if approve else None
        self.clock = SimClock(T0 + pd.Timedelta(seconds=clock_offset))
        self.port = PaperBroker(account_mode=account_mode, atomic_protection=atomic, faults=faults or FaultPlan(), **port_kw)
        self.port.report_capabilities(T0)
        ref = self.proposal["executable_reference_price"]
        sp = spread_pips * 0.01
        if direction > 0:
            ask = ref if ask is None else ask
            bid = round(ask - sp, 3) if bid is None else bid
        else:
            bid = ref if bid is None else bid
            ask = round(bid + sp, 3) if ask is None else ask
        self.port.add_quote(bid, ask, T0)
        self.setup, self.entry, self.cons = upstream()
        self.rates = rates()
        self.store = store or InMemoryExecutionStore()
        self.engine = ExecutionEngine(policy or ExecutionPolicy(), store=self.store, clock=self.clock, risk_engine=self.risk)
        self.h4 = "ALLOW_LONG" if direction > 0 else "ALLOW_SHORT"

    def deps(self, **over) -> GateDependencies:
        kw = dict(risk_state=self.risk.state(), account=account(ts=self.clock.now() - pd.Timedelta(seconds=1)),
                  construction=self.cons, entry_candidate=self.entry, setup=self.setup, h4_permission_now=self.h4,
                  rates=self.rates)
        kw.update(over)
        return GateDependencies(**kw)

    def ready(self, **intent_kw):
        """Startup reconciliation + intent creation."""
        self.engine.reconcile(self.port, startup=True)
        self.intent = self.engine.create_intent(self.approval, self.proposal, **intent_kw)
        return self.intent

    def quote(self, bid, ask, dt=0.0, session="OPEN"):
        self.port.add_quote(bid, ask, self.clock.now() + pd.Timedelta(seconds=dt), session)

    def advance(self, seconds):
        return self.clock.advance(seconds)

    def submit(self, **over):
        return self.engine.submit(self.intent.order_intent_id, self.port, self.deps(**over))

    def gate(self, **over):
        return self.engine.gate(self.intent.order_intent_id, self.port, self.deps(**over))

    def order(self):
        return self.engine.state().orders[self.intent.order_intent_id]

    def restart(self, **kw):
        """A new process over the same persisted store (and the same paper port - the adapter survives)."""
        self.engine = ExecutionEngine(kw.pop("policy", self.engine.policy), store=self.store, clock=self.clock,
                                      risk_engine=self.risk)
        return self.engine
