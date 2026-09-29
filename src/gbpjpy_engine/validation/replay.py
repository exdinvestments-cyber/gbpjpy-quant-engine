"""Point-in-time historical replay (Phase 1H).

Layer 1 - strategy decisions: the unchanged Phase 1A-1E engines run over the
dataset.  They are already causal (every feature is available only at its bar
close; decisions happen at the next bar OPEN), so a single pass is identical to
bar-by-bar replay; ``test_validation_lookahead`` proves it by mutating the
future and checking that earlier decisions do not change.

Layer 2 - strategy trade ledger (R-based edge analysis): each Phase 1E
proposal is filled at its decision bar OPEN and simulated to exit under an
explicit cost scenario.  Exposure rule: at most one open trade per direction
(mirrors the Phase 1F default that forbids pyramiding).  The accepted-trade
PATH is fixed by the BASELINE scenario and then re-priced under every other
scenario, so signals and trade selection are identical across cost scenarios
and only costs differ.

Layer 3 - account simulation (compounding, reported separately): the same path
through the Phase 1F ``AccountRiskEngine`` (sizing, limits, halts) and the
Phase 1G ``ExecutionEngine`` (order intent, final gate, submission, fill,
protection) driven by the deterministic paper port loaded with historical
quotes.  A trade's close is recorded only once simulated time reaches it.

No network, no platform, no real order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from decimal import Decimal

import numpy as np
import pandas as pd

from ..config import H4Config
from ..engine import H4MarketIntelligenceEngine
from ..entry import EntryConfig, EntryIntelligenceEngine
from ..execution import (ExecutionEngine, ExecutionPolicy, FaultPlan, GateDependencies, InMemoryExecutionStore, PaperBroker,
                         SimClock)
from ..h1 import H1Config, H1SetupEngine
from ..risk import AccountRiskEngine, InMemoryRiskStateStore, OpenPosition, RiskPolicy, rates_from_bars
from ..risk.money import D
from ..risk.research import ResearchAccount
from ..trade import TradeConfig, TradeConstructionEngine
from .costs import CostScenario
from .resampling import UTC_GRID, H4BarDefinition, resample_h1_to_h4
from .simulator import ExitPlan, SimulatedTrade, simulate_trade


@dataclass(frozen=True)
class StrategyConfigs:
    h4: H4Config = field(default_factory=H4Config)
    h1: H1Config = field(default_factory=H1Config)
    entry: EntryConfig = field(default_factory=EntryConfig)
    trade: TradeConfig = field(default_factory=TradeConfig)
    risk: RiskPolicy = field(default_factory=RiskPolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy)


@dataclass
class PipelineOutputs:
    h1_bars: pd.DataFrame
    h4_bars: pd.DataFrame
    h4: object
    h1: object
    entry: object
    trade: object
    configs: StrategyConfigs
    h4_definition: dict
    runtime_seconds: float
    ts_index: dict = field(default_factory=dict)

    def index_of(self, t) -> int | None:
        return self.ts_index.get(pd.Timestamp(t))


def run_pipeline(h1_bars: pd.DataFrame, h4_bars: pd.DataFrame | None = None, configs: StrategyConfigs | None = None,
                 h4_definition: H4BarDefinition = UTC_GRID) -> PipelineOutputs:
    """Run the unchanged Phase 1A-1E engines (strategy decisions only)."""
    cfg = configs or StrategyConfigs()
    t0 = time.perf_counter()
    h4_report = {"source": "provider"}
    if h4_bars is None:
        h4_bars, h4_report = resample_h1_to_h4(h1_bars, h4_definition)
    h4r = H4MarketIntelligenceEngine(cfg.h4).run(h4_bars)
    h1r = H1SetupEngine(cfg.h1).run(h1_bars, h4r)
    en = EntryIntelligenceEngine(cfg.entry).run(h1r, h4r)
    tr = TradeConstructionEngine(cfg.trade).run(en, h1r, h4r)
    out = PipelineOutputs(h1_bars, h4_bars, h4r, h1r, en, tr, cfg, h4_report, time.perf_counter() - t0)
    out.ts_index = {pd.Timestamp(t): i for i, t in enumerate(h1r.features["timestamp"])}
    return out


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------
def warmup(out: PipelineOutputs) -> dict:
    """First H1 bar at which every component is fully formed; nothing earlier is scored."""
    f1, f4, fr = out.h1.features, out.h4.features, out.h1.frame
    h1_ok = np.flatnonzero(f1["warmup_complete"].to_numpy(bool))
    h4_ok = np.flatnonzero(f4["warmup_complete"].to_numpy(bool))
    ctx_ok = np.flatnonzero((fr["h4_context_status"] == "OK").to_numpy())
    if not len(h1_ok) or not len(h4_ok) or not len(ctx_ok):
        return {"complete": False, "index": None, "timestamp": None,
                "reason": "insufficient history for warm-up (no scoring possible)"}
    h4_ready_at = pd.Timestamp(f4["available_at"].iloc[h4_ok[0]])
    h4_idx = int((pd.to_datetime(f1["timestamp"], utc=True) < h4_ready_at.tz_convert("UTC")).sum())
    ppl = int(out.configs.entry.spread.history_min)
    stop_pct = int(out.configs.trade.stop.percentile_lookback)
    idx = max(int(h1_ok[0]), h4_idx, int(ctx_ok[0]), ppl, stop_pct)
    idx = min(idx, len(f1) - 1)
    return {"complete": True, "index": idx, "timestamp": pd.Timestamp(f1["timestamp"].iloc[idx]).isoformat(),
            "components": {"h1_features_warmup_index": int(h1_ok[0]), "h4_features_ready_h1_index": h4_idx,
                           "h4_context_ok_index": int(ctx_ok[0]), "entry_spread_history_bars": ppl,
                           "stop_percentile_lookback_bars": stop_pct},
            "note": "EMA/ATR/percentile/swing/zone/structure/regime histories are all complete from this bar"}


# ---------------------------------------------------------------------------
# Signal ledger
# ---------------------------------------------------------------------------
def signal_ledger(out: PipelineOutputs, account_decisions: dict | None = None) -> pd.DataFrame:
    """One row per historical opportunity, rejected ones included."""
    fr = out.h1.frame
    cand_by_setup = {c.setup_id: c for c in out.entry.candidates}
    cons_by_cand = {c.entry_candidate_id: c for c in out.trade.constructions}
    acc = account_decisions or {}
    rows = []
    for s in out.h1.setups:
        i = int(s.created_index)
        q = s.qualified or {}
        c = cand_by_setup.get(s.setup_id)
        k = cons_by_cand.get(c.entry_candidate_id) if c else None
        p = k.proposal if (k is not None and k.decision == "PROPOSE_TRADE" and k.proposal) else None
        a = (c.accepted or {}) if c else {}
        pid = k.trade_proposal_id if k is not None else None
        ad = acc.get(pid, {}) if pid else {}
        rows.append({
            "opportunity_id": s.setup_id, "stage": "H1_SETUP", "created_at": s.created_at, "side": s.side.upper(),
            "setup_family": s.family, "h4_regime": fr["h4_regime"].iloc[i], "h4_permission": fr["h4_permission"].iloc[i],
            "h4_context_status": fr["h4_context_status"].iloc[i], "h4_primary_structure": fr["h4_primary_structure"].iloc[i],
            "setup_state": s.state, "setup_end_reason": s.end_reason, "qualified": bool(s.qualified),
            "setup_score": q.get("setup_score", s.peak_score), "setup_confidence": q.get("setup_confidence"),
            "entry_decision": c.decision if c else None, "entry_end_category": c.end_category if c else None,
            "entry_quality_score": a.get("entry_quality_score"), "confirmation_family": a.get("confirmation_family"),
            "trade_proposal_id": pid, "trade_decision": k.decision if k is not None else None,
            "trade_quality_score": p.get("trade_construction_quality_score") if p else None,
            "risk_decision": ad.get("risk_decision"), "execution_decision": ad.get("execution_decision"),
            "reason_codes": sorted(set((q.get("reason_codes") or []) + (a.get("reason_codes") or []) +
                                       ((k.reason_codes if k is not None else None) or []) + (ad.get("reason_codes") or []))),
        })
    seen = set()
    for g in out.h1.counterfactual:
        if g.get("kind") != "GATED_BY_H4":
            continue
        key = (g["side"], g["family"], g["anchor"])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"opportunity_id": f"GATED-{g['side']}-{g['family']}-{g['anchor']}", "stage": "H4_GATED",
                     "created_at": g["at"], "side": g["side"].upper(), "setup_family": g["family"],
                     "h4_permission": g.get("h4_permission"), "setup_state": "GATED_BY_H4",
                     "setup_score": g.get("setup_score"), "setup_confidence": g.get("setup_confidence"),
                     "reason_codes": sorted(g.get("blockers") or [])})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy trade ledger (R)
# ---------------------------------------------------------------------------
@dataclass
class TradeLedger:
    scenario: dict
    trades: list  # SimulatedTrade (scored)
    excluded: list  # {"trade_proposal_id", "reason"}
    path: list  # accepted proposal ids in order
    ambiguity_policy: str
    price_type: str

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])


def _proposals(out: PipelineOutputs) -> list:
    return sorted(out.trade.proposals, key=lambda p: (pd.Timestamp(p["timestamp"]), p["trade_proposal_id"]))


def strategy_trades(out: PipelineOutputs, scenario: CostScenario, price_type: str, ambiguity_policy: str = "CONSERVATIVE",
                    ltf: pd.DataFrame | None = None, path: list | None = None, blackout: set | None = None,
                    exit_plan: ExitPlan | None = None, restrict_to: tuple | None = None) -> TradeLedger:
    """Simulate proposals; ``path`` (from the BASELINE run) fixes which proposals are traded."""
    w = warmup(out)
    bars = out.h1.features
    black = blackout or set()
    trades, excluded, taken = [], [], []
    open_until = {"LONG": None, "SHORT": None}
    lo, hi = (pd.Timestamp(restrict_to[0]), pd.Timestamp(restrict_to[1])) if restrict_to else (None, None)
    for p in _proposals(out):
        pid, t = p["trade_proposal_id"], pd.Timestamp(p["timestamp"])
        if lo is not None and not (lo <= t < hi):
            continue
        idx = out.index_of(t)
        if idx is None:
            excluded.append({"trade_proposal_id": pid, "reason": "DECISION_BAR_MISSING"})
            continue
        if not w["complete"] or idx < w["index"]:
            excluded.append({"trade_proposal_id": pid, "reason": "WARMUP"})
            continue
        if idx in black:
            excluded.append({"trade_proposal_id": pid, "reason": "DATA_GAP_BLACKOUT"})
            continue
        if path is not None:
            if pid not in path:
                excluded.append({"trade_proposal_id": pid, "reason": "NOT_IN_FIXED_PATH"})
                continue
        else:
            busy = open_until[p["direction"]]
            if busy is not None and busy >= idx:
                excluded.append({"trade_proposal_id": pid, "reason": "EXPOSURE_ONE_PER_DIRECTION"})
                continue
        tr = simulate_trade(bars, idx, p, scenario, price_type, ambiguity_policy=ambiguity_policy, ltf=ltf, exit_plan=exit_plan)
        trades.append(tr)
        taken.append(pid)
        open_until[p["direction"]] = tr.exit_index
    return TradeLedger(scenario.describe(), trades, excluded, taken, ambiguity_policy, price_type)


def cost_scenario_ledgers(out: PipelineOutputs, scenarios: dict, price_type: str, ambiguity_policy="CONSERVATIVE",
                          ltf=None, blackout=None) -> dict:
    """BASELINE fixes the path; every scenario re-prices the identical path."""
    base = strategy_trades(out, scenarios["BASELINE"], price_type, ambiguity_policy, ltf, None, blackout)
    res = {"BASELINE": base}
    for name, sc in scenarios.items():
        if name != "BASELINE":
            res[name] = strategy_trades(out, sc, price_type, ambiguity_policy, ltf, base.path, blackout)
    return res


# ---------------------------------------------------------------------------
# Account simulation through Phase 1F + Phase 1G
# ---------------------------------------------------------------------------
@dataclass
class AccountRun:
    decisions: dict  # trade_proposal_id -> {risk_decision, execution_decision, reason_codes, ...}
    equity_curve: list  # (time, balance)
    closed: list
    starting_balance: str
    final_balance: str
    currency: str
    policy_name: str

    def to_dict(self) -> dict:
        return {"policy": self.policy_name, "currency": self.currency, "starting_balance": self.starting_balance,
                "final_balance": self.final_balance, "trades_closed": len(self.closed),
                "approved": sum(d.get("risk_decision") == "RISK_APPROVED" for d in self.decisions.values()),
                "filled": sum(d.get("execution_decision") == "FILLED" for d in self.decisions.values())}


def account_simulation(out: PipelineOutputs, ledger: TradeLedger, starting_balance="10000", currency: str = "GBP",
                       risk_policy: RiskPolicy | None = None, execution_policy: ExecutionPolicy | None = None,
                       price_type: str = "BID", policy_name: str = "baseline", latency_seconds: float = 1.0,
                       max_notional_multiple: float | None = 30.0) -> AccountRun:
    """Replay the fixed trade path through Phase 1F sizing and the Phase 1G execution engine.

    ``max_notional_multiple`` is the broker's margin leverage (EXTERNAL_CONSTRAINT; 30 is the UK retail cap for major
    pairs - declare the real broker value).  ``None`` leaves margin UNKNOWN and Phase 1F then rejects (fail closed)."""
    rp = risk_policy or out.configs.risk
    ep = execution_policy or out.configs.execution
    feats = out.h1.features
    rates = rates_from_bars(feats)
    risk = AccountRiskEngine(rp, store=InMemoryRiskStateStore(), rates=rates)
    acct = ResearchAccount("RESEARCH", currency, D(starting_balance), **{"broker_leverage": max_notional_multiple})
    by_pid = {t.trade_proposal_id: t for t in ledger.trades}
    cons = {c.trade_proposal_id: c for c in out.trade.constructions}
    cands = {c.entry_candidate_id: c for c in out.entry.candidates}
    setups = {s.setup_id: s for s in out.h1.setups}
    fr = out.h1.frame
    pending = []  # (exit_time, pid, pnl)
    decisions, curve, closed = {}, [], []

    def settle(upto):
        pending.sort(key=lambda x: (x[0], x[1]))
        while pending and pending[0][0] <= upto:
            et, pid, pnl = pending.pop(0)
            acct.positions.pop(f"POS-{pid}", None)
            acct.balance = D(acct.balance) + pnl
            from ..risk.account import ClosedTradeResult

            risk.record_closed_result(ClosedTradeResult(f"R-{pid}", f"POS-{pid}", pid, pnl, D(0), D(0), et, "SIMULATED_EXIT"))
            closed.append({"trade_proposal_id": pid, "closed_at": et.isoformat(), "pnl": str(pnl)})
            curve.append((et.isoformat(), str(acct.balance)))

    for p in _proposals(out):
        pid = p["trade_proposal_id"]
        tr: SimulatedTrade | None = by_pid.get(pid)
        if tr is None:
            continue
        t = pd.Timestamp(p["timestamp"])
        settle(t)
        idx = out.index_of(t)
        h4_now = fr["h4_permission"].iloc[max(idx - 1, 0)]
        c = cons[pid]
        dec = risk.approve(p, acct.snapshot(t), t, c, c.index, h4_now)
        rec = {"risk_decision": dec["decision"], "risk_rejection": dec.get("rejection_category"),
               "reason_codes": list(dec.get("reason_codes") or [])}
        decisions[pid] = rec
        if dec["decision"] != "RISK_APPROVED":
            rec["execution_decision"] = "NOT_SUBMITTED"
            continue
        a = dec["approved_trade"]
        clock = SimClock(t + pd.Timedelta(seconds=latency_seconds))
        port = PaperBroker(atomic_protection=True, faults=FaultPlan(slippage_pips=tr.slippage_entry_pips))
        port.report_capabilities(t)
        pip = 0.01
        o = float(feats["open"].iloc[idx])
        sp = tr.spread_entry_pips * pip
        bid, ask = {"ASK": (o - sp, o), "MID": (o - sp / 2, o + sp / 2)}.get(price_type, (o, o + sp))
        port.add_quote(round(bid, 3), round(ask, 3), t)
        eng = ExecutionEngine(ep, store=InMemoryExecutionStore(), clock=clock, risk_engine=risk)
        eng.reconcile(port, startup=True)
        intent = eng.create_intent(dec, p)
        deps = GateDependencies(risk_state=risk.state(), account=acct.snapshot(t), construction=c,
                                entry_candidate=cands.get(p["entry_candidate_id"]), setup=setups.get(p["setup_id"]),
                                h4_permission_now=h4_now, rates=rates)
        res = eng.submit(intent.order_intent_id, port, deps)
        eng.process_events(port)
        st = eng.state().orders[intent.order_intent_id]
        rec["reason_codes"] += list(res.get("reason_codes") or [])
        if st["state"] != "FILLED":
            rec["execution_decision"] = f"{res['outcome']}"
            risk.release_reservation(a["risk_approval_id"], "not filled in simulation", t)
            continue
        prot = eng.ensure_protection(intent.order_intent_id, port)
        rec["execution_decision"] = "FILLED"
        rec["protection"] = prot["status"]
        vol = D(st["filled_volume"])
        rec["filled_volume"] = str(vol)
        pos = OpenPosition(position_id=f"POS-{pid}", symbol=a["symbol"], direction=a["direction"], volume=vol,
                           entry_price=float(st["vwap"]), stop_price=a["stop"], opened_at=t, trade_proposal_id=pid,
                           entry_candidate_id=a["entry_candidate_id"], setup_id=a["setup_id"])
        acct.positions[pos.position_id] = (pos, D(a["estimated_margin"]) if a.get("estimated_margin") else D(0))
        pip_val = D(a["pip_value_per_volume_unit"])
        pnl = (D(str(tr.net_pips)) if tr.net_pips is not None else D(str(tr.gross_pips))) * pip_val * vol
        pending.append((pd.Timestamp(tr.exit_time) + pd.Timedelta(hours=1), pid, pnl.quantize(Decimal("0.01"))))
        rec["pnl"] = str(pnl.quantize(Decimal("0.01")))
    settle(pd.Timestamp.max.tz_localize("UTC"))
    return AccountRun(decisions, curve, closed, str(D(starting_balance)), str(acct.balance), currency, policy_name)


def replace_configs(cfg: StrategyConfigs, **changes) -> StrategyConfigs:
    return replace(cfg, **changes)
