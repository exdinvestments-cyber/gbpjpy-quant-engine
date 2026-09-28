"""Research mode for the account-risk layer (Phase 1F).

Keeps the three layers separate: strategy decisions and trade geometry come
from Phases 1A-1E (generated once); this module only replays those proposals
through an ``AccountRiskEngine`` over a SIMULATED account, so identical signals
can be compared under different risk policies without regenerating signals.

Trade outcomes are NEVER invented here: every close is supplied by the caller
(e.g. a later backtest, or an explicitly labelled synthetic sequence in a
test).  Synthetic sequences do not predict real results, and no risk policy may
be chosen from synthetic profitability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

import pandas as pd

from .account import AccountState, BalanceAdjustment, ClosedTradeResult, OpenPosition
from .money import ZERO, D


@dataclass
class ResearchAccount:
    account_id: str
    currency: str
    balance: Decimal
    broker_leverage: float | None = None
    positions: dict = field(default_factory=dict)  # position_id -> (OpenPosition, margin)

    def snapshot(self, as_of, profile: str = "PERSONAL") -> AccountState:
        used = sum((m for _, m in self.positions.values()), ZERO)
        eq = D(self.balance)
        return AccountState(
            account_id=self.account_id, currency=self.currency, balance=D(self.balance), equity=eq,
            free_margin=eq - used, used_margin=used, margin_level=(eq / used * 100) if used > 0 else None,
            floating_pnl=ZERO, timestamp=pd.Timestamp(as_of), open_positions=tuple(p for p, _ in self.positions.values()),
            profile=profile, source="research_simulation", **{"broker_leverage": self.broker_leverage})


class ResearchRunner:
    def __init__(self, engine, account: ResearchAccount, profile: str = "PERSONAL"):
        self.engine, self.account, self.profile = engine, account, profile
        self.decisions: list = []
        self.closes: list = []

    def propose(self, proposal: dict, as_of, construction=None, construction_index=None) -> dict:
        dec = self.engine.approve(proposal, self.account.snapshot(as_of, self.profile), as_of, construction, construction_index)
        self.decisions.append(dec)
        a = dec.get("approved_trade")
        if dec["decision"] == "RISK_APPROVED" and not dec.get("idempotent_replay"):
            pos = OpenPosition(position_id=f"POS-{proposal['trade_proposal_id']}", symbol=a["symbol"], direction=a["direction"],
                               volume=D(a["approved_volume"]), entry_price=a["entry_reference"], stop_price=a["stop"],
                               opened_at=pd.Timestamp(as_of), trade_proposal_id=a["trade_proposal_id"],
                               entry_candidate_id=a["entry_candidate_id"], setup_id=a["setup_id"])
            margin = D(a["estimated_margin"]) if a.get("estimated_margin") else ZERO
            self.account.positions[pos.position_id] = (pos, margin)
            self.engine.convert_reservation(a["risk_approval_id"], pos.position_id, as_of)
        return dec

    def close(self, trade_proposal_id: str, realised_pnl, as_of, reason: str = "RESEARCH_SUPPLIED") -> dict:
        pid = f"POS-{trade_proposal_id}"
        self.account.positions.pop(pid, None)
        net = D(realised_pnl)
        self.account.balance = D(self.account.balance) + net
        res = ClosedTradeResult(f"R-{trade_proposal_id}", pid, trade_proposal_id, net, ZERO, ZERO, pd.Timestamp(as_of), reason)
        self.closes.append(res)
        return self.engine.record_closed_result(res)

    def close_in_r(self, trade_proposal_id: str, r_multiple: float, as_of) -> dict:
        """Caller-supplied outcome in R, converted with the APPROVED actual risk of that trade."""
        dec = next(d for d in self.decisions if d["trade_proposal_id"] == trade_proposal_id and d["decision"] == "RISK_APPROVED")
        return self.close(trade_proposal_id, D(dec["approved_trade"]["actual_risk_currency"]) * D(r_multiple), as_of)

    def adjust(self, adjustment_id: str, kind: str, amount, as_of) -> dict:
        self.account.balance = D(self.account.balance) + D(amount)
        return self.engine.record_adjustment(BalanceAdjustment(adjustment_id, kind, D(amount), pd.Timestamp(as_of)))


def compare_policies(events: list, engines: dict, account_factory) -> dict:
    """Replay identical events (``("propose", proposal, as_of)`` / ``("close_r", proposal_id, r, as_of)``) under several
    engines/policies.  Returns per-policy decisions and final balance - for research comparison only."""
    out = {}
    for name, eng in engines.items():
        run = ResearchRunner(eng, account_factory())
        for ev in events:
            if ev[0] == "propose":
                run.propose(ev[1], ev[2])
            elif ev[0] == "close_r" and any(d["trade_proposal_id"] == ev[1] and d["decision"] == "RISK_APPROVED"
                                              and not d.get("idempotent_replay") for d in run.decisions):
                run.close_in_r(ev[1], ev[2], ev[3])
        out[name] = {"decisions": run.decisions, "final_balance": str(run.account.balance),
                     "approved": sum(d["decision"] == "RISK_APPROVED" for d in run.decisions)}
    return out


# ---------------------------------------------------------------------------
# Risk of ruin / Monte Carlo - interfaces only
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskOfRuinInputs:
    """Everything a meaningful risk-of-ruin estimate needs - none of it is validated yet."""

    win_probability: float | None = None
    win_loss_distribution_R: tuple = ()  # empirical outcome distribution in R
    tail_loss_distribution_R: tuple = ()  # gap / slippage losses beyond -1R
    trade_dependency: str | None = None  # e.g. serial correlation / regime clustering model
    slippage_model: str | None = None
    drawdown_distribution: tuple = ()
    ruin_threshold_percent: float | None = None


def risk_of_ruin(inputs: RiskOfRuinInputs):
    raise NotImplementedError("risk of ruin requires validated outcome distributions (win probability, R distribution, "
                              "tail losses, trade dependency, slippage, drawdown distribution); none exist yet")


def monte_carlo_records(decisions: list) -> list[dict]:
    """Clean per-trade risk records for a later Monte Carlo study (no outcomes are fabricated)."""
    rows = []
    for d in decisions:
        a = d.get("approved_trade")
        rows.append({"trade_proposal_id": d["trade_proposal_id"], "timestamp": d["timestamp"], "decision": d["decision"],
                     "direction": a["direction"] if a else None,
                     "actual_risk_percent": a["actual_risk_percent"] if a else None,
                     "actual_risk_currency": a["actual_risk_currency"] if a else None,
                     "binding_constraint": d.get("binding_constraint"), "rejection_category": d.get("rejection_category"),
                     "note": "risk inputs only; outcomes must come from validated data"})
    return rows
