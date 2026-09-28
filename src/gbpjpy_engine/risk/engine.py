"""Phase 1F Account Risk Engine: PROPOSED TRADE -> RISK-APPROVED TRADE.

The engine answers ONE question: "given this account and this valid trade,
what is the maximum safe exposure permitted by the risk framework?"  It never
asks what exposure a profit objective would need.  A risk-approved trade is
NOT an order: nothing here sends, modifies or closes anything.

Firewalls (enforced by tests):

* score/risk - no setup, entry, trade-quality or confidence value is read;
  quality decides whether a trade EXISTS, policy decides its exposure;
* losses never increase risk - every multiplier is <= 1 and only drawdown,
  consecutive losses and news can reduce it;
* leverage affects margin feasibility only, never the permitted risk;
* no profit-objective input exists anywhere in the risk path.

Pipeline (fail closed at every step): proposal revalidation -> account data ->
contract data -> conversion data -> global state -> drawdown -> daily / weekly /
monthly budgets -> consecutive losses -> exposure (duplicates, pyramiding,
positions) -> open / reserved / aggregate risk -> external ruleset -> weekend /
news -> strictest cap -> volume -> ROUND DOWN -> actual risk -> margin ->
approval + reservation.  load -> evaluate -> reserve -> save runs inside one
store transaction (thread- and process-safe) and approvals are idempotent per
trade_proposal_id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from ..trade.symbol import GBPJPY
from .config import RiskPolicy
from .fx import ConversionUnavailable, factor
from .instruments import GBPJPY_CONTRACT
from .margin import NotionalLeverageMarginModel
from .money import HUNDRED, ZERO, D, NumericError, money_down, money_up, pct, s
from .periods import hours_to_weekend_close, period_keys
from .policies import ExternalContext
from .sizing import normalise_volume, pip_value, position_remaining_risk, risk_per_volume
from .state import CorruptedRiskState, InMemoryRiskStateStore, RiskState

CAP_ORDER = ("PER_TRADE_CAP", "MAX_SINGLE_TRADE_CAP", "DRAWDOWN_CAP", "CONSECUTIVE_LOSS_CAP", "NEWS_RISK_CAP",
             "DAILY_REMAINING_RISK", "WEEKLY_REMAINING_RISK", "MONTHLY_REMAINING_RISK", "AGGREGATE_RISK_CAP",
             "SYMBOL_RISK_CAP", "EXTERNAL_POLICY_CAP")
RESET_CONFIRMATION = "CONFIRM_RESET_OF_RISK_HALT"


@dataclass(frozen=True)
class ResetAuthorisation:
    operator: str
    reason: str
    confirmation: str  # must equal RESET_CONFIRMATION


def geometry_fingerprint(p: dict) -> str:
    core = {k: p.get(k) for k in ("trade_proposal_id", "direction", "executable_reference_price", "proposed_stop_price")}
    core["target"] = (p.get("primary_target") or {}).get("price")
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


class AccountRiskEngine:
    def __init__(self, policy: RiskPolicy | None = None, store=None, contract=None, rates=None, margin_model=None,
                 external_policy=None, commission_model=None, symbol_spec=None):
        self.policy = policy or RiskPolicy()
        self.policy.validate()
        self.store = store or InMemoryRiskStateStore()
        self.contract = contract or GBPJPY_CONTRACT
        self.rates = rates
        self.margin_model = margin_model or NotionalLeverageMarginModel()
        self.external = external_policy
        self.commission_model = commission_model
        self.symbol_spec = symbol_spec or GBPJPY

    # ================================================================== state helpers
    def _load(self, at: str) -> tuple[RiskState, str | None]:
        try:
            return self.store.load(), None
        except CorruptedRiskState as exc:
            if hasattr(self.store, "quarantine"):
                self.store.quarantine()
            st = RiskState(global_state="HALTED", halt_reason=f"corrupted risk state: {exc}", halt_source="AUTO",
                           halted_at=at)
            st.note(at, "AUTO_HALT", st.halt_reason)
            self.store.save(st)
            return st, str(exc)

    def _halt(self, st: RiskState, at: str, reason: str, source: str) -> None:
        if st.global_state != "HALTED":
            st.global_state, st.halt_reason, st.halt_source, st.halted_at = "HALTED", reason, source, at
            st.note(at, f"{source}_HALT", reason)

    def _keys(self, ts):
        dp = self.policy.data
        return period_keys(ts, dp.timezone, dp.rollover_hour)

    def _capital(self, account) -> Decimal:
        b = self.policy.sizing.basis
        if b == "BALANCE":
            return D(account.balance)
        if b == "EQUITY":
            return D(account.equity)
        return min(D(account.balance), D(account.equity))

    def _observe(self, st: RiskState, account, as_of, at: str) -> None:
        keys = self._keys(as_of)
        cap = self._capital(account)
        for name, key in (("day", keys.day), ("week", keys.week), ("month", keys.month)):
            cur = st.periods.get(name)
            if cur is None or cur["key"] != key:
                st.periods[name] = {"key": key, "reference": s(cap), "set_at": at}
                st.note(at, f"{name.upper()}_REFERENCE_SET", {"key": key, "reference": s(cap)})
        st.hwm_balance = s(max(D(st.hwm_balance), D(account.balance))) if st.hwm_balance else s(D(account.balance))
        st.hwm_equity = s(max(D(st.hwm_equity), D(account.equity))) if st.hwm_equity else s(D(account.equity))
        st.last_account_timestamp = pd.Timestamp(account.timestamp).isoformat()
        if st.global_state == "PAUSED" and st.pause_until_day and keys.day > st.pause_until_day:
            st.global_state, st.pause_reason, st.pause_until_day = "ENABLED", None, None
            st.note(at, "PAUSE_LIFTED", "new trading day")

    def _realised(self, st: RiskState, period: str, key: str, as_of) -> Decimal:
        tot = ZERO
        for r in st.ledger:
            if r[period] == key and pd.Timestamp(r["close_timestamp"]) <= as_of:
                tot += D(r["net"])
        return tot

    def _active_reservations(self, st: RiskState, as_of) -> list:
        return [r for r in st.reservations.values() if r["state"] == "RESERVED" and pd.Timestamp(r["created_at"]) <= as_of]

    # ================================================================== public API
    def approve(self, proposal: dict, account, as_of, construction=None, construction_index: int | None = None,
                h4_permission_now: str | None = None) -> dict:
        """Risk-approve (or reject) one Phase 1E proposal at ``as_of``.  Never places anything."""
        as_of = pd.Timestamp(as_of)
        at = as_of.isoformat()
        with self.store.transaction():
            st, corrupt = self._load(at)
            pid = proposal.get("trade_proposal_id")
            if pid in st.approvals and corrupt is None:
                out = dict(st.approvals[pid])
                out["idempotent_replay"] = True
                return out
            dec = self._evaluate(st, proposal, account, as_of, at, construction, construction_index, h4_permission_now, corrupt)
            self.store.save(st)
            return dec

    def record_closed_result(self, result) -> dict:
        at = pd.Timestamp(result.close_timestamp).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            if any(r["result_id"] == result.result_id for r in st.ledger):
                return {"status": "DUPLICATE_IGNORED", "result_id": result.result_id}
            k = self._keys(result.close_timestamp)
            net = result.net
            st.ledger.append({"result_id": result.result_id, "position_id": result.position_id,
                              "trade_proposal_id": result.trade_proposal_id, "net": s(net), "realised_pnl": s(D(result.realised_pnl)),
                              "fees": s(D(result.fees)), "swap": s(D(result.swap)), "close_timestamp": at,
                              "close_reason": result.close_reason, "partial": result.partial,
                              "closed_volume": s(D(result.closed_volume)) if result.closed_volume is not None else None,
                              "day": k.day, "week": k.week, "month": k.month})
            if not result.partial:
                if net < 0:
                    st.consecutive_losses += 1
                elif net > 0:
                    st.consecutive_losses = 0
                    st.pause_trigger_count = None
            st.note(at, "CLOSED_RESULT", {"result_id": result.result_id, "net": s(net), "partial": result.partial,
                                          "consecutive_losses": st.consecutive_losses})
            self.store.save(st)
            return {"status": "RECORDED", "consecutive_losses": st.consecutive_losses}

    def record_adjustment(self, adj) -> dict:
        """Deposits / withdrawals shift high-water marks and period references - never counted as trading P&L."""
        at = pd.Timestamp(adj.timestamp).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            if any(a["adjustment_id"] == adj.adjustment_id for a in st.adjustments):
                return {"status": "DUPLICATE_IGNORED"}
            amt = D(adj.amount)
            st.adjustments.append({"adjustment_id": adj.adjustment_id, "kind": adj.kind, "amount": s(amt), "timestamp": at})
            if st.hwm_balance:
                st.hwm_balance = s(D(st.hwm_balance) + amt)
            if st.hwm_equity:
                st.hwm_equity = s(D(st.hwm_equity) + amt)
            for p in st.periods.values():
                p["reference"] = s(D(p["reference"]) + amt)
            st.note(at, f"BALANCE_{adj.kind}", s(amt))
            self.store.save(st)
            return {"status": "RECORDED"}

    def release_reservation(self, risk_approval_id: str, reason: str, as_of) -> dict:
        return self._set_reservation(risk_approval_id, "RELEASED", reason, as_of)

    def convert_reservation(self, risk_approval_id: str, position_id: str, as_of) -> dict:
        """Future execution: the filled position now carries the risk (reported through the account state)."""
        return self._set_reservation(risk_approval_id, "CONVERTED", f"position {position_id}", as_of)

    def _set_reservation(self, rid, to, reason, as_of):
        at = pd.Timestamp(as_of).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            r = st.reservations.get(rid)
            if r is None:
                return {"status": "UNKNOWN_RESERVATION"}
            if r["state"] != "RESERVED":
                return {"status": "ALREADY_" + r["state"]}
            r["state"], r["ended_at"], r["end_reason"] = to, at, reason
            st.note(at, f"RESERVATION_{to}", {"id": rid, "reason": reason})
            self.store.save(st)
            return {"status": to}

    def halt(self, reason: str, operator: str, as_of) -> dict:
        at = pd.Timestamp(as_of).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            self._halt(st, at, f"manual: {reason} (by {operator})", "MANUAL")
            self.store.save(st)
            return {"global_state": st.global_state}

    def pause(self, reason: str, operator: str, as_of) -> dict:
        at = pd.Timestamp(as_of).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            if st.global_state == "ENABLED":
                st.global_state, st.pause_reason, st.pause_until_day = "PAUSED", f"manual: {reason} (by {operator})", None
                st.note(at, "MANUAL_PAUSE", st.pause_reason)
            self.store.save(st)
            return {"global_state": st.global_state}

    def reset(self, auth: ResetAuthorisation, as_of) -> dict:
        """The ONLY way out of HALTED / PAUSED: an explicit, recorded authorisation."""
        if auth.confirmation != RESET_CONFIRMATION or not auth.operator or not auth.reason:
            return {"status": "REFUSED", "reason": "reset requires operator, reason and the exact confirmation phrase"}
        at = pd.Timestamp(as_of).isoformat()
        with self.store.transaction():
            st, _ = self._load(at)
            prev = st.global_state
            st.global_state, st.halt_reason, st.halt_source, st.halted_at = "ENABLED", None, None, None
            st.pause_reason, st.pause_until_day = None, None
            st.note(at, "AUTHORISED_RESET", {"from": prev, "operator": auth.operator, "reason": auth.reason})
            self.store.save(st)
            return {"status": "RESET", "from": prev}

    def state(self) -> RiskState:
        with self.store.transaction():
            return self._load("state-inspection")[0]

    @staticmethod
    def _construction_state(construction, as_of, cidx) -> str:
        """Phase 1E state as known at the approval TIME (a lapse later in the same bar is not yet known)."""
        st = "NO_TRADE_CONSTRUCTION"
        for idx, t_at, _, to, _ in construction.transitions:
            try:
                known = pd.Timestamp(t_at) <= as_of
            except (ValueError, TypeError):
                known = cidx is not None and idx <= cidx
            if known:
                st = to
        return st

    # ================================================================== evaluation
    def _evaluate(self, st, proposal, account, as_of, at, construction, cidx, h4_now, corrupt) -> dict:
        pol, c = self.policy, self.contract
        codes: list[str] = []
        audit: dict = {"as_of": at, "policy_hash": pol.policy_hash()}
        pid = proposal.get("trade_proposal_id")

        def reject(category, reason, *extra, halt=False):
            if halt:
                self._halt(st, at, reason, "AUTO")
                extra = extra + ("GLOBAL_HALT",)
            dec = {"decision": "RISK_REJECTED", "risk_approval_id": None, "trade_proposal_id": pid,
                   "entry_candidate_id": proposal.get("entry_candidate_id"), "setup_id": proposal.get("setup_id"),
                   "timestamp": at, "rejection_category": category, "rejection_reason": reason,
                   "reason_codes": list(dict.fromkeys(codes + list(extra) + ["RISK_REJECTED"])),
                   "binding_constraint": audit.get("binding_constraint"), "global_risk_state": st.global_state,
                   "audit": audit, "approved_trade": None, "not_an_order": True}
            st.note(at, "RISK_REJECTED", {"trade_proposal_id": pid, "category": category})
            return dec

        if corrupt is not None:
            return reject("CORRUPTED_RISK_STATE", f"risk state corrupted ({corrupt}); halted", "RISK_STATE_CORRUPTED")
        # ---- 1. proposal revalidation ---------------------------------------------------------
        try:
            d = 1 if proposal["direction"] == "LONG" else (-1 if proposal["direction"] == "SHORT" else 0)
            entry, stop = D(proposal["executable_reference_price"]), D(proposal["proposed_stop_price"])
            target = D((proposal.get("primary_target") or {})["price"])
        except (KeyError, TypeError, NumericError) as exc:
            return reject("PROPOSAL_INVALID", f"proposal geometry unusable: {exc}", "PROPOSAL_INVALID")
        if proposal.get("decision") != "PROPOSE_TRADE" or d == 0:
            return reject("PROPOSAL_INVALID", "not a proposed trade", "PROPOSAL_INVALID")
        if d * (entry - stop) <= 0 or d * (target - entry) <= 0:
            return reject("PROPOSAL_INVALID", "impossible stop/target geometry", "PROPOSAL_INVALID")
        if pd.Timestamp(proposal["timestamp"]) > as_of:
            return reject("PROPOSAL_INVALID", "proposal is from the future", "PROPOSAL_INVALID")
        if proposal.get("symbol") != c.symbol:
            return reject("PROPOSAL_INVALID", "proposal symbol does not match the contract", "PROPOSAL_INVALID")
        if construction is not None:
            if construction.decision != "PROPOSE_TRADE":
                return reject("PROPOSAL_INVALID", "Phase 1E construction was not a proposal", "PROPOSAL_INVALID")
            state_now = self._construction_state(construction, as_of, cidx)
            if state_now != "PROPOSED":
                return reject("PROPOSAL_NO_LONGER_VALID", f"Phase 1E proposal is {state_now} at approval time",
                              "PROPOSAL_NO_LONGER_VALID")
            if geometry_fingerprint(construction.proposal) != geometry_fingerprint(proposal):
                return reject("PROPOSAL_INVALID", "proposal geometry changed since construction", "PROPOSAL_GEOMETRY_CHANGED")
        if h4_now is not None and h4_now not in (("ALLOW_LONG", "ALLOW_BOTH") if d > 0 else ("ALLOW_SHORT", "ALLOW_BOTH")):
            return reject("PROPOSAL_NO_LONGER_VALID", f"H4 permission is now {h4_now}", "H4_PERMISSION_REVOKED")
        codes.append("PROPOSAL_VALID")
        # ---- 2. account data ---------------------------------------------------------------------
        issues = account.issues() if account is not None else ["no account state"]
        if issues:
            return reject("ACCOUNT_STATE_INVALID", "; ".join(issues), "ACCOUNT_STATE_INVALID",
                          halt=pol.data.halt_on_invalid_account)
        ats = pd.Timestamp(account.timestamp)
        if ats > as_of:
            return reject("ACCOUNT_STATE_INVALID", "account snapshot is newer than the decision time", "ACCOUNT_STATE_FROM_FUTURE")
        if (as_of - ats).total_seconds() > pol.data.account_max_age_seconds:
            return reject("ACCOUNT_STATE_STALE", f"account snapshot {as_of - ats} old", "ACCOUNT_STATE_STALE")
        if account.profile == "EXTERNAL_RULESET" and self.external is None:
            return reject("EXTERNAL_POLICY_MISSING", "external-ruleset account without a ruleset object", "EXTERNAL_POLICY_BLOCK")
        codes.append("ACCOUNT_STATE_VALID")
        ccy = account.currency
        # ---- 3. contract data --------------------------------------------------------------------
        ci = c.issues()
        if not ci and not c.matches(self.symbol_spec):
            ci = ["contract specification disagrees with the strategy symbol metadata"]
        if ci:
            return reject("SYMBOL_SPEC_INVALID", "; ".join(ci), "SYMBOL_SPEC_INVALID", halt=pol.data.halt_on_invalid_symbol)
        codes.append("SYMBOL_SPEC_VALID")
        # ---- 4. conversion -----------------------------------------------------------------------
        max_age = pd.Timedelta(seconds=pol.data.rate_max_age_seconds)
        try:
            fac, used = factor(c.quote_currency, ccy, as_of, self.rates, max_age)
        except ConversionUnavailable as exc:
            code = "CONVERSION_RATE_STALE" if "stale" in str(exc) else "CONVERSION_RATE_MISSING"
            return reject("CONVERSION_UNAVAILABLE", str(exc), code)
        audit["conversion"] = {"from": c.quote_currency, "to": ccy, "factor": s(fac), "rates": [r.to_dict() for r in used]}
        codes.append("CONVERSION_RATE_VALID")
        # ---- 5. global state ---------------------------------------------------------------------
        self._observe(st, account, as_of, at)
        if st.global_state == "HALTED":
            return reject("GLOBAL_HALT", f"global state HALTED ({st.halt_reason})", "GLOBAL_HALT")
        if st.global_state == "PAUSED":
            return reject("GLOBAL_PAUSE", f"global state PAUSED ({st.pause_reason})", "GLOBAL_PAUSE")
        capital = self._capital(account)
        keys = self._keys(as_of)
        # ---- 6. drawdown -------------------------------------------------------------------------
        hwm_e, hwm_b = D(st.hwm_equity), D(st.hwm_balance)
        dd_e = (hwm_e - D(account.equity)) / hwm_e * HUNDRED if hwm_e > 0 else ZERO
        dd_b = (hwm_b - D(account.balance)) / hwm_b * HUNDRED if hwm_b > 0 else ZERO
        tier, mult = "NORMAL", Decimal(1)
        for name, th, m in pol.drawdown.tiers:
            if dd_e >= D(th):
                tier, mult = name, D(m)
        audit["drawdown"] = {"hwm_equity": st.hwm_equity, "hwm_balance": st.hwm_balance, "equity_drawdown_pct": s(dd_e.quantize(Decimal("0.0001"))),
                             "balance_drawdown_pct": s(dd_b.quantize(Decimal("0.0001"))),
                             "equity_drawdown_abs": s(hwm_e - D(account.equity)), "state": tier, "multiplier": s(mult)}
        codes.append(f"DRAWDOWN_{tier}")
        if tier == "HALT" or mult <= 0:
            return reject("DRAWDOWN_HALT", f"equity drawdown {dd_e:.2f}% reached the HALT tier", halt=pol.drawdown.halt_on_critical)
        # ---- open / reserved risk ----------------------------------------------------------------
        open_rows, open_total, sym_open = [], ZERO, ZERO
        for p in account.open_positions:
            f_p = fac if p.symbol == c.symbol else None
            if f_p is None:
                return reject("OPEN_RISK_UNKNOWN", f"cannot value open position {p.position_id} ({p.symbol})", "OPEN_RISK_UNKNOWN")
            r = position_remaining_risk(p, c, f_p)
            if r["remaining_risk"] is None:
                return reject("OPEN_RISK_UNKNOWN", f"open position {p.position_id} has no protective stop", "OPEN_RISK_UNKNOWN")
            open_rows.append({**r, "remaining_risk": s(money_up(r["remaining_risk"], ccy))})
            open_total += r["remaining_risk"]
            sym_open += r["remaining_risk"]
        res = self._active_reservations(st, as_of)
        reserved = sum((D(r["amount"]) for r in res), ZERO)
        pending = D(account.pending_risk)
        audit["open_risk"] = {"positions": open_rows, "total_open_risk": s(money_up(open_total, ccy)),
                              "reserved_risk": s(reserved), "provider_pending_risk": s(pending)}
        # ---- 7. period budgets -------------------------------------------------------------------
        float_loss = max(-D(account.floating_pnl), ZERO)
        periods = {}
        for name, key, pct_lim, enabled, code in (
                ("day", keys.day, pol.limits.daily_loss_percent, True, "DAILY"),
                ("week", keys.week, pol.limits.weekly_loss_percent, True, "WEEKLY"),
                ("month", keys.month, pol.limits.monthly_loss_percent, pol.limits.monthly_enabled, "MONTHLY")):
            if not enabled:
                continue
            ref = D(st.periods[name]["reference"])
            realised = self._realised(st, name, key, as_of)
            used_ = max(-realised, ZERO) + float_loss + open_total + reserved + pending
            limit = D(pct_lim) / HUNDRED * ref
            remaining = limit - used_
            periods[name] = {"key": key, "reference": s(ref), "realised_net": s(realised), "limit": s(money_down(limit, ccy)),
                             "used_before": s(money_up(used_, ccy)), "remaining_before": s(money_down(remaining, ccy))}
            if remaining <= 0:
                audit["periods"] = periods
                return reject(f"{code}_LIMIT", f"{name} loss budget exhausted", f"{code}_LIMIT_REACHED",
                              halt=(name == "day" and pol.data.halt_on_daily_limit))
            codes.append(f"{code}_LIMIT_OK")
        audit["periods"] = periods
        # ---- 8. consecutive losses ---------------------------------------------------------------
        cl = pol.consecutive
        cons_mult = Decimal(1)
        if cl.pause_after and st.consecutive_losses >= cl.pause_after and st.pause_trigger_count != st.consecutive_losses:
            st.pause_trigger_count = st.consecutive_losses
            st.global_state = "PAUSED"
            st.pause_reason = f"{st.consecutive_losses} consecutive losses"
            st.pause_until_day = keys.day if cl.pause_until == "NEXT_TRADING_DAY" else None
            st.note(at, "AUTO_PAUSE", st.pause_reason)
            return reject("CONSECUTIVE_LOSS_PAUSE", st.pause_reason, "CONSECUTIVE_LOSS_PAUSE")
        if cl.reduce_after and st.consecutive_losses >= cl.reduce_after:
            cons_mult = D(cl.reduce_factor)
            codes.append("CONSECUTIVE_LOSS_REDUCED")
        else:
            codes.append("CONSECUTIVE_LOSS_LIMIT_OK")
        audit["consecutive_losses"] = {"count": st.consecutive_losses, "multiplier": s(cons_mult)}
        # ---- 9. exposure: duplicates, pyramiding, positions ---------------------------------------
        lim = pol.limits
        ids = {proposal.get("trade_proposal_id"), proposal.get("entry_candidate_id"), proposal.get("setup_id")} - {None}
        exposures = [{"direction": p.direction, "ids": {p.trade_proposal_id, p.entry_candidate_id, p.setup_id, p.position_id},
                      "floating": D(p.floating_pnl), "kind": "position"} for p in account.open_positions if p.symbol == c.symbol]
        exposures += [{"direction": r["direction"], "ids": {r["trade_proposal_id"], r["entry_candidate_id"], r["setup_id"]},
                       "floating": ZERO, "kind": "reservation"} for r in res if r["symbol"] == c.symbol]
        if any(ids & (e["ids"] - {None}) for e in exposures):
            return reject("DUPLICATE_EXPOSURE", "the same setup / entry / proposal already has exposure", "DUPLICATE_EXPOSURE")
        same = [e for e in exposures if e["direction"] == proposal["direction"]]
        if any(e["kind"] == "position" and e["floating"] < 0 for e in same):
            return reject("ADD_TO_LOSER_PROHIBITED", "adding to a losing position is prohibited", "ADD_TO_LOSER_PROHIBITED")
        if same and not lim.pyramiding_enabled:
            return reject("PYRAMIDING_DISABLED", "an exposure in this direction already exists (no pyramiding)", "PYRAMIDING_DISABLED")
        if len(same) >= lim.max_same_direction_positions:
            return reject("SYMBOL_EXPOSURE_LIMIT", "same-direction GBPJPY position limit reached", "SYMBOL_EXPOSURE_LIMIT")
        if any(e["direction"] != proposal["direction"] for e in exposures) and not lim.allow_opposite_direction:
            return reject("SYMBOL_EXPOSURE_LIMIT", "opposite GBPJPY exposure exists (hedging disabled)", "SYMBOL_EXPOSURE_LIMIT")
        if len(exposures) >= lim.max_symbol_positions:
            return reject("SYMBOL_EXPOSURE_LIMIT", "GBPJPY position limit reached", "SYMBOL_EXPOSURE_LIMIT")
        codes.append("EXPOSURE_OK")
        # ---- 10. external ruleset ------------------------------------------------------------------
        h2w = hours_to_weekend_close(as_of, pol.data.timezone, pol.data.rollover_hour)
        news = proposal.get("news_status") or "UNKNOWN"
        ext_cap, ext_vol, ext_status = None, None, "NOT_APPLICABLE"
        if self.external is not None:
            daily_hist: dict = {}
            for r in st.ledger:
                if pd.Timestamp(r["close_timestamp"]) <= as_of:
                    daily_hist[r["day"]] = daily_hist.get(r["day"], ZERO) + D(r["net"])
            today_loss = max(-self._realised(st, "day", keys.day, as_of), ZERO) + float_loss
            er = self.external.evaluate(ExternalContext(as_of, ccy, D(account.balance), D(account.equity), hwm_e, today_loss,
                                                        len(account.open_positions), h2w, news, daily_hist))
            ext_status = er.status
            audit["external_policy"] = {"name": getattr(self.external, "name", type(self.external).__name__), "status": er.status,
                                        "remaining": s(er.remaining_risk), "max_volume": s(er.max_volume),
                                        "reasons": list(er.reasons), "consistency": er.consistency}
            if er.status in ("BLOCK", "BREACH", "UNKNOWN"):
                return reject("EXTERNAL_POLICY", "; ".join(er.reasons) or f"external policy {er.status}", "EXTERNAL_POLICY_BLOCK",
                              halt=(er.status == "BREACH" and pol.data.halt_on_external_breach))
            ext_cap, ext_vol = er.remaining_risk, er.max_volume
            codes.append("EXTERNAL_POLICY_OK")
        # ---- 11. weekend / news ------------------------------------------------------------------------
        ev = pol.events
        weekend_status = "ALLOW"
        if ev.weekend != "ALLOW" and h2w <= ev.weekend_cutoff_hours:
            return reject("WEEKEND_POLICY", f"weekend policy {ev.weekend}: {h2w:.1f} h to the weekend close", "WEEKEND_POLICY_BLOCK")
        if ev.weekend != "ALLOW":
            weekend_status = f"{ev.weekend}_OUTSIDE_CUTOFF"
        news_mult, news_status = Decimal(1), f"{news}/{ev.news}"
        event = news in ("EVENT_IMMINENT", "EVENT_RECENT")
        if news == "UNKNOWN":
            codes.append("NEWS_UNKNOWN")
            if ev.block_unknown_news:
                return reject("NEWS_POLICY", "news status UNKNOWN and policy blocks unknown", "NEWS_POLICY_BLOCK")
        if (ev.news == "BLOCK_BEFORE" and news == "EVENT_IMMINENT") or (ev.news == "BLOCK_AFTER" and news == "EVENT_RECENT") \
                or (ev.news == "UNKNOWN" and event):
            return reject("NEWS_POLICY", f"news policy {ev.news} with status {news}", "NEWS_POLICY_BLOCK")
        if ev.news == "REDUCE_RISK" and event:
            news_mult = D(ev.news_reduce_factor)
        # ---- 12. risk budget: strictest applicable limit -----------------------------------------
        base = D(pol.sizing.base_risk_percent) / HUNDRED * capital
        caps = {"PER_TRADE_CAP": base, "MAX_SINGLE_TRADE_CAP": D(pol.sizing.max_single_trade_risk_percent) / HUNDRED * capital}
        if mult < 1:
            caps["DRAWDOWN_CAP"] = base * mult
        if cons_mult < 1:
            caps["CONSECUTIVE_LOSS_CAP"] = base * cons_mult
        if news_mult < 1:
            caps["NEWS_RISK_CAP"] = base * news_mult
        caps["DAILY_REMAINING_RISK"] = D(periods["day"]["remaining_before"])
        caps["WEEKLY_REMAINING_RISK"] = D(periods["week"]["remaining_before"])
        if "month" in periods:
            caps["MONTHLY_REMAINING_RISK"] = D(periods["month"]["remaining_before"])
        agg_before = open_total + reserved + pending
        caps["AGGREGATE_RISK_CAP"] = D(lim.max_aggregate_risk_percent) / HUNDRED * capital - agg_before
        caps["SYMBOL_RISK_CAP"] = D(lim.max_symbol_risk_percent) / HUNDRED * capital - sym_open - reserved
        if ext_cap is not None:
            caps["EXTERNAL_POLICY_CAP"] = D(ext_cap)
        binding = min(caps, key=lambda k: (caps[k], CAP_ORDER.index(k)))
        permitted = money_down(caps[binding], ccy)
        audit["risk_budget"] = {"sizing_capital": s(capital), "basis": pol.sizing.basis,
                                "caps": {k: s(money_down(v, ccy)) for k, v in caps.items()},
                                "binding_constraint": binding, "permitted_risk": s(permitted)}
        audit["binding_constraint"] = binding
        if permitted <= 0:
            return reject(binding, f"no risk available: {binding} leaves {permitted} {ccy}",
                          "AGGREGATE_RISK_EXCEEDED" if binding in ("AGGREGATE_RISK_CAP", "SYMBOL_RISK_CAP") else "TRADE_RISK_EXCEEDS_LIMIT")
        if binding in ("AGGREGATE_RISK_CAP", "SYMBOL_RISK_CAP") and lim.aggregate_policy == "REJECT" and caps[binding] < base:
            return reject(binding, "aggregate exposure would exceed the cap (policy REJECT)", "AGGREGATE_RISK_EXCEEDED")
        codes.append("AGGREGATE_RISK_OK")
        # ---- 13. volume ------------------------------------------------------------------------------
        costs = proposal.get("cost_assumptions", {}).get("components", {})
        sz = pol.sizing

        def comp(name, fallback):
            v = costs.get(name) or {}
            return (float(v["pips"]), "KNOWN") if v.get("status") == "KNOWN" and v.get("pips") is not None else (fallback, "ASSUMED")

        slip_stop, st_s = comp("slippage_exit", sz.stop_slippage_allowance_pips)
        slip_stop = max(slip_stop, sz.stop_slippage_allowance_pips)
        slip_entry, st_e = comp("slippage_entry", sz.entry_slippage_allowance_pips)
        slip_entry = max(slip_entry, sz.entry_slippage_allowance_pips)
        commission, st_c = comp("commission_round_turn", sz.assumed_commission_pips_round_turn)
        spread = costs.get("spread") or {}
        extra = D(slip_stop) + D(slip_entry) + D(commission)
        stop_dist = abs(entry - stop)
        per_vol = risk_per_volume(c, stop_dist, extra, fac)
        if per_vol <= 0:
            return reject("NUMERIC_INVALID", "non-positive risk per volume", "NUMERIC_INVALID")
        pv = pip_value(c, Decimal(1), fac)
        theoretical = permitted / per_vol
        volume, vnotes = normalise_volume(theoretical, c, ext_vol)
        audit["volume"] = {"risk_per_1_volume": s(per_vol), "theoretical_volume": s(theoretical), "normalised_volume": s(volume),
                           "volume_step": s(c.volume_step), "min_volume": s(c.min_volume), "max_volume": s(c.max_volume),
                           "notes": vnotes}
        codes += [n for n in vnotes if n == "VOLUME_ROUNDED_DOWN"]
        if volume < D(c.min_volume):
            budget = {"DAILY_REMAINING_RISK": ("DAILY_LIMIT", "DAILY_LIMIT_REACHED"),
                      "WEEKLY_REMAINING_RISK": ("WEEKLY_LIMIT", "WEEKLY_LIMIT_REACHED"),
                      "MONTHLY_REMAINING_RISK": ("MONTHLY_LIMIT", "MONTHLY_LIMIT_REACHED"),
                      "AGGREGATE_RISK_CAP": ("AGGREGATE_RISK_CAP", "AGGREGATE_RISK_EXCEEDED"),
                      "SYMBOL_RISK_CAP": ("SYMBOL_RISK_CAP", "AGGREGATE_RISK_EXCEEDED"),
                      "EXTERNAL_POLICY_CAP": ("EXTERNAL_POLICY", "EXTERNAL_POLICY_BLOCK")}.get(binding)
            why = f"minimum volume {c.min_volume} would risk more than the permitted {permitted} {ccy}"
            if budget:  # the binding budget is effectively exhausted - attribute the rejection to it
                return reject(budget[0], f"{binding} leaves {permitted} {ccy}: {why}", budget[1], "BROKER_MIN_VOLUME_TOO_LARGE")
            return reject("BROKER_MIN_VOLUME", why, "BROKER_MIN_VOLUME_TOO_LARGE")
        actual = volume * per_vol
        tol = D(sz.rounding_tolerance)
        if actual > permitted + tol:
            return reject("RISK_CALCULATION_INCONSISTENT", "actual risk exceeds permitted risk", "TRADE_RISK_EXCEEDS_LIMIT")
        codes.append("TRADE_RISK_WITHIN_LIMIT")
        # ---- 14. margin (feasibility; leverage never changes permitted risk) ------------------------
        mp = pol.margin
        est = self.margin_model.estimate(c, volume, account, as_of, self.rates, max_age)
        margin_info = {"status": est.status, "method": est.method, "estimated_margin": s(est.amount), "note": est.note}
        if est.status != "KNOWN" or account.free_margin is None:
            margin_info["status"] = "UNKNOWN"
            audit["margin"] = margin_info
            if mp.require_margin_known:
                return reject("MARGIN_UNKNOWN", "required margin cannot be estimated reliably", "MARGIN_UNKNOWN")
            codes.append("MARGIN_UNKNOWN")
            post_free = post_level = None
        else:
            used_m = D(account.used_margin or 0)
            equity = D(account.equity)

            def feasible(m):
                free_after = D(account.free_margin) - m
                level = equity / (used_m + m) * HUNDRED if (used_m + m) > 0 else None
                ok = free_after >= D(mp.min_free_margin_after_percent_of_equity) / HUNDRED * equity and \
                    (level is None or level >= D(mp.min_post_trade_margin_level_percent))
                return ok, free_after, level

            ok, post_free, post_level = feasible(est.amount)
            if not ok:
                if mp.margin_policy == "REJECT":
                    margin_info.update({"post_trade_free_margin": s(post_free), "post_trade_margin_level": s(post_level)})
                    audit["margin"] = margin_info
                    return reject("MARGIN_INSUFFICIENT", "post-trade free margin / margin level below the safety minimum",
                                  "MARGIN_INSUFFICIENT")
                per_m = est.amount / volume
                lo = floor_vol = ZERO
                v = volume
                while v >= D(c.min_volume):  # largest step-aligned volume that is margin-feasible
                    if feasible(per_m * v)[0]:
                        floor_vol = v
                        break
                    v -= D(c.volume_step)
                if floor_vol < D(c.min_volume):
                    audit["margin"] = margin_info
                    return reject("MARGIN_INSUFFICIENT", "even the minimum volume breaches margin safety", "MARGIN_INSUFFICIENT")
                volume, lo = floor_vol, per_m * floor_vol
                actual = volume * per_vol
                binding = "MARGIN_CAP"
                audit["binding_constraint"] = binding
                ok, post_free, post_level = feasible(lo)
                margin_info["estimated_margin"] = s(lo)
            margin_info.update({"status": "OK", "estimated_margin": s(money_up(D(margin_info["estimated_margin"]), ccy)),
                                "post_trade_free_margin": s(money_down(post_free, ccy)),
                                "post_trade_margin_level": s(post_level.quantize(Decimal("0.01"))) if post_level is not None else None})
            codes.append("MARGIN_OK")
        audit["margin"] = margin_info
        # ---- 15. approval + reservation --------------------------------------------------------------
        rid = f"RA-{pid}"
        actual_c = money_up(actual, ccy)
        pc = pct(actual, capital)
        daily_after = D(periods["day"]["remaining_before"]) - actual
        weekly_after = D(periods["week"]["remaining_before"]) - actual
        agg_after = agg_before + actual
        codes += ["RISK_APPROVED"]
        approved = {
            "risk_approval_id": rid, "trade_proposal_id": pid, "entry_candidate_id": proposal.get("entry_candidate_id"),
            "setup_id": proposal.get("setup_id"), "timestamp": at, "symbol": c.symbol, "direction": proposal["direction"],
            "entry_reference": float(entry), "stop": float(stop), "primary_target": float(target),
            "account_currency": ccy, "risk_basis": pol.sizing.basis, "sizing_capital": s(capital),
            "risk_cap_percent": s(D(pol.sizing.base_risk_percent)), "risk_amount": s(permitted),
            "planned_risk_currency": s(permitted), "planned_risk_percent": s(pct(permitted, capital)),
            "pip_value_per_volume_unit": s(pv["pip_value_account"].quantize(Decimal("0.00000001"))), "pip_value_quote_per_volume_unit": s(pv["pip_value_quote"]),
            "stop_distance_pips": s((stop_dist / D(c.pip_size)).quantize(Decimal("0.01"))),
            "risk_distance_pips_incl_allowances": s((stop_dist / D(c.pip_size) + extra).quantize(Decimal("0.01"))),
            "theoretical_volume": s(theoretical.quantize(Decimal("0.00000001"))), "approved_volume": s(volume),
            "normalised_volume": s(volume),
            "actual_risk_currency": s(actual_c), "actual_risk_percent": s(pc),
            "daily_risk_remaining": s(money_down(daily_after, ccy)), "weekly_risk_remaining": s(money_down(weekly_after, ccy)),
            "aggregate_risk_before_trade": s(money_up(agg_before, ccy)), "aggregate_risk_after_trade": s(money_up(agg_after, ccy)),
            "drawdown_state": tier, "consecutive_loss_state": {"count": st.consecutive_losses, "multiplier": s(cons_mult)},
            "margin_status": margin_info["status"], "estimated_margin": margin_info.get("estimated_margin"),
            "post_trade_free_margin": margin_info.get("post_trade_free_margin"),
            "post_trade_margin_level": margin_info.get("post_trade_margin_level"),
            "external_policy_status": ext_status, "news_risk_status": news_status, "weekend_policy_status": weekend_status,
            "global_risk_state": st.global_state, "binding_constraint": binding,
            "costs": {"spread": {"status": spread.get("status", "UNKNOWN"), "treatment": "embedded in the Phase 1E entry/stop prices"},
                      "stop_slippage_pips": {"value": slip_stop, "status": "ESTIMATED_ALLOWANCE" if st_s != "KNOWN" else "KNOWN"},
                      "entry_slippage_pips": {"value": slip_entry, "status": "ESTIMATED_ALLOWANCE" if st_e != "KNOWN" else "KNOWN"},
                      "commission_pips": {"value": commission, "status": st_c if st_c == "KNOWN" else "UNKNOWN_ASSUMED"},
                      "swap": {"status": "UNKNOWN", "note": "not included"}},
            "gap_risk_status": "NOT_BOUNDED_BY_STOP",
            "risk_note": "planned risk assumes the stop fills near its price; gaps and slippage can produce a loss larger "
                         "than 1R - it is NOT a guaranteed maximum loss",
            "slippage_assumption": {"stop_pips": slip_stop, "entry_pips": slip_entry},
            "conversion": audit["conversion"],
            "reason_codes": list(dict.fromkeys(codes)),
            "not_an_order": "risk approval only - no order, ticket or execution exists",
        }
        st.reservations[rid] = {"risk_approval_id": rid, "trade_proposal_id": pid,
                                "entry_candidate_id": proposal.get("entry_candidate_id"), "setup_id": proposal.get("setup_id"),
                                "symbol": c.symbol, "direction": proposal["direction"], "amount": s(actual_c),
                                "currency": ccy, "created_at": at, "state": "RESERVED"}
        audit["decision"] = {"why": f"approved: {binding} was the binding constraint; volume floored to the {c.volume_step} step",
                             "daily_before": periods["day"]["remaining_before"], "daily_after": approved["daily_risk_remaining"],
                             "aggregate_before": approved["aggregate_risk_before_trade"],
                             "aggregate_after": approved["aggregate_risk_after_trade"]}
        dec = {"decision": "RISK_APPROVED", "risk_approval_id": rid, "trade_proposal_id": pid,
               "entry_candidate_id": proposal.get("entry_candidate_id"), "setup_id": proposal.get("setup_id"), "timestamp": at,
               "rejection_category": None, "rejection_reason": None, "reason_codes": approved["reason_codes"],
               "binding_constraint": binding, "global_risk_state": st.global_state, "audit": audit,
               "approved_trade": approved, "not_an_order": True}
        st.approvals[pid] = dec
        st.note(at, "RISK_APPROVED", {"id": rid, "volume": s(volume), "risk": s(actual_c)})
        return dec
