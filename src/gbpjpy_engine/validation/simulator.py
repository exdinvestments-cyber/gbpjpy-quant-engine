"""Deterministic historical fill / exit simulator (Phase 1H) - research only, no network.

Price representation is explicit (``price_type`` of the dataset):

* BID bars: bid = bar, ask = bar + spread
* ASK bars: ask = bar, bid = bar - spread
* MID bars: bid = mid - spread/2, ask = mid + spread/2
* UNKNOWN: treated as BID and every trade is flagged ``PRICE_TYPE_UNKNOWN``

A LONG enters at the ASK and exits at the BID; a SHORT enters at the BID and
exits at the ASK.  A mid price is never called executable.

OHLC sequencing: inside one bar the order of high and low is unknowable.  When
a bar touches both the stop and the target the favourable outcome is NEVER
chosen automatically:

* CONSERVATIVE - the stop is assumed first;
* AMBIGUOUS - the stop is assumed first and the trade is flagged AMBIGUOUS
  (reported separately);
* LOWER_TIMEFRAME_REQUIRED - lower-timeframe bars inside that bar resolve the
  order; without them the trade is flagged UNRESOLVED and treated conservatively.

Lower-timeframe data is used ONLY for execution sequencing after the H1
decision; it never reaches a strategy decision.  Gaps through a stop fill at
the (worse) opening price; gaps through a target fill AT the target (the gap
gift is not taken).  Stops therefore can lose more than 1R.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .costs import CostScenario

AMBIGUITY_POLICIES = ("CONSERVATIVE", "AMBIGUOUS", "LOWER_TIMEFRAME_REQUIRED")


@dataclass(frozen=True)
class ExitPlan:
    """Exit architecture.  Default (Phase 1E): single target, no partials, no time exit, no trailing."""

    partial_fractions: tuple = ()  # e.g. ((0.5, "T1"),) - not active unless configured
    max_holding_bars: int | None = None  # research cap; None = hold until stop / target / end of data
    end_of_data_policy: str = "CLOSE_AT_LAST_BAR"


@dataclass
class SimulatedTrade:
    trade_id: str
    trade_proposal_id: str
    direction: str
    setup_family: str
    decision_time: str
    entry_time: str
    entry_index: int
    exit_time: str | None
    exit_index: int | None
    planned_entry: float
    stop: float
    target: float
    entry_fill: float
    exit_fill: float | None
    exit_reason: str  # STOP | TARGET | GAP_STOP | END_OF_DATA | MAX_HOLDING
    one_R_pips: float
    spread_entry_pips: float
    spread_exit_pips: float
    spread_status: str
    slippage_entry_pips: float
    slippage_exit_pips: float
    commission_pips: float | None
    commission_status: str
    swap_pips: float | None
    swap_status: str
    swap_nights: int
    gross_pips: float
    net_pips: float | None  # None only when a required cost is UNKNOWN and no scenario was supplied
    gross_R: float
    net_R: float | None
    mae_pips: float
    mfe_pips: float
    mae_R: float
    mfe_R: float
    mae_atr: float | None
    mfe_atr: float | None
    capture_ratio: float | None
    holding_bars: int
    holding_hours: float
    ambiguity: str = "NONE"  # NONE | AMBIGUOUS | UNRESOLVED | RESOLVED_LTF
    flags: list = field(default_factory=list)
    scenario: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _sides(o, h, l, c, spread_price, price_type):
    if price_type == "ASK":
        return (o - spread_price, h - spread_price, l - spread_price, c - spread_price), (o, h, l, c)
    if price_type == "MID":
        half = spread_price / 2
        return (o - half, h - half, l - half, c - half), (o + half, h + half, l + half, c + half)
    return (o, h, l, c), (o + spread_price, h + spread_price, l + spread_price, c + spread_price)


def _resolve_ltf(ltf: pd.DataFrame | None, bar_open: pd.Timestamp, bar_minutes: int, d: int, stop: float, target: float,
                 spread_price: float, price_type: str) -> str | None:
    """'STOP' / 'TARGET' from lower-timeframe bars inside the H1 bar, or None when unavailable/ambiguous."""
    if ltf is None or not len(ltf):
        return None
    end = bar_open + pd.Timedelta(minutes=bar_minutes)
    sub = ltf[(ltf["timestamp"] >= bar_open) & (ltf["timestamp"] < end)]
    if not len(sub):
        return None
    for _, r in sub.iterrows():
        bid, ask = _sides(r["open"], r["high"], r["low"], r["close"], spread_price, price_type)
        ex = bid if d > 0 else ask
        hi, lo = ex[1], ex[2]
        s_hit = lo <= stop if d > 0 else hi >= stop
        t_hit = hi >= target if d > 0 else lo <= target
        if s_hit and t_hit:
            return None  # still ambiguous at the lower timeframe
        if s_hit:
            return "STOP"
        if t_hit:
            return "TARGET"
    return None


def simulate_trade(bars: pd.DataFrame, entry_index: int, proposal: dict, scenario: CostScenario, price_type: str = "BID", *,
                   ambiguity_policy: str = "CONSERVATIVE", ltf: pd.DataFrame | None = None, exit_plan: ExitPlan | None = None,
                   atr_pips: float | None = None, pip: float = 0.01, bar_minutes: int = 60, trade_id: str | None = None) -> SimulatedTrade:
    """Simulate one proposal from the OPEN of ``entry_index`` (the decision bar) until exit."""
    if ambiguity_policy not in AMBIGUITY_POLICIES:
        raise ValueError(ambiguity_policy)
    if price_type not in ("BID", "ASK", "MID", "UNKNOWN"):
        raise ValueError(f"price_type must be BID / ASK / MID / UNKNOWN, not {price_type!r}")
    plan = exit_plan or ExitPlan()
    d = 1 if proposal["direction"] == "LONG" else -1
    stop = float(proposal["proposed_stop_price"])
    target = float(proposal["primary_target"]["price"])
    planned = float(proposal["executable_reference_price"])
    one_r = float(proposal["risk_unit"]["one_R_pips"])
    key = trade_id or proposal["trade_proposal_id"]
    flags = ["PRICE_TYPE_UNKNOWN"] if price_type == "UNKNOWN" else []
    ts = bars["timestamp"]
    O, Hh, L, C = (bars[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    S = bars["spread"].to_numpy(float) if "spread" in bars.columns else np.full(len(bars), np.nan)
    n = len(bars)
    atr_p = atr_pips if atr_pips is not None else float((proposal.get("volatility") or {}).get("h1_atr_pips") or 0.0)

    def spread_at(j):
        v, status = scenario.spread.spread_pips(S[j])
        return v, status

    sp_in, sp_status = spread_at(entry_index)
    slip_in = scenario.slippage.side_pips(sp_in, atr_p, f"{key}|entry")
    bid, ask = _sides(O[entry_index], Hh[entry_index], L[entry_index], C[entry_index], sp_in * pip, price_type)
    raw_entry = (ask if d > 0 else bid)[0]
    entry_fill = raw_entry + d * slip_in * pip
    exit_reason, exit_idx, exit_fill, raw_exit, amb = None, None, None, None, "NONE"
    adverse, favourable = 0.0, 0.0
    sp_out = sp_in
    last = n - 1 if plan.max_holding_bars is None else min(n - 1, entry_index + plan.max_holding_bars)
    j = entry_index
    while j <= last:
        sp_j, st_j = spread_at(j)
        if st_j != "HISTORICAL" and sp_status == "HISTORICAL":
            sp_status = "MIXED_HISTORICAL_AND_ASSUMED"
        b, a = _sides(O[j], Hh[j], L[j], C[j], sp_j * pip, price_type)
        ex = b if d > 0 else a  # the side the position exits on
        o_, h_, l_ = ex[0], ex[1], ex[2]
        if j > entry_index:  # opening gap through a level
            if (d > 0 and o_ <= stop) or (d < 0 and o_ >= stop):
                exit_reason, exit_idx, raw_exit = "GAP_STOP", j, o_
            elif (d > 0 and o_ >= target) or (d < 0 and o_ <= target):
                exit_reason, exit_idx, raw_exit = "TARGET", j, target
        if exit_reason is None:
            s_hit = (l_ <= stop) if d > 0 else (h_ >= stop)
            t_hit = (h_ >= target) if d > 0 else (l_ <= target)
            if s_hit and t_hit:
                res = None
                if ambiguity_policy == "LOWER_TIMEFRAME_REQUIRED":
                    res = _resolve_ltf(ltf, ts.iloc[j], bar_minutes, d, stop, target, sp_j * pip, price_type)
                    amb = "RESOLVED_LTF" if res else "UNRESOLVED"
                elif ambiguity_policy == "AMBIGUOUS":
                    amb = "AMBIGUOUS"
                else:
                    amb = "AMBIGUOUS_CONSERVATIVE"
                res = res or "STOP"
                exit_reason, exit_idx, raw_exit = res, j, (stop if res == "STOP" else target)
            elif s_hit:
                exit_reason, exit_idx, raw_exit = "STOP", j, stop
            elif t_hit:
                exit_reason, exit_idx, raw_exit = "TARGET", j, target
        # excursions (bar granularity; on the exit bar the exit-side extreme is capped at the exit level)
        adv_px = (entry_fill - l_) if d > 0 else (h_ - entry_fill)
        fav_px = (h_ - entry_fill) if d > 0 else (entry_fill - l_)
        if exit_reason in ("STOP", "GAP_STOP"):
            adv_px = d * (entry_fill - raw_exit) if raw_exit is not None else adv_px
        if exit_reason == "TARGET":
            fav_px = d * (raw_exit - entry_fill)
        adverse, favourable = max(adverse, adv_px), max(favourable, fav_px)
        if exit_reason:
            sp_out = sp_j
            break
        j += 1
    if exit_reason is None:  # end of data / holding cap: close at the last bar close on the exit side
        j = last
        sp_out, _ = spread_at(j)
        b, a = _sides(O[j], Hh[j], L[j], C[j], sp_out * pip, price_type)
        raw_exit = (b if d > 0 else a)[3]
        exit_idx = j
        exit_reason = "MAX_HOLDING" if plan.max_holding_bars is not None and j < n - 1 else "END_OF_DATA"
        flags.append("OPEN_AT_END" if exit_reason == "END_OF_DATA" else "TIME_CAPPED")
    slip_out = 0.0
    if exit_reason in ("STOP", "GAP_STOP"):
        slip_out = scenario.slippage.side_pips(sp_out, atr_p, f"{key}|exit") + max(0.0, scenario.slippage.stop_extra_pips)
    elif exit_reason in ("END_OF_DATA", "MAX_HOLDING"):
        slip_out = scenario.slippage.side_pips(sp_out, atr_p, f"{key}|exit")
    exit_fill = raw_exit - d * slip_out * pip  # limit target fills exactly at the target (no positive slippage)
    comm, comm_status = scenario.commission.pips()
    t_in, t_out = ts.iloc[entry_index], ts.iloc[exit_idx]
    swap, swap_status, nights = scenario.swap.pips(d, t_in, t_out + pd.Timedelta(minutes=bar_minutes))
    # gross: the same exit event measured on the data's own price representation (no spread, slippage, commission
    # or swap): entry at the bar open, exit at the exit-side level translated back into the representation
    sp_px = sp_out * pip
    side_offset = {"ASK": (-sp_px, 0.0), "MID": (-sp_px / 2, sp_px / 2)}.get(price_type, (0.0, sp_px))  # (bid, ask) - repr
    repr_exit = raw_exit - (side_offset[0] if d > 0 else side_offset[1])
    gross_pips = d * (repr_exit - O[entry_index]) / pip
    net_pips = d * (exit_fill - entry_fill) / pip
    if comm is None:
        flags.append("COMMISSION_UNKNOWN")
    else:
        net_pips -= comm
    if swap is None:
        if nights:
            flags.append("SWAP_UNKNOWN")
    else:
        net_pips += swap
    net_pips_final = None if comm is None else net_pips
    if exit_reason == "GAP_STOP":
        flags.append("GAP_THROUGH_STOP")
    net_r = None if net_pips_final is None else net_pips_final / one_r
    mfe_pips, mae_pips = favourable / pip, adverse / pip
    return SimulatedTrade(
        trade_id=key, trade_proposal_id=proposal["trade_proposal_id"], direction=proposal["direction"],
        setup_family=proposal.get("setup_family", ""), decision_time=str(proposal["timestamp"]), entry_time=t_in.isoformat(),
        entry_index=int(entry_index), exit_time=t_out.isoformat(), exit_index=int(exit_idx), planned_entry=planned,
        stop=stop, target=target, entry_fill=round(entry_fill, 6), exit_fill=round(exit_fill, 6), exit_reason=exit_reason,
        one_R_pips=one_r, spread_entry_pips=sp_in, spread_exit_pips=sp_out, spread_status=sp_status,
        slippage_entry_pips=slip_in, slippage_exit_pips=slip_out, commission_pips=comm, commission_status=comm_status,
        swap_pips=swap, swap_status=swap_status, swap_nights=nights, gross_pips=round(gross_pips, 4),
        net_pips=None if net_pips_final is None else round(net_pips_final, 4), gross_R=round(gross_pips / one_r, 6),
        net_R=None if net_r is None else round(net_r, 6), mae_pips=round(mae_pips, 3), mfe_pips=round(mfe_pips, 3),
        mae_R=round(mae_pips / one_r, 4), mfe_R=round(mfe_pips / one_r, 4),
        mae_atr=round(mae_pips / atr_p, 4) if atr_p else None, mfe_atr=round(mfe_pips / atr_p, 4) if atr_p else None,
        capture_ratio=round((net_r if net_r is not None else gross_pips / one_r) / (mfe_pips / one_r), 4) if mfe_pips > 0 else None,
        holding_bars=int(exit_idx - entry_index + 1), holding_hours=(t_out - t_in).total_seconds() / 3600 + bar_minutes / 60,
        ambiguity=amb, flags=flags, scenario=scenario.name)
