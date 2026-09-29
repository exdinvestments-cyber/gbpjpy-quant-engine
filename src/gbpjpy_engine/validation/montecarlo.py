"""Monte Carlo and stress research on HISTORICAL trade outcomes (Phase 1H).

Every routine takes an explicit seed and returns it with the result.  All of it
resamples or perturbs outcomes that were actually simulated from the data -
nothing here invents a trade.  Shuffled or resampled sequences are NOT a
perfect model of the future (regimes change, trades cluster); the caveat is
attached to every output.  A simulated probability of zero is never reported
as zero: the rule-of-three upper bound is reported instead.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..risk.research import RiskOfRuinInputs
from .metrics import BE_EPS_R, _arr, _resample_idx, wilson

CAVEAT = ("simulated from historical outcomes under stated assumptions; resampling cannot represent regime change, "
          "and future drawdowns can exceed every simulated value")


def _q(x) -> dict:
    x = np.asarray(x, float)
    if not len(x):
        return {}
    return {"p05": float(np.quantile(x, .05)), "p25": float(np.quantile(x, .25)), "median": float(np.median(x)),
            "p75": float(np.quantile(x, .75)), "p95": float(np.quantile(x, .95)), "max": float(x.max()), "min": float(x.min())}


def _path_stats(a: np.ndarray) -> tuple:
    eq = np.concatenate([[0.0], np.cumsum(a)])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    under = dd[1:] > 1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    ml = cl = 0
    for x in a:
        cl = cl + 1 if x < -BE_EPS_R else 0
        ml = max(ml, cl)
    return float(eq[-1]), float(dd.max()), ml, longest, float(under.mean()) if len(under) else 0.0


def sequence_monte_carlo(r, n_sims: int = 2000, seed: int = 0, method: str = "BOOTSTRAP", block: int | None = None) -> dict:
    """Alternative trade sequences: PERMUTATION (same trades, new order), BOOTSTRAP (i.i.d.) or BLOCK (keeps clustering)."""
    a = _arr(r)
    n = len(a)
    if n < 5:
        return {"status": "NOT_MEANINGFUL_SAMPLE_TOO_SMALL", "n": n, "seed": seed}
    rng = np.random.default_rng(seed)
    ends, dds, streak, uw_len, uw_frac = [], [], [], [], []
    for _ in range(n_sims):
        if method == "PERMUTATION":
            s = a[rng.permutation(n)]
        else:
            s = a[_resample_idx(n, rng, block if method == "BLOCK" else None)]
        e, d, ml, lu, fu = _path_stats(s)
        ends.append(e), dds.append(d), streak.append(ml), uw_len.append(lu), uw_frac.append(fu)
    hist = _path_stats(a)
    return {"method": method, "block": block, "n_trades": n, "n_sims": n_sims, "seed": seed,
            "ending_R": _q(ends), "max_drawdown_R": _q(dds), "max_losing_streak": _q(streak),
            "longest_underwater_trades": _q(uw_len), "fraction_time_underwater": _q(uw_frac),
            "probability_ending_R_le_0": float(np.mean(np.asarray(ends) <= 0)),
            "historical": {"ending_R": hist[0], "max_drawdown_R": hist[1], "max_losing_streak": hist[2]},
            "caveat": CAVEAT}


def cost_stress(df: pd.DataFrame, extra_pips=(0.0, 0.5, 1.0, 2.0, 3.0, 5.0), rcol: str = "net_R") -> dict:
    """Additional round-turn cost (pips) deducted from every trade; where does expectancy reach zero?"""
    if not len(df):
        return {"status": "NO_TRADES"}
    base = df[rcol].to_numpy(float)
    one_r = df["one_R_pips"].to_numpy(float)
    rows = []
    for x in extra_pips:
        r = base - x / one_r
        rows.append({"extra_round_turn_pips": x, "expectancy_R": float(r.mean()), "total_R": float(r.sum()),
                     "win_rate": float((r > BE_EPS_R).mean())})
    be = float(base.mean() / np.mean(1 / one_r)) if base.mean() > 0 else 0.0
    return {"grid": rows, "breakeven_extra_round_turn_pips": be,
            "interpretation": "edge disappears if real costs exceed the modelled costs by this many pips per round turn"
            if be > 0 else "expectancy is already <= 0 under the modelled costs"}


def random_cost_stress(df: pd.DataFrame, n_sims=1000, seed=0, spread_extra=(0.0, 2.0), slip_extra=(0.0, 1.5),
                       commission_extra=(0.0, 0.5), rcol="net_R") -> dict:
    if not len(df):
        return {"status": "NO_TRADES", "seed": seed}
    rng = np.random.default_rng(seed)
    base, one_r = df[rcol].to_numpy(float), df["one_R_pips"].to_numpy(float)
    exp = []
    for _ in range(n_sims):
        extra = rng.uniform(*spread_extra, len(base)) + rng.uniform(*slip_extra, len(base)) * 2 + rng.uniform(*commission_extra)
        exp.append(float((base - extra / one_r).mean()))
    return {"expectancy_R": _q(exp), "probability_expectancy_le_0": float(np.mean(np.asarray(exp) <= 0)), "seed": seed,
            "ranges_pips": {"spread_extra": spread_extra, "slippage_extra_per_side": slip_extra,
                            "commission_extra": commission_extra}, "caveat": CAVEAT}


def missed_trades(r, fraction: float = 0.1, n_sims: int = 1000, seed: int = 0) -> dict:
    """Random removal (downtime, disconnects, rejections, missed signals)."""
    a = _arr(r)
    if len(a) < 5:
        return {"status": "NOT_MEANINGFUL_SAMPLE_TOO_SMALL", "seed": seed}
    rng = np.random.default_rng(seed)
    exp, tot, dds = [], [], []
    for _ in range(n_sims):
        keep = a[rng.random(len(a)) >= fraction]
        if not len(keep):
            continue
        exp.append(keep.mean()), tot.append(keep.sum()), dds.append(_path_stats(keep)[1])
    return {"fraction_removed": fraction, "expectancy_R": _q(exp), "total_R": _q(tot), "max_drawdown_R": _q(dds),
            "seed": seed, "n_sims": n_sims, "caveat": CAVEAT}


def worse_fills(df: pd.DataFrame, max_entry_pips=2.0, max_exit_pips=2.0, n_sims=1000, seed=0, rcol="net_R") -> dict:
    if not len(df):
        return {"status": "NO_TRADES", "seed": seed}
    rng = np.random.default_rng(seed)
    base, one_r = df[rcol].to_numpy(float), df["one_R_pips"].to_numpy(float)
    exp = []
    for _ in range(n_sims):
        worse = rng.uniform(0, max_entry_pips, len(base)) + rng.uniform(0, max_exit_pips, len(base))
        exp.append(float((base - worse / one_r).mean()))
    return {"expectancy_R": _q(exp), "probability_expectancy_le_0": float(np.mean(np.asarray(exp) <= 0)),
            "max_entry_pips": max_entry_pips, "max_exit_pips": max_exit_pips, "seed": seed, "caveat": CAVEAT}


def tail_stress(r, probability: float = 0.02, extra_loss_R: tuple = (0.5, 2.0), n_sims: int = 1000, seed: int = 0) -> dict:
    """Occasional adverse gap / slippage events: a stop is NOT always exactly -1R."""
    a = _arr(r)
    if not len(a):
        return {"status": "NO_TRADES", "seed": seed}
    rng = np.random.default_rng(seed)
    exp, worst, dds = [], [], []
    for _ in range(n_sims):
        hit = rng.random(len(a)) < probability
        s = a - hit * rng.uniform(*extra_loss_R, len(a))
        exp.append(s.mean()), worst.append(s.min()), dds.append(_path_stats(s)[1])
    return {"event_probability": probability, "extra_loss_R_range": extra_loss_R, "expectancy_R": _q(exp),
            "worst_trade_R": _q(worst), "max_drawdown_R": _q(dds), "seed": seed, "caveat": CAVEAT}


def _prob_report(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    if k == 0:
        return {"observed": 0, "n": n, "reported": f"< {3 / n:.4g} (no ruin in {n} simulations; rule-of-three 95% bound)",
                "upper_95": 3 / n, "note": "never reported as zero"}
    return {"observed": k / n, "n": n, "wilson_95": [lo, hi]}


def risk_of_ruin(inputs: RiskOfRuinInputs, risk_percent: float, horizon_trades: int = 250, n_sims: int = 5000, seed: int = 0,
                 block: int | None = None) -> dict:
    """Empirical risk-of-ruin research using the Phase 1F ``RiskOfRuinInputs`` interface.

    Fixed-fractional compounding: equity *= 1 + risk% x R for R drawn from the empirical distribution (block
    resampling when ``block`` is set, as a crude dependency model), plus optional tail losses.  Ruin = equity falls
    below (1 - ruin_threshold_percent)."""
    dist = _arr(inputs.win_loss_distribution_R)
    if len(dist) < 5:
        return {"status": "NOT_ESTIMABLE_INSUFFICIENT_OUTCOMES", "seed": seed}
    thr = (inputs.ruin_threshold_percent or 50.0) / 100.0
    tail = _arr(inputs.tail_loss_distribution_R)
    rng = np.random.default_rng(seed)
    ruined, ends, mdd = 0, [], []
    f = risk_percent / 100.0
    for _ in range(n_sims):
        reps = math.ceil(horizon_trades / len(dist))
        s = dist[np.concatenate([_resample_idx(len(dist), rng, block) for _ in range(reps)])[:horizon_trades]]
        if len(tail):
            mask = rng.random(len(s)) < 0.02
            s = np.where(mask, np.minimum(s, rng.choice(tail, len(s))), s)
        eq = np.cumprod(1 + f * s)
        eq = np.maximum(eq, 0)
        pk = np.maximum.accumulate(np.concatenate([[1.0], eq]))
        mdd.append(float(((pk[1:] - eq) / pk[1:]).max()))
        if (eq <= 1 - thr).any():
            ruined += 1
        ends.append(float(eq[-1]))
    return {"method": "empirical R resampling with fixed-fractional compounding" + (" (block)" if block else " (i.i.d.)"),
            "risk_percent": risk_percent, "ruin_threshold_percent": thr * 100, "horizon_trades": horizon_trades,
            "probability_of_ruin": _prob_report(ruined, n_sims), "ending_equity_multiple": _q(ends),
            "max_drawdown_fraction": _q(mdd), "dependency_assumption": inputs.trade_dependency or ("block resampling"
                                                                                                    if block else "i.i.d."),
            "tail_losses_used": int(len(tail)), "seed": seed, "caveat": CAVEAT}


def compounding(df: pd.DataFrame, risk_percent: float, start_balance: float = 10000.0, rcol: str = "net_R") -> dict:
    """Historical-order fixed-fractional growth.  Reported SEPARATELY from R expectancy - it cannot rescue a weak edge."""
    if not len(df):
        return {"status": "NO_TRADES"}
    r = df.sort_values("entry_time")[rcol].to_numpy(float)
    eq = start_balance * np.cumprod(1 + risk_percent / 100 * r)
    pk = np.maximum.accumulate(np.concatenate([[start_balance], eq]))
    return {"risk_percent": risk_percent, "start_balance": start_balance, "end_balance": float(eq[-1]),
            "return_pct": float(eq[-1] / start_balance * 100 - 100), "max_drawdown_pct": float(((pk[1:] - eq) / pk[1:]).max() * 100),
            "curve": eq.tolist(), "note": "compounding amplifies both gains and losses; judge the edge from R statistics"}


def risk_policy_comparison(r, policies=(0.25, 0.5, 1.0), horizon_trades=250, n_sims=2000, seed=0,
                           ruin_threshold_percent=30.0, block=None) -> dict:
    """A SMALL pre-declared set of fixed-fractional risk levels - informational; no policy is selected by ending balance."""
    out = {}
    for p in policies:
        rr = risk_of_ruin(RiskOfRuinInputs(win_loss_distribution_R=tuple(_arr(r)),
                                           ruin_threshold_percent=ruin_threshold_percent), p, horizon_trades, n_sims, seed, block)
        out[f"{p}%"] = {k: rr.get(k) for k in ("probability_of_ruin", "ending_equity_multiple", "max_drawdown_fraction",
                                              "status")}
    return {"policies": out, "selection_rule": "NONE - the highest historical ending balance is never a selection criterion",
            "seed": seed}


def capital_scale(stop_pips_distribution, risk_percent: float, max_volume: float = 50.0, pip_value_per_volume_gbp: float = 6.6,
                  capitals=(1e4, 1e5, 1e6, 1e7)) -> dict:
    """Where would position size meet practical constraints?  Linear scaling is NOT assumed."""
    s = _arr(stop_pips_distribution)
    if not len(s):
        return {"status": "NO_TRADES"}
    rows = []
    for c in capitals:
        vol = c * risk_percent / 100 / (s * pip_value_per_volume_gbp)
        rows.append({"capital": c, "median_volume": float(np.median(vol)), "p95_volume": float(np.quantile(vol, .95)),
                     "share_trades_above_max_volume": float((vol > max_volume).mean())})
    return {"rows": rows, "max_volume": max_volume, "pip_value_per_volume_assumption_gbp": pip_value_per_volume_gbp,
            "unmodelled": ["market impact / slippage growth with size", "broker dealing limits and last-look",
                           "margin changes and leverage caps", "liquidity in thin sessions and around news"],
            "note": "results do not scale linearly to very large capital; these constraints are not simulated"}
