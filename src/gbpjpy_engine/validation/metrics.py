"""R-based performance statistics with uncertainty (Phase 1H).

Strategy edge is measured in R multiples (account-independent) before any
compounding.  Every summary carries a sample-size warning level; statistics on
small samples are reported with their uncertainty, never with false precision.
Bootstrap routines take an explicit seed and are deterministic.

Risk-adjusted ratios (Sharpe / Sortino / Calmar) are descriptive only and are
never an optimisation objective.  Sampling and annualisation assumptions are
returned with each value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

BE_EPS_R = 0.05  # |R| below this is a break-even trade (descriptive convention)
SAMPLE_LEVELS = ((30, "INSUFFICIENT_SAMPLE"), (100, "SMALL_SAMPLE"), (300, "MODERATE_SAMPLE"))


def sample_warning(n: int) -> dict:
    for limit, level in SAMPLE_LEVELS:
        if n < limit:
            return {"n": n, "level": level,
                    "message": f"{n} trades: statistics are unstable; treat point estimates as highly uncertain"}
    return {"n": n, "level": "ADEQUATE_FOR_BASIC_STATISTICS",
            "message": "sample adequate for basic statistics; this is NOT evidence of a persistent edge"}


def _arr(r) -> np.ndarray:
    a = np.asarray([x for x in r if x is not None and np.isfinite(x)], dtype=float)
    return a


def r_distribution(r) -> dict:
    a = _arr(r)
    if not len(a):
        return {"n": 0}
    pos, neg = a[a > BE_EPS_R], a[a < -BE_EPS_R]
    q = lambda x, p: float(np.quantile(x, p)) if len(x) else None  # noqa: E731
    return {"n": int(len(a)), "mean_R": float(a.mean()), "median_R": float(np.median(a)), "total_R": float(a.sum()),
            "std_R": float(a.std(ddof=1)) if len(a) > 1 else None, "min_R": float(a.min()), "max_R": float(a.max()),
            "positive": {"n": int(len(pos)), "mean": float(pos.mean()) if len(pos) else None, "q25": q(pos, .25),
                         "q50": q(pos, .5), "q75": q(pos, .75)},
            "negative": {"n": int(len(neg)), "mean": float(neg.mean()) if len(neg) else None, "q25": q(neg, .25),
                         "q50": q(neg, .5), "q75": q(neg, .75)},
            "tail": {"q01": q(a, .01), "q05": q(a, .05), "worse_than_minus_1R": int((a < -1.0 - 1e-9).sum()),
                     "worse_than_minus_1_5R": int((a < -1.5).sum()), "worst": float(a.min())}}


def expectancy(r) -> dict:
    a = _arr(r)
    n = len(a)
    if not n:
        return {"n": 0, "expectancy_R_direct": None}
    win, loss = a[a > BE_EPS_R], a[a < -BE_EPS_R]
    be = n - len(win) - len(loss)
    wr, lr = len(win) / n, len(loss) / n
    aw = float(win.mean()) if len(win) else 0.0
    al = float(loss.mean()) if len(loss) else 0.0
    be_mean = float(a[(a >= -BE_EPS_R) & (a <= BE_EPS_R)].mean()) if be else 0.0
    formula = wr * aw + lr * al + (be / n) * be_mean
    return {"n": n, "wins": int(len(win)), "losses": int(len(loss)), "break_even": int(be), "win_rate": wr, "loss_rate": lr,
            "average_win_R": aw if len(win) else None, "average_loss_R": al if len(loss) else None,
            "expectancy_R_formula": formula, "expectancy_R_direct": float(a.mean()),
            "note": "direct = mean of the complete R distribution (authoritative); formula shown for comparison"}


def profit_factor(r) -> float | None:
    a = _arr(r)
    gp, gl = a[a > 0].sum(), -a[a < 0].sum()
    if gl <= 0:
        return None  # undefined (no losses) - never reported as infinite edge
    return float(gp / gl)


def drawdown(r, times=None) -> dict:
    a = _arr(r)
    if not len(a):
        return {"max_drawdown_R": 0.0, "max_drawdown_trades": 0}
    eq = np.concatenate([[0.0], np.cumsum(a)])
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    i = int(dd.argmax())
    pk = int(np.flatnonzero(eq[: i + 1] == peak[i])[0])
    rec = np.flatnonzero(eq[i:] >= peak[i])
    rec_i = int(i + rec[0]) if len(rec) else None
    longest, cur = 0, 0
    for x in dd[1:]:
        cur = cur + 1 if x > 1e-12 else 0
        longest = max(longest, cur)
    out = {"max_drawdown_R": float(dd.max()), "peak_trade": pk, "trough_trade": i, "recovery_trade": rec_i,
           "max_drawdown_trades": i - pk, "trades_to_recover": (rec_i - i) if rec_i is not None else None,
           "recovered": rec_i is not None, "longest_underwater_trades": longest, "curve_R": eq.tolist(),
           "drawdown_curve_R": (-dd).tolist()}
    if times is not None and len(times) == len(a):
        t = pd.to_datetime(pd.Series(list(times)), utc=True)

        def at(k):  # equity point k exists after trade k (k >= 1)
            return t.iloc[max(k - 1, 0)]

        out["max_drawdown_duration_days"] = (at(i) - at(pk)).total_seconds() / 86400
        out["time_to_recover_days"] = (at(rec_i) - at(i)).total_seconds() / 86400 if rec_i is not None else None
    return out


def drawdown_episodes(r, times=None) -> list[dict]:
    a = _arr(r)
    eq = np.concatenate([[0.0], np.cumsum(a)])
    peak = np.maximum.accumulate(eq)
    eps, start = [], None
    for k in range(1, len(eq)):
        under = eq[k] < peak[k] - 1e-12
        if under and start is None:
            start = k - 1
        if not under and start is not None:
            seg = eq[start: k + 1]
            trough = start + int(seg.argmin())
            eps.append({"start": start, "trough": trough, "recovered_at": k, "depth_R": float(peak[start] - eq[trough]),
                        "trades_to_trough": trough - start, "trades_to_recover": k - trough})
            start = None
    if start is not None:
        trough = start + int(eq[start:].argmin())
        eps.append({"start": start, "trough": trough, "recovered_at": None, "depth_R": float(peak[start] - eq[trough]),
                    "trades_to_trough": trough - start, "trades_to_recover": None})
    return eps


def streaks(r) -> dict:
    a = _arr(r)
    mw = ml = cw = cl = 0
    for x in a:
        if x > BE_EPS_R:
            cw, cl = cw + 1, 0
        elif x < -BE_EPS_R:
            cl, cw = cl + 1, 0
        else:
            cw = cl = 0
        mw, ml = max(mw, cw), max(ml, cl)
    return {"max_consecutive_wins": mw, "max_consecutive_losses": ml}


def exposure_fraction(entries, exits, start, end) -> float | None:
    if not len(entries):
        return 0.0
    iv = sorted((pd.Timestamp(a), pd.Timestamp(b)) for a, b in zip(entries, exits))
    total, cs, ce = pd.Timedelta(0), None, None
    for a, b in iv:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            total += ce - cs
            cs, ce = a, b
    total += ce - cs
    span = pd.Timestamp(end) - pd.Timestamp(start)
    return float(total / span) if span > pd.Timedelta(0) else None


def risk_adjusted(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    a = _arr(df[rcol]) if len(df) else np.array([])
    n = len(a)
    res = {"sampling": "per-trade R and daily R (exit date; zero on business days without exits)",
           "annualisation": "per-trade Sharpe x sqrt(trades per year); daily Sharpe x sqrt(252)",
           "limitations": "trades are not i.i.d.; R ignores compounding; small samples make these ratios unstable; "
                          "descriptive only - never optimised"}
    if n < 2 or a.std(ddof=1) == 0:
        return dict(res, status="NOT_MEANINGFUL", sharpe_per_trade=None)
    t = pd.to_datetime(df["exit_time"], utc=True)
    years = max((t.max() - pd.to_datetime(df["entry_time"], utc=True).min()).days / 365.25, 1 / 365.25)
    tpy = n / years
    sh = a.mean() / a.std(ddof=1)
    down = a[a < 0]
    so = a.mean() / down.std(ddof=1) if len(down) > 1 and down.std(ddof=1) > 0 else None
    daily = pd.Series(a, index=t.dt.normalize().to_numpy()).groupby(level=0).sum()
    bdays = pd.bdate_range(daily.index.min(), daily.index.max(), tz="UTC")
    daily = daily.reindex(bdays, fill_value=0.0)
    dsh = daily.mean() / daily.std(ddof=1) * math.sqrt(252) if daily.std(ddof=1) > 0 else None
    mdd = drawdown(a)["max_drawdown_R"]
    calmar = (a.sum() / years) / mdd if mdd > 0 else None
    return dict(res, status="OK" if n >= 30 else "SMALL_SAMPLE_UNSTABLE", sharpe_per_trade=float(sh),
                sharpe_annualised_from_trades=float(sh * math.sqrt(tpy)), sortino_per_trade=None if so is None else float(so),
                sharpe_daily_annualised=None if dsh is None else float(dsh), calmar_R=None if calmar is None else float(calmar),
                trades_per_year=float(tpy), years=float(years))


# ---------------------------------------------------------------------------
# Bootstrap (deterministic, seeded)
# ---------------------------------------------------------------------------
def _resample_idx(n: int, rng, block: int | None) -> np.ndarray:
    if not block or block <= 1:
        return rng.integers(0, n, n)
    starts = rng.integers(0, max(n - block + 1, 1), math.ceil(n / block))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def bootstrap(r, stat, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05, block: int | None = None) -> dict:
    a = _arr(r)
    if len(a) < 2:
        return {"n": len(a), "status": "NOT_ESTIMABLE", "seed": seed}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat(a[_resample_idx(len(a), rng, block)])
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return {"n": int(len(a)), "estimate": stat(a), "median": float(np.median(vals)) if len(vals) else None,
            "ci_low": float(np.quantile(vals, alpha / 2)) if len(vals) else None,
            "ci_high": float(np.quantile(vals, 1 - alpha / 2)) if len(vals) else None,
            "p_le_zero": float((vals <= 0).mean()) if len(vals) else None, "n_boot": n_boot, "valid_draws": int(len(vals)),
            "seed": seed, "block": block, "method": "moving-block bootstrap" if block and block > 1 else "i.i.d. bootstrap",
            "caveat": "resampling assumes the historical trades are representative; it cannot capture regime change"}


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def confidence_intervals(r, seed: int = 0, n_boot: int = 2000, block: int | None = None) -> dict:
    a = _arr(r)
    wins = int((a > BE_EPS_R).sum())
    pf = lambda x: profit_factor(x)  # noqa: E731
    return {"win_rate_wilson_95": wilson(wins, len(a)),
            "win_rate": bootstrap(a, lambda x: float((x > BE_EPS_R).mean()), n_boot, seed, block=block),
            "mean_R": bootstrap(a, lambda x: float(x.mean()), n_boot, seed, block=block),
            "expectancy_R": bootstrap(a, lambda x: float(x.mean()), n_boot, seed + 1, block=block),
            "profit_factor": bootstrap(a, pf, n_boot, seed + 2, block=block),
            "sample": sample_warning(len(a))}


# ---------------------------------------------------------------------------
# Summaries and breakdowns
# ---------------------------------------------------------------------------
def summarise(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    if df is None or not len(df):
        return {"trades": 0, "sample": sample_warning(0)}
    df = df.sort_values("entry_time")
    r = df[rcol].to_numpy(float)
    ex = expectancy(r)
    dd = drawdown(r, df["exit_time"].tolist())
    hold = df["holding_hours"].to_numpy(float)
    return {"trades": int(len(df)), "wins": ex["wins"], "losses": ex["losses"], "break_even": ex["break_even"],
            "win_rate": ex["win_rate"], "loss_rate": ex["loss_rate"],
            "gross_profit_R": float(r[r > 0].sum()), "gross_loss_R": float(-r[r < 0].sum()), "profit_factor": profit_factor(r),
            "expectancy_R": ex["expectancy_R_direct"], "expectancy_R_formula": ex["expectancy_R_formula"],
            "mean_R": float(r.mean()), "median_R": float(np.median(r)), "total_R": float(r.sum()),
            "std_R": float(r.std(ddof=1)) if len(r) > 1 else None,
            "max_drawdown_R": dd["max_drawdown_R"], "max_drawdown_duration_days": dd.get("max_drawdown_duration_days"),
            "trades_to_recover": dd["trades_to_recover"], "time_to_recover_days": dd.get("time_to_recover_days"),
            "average_holding_hours": float(hold.mean()), "median_holding_hours": float(np.median(hold)),
            **streaks(r), "sample": sample_warning(len(df)), "r_column": rcol}


def breakdown(df: pd.DataFrame, by, rcol: str = "net_R") -> dict:
    if df is None or not len(df):
        return {}
    out = {}
    for k, g in df.groupby(by, dropna=False, sort=True):
        k = k if not isinstance(k, tuple) else "|".join(map(str, k))
        s = summarise(g, rcol)
        out[str(k)] = {x: s[x] for x in ("trades", "win_rate", "expectancy_R", "profit_factor", "total_R", "max_drawdown_R",
                                         "median_R")} | {"sample_level": s["sample"]["level"]}
    return out


def calendar_breakdowns(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    if df is None or not len(df):
        return {"year": {}, "quarter": {}, "month": {}}
    t = pd.to_datetime(df["entry_time"], utc=True)
    d = df.assign(_y=t.dt.year, _q=t.dt.year.astype(str) + "Q" + t.dt.quarter.astype(str),
                  _m=t.dt.strftime("%Y-%m"))
    return {"year": breakdown(d, "_y", rcol), "quarter": breakdown(d, "_q", rcol), "month": breakdown(d, "_m", rcol),
            "note": "losing periods are reported; no period is ever excluded"}


def concentration(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    if df is None or not len(df):
        return {}
    r = df[rcol].to_numpy(float)
    total = r.sum()
    srt = np.sort(r)[::-1]
    t = pd.to_datetime(df["entry_time"], utc=True)
    share = lambda x: float(x / total) if total > 0 else None  # noqa: E731
    per = {}
    for name, key in (("month", t.dt.strftime("%Y-%m")), ("quarter", t.dt.year.astype(str) + "Q" + t.dt.quarter.astype(str)),
                      ("year", t.dt.year.astype(str))):
        g = pd.Series(r).groupby(key.to_numpy()).sum()
        per[f"best_{name}"] = {"period": str(g.idxmax()), "R": float(g.max()), "share_of_total": share(g.max())}
    removed = {}
    for k in (1, 5, 10):
        if len(r) > k:
            rest = np.sort(r)[: len(r) - k]
            removed[f"without_best_{k}"] = {"trades": int(len(rest)), "total_R": float(rest.sum()), "mean_R": float(rest.mean()),
                                            "profit_factor": profit_factor(rest)}
        else:
            removed[f"without_best_{k}"] = {"status": "NOT_MEANINGFUL_SAMPLE_TOO_SMALL"}
    flag = None
    if total > 0 and len(r) > 5 and srt[:5].sum() >= total:
        flag = "OUTLIER_DEPENDENT: the best 5 trades account for all net profit"
    return {"total_R": float(total), "best_trade_share": share(srt[0]), "best_5_share": share(srt[:5].sum()),
            "best_10_share": share(srt[:10].sum()), **per, "remove_best": removed, "flag": flag,
            "worst": {"worst_trade_R": float(r.min()), "worst_5_R": [float(x) for x in np.sort(r)[:5]],
                      "gap_or_slippage_tail": int((r < -1.2).sum())}}


def distribution_stability(df: pd.DataFrame, rcol: str = "net_R", segments: int = 3) -> dict:
    if df is None or len(df) < 2 * segments:
        return {"status": "NOT_MEANINGFUL_SAMPLE_TOO_SMALL"}
    d = df.sort_values("entry_time")
    parts = np.array_split(d[rcol].to_numpy(float), segments)
    res = {"segments": [{"n": int(len(p)), "mean_R": float(p.mean()), "median_R": float(np.median(p)),
                         "win_rate": float((p > BE_EPS_R).mean()), "profit_factor": profit_factor(p)} for p in parts]}
    ks = []
    for i in range(segments):
        for j in range(i + 1, segments):
            a, b = np.sort(parts[i]), np.sort(parts[j])
            grid = np.concatenate([a, b])
            fa = np.searchsorted(a, grid, side="right") / len(a)
            fb = np.searchsorted(b, grid, side="right") / len(b)
            ks.append({"pair": [i, j], "ks_statistic": float(np.abs(fa - fb).max())})
    res["ks"] = ks
    res["note"] = "two-sample KS distance between chronological segments (descriptive; no p-value claimed)"
    return res


def trade_frequency(df: pd.DataFrame, start, end) -> dict:
    if df is None or not len(df):
        return {"trades": 0}
    t = pd.to_datetime(df["entry_time"], utc=True)
    span_days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)
    weekly = t.dt.tz_convert(None).dt.to_period("W").value_counts()
    all_weeks = pd.period_range(pd.Timestamp(start).tz_convert(None), pd.Timestamp(end).tz_convert(None), freq="W")
    wk = weekly.reindex(all_weeks, fill_value=0)
    monthly = t.dt.tz_convert(None).dt.to_period("M").value_counts()
    return {"trades": int(len(df)), "per_week_mean": float(len(df) / (span_days / 7)),
            "per_month_mean": float(len(df) / (span_days / 30.4375)), "per_year_mean": float(len(df) / (span_days / 365.25)),
            "weeks_with_zero_trades_pct": float((wk == 0).mean()) if len(wk) else None,
            "per_week_distribution": {int(k): int(v) for k, v in wk.value_counts().sort_index().items()},
            "per_month_max": int(monthly.max()), "per_month_min_observed": int(monthly.min())}
