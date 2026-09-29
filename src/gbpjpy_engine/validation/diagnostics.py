"""Descriptive research diagnostics (Phase 1H).

Everything here DESCRIBES historical behaviour and generates hypotheses.  No
function changes a strategy parameter, and nothing removes a family, session,
day or filter.  Counterfactual analyses of rejected opportunities are labelled
POST-DECISION DIAGNOSTIC and are never added to strategy results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import BE_EPS_R, breakdown, profit_factor, sample_warning

FAMILIES = ("TREND_PULLBACK_CONTINUATION", "BREAK_RETEST_CONTINUATION", "LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION",
            "COMPRESSION_EXPANSION_IN_H4_DIRECTION")
SCORE_COLUMNS = ("setup_score", "setup_confidence", "entry_quality_score", "trade_construction_quality_score")
POST_DECISION = "POST-DECISION DIAGNOSTIC (never counted as strategy performance)"


def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def enrich(out, trades: list) -> pd.DataFrame:
    """Trade ledger joined with the decision-time context (all values known at the decision)."""
    if not trades:
        return pd.DataFrame()
    props = {p["trade_proposal_id"]: p for p in out.trade.proposals}
    setups = {s.setup_id: s for s in out.h1.setups}
    fr, f1 = out.h1.frame, out.h1.features
    f4 = out.h4.features
    rows = []
    for t in trades:
        p = props[t.trade_proposal_id]
        i = max(t.entry_index - 1, 0)  # last completed bar at the decision
        s = setups.get(p.get("setup_id"))
        q = (s.qualified or {}) if s else {}
        h4i = fr["h4_index"].iloc[i]
        h4_atr_pct = float(f4["atr_percentile"].iloc[int(h4i)]) if pd.notna(h4i) and int(h4i) >= 0 else np.nan
        pt = p.get("primary_target") or {}
        sc = p.get("session_context") or {}
        rows.append({**t.to_dict(),
                     "setup_score": p.get("setup_score"), "setup_confidence": q.get("setup_confidence"),
                     "entry_quality_score": p.get("entry_quality_score"),
                     "trade_construction_quality_score": p.get("trade_construction_quality_score"),
                     "confirmation_family": p.get("confirmation_family"),
                     "h4_regime": fr["h4_regime"].iloc[i], "h4_permission": fr["h4_permission"].iloc[i],
                     "h1_volatility_regime": _g(p, "volatility", "h1_regime"), "h4_volatility_regime": _g(p, "volatility", "h4_regime"),
                     "h1_chop_score": float(fr["h1_chop_score"].iloc[i]) if "h1_chop_score" in fr else np.nan,
                     "h1_atr_percentile": float(f1["atr_percentile"].iloc[i]), "h4_atr_percentile": h4_atr_pct,
                     "session_label": sc.get("session_label"), "day_of_week": sc.get("day_of_week"), "hour_utc": sc.get("hour_utc"),
                     "news_status": p.get("news_status"), "stop_distance_atr": p.get("stop_distance_atr"),
                     "stop_quality_score": p.get("stop_quality_score"), "stop_noise_risk_score": p.get("stop_noise_risk_score"),
                     "target_distance_pips": pt.get("distance_pips"), "target_type": pt.get("type"),
                     "target_reachability_score": p.get("target_reachability_score"),
                     "barrier_density_score": p.get("barrier_density_score"), "barrier_count": _g(pt, "path", "barrier_count"),
                     "strong_barrier_count": _g(pt, "path", "strong_barrier_count"),
                     "planned_gross_R": p.get("gross_R"), "planned_net_R": p.get("estimated_net_R")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def family_coverage(out, trades_df: pd.DataFrame) -> dict:
    def cnt(items, key):
        c = {f: 0 for f in FAMILIES}
        for x in items:
            f = key(x)
            c[f] = c.get(f, 0) + 1
        return c

    created = cnt(out.h1.setups, lambda s: s.family)
    qualified = cnt(out.h1.qualified_setups, lambda s: s.family if hasattr(s, "family") else s.get("setup_family"))
    cands = cnt(out.entry.candidates, lambda c: c.setup_family)
    accepted = cnt([c for c in out.entry.candidates if c.accepted], lambda c: c.setup_family)
    props = cnt(out.trade.proposals, lambda p: p.get("setup_family"))
    traded = trades_df["setup_family"].value_counts().to_dict() if len(trades_df) else {}
    total_q = max(sum(qualified.values()), 1)
    res = {}
    for f in sorted(set(created) | set(FAMILIES)):
        q = qualified.get(f, 0)
        status = ("NEVER_OBSERVED" if created.get(f, 0) == 0 else "NEVER_QUALIFIED" if q == 0 else
                  "EXTREMELY_RARE" if q < 5 or q / total_q < 0.01 else "RARE" if q / total_q < 0.05 else "OBSERVED")
        res[f] = {"setups_created": created.get(f, 0), "qualified": q, "entry_candidates": cands.get(f, 0),
                  "accepted_entries": accepted.get(f, 0), "proposals": props.get(f, 0), "trades": int(traded.get(f, 0)),
                  "coverage_status": status}
    return res


def _spearman(x, y) -> float | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return None
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def score_calibration(df: pd.DataFrame, score_cols=SCORE_COLUMNS, rcol: str = "net_R", seed: int = 0) -> dict:
    """Do higher scores correspond to better outcomes?  Reported whatever the answer."""
    out = {}
    for col in score_cols:
        if col not in df.columns or df[col].notna().sum() == 0:
            out[col] = {"status": "NOT_AVAILABLE"}
            continue
        d = df[[col, rcol, "mae_R", "mfe_R"]].dropna(subset=[col, rcol])
        n = len(d)
        nb = 10 if n >= 100 else 5 if n >= 50 else 3 if n >= 15 else 0
        if nb == 0:
            out[col] = {"status": "NOT_MEANINGFUL_SAMPLE_TOO_SMALL", "n": n, "sample": sample_warning(n)}
            continue
        d = d.assign(_bin=pd.qcut(d[col].rank(method="first"), nb, labels=False))
        bins = []
        for b, g in d.groupby("_bin"):
            r = g[rcol].to_numpy(float)
            bins.append({"bin": int(b), "score_min": float(g[col].min()), "score_max": float(g[col].max()), "n": int(len(g)),
                         "win_rate": float((r > BE_EPS_R).mean()), "expectancy_R": float(r.mean()),
                         "mean_MAE_R": float(g["mae_R"].mean()), "mean_MFE_R": float(g["mfe_R"].mean())})
        exp = [b["expectancy_R"] for b in bins]
        diffs = np.diff(exp)
        rho = _spearman(d[col], d[rcol])
        lo = hi = None
        if n >= 15:
            xy = d[[col, rcol]].to_numpy(float)
            rng = np.random.default_rng(seed)
            vals = [v for v in (_spearman(*xy[rng.integers(0, n, n)].T) for _ in range(1000)) if v is not None]
            if vals:
                lo, hi = float(np.quantile(vals, .025)), float(np.quantile(vals, .975))
        if rho is None:
            verdict = "UNCLEAR"
        elif lo is not None and lo <= 0 <= hi:
            verdict = "STATISTICALLY_UNCLEAR"
        elif rho < 0:
            verdict = "NEGATIVE_RELATIONSHIP"
        elif (diffs >= 0).all():
            verdict = "MONOTONIC_INCREASING"
        else:
            verdict = "POSITIVE_BUT_NON_MONOTONIC"
        out[col] = {"n": n, "bins": bins, "spearman_score_vs_R": rho, "spearman_bootstrap_95": [lo, hi],
                    "monotonic_bins": bool((diffs >= 0).all()), "verdict": verdict,
                    "calibration_warning": None if verdict == "MONOTONIC_INCREASING" else
                    "higher scores do not reliably correspond to better outcomes - the scoring may not be calibrated",
                    "sample": sample_warning(n)}
    return out


def attribution(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    """Pre-defined segmentations only (no outcome-driven slicing)."""
    if not len(df):
        return {}
    d = df.copy()
    d["h1_atr_pct_bucket"] = pd.cut(d["h1_atr_percentile"], [-1, 25, 75, 101], labels=["LOW", "NORMAL", "HIGH"]).astype(str)
    d["h4_atr_pct_bucket"] = pd.cut(d["h4_atr_percentile"], [-1, 25, 75, 101], labels=["LOW", "NORMAL", "HIGH"]).astype(str)
    d["chop_bucket"] = pd.cut(d["h1_chop_score"], [-1, 40, 60, 101], labels=["LOW_CHOP", "MEDIUM_CHOP", "HIGH_CHOP"]).astype(str)
    d["hour_bucket"] = pd.cut(pd.to_numeric(d["hour_utc"], errors="coerce"), [-1, 6, 12, 16, 21, 24],
                              labels=["00-06", "07-12", "13-16", "17-21", "22-23"]).astype(str)
    res = {k: breakdown(d, col, rcol) for k, col in (
        ("direction", "direction"), ("setup_family", "setup_family"), ("h4_regime", "h4_regime"),
        ("h1_volatility_regime", "h1_volatility_regime"), ("h4_volatility_regime", "h4_volatility_regime"),
        ("h1_atr_percentile", "h1_atr_pct_bucket"), ("h4_atr_percentile", "h4_atr_pct_bucket"), ("chop", "chop_bucket"),
        ("session", "session_label"), ("day_of_week", "day_of_week"), ("hour_utc_bucket", "hour_bucket"),
        ("exit_reason", "exit_reason"))}
    res["note"] = "descriptive; no family, session, day or regime is removed on this evidence"
    return res


def long_short(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    if not len(df):
        return {}
    res = {}
    total_dd = 0.0
    for side, g in df.groupby("direction"):
        r = g.sort_values("entry_time")[rcol].to_numpy(float)
        eq = np.concatenate([[0], np.cumsum(r)])
        dd = float((np.maximum.accumulate(eq) - eq).max())
        total_dd += dd
        res[side] = {"trades": int(len(g)), "expectancy_R": float(r.mean()), "win_rate": float((r > BE_EPS_R).mean()),
                     "profit_factor": profit_factor(r), "r_quantiles": [float(np.quantile(r, q)) for q in (.1, .25, .5, .75, .9)],
                     "standalone_max_drawdown_R": dd, "families": g["setup_family"].value_counts().to_dict(),
                     "sample": sample_warning(len(g))["level"]}
    loss = df[df[rcol] < 0].groupby("direction")[rcol].sum().to_dict()
    tl = sum(loss.values())
    res["loss_contribution_share"] = {k: float(v / tl) for k, v in loss.items()} if tl else {}
    return res


def mae_mfe(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    """Distributions and HYPOTHESES only - no parameter is changed on this evidence."""
    if not len(df):
        return {}
    q = lambda s: {k: float(np.quantile(s, v)) for k, v in (("p25", .25), ("p50", .5), ("p75", .75), ("p90", .9))}  # noqa
    losers, winners = df[df[rcol] < -BE_EPS_R], df[df[rcol] > BE_EPS_R]
    res = {"mae_R": q(df["mae_R"]), "mfe_R": q(df["mfe_R"]), "mae_pips": q(df["mae_pips"]), "mfe_pips": q(df["mfe_pips"]),
           "losers_reaching_1R_favourable_first": float((losers["mfe_R"] >= 1.0).mean()) if len(losers) else None,
           "winners_with_mae_over_0_8R": float((winners["mae_R"] >= 0.8).mean()) if len(winners) else None,
           "capture_ratio": q(df["capture_ratio"].dropna()) if df["capture_ratio"].notna().any() else None,
           "by_family": {f: {"mae_R_median": float(g["mae_R"].median()), "mfe_R_median": float(g["mfe_R"].median()), "n": int(len(g))}
                         for f, g in df.groupby("setup_family")}}
    hyp = []
    if res["losers_reaching_1R_favourable_first"] and res["losers_reaching_1R_favourable_first"] > 0.3:
        hyp.append("many losing trades first moved >= 1R in favour (exit management hypothesis - requires untouched data)")
    if res["winners_with_mae_over_0_8R"] and res["winners_with_mae_over_0_8R"] > 0.3:
        hyp.append("many winners came within 0.2R of the stop (stop placement may be tight - hypothesis only)")
    res["hypotheses"] = hyp
    res["note"] = "bar-granularity excursions; within-bar ordering is unknown"
    return res


def stop_target_analysis(df: pd.DataFrame, rcol: str = "net_R") -> dict:
    if not len(df):
        return {}
    d = df.copy()
    cuts = {"stop_distance_atr": [0, 1, 2, 3, 100], "stop_quality_score": [-1, 40, 60, 80, 101],
            "stop_noise_risk_score": [-1, 30, 50, 70, 101], "target_reachability_score": [-1, 40, 60, 80, 101],
            "barrier_density_score": [-1, 20, 40, 60, 101], "planned_gross_R": [0, 2, 3, 4, 1000],
            "planned_net_R": [0, 2, 3, 4, 1000], "target_distance_pips": [0, 50, 100, 200, 10000]}
    res = {}
    for col, edges in cuts.items():
        if col in d and d[col].notna().any():
            b = pd.cut(pd.to_numeric(d[col], errors="coerce"), edges).astype(str)
            res[col] = breakdown(d.assign(_b=b), "_b", rcol)
    res["target_type"] = breakdown(d, "target_type", rcol)
    res["target_hit_rate"] = float((d["exit_reason"] == "TARGET").mean())
    res["note"] = "stops and targets are never moved retrospectively"
    return res


# ---------------------------------------------------------------------------
def _forward_move(f1: pd.DataFrame, idx: int, d: int, bars: int = 24) -> dict | None:
    if idx is None or idx + 1 >= len(f1):
        return None
    seg = f1.iloc[idx + 1: idx + 1 + bars]
    atr = float(f1["atr"].iloc[idx]) or np.nan
    o = float(f1["open"].iloc[idx + 1])
    fav = (seg["high"].max() - o) if d > 0 else (o - seg["low"].min())
    adv = (o - seg["low"].min()) if d > 0 else (seg["high"].max() - o)
    return {"fwd_return_atr": float(d * (seg["close"].iloc[-1] - o) / atr), "mfe_atr": float(fav / atr), "mae_atr": float(adv / atr)}


def counterfactuals(out, scenario, price_type: str = "BID") -> dict:
    """Rejected opportunities (entry and trade stages).  POST-DECISION DIAGNOSTIC only."""
    from .simulator import simulate_trade

    f1 = out.h1.features
    by_cat: dict = {}
    for c in out.entry.counterfactual:
        d = 1 if c["side"] == "long" else -1
        fm = _forward_move(f1, c.get("ended_index"), d)
        by_cat.setdefault(("ENTRY", c["category"]), []).append({"fwd": fm, "R": None})
    for r in out.trade.rejected:
        rec = r.get("research_record") or {}
        idx = out.index_of(pd.Timestamp(r["at"])) if r.get("at") else None
        d = 1 if r.get("direction") == "LONG" else -1
        R = None
        if rec.get("proposed_stop_price") and (rec.get("primary_target") or {}).get("price") and rec.get("risk_unit") \
                and idx is not None:
            p = dict(rec, trade_proposal_id=r["trade_proposal_id"], direction=r["direction"],
                     timestamp=f1["timestamp"].iloc[idx].isoformat())
            try:
                R = simulate_trade(f1, idx, p, scenario, price_type).net_R
            except Exception:  # noqa: BLE001 - incomplete research geometry
                R = None
        by_cat.setdefault(("TRADE", r.get("category")), []).append({"fwd": _forward_move(f1, idx, d), "R": R})
    res = {}
    for (stage, cat), items in sorted(by_cat.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        fw = [x["fwd"]["fwd_return_atr"] for x in items if x["fwd"]]
        rs = [x["R"] for x in items if x["R"] is not None]
        entry = {"stage": stage, "blocked": len(items), "fwd_return_atr_mean": float(np.mean(fw)) if fw else None,
                 "fwd_favourable_share": float(np.mean(np.asarray(fw) > 0)) if fw else None,
                 "simulated_R_n": len(rs)}
        if rs:
            a = np.asarray(rs)
            entry.update({"counterfactual_mean_R": float(a.mean()), "estimated_avoided_losses_R": float(-a[a < 0].sum()),
                          "estimated_missed_winners_R": float(a[a > 0].sum()),
                          "net_descriptive_contribution_R": float(-a.sum()),
                          "reading": "positive contribution = the filter avoided more loss than it missed"})
        res[f"{stage}:{cat}"] = entry
    return {"label": POST_DECISION, "filters": res,
            "note": "forward moves are measured over the next 24 H1 bars in ATR; simulated R exists only where the "
                    "rejected construction carried full geometry; nothing here is added to strategy results"}


# ---------------------------------------------------------------------------
def structural_limitations(out, trades_df: pd.DataFrame, rcol: str = "net_R") -> dict:
    f1, fr, st = out.h1.features, out.h1.frame, out.h1.structure
    n = len(f1)
    res = {}
    # 90 one-active-setup-per-direction capacity
    spans = {"long": [], "short": []}
    for s in out.h1.setups:
        if s.qualified_index is not None:
            spans[s.side].append((s.qualified_index, s.end_index if s.end_index is not None else n))
    overl = {}
    for side, iv in spans.items():
        iv.sort()
        overl[side] = sum(1 for k in range(1, len(iv)) if iv[k][0] <= max(e for _, e in iv[:k]))
    blocked_by_limit = sum("active" in str(c.get("reason", "")).lower() for c in out.entry.counterfactual)
    res["one_active_setup_limit"] = {"qualified_setups_overlapping_another_same_side": overl,
                                     "entry_candidates_blocked_by_active_limit": blocked_by_limit,
                                     "note": "limitation retained; overlap = potential opportunity cost only"}
    # 91 swing confirmation lag
    lags = np.array([sw.confirmation_lag_bars for sw in st.swings], float)
    dist = []
    for sw in st.swings:
        if 0 <= sw.confirm_index < n and sw.atr_at_confirmation:
            dist.append(abs(float(f1["close"].iloc[sw.confirm_index]) - sw.price) / sw.atr_at_confirmation)
    chase = sum(c.get("category") == "CHASE" for c in out.entry.counterfactual)
    res["h1_swing_lag"] = {"swings": int(len(lags)), "lag_bars_median": float(np.median(lags)) if len(lags) else None,
                           "lag_bars_p90": float(np.quantile(lags, .9)) if len(lags) else None,
                           "price_distance_at_confirmation_atr_median": float(np.median(dist)) if dist else None,
                           "price_distance_at_confirmation_atr_p90": float(np.quantile(dist, .9)) if dist else None,
                           "chase_rejections": chase, "note": "lag is retained; faster hindsight swings would be look-ahead"}
    # 92 limited swing horizon: does older structure disagree with the 12-swing view?
    H = int(getattr(st, "history_size", 12) or 12)
    diffs, sampled, saturated = 0, 0, 0
    for c in range(0, n, 24):
        alive = [s for s in st.swings if s.alive_at(c)]
        if len(alive) >= H:
            saturated += 1
        if len(alive) < H + 4:
            continue
        sampled += 1

        def lab(sw):
            up = sum(x.label in ("HH", "HL") for x in sw)
            dn = sum(x.label in ("LH", "LL") for x in sw)
            return "UP" if up > 1.5 * dn else "DOWN" if dn > 1.5 * up else "MIXED"

        if lab(alive[-H:]) != lab(alive[-2 * H:]):
            diffs += 1
    res["swing_horizon"] = {"history_size": H, "bars_sampled": sampled, "memory_saturated_share": saturated / max(1, len(range(0, n, 24))),
                            "older_structure_disagrees_share": diffs / sampled if sampled else None,
                            "note": "horizon unchanged during validation; disagreement = context that the current view discards"}
    # 93 multiple coexisting breaks
    if {"breaks_up_window", "breaks_down_window"} <= set(f1.columns):
        bu, bd = f1["breaks_up_window"].fillna(0).to_numpy(), f1["breaks_down_window"].fillna(0).to_numpy()
        res["multiple_breaks"] = {"bars_with_2plus_breaks_share": float(((bu + bd) >= 2).mean()),
                                  "bars_with_breaks_both_directions_share": float(((bu > 0) & (bd > 0)).mean()),
                                  "note": "only the most recent break is carried as primary context"}
    # 94 barriers: nearest-only vs stacked
    if len(trades_df):
        d = trades_df.assign(_stack=np.where(trades_df["barrier_count"].fillna(0) >= 2, "STACKED_2PLUS",
                                             np.where(trades_df["barrier_count"].fillna(0) == 1, "NEAREST_ONLY", "CLEAR")))
        res["barriers"] = {"trades_by_barrier_context": breakdown(d, "_stack", rcol),
                           "note": "historical rules unchanged; compares outcomes by stacked-barrier context"}
    # 95 weekend context-age blocking
    status = fr["h4_context_status"].astype(str)
    ts = pd.to_datetime(fr["timestamp"], utc=True)
    after_weekend = (ts.dt.dayofweek == 0) & (ts.dt.hour < 12) | ((ts.dt.dayofweek == 6))
    stale = status != "OK"
    blocked = stale & after_weekend
    atr = f1["atr"].to_numpy(float)
    moves = np.abs(np.diff(f1["close"].to_numpy(float), prepend=np.nan)) / atr
    res["weekend_context_age"] = {"bars_context_not_ok": int(stale.sum()), "after_weekend_bars_blocked": int(blocked.sum()),
                                  "blocked_bar_abs_move_atr_mean": float(np.nanmean(moves[blocked.to_numpy()])) if blocked.any()
                                  else None, "note": "policy unchanged; move size is the exposure avoided / opportunity missed"}
    # 96 news unknown
    props = out.trade.proposals
    unk = sum((p.get("news_status") in (None, "UNKNOWN")) for p in props)
    res["news"] = {"status": "NEWS_DATA_UNAVAILABLE", "proposals_with_news_unknown": unk, "proposals": len(props),
                   "trades_with_news_unknown": int((trades_df["news_status"].isin([None, "UNKNOWN"])).sum()) if len(trades_df) else 0,
                   "disclosure": "no historical economic-calendar data: the strategy has no event awareness in these results"}
    return res
