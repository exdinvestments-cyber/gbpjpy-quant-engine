"""Robustness research (Phase 1H): sensitivity, weight perturbation, threshold
alternatives, ablation, a simpler baseline, null baselines and an overfitting
risk diagnostic.

This is NOT optimisation.  Only a small, pre-declared set of parameters is
perturbed around the frozen baseline (x0.8 .. x1.2) and nothing is ever
written back to a production configuration.  The purpose is to find cliffs and
fragility, not a better parameter.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace

import numpy as np
import pandas as pd

from .metrics import BE_EPS_R, summarise

KEY_PARAMETERS = (  # pre-declared before any real-data result exists
    ("h1", "setup", "qualify_score"),
    ("h4", "permission", "min_context_score"),
    ("entry", "scoring", "min_entry_quality"),
    ("entry", "lifecycle", "run_away_atr"),
    ("trade", "stop", "buffer_atr"),
    ("trade", "target", "min_reachability"),
)
DEFAULT_FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)
WEIGHT_GROUPS = {
    "h1_setup_scoring": ("h1", "scoring", ("w_h4_context", "w_h1_structure", "w_pullback", "w_location", "w_displacement",
                                           "w_momentum", "w_liquidity", "w_market_quality", "w_room", "w_conflict")),
    "entry_quality": ("entry", "scoring", ("w_confirmation", "w_timing", "w_freshness", "w_price_quality", "w_market_quality",
                                           "w_structural_integrity", "w_room", "w_execution_conditions", "w_conflict")),
    "trade_quality": ("trade", "scoring", ("w_entry", "w_stop", "w_target", "w_asymmetry", "w_path", "w_context")),
}
ABLATIONS = {  # one logical feature family removed at a time (its H1 score weight -> 0, the rest renormalised)
    "without_liquidity": ("h1", "scoring", "w_liquidity"),
    "without_location": ("h1", "scoring", "w_location"),
    "without_momentum": ("h1", "scoring", "w_momentum"),
    "without_chop_market_quality": ("h1", "scoring", "w_market_quality"),
    "without_displacement": ("h1", "scoring", "w_displacement"),
    "without_room": ("h1", "scoring", "w_room"),
}


def get_param(configs, path):
    c, sec, name = path
    return getattr(getattr(getattr(configs, c), sec), name)


def set_params(configs, changes: dict):
    """New StrategyConfigs with ``{(config, section, name): value}`` applied (the originals are untouched)."""
    by_cfg: dict = {}
    for (c, sec, name), v in changes.items():
        by_cfg.setdefault(c, {}).setdefault(sec, {})[name] = v
    new = {}
    for c, secs in by_cfg.items():
        cfg = getattr(configs, c)
        new[c] = replace(cfg, **{sec: replace(getattr(cfg, sec), **vals) for sec, vals in secs.items()})
        if hasattr(new[c], "validate"):
            new[c].validate()
    return replace(configs, **new)


def perturb(configs, path, factor: float):
    v = get_param(configs, path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"{path} is not numeric")
    nv = type(v)(round(v * factor)) if isinstance(v, int) else float(v * factor)
    return set_params(configs, {path: nv})


def reweight(configs, group: str, name: str, factor: float):
    c, sec, names = WEIGHT_GROUPS[group]
    ws = {n: float(get_param(configs, (c, sec, n))) for n in names}
    total = sum(ws.values())
    ws[name] *= factor
    s = sum(ws.values())
    return set_params(configs, {(c, sec, n): w * total / s for n, w in ws.items()})


def ablate(configs, key: str):
    c, sec, name = ABLATIONS[key]
    group = next(g for g, (cc, ss, ns) in WEIGHT_GROUPS.items() if cc == c and ss == sec and name in ns)
    return reweight(configs, group, name, 0.0)


def _classify(base: float | None, variants: dict) -> str:
    if base is None:
        return "NO_TRADES"
    if base <= 0:
        return "NO_BASELINE_EDGE"
    near = [variants[f] for f in (0.9, 1.1) if f in variants and variants[f] is not None]
    far = [variants[f] for f in (0.8, 1.2) if f in variants and variants[f] is not None]
    bad = lambda x: x <= 0 or x < 0.5 * base  # noqa: E731
    if any(bad(x) for x in near):
        return "CLIFF_EDGE"
    if any(bad(x) for x in far):
        return "FRAGILE"
    return "STABLE_REGION"


def sensitivity(run_fn, configs, params=KEY_PARAMETERS, factors=DEFAULT_FACTORS) -> dict:
    """``run_fn(configs) -> trades DataFrame``.  Returns a table and a robustness flag per parameter."""
    base_df = run_fn(configs)
    base = summarise(base_df) if len(base_df) else {"trades": 0, "expectancy_R": None}
    table = {}
    for path in params:
        rows, exp = [], {}
        for f in factors:
            try:
                df = base_df if f == 1.0 else run_fn(perturb(configs, path, f))
                s = summarise(df) if len(df) else {"trades": 0, "expectancy_R": None, "total_R": 0.0, "win_rate": None}
                rows.append({"factor": f, "value": get_param(perturb(configs, path, f), path), "trades": s["trades"],
                             "expectancy_R": s.get("expectancy_R"), "total_R": s.get("total_R"), "win_rate": s.get("win_rate")})
                exp[f] = s.get("expectancy_R")
            except (ValueError, TypeError) as exc:
                rows.append({"factor": f, "status": f"INVALID_CONFIGURATION: {exc}"})
        table[".".join(path)] = {"rows": rows, "flag": _classify(base.get("expectancy_R"), exp)}
    flags = [v["flag"] for v in table.values()]
    return {"baseline": {k: base.get(k) for k in ("trades", "expectancy_R", "total_R")}, "parameters": table,
            "fragile_share": float(np.mean([f in ("CLIFF_EDGE", "FRAGILE") for f in flags])) if flags else None,
            "note": "perturbation around the frozen baseline only; no variant is adopted"}


def weight_sensitivity(run_fn, configs, factor: float = 1.2, groups=WEIGHT_GROUPS) -> dict:
    base_df = run_fn(configs)
    base = summarise(base_df).get("expectancy_R") if len(base_df) else None
    out = {}
    for g, (_, _, names) in groups.items():
        for n in names:
            res = {}
            for f in (1 / factor, factor):
                df = run_fn(reweight(configs, g, n, f))
                res[round(f, 3)] = summarise(df).get("expectancy_R") if len(df) else None
            vals = [v for v in res.values() if v is not None]
            fragile = base is not None and base > 0 and any(v <= 0 or v < 0.5 * base for v in vals)
            out[f"{g}.{n}"] = {"expectancy_R": res, "fragile": bool(fragile)}
    return {"baseline_expectancy_R": base, "weights": out,
            "fragile_weights": [k for k, v in out.items() if v["fragile"]]}


def threshold_alternatives(run_fn, configs, alternatives: dict) -> dict:
    """``{path: (alt1, alt2)}`` - a handful of nearby values per major gate (never hundreds)."""
    if sum(len(v) for v in alternatives.values()) > 20:
        raise ValueError("threshold alternatives are limited to a handful of nearby values")
    out = {}
    for path, vals in alternatives.items():
        rows = []
        for v in vals:
            df = run_fn(set_params(configs, {path: v}))
            s = summarise(df) if len(df) else {"trades": 0}
            rows.append({"value": v, "trades": s["trades"], "expectancy_R": s.get("expectancy_R")})
        out[".".join(path)] = rows
    return out


def ablation(run_fn, configs, keys=tuple(ABLATIONS)) -> dict:
    base_df = run_fn(configs)
    base = summarise(base_df) if len(base_df) else {"trades": 0}
    out = {}
    for k in keys:
        df = run_fn(ablate(configs, k))
        s = summarise(df) if len(df) else {"trades": 0}
        out[k] = {"trades": s["trades"], "expectancy_R": s.get("expectancy_R"), "total_R": s.get("total_R"),
                  "delta_expectancy_R": (s.get("expectancy_R") - base["expectancy_R"])
                  if s.get("expectancy_R") is not None and base.get("expectancy_R") is not None else None}
    return {"baseline": {"trades": base["trades"], "expectancy_R": base.get("expectancy_R")}, "ablations": out,
            "note": "a family whose removal does not worsen results may not contribute information; the strategy is NOT "
                    "rewritten on this evidence"}


# ---------------------------------------------------------------------------
# Simpler and null baselines (same data, same simulator, same costs)
# ---------------------------------------------------------------------------
def simple_baseline_proposals(out, stop_atr: float = 1.5, target_r: float = 2.0) -> list[dict]:
    """Basic H4 trend (EMA mid vs slow at the last CLOSED H4 bar) + H1 pullback to the fast EMA and re-cross.

    Decision at the next H1 open; stop = 1.5 ATR, target = 2R; entry reference = open (+ previous bar spread for
    longs).  Deliberately naive - a yardstick, not a strategy candidate."""
    f1, fr, f4 = out.h1.features, out.h1.frame, out.h4.features
    c = f1["close"].to_numpy(float)
    ema = f1["ema_fast"].to_numpy(float)
    atr = f1["atr"].to_numpy(float)
    spread = f1["spread"].to_numpy(float) if "spread" in f1 else np.full(len(f1), np.nan)
    h4i = fr["h4_index"].to_numpy()
    e_mid, e_slow = f4["ema_mid"].to_numpy(float), f4["ema_slow"].to_numpy(float)
    props = []
    for i in range(2, len(f1) - 1):
        k = h4i[i]
        if pd.isna(k) or int(k) < 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        k = int(k)
        trend = 1 if e_mid[k] > e_slow[k] else -1 if e_mid[k] < e_slow[k] else 0
        if trend == 0:
            continue
        crossed = (trend > 0 and c[i - 1] < ema[i - 1] and c[i] > ema[i]) or (trend < 0 and c[i - 1] > ema[i - 1] and c[i] < ema[i])
        if not crossed:
            continue
        o = float(f1["open"].iloc[i + 1])
        sp = spread[i] if np.isfinite(spread[i]) else 3.0
        ref = o + (sp * 0.01 if trend > 0 else 0.0)
        stop = ref - trend * stop_atr * atr[i]
        target = ref + trend * target_r * stop_atr * atr[i]
        props.append({"trade_proposal_id": f"BASE-{i + 1}", "timestamp": f1["timestamp"].iloc[i + 1].isoformat(),
                      "direction": "LONG" if trend > 0 else "SHORT", "setup_family": "SIMPLE_BASELINE",
                      "executable_reference_price": ref, "proposed_stop_price": round(stop, 3),
                      "primary_target": {"price": round(target, 3)},
                      "risk_unit": {"one_R_pips": round(stop_atr * atr[i] / 0.01, 2)},
                      "volatility": {"h1_atr_pips": atr[i] / 0.01}})
    return props


def simulate_proposals(out, proposals: list, scenario, price_type="BID", ambiguity="CONSERVATIVE", warmup_index=0) -> pd.DataFrame:
    from .simulator import simulate_trade

    f1 = out.h1.features
    busy = {"LONG": -1, "SHORT": -1}
    rows = []
    for p in sorted(proposals, key=lambda x: x["timestamp"]):
        idx = out.index_of(pd.Timestamp(p["timestamp"]))
        if idx is None or idx < warmup_index or busy[p["direction"]] >= idx:
            continue
        t = simulate_trade(f1, idx, p, scenario, price_type, ambiguity_policy=ambiguity)
        busy[p["direction"]] = t.exit_index
        rows.append(t.to_dict())
    return pd.DataFrame(rows)


def null_baselines(out, strategy_df: pd.DataFrame, scenario, n_runs: int = 50, seed: int = 0, price_type="BID",
                   warmup_index: int = 0) -> dict:
    """(a) random direction at the strategy's own entry times and stop/target distances;
    (b) random entry timing inside bars where H4 permission allowed that direction, with stop/target distances drawn
    from the strategy's empirical ATR multiples.  Distributions of expectancy across seeded runs."""
    if not len(strategy_df):
        return {"status": "NO_STRATEGY_TRADES"}
    f1, fr = out.h1.features, out.h1.frame
    rng = np.random.default_rng(seed)
    atr = f1["atr"].to_numpy(float)
    stop_mult = (strategy_df["one_R_pips"] * 0.01 / atr[strategy_df["entry_index"].to_numpy() - 1]).replace([np.inf], np.nan).dropna()
    rr = ((strategy_df["target"] - strategy_df["planned_entry"]).abs() / (strategy_df["planned_entry"] - strategy_df["stop"]).abs())
    perm = fr["h4_permission"].astype(str).to_numpy()
    rd, rt = [], []
    n = len(strategy_df)
    for run in range(n_runs):
        props_a, props_b = [], []
        for k, t in enumerate(strategy_df.itertuples()):
            d = 1 if rng.random() < 0.5 else -1
            one = abs(t.planned_entry - t.stop)
            props_a.append(_mk(f"NA-{run}-{k}", f1, t.entry_index, d, one, abs(t.target - t.planned_entry)))
        allowed = [i for i in range(max(warmup_index, 1), len(f1) - 1) if perm[i - 1] in ("ALLOW_LONG", "ALLOW_SHORT", "ALLOW_BOTH")]
        if allowed:
            for k, i in enumerate(rng.choice(allowed, size=min(n, len(allowed)), replace=False)):
                p = perm[i - 1]
                d = 1 if p == "ALLOW_LONG" else -1 if p == "ALLOW_SHORT" else (1 if rng.random() < 0.5 else -1)
                m = float(rng.choice(stop_mult.to_numpy())) if len(stop_mult) else 1.5
                r_ = float(rng.choice(rr.to_numpy())) if len(rr) else 2.0
                one = m * atr[i - 1]
                if np.isfinite(one) and one > 0:
                    props_b.append(_mk(f"NB-{run}-{k}", f1, int(i), d, one, one * r_))
        da = simulate_proposals(out, props_a, scenario, price_type, warmup_index=warmup_index)
        db = simulate_proposals(out, props_b, scenario, price_type, warmup_index=warmup_index)
        rd.append(float(da["net_R"].mean()) if len(da) else np.nan)
        rt.append(float(db["net_R"].mean()) if len(db) else np.nan)
    strat = float(strategy_df["net_R"].mean())

    def dist(x):
        x = np.asarray([v for v in x if np.isfinite(v)])
        return {"median": float(np.median(x)), "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)),
                "strategy_percentile": float((x < strat).mean() * 100)} if len(x) else {}

    return {"strategy_expectancy_R": strat, "random_direction": dist(rd), "random_timing_constrained_context": dist(rt),
            "n_runs": n_runs, "seed": seed,
            "reading": "a strategy percentile near 50 means results do not obviously exceed trivial behaviour"}


def _mk(pid, f1, idx, d, one, reward):
    o = float(f1["open"].iloc[idx])
    return {"trade_proposal_id": pid, "timestamp": f1["timestamp"].iloc[idx].isoformat(), "direction": "LONG" if d > 0 else "SHORT",
            "setup_family": "NULL", "executable_reference_price": o, "proposed_stop_price": round(o - d * one, 3),
            "primary_target": {"price": round(o + d * reward, 3)}, "risk_unit": {"one_R_pips": round(one / 0.01, 2)},
            "volatility": {"h1_atr_pips": float(f1["atr"].iloc[max(idx - 1, 0)]) / 0.01}}


# ---------------------------------------------------------------------------
def overfitting_risk(n_tunable_parameters: int, n_experiments: int, n_trades: int, fragile_share: float | None,
                     in_sample_expectancy: float | None, out_of_sample_expectancy: float | None,
                     best5_share: float | None, top_family_share: float | None) -> dict:
    """Heuristic 0-100 diagnostic (NOT a probability).  Each factor is scored 0..1 and reported."""
    f = {"parameters": min(1.0, np.log10(max(n_tunable_parameters, 1)) / 3.0),
         "experiments": min(1.0, n_experiments / 50.0),
         "small_sample": 1.0 if n_trades < 30 else 0.6 if n_trades < 100 else 0.3 if n_trades < 300 else 0.1,
         "sensitivity": fragile_share if fragile_share is not None else 0.5,
         "is_oos_degradation": (0.5 if in_sample_expectancy is None or out_of_sample_expectancy is None else
                                1.0 if in_sample_expectancy > 0 and out_of_sample_expectancy <= 0 else
                                max(0.0, min(1.0, (in_sample_expectancy - out_of_sample_expectancy) / abs(in_sample_expectancy)))
                                if in_sample_expectancy else 0.5),
         "return_concentration": min(1.0, best5_share) if best5_share is not None else 0.5,
         "family_concentration": top_family_share if top_family_share is not None else 0.5}
    score = float(np.mean(list(f.values())) * 100)
    level = "HIGH" if score >= 60 else "MODERATE" if score >= 35 else "LOW"
    return {"score_0_100": round(score, 1), "level": level, "factors": {k: round(float(v), 3) for k, v in f.items()},
            "note": "heuristic research diagnostic - not a formal probability of overfitting",
            "warnings": [k for k, v in f.items() if v >= 0.6]}


def win_rate(df) -> float | None:
    return float((df["net_R"] > BE_EPS_R).mean()) if len(df) else None


def config_is_dataclass_tree(configs) -> bool:
    return all(is_dataclass(getattr(configs, f.name)) for f in fields(configs))
