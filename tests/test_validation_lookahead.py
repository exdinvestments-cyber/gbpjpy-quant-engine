"""Phase 1H critical look-ahead test: randomly mutate FUTURE GBPJPY data; every decision before the mutation
point (H4 state, H1 setups, entries, stops, targets, proposals, risk, execution) must be identical.
Synthetic fixture - tests causality of the machinery only."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.validation import account_simulation, run_pipeline, standard_scenarios, strategy_trades

CUT_INDEX = 1750


def _mutate(h1: pd.DataFrame, cut: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    m = h1.copy()
    fut = m["timestamp"] >= cut
    n = int(fut.sum())
    shift = np.cumsum(rng.normal(0, 0.3, n))
    for c in ("open", "high", "low", "close"):
        m.loc[fut, c] = m.loc[fut, c].to_numpy() + shift
    m.loc[fut, "high"] = m.loc[fut, ["open", "high", "close"]].max(axis=1) + rng.uniform(0, 0.3, n)
    m.loc[fut, "low"] = m.loc[fut, ["open", "low", "close"]].min(axis=1) - rng.uniform(0, 0.3, n)
    m.loc[fut, "spread"] = rng.uniform(1, 9, n)
    return m


def _txt(df):
    return df.reset_index(drop=True).astype(str).replace({"NaT": "nan", "None": "nan", "<NA>": "nan"})


def _before(out, cut):
    f4 = out.h4.features
    h4 = f4[pd.to_datetime(f4["available_at"], utc=True) <= cut].drop(columns=["reason_codes"], errors="ignore")
    fr = out.h1.frame
    h1 = fr[pd.to_datetime(fr["available_at"], utc=True) <= cut]
    setups = {s.setup_id: [t for t in s.transitions if pd.Timestamp(t[1]) < cut] for s in out.h1.setups
              if pd.Timestamp(s.created_at) < cut}
    cands = {c.entry_candidate_id: [t for t in c.transitions if pd.Timestamp(t[2]) < cut] for c in out.entry.candidates
             if pd.Timestamp(c.qualified_at) < cut}
    props = [p for p in out.trade.proposals if pd.Timestamp(p["timestamp"]) < cut]
    return h4, h1, setups, cands, props


@pytest.mark.parametrize("seed", [1, 2])
def test_critical_future_mutation_leaves_every_earlier_decision_identical(seed):
    from gbpjpy_engine.synthetic_entry import generate_entry_scenario

    h1 = generate_entry_scenario("entry_uptrend").h1
    cut = pd.Timestamp(h1["timestamp"].iloc[CUT_INDEX])
    a = run_pipeline(h1)
    b = run_pipeline(_mutate(h1, cut, seed))
    assert not np.allclose(a.h1_bars["close"].to_numpy(), b.h1_bars["close"].to_numpy())  # the future really changed
    ha, fa, sa, ca, pa = _before(a, cut)
    hb, fb, sb, cb, pb = _before(b, cut)
    # values compared as text: a column that is all-NaT before the cut may get a different inferred dtype
    pd.testing.assert_frame_equal(_txt(ha), _txt(hb))  # H4 state
    pd.testing.assert_frame_equal(_txt(fa), _txt(fb))  # H1 setup intelligence
    assert sa == sb and ca == cb  # setups and entry decisions
    assert json.dumps(pa, sort_keys=True, default=str) == json.dumps(pb, sort_keys=True, default=str)  # stops/targets
    assert pa, "the fixture must contain at least one decision before the mutation point"
    sc = standard_scenarios("PER_BAR")["BASELINE"]
    la, lb = strategy_trades(a, sc, "BID"), strategy_trades(b, sc, "BID")
    early = [t.trade_proposal_id for t in la.trades if pd.Timestamp(t.decision_time) < cut]
    assert early == [t.trade_proposal_id for t in lb.trades if pd.Timestamp(t.decision_time) < cut]
    closed_early = [t.to_dict() for t in la.trades if pd.Timestamp(t.exit_time) + pd.Timedelta(hours=1) <= cut]
    assert closed_early == [t.to_dict() for t in lb.trades if pd.Timestamp(t.exit_time) + pd.Timedelta(hours=1) <= cut]
    ra, rb = account_simulation(a, la), account_simulation(b, lb)
    for pid in early:  # risk sizing and execution decisions taken before the cut
        for k in ("risk_decision", "execution_decision", "filled_volume", "protection"):
            assert ra.decisions[pid].get(k) == rb.decisions[pid].get(k)
