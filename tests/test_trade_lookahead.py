"""Phase 1E look-ahead and determinism: future candles cannot change historical stop selection, target selection,
R calculation or the proposal decision unless replay has actually advanced to them."""

from __future__ import annotations

import json

import pytest

from gbpjpy_engine.data.resample import resample_complete
from gbpjpy_engine.entry import EntryIntelligenceEngine
from gbpjpy_engine.snapshot import to_jsonable
from gbpjpy_engine.trade import TradeConstructionEngine


def js(v):
    return json.dumps(to_jsonable(v), sort_keys=True, default=str)


def pipeline(engine, h1_engine, h1_bars):
    h1_bars = h1_bars.reset_index(drop=True)
    h4r = engine.run(resample_complete(h1_bars, 60, 240)[0])
    h1r = h1_engine.run(h1_bars, h4r)
    er = EntryIntelligenceEngine().run(h1r, h4r)
    return TradeConstructionEngine().run(er, h1r, h4r)


def construction_at(rec):
    """Everything decided at the construction instant (later lapse transitions excluded)."""
    upto = next(k for k, t in enumerate(rec.transitions) if t[3] in ("PROPOSED", "REJECTED", "INVALIDATED", "EXPIRED"))
    return rec.decision, rec.category, js(rec.proposal), rec.transitions[: upto + 1]


@pytest.fixture(scope="module")
def up(trade_results):
    return trade_results["entry_uptrend"]


def test_truncated_replay_builds_identical_proposals(engine, h1_engine, up):
    sc, _, _, _, full = up
    rec = full.constructions[0]
    part = pipeline(engine, h1_engine, sc.h1.iloc[: rec.index + 1])
    r = part.construction(rec.trade_proposal_id)
    assert construction_at(r) == construction_at(rec)
    assert {x.trade_proposal_id for x in part.constructions} == \
        {x.trade_proposal_id for x in full.constructions if x.index <= rec.index}
    early = pipeline(engine, h1_engine, sc.h1.iloc[: rec.index])  # replay has not reached the decision yet
    assert rec.trade_proposal_id not in {x.trade_proposal_id for x in early.constructions}


def test_future_candles_cannot_change_stop_targets_r_or_decision(engine, h1_engine, up):
    sc, _, _, _, full = up
    rec = full.constructions[0]
    j = rec.index
    alt = sc.h1.copy()
    alt.loc[j, "high"] = alt.loc[j, "high"] + 3.0  # the decision bar's range/close and every later bar change
    alt.loc[j, "low"] = alt.loc[j, "low"] - 3.0
    alt.loc[j, "close"] = alt.loc[j, "low"] + 0.2
    later = alt.index > j
    for col in ("open", "high", "low", "close"):
        alt.loc[later, col] = alt.loc[later, col] + 5.0  # a huge move that would have "hit" any target
    res = pipeline(engine, h1_engine, alt)
    r = res.construction(rec.trade_proposal_id)
    assert construction_at(r) == construction_at(rec)
    p, q = r.proposal, rec.proposal
    for k in ("proposed_stop_price", "raw_invalidation_price", "gross_R", "estimated_net_R", "primary_target", "candidate_targets"):
        assert js(p[k]) == js(q[k]), k


def test_deterministic_replay(engine, h1_engine, up):
    sc, h4, h1, er, full = up
    again = TradeConstructionEngine().run(er, h1, h4)
    assert [js(c.to_dict()) for c in again.constructions] == [js(c.to_dict()) for c in full.constructions]
    assert js(again.rejected) == js(full.rejected)
    fresh = pipeline(engine, h1_engine, sc.h1)
    assert [c.trade_proposal_id for c in fresh.constructions] == [c.trade_proposal_id for c in full.constructions]
    assert js(fresh.proposals) == js(full.proposals)
