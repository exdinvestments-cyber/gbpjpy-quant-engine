"""Phase 1D critical look-ahead, no-hindsight and determinism tests.

Future H1 candles, future H4 information, future spread, future breakout outcomes, future news and future
execution outcomes must not change any past entry decision.  The decision at the first executable price after
bar ``c`` may use that one price and the spread known at that moment - never the rest of the next bar.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.data.resample import resample_complete
from gbpjpy_engine.entry import EntryIntelligenceEngine, ExecutableQuote, NewsEvent
from gbpjpy_engine.snapshot import to_jsonable
from test_h1_lookahead import mid_bucket, same_rows

HINDSIGHT = ("mfe", "mae", "max_favourable", "max_favorable", "max_adverse", "outcome", "pnl", "profit", "future")


def js(v):
    return json.dumps(to_jsonable(v), sort_keys=True, default=str)


def pipeline(engine, h1_engine, entry_engine, h1_bars, **kw):
    h1_bars = h1_bars.reset_index(drop=True)
    h4r = engine.run(resample_complete(h1_bars, 60, 240)[0])
    h1r = h1_engine.run(h1_bars, h4r)
    return h4r, h1r, entry_engine.run(h1r, h4r, **kw)


@pytest.fixture(scope="module")
def up(entry_results):
    return entry_results["entry_uptrend"]


@pytest.fixture(scope="module")
def acc(up):
    return next(c for c in up[3].candidates if c.accepted is not None)


def check_entry_prefix(full, part, cut):
    same_rows(full.frame, part.frame, cut)
    for i in (cut - 20, cut - 1, cut):
        assert js(part.details[i]) == js(full.details[i]), i
    fc = {c.entry_candidate_id: c for c in full.candidates}
    assert {c.entry_candidate_id for c in part.candidates} == \
        {c.entry_candidate_id for c in full.candidates if c.qualified_index <= cut}
    for c in part.candidates:
        f = fc[c.entry_candidate_id]
        assert [t for t in f.transitions if t[0] <= cut] == c.transitions
        assert [js(o) for o in f.opportunities if o["index"] <= cut] == [js(o) for o in c.opportunities]
        assert f.state_at(cut) == c.state
    assert part.accepted == [a for a in full.accepted if a["decision_bar_index"] <= cut]


def test_future_h1_candles_cannot_change_past_decisions(engine, h1_engine, entry_engine, up, acc):
    sc, _, _, full = up
    ci, j = acc.confirmation_index, acc.accepted_index
    for cut in (ci, j, mid_bucket(full, j + 6)):  # confirmation pending / decision bar / H4 candle still forming
        _, _, part = pipeline(engine, h1_engine, entry_engine, sc.h1.iloc[: cut + 1])
        check_entry_prefix(full, part, cut)
    _, _, pending = pipeline(engine, h1_engine, entry_engine, sc.h1.iloc[: ci + 1])
    c = next(x for x in pending.candidates if x.setup_id == acc.setup_id)
    assert c.state == "CONFIRMING" and c.awaiting_price and c.opportunities == []  # waits for a real next price


def test_decision_uses_only_the_next_open_not_the_rest_of_that_bar(engine, h1_engine, entry_engine, up, acc):
    sc, _, _, full = up
    j = acc.accepted_index
    alt = sc.h1.copy()
    alt.loc[j, "high"] = alt.loc[j, "high"] + 2.0  # the decision bar's range, close and spread change ...
    alt.loc[j, "low"] = alt.loc[j, "low"] - 2.0
    alt.loc[j, "close"] = alt.loc[j, "low"] + 0.1
    alt.loc[j, "spread"] = 40.0
    later = alt.index > j
    for col in ("open", "high", "low", "close"):
        alt.loc[later, col] = alt.loc[later, col] - 3.0  # ... and the future crashes
    _, _, res = pipeline(engine, h1_engine, entry_engine, alt)
    c = next(x for x in res.candidates if x.setup_id == acc.setup_id)
    assert js(c.accepted) == js(acc.accepted)  # identical decision: only the open of bar j was used
    assert js(c.opportunities[0]) == js(acc.opportunities[0])
    assert c.state_at(j - 1) == acc.state_at(j - 1)
    assert c.transitions[-1][0] >= j and c.state in ("EXPIRED", "INVALIDATED")  # only the ending changes


def test_live_quote_after_the_final_close_gives_the_same_decision(engine, h1_engine, entry_engine, up, acc):
    sc, _, _, _ = up
    ci = acc.confirmation_index

    class Live:
        def quote_after(self, index):
            if index != ci:
                return None
            return ExecutableQuote(sc.h1["timestamp"].iloc[ci + 1], float(sc.h1["open"].iloc[ci + 1]),
                                   float(sc.h1["spread"].iloc[ci]), "last_completed_bar_report", "next_bar_open")

    _, _, live = pipeline(engine, h1_engine, entry_engine, sc.h1.iloc[: ci + 1], quotes=Live())
    c = next(x for x in live.candidates if x.setup_id == acc.setup_id)
    assert js(c.accepted) == js(acc.accepted) and c.state == "ENTRY_CANDIDATE"


def test_future_spread_cannot_leak_backward(up, acc):
    _, h4, h1, full = up
    j = acc.accepted_index
    feats = h1.features.copy()
    feats.loc[feats.index >= j, "spread"] = 30.0  # spreads reported from the decision bar onward explode
    res = EntryIntelligenceEngine().run(dataclasses.replace(h1, features=feats), h4)
    c = next(x for x in res.candidates if x.setup_id == acc.setup_id)
    assert js(c.accepted) == js(acc.accepted)
    feats2 = h1.features.copy()
    feats2.loc[j - 1, "spread"] = 30.0  # the spread KNOWN at the decision (last completed bar) is what counts
    c2 = next(x for x in EntryIntelligenceEngine().run(dataclasses.replace(h1, features=feats2), h4).candidates
              if x.setup_id == acc.setup_id)
    assert c2.state == "REJECTED" and c2.end_category == "SPREAD"


def test_future_h4_information_cannot_leak(up):
    _, _, h1, er = up
    f = h1.frame
    for c in er.candidates:
        for o in c.opportunities:
            p = o["based_on_close_of"]
            assert o["index"] == p + 1
            k = f["h4_index"].iloc[p]
            if k >= 0:
                assert f["h4_available_at"].iloc[p] <= f["available_at"].iloc[p]
                assert pd.Timestamp(o["at"]) >= f["available_at"].iloc[p]


def test_future_breakout_outcomes_cannot_leak(up):
    """Family scores at bar c read breaks through state_at(c); rewriting a break's LATER lifecycle changes nothing."""
    _, h4, h1, er = up
    st = h1.structure
    brks = []
    for b in st.breaks:
        nb = dataclasses.replace(b, transitions=list(b.transitions))
        last = nb.transitions[-1]
        nb.transitions.append(dataclasses.replace(last, index=last.index + 500, from_state=last.to_state, to_state="FAILED",
                                                  reason="edited future failure"))
        nb.max_follow_through_atr = 99.0
        nb.structure_after = "edited"
        brks.append(nb)
    alt = EntryIntelligenceEngine().run(dataclasses.replace(h1, structure=dataclasses.replace(st, breaks=brks)), h4)
    lim = min(t.index for b in brks for t in b.transitions[-1:])  # earliest edited (future) transition
    same_rows(er.frame, alt.frame, lim - 1)


def test_future_news_cannot_leak_backward(up, acc):
    _, h4, h1, er = up
    t = pd.Timestamp(acc.accepted["timestamp"])

    class Calendar:
        def events(self, start, end, known_at):
            return [NewsEvent(t + pd.Timedelta(minutes=10), "GBP", "HIGH", "unscheduled at decision time",
                              known_since=t + pd.Timedelta(hours=2)),
                    NewsEvent(t + pd.Timedelta(days=3), "JPY", "HIGH", "far future event")]

    res = EntryIntelligenceEngine().run(h1, h4, news=Calendar())
    c = next(x for x in res.candidates if x.setup_id == acc.setup_id)
    assert c.accepted["news_status"] == "CLEAR"
    assert not {f.name for f in dataclasses.fields(NewsEvent)} & {"actual", "forecast", "outcome", "previous"}


def test_execution_outcomes_cannot_influence_qualification(up, entry_engine):
    _, h4, h1, _ = up
    before = (js(h1.frame.to_dict("records")), js([s.to_dict() for s in h1.setups]), js(h1.qualified_setups))
    entry_engine.run(h1, h4)
    EntryIntelligenceEngine().run(h1, h4, quotes=None, news=None)
    after = (js(h1.frame.to_dict("records")), js([s.to_dict() for s in h1.setups]), js(h1.qualified_setups))
    assert before == after  # Phase 1D never writes back into Phase 1C results


def test_no_hindsight_fields_exist(up):
    for _, _, _, er in [up]:
        blob = js([[c.to_dict() for c in er.candidates], er.accepted, er.counterfactual])
        keys = set()

        def walk(v):
            if isinstance(v, dict):
                for k, x in v.items():
                    keys.add(str(k).lower())
                    walk(x)
            elif isinstance(v, list):
                for x in v:
                    walk(x)

        walk(json.loads(blob))
        assert not [k for k in keys if any(h in k for h in HINDSIGHT)], keys


def test_deterministic_replay(engine, h1_engine, entry_engine, up):
    sc, h4, h1, full = up
    again = EntryIntelligenceEngine().run(h1, h4)
    same_rows(full.frame, again.frame, len(full.frame) - 1)
    assert [js(c.to_dict()) for c in again.candidates] == [js(c.to_dict()) for c in full.candidates]
    assert again.accepted == full.accepted and js(again.counterfactual) == js(full.counterfactual)
    _, _, fresh = pipeline(engine, h1_engine, entry_engine, sc.h1)
    assert [c.entry_candidate_id for c in fresh.candidates] == [c.entry_candidate_id for c in full.candidates]
    assert fresh.accepted == full.accepted


def test_entry_harness_detects_leakage(up):
    _, _, _, er = up
    full = er.frame[["long_conf_reacceleration_score"]].copy()
    full["future"] = full["long_conf_reacceleration_score"].shift(-1)
    part = full.iloc[:1500].copy()
    part["future"] = part["long_conf_reacceleration_score"].shift(-1)
    with pytest.raises(AssertionError, match="H1 look-ahead"):
        same_rows(full, part, 1499)
    assert np.isfinite(full["long_conf_reacceleration_score"]).all()
