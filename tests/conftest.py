from __future__ import annotations

import pandas as pd
import pytest

from gbpjpy_engine import H4Config, H4MarketIntelligenceEngine
from gbpjpy_engine.synthetic import SCENARIOS, generate_scenario, random_walk_bars


@pytest.fixture(scope="session")
def engine() -> H4MarketIntelligenceEngine:
    return H4MarketIntelligenceEngine(H4Config())


@pytest.fixture(scope="session")
def rw_bars() -> pd.DataFrame:
    return random_walk_bars(n=900, seed=11)


@pytest.fixture(scope="session")
def rw_result(engine, rw_bars):
    return engine.run(rw_bars)


@pytest.fixture(scope="session")
def scenario_results(engine):
    out = {}
    for name in SCENARIOS:
        sc = generate_scenario(name)
        out[name] = (sc, engine.run(sc.bars))
    return out


# ---------------------------------------------------------------------------
# Phase 1C (H1) fixtures - additive
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def h1_engine():
    from gbpjpy_engine.h1 import H1Config, H1SetupEngine

    return H1SetupEngine(H1Config())


@pytest.fixture(scope="session")
def h1_results(engine, h1_engine):
    from gbpjpy_engine.synthetic_h1 import H1_SCENARIOS, generate_h1_scenario

    out = {}
    for name in H1_SCENARIOS:
        sc = generate_h1_scenario(name)
        h4 = engine.run(sc.h4)
        out[name] = (sc, h4, h1_engine.run(sc.h1, h4))
    return out


# ---------------------------------------------------------------------------
# Phase 1D (entry) fixtures - additive
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def entry_engine():
    from gbpjpy_engine.entry import EntryConfig, EntryIntelligenceEngine

    return EntryIntelligenceEngine(EntryConfig())


@pytest.fixture(scope="session")
def entry_results(engine, h1_engine, entry_engine):
    """{scenario: (scenario, H4 result, H1 setup result, entry result)} at realistic GBPJPY H1 volatility."""
    from gbpjpy_engine.synthetic_entry import ENTRY_SCENARIOS, generate_entry_scenario

    out = {}
    for name in ENTRY_SCENARIOS:
        sc = generate_entry_scenario(name)
        h4 = engine.run(sc.h4)
        h1 = h1_engine.run(sc.h1, h4)
        out[name] = (sc, h4, h1, entry_engine.run(h1, h4))
    return out
