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
