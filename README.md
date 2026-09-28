# gbpjpy-quant-engine

Research-grade algorithmic analysis system dedicated to **GBPJPY**.

**Current phase: 1D — Entry Intelligence Engine**, built on the Phase 1C H1 Setup Intelligence Engine, the Phase 1B
H4 Context & Directional Permission Engine and the Phase 1A/1A.1 H4 Market Intelligence Engine. Phase 1D decides
whether a QUALIFIED H1 setup has developed into a precise, timely and executable ENTRY CANDIDATE (confirmation,
freshness, chase/extension, bid/ask-aware executable reference, spread, gaps, room, abnormal conditions). An entry
candidate is **not an order**: no position size, risk amount, stop loss, take profit, order or broker exists.

Phase 1B (H4): It describes the H4 environment (structure hierarchy, legs, displacement, breakout lifecycle,
liquidity, zones, location, maturity, compression, room to move) and produces a directional permission —
`ALLOW_LONG`, `ALLOW_SHORT`, `ALLOW_BOTH` or `BLOCK_ALL` — stating what a future H1 engine may *search for*, with
machine-readable reasons for every decision. Permission is context, not a trade signal.

> This repository contains **no** order placement, broker connectivity, position sizing, stop-loss / take-profit
> logic or optimisation. Nothing here has been validated as profitable, and no performance claims are made.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # full test suite incl. anti-look-ahead tests

# describe every closed bar of a CSV (timestamp = H4 bar OPEN time)
python -m gbpjpy_engine run --csv gbpjpy_h4.csv --tz UTC --out output/features.parquet --log output/eval.jsonl

# explain one bar in plain language
python -m gbpjpy_engine inspect --csv gbpjpy_h4.csv --tz UTC --timestamp 2024-03-01T08:00:00Z
python -m gbpjpy_engine inspect --csv gbpjpy_h4.csv --tz UTC --why      # why is H1 allowed/blocked?

# list every parameter and its purpose
python -m gbpjpy_engine config

# Phase 1C: H1 setup intelligence (H4 aggregated from complete H1 buckets unless --h4-csv is given)
python -m gbpjpy_engine h1 --h1-csv gbpjpy_h1.csv --tz UTC --out output/h1_setups.parquet

# Phase 1D: entry intelligence (declare the spread units in config/entry_default.yaml: execution.spread_unit)
python -m gbpjpy_engine entry --h1-csv gbpjpy_h1.csv --tz UTC --out output/entry.parquet
```

```python
from gbpjpy_engine import H4MarketIntelligenceEngine, load_config
from gbpjpy_engine.data import load_csv
from gbpjpy_engine.research import inspect

bars = load_csv("gbpjpy_h4.csv", assume_timezone="Europe/Athens")   # broker server time -> UTC
result = H4MarketIntelligenceEngine(load_config("config/h4_default.yaml")).run(bars)
snap = result.latest()            # H4Snapshot for the most recent CLOSED bar
print(snap.regime, snap.h4_bias, snap.bias_confidence, snap.reason_codes)
print(inspect(result))            # human-readable explanation
print(snap.directional_permission, snap.permission_confidence, snap.permission_reason_codes)
print(result.explain_permission())  # exact reasons behind the latest permission
```

CSV input needs `timestamp, open, high, low, close` and optionally `volume` and `spread`.
Naive timestamps must be given an explicit timezone (`--tz` / `assume_timezone`); the engine never guesses.
`volume` is broker **tick** volume (not exchange volume); volume and spread are carried through for later
execution/market-quality modules but are not used by any Phase 1A feature.

The core is platform agnostic (stdlib + numpy + pandas + PyYAML). Future MT4/MT5 adapters must convert broker
server time to canonical UTC bars before data reaches the engine; no adapter exists yet.

## Documentation

* [`docs/PHASE_1A_H4_MARKET_INTELLIGENCE.md`](docs/PHASE_1A_H4_MARKET_INTELLIGENCE.md) — features, methodology,
  look-ahead protections, swing confirmation, regime and bias (Phase 1A / 1A.1).
* [`docs/PHASE_1B_H4_CONTEXT_PERMISSION.md`](docs/PHASE_1B_H4_CONTEXT_PERMISSION.md) — context layer and
  directional permission (Phase 1B).
* [`docs/PHASE_1C_H1_SETUP_INTELLIGENCE.md`](docs/PHASE_1C_H1_SETUP_INTELLIGENCE.md) — H1 setup intelligence,
  H4/H1 point-in-time alignment, setup families and lifecycle (Phase 1C).
* [`docs/PHASE_1D_ENTRY_INTELLIGENCE.md`](docs/PHASE_1D_ENTRY_INTELLIGENCE.md) — entry confirmation, timing,
  bid/ask executable references, spread, historical execution realism (Phase 1D).
* [`docs/BROKER_NEUTRAL_INTERFACES.md`](docs/BROKER_NEUTRAL_INTERFACES.md) — future MT4/MT5 adapter contracts
  (specification only).

## Layout

```
src/gbpjpy_engine/   engine package (data, features, classification, context [Phase 1B], h1 [Phase 1C], entry [Phase 1D], engine, snapshot, logging, research, cli)
config/              documented baseline configuration (h4_default.yaml, h1_default.yaml, entry_default.yaml)
docs/                design and methodology documentation
tests/               unit, anti-look-ahead and synthetic-scenario tests
```
