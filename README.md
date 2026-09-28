# gbpjpy-quant-engine

Research-grade algorithmic analysis system dedicated to **GBPJPY**.

**Current phase: 1A.1 — H4 Market Intelligence Engine (hardened).** It describes the H4 environment (structure, trend,
volatility, momentum, chop, levels, location, regime) and produces a strategic LONG / SHORT / NEUTRAL context bias
with machine-readable reasons.

> This repository contains **no** order placement, broker connectivity, position sizing, H1 entry logic or
> optimisation. Nothing here has been validated as profitable, and no performance claims are made.

## Quick start

```bash
pip install -e ".[dev]"
pytest                                   # full test suite incl. anti-look-ahead tests

# describe every closed bar of a CSV (timestamp = H4 bar OPEN time)
python -m gbpjpy_engine run --csv gbpjpy_h4.csv --tz UTC --out output/features.parquet --log output/eval.jsonl

# explain one bar in plain language
python -m gbpjpy_engine inspect --csv gbpjpy_h4.csv --tz UTC --timestamp 2024-03-01T08:00:00Z

# list every parameter and its purpose
python -m gbpjpy_engine config
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
```

CSV input needs `timestamp, open, high, low, close` and optionally `volume` and `spread`.
Naive timestamps must be given an explicit timezone (`--tz` / `assume_timezone`); the engine never guesses.
`volume` is broker **tick** volume (not exchange volume); volume and spread are carried through for later
execution/market-quality modules but are not used by any Phase 1A feature.

The core is platform agnostic (stdlib + numpy + pandas + PyYAML). Future MT4/MT5 adapters must convert broker
server time to canonical UTC bars before data reaches the engine; no adapter exists yet.

## Documentation

See [`docs/PHASE_1A_H4_MARKET_INTELLIGENCE.md`](docs/PHASE_1A_H4_MARKET_INTELLIGENCE.md) for every feature, its
methodology, parameters, look-ahead protections, swing confirmation, regime rules, bias derivation, assumptions
and known limitations.

## Layout

```
src/gbpjpy_engine/   engine package (data, features, classification, engine, snapshot, logging, research, cli)
config/              documented baseline configuration (h4_default.yaml)
docs/                design and methodology documentation
tests/               unit, anti-look-ahead and synthetic-scenario tests
```
