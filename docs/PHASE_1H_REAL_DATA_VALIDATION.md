# Phase 1H — Real-Data Validation & Research Infrastructure

Question: **does this system survive real GBPJPY data?**

The aim is to *falsify* the strategy, not to prove it. A disappointing truthful result is worth more than an impressive
false one. Nothing in this phase changes a strategy, risk or execution parameter. There is no auto-tuning, no network
access and no order.

## 0. Current status: REAL_DATA_REQUIRED

No real GBPJPY dataset exists in the repository or the research store. The infrastructure is complete and tested, but
**no real-data validation result exists**. No synthetic data has been substituted. The tests use the synthetic
Phase 1D fixture only to exercise the machinery. Every such run is labelled `SYNTHETIC_TEST_ONLY` / `NOT_VALIDATED`.

## 1. Architecture (`src/gbpjpy_engine/validation/`)

| Module | Role |
|---|---|
| `provenance.py` | `DatasetProvenance` (provider, symbol, timeframe, coverage, source timezone, price type BID/ASK/MID/UNKNOWN, volume type, spread availability/unit/resolution, missing-data notes, declared outages, retrieval/import time, SHA-256), `HistoricalDataProvider` protocol, `FileProvider` (CSV, Parquet, `TERMINAL_TAB_EXPORT`) |
| `datastore.py` | RAW → NORMALISED → DERIVED layers; RAW is byte-exact, read-only and re-hashed before every use |
| `quality.py` | Pre-backtest data-quality report, gap classification and the fail-safe gap blackout; `quality_gate` blocks corrupt data |
| `resampling.py` | Versioned `H4BarDefinition`; deterministic H1→H4; exact aggregation check; provider-H4 vs derived-H4 comparison |
| `costs.py` | Spread / commission / slippage / swap models and DIAGNOSTIC / BASELINE / STRESSED scenarios |
| `simulator.py` | Deterministic historical fill/exit simulator with explicit bid/ask, gap and OHLC-ambiguity handling |
| `replay.py` | Point-in-time replay of Phases 1A–1E, warm-up, signal ledger, strategy trade ledger, account simulation through Phase 1F + Phase 1G |
| `metrics.py` | R metrics, expectancy (two ways), drawdown/recovery, streaks, exposure, Sharpe/Sortino/Calmar, seeded bootstrap CIs, breakdowns, concentration, stability, frequency |
| `splits.py` | Chronological splits with embargo, `SplitGuard`, `HoldoutLock`, walk-forward windows and aggregation |
| `experiments.py` | Append-only hash-chained experiment manifest, research budget, hypothesis register |
| `montecarlo.py` | Sequence / block bootstrap, cost / missed-trade / worse-fill / tail stress, risk of ruin, compounding, risk-policy comparison, capital scale |
| `diagnostics.py` | Family coverage, score calibration, attribution, long/short, MAE/MFE, stops/targets, counterfactual filter contribution, structural-limitation measurements |
| `robustness.py` | Parameter / weight sensitivity, threshold alternatives, ablation, a simpler baseline, null baselines, overfitting-risk diagnostic |
| `gates.py` | Validation statuses and promotion gates |
| `objective.py` | The £500k ambition: analysed only after validation and isolated from every decision |
| `report.py` | Canonical `BacktestResult`, dashboard datasets (CSV/JSON), text report with a marketing-language check |
| `runner.py` | `first_real_data_run` (REAL_DATA_REQUIRED or the frozen baseline run once) and `run_validation` |
| `snapshot.py` | Strategy snapshot and configuration freeze |

**Separation from production.** The package reads strategy outputs. It is imported by no strategy, risk or
execution module (tested). It never assigns to a configuration, never writes `config/*.yaml`, and builds every
sensitivity variant as a new object.

## 2. Repository and configuration freeze

`strategy_snapshot` records:

- the package version, git commit and working-tree state;
- the SHA-256 of every strategy source file;
- hashes of the six configuration objects (H4, H1, entry, trade, risk, execution);
- the execution/cost assumptions, the Python/platform/numpy/pandas versions and a timestamp.

Its `snapshot_id` depends only on code, configuration and assumptions, not on the timestamp.

`configuration_freeze` lists every parameter (about 650) with a kind (threshold / weight / lookback / expiry /
ATR multiplier / buffer) and a label:

- `UNFITTED_BASELINE`: all strategy parameters at the start of Phase 1H.
- `EXTERNAL_CONSTRAINT`: symbol, pip size, timeframe, spread units, sessions and similar.
- `EMPIRICALLY_VALIDATED`: none.

## 3. Data: provenance, immutability, normalisation, quality

- **Provenance.** Required for every dataset. Backtests refuse a frame without it (`require_known_provenance`).
  A provider name that looks synthetic must be flagged `is_synthetic`. Synthetic datasets live in a separate RAW
  namespace, and the real-data run refuses them.
- **RAW.** The provider file is copied unchanged, hashed and made read-only. The source file is never touched.
  `verify_raw` re-hashes it before every use, and any change raises `RawDataModified`.
- **NORMALISED.** Canonical UTC bars. The declared source timezone is applied **only** to naive timestamps.
  Ambiguous or non-existent DST times raise an error rather than being guessed. Spread is converted to pips from its
  declared unit (POINTS ÷ 10, PRICE × 100). There is no sorting, de-duplication, filling or repair. A transformation
  log and the normalised hash are stored, and tampering with the normalised file is detected.
- **DERIVED.** Deterministic products such as H4 from H1, tagged with the bar definition and the parent hash.
- **Quality report.** Covers:
  - duplicates, missing and non-monotonic timestamps;
  - OHLC inconsistencies, zero/negative and implausible prices, price spikes;
  - weekend bars, DST irregularities (the weekly-open hour distribution) and grid shifts;
  - abnormal and zero spreads, missing volume/spread;
  - stale sequences and flat or repeated bars.

  Any ERROR sets the status to BLOCKED and `quality_gate` raises. Warnings travel into every result.
- **Gap policy.** Each gap is classified as EXPECTED_MARKET_CLOSURE (weekend, Christmas / New Year),
  SHORT_UNEXPLAINED_GAP, MAJOR_MISSING_INTERVAL, PROVIDER_OUTAGE (declared) or UNKNOWN. After a major, outage or
  unknown gap, no new decisions are taken for 24 bars (fail safe).
- **H1/H4 consistency.** The default H4 is derived from H1 on the UTC 00/04/08/12/16/20 grid (`H4_UTC_GRID_00_v1`).
  `verify_aggregation` checks it exactly. A provider H4 file is compared bar by bar (missing bars, OHLC mismatches,
  grid alignment) and never mixed in. Server-clock grids (for example `Europe/Athens`) follow that clock's DST, and
  irregular DST-day buckets are dropped and reported.

## 4. Price representation and costs

- **Bid/ask by price type.** BID data: ask = bid + spread. ASK data: bid = ask − spread. MID data: ± spread/2.
  UNKNOWN is treated as BID and every trade is flagged `PRICE_TYPE_UNKNOWN`. A LONG enters at the ASK and exits at
  the BID. A SHORT enters at the BID and exits at the ASK. A mid price is never called executable.
- **Spread.** HISTORICAL is the bar's own timestamp-appropriate spread. A bar with no spread raises `MissingSpread`
  unless an explicit fallback is declared. ASSUMED is an explicit value when the data has no spread (default 3 pips,
  the Phase 1D assumption), always stress-tested. ZERO exists only in the DIAGNOSTIC scenario. Zero is never used
  silently.
- **Commission.**
  - KNOWN: from the broker.
  - ZERO_DECLARED: must cite its source.
  - UNKNOWN: net results stay undefined unless a scenario is chosen.
  - SCENARIO: the default 0.5 pip round turn from Phase 1E, labelled as an assumption.
- **Slippage.** ZERO_SLIPPAGE_DIAGNOSTIC_ONLY, FIXED, SPREAD_DEPENDENT, VOLATILITY_DEPENDENT or EMPIRICAL (resampled
  deterministically per trade and seed). Slippage is always adverse. Stop exits can add extra slippage.
- **Swap.** UNKNOWN (default, disclosed as `SWAP_UNKNOWN`), KNOWN or SCENARIO. Supports long/short rates per night,
  a triple-rollover weekday (declare it per provider) and the rollover hour.
- **Scenarios.** `ZERO_COST_DIAGNOSTIC` (an upper bound only, never presented as realistic), `BASELINE` and
  `STRESSED` (spread × 1.5 + 1 pip, slippage × 2, stop +2 pips, commission +0.5 pip, adverse swap 1 pip/night).

## 5. Historical execution simulator

A proposal is filled at the OPEN of its decision bar, which is the Phase 1D decision time. The simulator then walks
forward bar by bar.

- **Gaps.** An opening gap through the stop fills at the worse open (a loss beyond −1R is possible). A gap through
  the target fills AT the target; the gap gift is not taken.
- **Target fills.** Targets fill as limit orders, with no positive slippage.
- **Bars that touch both stop and target** (intrabar order unknowable):
  - **CONSERVATIVE**: the stop is assumed first.
  - **AMBIGUOUS**: stop first, and the trade is flagged and also reported excluded.
  - **LOWER_TIMEFRAME_REQUIRED**: M15/M5/M1 bars inside that H1 bar decide the order. Without them the trade is
    UNRESOLVED and treated conservatively.

  Lower-timeframe data is used only for sequencing after the H1 decision and never reaches a strategy decision (tested).
- **End of data.** Trades still open close at the last bar and are flagged `OPEN_AT_END`.
- **MAE/MFE.** Recorded at bar granularity in pips, ATR and R. `capture_ratio` = realised R / MFE R.
- **Gross vs net.** Gross R is the representation-price move from the entry open to the exit event, with no costs.
  Net R includes spread on both sides, slippage, commission and swap.
- **Exit architecture.** Partial exits and a holding cap are supported but not active, matching Phase 1E
  (single target, no trailing, no time exit).

## 6. Replay, warm-up and ledgers

- **Point-in-time replay.** The Phase 1A–1E engines are unchanged and already causal: features become available
  at bar close, and decisions happen at the next open. One pass is therefore identical to bar-by-bar replay.
  `test_validation_lookahead` mutates the future and checks every earlier H4 state, H1 setup, entry, stop, target,
  proposal, risk decision and execution decision.
- **Warm-up.** The first H1 bar at which H1 features, H4 features (converted to their H1 availability time), H4
  context, the spread history and the stop-percentile history are all complete. Nothing earlier is scored.
- **Signal ledger.** Every opportunity, including rejected ones: H4 regime/permission/context status, family,
  setup score/confidence/state, entry decision and quality, proposal decision and quality, risk decision, execution
  decision and reason codes. Setups gated by H4 are included.
- **Trade ledger (R).** At most one open trade per direction (mirrors the Phase 1F no-pyramiding default). The
  BASELINE run fixes the path, and every other cost scenario re-prices that identical path. Signals and trade
  selection are therefore identical across scenarios, and only costs differ (critical cost test).
- **Account simulation.** The same path runs through the Phase 1F `AccountRiskEngine` and the Phase 1G
  `ExecutionEngine`, using the deterministic paper port loaded with historical bid/ask quotes. A trade's close is
  recorded only when simulated time reaches it. The margin leverage is an explicit EXTERNAL_CONSTRAINT: 30 is the UK
  retail cap, so declare the broker's real value. Without it, Phase 1F fails closed with `MARGIN_UNKNOWN`.

## 7. Statistics

- **R metrics first.** R-based metrics (mean, median, total, standard deviation, positive/negative distributions,
  tails) come before any compounding. Expectancy is computed directly from the full R distribution and also via the
  win-rate formula.
- **Core metrics.** Profit factor (undefined, not infinite, without losses), maximum drawdown and its duration,
  time and trades to recover, holding times, streaks, exposure, frequency and time in market.
- **Risk-adjusted ratios.** Sharpe/Sortino/Calmar come with their sampling and annualisation assumptions and are
  never an optimisation objective.
- **Uncertainty.** Seeded i.i.d. and moving-block bootstrap CIs for win rate, mean R, expectancy and profit factor,
  plus Wilson intervals and P(expectancy ≤ 0). Sample warning levels: INSUFFICIENT (< 30), SMALL (< 100),
  MODERATE (< 300).
- **Breakdowns.** Year / quarter / month (losing periods are never hidden). Long vs short. Each setup family.
  Pre-existing regimes (H4 regime, H1/H4 volatility regime, ATR-percentile buckets, chop). Session, day of week and
  pre-defined UTC buckets.
- **Concentration.** Best trade/5/10, best month/quarter/year, results after removing the best 1/5/10, worst trades
  and tails, and chronological distribution stability (KS distance).

## 8. Process safeguards

- **Chronological splits.** DEVELOPMENT → VALIDATION → FINAL_HOLDOUT with an embargo. Never shuffled.
- **`SplitGuard`.** The only gateway to bars. Each request is truncated at its segment end, so later bars never
  enter a computation.
- **Holdout lock.** FINAL_HOLDOUT bars are refused until `HoldoutLock.unlock`, which requires a strategy snapshot,
  a configuration hash, declared hypotheses, a validation-results id, an operator and the confirmation phrase.
  Parameter exploration, sensitivity tuning, ablation and walk-forward calibration are refused on the holdout even
  after unlock. The first access is recorded as `HOLDOUT_RESULT`; every later access is `POST_HOC`. The lock is
  file-backed.
- **Walk-forward.** Anchored or rolling windows inside the non-holdout range. The strategy is not re-fitted, since
  no calibration exists. Reports cover each window, dispersion, profitable and unprofitable windows, and
  concentration in one window.
- **Experiment manifest.** Append-only JSON lines with a hash chain. Failed and abandoned experiments are kept, ids
  are never reused, and `verify` detects rewriting.
- **Research budget.** Default 40 exploratory experiments per segment. After that, a new untouched validation
  period is required.
- **Hypothesis register.** Records the statement, evidence, affected component, possible change, the data that
  generated it, and the untouched data required to test it.
- **Analysis labels.** Every report is labelled PRE_DECLARED_TEST, EXPLORATORY_ANALYSIS, POST_HOC_ANALYSIS or
  HOLDOUT_RESULT.
- **Reproducibility.** Strategy snapshot + configuration hash + dataset hash + experiment id + seed. Two runs with
  identical inputs produce identical results (tested).

## 9. Robustness, baselines, overfitting

- **Sensitivity.** A pre-declared set of six parameters is perturbed at × 0.8 / 0.9 / 1.0 / 1.1 / 1.2 around the
  frozen baseline: H1 qualify score, H4 minimum context score, minimum entry quality, run-away ATR, stop buffer and
  minimum target reachability. Each is flagged CLIFF_EDGE, FRAGILE, STABLE_REGION or NO_BASELINE_EDGE. Variants are
  never adopted.
- **Weight sensitivity.** Each score weight is perturbed ± 20% with its group renormalised.
- **Threshold alternatives.** At most 20 nearby values in total.
- **Ablation.** One H1 feature family's weight is set to zero at a time (liquidity, location, momentum,
  chop/market quality, displacement, room).
- **Simple baseline.** H4 EMA trend plus an H1 fast-EMA re-cross, with a 1.5-ATR stop and a 2R target, using the
  same data, simulator and costs.
- **Null baselines.** Random direction at the strategy's own entry times, and random timing inside H4-permitted
  bars with the strategy's own stop/target distributions (seeded). The strategy's percentile against these is
  reported.
- **Overfitting risk.** A heuristic 0–100 score from tunable-parameter count, experiment count, sample size,
  sensitivity, in-sample → out-of-sample degradation, return concentration and family concentration. It is **not**
  a probability.
- **Diagnostics.**
  - Score calibration with monotonicity: MONOTONIC_INCREASING, POSITIVE_BUT_NON_MONOTONIC, NEGATIVE_RELATIONSHIP or
    STATISTICALLY_UNCLEAR.
  - Setup-family coverage: NEVER_OBSERVED, NEVER_QUALIFIED, EXTREMELY_RARE, RARE or OBSERVED.
  - MAE/MFE hypotheses.
  - Stop and target analysis.
  - Rejected-opportunity counterfactuals and filter contribution, labelled POST-DECISION DIAGNOSTIC.
  - One-active-setup overlap, H1 swing-confirmation lag, 12-swing horizon disagreement, multiple coexisting breaks,
    nearest vs stacked barriers, weekend context-age blocking.
  - NEWS_DATA_UNAVAILABLE disclosure.

## 10. Monte Carlo and risk research

- **Trade sequences.** Permutation, i.i.d. and block bootstrap, giving ending R, drawdown, losing streak and time
  under water.
- **Stress tests.**
  - Deterministic cost grid, with the break-even extra cost in pips.
  - Randomised costs, missed trades (10% / 25%), worse fills, and tail gap events beyond −1R.
- **Risk of ruin.** Uses the Phase 1F `RiskOfRuinInputs` interface with the empirical R distribution, tail losses
  and i.i.d. or block dependency. It is never reported as zero: the rule-of-three bound is reported instead.
- **Compounding.** Reported separately and never used to disguise weak expectancy.
- **Risk policies.** A small pre-declared set (0.25 / 0.5 / 1.0%), shown for information only. The highest ending
  balance is never a selection criterion.
- **Capital scale.** Shows when volume would hit broker maximums. Slippage, impact, margin and liquidity limits at
  size are listed as unmodelled.

## 11. Statuses, gates and the £500k objective

- **Statuses.** NOT_VALIDATED, DATA_VALIDATED, IN_SAMPLE_ONLY, OUT_OF_SAMPLE_TESTED, WALK_FORWARD_TESTED,
  HOLDOUT_TESTED, PAPER_TRADING_REQUIRED, FAILED_VALIDATION. The status comes from the available evidence, never
  from positive P&L.
- **Promotion gates.** All must pass:
  - adequate out-of-sample sample and positive out-of-sample expectancy;
  - bootstrap P(E ≤ 0) and stressed-cost expectancy;
  - drawdown and walk-forward consistency;
  - parameter stability and concentration;
  - execution stress and data coverage.

  A missing metric counts as a failure (UNAVAILABLE). Failures are listed.
- **The £500,000 objective.** Analysed only at HOLDOUT_TESTED / PAPER_TRADING_REQUIRED, and only then as capital
  required per risk level with a range from the expectancy CI. If validated expectancy is not positive, the result
  says so. The module imports no decision layer, and tests show that changing the objective changes no signal,
  entry, stop, target, risk policy or acceptance.

## 12. How to import real data (REAL_DATA_REQUIRED)

**Required:** GBPJPY **H1**. Optional: a provider **H4** file (compared against H1-derived H4) and **M15 / M5 / M1**
or ticks (used only for execution sequencing).

**Formats:**
- `CSV` with a header row.
- `PARQUET`.
- `TERMINAL_TAB_EXPORT`: tab-separated platform history export, for example from MT5 (*History Center / Bars →
  Export*) with the header `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>`.

**Columns:**
- Required: `timestamp` (bar OPEN time), `open`, `high`, `low`, `close`.
- Recommended: `volume` (tick volume) and `spread` (per bar).
- Ticks: `timestamp`, `bid`, `ask`.

**Provenance to declare (never guessed):**
- provider;
- the source timezone (for platform exports, the broker **server** timezone, for example `Europe/Athens` or
  `Etc/GMT-2`);
- price type (platform bar exports are normally BID);
- volume type;
- spread availability and unit (platform exports report spread in POINTS);
- retrieval time.

```
python -m gbpjpy_engine validate import --file GBPJPY_H1.csv --timeframe H1 --provider broker_export \
    --source-tz Europe/Athens --price-type BID --volume-type TICK_VOLUME --spread PER_BAR --spread-unit POINTS \
    --format TERMINAL_TAB_EXPORT --retrieved-at 2025-01-15T10:00:00Z
python -m gbpjpy_engine validate status
python -m gbpjpy_engine validate run        # the frozen baseline, run once, unchanged
```

**Recommended coverage.** As many years as available, spanning trending, ranging, high- and low-volatility
environments. The actual coverage and regime diversity are reported; no particular number of years is assumed to be
adequate.

**Storage.** The store defaults to `data/research_store`, which is git-ignored for raw data and parquet. Market data
is never committed.

## 13. Performance

The research engine is slower than live execution by design.

- The Phase 1A–1E feature, structure and zone computation is pure Python and dominates runtime: about 200 H1
  bars/s on the synthetic fixture.
- Every sensitivity variant repeats the full replay.
- The runner reports bars processed, trades, runtime, bars per second, peak process memory and the bottleneck.

## 14. Known limitations

- No real data has been validated; every number in the test suite comes from synthetic fixtures.
- Intrabar order is unknown without lower-timeframe data; ambiguous bars are resolved conservatively.
- Historical spreads are per-bar summaries, not the quote at the fill.
- Commission and swap are UNKNOWN unless the provider declares them; baseline results use labelled assumptions.
- News: NEWS_DATA_UNAVAILABLE. The strategy has no event awareness.
- The account simulation's equity ignores floating P&L between decisions (ResearchAccount); rate conversion for P&L
  uses the approval-time pip value.
- The exposure rule (one open trade per direction) is fixed by the BASELINE path.
- The walk-forward has nothing to re-fit (by design); it measures stability, not adaptation.
- Monte Carlo resamples history and cannot represent regime change; the overfitting score is heuristic.
- Structural-limitation measurements (swing horizon, multiple breaks, barriers) are descriptive proxies.
- H4 from a server-clock grid drops DST-transition buckets.
