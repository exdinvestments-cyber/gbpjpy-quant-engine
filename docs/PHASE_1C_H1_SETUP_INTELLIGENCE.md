# Phase 1C — GBPJPY H1 Setup Intelligence Engine

**Question answered.** H4 says which direction, if any, may be investigated. H1 answers: *is a sufficiently
high-quality setup developing in that permitted direction?*

**An H1 setup is NOT a trade.** A `QUALIFIED` setup carries no entry price, stop loss, take profit, position size or
order — none of these exist anywhere in the codebase. `invalidation_reference` is the structural premise level of a
setup, stored for research; it is not a stop. No score or threshold has been validated against outcomes; all are
descriptive baselines, and no logic was engineered to make scores high or to fit profit.

```
MARKET DATA -> CANONICAL NORMALISATION -> H4 ENGINE (1A/1A.1) -> H4 PERMISSION (1B) -> H1 SETUP ENGINE (1C)
            -> [future] ENTRY ENGINE -> RISK ENGINE -> BROKER-NEUTRAL EXECUTION -> MT4 / MT5 ADAPTERS
```

## 1. Reuse and integration

| Need | Reused component (unchanged behaviour) |
|---|---|
| H1 bars, validation | canonical model, `validate_bars` (H1 `DataConfig`, 60 min) |
| H1 features | `features/pipeline.compute_feature_frame` — the Phase 1A stack extracted from the H4 engine (H4 output proven bit-identical) |
| swings, HH/HL, break lifecycle | `compute_structure` |
| H1 zones | incremental `ZoneBook` (`compute_levels`) |
| memory, legs, hierarchy | `SwingMemoryTracker`, `LegEngine`, `classify_layer` |
| displacement | `displacement_features` |
| breakout quality / acceptance / failure / false-break risk | `BreakoutAnalyzer` |
| liquidity references, sweeps | `LiquidityMap` |
| origin zones, compression, expansion | `OriginZoneTracker`, `compression`, `ExpansionTracker` |
| retracement / range position / room | `premium_discount`, `room` |

New: `data/resample.py` (complete-bucket H1→H4 aggregation), `h1/config.py`, `h1/alignment.py`, `h1/signals.py`
(pullback, transition, reclaim, rejection, candle patterns, confluence, H1 room), `h1/setups.py` (families,
conflict, scoring, lifecycle), `h1/engine.py`, `h1/report.py`, `synthetic_h1.py`.

## 2. H4 -> H1 alignment (point-in-time)

An H1 bar closing at `t` is joined (backward as-of join on **close times**) to the latest H4 bar whose close is
`<= t`. An H4 candle still forming at `t` has a close later than `t` and is therefore unreachable. The H1 bar that
closes at the same instant as an H4 candle may see it (both complete). Joining on close times makes the join
independent of the H4 grid anchor (e.g. brokers opening H4 at 21:00 UTC after DST) and robust to missing bars.

Safeguards: a post-join assertion (`AlignmentError` → every H1 bar blocked with `H4_ALIGNMENT_ERROR`); context age
per bar with `NONE` / `STALE` (> `alignment.max_context_age_hours`, default 8 h) statuses that block H1; H4 objects
(zones, origin zones, room barriers) are read only through the aligned H4 bar index.
`resample_complete` emits only complete buckets, so a partially observed H4 candle never exists as a bar.

Tests (`tests/test_h1_data_alignment.py`, `tests/test_h1_lookahead.py`): hour-by-hour mapping, exact-close boundary,
missing H4 candles (stale, never a later bar), DST-shifted and irregular H4 grids, missing H1 candles, and the
decisive test — rewriting the rest of the H4 candle that is still forming at `T` (so the completed H4 bar really
differs) leaves every H1 row, detail and setup up to `T` unchanged.

## 3. H1 intelligence

* **Structure / hierarchy**: confirmed swings with occurrence and confirmation timestamps (3-bar confirmation);
  H1 primary (last 12 swings) and immediate (last 4 swings + provisional close) layers.
* **Alignment states**: `FULL_BULLISH/BEARISH_ALIGNMENT`, `H4_BULL/BEAR_H1_PULLBACK` (an active counter-direction
  correction or opposing immediate structure), `H1_COUNTERTREND`, `H1_TRANSITION`, `H1_RANGE`, `CONFLICTED`,
  `H4_TWO_WAY`, `BLOCKED_BY_H4`. Full alignment is not assumed superior; states are stored for testing.
* **Legs**: Phase 1B leg metrics/classes, classified relative to the H4-permitted direction.
* **Displacement, momentum, volatility, chop, efficiency**: reused causal features; `h1_chop_score` = 0.85 × Phase
  1A chop score + 0.15 × wickiness; H1 vs H4 ATR% ratio reported.
* **Pullback engine** (per side): correction of the latest impulse ≥ 1.5 ATR, measured from its end pivot:
  depth (max and current %), ATR distance, duration, velocity, efficiency, overlap, structural damage, counter-momentum
  deterioration. States `NO_PULLBACK`, `SHALLOW`, `HEALTHY`, `DEEP`, `STRUCTURE_THREATENING`,
  `FAILED_PULLBACK_CONTEXT`. `pullback_quality_score` blends controlled velocity, reasonable depth (35–65 % band —
  untested, not a Fibonacci claim), declining counter momentum, structural integrity, corrective overlap and an
  orderly path. Deeper is not assumed better.
* **Structural transition**: a new H1 break against / out of the prevailing H1 structure (not a continuation BOS)
  after a counter leg ≥ 1 ATR. Score: break magnitude, displacement, counter-momentum deterioration, bars since the
  counter extreme (stall), size of the counter move, lifecycle state (0 once FAILED/INVALIDATED). Returns direction,
  score, broken level, confirmation timestamp (only once known), reason codes.
* **Reclaim**: when an H1 break becomes FAILED/INVALIDATED, the lost level is reclaimed the other way: distance lost,
  duration, reclaim strength, close quality, displacement, follow-through (measured as it happens).
* **Rejection**: bullish/bearish scores from wick/range, wick/body, close location, ATR-normalised wick,
  historical wick percentile, level interaction (H1/H4 zones, origin zones, H4 swing extremes, liquidity references)
  and follow-through to the current bar.
* **Candle patterns** (engulfing, pins, inside/outside, strong closes) are output columns only; no setup logic reads
  them (tested).
* **Confluence**: H1 zones, H1/H4 origin zones, H4 zones and H4 swing extremes are merged into clusters
  (`confluence.cluster_tolerance_atr`); each cluster counts once, with credit for having both timeframes, distinct
  source types and maximum strength — duplicates of the same price information add nothing (tested).
* **Liquidity / sweeps / breakouts / compression / expansion**: reused Phase 1B components on H1 bars.
* **Location** (per side): confluence cluster, H1 and H4 structural range position, break/retest area, liquidity
  interaction, proximity to the last H1 swing. Descriptive; location never creates a setup by itself.
* **Room**: `min(H1 room, H4 nearest barrier converted into H1 ATR)`; no R:R.

## 4. Setup families, scores and lifecycle

Archetypes (candidate structures for later testing, not assumed profitable):

| Family | Premise (eligibility) | Trigger |
|---|---|---|
| TREND_PULLBACK_CONTINUATION | shallow/healthy/deep pullback of a qualifying impulse | max(transition, reclaim, rejection, displacement) |
| BREAK_RETEST_CONTINUATION | recent confirmed/accepted H1 break in the direction **with a successful retest** | max(rejection, displacement) |
| LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION | recent rejected sweep implying the direction | max(transition, displacement, reclaim) |
| COMPRESSION_EXPANSION_IN_H4_DIRECTION | recent expansion out of compression in the direction, not already extremely extended | expansion quality |

* **Conflict** (`h1_setup_conflict_score`, noisy-OR): strong opposing displacement, high chop, trigger under an
  opposing barrier, sweep without follow-through, high breakout-failure risk, extreme extension, H1 structure broken
  against the direction.
* **Quality families**: H4_CONTEXT, H1_STRUCTURE, PULLBACK_QUALITY, LOCATION, DISPLACEMENT, MOMENTUM,
  LIQUIDITY_CONTEXT, MARKET_QUALITY, ROOM_TO_MOVE, CONFLICT — one score each, weighted (`scoring.w_*`).
* **long_setup_score / short_setup_score** = (½ archetype score + ½ quality score) × (1 − ½ conflict/100); stored
  separately per side. Never a sizing input.
* **setup_confidence** = weighted share of quality families ≥ 50, × (1 − conflict/100). It measures **evidence
  consistency; it is not a probability of success** and must not be used for sizing.
* **Lifecycle**: `NO_SETUP → WATCHING` (premise present, H4 permits, no blockers) `→ DEVELOPING` (score ≥ 45)
  `→ QUALIFIED` (score ≥ 60, confidence ≥ 50, conflict ≤ 50, room ≥ 30, trigger ≥ 50, eligible, permitted, no
  blockers). Phase 1C stops at QUALIFIED.
* **Invalidation** (exact reason stored): H4 permission withdrawn; hard blocker; close beyond the invalidation
  reference (impulse origin while developing; frozen to the correction extreme at qualification); pullback broke
  the impulse origin; breakout failed; opposing displacement ≥ 70; room disappeared.
* **Expiry**: not qualified within 24 bars; qualified setup stale after 12 bars; price travelled ≥ 3 ATR without the
  setup; H4 regime changed; premise gone or superseded; a qualified setup whose room vanished (premise consumed).
* **Duplicate protection**: one active setup per side; stable ids `H1-<SIDE>-<FAMILY>-<anchor>-<bar>`; each anchor
  event (impulse leg / break / sweep / expansion) can create at most one setup.
* **Gating**: `actionable_setup_permission_<side>` requires H4 `ALLOW_<SIDE>`/`ALLOW_BOTH`, current H4 context and no
  H1 blockers. Nothing overrides an H4 hard blocker. H1 still analyses blocked sides for research.
* **Hard blockers**: INVALID_H1_DATA, STALE_H1_DATA, INSUFFICIENT_HISTORY (H1 or H4), EXTREME_H1_VOLATILITY,
  SEVERE_H1_CHOP, H4_BLOCK_ALL, H4_PERMISSION_CONFLICT (side), UNRESOLVED_DATA_GAP, NO_H4_CONTEXT,
  STALE_H4_CONTEXT, H4_ALIGNMENT_ERROR, H1_CONTEXT_ERROR (fail-safe: any exception blocks that bar and all later bars).

## 5. Outputs, audit and research storage

* `H1SetupResult.frame` — one row per H1 bar (alignment, structure, pullbacks, displacement, momentum, volatility,
  chop, efficiency, confluence, sweeps, breakouts, transitions, reclaims, rejection, patterns, compression, location,
  room, conflict, scores, confidence, families, lifecycle states and ids, actionable permissions, blockers, reason
  codes). `export(path)` → Parquet/CSV/JSONL.
* `details[i]` — full evidence per bar, including a per-side explanation (quality families, archetype scores and
  eligibility, missing requirements, conflict hits, blockers, reason codes).
* `setups` (full lifecycle with timestamped transitions), `qualified_setups` (structured objects), `transitions`,
  `reclaims`, `sweep_events`, `expansions`, `errors`.
* **Counterfactual log** (`counterfactual`): setups that never qualified but reached a peak score ≥ 45
  (`FAILED_QUALIFICATION`: score, missing requirements, blockers, invalidation/expiry reason) and promising candidates
  blocked by H4 (`GATED_BY_H4`, once per anchor). Recorded at the time — rules are never re-applied retrospectively.
* `explain_setup(id)` / `h1.report.explain_h1_setup` answer "why did this setup qualify / fail?";
  `python -m gbpjpy_engine h1 --h1-csv ...` runs the pipeline.

## 6. Known limitations

* Validated only on deterministic synthetic data; no threshold, weight or band has been tested against outcomes.
* The H1 primary layer sees only the last 12 confirmed swings; swing confirmation lags by 3 H1 bars.
* Only the most recent H1 break within its monitoring window is analysed in depth.
* Context-age limit (8 h) blocks the first H1 bars after weekends until a new H4 candle completes.
* Room uses the nearest H4 barrier from the aligned H4 context plus H1 barriers; deeper barrier stacks are ignored.
* One active setup per side; a second, simultaneous archetype on the same side waits until the active one ends.
* Intrabar sequencing within an H1 bar is unknown (completed bars only).
* No news/calendar awareness.
* On the bundled synthetic scenarios almost every setup is TREND_PULLBACK_CONTINUATION (one BREAK_RETEST);
  LIQUIDITY_SWEEP_REVERSAL and COMPRESSION_EXPANSION are covered by unit tests only, not observed end-to-end.
* The per-bar loop is pure Python (~3 ms per H1 bar; ~6 s for 1,920 bars). The main costs are pandas row
  access, the pullback/leg metrics and confluence clustering. Fine for research, not yet tuned for tick-level use.
