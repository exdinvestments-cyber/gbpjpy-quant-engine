# Phase 1A — GBPJPY H4 Market Intelligence Engine

**Scope.** A descriptive engine that turns closed GBPJPY H4 bars into a structured, auditable description of the market
environment: structure, trend, volatility, momentum, chop, levels, location, regime and a strategic
LONG / SHORT / NEUTRAL bias.

**Out of scope (by design).** H1 entry logic, trade signals, order placement, broker connectivity, position sizing,
leverage, optimisation. None of these exist in the codebase. The bias is *strategic context* for later phases, not an
entry signal. No component has been validated for profitability and no performance claim is made anywhere.

---

## 1. Architecture

```
data/          canonical bar model + validation (flags, never repairs)
features/      pure, causal feature calculators
  indicators   EMA, Wilder RMA/ATR/ADX, trailing percentile, efficiency ratio
  candles      candle geometry + descriptive classes
  volatility   ATR metrics, volatility regime, shock detection
  trend        EMA trend engine + ADX/DI
  momentum     price-derived momentum
  structure    confirmed swings, HH/HL/LH/LL, BOS/CHoCH, structure quality   (forward walk)
  chop         directional efficiency + chop score + market quality
  levels       support/resistance zones                                     (forward walk)
  context      round numbers, range location, extension, sessions, time
classification/
  regime       ordered, transparent rule list with evidence / conflicting evidence
  bias         family-weighted evidence, dampeners, LONG/SHORT/NEUTRAL
engine.py      orchestration -> H4AnalysisResult (feature table + snapshots)
snapshot.py    H4Snapshot (per-bar structured output)
logging_utils  JSONL evaluation log, CSV/Parquet/JSONL export
research.py    human-readable debug view of one bar
synthetic.py   deterministic synthetic scenarios (engineering tests only)
cli.py         `python -m gbpjpy_engine {run,inspect,synthetic,config}`
```

Data → features → classification → output are strictly separated. Later phases consume `H4Snapshot` (or the flat
feature table) and never reach into feature internals.

## 2. Data model and conventions

| Field | Meaning |
|---|---|
| `timestamp` | bar **open** time, timezone-aware, normalised to **UTC** |
| `available_at` | `timestamp + 4h` — the bar's close. Every feature for the bar is only knowable at this time |
| `open/high/low/close` | JPY prices (float) |
| `volume` | **broker tick volume** if supplied, else NaN (flagged). Not exchange volume — see §2.2 |
| `spread` | source-reported spread if supplied, else NaN (flagged). Units are those of the source |
| `source` | data source label |

Historical files and future live feeds use the same canonical frame (`to_canonical`, `load_csv`, `Bar`,
`bars_to_frame`). Naive timestamps are **rejected** unless the caller explicitly names their timezone
(e.g. broker server time `Europe/Athens`); DST-ambiguous local times raise rather than being guessed.
`exclude_unclosed_bars(df, as_of)` / `engine.run(bars, as_of=...)` drop a still-forming bar so it can never be
evaluated as if closed.

### 2.1 Broker timestamps and future MT4/MT5 adapters

Canonical internal timestamps are always timezone-aware UTC (`data.interfaces.assert_canonical` enforces this at
the engine boundary). The core never corrects broker timestamps. **Future MT4/MT5 adapters must normalise broker
server time (commonly EET/EEST, UTC+2/+3 with DST) into the canonical format — UTC bar-open timestamps of closed
bars — before sending data into the strategy engine**, e.g. `to_canonical(raw, assume_timezone="<broker tz>")`.
DST-induced shifts of the H4 grid that remain after conversion are *flagged* (`TIMEZONE_ALIGNMENT_SHIFT`), never
silently repaired.

Target architecture (only the core and the neutral interface exist today):

```
GBPJPY Strategy Core            (this package, platform agnostic: stdlib + numpy + pandas + PyYAML)
        |
Broker-neutral interfaces       (data/interfaces.py: BarSource protocol, assert_canonical)
      /            \
MT4 Adapter     MT5 Adapter     (NOT implemented - future phases)
```

`tests/test_portability.py` parses every module's imports and fails if the core imports anything beyond the
standard library, numpy, pandas and PyYAML, or anything MetaTrader/network related.

### 2.2 Volume and spread

`volume` in retail FX feeds is **broker tick volume**: the number of price updates on one broker's feed. Spot FX
has no centralised exchange, so tick volume is not traded volume, is not comparable across brokers and can change
with a broker's feed infrastructure. `spread` is broker-specific too. Both are preserved unchanged in the canonical
frame, the feature table and each snapshot's `market_data` block so later execution / market-quality modules can
consume them, but **no Phase 1A strategy feature uses them** (tested: randomising volume and spread changes no
feature, regime or bias).

### 2.3 Validation (`validate_bars`)

Validation never modifies data. ERROR issues make the engine refuse to run (`fail_on_data_errors=True`);
WARNING issues are attached to each affected bar (`data_quality_flags`, `data_quality_status`).

| Issue | Severity |
|---|---|
| `TIMEZONE_NAIVE`, `TIMEZONE_NOT_UTC` | ERROR |
| `MISSING_PRICE`, `NONPOSITIVE_PRICE`, `INVALID_OHLC` (high < max(o,c), low > min(o,c), high < low) | ERROR |
| `DUPLICATE_TIMESTAMP`, `OUT_OF_ORDER`, `NEGATIVE_SPREAD` | ERROR |
| `TIMEZONE_ALIGNMENT_SHIFT` — bar-open offset from the H4 grid differs from the dominant offset *seen so far* (typical of DST-shifted broker data) | WARNING |
| `MISSING_BARS` (mid-week gap), `EXTENDED_MARKET_CLOSURE` (weekend gap > `max_weekend_gap_hours`) | WARNING |
| `ABNORMAL_PRICE_GAP` — open-vs-previous-close gap > `abnormal_gap_range_mult` × trailing median range | WARNING |
| `MISSING_VOLUME`, `MISSING_SPREAD`, `ZERO_VOLUME`, `MIXED_SOURCES` | WARNING |

`data_quality_status` per bar: `ERROR` > `WARNING` (price/time integrity concern) > `INFO` (only volume/spread
unavailable) > `OK`. Every validation check is itself causal, so appending future bars cannot change an earlier
bar's flags.

## 3. Look-ahead protection

1. **Vectorised features** use only causal operations: `ewm(adjust=False)`, Wilder recursions, trailing
   `rolling(...)` windows, trailing percentile ranks (`rolling().rank(pct=True)`, window includes the current bar),
   and `shift(+k)` (never negative shifts). Shock and candle normalisation use `atr_prev` (ATR known *before* the
   bar) so a large bar cannot dampen its own abnormality.
2. **Swings, structure, breaks and zones** are computed in an explicit forward walk: at bar `c` only arrays up to
   `c` are read, and per-bar outputs are written once and never revisited.
3. **Swing confirmation.** A fractal pivot at bar `i` needs `right_bars` later bars. It is evaluated at
   `c = i + right_bars`. Every swing carries **two explicit timestamps**: `occurred_at` (open of the pivot bar)
   and `confirmed_at` (close of bar `c`, when it became knowable). Per-bar columns
   (`last_swing_high_occurred_at` / `last_swing_high_confirmed_at`, same for lows), snapshots and the swing history
   keep both. The swing does not exist in any state before `c`. The confirmation lag is intentional anti-repaint
   protection; its sensitivity (`swing.left_bars`, `swing.right_bars`) is configurable and not shortened.
4. **Superseded swings** (a more extreme same-type pivot arrives before a meaningful opposite swing) are marked
   with `removed_index`/`replaced_by`; history is not rewritten, so earlier bars still show the swing that was
   actually known then.
5. **Break lifecycle** is an append-only list of transitions, each stamped with the bar at which it became known.
   The per-bar frame stores the state *as known at that bar*; `BreakEvent.state_at(c)` / `snapshot_at(c)`
   reproduce it later. A break CONFIRMED at T that FAILS at T+N still reads CONFIRMED at T.
6. **Zones** are maintained by an incremental engine whose event log (`zone_events`) is append-only and indexed by
   the bar at which each event became known; a truncated run's events are exactly the prefix of the full run's.
7. **Tests** (`tests/test_lookahead.py`, plus point-in-time tests in `test_break_lifecycle.py`,
   `test_structure_memory.py`, `test_levels_incremental.py`):
   * truncation invariance — running on `bars[:T+1]` reproduces rows `0..T` of the full run *exactly* for every
     column, plus identical snapshots (zones, reasons) at `T`;
   * future perturbation — replacing everything after `T` with different data leaves rows `0..T` unchanged;
   * pivot perturbation — destroying a pivot's right-hand confirming bars changes nothing up to the pivot bar;
   * swings/levels/breaks are never visible before their confirmation / availability time;
   * a canary proves the harness *fails* on a deliberately leaky (centred) feature.

## 4. Features

### 4.1 Market structure (`features/structure.py`)

* **Candidate pivots**: high pivot at `i` if `high[i] > max(high[i-L:i])` and `high[i] >= max(high[i+1:i+R+1])`
  (mirror for lows). `L = left_bars`, `R = right_bars`.
* **Meaningful swings**: a candidate is accepted only if it alternates with the previous swing and is at least
  `min_swing_atr × ATR(at confirmation)` away from it. A more extreme same-type candidate supersedes the previous
  swing. Candle noise therefore never becomes structure.
* **Labels**: each swing vs the previous same-type swing: `HH/LH/EH` or `HL/LL/EL` (equal within
  `equal_level_atr × ATR`).
* **Structural memory**: a rolling window of the `structure.swing_history_size` (default 16) most recent confirmed
  swings is kept. Each record holds swing id, type, `occurred_at`, `confirmed_at`, price, classification
  (HH/LH/EH/HL/LL/EL), `significance_atr` (distance from the previous opposite swing in ATR at confirmation) and
  confirmation lag. `StructureResult.history_at(c)` returns exactly the window held at bar `c`; snapshots expose it
  as `market_structure.swing_history`; per-bar columns add `swing_sequence` (e.g. `HL>HH>HL>HH`), label counts
  (`hist_hh/hl/lh/ll`) and `hist_avg_significance_atr`. Leaving the window never alters a swing record.
* **Swing state** (latest high label, latest low label): `HH+HL → bullish`, `LH+LL → bearish`,
  `HH+LL → transitional` (expanding/conflicting), anything else → `neutral`. Fewer than two highs and two lows →
  `neutral` with `structure_defined=False` (`STRUCTURE_UNCLEAR`).
* **Structural breaks**: at bar `c`, against the latest swing high/low known at the end of `c-1`, a break requires
  the **close** beyond the level by `≥ break_min_atr × ATR`. Wick-only excursions and marginal closes are not
  breaks. Each level can break once. Stored per event: level, swing id, level time, break time, availability time,
  close, magnitude, ATR-normalised magnitude, wick extension, `closed_beyond`, structure before and after,
  lifecycle transitions.
* **Type**: with prevailing structure → `BOS`; against it → `CHOCH`; from neutral/transitional → `BREAKOUT`.
* **Break lifecycle** (evaluated on each later bar's close, only while `bars since break ≤ break_monitor_bars`):

  | From | To | Rule |
  |---|---|---|
  | — | `CANDIDATE` | close beyond the level by `≥ break_min_atr × ATR` |
  | `CANDIDATE` | `INVALIDATED` | any close back through the level within the confirmation window |
  | `CANDIDATE` | `CONFIRMED` | every close for `break_confirm_bars` bars held beyond the level |
  | `CONFIRMED` | `ACCEPTED` | `≥ break_accept_bars` since the break and best close beyond the level `≥ break_accept_atr × ATR(break)` |
  | `CONFIRMED` / `ACCEPTED` | `FAILED` | a later close back through the level by `≥ break_fail_atr × ATR` |

  `FAILED` and `INVALIDATED` are terminal; after `break_monitor_bars` the last state is frozen (a much later return
  through the level is new structure, not failure of this break). Each transition stores bar, timestamp, previous
  state, new state and a reason; the event exposes break time, confirmation, acceptance, failure and invalidation
  timestamps, current and previous state and transition reason. Per-bar columns: `last_break_status`,
  `last_break_previous_status`, `last_break_transition_reason`, `last_break_confirmed_at`, `last_break_failed_at`.
* **Final structure state**: the swing state, overridden to `transitional` when a *live* (CANDIDATE, CONFIRMED or
  ACCEPTED) break within `transition_memory_bars` contradicts it (a live CHoCH), or when a CHoCH/breakout occurs
  from a neutral state. When that break FAILS or is INVALIDATED the override lapses on that bar.

### 4.2 Structure quality (0–100)

Weighted mean of components (each 0–1), all kept in the output as `sq_*`:

| Component | Measure | Weight |
|---|---|---|
| clarity | share of the last `quality_swings` labels agreeing with the dominant direction | 0.20 |
| impulse_pullback | mean impulse leg / mean pullback leg (ATR), 1.0→0, 2.5→1 | 0.15 |
| impulse_size | mean impulse leg in ATR, 1→0, 4→1 | 0.05 |
| break_conflict | 1 − two-sided break share in window, reduced by the share of INVALIDATED/FAILED breaks | 0.10 |
| reversal_frequency | 1 − scaled number of structure-state changes in window | 0.10 |
| swing_spacing | mean bars between swings, 2→0, 8→1 | 0.05 |
| persistence | share of window bars in the current directional state | 0.15 |
| overlap | 1 − scaled mean candle overlap | 0.10 |
| wickiness | 1 − scaled mean wick/range | 0.10 |

Raw values (`sq_avg_impulse_atr`, `sq_avg_pullback_atr`, `sq_impulse_pullback_ratio`, `sq_avg_swing_spacing`,
`sq_state_changes`) are also stored.

### 4.3 Trend engine (`features/trend.py`)

Independent of the swing classifier. EMA(20/50/200) components in [-1, 1]:
order (sign agreement of the three EMA pairs), slope (tanh of EMA slopes in ATR/bar), price position vs each EMA,
EMA20–EMA50 separation in ATR, price-to-EMA50 distance in ATR, and alignment persistence over
`persistence_window`. `trend_score = 100 × Σ w·c` (weights 0.25/0.25/0.15/0.15/0.05/0.15) ∈ [-100, 100];
`trend_strength = |trend_score|`. States: `strong_bullish ≥ 60`, `bullish ≥ 30`, `neutral`, mirrored for bearish.
The trend engine alone never sets the regime — the regime requires structure agreement.

### 4.4 ADX / DI

Wilder ADX, +DI, −DI (period 14), `adx_slope`, `di_spread = +DI − −DI`, `directional_persistence` (mean sign of DI
spread), `adx_state`. ADX never produces a signal; in the bias it only scales the DI contribution inside the trend
family.

### 4.5 Volatility

Wilder ATR(14), ATR as % of price, **trailing percentile of ATR%** over 500 bars (relative, not pip thresholds),
ATR(5)/ATR(50) ratio → `expanding`/`stable`/`contracting`. Regime by percentile: `very_low ≤ 10`, `low ≤ 30`,
`normal < 70`, `high < 92`, `extreme`.

### 4.6 Volatility shock

Five components scaled linearly from a start to an extreme threshold (0–100), all normalised by the *previous*
bar's ATR: range, open gap, largest wick, close-to-close displacement, ATR(5)/ATR(50) expansion.
`shock_severity = max(components)`, `shock_driver` names the triggering component, `volatility_shock =
severity ≥ 50`. Phase 1A does not decide whether trading should be disabled.

### 4.7 Momentum

Components in [-1, 1]: ROC over 3/6/12 bars normalised by `ATR·√n` (tanh), impulse velocity (ATR/bar over 6 bars),
mean signed body/range over 6 bars, consecutive same-direction closes (capped at 5).
`bullish = 100·Σ w·max(c,0)`, `bearish = 100·Σ w·max(−c,0)`, `net = bullish − bearish`. ROC and velocity are
correlated and share the displacement budget (0.35 + 0.20). `momentum_percentile` compares |net| to its 500-bar
history; `momentum_acceleration` compares `net` with its value 3 bars earlier (`accelerating`, `decelerating`,
`steady`, `flat`).

### 4.8 Candle quality

Range, body, body %, upper/lower wick and wick %, range in previous-bar ATR, close location (0 = low, 1 = high),
and a descriptive class: `strong_bullish_close`, `strong_bearish_close`, `bullish_rejection`, `bearish_rejection`,
`indecision`, `weak_bullish_close`, `weak_bearish_close`, plus the `abnormal_expansion` flag. Descriptive only.

### 4.9 Directional efficiency

`|close_t − close_{t−n}| / Σ|close_i − close_{i−1}|` over `lookback = 20`, its signed version and 500-bar percentile.
150 pips net after 700 pips travelled → 0.21; a straight line → 1.0.

### 4.10 Chop score (0–100) and market quality

Weighted components (0–1, higher = choppier): efficiency (0.18), Choppiness Index 38.2→61.8 (0.10), mean candle
overlap (0.12), EMA50 crossings (0.14), flat EMA50 (0.08), compressed EMA20/50 (0.08), weak ADX (0.10),
two-sided structural breaks (0.08), structure-state instability (0.07), small displacement vs ATR·√n (0.05).
Correlated components (efficiency / CI / displacement) have modest individual weights.
`market_quality`: `severe_chop ≥ 72`, `range ≥ 55`, `clean_trend` (chop < 38 and trend strength ≥ 50),
`trend_with_noise` (trend strength ≥ 30), else `transition`.

### 4.11 Support / resistance zones

Maintained **incrementally** (`features/levels.py`, `ZoneBook`). Level sources known at bar `c`: swings accepted by
then with pivot inside the 300-bar lookback (including later-superseded swings), break levels, and the trailing
60-bar range high/low. Sources are single-linkage clustered with tolerance `0.5 × ATR`; zones have a minimum
half-width of `0.1 × ATR`. On each new bar:

* **update** — every zone's sliding-window counters (touch bars, interaction episodes, rejections, side-to-side
  crossings, last interaction) are advanced by one bar in O(1) instead of recounting the lookback;
* **create / update / merge** — a new source within tolerance of an existing zone joins it (the zone keeps its id
  and creation bar; a source bridging two zones merges them); otherwise a new zone is created;
* **age / expire** — sources leave when they fall out of the lookback; a zone that loses all sources expires, one
  whose remaining sources separate is split;
* **invalidate** — a zone crossed `invalidate_after_breaks` (default 4) times inside the lookback is INVALIDATED and
  its sources retired (`0` disables this and restores the Phase 1A behaviour).

Every event (`created`, `updated`, `merged`, `expired`, `invalidated`) is appended to `H4AnalysisResult.zone_events`
with the bar at which it became known; invalidated zones are also listed in `invalidated_zones`. Zones carry
`zone_id`, `created_index`, `bars_since_created` and `age_bars` (bars since their oldest live source).
Zone geometry (tolerance, minimum width) uses the ATR at the zone's last membership change and is then held fixed.
The previous full-rebuild algorithm is kept as `compute_levels_rebuild` (reference implementation); with a constant
ATR and invalidation disabled the incremental engine reproduces it exactly, bar by bar (tested). Per zone: bounds, midpoint, type
(support below price / resistance above; inside → by midpoint), sources, interaction episodes, touch bars, last
interaction and bars since, age, rejections (touch bars that close outside the zone — above it after probing down into it, or below it after probing up), rejection
strength, breaks (closes flipping from one side of the zone to the other), distance and ATR distance, and
`level_strength_score = 30·interactions + 25·rejection + 15·source diversity + 10·swing count + 10·recency
+ 10·(1 − breaks)` (each term capped, total 0–100). Only the `max_zones = 12` strongest are kept.

### 4.12 Context

* **Round numbers**: nearest multiple of each configured JPY interval (1.00 and 0.50), signed distance in pips and
  ATR. Context only — no reversal assumption.
* **Range location**: `(close − low_n) / (high_n − low_n) × 100` for 20/60/180 bars. No directional meaning.
* **Extension**: distances to EMA20/EMA50 in ATR, recent impulse, composite metric `0.6·|d20| + 0.2·|d50|`,
  `extension_score = ½·trailing percentile + ½·absolute (3 ATR = 100)`; `normal`, `extended ≥ 65`,
  `extremely_extended ≥ 85`, with direction.
* **Sessions**: minutes of overlap between the H4 bar interval and Asia (Tokyo 09–18 local), London (08–17
  Europe/London) and New York (08–17 America/New_York), resolved per date with `zoneinfo` so DST is exact;
  primary session, active sessions (≥ 60 min), and pairwise overlaps.
* **Time**: day of week, UTC hour, month, quarter, year (metadata only).

## 5. Regime classification

Ordered rules on one bar's features (first match wins); the result carries `rule`, `evidence` and
`conflicting_evidence` (reason codes):

| # | Condition | Regime |
|---|---|---|
| 0 | warm-up incomplete (< 250 bars) | `UNCLEAR` |
| 1 | `volatility_shock` | `VOLATILITY_SHOCK` |
| 2 | structure and trend aligned, strong trend state, ADX ≥ 20, chop ≤ 45, structure quality ≥ 50 | `STRONG_BULL/BEAR_TREND` |
| 3 | structure and trend aligned, chop < 55 | `BULL/BEAR_TREND` |
| 4 | ATR percentile ≤ 20, ATR(5)/ATR(50) ≤ 0.9, trend not strong | `LOW_VOLATILITY_COMPRESSION` |
| 5 | chop ≥ 55, trend not strong | `HIGH_VOLATILITY_RANGE` (ATR pct ≥ 70) / `RANGE` |
| 6 | one of structure/trend directional, the other neutral (not opposing, not transitional), chop < 65 | `WEAK_BULL/BEAR_TREND` |
| 7 | structure vs trend conflict, transitional structure, or recent CHoCH | `TRANSITION` |
| 8 | chop ≥ 45 → `RANGE`, otherwise | `UNCLEAR` |

Conflicting evidence is regime-aware (e.g. for bullish regimes: bearish momentum, extension up, near strong
resistance, high chop, low efficiency, weak ADX, invalidated (`BREAK_REJECTED`) or failed (`BREAK_FAILED`) break; acceleration/deceleration is resolved using the
momentum sign).

## 6. Directional bias

Families (correlated features blended inside, families weighted — see `classification/bias.py`):

| Family | Weight | Contents |
|---|---|---|
| structure | 0.40 | swing state × (0.5 + 0.5·quality); live CHoCH gives 0.35 in its direction; CONFIRMED/ACCEPTED BOS +0.1; INVALIDATED/FAILED breaks give nothing |
| trend | 0.35 | 0.8 × trend_score + 0.2 × DI spread scaled by ADX (EMA and ADX/DI are one family) |
| momentum | 0.25 | bullish / bearish momentum scores |
| market quality | dampener | × (1 − 0.4 · chop/100), both sides |
| volatility | dampener | × 0.85 extreme volatility, × 0.70 shock, both sides |
| location | dampener | −0.10 extended / −0.20 extremely extended on the stretched side; −0.10 near a strong opposing zone |

`bullish_evidence_score` / `bearish_evidence_score` ∈ [0, 100]. Directional bias requires **all** of:
a directional regime (range, compression, transition and unclear regimes → `NEUTRAL`), evidence agreeing with the
regime direction, not both sides ≥ 35 (`BIAS_CONFLICT`), dominant ≥ 45 (`BIAS_INSUFFICIENT_EVIDENCE`), separation ≥ 20
(`BIAS_INSUFFICIENT_SEPARATION`). `bias_confidence = 100·√(dominant/100 · separation/100)` for LONG/SHORT and 0 for
NEUTRAL. Saturating every trend input cannot push evidence beyond the trend family's share (tested). The engine is
designed to return NEUTRAL often (≈ 70 % of bars on long random-walk data).

## 7. Outputs

* `H4AnalysisResult.features` — flat table, one row per closed bar (≈ 190 columns incl. all components).
* `H4Snapshot` — structured per-bar record with every field listed in the Phase 1A spec, plus the rolling swing
  history, break lifecycle fields, zones (with ids), evidence, family scores, modifiers, `market_data`
  (volume/spread/source, not used by features), warnings, engine version and config hash. `to_dict()` / `to_json()`.
* `H4AnalysisResult.structure` (swings, breaks with transitions), `zone_events`, `invalidated_zones`.
* `write_evaluation_log(path)` — JSONL, one record per evaluation: timestamp, input-data status and flags, all
  feature values, regime, evidence, bias, confidence, reason codes, warnings, config hash.
* `export_features(path)` — `.parquet`, `.csv` or `.jsonl`.
* `research.inspect(result, timestamp)` / `python -m gbpjpy_engine inspect` — human-readable explanation.

## 8. Reason codes

Defined in `reason_codes.py` (`ReasonCode` enum + descriptions): structure (`STRUCTURE_*`, `BOS_*`, `CHOCH_*`,
`BREAK_REJECTED` (invalidated), `BREAK_FAILED`, `HIGH/LOW_STRUCTURE_QUALITY`), trend (`TREND_*`, `EMA_SLOPE_*`, `STRUCTURE_TREND_CONFLICT`),
ADX (`ADX_STRONG/WEAK`, `DI_BULLISH/BEARISH`), momentum, chop/efficiency, volatility (`VOLATILITY_*`), location
(`NEAR_STRONG_SUPPORT/RESISTANCE`, `NEAR_ROUND_NUMBER`, `EXTENDED_*`, `OVEREXTENDED_*`, `RANGE_*_LOCATION`), bias
(`BIAS_*`) and engine (`INSUFFICIENT_HISTORY`, `DATA_QUALITY_WARNING`).

## 9. Parameters

All parameters live in `config.py` (dataclasses; each field has a `doc` string) and are mirrored with comments in
`config/h4_default.yaml`. `python -m gbpjpy_engine config` prints every parameter with its purpose. Values are
baseline defaults, **not optimised**. `H4Config.config_hash()` is written into every snapshot/log record.

## 10. Assumptions

* Input bars are H4 bars whose `timestamp` is the bar open; the engine does not resample.
* 1 pip = 0.01 JPY.
* Tokyo/London/New York session windows are conventional approximations and configurable.
* Warm-up of 250 bars (EMA200 plus percentile history) is required before a regime or bias is issued.
* A bar is evaluated only after it closes.

## 11. Known limitations

* **Not validated on real GBPJPY history yet.** Behavioural tests use deterministic synthetic scenarios, which are
  engineering checks, not evidence of predictive value or profitability.
* Thresholds (chop, regime, bias, strength scores, break lifecycle and zone invalidation) are heuristic baselines;
  weights inside composite scores are judgement-based and not fitted.
* Swing detection is fractal + ATR-amplitude based; confirmation latency is `right_bars` bars (12 h by default), so
  structure always lags price by design. Whether 3 bars is appropriate is left to later research.
* The structure *state* rule still reads the latest swing high/low labels (plus live CHoCH); the richer rolling
  history (16 swings) is exposed for Phase 1B but does not yet feed the state classification beyond the quality
  score.
* Break lifecycle monitoring stops after `break_monitor_bars` (60 bars); a break's state is frozen after that.
  Failure uses a close-based threshold; intrabar spikes back through a level do not fail a break.
* Incremental zones fix their geometry at the last membership change; zones therefore differ from a
  current-ATR rebuild when volatility changes (on the 6,000-bar benchmark, geometry alone changed the nearest
  support/resistance on ~17–19 % of bars). Zone invalidation (default 4 crossings) is a new, deliberate behaviour
  change and removes heavily-crossed zones that previously remained listed with low strength.
* Percentile-based features need ~100–500 bars of history; in very persistent trends the extension percentile can
  stay high for long periods.
* Runtime ~0.7 ms/bar (6,000 bars ≈ 4.3 s in this environment); the swing/structure forward walk and pandas
  feature assembly now dominate.
* Tick volume and spread are validated and preserved but not used in features; tick volume is broker-specific.
* Broker data in server time with DST shifts produces H4 grids that move relative to UTC; this is flagged
  (`TIMEZONE_ALIGNMENT_SHIFT`), not corrected. Adapters must supply UTC canonical bars.
* No MT4/MT5 adapter exists yet; only the broker-neutral `BarSource` protocol is defined.
* No economic-calendar/news awareness; shocks are detected only after the bar closes.

## 12. Change log — Phase 1A.1 (hardening)

* Structural memory: rolling history of `swing_history_size` confirmed swings with occurrence and confirmation
  timestamps, classification and significance.
* Break lifecycle CANDIDATE / CONFIRMED / ACCEPTED / FAILED / INVALIDATED with append-only, point-in-time transitions
  (replaces `pending` / `confirmed` / `rejected`; mapping: pending→CANDIDATE, rejected→INVALIDATED,
  confirmed→CONFIRMED and may later become ACCEPTED or FAILED).
* Incremental zone engine with create/update/merge/expire/invalidate events; reference rebuild retained for
  equivalence testing.
* Explicit `occurred_at` vs `confirmed_at` everywhere (`last_swing_*_time` columns renamed to
  `last_swing_*_occurred_at`).
* Broker-neutral `BarSource` protocol and `assert_canonical`; documentation of adapter timestamp normalisation and
  tick-volume semantics; platform-isolation tests.
