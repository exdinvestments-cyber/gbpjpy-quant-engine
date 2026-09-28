# Phase 1D — Entry Intelligence Engine

```
H4 MARKET INTELLIGENCE -> H4 CONTEXT -> H4 DIRECTIONAL PERMISSION -> H1 SETUP INTELLIGENCE -> QUALIFIED SETUP
    -> ENTRY INTELLIGENCE (Phase 1D) -> EXECUTABLE ENTRY CANDIDATE
```

An **EXECUTABLE ENTRY CANDIDATE is not an order.** Phase 1D contains no position size, risk amount, stop loss,
take profit, order, ticket, broker connection or account access. Nothing has been validated as profitable; every
threshold is an untested engineering baseline (`config/entry_default.yaml`) and none was tuned against profit.

Package: `src/gbpjpy_engine/entry/`

| Module | Purpose |
|---|---|
| `config.py` | documented parameters, family-specific confirmation policies, validation, YAML loader |
| `interfaces.py` | broker-neutral protocols: `QuoteSource` (+ historical `BarOpenQuotes`), `SlippageModel`, `NewsProvider`, `LowerTimeframeProvider`, `IntrabarConfirmationProvider` |
| `confirmation.py` | six confirmation families + confirmation confluence |
| `execution.py` | spread, bid/ask executable reference, slippage hook, gap, price deterioration, abnormality, stacked barriers, session, news |
| `scoring.py` | freshness, chase risk, entry extension, entry market quality, entry conflict, entry quality |
| `lifecycle.py` | `EntryCandidate` state machine (append-only transitions and audit trail) |
| `engine.py` | `EntryIntelligenceEngine.run(h1_result, h4_result, quotes=None, news=None, slippage_model=None, intrabar=None)` |
| `report.py` | `explain_entry_candidate(result, id)` — human-readable audit |

## 1. How Phase 1D consumes Phase 1C

* The input is an unchanged Phase 1C `H1SetupResult` and the H4 result it was aligned to. Phase 1D never
  writes back into either (tested).
* Each Phase 1C setup that reaches QUALIFIED creates **one** entry candidate `E-{setup_id}` at its
  qualification bar (`WAITING_FOR_CONFIRMATION`). Setup state is read point-in-time with `Setup.state_at(c)`.
* Reused: Phase 1A break lifecycle (`BreakEvent.state_at`), swings, zones; Phase 1B multi-candle displacement
  (`displacement_features`) and room scale (`RoomConfig`); Phase 1C sweeps, reclaims, expansions, pullback
  context, H4-aligned frame (`h4_permission`, `h4_context_status`, `h4_index`) and H1 blockers; the H4 context
  room barriers (`context.details[k]["room"]`). Nothing in Phases 1A–1C was changed for Phase 1D.
* **One active setup per side**: Phase 1C tracks one active setup per direction, so Phase 1D tracks at most one
  candidate per side (`lifecycle.max_active_per_side` must be 1). Candidates are keyed by id so the tracker can
  support several simultaneous candidates later if validation justifies it.

## 2. Closed-H1 decision model and timing

* **Confirmation** is judged at the CLOSE of H1 bar `c`, from completed bars only.
* **Intrabar sequencing is never assumed.** When a completed H1 candle contains both a level interaction and a
  subsequent move, the order of those events is unknown; every family reasons about closes, extremes and bodies
  of completed bars. No intrabar path is reconstructed.
* **The decision** (accept / defer / reject / expire / invalidate) is taken at the first price observable after
  that close — historically the OPEN of bar `c+1` (`BarOpenQuotes`). It uses information known at the close of
  `c` plus that single price and the spread known at that moment. The high, low, close and spread report of bar
  `c+1` are never used for that decision (tested by rewriting them).
* Transitions carry a phase: `OPEN` (decided at the first executable price) or `CLOSE` (decided at a bar close).
  `state_at(i)` replays history exactly.
* **Live use**: a `QuoteSource` that already provides the first price after the final close produces the same
  decision the historical replay produces (tested).
* **Lower-timeframe extension**: `LowerTimeframeProvider` / `IntrabarConfirmationProvider` can supply M15/M5/M1
  or tick evidence. Phase 1D does not depend on them; when supplied, their observation is attached to the
  confirmation as research evidence only and does not change decisions (tested).

## 3. State machine

`NO_ENTRY -> WAITING_FOR_CONFIRMATION -> CONFIRMING -> ENTRY_CANDIDATE`, with `DEFERRED`, `REJECTED`, `EXPIRED`
and `INVALIDATED`.

| From | To | When |
|---|---|---|
| (qualified) | WAITING_FOR_CONFIRMATION | Phase 1C setup QUALIFIED |
| WAITING | CONFIRMING | an allowed family reaches `confirming_score` (partial) or the policy is met (complete) |
| CONFIRMING | WAITING | partial evidence faded |
| CONFIRMING/DEFERRED | ENTRY_CANDIDATE | all checks pass at the first executable price |
| CONFIRMING/DEFERRED | DEFERRED | gap against the thesis, abnormal conditions, late-Friday cutoff, market closure |
| DEFERRED | WAITING | a market closure intervened: the pre-closure confirmation is discarded |
| any active | REJECTED | spread, chase, extension, room, staleness, deterioration, gap, abnormality, conflict, quality, news (configured) |
| any active | EXPIRED | no confirmation within `max_wait_bars`, price ran without confirmation, window elapsed, Phase 1C setup expired |
| any active | INVALIDATED | H4 permission revoked/BLOCK_ALL/stale, Phase 1C setup invalidated, H1 hard blocker, opposing structure break, invalidation reference lost, confirmation level lost |
| ENTRY_CANDIDATE | EXPIRED / INVALIDATED | availability window elapsed or price moved away / structure failed (the accepted record is kept) |

**Duplicate protection**: one candidate per setup (`E-{setup_id}`, stable and deterministic); at most one
acceptance per candidate.

## 4. Confirmation families

Each family is scored 0–100 on every bar for both sides and stored independently (`{side}_conf_*_score`
columns), so later validation can test them one by one. None is assumed to be as useful as another. (While a
candidate is active, the structural-break score only counts breaks after that candidate's qualification bar;
otherwise any break of the last `structural_lookback_bars` bars is scored.)

| Family | Evidence (completed bars) |
|---|---|
| STRUCTURAL_BREAK_CONFIRMATION | a Phase 1A break (close beyond a confirmed swing by >= 0.25 ATR) after qualification: level importance (swing significance + zone strength), close beyond, penetration, body, close location, displacement, follow-through, lifecycle; optional minimum lifecycle state |
| DISPLACEMENT_CONFIRMATION | Phase 1B multi-candle displacement; an isolated giant candle (no multi-bar support, range >= 2.5 ATR) is multiplied by 0.6 |
| BREAK_RETEST_CONFIRMATION | confirmed/accepted break, return within tolerance, controlled penetration, directional response; a deep close back through = BREAKOUT_FAILURE |
| SWEEP_RECLAIM_CONFIRMATION | rejected sweep, then a close beyond the sweep candle (or a Phase 1C reclaim) plus displacement or structural consequence; a sweep alone never confirms |
| MOMENTUM_REACCELERATION_CONFIRMATION | counter-momentum deterioration, resumed and rising permitted-direction momentum, body progression, directional closes, structural progress, efficiency |
| COMPRESSION_EXPANSION_CONFIRMATION | Phase 1B expansion-from-compression event: range expansion, body, direction, structural consequence, H4 confidence; expansion >= 3 ATR capped at 40 (chasing) |

**Confirmation confluence** (`confirmation_quality_score`): `0.6 x primary family + 0.4 x support`, where support is
the weighted mean of the OTHER evidence groups (STRUCTURE, DISPLACEMENT, RECLAIM_RETEST, MOMENTUM,
MARKET_QUALITY), taking the maximum within a group and excluding the primary's own group — correlated evidence is
never counted twice (tested).

**Family-specific policies** (`policy.policies`): allowed confirmation families, minimum family score and minimum
confirmation quality per Phase 1C setup family — e.g. a trend pullback may confirm by structural break,
displacement or reacceleration; a liquidity-sweep reversal only by sweep+reclaim or structural break. Not
optimised.

## 5. Prices: signal vs executable, bid/ask, spread, slippage

* `signal_price` = the confirmation bar's close on the chart basis. It is a **reference**, not a fill: nobody can
  trade at a candle close after observing it.
* `executable_reference_price` = the first price logically available after confirmation (historically the next
  H1 open), moved to the side of the book the trade would use: **LONG buys the ASK, SHORT sells the BID**
  (`execution.price_basis`: `bid` default for typical retail FX charts, `mid` or `ask`). Stops/closures would use
  the opposite side; no stop or close exists in Phase 1D.
* **Spread** used at the decision is the spread KNOWN at that moment — the report of the bar that just closed
  (`BarOpenQuotes`) or a live quote. Units are declared (`execution.spread_unit`: pips / points / price), never
  guessed. Status NORMAL / ELEVATED / HIGH / EXTREME = worst of an absolute (pips), ATR-relative and
  distribution (vs trailing median) test; `UNKNOWN` when missing.
* **Missing spread is never treated as zero**: status UNKNOWN stays visible (`SPREAD_UNKNOWN`), the executable
  reference uses an explicitly flagged conservative assumption (`spread_assumed: true`), and UNKNOWN is blocked
  only if `spread.block_unknown` is set. HIGH/EXTREME are blocked by default (`SPREAD_TOO_HIGH`).
* **Slippage** is UNKNOWN by default (`SLIPPAGE_UNKNOWN`). `FixedSlippage`, `SpreadDependentSlippage`,
  `VolatilityDependentSlippage` and `EmpiricalSlippage` apply only numbers a researcher supplies; the estimate is
  reported separately and never folded into the executable reference.

## 6. Timing, chase and extension

* **Freshness** (0–100; FRESH / AGING / STALE / EXPIRED): bars since confirmation and qualification, directional
  move since each (ATR), wall-clock hours since confirmation. STALE rejects; EXPIRED expires.
* **Chase risk** (0–100; LOW / MODERATE / HIGH / EXTREME): distance from the confirmation reference and from the
  setup location (ATR), share of the reference impulse already travelled, proximity to the nearest opposing
  barrier cluster. HIGH and above reject by default (`chase.reject_at`).
* **Entry extension** (0–100; NOT_EXTENDED / EXTENDED / OVEREXTENDED): distance from the H1 EMA baseline and from
  the last opposite structural swing, recent 10-bar impulse, Phase 1A extension percentile, aligned H4 extension.
  Used for timing only; extension is not assumed to predict reversal. OVEREXTENDED rejects.

## 7. Execution conditions

* **Gap** = next price vs the last close (ATR): NONE / SMALL / LARGE / EXTREME. EXTREME rejects; LARGE against the
  thesis defers (structure re-checked at the next close).
* **Price deterioration** = adverse difference between the executable reference and the signal price
  (`price_deterioration_pips`, `_atr`, `_score` with 0 = none, 100 = severe). Entry window AVAILABLE
  (<= 0.25 ATR) / DETERIORATING / EXPIRED (>= 0.6 ATR, rejected).
* **Entry window**: at most `window_bars` executable opportunities and `window_max_hours` after confirmation;
  an accepted candidate stays available for `candidate_valid_bars` and expires earlier if price moves away.
* **Barrier stacking**: H1 zones (strength >= 50), H1 confirmed swings and the aligned H4 room barriers beyond the
  executable reference are clustered (0.5 ATR) so overlapping H1/H4 levels count once; the nearest five clusters
  are stored with type, price, strength, timeframe, distance and ATR distance. `remaining_room_score` uses the
  Phase 1B room scale; `barrier_density_score` weights clusters by strength, proximity and timeframe confluence.
* **Entry market quality**: execution-time chop, directional efficiency, volatility regime and gap (confirmation,
  spread and deterioration are separate families, so they are not counted again).
* **Abnormal movement** (0–100, max of components): extreme candle, volatility expansion, gap, spread explosion,
  Phase 1A shock severity. >= 60 defers, >= 85 rejects.
* **News**: `NewsProvider` returns scheduled GBP/JPY (or GLOBAL) events with `known_since`, so a schedule published
  later cannot leak backwards; outcomes (actual/forecast) are not part of the contract. Without a provider
  `news_status = UNKNOWN`, which is recorded and not blocked unless `news.block_unknown` is set. No event is
  ever invented.
* **Session** at the execution time: active sessions and overlaps (Asia / London / New York, DST-aware via
  zoneinfo) or `off_peak`. Descriptive only.

## 8. Weekend / market reopen

* A closure (> 2 h between a bar close and the next price) between confirmation and the first executable price
  DEFERS the candidate and discards the confirmation (`WEEKEND_REOPEN_REVALIDATION`); a post-reopen confirmation is
  required.
* The first completed bar after any closure cannot confirm (`gap.reopen_revalidation_bars`); a setup qualified
  before a closure is flagged `weekend_carryover` and needs post-reopen confirmation.
* Confirmations closing on Friday at/after `friday_cutoff_hour_utc` are deferred, never accepted into the weekend.
* With default Phase 1C settings a setup rarely survives a weekend at all (stale H4 context after the reopen
  invalidates it); the Phase 1D protection is tested on an edited setup that does.

## 9. Conflict, quality, decision

* `entry_conflict_score` (noisy-OR): momentum deteriorating, price extended, room collapsed, poor spread, abnormal
  volatility, H4 context weakened since qualification, opposing barrier cluster.
* `entry_quality_score`: weighted families CONFIRMATION, TIMING, FRESHNESS, PRICE_QUALITY, MARKET_QUALITY,
  STRUCTURAL_INTEGRITY, ROOM, EXECUTION_CONDITIONS, CONFLICT, reduced by conflict. **Not a probability and never a
  position-sizing input.**
* Decision order at an executable opportunity: market closure -> window / freshness expiry -> hard rejects (all
  reasons recorded) -> Friday cutoff -> deferrals -> ACCEPT_ENTRY_CANDIDATE.
* Structural revalidation happens at every close: Phase 1C setup state, H4 permission and context status, H1
  hard blockers, opposing H1 break, setup invalidation reference, confirmation level. A qualified setup has no
  permanent permission; stale setups are not grandfathered.

## 10. Outputs, audit and counterfactuals

* `EntryResult.frame`: one row per H1 bar — entry state and id per side, all six family scores per side,
  confirmation family/quality, window status, events, reason codes, reopen flags, spread.
* `EntryResult.accepted`: accepted candidate objects (ids, timestamps, direction, families, `signal_price`,
  `executable_reference_price`, execution side, spread status/pips, slippage status, price deterioration,
  chase, extension, freshness, confirmation quality, entry market quality, remaining room, barrier density,
  conflict, quality, H4 permission, H1 setup score, session context, news status, reason codes). No size, risk
  amount, stop loss, take profit or ticket.
* `EntryCandidate.audit`: every step (SETUP QUALIFIED -> WAITING FOR CONFIRMATION -> e.g. BULLISH STRUCTURAL BREAK ->
  SPREAD CHECK PASSED -> ROOM CHECK PASSED -> CHASE CHECK PASSED -> ENTRY CANDIDATE ACCEPTED); rejected candidates
  carry every failed check. `explain_entry_candidate()` renders it.
* **Counterfactual log**: every candidate that was not accepted (rejected, expired or invalidated, confirmed or
  not) with its category, reason, end code and — when confirmed — the hypothetical executable reference and scores,
  so later validation can test whether each filter helps. Filters must not be retro-fitted to individual rejected
  winners.

## 11. Historical execution realism (OHLC-only research)

What cannot be known from H1 OHLC and is therefore **not assumed**:

* the order of events inside a completed candle (intrabar sequence);
* a fill at the candle close (the signal price is a reference only);
* zero latency: the next open is the earliest honest proxy, not a guaranteed fill;
* zero spread: the spread known at the decision is used; missing spread is UNKNOWN (conservative assumption, flagged);
* zero slippage: slippage is reported UNKNOWN unless a researcher supplies a model;
* the spread during the next bar (its report aggregates the whole hour and is future information);
* queue position, requotes, partial fills, freeze/stop levels.

Future lower-timeframe replay plugs in through `LowerTimeframeProvider` / `IntrabarConfirmationProvider` and a
tick-based `QuoteSource`.

## 12. Portability

The package imports only the standard library, numpy and pandas (plus PyYAML for configuration). All broker
contact goes through the protocols in `entry/interfaces.py`, implemented outside the core by future MT4/MT5
adapters (see `docs/BROKER_NEUTRAL_INTERFACES.md`). Tests forbid platform SDKs and order / sizing / stop / target
identifiers in the package.

## 13. Known limitations

* Validated only on deterministic synthetic data; no threshold, weight, band or policy has been tested against
  outcomes.
* The Phase 1C synthetic H1 series has an H1 ATR of ~15 pips, so its 1.5–3.5 pip spreads are HIGH relative to ATR
  and most candidates on it are spread-rejected; Phase 1D engineering scenarios (`synthetic_entry.py`) use ATR ~28
  pips.
* On the synthetic scenarios MOMENTUM_REACCELERATION is the most permissive family and confirms almost every
  accepted candidate; compression -> expansion never occurs end-to-end (unit-tested only). Their relative
  usefulness is unknown.
* The historical executable reference is the next H1 open: real first-tradable quotes, latency and intrabar spread
  are unknown; bar spread reports are coarse.
* Slippage is unknown; news status is UNKNOWN (no provider).
* One active candidate per side (inherited from Phase 1C).
* Qualified Phase 1C setups often last only a few bars, which limits how many candidates can develop.
* Weekend carry-over is protected but rarely exercised naturally (Phase 1C already invalidates most setups across
  weekends).
* The candidate availability window after acceptance uses completed H1 closes; availability inside the hour is
  unknown.
