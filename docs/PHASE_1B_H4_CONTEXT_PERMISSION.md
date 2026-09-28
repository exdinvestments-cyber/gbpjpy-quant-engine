# Phase 1B — GBPJPY H4 Context & Directional Permission Engine

**Scope.** Turns the Phase 1A/1A.1 H4 intelligence into structural context and a *directional permission*
(`ALLOW_LONG`, `ALLOW_SHORT`, `ALLOW_BOTH`, `BLOCK_ALL`) that states what a future H1 engine may **search for**.

**Not in scope.** H1 entries, trade signals, position sizing, execution, broker connectivity, optimisation. The
permission is not a trade signal and `permission_confidence` must never be used for sizing. Nothing here has been
validated for profitability; no performance claim is made.

## 1. Integration with the protected foundation

```
bars -> Phase 1A features (unchanged) -> structure (swings, break lifecycle) -> ZoneBook (zones)
                                                     |
                                   Phase 1B ContextEngine (src/gbpjpy_engine/context/)
                                                     |
                           result.context  (frame + per-bar details + event histories)
                           H4Snapshot (additive fields), JSONL log ("context" block), research view
```

* The context layer only **consumes** Phase 1A/1A.1 outputs: the feature row, the swing records
  (`accepted_index`/`removed_index`), break lifecycles via `transitions_known_at(c)`, and `zones_by_bar[c]`. It
  never re-detects swings, recomputes indicators or edits any Phase 1A value; `h4_bias`, `regime` and the feature
  table are untouched (tested).
* The only change to a Phase 1A.1 interface is the additive `Zone.formation_atr` field (ATR at which a zone's
  geometry was fixed), required for dynamic zone context.
* Run automatically by `H4MarketIntelligenceEngine.run`; `ContextEngine(cfg).run(result)` can also be run
  separately.

## 2. Point-in-time rules

At bar `c` the engine reads only information available at the close of `c`. Stateful components (liquidity sweeps,
origin zones, role reversal, expansion, state machine) use append-only transition histories stamped with the bar at
which each change became known; historical states are never rewritten. Tests prove that truncating the data at `T`,
or replacing everything after `T`, leaves every context row, every per-bar detail, every event history prefix and
every snapshot up to `T` identical (`tests/test_context_lookahead.py`), and a canary proves the harness catches
leakage.

## 3. Components

### 3.1 Full swing-history interpretation and hierarchy (`context/hierarchy.py`)

`SwingMemoryTracker` replays the Phase 1A.1 rolling memory (verified equal to `history_at`). Three windows are
classified with one rule set:

| Layer | Window |
|---|---|
| PRIMARY | last 16 swings (`hierarchy.primary_swings`) |
| INTERMEDIATE | last 8 swings |
| IMMEDIATE | last 4 swings + current close (close beyond the last swing high/low by 0.10 ATR = provisional HH/LL) |

Measures: recency-weighted label consistency (linear weights, newest = 2 × oldest — documented, not exponential,
not fitted), net progression of highs and lows (ATR), swing-amplitude expansion/contraction, swing duration,
correction depth, violations. Classes: RANGING, BULLISH, BEARISH, TRANSITIONAL, NEUTRAL, UNCLEAR, each with a
0–100 confidence. PRIMARY and INTERMEDIATE become TRANSITIONAL when the close breaks the *protected* swing (the
opposite swing that launched the latest extreme) by 0.25 ATR — a few bearish candles cannot flip a bullish primary
structure. Evidence returned per layer: swing ids, labels, weights, consistency, progression, violations.

### 3.2 Structural legs, impulse vs correction (`context/legs.py`)

A leg joins two consecutive confirmed swings (the active leg runs from the last swing to the current close). Stored
per leg: start/end time, direction, price and pip distance, duration, ATR distance, average body and range (ATR),
directional efficiency, candle overlap, momentum (mean signed body/range), velocity, maximum adverse and favourable
excursion, `impulse_score`, nature and classification. `impulse_score` blends six independent views (efficiency
0.25, distance 0.20, low overlap 0.15, body dominance 0.15, velocity 0.15, small adverse excursion 0.10).
Classification relative to the prevailing structure: STRONG/NORMAL/WEAK_IMPULSE, FAILED_IMPULSE (end swing failed to
exceed the prior extreme), HEALTHY/DEEP/CHOPPY_CORRECTION, UNCLEAR.

### 3.3 Displacement (`context/displacement.py`)

Multi-candle score (bullish and bearish, 0–100) from body/ATR, close location, range (only with a strong close),
net 3-bar move, window efficiency, low overlap, directional closes, same-direction structural break, and historical
percentile. STRONG/EXTREME additionally require ≥1.5 ATR net move, ≥2 directional closes and efficiency ≥0.5 — a
single large candle is capped below STRONG. Normalisation uses ATR from before the window. Classes NONE, WEAK,
MODERATE, STRONG, EXTREME.

### 3.4 Breakout lifecycle integration, quality, acceptance, failure, false-break risk (`context/breakouts.py`)

Reuses the Phase 1A.1 lifecycle (CANDIDATE → CONFIRMED → ACCEPTED / FAILED, CANDIDATE → INVALIDATED; monitoring
horizon remains `structure.break_monitor_bars`, configurable). At each bar, the latest break within that horizon is
evaluated from its transitions **as known at that bar** (break, confirmation, acceptance, failure, invalidation
timestamps, current and previous state, reason) and closes up to that bar (maximum excursion is recomputed; the
event's final value is never used).

* `breakout_quality_score`: level importance (swing significance + zone strength before the break), close beyond,
  body strength, close location, displacement, efficiency, follow-through, lifecycle support → HIGH_QUALITY_BREAK /
  MODERATE_BREAK / WEAK_BREAK / FALSE_BREAK_CANDIDATE.
* `acceptance_state`: ACCEPTING / PARTIALLY_ACCEPTING / UNRESOLVED / REJECTING from acceptance evidence (multiple
  closes beyond, successful retest, new structure beyond the level, continued displacement, lifecycle ACCEPTED) vs
  rejection evidence (close back inside, opposing displacement, rapid reversal, failed follow-through, lifecycle
  FAILED/INVALIDATED). Never decided on the break bar itself.
* `failure_score`: return depth, short time beyond the level, small excursion, close inside, opposing displacement,
  structural consequence (opposite break afterwards). Each failure is recorded once, at the bar it became known
  (`result.context.failed_breaks`). A failed break is context, never a reversal signal.
* `false_break_risk_score`: weak penetration, poor close, rejection wick, weak displacement, poor follow-through,
  high chop, low efficiency, strong opposing zone within 1 ATR, rapid return toward the level; every component ≥0.6 is
  listed as a reason.

**Intrabar limitation / extension point.** All of the above uses completed H4 closes. `BreakoutAnalyzer.evaluate`
accepts optional `IntrabarEvidence`; `data.interfaces.IntrabarProvider` defines how a future lower-timeframe/tick
provider supplies it. Nothing intrabar is assumed or fabricated today.

### 3.5 Liquidity references and sweeps (`context/liquidity.py`)

References are price-action only and labelled `POTENTIAL_LIQUIDITY_REFERENCE`: SWING_LIQUIDITY (confirmed swings,
known from confirmation), STRUCTURAL_LIQUIDITY_REFERENCE (previous break levels), clustered with volatility-aware
tolerances into EQUAL_HIGHS/LOWS (0.15 ATR) and NEAR_EQUAL_HIGHS/LOWS (0.35 ATR); the outermost cluster is flagged as
the structural extreme. No claim is made about actual resting orders. When a bar trades through references a sweep
event starts UNRESOLVED and evolves: SWEEP_AND_REJECT (all closes back inside for 3 bars) or BREAK_AND_ACCEPT (close
beyond by 0.25 ATR from the next bar; also "later reclaimed"). Scores (penetration, wick, close position, rejection,
reference importance, confirmation, structural effect) give `bullish_sweep_score` (downside liquidity swept and
rejected) and `bearish_sweep_score`.

### 3.6 Zones (`context/zones.py`)

* **Displacement-origin zones**: when STRONG displacement first appears, the zone is the high–low envelope of the 2
  bars preceding the displacement window (demand for bullish, supply for bearish), created at the displacement bar's
  close, boundaries fixed. Stored: boundaries, creation time, direction, origin structure, displacement strength,
  freshness, revisits, penetration depth, reaction strength, invalidation status (close through the far side).
* **Freshness**: FRESH / LIGHTLY_TESTED / TESTED / HEAVILY_TESTED / INVALIDATED — descriptive only.
* **Role reversal** on the Phase 1A.1 zones: ORIGINAL → BROKEN → ACCEPTED (2 closes beyond) → RETESTED → FLIPPED
  (0.5 ATR reaction), FAILED_FLIP on a close back on the original side. Crossing alone never flips a role.
* **Dynamic zone context**: original boundaries are preserved; current ATR, width/current ATR, distance/current ATR
  and current ATR / formation ATR are reported separately.

### 3.7 Location and retracement (`context/location.py`)

Premium/discount on the latest confirmed swing range (≥1.5 ATR): DEEP_DISCOUNT < 20 ≤ DISCOUNT < 45 ≤ EQUILIBRIUM ≤
55 < PREMIUM ≤ 80 < DEEP_PREMIUM (raw percentile and inside/above/below kept). Retracement of the active impulse
(last with-structure leg, or the developing with-structure active leg): depth %, current %, ATR distance, duration,
velocity, efficiency; SHALLOW < 30 ≤ NORMAL < 55 ≤ DEEP < 80 ≤ VERY_DEEP. Descriptive bands, no Fibonacci claim; no
"discount = buy".

### 3.8 Maturity, deterioration, compression, expansion (`context/maturity.py`)

* **Trend maturity**: impulses, corrections, distance from the trend origin (ATR), duration, extension, deterioration
  → EARLY / DEVELOPING / MATURE / EXTENDED / EXHAUSTION_RISK / UNKNOWN. Never a counter-trend signal.
* **Momentum deterioration** (0–100): smaller impulses, deeper corrections, slower velocity, weaker impulse quality,
  lower efficiency, more overlap, failed impulses, failed/invalidated breaks in the trend direction.
* **Compression** (0–100): swing-amplitude contraction, falling ATR ratio, contracting ranges, overlap, converging
  swings, low efficiency. No directional implication.
* **Expansion after compression**: episode of ≥4 compression bars, then (during it or within 3 bars) a break or
  MODERATE+ displacement with bar range ≥1.5 × the episode's mean range → event with compression duration, direction,
  displacement, structural break; EXPANSION_ACCEPTED / EXPANSION_FAILED from the linked break, EXPANSION_FADED if
  nothing is accepted within the window.

### 3.9 Room to move (`context/room.py`)

ATR distance to the nearest opposing barrier: strong zones (distance 0 when inside a resistance/support-type zone),
confirmed swing highs/lows in memory, valid supply/demand origin zones, trailing range extreme. Barriers formed
entirely inside the current active leg (the move's own running extreme) are excluded. `room_score` = 0 at 0.5 ATR,
100 at 4 ATR, 100 with no barrier (`NO_BARRIER_FOUND`). No R:R (no entry/stop exists).

### 3.10 Conflict, quality, LONG/SHORT scores (`context/permission.py`)

* **Conflict** (noisy-OR of documented weights): primary vs strong opposing displacement (0.35), opposing level
  immediately ahead (0.25), failed break in the structural direction (0.30), extreme extension (0.20), strong
  structure during a shock (0.35), primary vs intermediate disagreement (0.20), structure vs EMA trend engine (0.20),
  opposing sweep (0.15), high false-break risk on the structural break (0.15).
* **Quality**: eight families, one score each (STRUCTURE 0.20, DISPLACEMENT 0.10, MOMENTUM 0.10, VOLATILITY 0.15,
  LOCATION 0.10, LIQUIDITY_CONTEXT 0.10, ROOM_TO_MOVE 0.10, MARKET_QUALITY 0.15). Correlated inputs share a family.
* **LONG / SHORT scores**: families structure 0.35, continuation 0.15, displacement 0.15, level behaviour 0.10,
  liquidity 0.10, room 0.15; multiplicative penalties for insufficient room, extreme extension, recent failed break,
  transition against the side, severe chop, conflict, extreme volatility and shock. Same rules for both sides; the
  two scores are always stored separately (no assumption that up- and down-moves behave identically).

### 3.11 Hard blockers and permission

Hard blockers (configurable): INVALID_DATA, STALE_DATA (latest bar, when `as_of` is supplied), INSUFFICIENT_HISTORY,
UNRESOLVED_DATA_GAP (6 bars after a gap/grid-shift flag), EXTREME_VOLATILITY_SHOCK (severity ≥85),
SEVERE_CHOP, UNCLASSIFIABLE_STRUCTURE, CONTEXT_ERROR. Any blocker → `BLOCK_ALL` with `BLOCKER_*` codes.

Otherwise a side qualifies when its score ≥55 (≥65 in TRANSITION/UNCLEAR regimes), its room ≥35, conflict ≤55 and
quality ≥45. Both qualify → the stronger if ≥15 points apart, else `ALLOW_BOTH` in RANGE/HIGH_VOLATILITY_RANGE and
`BLOCK_ALL` (TWO_WAY_CONTEXT_OUTSIDE_RANGE) elsewhere. Nothing qualifies → `BLOCK_ALL` with the failed requirements.
`permission_confidence` = geometric mean of score, quality, (100 − conflict) and room for the permitted side; 0 for
BLOCK_ALL. Requirements never loosen with time or inactivity.

**Fail-safe**: any exception during a bar's evaluation sets CONTEXT_ERROR (BLOCK_ALL) for that bar and all later
bars of the run; the error is logged in `result.context.errors`.

### 3.12 State machine and temporal stability (`context/state_machine.py`)

States: TREND, PULLBACK, DEEP_PULLBACK, CONTINUATION_ATTEMPT, BREAK, ACCEPTANCE, FAILURE, TRANSITION,
REVERSAL_ATTEMPT, COMPRESSION, EXPANSION, RANGE, BLOCKED, UNDEFINED. Expected transitions are declared (e.g.
TREND → PULLBACK → CONTINUATION_ATTEMPT → BREAK → ACCEPTANCE; TREND → DEEP_PULLBACK → TRANSITION → REVERSAL_ATTEMPT;
COMPRESSION → BREAK/EXPANSION → ACCEPTANCE/FAILURE). Unexpected transitions are applied (never hidden) and logged in
`result.context.unexpected_transitions`. Output: current and previous state, bars in state, state-change time,
transition validity and the number of changes in the last 20 bars. No artificial confirmation lag is added.

## 4. Outputs and auditability

* `result.context.frame` — one row per bar (research storage: structure, legs, displacement, breakout quality,
  acceptance, failure, sweeps, zones, maturity, compression, room, conflict, quality, scores, permission, state).
  `result.export_context(path)` writes Parquet/CSV/JSONL.
* `result.context.details[i]` — full per-bar evidence: hierarchy evidence, last 8 legs, retracement, premium/discount,
  breakout context, liquidity clusters and latest sweep, origin zones, S/R dynamic context, role reversal, maturity,
  deterioration and compression components, expansion, room barriers, conflict hits, quality families, LONG/SHORT
  breakdowns, decision, state, explanation.
* Event histories: `legs`, `sweep_events`, `origin_zones`, `role_reversals`, `failed_breaks`, `expansions`,
  `unexpected_transitions`, `errors`.
* `result.explain_permission(timestamp)` / `research.explain_permission` / `python -m gbpjpy_engine inspect --why`
  answer "why was H1 allowed / not allowed to search here?" with the decision, confidence, supporting evidence,
  opposing evidence, blockers, unmet requirements and all permission reason codes.
* The H4 snapshot carries every Phase 1B field from the specification; the JSONL log adds a `context` block.

## 5. Parameters

Sections `hierarchy`, `legs`, `displacement`, `breakout_context`, `liquidity`, `origin_zones`, `role_reversal`,
`location_context`, `maturity`, `compression`, `room`, `context_scoring`, `permission`, `blockers`,
`context_state` in `config.py` / `config/h4_default.yaml`, each field documented. Baseline values, **not
optimised**.

## 6. Known limitations

* Not validated on real GBPJPY history; all thresholds and weights are judgement-based baselines.
* The primary layer and maturity are bounded by the 16-swing memory; very long trends are only partly visible.
* Swing-based layers inherit the 3-bar confirmation lag; only the immediate layer reacts to the current close.
* Breakout analysis evaluates the most recent break within the monitoring horizon; older concurrent breaks are only
  reflected through failed-break records and conflict flags.
* Role reversal is tracked only while a zone is in the emitted zone map (top `max_zones`).
* Liquidity references are inferred from price; nothing is known about actual orders.
* Completed-bar evaluation only — intrabar sequencing is unknown until an `IntrabarProvider` exists.
* `ALLOW_BOTH` requires a range regime and two independently strong sides; it is rare by construction.
* Runtime ~2.9 ms/bar including Phase 1A (6,000 bars ≈ 17 s in the development container; Phase 1A alone ≈ 3.8 s).
  The per-bar Python walk dominates: hierarchy ~15 %, breakout evaluation ~10 %, active-leg metrics ~9 %,
  liquidity clustering ~6 %; no single hotspot. No shortcut that could compromise point-in-time correctness was taken.
* No news/calendar blocker yet.
