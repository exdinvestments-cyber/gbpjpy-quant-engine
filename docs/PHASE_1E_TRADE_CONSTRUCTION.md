# Phase 1E — Trade Construction Engine

```
... -> ENTRY INTELLIGENCE -> EXECUTABLE ENTRY CANDIDATE (Phase 1D)
    -> TRADE CONSTRUCTION (Phase 1E) -> PROPOSED TRADE
    -> [future: account risk engine -> execution contract -> MT4/MT5 adapter]
```

**A PROPOSED TRADE IS NOT AN ORDER.** Phase 1E outputs prices, distances and R relationships only: no volume, lot
size, currency risk, percentage risk, leverage, order or broker contact. Every threshold is an untested baseline
(`config/trade_default.yaml`); nothing was fitted to profit.

Package `src/gbpjpy_engine/trade/`:

| Module | Purpose |
|---|---|
| `symbol.py` | `SymbolSpec` — the ONLY place pip size, point, digits and price normalisation live; `SymbolSpecSource` protocol |
| `config.py` | documented parameters, family-specific stop policies, minimum asymmetry per family |
| `interfaces.py` | `BrokerConstraintSource` (UNKNOWN by default), `TransactionCostModel`, exit-plan / break-even / trailing / time-exit architecture (disabled) |
| `stops.py` | reference selection, noise buffer, sanity, volatility adequacy, noise risk, stop quality — no target/R input |
| `targets.py` | structural target discovery, clustering, ladder, barrier paths, reachability, quality, primary selection — no stop/R input |
| `construct.py` | `TradeContext`, the state machine, R and cost maths, conflict, quality, decision |
| `engine.py` | `TradeConstructionEngine.run(entry_result, h1_result, h4_result, broker=None, costs=None)` |
| `report.py` | `explain_trade(result, id)` — the audit answers |
| `outcomes.py` | MAE/MFE in R computed AFTER construction (never imported by construction) |

## 1. Risk before reward

The engine does not ask "how can this trade reach 1:3?"; it asks "where is this thesis objectively wrong?".

1. **EVALUATING_STOP** — structural invalidation, buffered by observed noise. No target or R information exists
   at this point (the stop module cannot see it; tested).
2. **EVALUATING_TARGETS** — structural targets from existing structure. No stop, R or minimum-R input.
3. **EVALUATING_RISK_REWARD** — `1R = |executable entry - proposed stop|`; gross and cost-adjusted R per target;
   the minimum-asymmetry gate is applied to the PRIMARY target of this fixed geometry.
4. **PROPOSED** or **REJECTED** (INVALIDATED / EXPIRED when the entry candidate or thesis fails first).

If the natural structure does not offer the configured asymmetry the trade is REJECTED. The stop is **never moved
closer** and the target is **never moved farther** to pass the gate: changing the minimum R changes only the
decision, never the stop, the targets or the primary target (tested for thresholds from 0.25 to 10 R).

## 2. Input and revalidation

Only Phase 1D candidates with an accepted `ENTRY_CANDIDATE` reach construction. Construction happens at the moment of
acceptance (the first executable price after the close of bar `p`), using information known at the close of `p`.
Revalidated first: entry candidate state at that instant, freshness (FRESH/AGING), H4 permission and context status,
Phase 1C setup still QUALIFIED, no H1 hard blocker. Immediately before finalising: entry validity, H4 permission,
remaining room (Phase 1D), stop thesis still intact (last close beyond the invalidation level) and target still beyond
the entry — otherwise INVALIDATED. After a proposal, it lapses (EXPIRED / INVALIDATED) together with its entry
candidate; the proposal record itself is never rewritten.

## 3. Price / pip mathematics

`SymbolSpec` (canonical GBPJPY: 3 digits, point 0.001, pip 0.01) provides `to_pips`, `to_points`, `pips_to_price`,
`normalize` (exact decimal) and `normalize_away` (stops rounded away from the entry, targets toward it — rounding
never improves R). A future adapter supplies the broker's real specification through `SymbolSpecSource`; the engine
checks it against the data configuration.

## 4. Stops — thesis invalidation

Stop families: STRUCTURAL_INVALIDATION (Phase 1C premise level: pullback extreme, broken level, sweep extreme or
compression extreme), SWING_INVALIDATION (most recent confirmed H1 swing), ZONE_INVALIDATION (far edge of the
nearest H1 zone), RECLAIM_FAILURE (swept liquidity extreme), BREAK_RETEST_FAILURE (retest extreme / broken level),
VOLATILITY_ADJUSTED_STRUCTURE (when the preferred level lies inside ordinary noise, the next structure FARTHER away).
No fixed-pip stop exists.

**Family policies** (ordered; first valid reference wins): trend pullback → structural, swing, zone; break-retest →
break-retest failure, structural, swing; liquidity sweep → reclaim failure, structural, swing; compression →
structural, swing, zone.

**Buffer**: `max(buffer_atr x ATR, quantile of recent adverse wicks)` plus the spread needed to express the chart
level on the triggering side of the book (a LONG stop sells on the BID, a SHORT stop buys on the ASK). Outputs
`raw_invalidation_price`, `stop_buffer`, `proposed_stop_price`.

**Sanity**: finite values, correct side, non-zero distance, minimum distance (`min_stop_pips`), symbol precision,
known broker stop level. A stop that fails is REJECTED — never widened or tightened to comply.

**Volatility adequacy**: `stop_distance_pips`, `stop_distance_atr`, `stop_distance_percentile` (vs trailing 5-bar
ranges) → TOO_TIGHT / NORMAL / WIDE / EXTREME; EXTREME is rejected (STOP_DISTANCE_EXCESSIVE).

**Noise risk** (0–100; not a hit probability): stop distance vs the q90 of recent 5-bar adverse excursions, vs ATR,
local chop, spread share, recent level interactions at the stop, buffer thickness. ≥ 70 rejects (STOP_INSIDE_NOISE).

**Stop quality** (0–100): structural relevance, noise clearance, volatility appropriateness, distance, setup-family
relevance (components exposed).

**Broker constraints**: `BrokerConstraintSource` → `BrokerStopConstraints`; without an adapter
`broker_constraints_status = UNKNOWN`.

## 5. Targets — structure and room

Sources (all known at proposal time): confirmed H1 swings, H1 zones, the aligned H4 room barriers (strong zones,
swings, range extreme, supply/demand origin zones), confirmed H4 swings, the H1 range extreme. Levels within 0.5 ATR
are one cluster (H1/H4 overlaps counted once). Clusters nearer than `min_target_atr` are barriers, not targets.

Each target (T1..T3) sits `front_run_atr` BEFORE its level (plus the exit-side spread for a SHORT) — never beyond
it — and stores price, type, timeframe, strength, distance, gross and net R, barriers before it, and its path
(barrier count, strong count, maximum strength, density, nearest barrier and its distance).

**Reachability** (0–100, entry-time information only): distance in ATR, H4 side context and H1 structure alignment,
momentum, volatility regime. Barriers are scored separately in the path (not counted twice). A target beyond
`reach_zero_atr` or below `min_reachability` is TARGET_UNREALISTIC.

**Target quality** = structural legitimacy (strength, timeframe confluence) + reachability + path quality.

**Primary target**: the highest-quality realistic target whose path is not congested (≥ 2 strong barriers or path
density ≥ 60). R is not a selection input. Alternatives are stored. If none: NO_REALISTIC_TARGET,
TARGET_UNREALISTIC or BARRIER_CONGESTION.

## 6. R and costs

`gross_R = |target - entry| / |entry - stop|` with exact directional maths on executable-side prices.
`estimated_net_R = (reward - costs) / (risk + costs)` with costs = entry slippage + exit slippage + round-turn
commission (pip-equivalent). Spread is embedded bid/ask-correctly in the entry, stop and target prices at the
spread known at the decision. Unknown slippage/commission use flagged CONSERVATIVE assumptions (never zero); swap
is UNKNOWN and excluded. `cost_assumptions` lists every component and its status (KNOWN / ASSUMED /
UNKNOWN_ASSUMED / UNKNOWN).

**Minimum asymmetry**: estimated net R of the primary target ≥ an unfitted per-family baseline (1.5). Later
validation may set values by family, direction, regime and volatility. Rejections distinguish INSUFFICIENT_RR from
EXECUTION_COSTS (gross passes, costs push net below: COSTS_DEGRADE_RR).

## 7. Quality, conflict, decision

`trade_construction_quality_score` families: ENTRY_QUALITY, STOP_QUALITY, TARGET_QUALITY (legitimacy +
reachability), ASYMMETRY, PATH_QUALITY, MARKET_CONTEXT — reduced by conflict.
`trade_construction_conflict_score` (noisy-OR): excellent entry + wide stop, good R with a poorly reachable target,
dense barriers before the target, stop close to noise, costs destroying asymmetry, H4 context weakening.
Decisions: PROPOSE_TRADE, REJECT_TRADE, INVALIDATE, EXPIRE (still no execution).

## 8. Management architecture (not active)

`ExitPlan` (single target by default; multi-target splits such as T1/T2/runner can be represented and validated),
`BreakEvenPolicy`, `TrailingPolicy`, `TimeExitPolicy` protocols with `BreakEvenSpec` (R achieved, structural progress,
target reached, new H1 structure), `TrailingSpec` (structure, ATR, swing, fixed-R) and `TimeExitSpec` (max bars,
no progress, session transition). All are disabled by default; partial exits, break-even, trailing and time exits
are not assumed to help and nothing is optimised.

## 9. Risk unit and account separation

`risk_unit.one_R_price / one_R_pips` defines 1R. All geometry is account-independent: the engine accepts no account,
balance, risk-percentage or leverage input and outputs none, so a proposal is identical for £1,000 or £1,000,000.
Sizing belongs to a future account-level risk engine.

## 10. Historical honesty and MAE/MFE

Construction uses only information known at the proposal time — never later highs, lows, swings, targets reached,
stop outcomes, volatility or news results (tested by rewriting the decision bar and all later bars).
`trade.outcomes.compute_excursions` measures MAE/MFE in R from bars AFTER the proposal for later research; the
construction modules never import it.

## 11. Research storage

Proposals and retained rejections carry: long/short direction (statistics kept separately via
`TradeResult.direction_stats()`), H1/H4 volatility regimes and ATR, execution-time session, news status
(`NEWS_UNKNOWN` without a provider), cost assumptions and broker-constraint status. Rejections keep their category
(stop too wide, stop inside noise, insufficient R, target unrealistic, barrier congestion, execution costs,
H4 permission changed, candidate expired, ...) and the full research geometry so later work can test whether each
filter adds value.

## 12. Known limitations

* Validated only on deterministic synthetic data; no threshold, weight, buffer or minimum R has been tested against
  outcomes.
* On the synthetic scenarios the stop is always the Phase 1C structural reference (trend pullbacks) and the ladder
  usually holds one or two targets; STOP_TOO_WIDE and BARRIER_CONGESTION occur only in constructed edge cases.
* Target sources exclude liquidity references not persisted by Phase 1C (equal highs/lows pools); H4 room barriers
  are limited to the five nearest.
* Spread at exit, slippage, commission and swap are unknown; conservative assumptions are flagged, swap is excluded.
* Broker stop/freeze levels are UNKNOWN without an adapter.
* Management policies are representations only.
* Proposals inherit the one-active-candidate-per-side limitation.
