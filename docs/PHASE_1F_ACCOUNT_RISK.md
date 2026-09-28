# Phase 1F — Account Risk Engine

```
MARKET DATA -> H4 -> H1 SETUP -> ENTRY CANDIDATE -> TRADE CONSTRUCTION -> PROPOSED TRADE (Phases 1A-1E)
    -> ACCOUNT RISK ENGINE -> POSITION VOLUME -> RISK-APPROVED TRADE (Phase 1F)
    -> [future: execution engine -> platform adapter]
```

**A RISK-APPROVED TRADE IS NOT AN ORDER.** Nothing in `src/gbpjpy_engine/risk/` connects to a platform or places,
modifies or closes anything. Every policy value in `config/risk_default.yaml` is documented as
**CONFIGURATION — NOT VALIDATED EDGE**; none was chosen from (synthetic) profitability.

## 1. Capital philosophy

The engine asks: *"given this account and this valid trade, what is the maximum safe exposure permitted by the
risk framework?"* — never *"what must we risk to reach a target?"*.

* **£500k / profit-objective firewall** — there is no profit-objective input anywhere: no config field (the loader
  rejects unknown keys such as `annual_profit_target`), no `approve()` parameter, and a source test forbids
  target/goal/recovery identifiers in the risk package. Extra keys added to a proposal are ignored; tests show risk
  %, volume, stop and target are unchanged.
* **Score/risk firewall** — setup, entry, trade-quality and confidence values are never read (source-scanned).
  Quality decides whether a trade exists; policy decides exposure. A 95/100 setup gets the same risk as an 80/100 one.
* **No loss escalation** — every multiplier is ≤ 1 (validated); losses can only reduce risk (consecutive-loss
  reduction, drawdown tiers) or stop trading (pause, limits, halt). Wins never raise the percentage: compounding
  comes only from the current capital × the fixed policy percentage.
* **Leverage firewall** — leverage only feeds the margin feasibility check; 1:30 and 1:500 accounts receive
  identical permitted risk and volume when both can carry the margin (tested).
* No martingale, grid, averaging down, doubling after losses, revenge or recovery sizing exists.

## 2. Components

| Module | Purpose |
|---|---|
| `money.py` | numeric strategy (Decimal), rounding helpers |
| `instruments.py` | `ContractSpec` (base/quote currency, contract size, pip, point, digits, min/max volume, step, stop level) |
| `fx.py` | timestamp-aware conversion (`RateProvider`, `StaticRates`, `rates_from_bars`, `factor`) |
| `account.py` | canonical `AccountState`, `OpenPosition`, `BalanceAdjustment`, `ClosedTradeResult`, `StopChange` |
| `periods.py` | deterministic trading day / week / month keys (timezone + roll hour; DST via zoneinfo) |
| `config.py` | `RiskPolicy` — every value centralised and documented |
| `sizing.py` | pip value, per-volume risk, round-down volume, open-position risk, partial-close/stop-move helpers |
| `margin.py` | `MarginModel`, notional/leverage estimate, UNKNOWN when leverage/rates are missing |
| `policies.py` | `ExternalRiskPolicy`, `GenericRuleset` (prop-style envelopes, consistency reporting), exposure extension point |
| `state.py` | persistent `RiskState`, in-memory and JSON file stores (checksum, atomic replace, lock file) |
| `engine.py` | `AccountRiskEngine` — the approval pipeline, reservations, ledger, halts |
| `research.py` | research simulation, policy comparison, risk-of-ruin and Monte Carlo interfaces |
| `report.py` | `explain_risk_decision` — the audit answers |

## 3. GBPJPY pip value

GBPJPY P&L is in JPY: `P&L_JPY = Δprice × contract_size × volume`. One pip per 1.0 volume = `pip_size ×
contract_size` JPY (1,000 JPY for a 100,000 GBP contract). Account-currency pip value =
`pip_size × contract_size × volume × factor(JPY → account, as_of)`:

* JPY account: factor 1; GBP account: 1 / GBPJPY; USD: 1 / USDJPY; EUR: 1 / EURJPY; other currencies: direct,
  inverse, or one hop through GBP / USD / EUR / JPY.
* Example: GBP account, GBPJPY 190 → 1,000 / 190 = 5.263158 GBP per pip per 1.0 volume. No approximate pip value
  exists anywhere; tests check exact values across prices, currencies and contract sizes.

## 4. Conversion

`RateProvider.rate(base, quote, as_of)` returns the latest quote known at or before `as_of` (a later quote is
invisible). Rates older than `data.rate_max_age_seconds` are stale. Missing, stale, future or non-positive rates
raise `ConversionUnavailable` → the approval is rejected (`CONVERSION_RATE_MISSING` / `_STALE`) — never guessed. Every
rate used is recorded in the audit. `rates_from_bars` turns GBPJPY bar closes into rates available at each bar close.

## 5. Sizing

1. Sizing capital by `sizing.basis`: BALANCE, EQUITY or **LOWER_OF_BALANCE_OR_EQUITY** (default — floating profit is
   never sized on, floating loss always is).
2. Permitted risk = the strictest of: `PER_TRADE_CAP` (base %), `MAX_SINGLE_TRADE_CAP` (hard ceiling, ≤ 5 %),
   `DRAWDOWN_CAP`, `CONSECUTIVE_LOSS_CAP`, `NEWS_RISK_CAP`, `DAILY/WEEKLY/MONTHLY_REMAINING_RISK`,
   `AGGREGATE_RISK_CAP`, `SYMBOL_RISK_CAP`, `EXTERNAL_POLICY_CAP` — rounded DOWN to the currency minor unit. The
   binding constraint is always reported; `MARGIN_CAP` appears when margin reduces the volume.
3. Risk per 1.0 volume = (stop distance + stop slippage allowance + entry slippage allowance + commission, in pips)
   × pip size × contract size × conversion. The spread is already embedded in the Phase 1E entry/stop prices.
4. Theoretical volume = permitted / risk-per-volume → **ROUND DOWN** to the volume step (never above max volume or
   an external cap). Below the minimum volume → REJECT (attributed to the binding budget when a budget is
   effectively exhausted).
5. Actual risk = volume × risk-per-volume, verified ≤ permitted (explicit tolerance); actual % reported.

**Costs** stay explicit: spread KNOWN/ASSUMED (embedded), slippage ESTIMATED_ALLOWANCE (configurable; replaced by
KNOWN data when a model supplies it), commission KNOWN or UNKNOWN_ASSUMED (never zero), swap UNKNOWN.

**Gap risk**: the approval records `planned_risk`, `gap_risk_status = NOT_BOUNDED_BY_STOP` and the slippage
assumption. Planned risk is never described as a guaranteed maximum loss.

## 6. Budgets, drawdown, losses, exposure

* **Daily / weekly / monthly** budgets use the reference capital captured at the first observation of each period
  (persisted — a restart never resets it). Used budget = realised trading loss in the period (ledger; deposits
  excluded) + floating loss + remaining stop risk of open positions + reserved risk + provider-reported pending
  risk (deliberately conservative). Exhausted → `BLOCK_NEW_TRADES`; the monthly budget can be disabled.
* **Boundaries**: trading day rolls at `rollover_hour` in `timezone` (default 17:00 America/New_York, DST-correct);
  ISO weeks of the trading date; calendar months of the trading date. The machine clock is never used.
* **Drawdown**: persistent balance and equity high-water marks; absolute and % drawdown; tiers NORMAL / CAUTION
  (×0.75) / DEFENSIVE (×0.5) / HALT (≥ 15 %, sets HALTED). Asymmetric: bad conditions → less exposure, good
  conditions → the normal baseline, never more.
* **Consecutive losses** (closed, non-partial results): ≥ 3 → risk × 0.5; ≥ 5 → PAUSED until the next trading day
  (or manual reset). A pause re-triggers only after further losses; no psychological timing is assumed.
* **Exposure**: GBPJPY position / direction limits, duplicate-exposure check on setup / entry / proposal / position
  ids, **no pyramiding by default**, adding to a losing position always prohibited, hedging off by default.
* **Open and reserved risk**: remaining stop risk per open position (a position without a stop fails closed);
  approvals reserve their actual risk so simultaneous candidates cannot each assume the whole budget. Reservations
  are released (not placed) or converted (filled) explicitly.
* **Deposits / withdrawals** are `BalanceAdjustment` events: they shift high-water marks and period references and
  never count as trading profit or loss.

## 7. Margin

`MarginModel.estimate` (default: volume × contract size in GBP, converted, / leverage). Approval requires BOTH
loss-risk approval and margin approval: free margin after the trade ≥ 50 % of equity and margin level ≥ 300 %.
UNKNOWN margin rejects by default; `margin_policy = REDUCE` lowers the volume to what margin safely allows
(`MARGIN_CAP`).

## 8. External rulesets, weekend, news

`AccountState.profile` is PERSONAL or EXTERNAL_RULESET; the same strategy runs inside any `ExternalRiskPolicy`.
`GenericRuleset` supports daily loss, static/trailing drawdown, maximum volume, maximum open positions, weekend
holding and news restrictions; consistency rules are reported, never used to distort trades. Weekend policy ALLOW /
DISALLOW / CLOSE_BEFORE_WEEKEND / UNKNOWN_EXTERNAL_POLICY blocks new approvals inside the configured cutoff before the
Friday roll. News policy ALLOW / BLOCK_BEFORE / BLOCK_AFTER / REDUCE_RISK / UNKNOWN acts on the Phase 1D news status
(UNKNOWN without a provider — no events are fabricated).

## 9. Kill switch, fail closed, persistence, concurrency

* Global state ENABLED / PAUSED / HALTED is persisted. HALTED blocks every approval regardless of strategy signals.
  Automatic halts: critical drawdown, invalid account state, invalid contract specification, corrupted risk state,
  external breach (and optionally the daily limit). Manual `halt()` persists across restarts; only `reset()` with a
  `ResetAuthorisation` (operator, reason, exact confirmation phrase) clears it, and the reset is audited.
* **Fail closed**: missing/stale conversion, stale or future account state, invalid account or contract, unusable or
  expired proposal, changed geometry, lost H4 permission, unbounded open risk, unknown margin, non-finite values,
  corrupted state → explicit rejection.
* **Persistence**: `JsonFileRiskStateStore` — checksummed JSON (Decimals as exact strings), atomic replace; a
  corrupted file is quarantined and the engine HALTS until an authorised reset.
* **Idempotency**: approvals are keyed by `trade_proposal_id` (`RA-{id}`); reprocessing returns the stored decision
  and never reserves twice. Closed results and adjustments are idempotent by id.
* **Concurrency**: load → evaluate → reserve → save runs inside one transaction (thread lock + exclusive lock file),
  so two near-simultaneous candidates cannot consume the same budget (tested with threads and two engine instances on
  one file).

## 10. Numeric strategy

Decimal (28 digits) for money, volumes, steps and percentages; floats converted via `repr`. Permitted amounts round
DOWN, reported losses round UP, volumes floor to the step. Non-finite values raise `NumericError` → fail closed.

## 11. Look-ahead and determinism

Approval at `T` uses only the account snapshot (≤ T, fresh), rates known at T, ledger entries closed at ≤ T and the
persisted state. Future rates and results are invisible (tested). Identical inputs give identical approvals and ids;
decision time is always the supplied `as_of`, never the machine clock.

## 12. Research mode

`ResearchRunner` replays Phase 1E proposals over a simulated account; `compare_policies` runs identical signals
under different policies without regenerating signals. Outcomes are always supplied by the caller — nothing is
invented, and synthetic sequences say nothing about real results. `risk_of_ruin` is an interface that refuses to
compute without validated inputs (win probability, R distribution, tail losses, dependency, slippage, drawdown
distribution); `monte_carlo_records` exports clean risk records.

## 13. What a future MT4/MT5 adapter must provide

Account state (currency, balance, equity, margin fields, floating P&L, leverage, positions with stops, timestamp),
contract specifications (`ContractSpec`), timestamped conversion rates (`RateProvider`), margin requirements
(`MarginModel`), position state (open positions, partial closes, stop changes), closed-trade results
(`ClosedTradeResult`), balance adjustments, and — later — order state for the execution engine. See
`docs/BROKER_NEUTRAL_INTERFACES.md`.

## 14. Known limitations

* All policy values are configuration, not validated edge; no policy has been compared on real outcomes.
* The engine trusts the adapter's account snapshot; floating P&L per position is informational.
* Daily/weekly budgets count floating loss and remaining stop risk together (conservative double count).
* Reservations persist until released or converted; a missing release keeps budget locked (safe side).
* Margin uses a notional/leverage approximation unless an adapter supplies the real requirement.
* Commission, swap and real slippage are unknown until provided; allowances are assumptions.
* Consecutive-loss counting uses final (non-partial) closes only.
* Cross-process locking is a lock file for cooperating processes on one host.
