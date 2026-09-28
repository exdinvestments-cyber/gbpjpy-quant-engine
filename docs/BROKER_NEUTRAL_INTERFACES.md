# Broker-neutral interfaces (future MT4 / MT5 adapters)

Status: **specification only — not implemented.** No adapter, connection, order or account code exists in this
repository. This document fixes the contracts future adapters must satisfy so that the GBPJPY strategy core never
depends on MetaTrader 4, MetaTrader 5, MQL4/MQL5, broker credentials or broker-specific structures.

```
                 GBPJPY STRATEGY CORE
   (market data -> H4 features -> H4 context/permission -> H1 setup intelligence (Phase 1C)
    -> entry intelligence / executable entry candidate (Phase 1D, NOT an order)
    -> trade construction / proposed trade (Phase 1E, NOT an order: prices and R only)
    -> account risk engine / risk-approved trade (Phase 1F, NOT an order: volume and money risk)
    -> [future: execution engine, execution contract])
                          |
               BROKER-NEUTRAL INTERFACES   (this document; market-data protocols in data/interfaces.py)
                     /                \
              MT4 ADAPTER          MT5 ADAPTER        (future; separate package; owns all platform code)
```

Separation of concerns (target):

| Layer | Owns | May depend on |
|---|---|---|
| Market data | canonical bars/ticks, validation | stdlib, numpy, pandas |
| Feature engineering (Phase 1A) | causal H4 features, structure, zones | market data |
| Context engine (Phase 1B) | hierarchy, breakouts, liquidity, permission | features |
| H1 setup intelligence (Phase 1C) | H1 setups up to QUALIFIED | context |
| Trade construction (Phase 1E) | structural stop, targets, 1R, gross/net R - no volume or money | entry candidates + the protocols below |
| Entry intelligence (Phase 1D) | confirmation, timing, spread/bid-ask, executable entry candidate (not an order) | setups + the protocols below |
| Account risk engine (Phase 1F) | sizing, limits, drawdown, margin, halts - no orders | proposals, account state (neutral) |
| Execution engine (future) | order intent → neutral order request | risk, neutral interfaces |
| Broker/platform adapter (future) | MT4/MT5 translation, sessions, reconnection | neutral interfaces + platform SDK |
| Account state / persistence / monitoring (future) | neutral records | neutral interfaces |

Rules for every adapter:

1. Convert all timestamps to timezone-aware **UTC** before data leaves the adapter (broker server time is commonly
   EET/EEST). Never deliver an unfinished bar as closed.
2. Prices are broker quotes; the core never assumes a particular broker's digits — they come from the symbol
   specification.
3. Tick volume is broker-specific quote-update count, not exchange volume.
4. All platform identifiers (tickets, magic numbers, platform enums, error codes) stay inside the adapter and are
   mapped to the neutral fields below.

## Bars

Implemented as `data.interfaces.BarSource` (protocol) + the canonical frame in `data/model.py`:
`timestamp` (UTC bar open), `open`, `high`, `low`, `close`, `volume` (tick volume, optional), `spread`
(optional, units declared by the adapter), `source`. Only closed bars (`timestamp + timeframe <= as_of`).

## Ticks

Protocol `data.interfaces.TickSource`: rows `time` (UTC), `bid`, `ask`, optional `last`, `flags`. Used later for
spread monitoring and intrabar evidence; not consumed by Phase 1A/1B.

Intrabar extension point: `data.interfaces.IntrabarProvider.observe(bar_open, level, direction)` returns an
`IntrabarObservation` (e.g. whether price traded back through a level inside a completed H4 bar). Phase 1B accepts
this evidence as optional input and currently runs with none.

## Spread

Neutral record: `time` (UTC), `bid`, `ask`, `spread_price = ask - bid`, `spread_pips` (using the symbol pip size),
`source`. Derived from ticks or bar spread; broker-reported "points" must be converted by the adapter.

## Quotes and executable prices

Implemented as the protocol `entry.interfaces.QuoteSource.quote_after(index) -> ExecutableQuote | None`: the first
price observable after the close of H1 bar `index` (`time` UTC, chart-basis `price`, the `spread` KNOWN at that
time in declared units, `spread_source`). The historical default `BarOpenQuotes` uses the next bar's open and the
spread reported by the bar that just closed. A future adapter supplies real quotes (bid/ask); LONG references the
ASK, SHORT the BID. Missing spread must be delivered as missing - never as zero.

## Slippage

Protocol `entry.interfaces.SlippageModel.estimate(SlippageContext) -> SlippageEstimate(pips | None, status)`.
Default `UnknownSlippage` (status UNKNOWN). Fixed, spread-dependent, volatility-dependent and empirical models
apply only researcher-supplied numbers; a future adapter may feed observed broker slippage samples.

## News calendar

Protocol `entry.interfaces.NewsProvider.events(start, end, known_at) -> list[NewsEvent]` with `time` (UTC),
`currency` (GBP / JPY / GLOBAL), `importance`, `name`, `category`, `known_since`. Events whose schedule was not
known at `known_at` must not be returned (the core filters them again). Outcomes are not part of the contract.
No provider exists yet: news status is UNKNOWN.

## Lower-timeframe data

Protocols `entry.interfaces.LowerTimeframeProvider.closed_bars(timeframe, start, end, as_of)` (M15/M5/M1/TICK,
closed data only) and `IntrabarConfirmationProvider.confirm(direction, level, start, end, as_of)`. Optional
research inputs; Phase 1D decisions never depend on them.

## Trade proposal

Produced by Phase 1E (`trade.TradeConstructionEngine`): direction, executable entry reference, proposed stop
(LONG: sell stop on the BID; SHORT: buy stop on the ASK), structural target ladder and primary target, 1R, gross and
estimated net R, cost assumptions. It contains no volume: a future account-level risk engine converts R into volume,
and a future execution contract / MT4-MT5 adapter translates canonical entry, stop, target and volume into
platform order structures.

## Broker stop constraints

Protocol `trade.interfaces.BrokerConstraintSource.stop_constraints(symbol, time) -> BrokerStopConstraints(status,
min_stop_distance_points, freeze_level_points, source)`. Without an adapter the status is UNKNOWN. A stop inside a
known broker stop level is rejected, never silently widened.

## Transaction costs

Protocol `trade.interfaces.TransactionCostModel.estimate(symbol, time, direction) -> CostEstimate` (pip-equivalent
round-turn commission, account-independent). Unknown costs are represented explicitly, never as zero.

## Account risk adapter requirements

Phase 1F (`risk.AccountRiskEngine`) consumes, and a future adapter must supply:

* **Account state** (`risk.account.AccountState`, via `AccountStateSource`): account id, currency (any ISO code),
  balance, equity, free and used margin, margin level, floating P&L, open positions (id, direction, volume, entry,
  stop, current price, floating P&L, linked proposal ids), provider-pending risk, leverage where known, timestamp (UTC).
* **Contract specifications** (`risk.instruments.ContractSpec`, via `ContractSpecSource`): base/quote currency,
  contract size, digits, point, pip size, minimum/maximum volume, volume step, minimum stop distance, margin mode.
* **Conversion rates** (`risk.fx.RateProvider`): timestamped quotes; the core never guesses a missing rate.
* **Margin information** (`risk.margin.MarginModel`): the provider's margin requirement where the notional/leverage
  approximation is not reliable.
* **Position state**: partial closes and stop changes (`StopChange`), so remaining risk can be recomputed.
* **Closed-trade results** (`ClosedTradeResult`: id, realised P&L, fees, swap, close time, close reason) and balance
  adjustments (`BalanceAdjustment`: deposits, withdrawals, corrections).
* **Order state** - for the future execution engine only; Phase 1F places nothing.

## Symbol specification

Canonical form consumed by the core: `trade.symbol.SymbolSpec(symbol, digits, point, pip_size, source)`, supplied by
an adapter through `SymbolSpecSource`. Adapter-side broker fields: `symbol` (canonical `GBPJPY`), `broker_symbol` (e.g. with suffix), `digits`, `point`, `pip_size` (0.01),
`contract_size`, `min_volume`, `max_volume`, `volume_step`, `margin_currency`, `profit_currency`,
`trading_sessions` (UTC windows), `stop_level_points`, `freeze_level_points`, `swap_long`, `swap_short`.

## Account state

`account_id` (opaque), `currency`, `balance`, `equity`, `margin_used`, `margin_free`, `margin_level`,
`leverage` (read-only information, never an input to strategy decisions), `timestamp` (UTC), `source`.

## Order request

Neutral intent produced by a future execution engine: `request_id` (UUID), `symbol`, `side` (BUY/SELL),
`order_type` (MARKET/LIMIT/STOP), `quantity` (in neutral lots, validated against the symbol specification by the
risk engine), `price` (for pending orders), `stop_loss_price`, `take_profit_price`, `time_in_force`,
`max_slippage_pips`, `reason_codes` (audit trail back to the H4/H1 decision), `created_at` (UTC).

## Order result

`request_id`, `status` (ACCEPTED/REJECTED/FILLED/PARTIALLY_FILLED/CANCELLED/EXPIRED), `broker_order_id` (opaque
string), `filled_quantity`, `average_fill_price`, `slippage_pips`, `commission`, `error_code` (neutral enum),
`error_message`, `timestamp` (UTC).

## Position state

`position_id` (opaque), `symbol`, `side`, `quantity`, `open_price`, `open_time` (UTC), `stop_loss_price`,
`take_profit_price`, `unrealised_pnl`, `swap`, `commission`, `origin_request_id`, `timestamp` (UTC).

---

None of these interfaces imply that martingale, grid, averaging-down, loss-recovery sizing or unbounded leverage
will ever be supported; the future risk engine is expected to reject such behaviour explicitly.
