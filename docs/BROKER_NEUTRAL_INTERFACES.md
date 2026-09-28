# Broker-neutral interfaces (future MT4 / MT5 adapters)

Status: **specification only — not implemented.** No adapter, connection, order or account code exists in this
repository. This document fixes the contracts future adapters must satisfy so that the GBPJPY strategy core never
depends on MetaTrader 4, MetaTrader 5, MQL4/MQL5, broker credentials or broker-specific structures.

```
                 GBPJPY STRATEGY CORE
   (market data -> H4 features -> H4 context/permission -> H1 setup intelligence (Phase 1C)
    -> [future: entry engine, risk engine, execution logic])
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
| Risk engine (future) | exposure limits, drawdown guards | context, account state (neutral) |
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

## Symbol specification

`symbol` (canonical `GBPJPY`), `broker_symbol` (e.g. with suffix), `digits`, `point`, `pip_size` (0.01),
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
