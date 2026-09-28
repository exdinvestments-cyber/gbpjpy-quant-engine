# Phase 1G — Execution Safety

RISK-APPROVED TRADE → EXECUTION SAFETY → ORDER INTENT → platform-neutral ORDER LIFECYCLE.

There is **no platform connection, no network code and no real order** anywhere in this phase. The only
implementation of the execution port is the deterministic, testing-only `PaperBroker`. Every policy value in
`config/execution_default.yaml` is **CONFIGURATION — NOT VALIDATED EDGE**; none was optimised against profit.

Responsibilities stay separate:

| Layer | Question it answers |
|---|---|
| Phases 1A–1E (strategy) | Does this trade deserve to exist? |
| Phase 1F (risk) | Can the account afford it, and how much volume at most? |
| Phase 1G (execution) | Is it still safe and technically valid to send, exactly once, and is it protected? |

Execution may only **ACCEPT, DEFER, REJECT, CANCEL, RECONCILE or HALT**. It never improves a price, tightens or
widens a stop, extends a target, or raises a volume. It never assumes zero latency, zero slippage or zero spread,
and it never retries indefinitely.

## 1. Components (`src/gbpjpy_engine/execution/`)

| Module | Content |
|---|---|
| `model.py` | Canonical objects: `OrderIntent` (frozen), `OwnershipTag`, `StrategyIdentity`, `MarketSnapshot`, `ConnectionStatus`, `BrokerCapabilities`, `PortResponse`, `FillEvent`, `BrokerOrderView`, `BrokerPositionView`, `CloseRequest`, `ModificationRequest`, `CancelRequest`, `Position`; lifecycle, connection, session and rejection vocabularies |
| `port.py` | `ExecutionPort` protocol (the ONLY surface a future adapter implements), `Clock` protocol + deterministic `SimClock`, `SymbolMapper` / `SymbolMapping`, `canonical_rejection` |
| `config.py` | `ExecutionPolicy` (freshness, intent, retry, fills, protection, breaker) with validation and YAML loading |
| `store.py` | `ExecutionState`, `InMemoryExecutionStore`, `JsonFileExecutionStore` (checksummed JSON, atomic replace, lock file, quarantine of corrupted state) |
| `engine.py` | `ExecutionEngine`: intent creation, final gate, write-ahead submission, event handling, fills, protection, reconciliation, cancellation/expiry, modification/close contracts, breaker, health, metrics, latency, audit |
| `paper.py` | `PaperBroker` + `FaultPlan` — deterministic fault injection. **Testing only, not realism.** |
| `report.py` | `explain_execution` — human-readable audit |

Reused: Phase 1F `AccountRiskEngine` (live risk state, reservations, `release_reservation`, `convert_reservation`,
`RESET_CONFIRMATION`), `risk.fx.factor` and `risk.money` (Decimal, round-down), `risk.instruments.GBPJPY_CONTRACT`,
`trade.symbol.GBPJPY` (price normalisation), the point-in-time transition histories of the Phase 1C setup, 1D entry
candidate and 1E trade construction, and the shared `reason_codes` registry.

## 2. Naming (specification verbs → canonical names)

The protected Phase 1A safety test forbids certain order-function identifiers anywhere in the package, so the
canonical port verbs differ from the specification's wording. The mapping lives only here:

| Specification name | Canonical port method |
|---|---|
| `submit_order` | `request_submission` |
| `modify_order` | `request_modification` |
| `cancel_order` | `request_cancellation` |
| `close_position` | `request_close` |
| stop loss / take profit | `stop_price` / `target_price` |
| magic number | `StrategyIdentity.numeric_id` (mapped by the adapter) |
| ticket | `broker_ref` / `position_id` (opaque strings) |

Order types: `MARKET`, `BUY_LIMIT`, `SELL_LIMIT`, `BUY_STOP`, `SELL_STOP`.

## 3. Two-phase safety

**Phase A — `create_intent(approval, proposal, order_type, pending_price)`.** Only a Phase 1F `RISK_APPROVED`
decision whose proposal id matches can become an intent. The `OrderIntent` is immutable (frozen dataclass), has a
stable id `OI-{risk_approval_id}` and a stable idempotency key `client_order_key = sha256(strategy|instance|intent)[:20]`.
The same approval always yields the same intent (idempotent). It carries the structural stop and target, the
Phase 1F approved volume (a ceiling) and permitted risk, the cost assumptions, the H1 ATR, ownership metadata and a
finite validity window (30 min; AGING after half). The reference price is **not** a guaranteed fill.

**Phase B — `submit(intent_id, port, deps)`.** The final gate is re-run at the current clock time; only
`SUBMIT_READY` proceeds to ONE request through the port.

## 4. Final submission gate (order of checks)

Outcomes: `SUBMIT_READY`, `DEFER`, `REJECT`, `HALT` — each with reason codes.

1. **Phase 1F risk state** — read LIVE from the risk engine when one is attached (a halt raised after the caller
   built its dependencies still blocks). HALTED → HALT; PAUSED → DEFER; unavailable → HALT (fail closed).
2. Execution breaker: HALTED → HALT; PAUSED → DEFER.
3. Duplicate protection: an intent already sent / live / terminal, or another live intent for the same setup → REJECT.
4. Startup reconciliation not complete, or any open critical issue/incident → DEFER.
5. Validity window (EXPIRED → REJECT), AGING recorded; bounded backoff in progress → DEFER.
6. Phase 1F reservation still `RESERVED`.
7. Setup `QUALIFIED`, entry candidate `ENTRY_CANDIDATE`, proposal `PROPOSED` — evaluated from their transition
   histories **as known at the current time**; H4 permission still allows the direction.
8. Connection CONNECTED (DEGRADED only if policy allows).
9. Capabilities fresh, order type supported, netting conflict (a netting account already holding GBPJPY).
10. Quote present, same canonical symbol, not from the future, fresh (≤ 10 s); repeated stale quotes PAUSE.
11. Session OPEN (UNKNOWN only if policy allows).
12. Account snapshot fresh and not from the future.
13. Executable price: **LONG buys the ASK, SHORT sells the BID** — never a mid. Pending geometry: BUY_LIMIT below
    ASK, SELL_LIMIT above BID, BUY_STOP above ASK, SELL_STOP below BID.
14. Directional deterioration vs the approved reference (pips and ATR). Adverse beyond the limit → REJECT (never
    chased). A favourable move is not deterioration and is not "improved upon".
15. Spread (pips and fraction of ATR) → DEFER or REJECT by policy; percentile recorded, never optimised.
16. Stop/target geometry at the executable price and against the adapter-reported stop level.
17. **Risk recheck at the executable price** with fresh conversion rates: if the approved volume would now risk
    more than the Phase 1F permitted amount, the volume is reduced and **rounded DOWN**, or the intent is rejected
    (policy). Volume can only stay equal or fall.
18. Volume against adapter min/max/step (floored, never raised).
19. Gross and cost-adjusted net R at the executable price ≥ the proposal's minimum.
20. Room from the executable price to the first opposing structure ≥ minimum ATR.

A gate REJECT/EXPIRE returns the Phase 1F reservation.

## 5. Execution state architecture

Persisted `ExecutionState`: execution state (ENABLED/PAUSED/HALTED + reason/source), intents (immutable content),
orders (lifecycle state, full transition history, client key, attempts, requotes, adapter reference, validated
volume, fills, VWAP, last gate, backoff, latency marks), fills (by fill id, with a cost ledger), positions,
incidents, reconciliation (startup flag, open issues, last run, foreign items), control requests (modify / close /
cancel-remainder / emergency-close, keyed by idempotency key), heartbeat, counters, and an **append-only event log**
(monotonic sequence; no API edits or deletes events).

Lifecycle states: CREATED, VALIDATING, SUBMIT_READY, SUBMISSION_REQUESTED, ACKNOWLEDGED, PARTIALLY_FILLED, FILLED,
REJECTED, CANCEL_REQUESTED, CANCEL_ACKNOWLEDGED, CANCELLED, CANCEL_FAILED, EXPIRED, UNKNOWN, RECONCILIATION_REQUIRED.
Transitions are validated against an explicit legal map (`IllegalTransition` otherwise). UNKNOWN can never go back
to VALIDATING without passing through reconciliation.

## 6. Idempotency and exactly-once intent

* One intent per approval (`OI-{risk_approval_id}`), one client order key per intent, reused on any re-send.
* **Write-ahead:** SUBMISSION_REQUESTED is persisted *before* the port is called.
* A timeout, disconnect, error, exception or late unreferenced acknowledgement is **never read as a rejection**:
  the order becomes UNKNOWN → RECONCILIATION_REQUIRED. Nothing is re-sent until reconciliation proves the order
  absent (no working order, no position, no fills and the port's order status `NOT_FOUND`).
* After a restart every in-flight submission becomes RECONCILIATION_REQUIRED and the startup flag is cleared, so no
  submission is possible before startup reconciliation.
* All engine operations run inside a store transaction (in-process lock + exclusive lock file), so concurrent
  callers — threads or processes sharing one state file — cannot send the same intent twice.
* Fills are deduplicated by fill id; late or duplicate acknowledgements are recorded and ignored.

**How duplicate live trades will eventually be prevented:** the future adapter must carry the client order key
with the order (comment field or an external key↔ticket mapping) and must be able to answer "does an order,
position or deal with this key exist?". Together with the write-ahead record this makes resubmission after any
ambiguous outcome impossible without positive proof of absence.

## 7. Retries, rejections and the circuit breaker

Canonical rejections: INVALID_VOLUME, INVALID_PRICE, INVALID_STOPS, MARKET_CLOSED, INSUFFICIENT_MARGIN,
TRADE_DISABLED, PRICE_CHANGED, OFF_QUOTES, SYMBOL_DISABLED, REQUOTE, TRANSIENT_INFRASTRUCTURE,
UNKNOWN_BROKER_REJECTION. Only PRICE_CHANGED / OFF_QUOTES / REQUOTE / TRANSIENT_INFRASTRUCTURE are retryable, with
a deterministic backoff (1 s, 2 s, 4 s … capped at 8 s), at most 3 attempts and 2 requotes, each attempt re-running
the complete gate. Anything else is terminal and releases the reservation.

Breaker: 3 consecutive rejections → HALT; 2 critical reconciliation runs → HALT; 5 consecutive stale quotes → PAUSE;
3 disconnects → PAUSE; overfill, unprotected position or corrupted state → HALT. Leaving PAUSED/HALTED requires the
explicit authorisation (`ResetAuthorisation` with the Phase 1F confirmation phrase), which is recorded. A Phase 1F
HALT always overrides execution.

## 8. Fills, slippage and the cost ledger

Partial fills accumulate a Decimal VWAP. Slippage is signed per fill against the intended price (reference for
market, pending price for pending orders): **positive = adverse; favourable slippage is kept (negative), never
discarded.** The ledger records spread at submission, commission, slippage in pips and money, and swap / other
charges as unknown (`None`). Excessive slippage raises an incident. A fill beyond the validated volume is an
OVERFILL (critical, HALT). Partial-fill policies: ACCEPT_PARTIAL, CANCEL_REMAINDER (records the remainder
cancellation), CONTINUE_WITHIN_VALIDITY_WINDOW. The first fill converts the Phase 1F reservation.

## 9. Protection and unprotected positions

After a fill, `ensure_protection` attaches the structural stop and primary target (atomically with the entry when
the adapter reports `atomic_protection`, otherwise through `request_protection`) and then **verifies** symbol,
direction, volume, stop and target on the adapter's position view. If the stop is missing, different, or the
position cannot be verified, the position is `UNPROTECTED_POSITION`: a CRITICAL incident is opened, execution is
HALTED, and (policy `EMERGENCY_CLOSE_INTENT`) an idempotent emergency-close **intent** is recorded with status
`INTENT_RECORDED_NOT_SENT` — nothing is closed in Phase 1G. A missing target alone is recorded as
`PROTECTED_STOP_ONLY`.

## 10. Reconciliation (the adapter is the execution source of truth)

`reconcile(port, startup)` compares adapter orders, positions and fills with internal state:

* foreign items (no key and not our numeric id) are recorded and **never touched**;
* our identity without a known key → ORPHAN_BROKER_ORDER / UNEXPECTED_BROKER_POSITION (critical);
* UNKNOWN submissions: found → adopted (fills applied); working order → ACKNOWLEDGED; `NOT_FOUND` → VALIDATING (or
  EXPIRED past validity); any other status → SUBMISSION_STATUS_UNRESOLVED (critical);
* fills the event stream missed are adopted;
* MISSING_BROKER_ORDER / MISSING_BROKER_POSITION / VOLUME_MISMATCH / STATE_MISMATCH / STOP_MISMATCH are critical;
  PRICE_MISMATCH / TARGET_MISMATCH are recorded. Internal state is never silently overwritten; critical issues
  block new submissions until an operator resolves them (`resolve_issue`, audited).

## 11. Connection, clock, sessions, symbols

Connection states CONNECTED / DEGRADED / DISCONNECTED / UNKNOWN; heartbeat timestamps for connection, quote, account,
last success and reconciliation; `health()` reports DEGRADED_MONITORING when positions are open without a
connection (a server-side stop still exists, monitoring does not). All time comes from an explicit `Clock`:
`now()` for event time, `monotonic()` for timeouts and latency — no wall-clock arithmetic. Latency marks: decision →
submit, submit → ack, ack → first fill, total. Symbols are canonical (`GBPJPY`); adapter suffixes live only in a
`SymbolMapping`. Session status comes with every quote.

## 12. Modification and close contracts

`request_modification`: idempotent by key; a stop is never widened and a target never extended without an explicit
`risk_policy_approval`; a stop on the wrong side of the current exit price is rejected; trailing is not active.
`request_close`: full, partial or emergency; idempotent; unknown/foreign positions and invalid volumes are refused;
nothing is sent unless a port is passed explicitly.

## 13. Paper port — testing only

`PaperBroker` fills market orders at the current ASK/BID plus configured **adverse** slippage; it never produces a
better price unless a test explicitly requests negative slippage. Faults (`FaultPlan`): timeout / disconnect /
error / exception with or without receipt, rejections and requotes, partial fills, fills before acknowledgement,
duplicated and delayed events, hidden fills, protection rejection, wrong stop, dropped target, cancel/fill race,
cancel timeout. It models no liquidity, queue position or last-look and must not be used to estimate live
execution quality.

## 14. Look-ahead safety and determinism

Quotes, fills and events are visible only at or after their timestamps; a quote from the future is refused; upstream
states are read from transition histories as known at the current time; the account snapshot may not be from the
future. With `SimClock` and the paper port every run is bit-for-bit reproducible (identical event logs).

## 15. Known limitations

* No real adapter exists; all adapter behaviour is simulated by the testing-only paper port.
* Emergency closes are recorded as intents only; nothing is closed automatically.
* Trailing stops, break-even moves and scaling are not active.
* Swap and other charges are recorded as unknown.
* Reconciliation relies on the adapter carrying the client order key; an adapter that cannot will leave some
  ambiguous outcomes permanently in RECONCILIATION_REQUIRED (safe, but manual).
* A position may still lose more than 1R through gaps or stop slippage; protection verification cannot prevent that.
* Netting accounts are handled conservatively: any existing GBPJPY exposure blocks a new intent.
* Spread percentile is descriptive only.
* The whole execution state, including the append-only event log, is re-serialised on every save, so save cost
  grows linearly with the log; a production store should segment or rotate the log (append-only files).
* The paper port's fills are instantaneous; real latency, queueing and partial-fill behaviour are unknown.

## 16. MT4 adapter requirements

A future MT4 adapter (outside this package) must implement `ExecutionPort` exactly and:

* map `GBPJPY` ↔ the terminal symbol through `SymbolMapping` (suffixes configured, never hard-coded);
* translate `StrategyIdentity.numeric_id` to the MQL4 magic number and carry `client_order_key` in the order comment
  (and keep an external key ↔ ticket journal because comments can be altered);
* translate MQL4 error codes to the canonical rejection reasons with its own table (`canonical_rejection`);
* report capabilities: hedging mode, market and pending support, whether stop/target can be sent with the entry,
  `MODE_MINLOT` / `MODE_MAXLOT` / `MODE_LOTSTEP`, `MODE_STOPLEVEL`, `MODE_FREEZELEVEL`, with a report timestamp;
* return quotes with bid, ask, server time converted to UTC and a session status; never a mid;
* answer `get_order_status(key)` from open orders, open trades and account history (`NOT_FOUND` only with proof);
* expose fills per order (MT4 has no deal list: derive fills from the history and report partials explicitly);
* never retry internally and never alter prices, stops, targets or volumes it receives;
* report connection state (terminal connected / trade allowed / expert enabled);
* treat "trade context busy" as TRANSIENT_INFRASTRUCTURE and a timeout as TIMEOUT (never as a rejection).

## 17. MT5 adapter requirements

A future MT5 adapter must implement `ExecutionPort` exactly and:

* support both account margin modes and report `account_mode` HEDGING or NETTING;
* map the magic number from `numeric_id` and carry `client_order_key` in the request comment plus an external
  key ↔ order/deal/position journal;
* translate trade server return codes (`retcode`) to canonical rejection reasons;
* report filling modes (FOK / IOC / RETURN) and choose one without changing the requested volume;
* report volume limits, stop and freeze levels and whether protection can be attached with the entry;
* report orders, deals and positions separately: fills come from deals (each with a unique fill id, volume, price,
  commission, position id); positions carry stop and target for verification;
* answer `get_order_status(key)` from orders, history orders and deals;
* never retry internally, never modify prices/levels/volumes, and never treat a timeout as a rejection;
* report connection and trading-permission status and server time converted to UTC.

## 18. Portability and scope

The execution package imports only the standard library, pandas and the engine's own modules; the port and clock
are protocols; there is no network, platform or credential code. Phase 1G adds execution safety only: it does not
change strategy behaviour, risk rules or any earlier phase's decisions, and it does not begin Phase 1H.
