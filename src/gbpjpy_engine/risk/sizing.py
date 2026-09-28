"""GBPJPY pip value, per-volume risk and round-down volume (Phase 1F).

Pip-value methodology
---------------------
GBPJPY is quoted in JPY per GBP.  For volume ``v`` (in contract units of
``contract_size`` GBP), a price move of ``dP`` produces a profit/loss of

    P&L_JPY = dP x contract_size x v                      (quote currency)

so one pip per 1.0 volume is ``pip_size x contract_size`` JPY (1,000 JPY for the
canonical 100,000 GBP contract).  Converting to the account currency uses the
timestamp-aware JPY -> account factor (``fx.factor``):

    pip_value_account = pip_size x contract_size x v x factor(JPY -> ACC, as_of)

A JPY account needs no conversion; a GBP account uses 1 / GBPJPY; a USD account
1 / USDJPY; other currencies a direct/inverse pair or one pivot hop.  No
approximate hard-coded pip values exist anywhere.

Quality scores are NOT inputs to anything in this module.
"""

from __future__ import annotations

from decimal import Decimal

from .money import D, floor_to_step


def pip_value(contract, volume, jpy_to_account: Decimal) -> dict:
    per_pip_quote = D(contract.pip_size) * D(contract.contract_size) * D(volume)
    return {"pip_value_quote": per_pip_quote, "quote_currency": contract.quote_currency,
            "pip_value_account": per_pip_quote * jpy_to_account}


def risk_per_volume(contract, stop_distance_price, extra_pips, jpy_to_account: Decimal) -> Decimal:
    """Account-currency risk of 1.0 volume: (stop distance + cost/slippage allowances) x contract x conversion."""
    dist = D(stop_distance_price) + D(extra_pips) * D(contract.pip_size)
    return dist * D(contract.contract_size) * jpy_to_account


def normalise_volume(theoretical, contract, extra_cap=None) -> tuple[Decimal, list[str]]:
    """Floor to the volume step (ROUND DOWN), never above max_volume or an external volume cap."""
    notes = []
    cap = D(contract.max_volume)
    if extra_cap is not None and D(extra_cap) < cap:
        cap = D(extra_cap)
        notes.append("EXTERNAL_MAX_VOLUME")
    v = D(theoretical)
    if v > cap:
        v = cap
        notes.append("MAX_VOLUME_CAP")
    out = floor_to_step(v, contract.volume_step)
    if out < D(theoretical):
        notes.append("VOLUME_ROUNDED_DOWN")
    return out, notes


def position_remaining_risk(position, contract, jpy_to_account: Decimal, new_stop: float | None = None) -> dict:
    """Remaining planned stop risk of an open position (account currency).

    Measured from the current price when known (else the entry) to the stop; a stop already beyond the price
    in the profitable direction leaves zero PLANNED risk - but gaps and slippage can still produce a loss, so the
    figure is never a guaranteed maximum."""
    d = 1 if position.direction == "LONG" else -1
    stop = new_stop if new_stop is not None else position.stop_price
    if stop is None:
        return {"position_id": position.position_id, "remaining_risk": None, "status": "UNBOUNDED_NO_STOP"}
    ref = position.current_price if position.current_price is not None else position.entry_price
    dist = max(D(d) * (D(ref) - D(stop)), D(0))
    risk = dist * D(contract.contract_size) * D(position.volume) * jpy_to_account
    return {"position_id": position.position_id, "remaining_risk": risk, "status": "BOUNDED_BY_STOP",
            "stop": stop, "reference_price": ref, "volume": str(position.volume),
            "gap_risk_status": "NOT_BOUNDED_BY_STOP (gaps/slippage can exceed planned risk)"}


def partial_close_remaining(position, closed_volume) -> Decimal:
    """Volume left after a partial close (future execution reports it; remaining risk is recomputed from it)."""
    left = D(position.volume) - D(closed_volume)
    if left < 0:
        raise ValueError("closed volume exceeds the open volume")
    return left
