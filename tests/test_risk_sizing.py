"""Phase 1F: numeric precision, contract spec, FX conversion, GBPJPY pip value, position sizing, rounding, costs,
gap disclosure, single-trade ceiling, balance/equity basis, margin and the leverage firewall."""

from __future__ import annotations

import itertools
from dataclasses import replace
from decimal import Decimal as Dec

import pandas as pd
import pytest

from gbpjpy_engine.risk import (GBPJPY_CONTRACT, ContractSpec, ConversionRate, ConversionUnavailable, FixedMarginPerVolume,
                                StaticRates, UnknownMargin, factor)
from gbpjpy_engine.risk.money import NumericError, D, floor_to_step, money_down, money_up
from gbpjpy_engine.risk.sizing import normalise_volume, partial_close_remaining, pip_value, position_remaining_risk
from risk_helpers import RATE_TS, T0, account, engine, policy, position, proposal, rates

H = pd.Timedelta(hours=2)


# ------------------------------------------------------------------ numeric strategy
def test_decimal_money_and_volume_rounding():
    assert D(0.1) == Dec("0.1") and D("190.123") == Dec("190.123")
    for bad in (float("nan"), float("inf"), None, "abc", Dec("NaN")):
        with pytest.raises(NumericError):
            D(bad)
    assert money_down("46.539", "GBP") == Dec("46.53") and money_up("46.531", "GBP") == Dec("46.54")
    assert money_down("7012.9", "JPY") == Dec("7012") and money_up("7012.1", "JPY") == Dec("7013")
    assert floor_to_step("0.13999", "0.01") == Dec("0.13") and floor_to_step("1.9", "1") == Dec("1")
    assert floor_to_step("0.0379", "0.001") == Dec("0.037")


def test_contract_specification_is_validated_not_assumed():
    assert GBPJPY_CONTRACT.issues() == [] and (GBPJPY_CONTRACT.base_currency, GBPJPY_CONTRACT.quote_currency) == ("GBP", "JPY")
    assert ContractSpec(contract_size=Dec("0")).issues()
    assert ContractSpec(min_volume=Dec("0.015"), volume_step=Dec("0.01")).issues()
    assert ContractSpec(point=Dec("0.01"), digits=3).issues()
    mini = ContractSpec(contract_size=Dec("10000"), min_volume=Dec("0.1"), volume_step=Dec("0.1"), source="provider_x")
    assert mini.issues() == []


# ------------------------------------------------------------------ FX conversion
def test_conversion_direct_inverse_cross_and_timestamps():
    r = rates(gbpusd="1.27")
    f, used = factor("JPY", "GBP", T0, r, H)
    assert f == Dec(1) / Dec("190") and used[0].base == "GBP"  # inverse of GBPJPY
    assert factor("JPY", "JPY", T0, r, H) == (Dec(1), [])
    assert factor("JPY", "USD", T0, r, H)[0] == Dec(1) / Dec("150")
    chf = StaticRates([ConversionRate("USD", "JPY", Dec("150"), RATE_TS), ConversionRate("USD", "CHF", Dec("0.9"), RATE_TS)])
    f2, used2 = factor("JPY", "CHF", T0, chf, H)
    assert f2 == Dec(1) / Dec("150") * Dec("0.9") and len(used2) == 2  # one pivot hop, both rates recorded
    with pytest.raises(ConversionUnavailable):
        factor("JPY", "SEK", T0, r, H)
    with pytest.raises(ConversionUnavailable, match="stale"):
        factor("JPY", "GBP", T0 + pd.Timedelta(hours=5), rates(span_hours=0), H)
    future = StaticRates([ConversionRate("GBP", "JPY", Dec("150"), T0 + pd.Timedelta(minutes=1))])
    with pytest.raises(ConversionUnavailable):
        factor("JPY", "GBP", T0, future, H)  # a rate published after T0 is invisible at T0


# ------------------------------------------------------------------ pip value
@pytest.mark.parametrize("price", ["150", "190", "210.555"])
def test_gbpjpy_pip_value_is_exact_in_every_account_currency(price):
    r = rates(gbpjpy=price, usdjpy="151.25", eurjpy="163.4")
    for ccy, expected in (("JPY", Dec("1000")), ("GBP", Dec("1000") / Dec(price)), ("USD", Dec("1000") / Dec("151.25")),
                          ("EUR", Dec("1000") / Dec("163.4"))):
        f, _ = factor("JPY", ccy, T0, r, H)
        pv = pip_value(GBPJPY_CONTRACT, Dec(1), f)
        assert pv["pip_value_quote"] == Dec("1000.00") and pv["pip_value_account"] == expected
        assert pip_value(GBPJPY_CONTRACT, Dec("0.37"), f)["pip_value_account"] == expected * Dec("0.37")
    mini = ContractSpec(contract_size=Dec("10000"))
    assert pip_value(mini, Dec(1), Dec(1))["pip_value_quote"] == Dec("100.00")


# ------------------------------------------------------------------ CRITICAL: position sizing grid
def test_actual_risk_never_exceeds_permitted_after_normalisation():
    checked = 0
    grid = itertools.product(("150", "190", "212.5"), ((1, {}), (1, {"stop_refs": {"STRUCTURAL_INVALIDATION": {"price": 189.75, "reason": "x"}}}),
                                                    (-1, {})),
                             (("GBP", "10000"), ("USD", "2500"), ("EUR", "250000"), ("JPY", "1500000"), ("GBP", "1000000")),
                             ((Dec("100000"), Dec("0.01"), Dec("0.01")), (Dec("10000"), Dec("0.1"), Dec("0.1")),
                              (Dec("100000"), Dec("0.001"), Dec("0.001")), (Dec("1000"), Dec("1"), Dec("1"))))
    for price, (d, over), (ccy, bal), (size, step, vmin) in grid:
        c = replace(GBPJPY_CONTRACT, contract_size=size, volume_step=step, min_volume=vmin)
        eng = engine(rate_provider=rates(gbpjpy=price), contract=c)
        dec = eng.approve(proposal(d, **over), account(balance=bal, currency=ccy, lev=500), T0)
        if dec["decision"] != "RISK_APPROVED":
            assert dec["rejection_category"] == "BROKER_MIN_VOLUME", dec["rejection_reason"]
            continue
        a = dec["approved_trade"]
        vol, permitted, actual = Dec(a["approved_volume"]), Dec(a["risk_amount"]), Dec(a["actual_risk_currency"])
        assert vol >= vmin and (vol / step) % 1 == 0 and vol <= Dec(a["theoretical_volume"]) + Dec("1e-8")
        per = Dec(dec["audit"]["volume"]["risk_per_1_volume"])
        assert vol * per <= permitted  # exact, before display rounding
        assert actual <= permitted + Dec("0.01") * (1 if ccy != "JPY" else 100)
        assert Dec(a["actual_risk_percent"]) <= Dec("0.5001")
        checked += 1
    assert checked > 100


def test_broker_minimum_volume_rejects_instead_of_exceeding_risk():
    big = replace(GBPJPY_CONTRACT, min_volume=Dec("1.00"), volume_step=Dec("0.01"))
    dec = engine(contract=big).approve(proposal(), account(balance="1000"), T0)
    assert dec["decision"] == "RISK_REJECTED" and "BROKER_MIN_VOLUME_TOO_LARGE" in dec["reason_codes"]


def test_volume_rounds_down_and_respects_caps():
    v, notes = normalise_volume(Dec("0.13999"), GBPJPY_CONTRACT)
    assert v == Dec("0.13") and "VOLUME_ROUNDED_DOWN" in notes
    v2, n2 = normalise_volume(Dec("250"), GBPJPY_CONTRACT)
    assert v2 == Dec("100") and "MAX_VOLUME_CAP" in n2
    v3, n3 = normalise_volume(Dec("5"), GBPJPY_CONTRACT, Dec("2.505"))
    assert v3 == Dec("2.50") and "EXTERNAL_MAX_VOLUME" in n3


# ------------------------------------------------------------------ risk %, basis, ceiling, costs, gap
def test_percentage_risk_and_sizing_basis():
    p = proposal()
    base = engine().approve(p, account(balance="10000", equity="9000"), T0)["approved_trade"]
    assert base["risk_basis"] == "LOWER_OF_BALANCE_OR_EQUITY" and base["sizing_capital"] == "9000"
    assert Dec(base["risk_amount"]) == Dec("45.00")
    bal = engine(policy(sizing={"basis": "BALANCE"})).approve(p, account(balance="10000", equity="9000"), T0)["approved_trade"]
    eq = engine(policy(sizing={"basis": "EQUITY"})).approve(p, account(balance="10000", equity="11000"), T0)["approved_trade"]
    assert bal["risk_amount"] == "50.00" and eq["risk_amount"] == "55.00"
    low = engine().approve(p, account(balance="10000", equity="11000"), T0)["approved_trade"]
    assert low["risk_amount"] == "50.00"  # floating profit is not sized on by default


def test_single_trade_ceiling_cannot_be_exceeded():
    with pytest.raises(ValueError):
        policy(sizing={"base_risk_percent": 2.0, "max_single_trade_risk_percent": 1.0}).validate()
    with pytest.raises(ValueError):
        policy(sizing={"max_single_trade_risk_percent": 7.0}).validate()  # absolute sanity ceiling
    a = engine(policy(sizing={"base_risk_percent": 1.0, "max_single_trade_risk_percent": 1.0}),
               ).approve(proposal(), account(), T0)["approved_trade"]
    assert Dec(a["actual_risk_percent"]) <= Dec("1.0")


def test_costs_are_explicit_and_slippage_allowance_widens_risk():
    a = engine().approve(proposal(), account(), T0)["approved_trade"]
    c = a["costs"]
    assert c["spread"]["status"] == "KNOWN" and c["commission_pips"]["status"] == "UNKNOWN_ASSUMED"
    assert c["stop_slippage_pips"]["status"] == "ESTIMATED_ALLOWANCE" and c["swap"]["status"] == "UNKNOWN"
    assert Dec(a["risk_distance_pips_incl_allowances"]) > Dec(a["stop_distance_pips"])
    wide = engine(policy(sizing={"stop_slippage_allowance_pips": 5.0})).approve(proposal(), account(), T0)["approved_trade"]
    assert Dec(wide["approved_volume"]) <= Dec(a["approved_volume"])
    with pytest.raises(ValueError):
        policy(sizing={"assumed_commission_pips_round_turn": 0.0}).validate()  # never zero


def test_gap_risk_is_disclosed_never_a_guaranteed_maximum():
    a = engine().approve(proposal(), account(), T0)["approved_trade"]
    assert a["gap_risk_status"] == "NOT_BOUNDED_BY_STOP" and "NOT a guaranteed maximum loss" in a["risk_note"]
    assert a["planned_risk_currency"] == a["risk_amount"] and a["slippage_assumption"]["stop_pips"] >= 1.0
    r = position_remaining_risk(position(), GBPJPY_CONTRACT, Dec(1) / Dec(190))
    assert "NOT_BOUNDED_BY_STOP" in r["gap_risk_status"]


# ------------------------------------------------------------------ margin & leverage firewall
def test_leverage_changes_feasibility_never_risk_appetite():
    p = proposal()
    outs = {lev: engine().approve(p, account(lev=lev), T0) for lev in (30, 100, 500)}
    vols = {lev: o["approved_trade"]["approved_volume"] for lev, o in outs.items()}
    risks = {lev: o["approved_trade"]["risk_amount"] for lev, o in outs.items()}
    assert len(set(vols.values())) == 1 and len(set(risks.values())) == 1
    margins = [Dec(outs[lev]["approved_trade"]["estimated_margin"]) for lev in (30, 100, 500)]
    assert margins[0] > margins[1] > margins[2]


def test_margin_insufficient_unknown_and_reduce_policy():
    p = proposal()
    tight = account(balance="10000", free="600", used="9400", lev=30)
    rej = engine().approve(p, tight, T0)
    assert rej["rejection_category"] == "MARGIN_INSUFFICIENT" and "MARGIN_INSUFFICIENT" in rej["reason_codes"]
    unk = engine().approve(p, account(lev=None), T0)
    assert unk["rejection_category"] == "MARGIN_UNKNOWN"
    lenient = engine(policy(margin={"require_margin_known": False})).approve(p, account(lev=None), T0)
    assert lenient["decision"] == "RISK_APPROVED" and lenient["approved_trade"]["margin_status"] == "UNKNOWN"
    red = engine(policy(margin={"margin_policy": "REDUCE"}), margin_model=FixedMarginPerVolume(Dec("40000"))).approve(
        p, account(balance="10000", free="10000", used="0"), T0)
    a = red["approved_trade"]
    assert red["binding_constraint"] == "MARGIN_CAP" and Dec(a["approved_volume"]) < Dec("0.13")
    assert Dec(a["post_trade_free_margin"]) >= Dec("5000") and Dec(a["post_trade_margin_level"]) >= Dec("300")
    none = engine(margin_model=UnknownMargin()).approve(p, account(), T0)
    assert none["rejection_category"] == "MARGIN_UNKNOWN"


def test_partial_close_and_stop_movement_architecture():
    pos = position(volume="0.30", entry=190.0, stop=189.5, current_price=190.2)
    f = Dec(1) / Dec(190)
    before = position_remaining_risk(pos, GBPJPY_CONTRACT, f)["remaining_risk"]
    assert before == Dec("0.7") * Dec("100000") * Dec("0.30") * f
    moved = position_remaining_risk(pos, GBPJPY_CONTRACT, f, new_stop=190.1)
    assert moved["remaining_risk"] == Dec("0.1") * Dec("100000") * Dec("0.30") * f
    assert "NOT_BOUNDED_BY_STOP" in moved["gap_risk_status"]  # moving a stop never guarantees less realised risk
    assert partial_close_remaining(pos, Dec("0.10")) == Dec("0.20")
    with pytest.raises(ValueError):
        partial_close_remaining(pos, Dec("0.5"))
    assert position_remaining_risk(replace(pos, stop_price=None), GBPJPY_CONTRACT, f)["status"] == "UNBOUNDED_NO_STOP"
