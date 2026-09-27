from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.data import (
    Bar,
    DataIntegrityError,
    bars_to_frame,
    exclude_unclosed_bars,
    to_canonical,
    validate_bars,
)
from helpers import make_bars


def _types(report):
    return {i.issue_type for i in report.issues}


def test_clean_data_is_ok():
    rep = validate_bars(make_bars(185 + np.arange(50) * 0.01))
    assert rep.summary()["status"] == "OK"
    assert not rep.has_errors


def test_invalid_ohlc_flagged_not_repaired():
    bars = make_bars(185 + np.arange(30) * 0.01)
    bars.loc[10, "high"] = bars.loc[10, "low"] - 0.5  # high below low
    before = bars.copy()
    rep = validate_bars(bars)
    assert "INVALID_OHLC" in _types(rep)
    assert "INVALID_OHLC" in rep.bar_flags[10]
    pd.testing.assert_frame_equal(bars, before)  # never modified


def test_nonpositive_and_missing_prices():
    bars = make_bars(185 + np.arange(30) * 0.01)
    bars.loc[3, "low"] = 0.0
    bars.loc[5, "close"] = np.nan
    rep = validate_bars(bars)
    assert {"NONPOSITIVE_PRICE", "MISSING_PRICE"} <= _types(rep)
    assert rep.has_errors


def test_duplicates_and_out_of_order():
    bars = make_bars(185 + np.arange(30) * 0.01)
    dup = pd.concat([bars.iloc[:10], bars.iloc[[9]], bars.iloc[10:]], ignore_index=True)
    rep = validate_bars(dup)
    assert "DUPLICATE_TIMESTAMP" in _types(rep)
    swapped = bars.copy()
    swapped.loc[[5, 6]] = swapped.loc[[6, 5]].to_numpy()
    rep2 = validate_bars(swapped)
    assert "OUT_OF_ORDER" in _types(rep2)


def test_engine_refuses_error_data(engine):
    bars = make_bars(185 + np.arange(30) * 0.01)
    bars = pd.concat([bars, bars.iloc[[5]]], ignore_index=True)
    with pytest.raises(DataIntegrityError) as ei:
        engine.run(bars)
    assert ei.value.report is not None and ei.value.report.has_errors


def test_missing_bars_vs_weekend_gap():
    bars = make_bars(185 + np.arange(60) * 0.01)  # grid already skips weekends
    rep = validate_bars(bars)
    assert "MISSING_BARS" not in _types(rep)
    holed = bars.drop(index=[20, 21]).reset_index(drop=True)  # midweek hole
    rep2 = validate_bars(holed)
    assert "MISSING_BARS" in _types(rep2)


def test_abnormal_gap_flagged():
    bars = make_bars(185 + np.arange(60) * 0.01)
    bars.loc[40:, ["open", "high", "low", "close"]] += 3.0  # jump between close[39] and open[40]
    rep = validate_bars(bars)
    assert "ABNORMAL_PRICE_GAP" in rep.bar_flags[40]


def test_missing_volume_and_spread_flagged():
    bars = make_bars(185 + np.arange(20) * 0.01, volume=False, spread=False)
    rep = validate_bars(bars)
    assert {"MISSING_VOLUME", "MISSING_SPREAD"} <= _types(rep)
    assert not rep.has_errors
    bars2 = make_bars(185 + np.arange(20) * 0.01)
    bars2.loc[4, "volume"] = np.nan
    assert "MISSING_VOLUME" in validate_bars(bars2).bar_flags[4]


def test_naive_timestamps_rejected_without_explicit_timezone():
    raw = pd.DataFrame(
        {"time": pd.date_range("2024-01-02", periods=5, freq="4h"), "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0}
    )
    with pytest.raises(DataIntegrityError):
        to_canonical(raw)
    df = to_canonical(raw, assume_timezone="UTC")
    assert str(df["timestamp"].dt.tz) == "UTC"


def test_broker_timezone_conversion_handles_dst():
    # Broker server time (Europe/Athens, UTC+2 winter / UTC+3 summer) with H4 bars at 00,04,...
    idx = pd.date_range("2024-03-28 00:00", "2024-04-03 20:00", freq="4h")
    idx = idx[idx.dayofweek < 5]
    raw = pd.DataFrame({"timestamp": idx, "open": 190.0, "high": 190.2, "low": 189.8, "close": 190.0})
    df = to_canonical(raw, assume_timezone="Europe/Athens")
    hours = set(df["timestamp"].dt.hour)
    assert 22 in hours and 21 in hours  # before DST: 22:00 UTC; after: 21:00 UTC
    rep = validate_bars(df)
    # the grid shift is surfaced as a timezone-consistency warning, not silently accepted
    assert "TIMEZONE_ALIGNMENT_SHIFT" in _types(rep)


def test_non_utc_frame_is_error():
    bars = make_bars(185 + np.arange(10) * 0.01)
    bars["timestamp"] = bars["timestamp"].dt.tz_convert("Europe/London")
    assert "TIMEZONE_NOT_UTC" in _types(validate_bars(bars))


def test_bar_dataclass_roundtrip():
    b = [
        Bar(pd.Timestamp("2024-01-02 00:00", tz="UTC"), 190.0, 190.5, 189.5, 190.2, 1000, 2.0, "live"),
        Bar(pd.Timestamp("2024-01-02 04:00", tz="UTC"), 190.2, 190.6, 190.0, 190.4, 900, 2.1, "live"),
    ]
    df = bars_to_frame(b)
    assert list(df["source"]) == ["live", "live"]
    assert not validate_bars(df).has_errors


def test_exclude_unclosed_bars():
    bars = make_bars(185 + np.arange(10) * 0.01)
    as_of = bars["timestamp"].iloc[-1] + pd.Timedelta(hours=3, minutes=59)
    assert len(exclude_unclosed_bars(bars, as_of)) == 9
    as_of2 = bars["timestamp"].iloc[-1] + pd.Timedelta(hours=4)
    assert len(exclude_unclosed_bars(bars, as_of2)) == 10
    with pytest.raises(DataIntegrityError):
        exclude_unclosed_bars(bars, pd.Timestamp("2024-01-01"))
