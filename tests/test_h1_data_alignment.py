"""Phase 1C: H1 data model/validation, H1->H4 aggregation and point-in-time H4/H1 alignment."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.data import DataIntegrityError, validate_bars
from gbpjpy_engine.data.model import to_canonical
from gbpjpy_engine.data.resample import resample_complete
from gbpjpy_engine.h1 import H1Config, H1SetupEngine, align_h4_to_h1
from gbpjpy_engine.h1.config import H1AlignmentConfig, h1_config_from_dict
from gbpjpy_engine.synthetic_h1 import h1_weekday_grid

H1CFG = H1Config()


def h1_bars(n=48, start="2024-01-02 00:00", price=190.0):
    t = h1_weekday_grid(n, start)
    c = price + np.cumsum(np.sin(np.arange(n)) * 0.05)
    o = np.concatenate([[price], c[:-1]])
    raw = pd.DataFrame({"timestamp": t, "open": o, "high": np.maximum(o, c) + 0.03, "low": np.minimum(o, c) - 0.03,
                        "close": c, "volume": 100.0, "spread": 2.0})
    return to_canonical(raw, source="test")


def fake_h4(opens):
    t = pd.DatetimeIndex(pd.to_datetime(opens, utc=True))
    f = pd.DataFrame({"timestamp": t, "available_at": t + pd.Timedelta(hours=4), "regime": [f"R{i}" for i in range(len(t))]})
    return SimpleNamespace(features=f, context=None)


# ------------------------------------------------------------------ H1 data model
def test_h1_config_and_validation_reuse():
    assert H1CFG.data.timeframe_minutes == 60
    with pytest.raises(ValueError):
        h1_config_from_dict({"data": {"timeframe_minutes": 240}})
    bars = h1_bars()
    rep = validate_bars(bars, H1CFG.data)
    assert not rep.has_errors and "TIMEZONE_ALIGNMENT_SHIFT" not in {i.issue_type for i in rep.issues}
    holed = bars.drop(index=[10, 11]).reset_index(drop=True)
    assert "MISSING_BARS" in {i.issue_type for i in validate_bars(holed, H1CFG.data).issues}
    dup = pd.concat([bars, bars.iloc[[3]]], ignore_index=True)
    assert "DUPLICATE_TIMESTAMP" in {i.issue_type for i in validate_bars(dup, H1CFG.data).issues}
    bad = bars.copy()
    bad.loc[5, "high"] = bad.loc[5, "low"] - 1
    assert "INVALID_OHLC" in {i.issue_type for i in validate_bars(bad, H1CFG.data).issues}
    shifted = bars.copy()
    shifted.loc[20:, "timestamp"] = shifted.loc[20:, "timestamp"] + pd.Timedelta(minutes=30)
    assert "TIMEZONE_ALIGNMENT_SHIFT" in {i.issue_type for i in validate_bars(shifted, H1CFG.data).issues}


def test_h1_engine_refuses_invalid_data(h1_engine):
    bars = h1_bars(60)
    bad = pd.concat([bars, bars.iloc[[5]]], ignore_index=True)
    with pytest.raises(DataIntegrityError):
        h1_engine.run(bad, SimpleNamespace(features=pd.DataFrame(), context=None))
    naive = bars.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(DataIntegrityError):
        h1_engine.run(naive, SimpleNamespace(features=pd.DataFrame(), context=None))


def test_resample_complete_buckets_only():
    bars = h1_bars(24)
    h4, dropped = resample_complete(bars, 60, 240)
    assert len(h4) == 6 and not dropped
    first = bars.iloc[:4]
    r = h4.iloc[0]
    assert r["open"] == first["open"].iloc[0] and r["close"] == first["close"].iloc[-1]
    assert r["high"] == first["high"].max() and r["low"] == first["low"].min() and r["volume"] == 400.0
    holed = bars.drop(index=[5]).reset_index(drop=True)
    h4b, dropped_b = resample_complete(holed, 60, 240)
    assert len(h4b) == 5 and dropped_b == [pd.Timestamp("2024-01-02 04:00", tz="UTC")]  # never filled
    partial, dp = resample_complete(bars.iloc[:6], 60, 240)  # 2nd H4 bucket still forming -> not emitted
    assert len(partial) == 1 and len(dp) == 1
    off, _ = resample_complete(bars.iloc[1:], 60, 240, offset="1h")  # broker grid opening at 01:00 UTC
    assert off["timestamp"].iloc[0] == pd.Timestamp("2024-01-02 01:00", tz="UTC")


# ------------------------------------------------------------------ alignment
def _align(h1_opens, h4_opens, cfg=H1AlignmentConfig()):
    h1 = pd.DatetimeIndex(pd.to_datetime(h1_opens, utc=True))
    return align_h4_to_h1(pd.Series(h1 + pd.Timedelta(hours=1)), fake_h4(h4_opens), cfg)


def test_h1_sees_only_completed_h4():
    h1 = pd.date_range("2024-01-02 00:00", periods=12, freq="1h", tz="UTC")
    a = _align(h1, ["2024-01-02 00:00", "2024-01-02 04:00", "2024-01-02 08:00"])
    # H1 bars closing 01:00-03:00 are inside the first (unfinished) H4 candle -> nothing to see
    assert list(a["h4_index"][:3]) == [-1, -1, -1] and set(a["h4_context_status"][:3]) == {"NONE"}
    # the H1 bar closing 04:00 closes together with the first H4 candle -> may see it
    assert a["h4_index"].iloc[3] == 0
    # H1 bars inside the second H4 candle still see the FIRST candle only
    assert list(a["h4_index"][4:7]) == [0, 0, 0] and a["h4_index"].iloc[7] == 1
    assert (a["h4_available_at"].dropna() <= (h1 + pd.Timedelta(hours=1))[a["h4_index"] >= 0]).all()
    assert list(a["h4_context_age_hours"][3:8]) == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_missing_h4_bars_make_context_stale_not_future():
    h1 = pd.date_range("2024-01-02 00:00", periods=20, freq="1h", tz="UTC")
    a = _align(h1, ["2024-01-02 00:00", "2024-01-02 12:00"])  # 04:00 and 08:00 H4 candles missing
    assert a["h4_index"].iloc[10] == 0  # 10:00-11:00 H1 still sees the 00:00 H4 bar, never the 12:00 one
    assert a["h4_context_status"].iloc[12] == "STALE"  # 12:00-13:00 H1: last completed H4 closed 04:00 (9h ago)
    assert a["h4_index"].iloc[15] == 1


def test_dst_shifted_broker_h4_grid():
    # after a DST change the broker's H4 candles open at 21:00, 01:00, 05:00 UTC
    h1 = pd.date_range("2024-03-31 21:00", periods=12, freq="1h", tz="UTC")
    a = _align(h1, ["2024-03-31 21:00", "2024-04-01 01:00", "2024-04-01 05:00"])
    assert list(a["h4_index"][:3]) == [-1, -1, -1]
    assert a["h4_index"].iloc[3] == 0 and a["h4_available_at"].iloc[3] == pd.Timestamp("2024-04-01 01:00", tz="UTC")
    assert list(a["h4_index"][4:7]) == [0, 0, 0] and a["h4_index"].iloc[7] == 1


def test_irregular_h4_timestamp_joined_by_close_time():
    h1 = pd.date_range("2024-01-02 00:00", periods=8, freq="1h", tz="UTC")
    a = _align(h1, ["2024-01-02 02:00"])  # irregular candle 02:00-06:00
    assert list(a["h4_index"]) == [-1, -1, -1, -1, -1, 0, 0, 0]


def test_alignment_error_fails_safe(monkeypatch, h1_results):
    import gbpjpy_engine.h1.engine as eng
    from gbpjpy_engine.h1.alignment import AlignmentError

    sc, h4, _ = h1_results["h1_range"]

    def boom(*a, **k):
        raise AlignmentError("simulated")

    monkeypatch.setattr(eng, "align_h4_to_h1", boom)
    res = H1SetupEngine().run(sc.h1.iloc[:1300].reset_index(drop=True), h4)
    f = res.frame
    assert (~f["actionable_setup_permission_long"]).all() and (~f["actionable_setup_permission_short"]).all()
    assert all(b == ["H4_ALIGNMENT_ERROR"] for b in f["h1_blockers"])
    assert res.setups == [] and res.qualified_setups == []


def test_engine_alignment_columns(h1_results):
    sc, h4, res = h1_results["h1_uptrend_pullbacks"]
    f = res.frame
    known = f["h4_index"] >= 0
    h4_close = h4.features["available_at"].to_numpy()[f.loc[known, "h4_index"].to_numpy()]
    assert (h4_close <= f.loc[known, "available_at"].to_numpy()).all()
    # every H1 bar sees the LATEST completed H4 bar (nothing more recent, nothing older than necessary)
    closes = h4.features["available_at"]
    for i in range(1300, len(f), 97):
        expected = int(np.searchsorted(closes.to_numpy(), f["available_at"].to_numpy()[i], side="right") - 1)
        assert f["h4_index"].iloc[i] == expected
    assert set(f["h4_context_status"]) <= {"OK", "NONE", "STALE"}
