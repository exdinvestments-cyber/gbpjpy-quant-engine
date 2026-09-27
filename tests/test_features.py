from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.config import (
    CandleConfig,
    DataConfig,
    ExtensionConfig,
    RangeLocationConfig,
    RoundNumberConfig,
    SessionConfig,
    VolatilityConfig,
)
from gbpjpy_engine.features.candles import candle_features
from gbpjpy_engine.features.context import (
    extension_features,
    range_location_features,
    round_number_features,
    session_features,
    time_features,
)
from gbpjpy_engine.features.trend import trend_features
from gbpjpy_engine.features.volatility import volatility_features
from helpers import make_bars


def test_candle_geometry_and_classes():
    bars = make_bars(
        closes=[100.0, 101.0, 100.1, 100.02, 100.9],
        opens=[100.0, 100.0, 100.3, 100.0, 100.8],
        highs=[100.1, 101.05, 101.0, 100.41, 100.95],
        lows=[99.9, 99.95, 100.0, 99.61, 99.0],
    )
    prev_atr = pd.Series([np.nan, 0.5, 0.5, 0.5, 0.5])
    f = candle_features(bars, prev_atr, CandleConfig())
    r = f.iloc[1]
    assert r["candle_range"] == pytest.approx(1.10)
    assert r["candle_body"] == pytest.approx(1.0)
    assert r["upper_wick"] == pytest.approx(0.05)
    assert r["lower_wick"] == pytest.approx(0.05)
    assert r["range_atr"] == pytest.approx(2.2)
    assert r["candle_class"] == "strong_bullish_close"
    assert bool(r["abnormal_expansion"])
    assert f.iloc[2]["candle_class"] == "bearish_rejection"  # long upper wick, close near low
    assert f.iloc[3]["candle_class"] == "indecision"
    assert f.iloc[4]["candle_class"] == "bullish_rejection"
    assert 0 <= f["close_location"].min() and f["close_location"].max() <= 1


def test_volatility_percentile_and_regimes():
    n = 700
    rng = np.random.default_rng(1)
    closes = 190 + np.cumsum(rng.normal(0, 0.05, n))
    spread = np.where(np.arange(n) < 600, 0.2, 1.5)  # volatility jumps late
    bars = make_bars(closes, highs=closes + spread, lows=closes - spread)
    v = volatility_features(bars, VolatilityConfig())
    assert v["atr_percentile"].iloc[50] != v["atr_percentile"].iloc[50]  # NaN before min periods
    assert v["atr_percentile"].iloc[-1] >= 90
    assert v["volatility_regime"].iloc[-1] in ("high", "extreme")
    assert v["volatility_trend"].iloc[605] == "expanding"
    assert v["volatility_regime"].iloc[590] in ("very_low", "low", "normal")


def test_trend_features_bullish_on_rising_prices():
    closes = pd.Series(150 + np.arange(400) * 0.1)
    atr = pd.Series(np.full(400, 0.5))
    from gbpjpy_engine.config import TrendConfig

    t = trend_features(closes, atr, TrendConfig())
    assert t["trend_state"].iloc[-1] == "strong_bullish"
    assert 0 <= t["trend_strength"].iloc[-1] <= 100
    t2 = trend_features(pd.Series(150 - np.arange(400) * 0.1), atr, TrendConfig())
    assert t2["trend_state"].iloc[-1] == "strong_bearish"
    flat = trend_features(pd.Series(150 + 0.3 * np.sin(np.arange(400))), atr, TrendConfig())
    assert flat["trend_state"].iloc[-1] == "neutral"


def test_extension_calculation():
    n = 400
    closes = np.full(n, 190.0)
    closes[-5:] = [190.5, 191.0, 191.5, 192.0, 192.5]
    bars = make_bars(closes)
    atr = pd.Series(np.full(n, 0.5))
    from gbpjpy_engine.config import TrendConfig

    trend = trend_features(bars["close"], atr, TrendConfig())
    ext = extension_features(bars, trend, atr, ExtensionConfig())
    last = ext.iloc[-1]
    assert last["dist_ema_fast_atr"] == pytest.approx((192.5 - trend["ema_fast"].iloc[-1]) / 0.5)
    assert last["extension_direction"] == "up"
    assert last["extension_state"] in ("extended", "extremely_extended")
    assert ext["extension_state"].iloc[-10] == "normal"


def test_round_numbers():
    closes = pd.Series([191.87, 192.49, 192.51])
    atr = pd.Series([0.5, 0.5, 0.5])
    rn = round_number_features(closes, atr, RoundNumberConfig(), DataConfig())
    assert list(rn["nearest_round_number"]) == [192.0, 192.0, 193.0]
    assert rn["round_number_distance_pips"].iloc[0] == pytest.approx(-13.0)
    assert rn["round_number_distance_atr"].iloc[0] == pytest.approx(-0.26)
    assert list(rn["nearest_round_number_0p5"]) == [192.0, 192.5, 192.5]


def test_range_location():
    closes = list(np.linspace(100, 110, 30))
    bars = make_bars(closes)
    rl = range_location_features(bars, RangeLocationConfig(short=10, medium=20, long=30))
    assert rl["range_location_short"].iloc[-1] > 90
    bars2 = make_bars(list(np.linspace(110, 100, 30)))
    rl2 = range_location_features(bars2, RangeLocationConfig(short=10, medium=20, long=30))
    assert rl2["range_location_long"].iloc[-1] < 10
    assert rl["range_location_short"].iloc[:8].isna().all()


def test_sessions_account_for_dst():
    tf = pd.Timedelta(hours=4)
    ts = pd.Series(pd.to_datetime(["2024-01-10 08:00", "2024-07-10 08:00", "2024-07-10 12:00", "2024-01-10 00:00",
                                   "2024-07-10 04:00"], utc=True))
    s = session_features(ts, SessionConfig(), tf)
    # Winter: London opens 08:00 UTC -> 08-12 UTC bar fully London (240 min)
    assert s["session_london_minutes"].iloc[0] == 240
    # Summer (BST): London 07:00-16:00 UTC -> 08-12 UTC still fully London
    assert s["session_london_minutes"].iloc[1] == 240
    # Summer: New York 12:00-21:00 UTC -> 12-16 UTC bar = London/NY overlap
    assert "london_new_york_overlap" in s["session_overlap"].iloc[2]
    assert s["session_new_york_minutes"].iloc[2] == 240
    # Tokyo 00:00-09:00 UTC (no DST) -> 00-04 UTC is Asia
    assert s["primary_session"].iloc[3] == "asia"
    # Summer 04-08 UTC: London opens 07:00 UTC -> 60 min London, overlap with Tokyo
    assert s["session_london_minutes"].iloc[4] == 60
    assert "asia_london_overlap" in s["session_overlap"].iloc[4]


def test_winter_vs_summer_new_york_open_shift():
    tf = pd.Timedelta(hours=4)
    ts = pd.Series(pd.to_datetime(["2024-01-10 12:00", "2024-07-10 12:00"], utc=True))
    s = session_features(ts, SessionConfig(), tf)
    assert s["session_new_york_minutes"].iloc[0] == 180  # EST: NY 13:00 UTC open
    assert s["session_new_york_minutes"].iloc[1] == 240  # EDT: NY 12:00 UTC open


def test_time_features():
    ts = pd.Series(pd.to_datetime(["2024-05-17 20:00"], utc=True))
    t = time_features(ts).iloc[0]
    assert (t["day_of_week"], t["hour_utc"], t["month"], t["quarter"], t["year"]) == (4, 20, 5, 2, 2024)
    assert t["day_name"] == "Friday"
