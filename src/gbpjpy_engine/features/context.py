"""Location and time context: round numbers, range location, extension,
trading sessions and calendar metadata.  All descriptive; none of these
features implies a trade direction on its own."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from ..config import DataConfig, ExtensionConfig, RangeLocationConfig, RoundNumberConfig, SessionConfig
from .indicators import rolling_percentile

# ---------------------------------------------------------------------------
# Round numbers
# ---------------------------------------------------------------------------


def round_number_features(close: pd.Series, atr_s: pd.Series, cfg: RoundNumberConfig, data_cfg: DataConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=close.index)
    for k, interval in enumerate(cfg.intervals):
        rn = (close / interval).round() * interval
        tag = "" if k == 0 else f"_{str(interval).replace('.', 'p')}"
        out[f"nearest_round_number{tag}"] = rn.round(3)
        out[f"round_number_distance_pips{tag}"] = ((close - rn) / data_cfg.pip_size).round(1)
        out[f"round_number_distance_atr{tag}"] = (close - rn) / atr_s
    return out


# ---------------------------------------------------------------------------
# Range location
# ---------------------------------------------------------------------------


def range_location_features(df: pd.DataFrame, cfg: RangeLocationConfig) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for name, n in (("short", cfg.short), ("medium", cfg.medium), ("long", cfg.long)):
        hi = df["high"].rolling(n, min_periods=n).max()
        lo = df["low"].rolling(n, min_periods=n).min()
        out[f"range_location_{name}"] = ((df["close"] - lo) / (hi - lo).replace(0.0, np.nan) * 100.0).round(2)
    return out


# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------


def extension_features(df: pd.DataFrame, trend: pd.DataFrame, atr_s: pd.Series, cfg: ExtensionConfig) -> pd.DataFrame:
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    out["dist_ema_fast_atr"] = (c - trend["ema_fast"]) / atr_s
    out["dist_ema_mid_atr"] = (c - trend["ema_mid"]) / atr_s
    n = cfg.impulse_window
    up_imp = (c - df["low"].rolling(n, min_periods=n).min()) / atr_s
    dn_imp = (df["high"].rolling(n, min_periods=n).max() - c) / atr_s
    out["impulse_up_atr"] = up_imp
    out["impulse_down_atr"] = dn_imp

    # composite distance: EMA20 distance dominates, EMA50 distance contributes at half scale
    metric = 0.6 * out["dist_ema_fast_atr"].abs() + 0.4 * out["dist_ema_mid_atr"].abs() * 0.5
    out["extension_metric_atr"] = metric
    pct = rolling_percentile(metric, cfg.percentile_lookback, cfg.percentile_min_periods)
    out["extension_percentile"] = pct
    absolute = np.clip(metric / cfg.absolute_norm_atr, 0.0, 1.0) * 100.0
    score = np.where(pct.isna(), absolute, 0.5 * pct + 0.5 * absolute)
    score = pd.Series(score, index=df.index).where(metric.notna())
    out["extension_score"] = score.round(2)
    direction = np.sign(out["dist_ema_fast_atr"])
    out["extension_direction"] = np.select([direction > 0, direction < 0], ["up", "down"], default="flat")
    out["extension_state"] = np.select(
        [score.isna(), score >= cfg.extreme_score, score >= cfg.extended_score],
        ["unknown", "extremely_extended", "extended"],
        default="normal",
    )
    return out


# ---------------------------------------------------------------------------
# Sessions (DST-aware via zoneinfo)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=65536)
def _session_window_utc(tz_name: str, day: datetime, start: str, end: str) -> tuple[datetime, datetime]:
    tz = ZoneInfo(tz_name)
    sh, sm = map(int, start.split(":"))
    eh, em = map(int, end.split(":"))
    s = datetime.combine(day.date(), time(sh, sm), tzinfo=tz)
    e = datetime.combine(day.date(), time(eh, em), tzinfo=tz)
    if e <= s:
        e = e + timedelta(days=1)
    utc = ZoneInfo("UTC")
    return s.astimezone(utc), e.astimezone(utc)


def _windows_for_bar(tz_name: str, bar_start: datetime, bar_end: datetime, start: str, end: str) -> list[tuple[datetime, datetime]]:
    tz = ZoneInfo(tz_name)
    local_day = bar_start.astimezone(tz).replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    wins = []
    for k in (-1, 0, 1):
        d = local_day + timedelta(days=k)
        s, e = _session_window_utc(tz_name, d, start, end)
        if s < bar_end and e > bar_start:
            wins.append((max(s, bar_start), min(e, bar_end)))
    return wins


def _minutes(wins: list[tuple[datetime, datetime]]) -> float:
    return sum((e - s).total_seconds() for s, e in wins) / 60.0


def _intersect(a: list[tuple[datetime, datetime]], b: list[tuple[datetime, datetime]]) -> float:
    tot = 0.0
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e > s:
                tot += (e - s).total_seconds()
    return tot / 60.0


def session_features(timestamps: pd.Series, cfg: SessionConfig, timeframe: pd.Timedelta) -> pd.DataFrame:
    names = [s[0] for s in cfg.sessions]
    rows = []
    for t in timestamps:
        bs = t.to_pydatetime()
        be = bs + timeframe.to_pytimedelta()
        wins = {name: _windows_for_bar(tz, bs, be, st, en) for name, tz, st, en in cfg.sessions}
        mins = {name: _minutes(w) for name, w in wins.items()}
        active = [nm for nm in names if mins[nm] >= cfg.min_overlap_minutes]
        overlaps = []
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if _intersect(wins[a], wins[b]) >= cfg.min_overlap_minutes:
                    overlaps.append(f"{a}_{b}_overlap")
        primary = max(names, key=lambda nm: (mins[nm], -names.index(nm))) if max(mins.values()) > 0 else "off_session"
        row = {f"session_{nm}_minutes": mins[nm] for nm in names}
        row["primary_session"] = primary
        row["active_sessions"] = "|".join(active) if active else "none"
        row["session_overlap"] = "|".join(overlaps) if overlaps else "none"
        rows.append(row)
    return pd.DataFrame(rows, index=timestamps.index)


# ---------------------------------------------------------------------------
# Calendar metadata
# ---------------------------------------------------------------------------


def time_features(timestamps: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=timestamps.index)
    out["day_of_week"] = timestamps.dt.dayofweek
    out["day_name"] = timestamps.dt.day_name()
    out["hour_utc"] = timestamps.dt.hour
    out["month"] = timestamps.dt.month
    out["quarter"] = timestamps.dt.quarter
    out["year"] = timestamps.dt.year
    return out
