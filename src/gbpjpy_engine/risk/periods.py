"""Deterministic day / week / month boundaries for risk accounting (Phase 1F).

Boundaries never use the local machine clock.  A trading day starts at
``rollover_hour`` in ``timezone`` (default 17:00 America/New_York - the common FX
day roll, DST-correct via zoneinfo).  The trading date of an instant is the
local date after shifting by ``24 - rollover_hour`` hours, so Monday 17:00 New
York starts Tuesday's trading day.  Weeks are ISO weeks of the trading date
(a new week starts at the Sunday roll); months are calendar months of the
trading date.  The weekend close is Friday at the roll hour.
"""

from __future__ import annotations

from dataclasses import dataclass
from zoneinfo import ZoneInfo

import pandas as pd


@dataclass(frozen=True)
class PeriodKeys:
    day: str
    week: str
    month: str
    trading_date: str


def trading_date(ts, tz: str, rollover_hour: int):
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    local = t.tz_convert(ZoneInfo(tz))
    return (local + pd.Timedelta(hours=(24 - rollover_hour) % 24)).date()


def period_keys(ts, tz: str = "America/New_York", rollover_hour: int = 17) -> PeriodKeys:
    d = trading_date(ts, tz, rollover_hour)
    iso = d.isocalendar()
    return PeriodKeys(day=d.isoformat(), week=f"{iso[0]}-W{iso[1]:02d}", month=f"{d.year}-{d.month:02d}",
                      trading_date=d.isoformat())


def hours_to_weekend_close(ts, tz: str = "America/New_York", rollover_hour: int = 17) -> float:
    """Hours until the coming Friday roll (0 or negative inside the weekend)."""
    t = pd.Timestamp(ts).tz_convert(ZoneInfo(tz))
    days_ahead = (4 - t.dayofweek) % 7
    friday = (t + pd.Timedelta(days=days_ahead)).normalize() + pd.Timedelta(hours=rollover_hour)
    if t.dayofweek == 4 and t >= friday or t.dayofweek in (5,) or (t.dayofweek == 6 and t.hour < rollover_hour):
        return 0.0
    return (friday - t).total_seconds() / 3600.0
