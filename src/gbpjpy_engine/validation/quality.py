"""Research data-quality report and gap policy (Phase 1H).

Extends ``data.validation.validate_bars`` (duplicates, ordering, OHLC
consistency, non-positive prices, grid/DST alignment, missing bars, abnormal
gaps, missing volume/spread) with research checks: weekend bars, DST
irregularities, implausible prices, price spikes, abnormal and zero spreads,
stale sequences, flat and repeated bars, and a classified gap list.

Nothing is repaired.  ``QualityGate`` decides whether a backtest may begin:
any ERROR blocks it (never silently), warnings are carried into every result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..data.validation import validate_bars

GAP_CLASSES = ("EXPECTED_MARKET_CLOSURE", "SHORT_UNEXPLAINED_GAP", "MAJOR_MISSING_INTERVAL", "PROVIDER_OUTAGE", "UNKNOWN")


@dataclass(frozen=True)
class QualityPolicy:
    """Research data-quality thresholds - CONFIGURATION, NOT VALIDATED EDGE (external plausibility only)."""

    plausible_min_price: float = 80.0  # GBPJPY has not traded outside roughly 110-260 in modern history
    plausible_max_price: float = 400.0
    spike_range_mult: float = 15.0  # bar range vs trailing median range
    abnormal_spread_pips: float = 30.0
    abnormal_spread_mult: float = 5.0  # vs trailing median spread
    stale_run_bars: int = 5  # consecutive identical closes
    max_weekend_gap_hours: float = 72.0
    short_gap_bars: int = 3
    major_gap_hours: float = 24.0
    blackout_bars_after_gap: int = 24  # no NEW decisions for this many bars after an unresolved gap
    holidays_mmdd: tuple = ("12-25", "01-01")


@dataclass
class GapRecord:
    index: int  # bar AFTER the gap
    start: str  # last bar before the gap (open time)
    end: str  # first bar after the gap
    hours: float
    missing_bars: int
    classification: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ResearchQualityReport:
    timeframe: str
    n_bars: int
    base: dict  # summary of data.validation
    issues: list = field(default_factory=list)  # (type, severity, index, timestamp, detail)
    gaps: list = field(default_factory=list)
    blackout: list = field(default_factory=list)  # bar indices where new decisions are blocked
    dst_summary: dict = field(default_factory=dict)
    status: str = "OK"  # OK | WARNINGS | BLOCKED

    @property
    def counts(self) -> dict:
        c: dict = dict(self.base.get("counts", {}))
        for t, *_ in self.issues:
            c[t] = c.get(t, 0) + 1
        g: dict = {}
        for x in self.gaps:
            g[x.classification] = g.get(x.classification, 0) + 1
        c.update({f"GAP_{k}": v for k, v in g.items()})
        return c

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i[1] == "ERROR"] + (["BASE_ERRORS"] if self.base.get("errors") else [])

    def summary(self) -> dict:
        return {"timeframe": self.timeframe, "n_bars": self.n_bars, "status": self.status, "counts": self.counts,
                "gaps": {k: sum(g.classification == k for g in self.gaps) for k in GAP_CLASSES},
                "blackout_bars": len(self.blackout), "dst": self.dst_summary}


def classify_gaps(df: pd.DataFrame, timeframe_minutes: int, policy: QualityPolicy, known_outages=()) -> list[GapRecord]:
    ts = df["timestamp"]
    step = pd.Timedelta(minutes=timeframe_minutes)
    out = []
    outages = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b, *_ in known_outages]
    for i in range(1, len(df)):
        d = ts.iloc[i] - ts.iloc[i - 1]
        if d <= step:
            continue
        a, b = ts.iloc[i - 1], ts.iloc[i]
        hours = d.total_seconds() / 3600
        missing = int(d / step) - 1
        days = pd.date_range(a.normalize(), b.normalize(), freq="D")
        spans_weekend = bool((days.dayofweek == 5).any())
        holiday = any(x.strftime("%m-%d") in policy.holidays_mmdd for x in days)
        if any(a < oe and b > os_ for os_, oe in outages):
            cls = "PROVIDER_OUTAGE"
        elif spans_weekend and hours <= policy.max_weekend_gap_hours:
            cls = "EXPECTED_MARKET_CLOSURE"
        elif holiday and hours <= 48:
            cls = "EXPECTED_MARKET_CLOSURE"
        elif missing <= policy.short_gap_bars:
            cls = "SHORT_UNEXPLAINED_GAP"
        elif hours >= policy.major_gap_hours:
            cls = "MAJOR_MISSING_INTERVAL"
        else:
            cls = "UNKNOWN"
        out.append(GapRecord(i, a.isoformat(), b.isoformat(), round(hours, 2), missing, cls))
    return out


def research_quality_report(df: pd.DataFrame, timeframe: str = "H1", policy: QualityPolicy | None = None,
                            known_outages=(), spread_expected: bool = False) -> ResearchQualityReport:
    """Full pre-backtest data-quality report.  ``df`` is canonical and never modified."""
    pol = policy or QualityPolicy()
    tfm = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240}[timeframe]
    base = validate_bars(df, DataConfig(timeframe_minutes=tfm, max_weekend_gap_hours=pol.max_weekend_gap_hours))
    rep = ResearchQualityReport(timeframe, len(df), {"counts": base.counts(), "errors": bool(base.has_errors),
                                                     "status": base.summary()["status"],
                                                     "error_details": [e.to_dict() for e in base.errors][:100]})
    n = len(df)
    if n == 0:
        rep.status = "BLOCKED"
        return rep
    ts = df["timestamp"]
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))

    def add(t, sev, i, detail):
        rep.issues.append((t, sev, None if i is None else int(i), None if i is None else ts.iloc[int(i)].isoformat(), detail))

    # implausible prices / spikes
    with np.errstate(invalid="ignore"):
        bad = (l < pol.plausible_min_price) | (h > pol.plausible_max_price)
    for i in np.flatnonzero(bad):
        add("IMPLAUSIBLE_PRICE", "ERROR", i, f"l={l[i]} h={h[i]} outside [{pol.plausible_min_price}, {pol.plausible_max_price}]")
    rng = pd.Series(h - l)
    ref = rng.rolling(200, min_periods=20).median().shift(1).to_numpy()
    for i in np.flatnonzero(~np.isnan(ref) & (rng.to_numpy() > pol.spike_range_mult * ref)):
        add("PRICE_SPIKE", "WARNING", i, f"range {rng.iloc[i]:.3f} vs trailing median {ref[i]:.3f}")
    # weekend bars (UTC): Saturday, Sunday before 20:00, Friday from 22:00
    dow, hr = ts.dt.dayofweek.to_numpy(), ts.dt.hour.to_numpy()
    wk = (dow == 5) | ((dow == 6) & (hr < 20)) | ((dow == 4) & (hr >= 22))
    for i in np.flatnonzero(wk):
        add("WEEKEND_BAR", "WARNING", i, "bar inside the usual weekend closure (UTC)")
    # flat / repeated / stale
    flat = (o == h) & (h == l) & (l == c)
    for i in np.flatnonzero(flat):
        add("FLAT_BAR", "WARNING", i, "open = high = low = close")
    rep_ = np.zeros(n, bool)
    rep_[1:] = (o[1:] == o[:-1]) & (h[1:] == h[:-1]) & (l[1:] == l[:-1]) & (c[1:] == c[:-1])
    for i in np.flatnonzero(rep_):
        add("REPEATED_BAR", "WARNING", i, "identical OHLC to the previous bar")
    run = 1
    for i in range(1, n):
        run = run + 1 if c[i] == c[i - 1] else 1
        if run == pol.stale_run_bars:
            add("STALE_SEQUENCE", "WARNING", i, f"{run} consecutive identical closes")
    # spreads
    spr = df["spread"].to_numpy(float) if "spread" in df.columns else np.full(n, np.nan)
    if np.isfinite(spr).any():
        med = pd.Series(spr).rolling(500, min_periods=20).median().shift(1).to_numpy()
        with np.errstate(invalid="ignore"):
            abn = (spr > pol.abnormal_spread_pips) | (np.isfinite(med) & (spr > pol.abnormal_spread_mult * med))
        for i in np.flatnonzero(abn):
            add("ABNORMAL_SPREAD", "WARNING", i, f"spread {spr[i]:.2f} pips")
        for i in np.flatnonzero(spr == 0):
            add("ZERO_SPREAD", "WARNING", i, "zero spread reported (suspicious; never used as a cost-free fill)")
    elif spread_expected:
        add("MISSING_SPREAD_DATA", "WARNING", None, "spread declared available but no values present")
    # DST / grid irregularities (reported, never corrected)
    shifts = [k for k in base.issues if k.issue_type == "TIMEZONE_ALIGNMENT_SHIFT"]
    week_open = df.assign(_w=ts.dt.tz_convert(None).dt.to_period("W-SUN")).groupby("_w")["timestamp"].min()
    open_hours = week_open.dt.hour.value_counts().sort_index()
    rep.dst_summary = {"grid_alignment_shifts": len(shifts),
                       "week_open_hour_utc_distribution": {int(k): int(v) for k, v in open_hours.items()},
                       "note": "a seasonal 1h change in the weekly open hour usually reflects DST in the provider clock; "
                               "it is reported, not corrected"}
    if len(open_hours) > 1:
        add("DST_IRREGULARITY", "WARNING", None, f"weekly open hour varies: {dict(open_hours)}")
    # gaps + blackout (fail safe around unresolved gaps)
    rep.gaps = classify_gaps(df, tfm, pol, known_outages)
    black = set()
    for g in rep.gaps:
        if g.classification in ("MAJOR_MISSING_INTERVAL", "PROVIDER_OUTAGE", "UNKNOWN"):
            black.update(range(g.index, min(n, g.index + pol.blackout_bars_after_gap)))
    rep.blackout = sorted(black)
    rep.status = "BLOCKED" if rep.errors else ("WARNINGS" if rep.issues or rep.base["status"] != "OK" or rep.gaps else "OK")
    return rep


class DataQualityBlocked(RuntimeError):
    def __init__(self, report: ResearchQualityReport):
        super().__init__(f"data quality BLOCKED the backtest: {report.counts}")
        self.report = report


def quality_gate(report: ResearchQualityReport) -> ResearchQualityReport:
    """Raise unless the data may be used; a backtest never begins silently on corrupt data."""
    if report.status == "BLOCKED":
        raise DataQualityBlocked(report)
    return report
