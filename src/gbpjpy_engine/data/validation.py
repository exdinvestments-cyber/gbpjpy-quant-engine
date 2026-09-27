"""Data-integrity validation for canonical H4 bars.

Validation FLAGS problems; it never modifies the data.  ERROR-severity
issues make the data unsafe for the engine (which refuses to run by default).
WARNING-severity issues are attached to the affected bars and propagate to
each snapshot's ``data_quality_status``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DataConfig
from .model import PRICE_COLUMNS

logger = logging.getLogger(__name__)

ERROR = "ERROR"
WARNING = "WARNING"


@dataclass(frozen=True)
class DataIssue:
    issue_type: str
    severity: str
    index: int | None
    timestamp: str | None
    detail: str

    def to_dict(self) -> dict:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity,
            "index": self.index,
            "timestamp": self.timestamp,
            "detail": self.detail,
        }


@dataclass
class DataQualityReport:
    n_bars: int
    issues: list[DataIssue] = field(default_factory=list)
    bar_flags: list[list[str]] = field(default_factory=list)

    @property
    def errors(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> list[DataIssue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    def counts(self) -> dict[str, int]:
        return dict(Counter(i.issue_type for i in self.issues))

    def summary(self) -> dict:
        return {
            "n_bars": self.n_bars,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "counts": self.counts(),
            "status": "ERROR" if self.has_errors else ("WARNING" if self.warnings else "OK"),
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.to_dict() for i in self.issues], columns=["issue_type", "severity", "index", "timestamp", "detail"])


def _ts(df: pd.DataFrame, i: int) -> str | None:
    try:
        return pd.Timestamp(df["timestamp"].iloc[i]).isoformat()
    except Exception:  # pragma: no cover - defensive
        return None


def validate_bars(df: pd.DataFrame, cfg: DataConfig | None = None) -> DataQualityReport:
    """Validate canonical bars and return a report.  ``df`` is not modified."""
    cfg = cfg or DataConfig()
    n = len(df)
    report = DataQualityReport(n_bars=n, bar_flags=[[] for _ in range(n)])

    def add(issue_type: str, severity: str, i: int | None, detail: str) -> None:
        report.issues.append(DataIssue(issue_type, severity, i, _ts(df, i) if i is not None else None, detail))
        if i is not None:
            report.bar_flags[i].append(issue_type)

    if n == 0:
        add("EMPTY_DATA", ERROR, None, "no bars supplied")
        return report

    # ---- timezone ------------------------------------------------------
    ts = df["timestamp"]
    tz = getattr(ts.dt, "tz", None)
    if tz is None:
        add("TIMEZONE_NAIVE", ERROR, None, "timestamps are timezone-naive")
        return _finish(report)
    if str(tz) != "UTC":
        add("TIMEZONE_NOT_UTC", ERROR, None, f"timestamps are in {tz}; canonical form requires UTC")

    # ---- prices ----------------------------------------------------------
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in PRICE_COLUMNS)
    for i in np.flatnonzero(np.isnan(o) | np.isnan(h) | np.isnan(l) | np.isnan(c)):
        add("MISSING_PRICE", ERROR, int(i), "one or more OHLC values missing")
    with np.errstate(invalid="ignore"):
        nonpos = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
    for i in np.flatnonzero(nonpos):
        add("NONPOSITIVE_PRICE", ERROR, int(i), f"o={o[i]} h={h[i]} l={l[i]} c={c[i]}")
    with np.errstate(invalid="ignore"):
        bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    for i in np.flatnonzero(bad):
        add("INVALID_OHLC", ERROR, int(i), f"o={o[i]} h={h[i]} l={l[i]} c={c[i]}")

    # ---- ordering / duplicates ------------------------------------------
    dup = ts.duplicated(keep="first").to_numpy()
    for i in np.flatnonzero(dup):
        add("DUPLICATE_TIMESTAMP", ERROR, int(i), "timestamp already present earlier in the data")
    tv = ts.to_numpy()
    for i in range(1, n):
        if tv[i] < tv[i - 1]:
            add("OUT_OF_ORDER", ERROR, i, f"timestamp earlier than previous bar ({_ts(df, i - 1)})")

    # ---- grid alignment (detects DST-shifted broker data, odd bars) ------
    tf = cfg.timeframe_minutes
    minutes = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
    offsets = minutes % tf
    seconds = ts.dt.second.to_numpy()
    # causal: compare each bar with the dominant offset among bars up to and including it,
    # so appending future data can never change an earlier bar's flag
    seen: Counter = Counter()
    mode_offset = int(offsets[0])
    for i in range(n):
        off = int(offsets[i])
        seen[off] += 1
        if seen[off] > seen[mode_offset]:
            mode_offset = off
        if off != mode_offset or seconds[i] != 0:
            add(
                "TIMEZONE_ALIGNMENT_SHIFT",
                WARNING,
                i,
                f"bar open offset {off}min from H4 grid differs from dominant offset {mode_offset}min so far "
                "(possible DST/timezone inconsistency in source data)",
            )

    # ---- missing bars / weekend gaps ------------------------------------
    step = pd.Timedelta(minutes=tf)
    diffs = ts.diff()
    for i in range(1, n):
        d = diffs.iloc[i]
        if pd.isna(d) or d <= step:
            continue
        start, end = ts.iloc[i - 1], ts.iloc[i]
        hours = d.total_seconds() / 3600.0
        spans_saturday = bool((pd.date_range(start.normalize(), end.normalize(), freq="D").dayofweek == 5).any())
        if spans_saturday and hours <= cfg.max_weekend_gap_hours:
            continue
        missing = int(d / step) - 1
        kind = "EXTENDED_MARKET_CLOSURE" if spans_saturday else "MISSING_BARS"
        add(kind, WARNING, i, f"gap of {hours:.1f}h before this bar (~{missing} bar(s) absent)")

    # ---- abnormal price gaps --------------------------------------------
    rng = pd.Series(h - l)
    ref = rng.rolling(cfg.gap_reference_window, min_periods=10).median().shift(1).to_numpy()
    gap = np.abs(o[1:] - c[:-1])
    for j in np.flatnonzero(~np.isnan(ref[1:]) & (gap > cfg.abnormal_gap_range_mult * ref[1:])):
        i = int(j + 1)
        add("ABNORMAL_PRICE_GAP", WARNING, i, f"open-vs-prev-close gap {gap[j]:.3f} vs median range {ref[i]:.3f}")

    # ---- volume / spread --------------------------------------------------
    present = df.attrs.get("columns_present", {})
    for col, code in (("volume", "MISSING_VOLUME"), ("spread", "MISSING_SPREAD")):
        vals = df[col].to_numpy(dtype=float)
        if present.get(col) is False or np.isnan(vals).all():
            report.issues.append(DataIssue(code, WARNING, None, None, f"{col} not available in source"))
            for i in range(n):
                report.bar_flags[i].append(code)
            continue
        for i in np.flatnonzero(np.isnan(vals)):
            add(code, WARNING, int(i), f"{col} missing for this bar")
    vol = df["volume"].to_numpy(dtype=float)
    for i in np.flatnonzero(vol == 0):
        add("ZERO_VOLUME", WARNING, int(i), "tick volume is zero")
    spr = df["spread"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        neg = spr < 0
    for i in np.flatnonzero(neg):
        add("NEGATIVE_SPREAD", ERROR, int(i), f"spread={spr[i]}")

    sources = df["source"].unique()
    if len(sources) > 1:
        report.issues.append(DataIssue("MIXED_SOURCES", WARNING, None, None, f"multiple sources: {list(sources)}"))

    return _finish(report)


def _finish(report: DataQualityReport) -> DataQualityReport:
    s = report.summary()
    if s["status"] != "OK":
        log = logger.error if report.has_errors else logger.warning
        log("data validation %s: %s", s["status"], s["counts"])
    else:
        logger.info("data validation OK (%d bars)", report.n_bars)
    return report
