"""Chronological splits, final-holdout lock and walk-forward windows (Phase 1H).

* Splits are chronological only (never shuffled): DEVELOPMENT -> VALIDATION ->
  FINAL_HOLDOUT, separated by an embargo so a trade opened in one segment
  cannot be scored with bars of the next.
* ``SplitGuard`` is the ONLY way research code obtains bars.  Development and
  validation requests are truncated at their segment end, so later bars never
  enter a computation.  FINAL_HOLDOUT bars are refused until the
  ``HoldoutLock`` is explicitly unlocked with a strategy snapshot, a
  configuration snapshot, declared hypotheses and validation results.
  Parameter exploration, sensitivity tuning and walk-forward calibration are
  refused on the holdout even after unlock.
* The first holdout access is recorded; every later analysis on it is labelled
  POST_HOC.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ("DEVELOPMENT", "VALIDATION", "FINAL_HOLDOUT")
HOLDOUT_FORBIDDEN_PURPOSES = ("PARAMETER_EXPLORATION", "SENSITIVITY_TUNING", "WALK_FORWARD_CALIBRATION", "ABLATION",
                              "DEVELOPMENT")
UNLOCK_CONFIRMATION = "I_CONFIRM_THE_STRATEGY_IS_FROZEN_AND_THE_HOLDOUT_IS_EVALUATED_ONCE"


class HoldoutLocked(PermissionError):
    pass


@dataclass(frozen=True)
class ChronologicalSplit:
    start: str
    development_end: str
    validation_end: str
    end: str
    embargo_hours: float = 120.0  # no trade may straddle a boundary into the next segment's scoring

    @staticmethod
    def by_fraction(timestamps, development=0.6, validation=0.2, embargo_hours=120.0) -> "ChronologicalSplit":
        t = pd.to_datetime(pd.Series(list(timestamps)), utc=True).sort_values().reset_index(drop=True)
        if development <= 0 or validation <= 0 or development + validation >= 1:
            raise ValueError("fractions must leave a non-empty holdout")
        a, b = t.iloc[0], t.iloc[-1]
        span = b - a
        return ChronologicalSplit(a.isoformat(), (a + span * development).isoformat(),
                                  (a + span * (development + validation)).isoformat(), b.isoformat(), embargo_hours)

    def bounds(self, segment: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        e = pd.Timedelta(hours=self.embargo_hours)
        s, d, v, x = (pd.Timestamp(v_) for v_ in (self.start, self.development_end, self.validation_end, self.end))
        return {"DEVELOPMENT": (s, d), "VALIDATION": (d + e, v), "FINAL_HOLDOUT": (v + e, x + pd.Timedelta(days=7))}[segment]  # last segment: to data end

    def segment_of(self, t) -> str | None:
        t = pd.Timestamp(t)
        for seg in SEGMENTS:
            a, b = self.bounds(seg)
            if a <= t < b:
                return seg
        return None  # inside an embargo

    def to_dict(self) -> dict:
        return asdict(self)


class HoldoutLock:
    """Persistent research-process safeguard (file-backed so it survives restarts)."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            self._write({"state": "LOCKED", "unlock": None, "first_accessed_at": None, "accesses": []})

    def _read(self) -> dict:
        return json.loads(self.path.read_text())

    def _write(self, d: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, sort_keys=True, indent=2))
        tmp.replace(self.path)

    @property
    def state(self) -> str:
        return self._read()["state"]

    @property
    def first_accessed_at(self):
        return self._read()["first_accessed_at"]

    def unlock(self, strategy_snapshot_id: str, configuration_hash: str, hypotheses: list, validation_results_id: str,
               operator: str, confirmation: str, at=None) -> dict:
        missing = [n for n, v in (("strategy_snapshot_id", strategy_snapshot_id), ("configuration_hash", configuration_hash),
                                  ("hypotheses", hypotheses), ("validation_results_id", validation_results_id),
                                  ("operator", operator)) if not v]
        if missing:
            raise HoldoutLocked(f"holdout unlock refused - missing: {missing}")
        if confirmation != UNLOCK_CONFIRMATION:
            raise HoldoutLocked("holdout unlock refused - explicit confirmation phrase required")
        d = self._read()
        if d["state"] == "UNLOCKED":
            return d["unlock"]
        d["state"] = "UNLOCKED"
        d["unlock"] = {"strategy_snapshot_id": strategy_snapshot_id, "configuration_hash": configuration_hash,
                       "hypotheses": list(hypotheses), "validation_results_id": validation_results_id, "operator": operator,
                       "at": pd.Timestamp(at or pd.Timestamp.now(tz="UTC")).isoformat()}
        self._write(d)
        return d["unlock"]

    def record_access(self, purpose: str, snapshot_id: str | None, at=None) -> str:
        d = self._read()
        now = pd.Timestamp(at or pd.Timestamp.now(tz="UTC")).isoformat()
        first = d["first_accessed_at"] is None
        if first:
            d["first_accessed_at"] = now
        label = "HOLDOUT_RESULT" if first else "POST_HOC"
        if not first and d["unlock"] and snapshot_id != d["unlock"]["strategy_snapshot_id"]:
            label = "POST_HOC"
        d["accesses"].append({"at": now, "purpose": purpose, "snapshot_id": snapshot_id, "label": label})
        self._write(d)
        return label


class SplitGuard:
    """The single gateway to research bars.  Holdout bars never leak into earlier segments."""

    def __init__(self, bars: pd.DataFrame, split: ChronologicalSplit, lock: HoldoutLock):
        self._bars = bars
        self.split = split
        self.lock = lock
        self.log: list = []

    def data(self, segment: str, purpose: str, snapshot_id: str | None = None, at=None) -> tuple[pd.DataFrame, dict]:
        """Bars from the dataset start up to the END of ``segment`` (earlier history is legitimate warm-up)."""
        if segment not in SEGMENTS:
            raise ValueError(segment)
        purpose = purpose.upper()
        label = "PRE_DECLARED_TEST" if purpose == "PRE_DECLARED_TEST" else ("EXPLORATORY_ANALYSIS" if segment != "FINAL_HOLDOUT"
                                                                            else None)
        if segment == "FINAL_HOLDOUT":
            if purpose in HOLDOUT_FORBIDDEN_PURPOSES:
                raise HoldoutLocked(f"the final holdout may never be used for {purpose}")
            if self.lock.state != "UNLOCKED":
                raise HoldoutLocked("the final holdout is LOCKED - unlock requires snapshots, hypotheses and validation "
                                    "results")
            label = self.lock.record_access(purpose, snapshot_id, at)
        a, b = self.split.bounds(segment)
        ts = pd.to_datetime(self._bars["timestamp"], utc=True)
        # a bar is usable only once CLOSED before the segment end
        tf = ts.diff().median() if len(ts) > 1 else pd.Timedelta(hours=1)
        out = self._bars.loc[(ts + tf) <= b].reset_index(drop=True)
        out.attrs = dict(self._bars.attrs)
        info = {"segment": segment, "purpose": purpose, "score_from": a.isoformat(), "score_to": b.isoformat(),
                "bars": int(len(out)), "last_bar": pd.Timestamp(out["timestamp"].iloc[-1]).isoformat() if len(out) else None,
                "analysis_label": label}
        self.log.append(info)
        return out, info


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WalkForwardWindow:
    k: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    mode: str

    def to_dict(self) -> dict:
        return asdict(self)


def walk_forward_windows(start, end, test_months: int = 6, train_months: int | None = 12, mode: str = "ANCHORED",
                         min_train_months: int = 6) -> list[WalkForwardWindow]:
    """Anchored (expanding) or rolling windows inside [start, end) - callers pass the NON-holdout range only."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out, k = [], 0
    test_start = s + pd.DateOffset(months=max(train_months or min_train_months, min_train_months))
    while test_start < e:
        test_end = min(test_start + pd.DateOffset(months=test_months), e)
        tr_s = s if mode == "ANCHORED" else max(s, test_start - pd.DateOffset(months=train_months or min_train_months))
        out.append(WalkForwardWindow(k, tr_s.isoformat(), test_start.isoformat(), test_start.isoformat(), test_end.isoformat(),
                                     mode))
        k += 1
        test_start = test_end
    return out


def walk_forward_report(trades: pd.DataFrame, windows: list[WalkForwardWindow], rcol: str = "net_R") -> dict:
    from .metrics import profit_factor, summarise

    rows = []
    t = pd.to_datetime(trades["entry_time"], utc=True) if len(trades) else pd.Series([], dtype="datetime64[ns, UTC]")
    for w in windows:
        m = (t >= pd.Timestamp(w.test_start)) & (t < pd.Timestamp(w.test_end)) if len(trades) else []
        g = trades.loc[m] if len(trades) else trades
        s = summarise(g, rcol) if len(g) else {"trades": 0}
        rows.append({"window": w.k, "test_start": w.test_start, "test_end": w.test_end, "trades": s["trades"],
                     "expectancy_R": s.get("expectancy_R"), "total_R": s.get("total_R", 0.0),
                     "profit_factor": profit_factor(g[rcol]) if len(g) else None, "win_rate": s.get("win_rate"),
                     "max_drawdown_R": s.get("max_drawdown_R"), "sample_level": (s.get("sample") or {}).get("level")})
    tot = [r["total_R"] for r in rows]
    ex = [r["expectancy_R"] for r in rows if r["expectancy_R"] is not None]
    pos_total = sum(x for x in tot if x > 0)
    return {"windows": rows, "n_windows": len(rows), "profitable_windows": sum(x > 0 for x in tot),
            "unprofitable_windows": sum(x < 0 for x in tot), "empty_windows": sum(r["trades"] == 0 for r in rows),
            "expectancy_dispersion_std": float(np.std(ex, ddof=1)) if len(ex) > 1 else None,
            "best_window_share_of_positive_R": (max(tot) / pos_total) if pos_total > 0 else None,
            "fragility_flag": ("CONCENTRATED_IN_ONE_WINDOW" if pos_total > 0 and max(tot) / pos_total > 0.6 and len(rows) > 2
                               else None),
            "method": "the strategy is not re-fitted between windows (no calibration exists); each test window is scored "
                      "only on trades decided inside it, with all earlier bars as causal history"}
