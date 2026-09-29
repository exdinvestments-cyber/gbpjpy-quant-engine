"""Experiment manifest, research budget and hypothesis register (Phase 1H).

* ``ExperimentManifest`` is append-only JSON-lines with a hash chain: every
  experiment (including failed and abandoned ones) is recorded with its
  hypothesis, changed parameters, reason, dataset, segment, metrics viewed,
  result, seed and analysis label.  There is no delete or edit API, and
  ``verify`` detects any rewrite of history.
* ``ResearchBudget`` caps exploratory parameter experiments per dataset
  segment; once exhausted, a new untouched validation period is required.
* ``HypothesisRegister`` separates discovery from confirmation: each
  hypothesis states its evidence, the data that generated it and the new
  untouched data required to test it.  Nothing here changes a strategy
  parameter (no auto-tuning).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

ANALYSIS_LABELS = ("PRE_DECLARED_TEST", "EXPLORATORY_ANALYSIS", "POST_HOC_ANALYSIS", "HOLDOUT_RESULT")


class ResearchBudgetExhausted(RuntimeError):
    pass


class ManifestTampered(RuntimeError):
    pass


@dataclass
class ExperimentRecord:
    experiment_id: str
    hypothesis: str
    parameters_changed: dict
    reason: str
    dataset_id: str
    segment: str
    analysis_label: str
    strategy_snapshot_id: str
    configuration_hash: str
    seed: int | None = None
    metrics_viewed: list = field(default_factory=list)
    result: dict = field(default_factory=dict)
    status: str = "COMPLETED"  # COMPLETED | FAILED | ABANDONED
    exploratory: bool = True
    at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentManifest:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def _lines(self) -> list[dict]:
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]

    def records(self) -> list[dict]:
        return [x["record"] for x in self._lines()]

    def verify(self) -> bool:
        prev = "GENESIS"
        for x in self._lines():
            h = hashlib.sha256((prev + json.dumps(x["record"], sort_keys=True, default=str)).encode()).hexdigest()
            if x["prev"] != prev or x["hash"] != h:
                raise ManifestTampered(f"manifest chain broken at {x['record'].get('experiment_id')}")
            prev = h
        return True

    def append(self, rec: ExperimentRecord, at=None) -> dict:
        if rec.analysis_label not in ANALYSIS_LABELS:
            raise ValueError(f"analysis_label must be one of {ANALYSIS_LABELS}")
        self.verify()
        lines = self._lines()
        if any(x["record"]["experiment_id"] == rec.experiment_id for x in lines):
            raise ValueError(f"experiment id {rec.experiment_id} already recorded (ids are never reused)")
        rec.at = rec.at or pd.Timestamp(at or pd.Timestamp.now(tz="UTC")).isoformat()
        prev = lines[-1]["hash"] if lines else "GENESIS"
        body = rec.to_dict()
        h = hashlib.sha256((prev + json.dumps(body, sort_keys=True, default=str)).encode()).hexdigest()
        entry = {"prev": prev, "hash": h, "record": body}
        with self.path.open("a") as fh:
            fh.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    def next_id(self, prefix: str = "EXP") -> str:
        return f"{prefix}-{len(self._lines()) + 1:05d}"

    def count(self, segment: str | None = None, exploratory_only: bool = True) -> int:
        return sum(1 for r in self.records() if (segment is None or r["segment"] == segment)
                   and (not exploratory_only or r["exploratory"]))


@dataclass(frozen=True)
class ResearchBudget:
    max_exploratory_experiments: int = 40  # per segment - process discipline, not a statistical guarantee

    def check(self, manifest: ExperimentManifest, segment: str) -> dict:
        used = manifest.count(segment)
        if used >= self.max_exploratory_experiments:
            raise ResearchBudgetExhausted(f"{used} exploratory experiments on {segment}: a new untouched validation period "
                                          "is required before further exploration")
        return {"segment": segment, "used": used, "remaining": self.max_exploratory_experiments - used}


@dataclass
class Hypothesis:
    hypothesis_id: str
    statement: str
    evidence: str
    affected_component: str
    possible_change: str
    generated_from: str  # dataset id + segment the idea came from
    untouched_data_required: str
    status: str = "OPEN"  # OPEN | TESTED_SUPPORTED | TESTED_REJECTED | WITHDRAWN
    at: str = ""


class HypothesisRegister:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def add(self, h: Hypothesis, at=None) -> dict:
        h.at = h.at or pd.Timestamp(at or pd.Timestamp.now(tz="UTC")).isoformat()
        if any(x["hypothesis_id"] == h.hypothesis_id for x in self.all()):
            raise ValueError("hypothesis ids are never reused")
        with self.path.open("a") as fh:
            fh.write(json.dumps(asdict(h), sort_keys=True) + "\n")
        return asdict(h)

    def all(self) -> list[dict]:
        return [json.loads(x) for x in self.path.read_text().splitlines() if x.strip()]
