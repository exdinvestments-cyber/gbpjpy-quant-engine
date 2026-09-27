"""Structured research logging and export.

* ``StructuredEvaluationLogger`` writes one JSON object per H4 evaluation
  (JSON Lines), suitable for loading into pandas/DuckDB for research.
* ``export_frame`` writes the flat feature table as CSV, JSON Lines or Parquet
  depending on the file extension.

Console output is left to the standard ``logging`` module; nothing here prints.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

from .snapshot import H4Snapshot, to_jsonable

logger = logging.getLogger(__name__)


class StructuredEvaluationLogger:
    def __init__(self, path: str | Path, include_zones: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.include_zones = include_zones

    def record(self, snap: H4Snapshot, features: dict) -> dict:
        rec = {
            "timestamp": snap.timestamp,
            "available_at": snap.available_at,
            "symbol": snap.symbol,
            "input_data_status": snap.data_quality_status,
            "input_data_flags": snap.data_quality_flags,
            "warmup_complete": snap.warmup_complete,
            "regime": snap.regime,
            "regime_evidence": snap.regime_evidence,
            "regime_conflicting_evidence": snap.regime_conflicting_evidence,
            "bias": snap.h4_bias,
            "confidence": snap.bias_confidence,
            "bullish_evidence_score": snap.bullish_evidence_score,
            "bearish_evidence_score": snap.bearish_evidence_score,
            "reason_codes": snap.reason_codes,
            "warnings": snap.warnings,
            "features": to_jsonable(features),
            "engine_version": snap.engine_version,
            "config_hash": snap.config_hash,
        }
        if self.include_zones:
            rec["zones"] = snap.zones
        return rec

    def write(self, records: Iterable[dict], append: bool = False) -> int:
        n = 0
        with self.path.open("a" if append else "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, separators=(",", ":")) + "\n")
                n += 1
        logger.info("wrote %d evaluation records to %s", n, self.path)
        return n


def read_evaluation_log(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def export_frame(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: json.dumps(to_jsonable(v)) if isinstance(v, (list, dict)) else v)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        out.to_parquet(path, index=False)
    elif suffix in (".jsonl", ".json"):
        out.to_json(path, orient="records", lines=True, date_format="iso")
    else:
        out.to_csv(path, index=False)
    logger.info("exported %d rows to %s", len(out), path)
    return path
