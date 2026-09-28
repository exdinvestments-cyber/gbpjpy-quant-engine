"""H4 Market Intelligence Engine - orchestration.

Pipeline:  canonical bars -> validation -> causal features -> structure/levels
(forward walk) -> regime -> bias -> snapshots / structured log.

The engine is descriptive.  It contains no order placement, no position
sizing and no entry logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .classification.bias import BiasResult, derive_bias
from .classification.regime import RegimeResult, classify_regime
from .config import H4Config
from .data.interfaces import assert_canonical
from .data.model import DataIntegrityError, exclude_unclosed_bars
from .data.validation import DataQualityReport, validate_bars
from .features.candles import candle_features
from .features.chop import chop_features, efficiency_features
from .features.context import (
    extension_features,
    range_location_features,
    round_number_features,
    session_features,
    time_features,
)
from .context.engine import ContextEngine
from .features.levels import Zone
from .features.pipeline import _BENIGN_FLAGS, _ERROR_TYPES, _bar_status, compute_feature_frame  # noqa: F401
from .features.momentum import momentum_features
from .features.structure import StructureResult, compute_structure
from .features.trend import adx_features, trend_features
from .features.volatility import shock_features, volatility_features
from .logging_utils import StructuredEvaluationLogger, export_frame
from .reason_codes import ReasonCode as RC
from .snapshot import H4Snapshot, build_snapshot

logger = logging.getLogger(__name__)


@dataclass
class H4AnalysisResult:
    features: pd.DataFrame
    structure: StructureResult
    zones_by_bar: list[list[Zone]]
    regimes: list[RegimeResult]
    biases: list[BiasResult]
    reason_codes: list[list[str]]
    warnings: list[list[str]]
    report: DataQualityReport
    config: H4Config
    zone_events: list = field(default_factory=list)  # ZoneEvent audit trail (index = bar it became known)
    invalidated_zones: list = field(default_factory=list)
    engine_version: str = __version__
    context: object = None  # Phase 1B ContextResult (context & directional permission)
    _snapshots: dict = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    def index_of(self, timestamp) -> int:
        ts = pd.Timestamp(timestamp)
        if ts.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        ts = ts.tz_convert("UTC")
        hits = np.flatnonzero((self.features["timestamp"] == ts).to_numpy())
        if len(hits) == 0:
            raise KeyError(f"no H4 bar opening at {ts.isoformat()}")
        return int(hits[0])

    def snapshot_at(self, i: int) -> H4Snapshot:
        if i < 0:
            i += len(self.features)
        if i not in self._snapshots:
            row = self.features.iloc[i].to_dict()
            self._snapshots[i] = build_snapshot(
                row, self.zones_by_bar[i], self.regimes[i], self.biases[i], self.reason_codes[i],
                self.warnings[i], self.config.data.symbol, self.engine_version, self.config.config_hash(),
                swing_history=self.structure.history_at(i),
                context_row=self.context.frame.iloc[i].to_dict() if self.context is not None else None,
                context_detail=self.context.details[i] if self.context is not None else None,
            )
        return self._snapshots[i]

    def snapshot(self, timestamp) -> H4Snapshot:
        return self.snapshot_at(self.index_of(timestamp))

    def latest(self) -> H4Snapshot:
        return self.snapshot_at(len(self.features) - 1)

    def snapshots(self):
        for i in range(len(self.features)):
            yield self.snapshot_at(i)

    # ------------------------------------------------------------------
    def write_evaluation_log(self, path: str | Path, include_zones: bool = False) -> int:
        slog = StructuredEvaluationLogger(path, include_zones=include_zones)
        feats = self.features
        records = (slog.record(self.snapshot_at(i), feats.iloc[i].to_dict()) for i in range(len(feats)))
        return slog.write(records)

    def export_features(self, path: str | Path) -> Path:
        return export_frame(self.features, path)

    def export_context(self, path: str | Path) -> Path:
        """Per-bar Phase 1B context/permission table (research storage)."""
        if self.context is None:
            raise ValueError("context engine was not run")
        return export_frame(self.context.frame, path)

    def explain_permission(self, timestamp=None) -> dict:
        """Exact explanation of the directional permission at a bar (default: latest)."""
        if self.context is None:
            raise ValueError("context engine was not run")
        i = len(self.features) - 1 if timestamp is None else self.index_of(timestamp)
        return self.context.explain(i)


class H4MarketIntelligenceEngine:
    """Descriptive GBPJPY H4 market-intelligence engine (Phase 1A)."""

    def __init__(self, config: H4Config | None = None):
        self.config = config or H4Config()
        self.config.validate()

    # ------------------------------------------------------------------
    def run(self, bars: pd.DataFrame, as_of: pd.Timestamp | None = None) -> H4AnalysisResult:
        cfg = self.config
        assert_canonical(bars)
        if as_of is not None:
            bars = exclude_unclosed_bars(bars, as_of, cfg.data.timeframe_minutes)

        report = validate_bars(bars, cfg.data)
        if report.has_errors and cfg.engine.fail_on_data_errors:
            raise DataIntegrityError(f"data integrity errors: {report.counts()}", report=report)
        if report.has_errors:
            logger.error("continuing despite data errors because fail_on_data_errors=False: %s", report.counts())

        df = bars.reset_index(drop=True)
        feats, structure, zones_by_bar, zone_book = compute_feature_frame(df, cfg, report)
        warm = feats["warmup_complete"].to_numpy()

        regimes: list[RegimeResult] = []
        biases: list[BiasResult] = []
        all_codes: list[list[str]] = []
        all_warn: list[list[str]] = []
        records = feats.to_dict("records")
        for i, row in enumerate(records):
            w = bool(warm[i])
            r = classify_regime(row, cfg, warmup_complete=w)
            b = derive_bias(row, r.regime, cfg, warmup_complete=w)
            codes = _merge_codes(r, b, row)
            warnings = []
            if not w:
                warnings.append("warm-up period: outputs not fully formed")
            if row["data_quality_status"] not in ("OK", "INFO"):
                warnings.append("input data flags: " + ",".join(row["data_quality_flags"]))
            regimes.append(r)
            biases.append(b)
            all_codes.append(codes)
            all_warn.append(warnings)

        classified = pd.DataFrame(
            {
                "regime": [r.regime.value for r in regimes],
                "regime_rule": [r.rule for r in regimes],
                "bullish_evidence_score": [b.bullish_evidence_score for b in biases],
                "bearish_evidence_score": [b.bearish_evidence_score for b in biases],
                "h4_bias": [b.bias for b in biases],
                "bias_confidence": [b.bias_confidence for b in biases],
                "reason_codes": all_codes,
            },
            index=feats.index,
        )
        feats = pd.concat([feats, classified], axis=1)

        logger.info(
            "H4 evaluation complete: %d bars, config=%s, regimes=%s",
            len(feats), cfg.config_hash(), feats["regime"].value_counts().to_dict(),
        )
        result = H4AnalysisResult(
            features=feats, structure=structure, zones_by_bar=zones_by_bar, regimes=regimes, biases=biases,
            reason_codes=all_codes, warnings=all_warn, report=report, config=cfg,
            zone_events=zone_book.events, invalidated_zones=zone_book.invalidated,
        )
        # Phase 1B: context & directional permission, layered on top of the Phase 1A/1A.1 result
        result.context = ContextEngine(cfg).run(result, as_of=as_of)
        return result


def _merge_codes(r: RegimeResult, b: BiasResult, row: dict) -> list[str]:
    seen: dict[str, None] = {}
    for c in [*r.evidence, *r.conflicting_evidence, *b.reason_codes]:
        seen.setdefault(c, None)
    if not row.get("warmup_complete", True):
        seen.setdefault(RC.INSUFFICIENT_HISTORY.value, None)
    if any(f not in _BENIGN_FLAGS for f in row.get("data_quality_flags") or []):
        seen.setdefault(RC.DATA_QUALITY_WARNING.value, None)
    return list(seen)
