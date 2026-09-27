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
from .data.model import CANONICAL_COLUMNS, DataIntegrityError, exclude_unclosed_bars
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
from .features.levels import Zone, compute_levels
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
    engine_version: str = __version__
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


class H4MarketIntelligenceEngine:
    """Descriptive GBPJPY H4 market-intelligence engine (Phase 1A)."""

    def __init__(self, config: H4Config | None = None):
        self.config = config or H4Config()
        self.config.validate()

    # ------------------------------------------------------------------
    def run(self, bars: pd.DataFrame, as_of: pd.Timestamp | None = None) -> H4AnalysisResult:
        cfg = self.config
        missing = [c for c in CANONICAL_COLUMNS if c not in bars.columns]
        if missing:
            raise DataIntegrityError(f"bars are not canonical; missing {missing} (use data.to_canonical)")
        if as_of is not None:
            bars = exclude_unclosed_bars(bars, as_of, cfg.data.timeframe_minutes)

        report = validate_bars(bars, cfg.data)
        if report.has_errors and cfg.engine.fail_on_data_errors:
            raise DataIntegrityError(f"data integrity errors: {report.counts()}", report=report)
        if report.has_errors:
            logger.error("continuing despite data errors because fail_on_data_errors=False: %s", report.counts())

        df = bars.reset_index(drop=True)
        tf = pd.Timedelta(minutes=cfg.data.timeframe_minutes)

        vol = volatility_features(df, cfg.volatility)
        atr_s = vol["atr"]
        candles = candle_features(df, vol["atr_prev"], cfg.candle)
        shock = shock_features(df, vol, candles, cfg.shock)
        trend = trend_features(df["close"], atr_s, cfg.trend)
        adx_df = adx_features(df, cfg.adx)
        mom = momentum_features(df, atr_s, cfg.momentum)
        eff = efficiency_features(df["close"], cfg.efficiency)

        W = cfg.structure.quality_window_bars
        overlap_mean = candles["overlap_prev"].rolling(W, min_periods=5).mean()
        wick_mean = candles["wick_ratio"].rolling(W, min_periods=5).mean()
        structure = compute_structure(df, atr_s, overlap_mean, wick_mean, cfg.swing, cfg.structure, tf)
        chop = chop_features(df, atr_s, trend, adx_df, candles, eff, structure.frame, cfg.chop)
        levels, zones_by_bar = compute_levels(df, atr_s, structure, cfg.levels)
        rn = round_number_features(df["close"], atr_s, cfg.round_numbers, cfg.data)
        rloc = range_location_features(df, cfg.range_location)
        ext = extension_features(df, trend, atr_s, cfg.extension)
        sess = session_features(df["timestamp"], cfg.sessions, tf)
        tfeat = time_features(df["timestamp"])

        base = df.copy()
        base.insert(1, "available_at", df["timestamp"] + tf)
        base.insert(0, "symbol", cfg.data.symbol)
        feats = pd.concat(
            [base, candles, vol, shock, trend, adx_df, mom, eff, structure.frame, chop, levels, rn, rloc, ext, sess, tfeat],
            axis=1,
        )
        feats = feats.loc[:, ~feats.columns.duplicated()].copy()

        warm = np.arange(len(df)) >= cfg.engine.warmup_bars
        feats["warmup_complete"] = warm
        flags = report.bar_flags if report.bar_flags else [[] for _ in range(len(df))]
        feats["data_quality_flags"] = [list(f) for f in flags]
        feats["data_quality_status"] = [_bar_status(fl) for fl in flags]

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
        return H4AnalysisResult(
            features=feats, structure=structure, zones_by_bar=zones_by_bar, regimes=regimes, biases=biases,
            reason_codes=all_codes, warnings=all_warn, report=report, config=cfg,
        )


_ERROR_TYPES = {
    "MISSING_PRICE", "NONPOSITIVE_PRICE", "INVALID_OHLC", "DUPLICATE_TIMESTAMP", "OUT_OF_ORDER", "NEGATIVE_SPREAD",
}
# Warnings that only say volume/spread are unavailable do not degrade price-based features.
_BENIGN_FLAGS = {"MISSING_VOLUME", "MISSING_SPREAD", "ZERO_VOLUME"}


def _bar_status(flags: list[str]) -> str:
    """ERROR > WARNING (price/time integrity concern) > INFO (only volume/spread unavailable) > OK."""
    if any(f in _ERROR_TYPES for f in flags):
        return "ERROR"
    if any(f not in _BENIGN_FLAGS for f in flags):
        return "WARNING"
    return "INFO" if flags else "OK"


def _merge_codes(r: RegimeResult, b: BiasResult, row: dict) -> list[str]:
    seen: dict[str, None] = {}
    for c in [*r.evidence, *r.conflicting_evidence, *b.reason_codes]:
        seen.setdefault(c, None)
    if not row.get("warmup_complete", True):
        seen.setdefault(RC.INSUFFICIENT_HISTORY.value, None)
    if any(f not in _BENIGN_FLAGS for f in row.get("data_quality_flags") or []):
        seen.setdefault(RC.DATA_QUALITY_WARNING.value, None)
    return list(seen)
