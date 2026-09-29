"""Dataset provenance and provider-neutral historical data sources (Phase 1H).

Every dataset used for research must carry a complete ``DatasetProvenance``:
provider, symbol, timeframe, coverage, timezone, price type (BID / ASK / MID /
UNKNOWN), volume type, spread availability, missing-data notes, import time and
a content hash.  A backtest refuses a dataset with unknown provenance.

Providers only READ files.  They never modify a source file, never guess a
timezone and never fill missing values.  Nothing here downloads data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable

import pandas as pd

from ..data.model import to_canonical

PRICE_TYPES = ("BID", "ASK", "MID", "UNKNOWN")
VOLUME_TYPES = ("TICK_VOLUME", "REAL_VOLUME", "NONE", "UNKNOWN")
SPREAD_AVAILABILITY = ("PER_BAR", "SAMPLED", "NONE", "UNKNOWN")
SPREAD_UNITS = ("PIPS", "POINTS", "PRICE", "UNKNOWN")
TIMEFRAMES = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "TICK": 0}
SYNTHETIC_MARKERS = ("synthetic", "generated", "simulated", "random")


class ProvenanceError(ValueError):
    """Raised when a dataset's provenance is missing, incomplete or inconsistent."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Content hash of a canonical frame (column order and values; independent of in-memory layout)."""
    cols = [c for c in df.columns]
    payload = df[cols].to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S%z", float_format="%.10g")
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class DatasetProvenance:
    provider: str  # e.g. "broker_export", "terminal_export", "institutional_vendor", "csv"
    symbol: str
    timeframe: str  # M1 / M5 / M15 / H1 / H4 / TICK
    source_timezone: str  # IANA timezone of the SOURCE timestamps (e.g. "UTC", "Europe/Athens", "Etc/GMT-2")
    price_type: str  # BID / ASK / MID / UNKNOWN
    volume_type: str = "UNKNOWN"
    spread_availability: str = "UNKNOWN"
    spread_unit: str = "UNKNOWN"
    spread_resolution: str = ""  # e.g. "per-bar average", "sampled every 10 minutes"
    start: str | None = None  # first bar open (UTC ISO), filled at import
    end: str | None = None  # last bar open (UTC ISO), filled at import
    rows: int | None = None
    missing_data_notes: str = ""
    known_outages: tuple = ()  # ((start_iso, end_iso, note), ...) declared by the provider
    retrieved_at: str | None = None  # when the provider file was obtained (declared by the importer)
    imported_at: str | None = None  # when it entered the RAW layer
    source_file: str | None = None
    source_sha256: str | None = None
    server_offset_note: str = ""  # broker/server clock convention (documentation only)
    dst_policy: str = "DECLARED_BY_SOURCE_TIMEZONE"  # never inferred
    is_synthetic: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["known_outages"] = [list(x) for x in self.known_outages]
        return d

    @staticmethod
    def from_dict(d: dict) -> "DatasetProvenance":
        kw = dict(d)
        kw["known_outages"] = tuple(tuple(x) for x in kw.get("known_outages") or ())
        return DatasetProvenance(**kw)

    @property
    def dataset_id(self) -> str:
        """Stable id: provider|symbol|timeframe|content hash (identical bytes -> identical id)."""
        if not self.source_sha256:
            raise ProvenanceError("dataset has no content hash - import it through the DataStore first")
        return f"{self.symbol}_{self.timeframe}_{self.provider}_{self.source_sha256[:16]}"

    def validate(self) -> None:
        problems = []
        for name in ("provider", "symbol", "timeframe", "source_timezone", "price_type"):
            if not getattr(self, name):
                problems.append(f"{name} missing")
        if self.price_type not in PRICE_TYPES:
            problems.append(f"price_type must be one of {PRICE_TYPES}")
        if self.volume_type not in VOLUME_TYPES:
            problems.append(f"volume_type must be one of {VOLUME_TYPES}")
        if self.spread_availability not in SPREAD_AVAILABILITY:
            problems.append(f"spread_availability must be one of {SPREAD_AVAILABILITY}")
        if self.spread_unit not in SPREAD_UNITS:
            problems.append(f"spread_unit must be one of {SPREAD_UNITS}")
        if self.timeframe not in TIMEFRAMES:
            problems.append(f"timeframe must be one of {sorted(TIMEFRAMES)}")
        if self.spread_availability in ("PER_BAR", "SAMPLED") and self.spread_unit == "UNKNOWN":
            problems.append("spread is available but its unit is UNKNOWN - declare PIPS / POINTS / PRICE")
        if any(m in (self.provider or "").lower() for m in SYNTHETIC_MARKERS) and not self.is_synthetic:
            problems.append("provider name indicates synthetic data but is_synthetic is False")
        if problems:
            raise ProvenanceError("; ".join(problems))

    def with_import(self, **kw) -> "DatasetProvenance":
        return replace(self, **kw)


# ---------------------------------------------------------------------------
# Provider-neutral sources
# ---------------------------------------------------------------------------
@runtime_checkable
class HistoricalDataProvider(Protocol):
    """Reads one provider file into (raw frame, provenance).  Must never modify the file."""

    provenance: DatasetProvenance

    def read_raw(self, path) -> pd.DataFrame:
        ...

    def to_canonical_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass
class ImportSpec:
    """How to interpret a provider file (declared by the person importing it - never guessed)."""

    provenance: DatasetProvenance
    fmt: str = "CSV"  # CSV | PARQUET | TERMINAL_TAB_EXPORT
    column_map: Mapping[str, str] = field(default_factory=dict)
    csv_kwargs: Mapping = field(default_factory=dict)
    spread_to_pips: float | None = None  # multiplier converting the source spread unit into pips


def _spread_multiplier(prov: DatasetProvenance, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    return {"PIPS": 1.0, "POINTS": 0.1, "PRICE": 100.0}.get(prov.spread_unit)  # GBPJPY: 1 pip = 10 points = 0.01


class FileProvider:
    """CSV / Parquet / platform tab-separated terminal exports (``<DATE>\\t<TIME>\\t<OPEN>...``)."""

    def __init__(self, spec: ImportSpec):
        spec.provenance.validate()
        if spec.fmt not in ("CSV", "PARQUET", "TERMINAL_TAB_EXPORT"):
            raise ProvenanceError(f"unsupported format {spec.fmt}")
        self.spec = spec
        self.provenance = spec.provenance

    def read_raw(self, path) -> pd.DataFrame:
        path = Path(path)
        if self.spec.fmt == "PARQUET":
            return pd.read_parquet(path)
        if self.spec.fmt == "TERMINAL_TAB_EXPORT":
            return pd.read_csv(path, sep="\t", **dict(self.spec.csv_kwargs))
        return pd.read_csv(path, **dict(self.spec.csv_kwargs))

    def to_canonical_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
        if self.spec.fmt == "TERMINAL_TAB_EXPORT" and "date" in df.columns and "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="mixed")
            df = df.drop(columns=["date", "time"])
            if "tickvol" in df.columns:
                df = df.rename(columns={"tickvol": "volume"})
                if "vol" in df.columns:
                    df = df.drop(columns=["vol"])
        elif self.spec.fmt == "TERMINAL_TAB_EXPORT" and "date" in df.columns:
            df = df.rename(columns={"date": "timestamp"})
        cmap = {str(k).strip().strip("<>").lower(): v for k, v in dict(self.spec.column_map).items()}
        # the declared source timezone is applied ONLY to naive timestamps; aware ones are converted to UTC
        out = to_canonical(df, source=f"{self.provenance.provider}:{self.provenance.symbol}:{self.provenance.timeframe}",
                           assume_timezone=self.provenance.source_timezone, column_map=cmap or None)
        mult = _spread_multiplier(self.provenance, self.spec.spread_to_pips)
        if self.provenance.spread_availability in ("NONE", "UNKNOWN"):
            out["spread"] = float("nan")
            out.attrs["columns_present"] = dict(out.attrs.get("columns_present", {}), spread=False)
        elif mult is None:
            raise ProvenanceError("spread present but unit unknown - cannot convert to pips")
        else:
            out["spread"] = out["spread"] * mult
        out.attrs["price_type"] = self.provenance.price_type
        return out


def describe_accepted_formats() -> dict:
    """Machine-readable import specification (also rendered in docs/PHASE_1H_REAL_DATA_VALIDATION.md)."""
    return {
        "required_timeframes": ["H1", "H4 (optional: derived from H1 if absent; a provider H4 file is compared)"],
        "optional_timeframes": ["M15", "M5", "M1", "TICK (execution sequencing only)"],
        "formats": {
            "CSV": "comma-separated with a header row",
            "PARQUET": "columnar file with the same columns",
            "TERMINAL_TAB_EXPORT": "tab-separated platform history export with <DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> "
                                   "<TICKVOL> <VOL> <SPREAD> header",
        },
        "required_columns": {"timestamp": "bar OPEN time (ISO-8601; naive timestamps require a declared source_timezone)",
                             "open": "float", "high": "float", "low": "float", "close": "float"},
        "recommended_columns": {"volume": "tick volume (declare volume_type)",
                                "spread": "per-bar spread (declare spread_unit PIPS / POINTS / PRICE)"},
        "required_provenance": ["provider", "symbol", "timeframe", "source_timezone", "price_type (BID/ASK/MID/UNKNOWN)",
                                "volume_type", "spread_availability", "spread_unit", "retrieved_at"],
        "tick_columns": {"timestamp": "quote time", "bid": "float", "ask": "float"},
        "recommended_coverage": "as many years as available, spanning trending, ranging, high- and low-volatility "
                                "environments (actual coverage and regime diversity are reported, never assumed)",
    }


def provenance_json(prov: DatasetProvenance) -> str:
    return json.dumps(prov.to_dict(), sort_keys=True, indent=2)
