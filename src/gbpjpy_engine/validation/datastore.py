"""RAW -> NORMALISED -> DERIVED research data layers (Phase 1H).

* RAW: the provider file, copied byte-for-byte, hashed and made read-only, with
  its provenance.  Nothing ever writes to a RAW file again; ``verify_raw``
  re-hashes it before every use.
* NORMALISED: canonical UTC bars produced from RAW by a declared, logged
  transformation (column mapping, timezone conversion, spread unit).  No
  sorting, de-duplication or repair: defects are reported, not fixed.
* DERIVED: deterministic products (e.g. H4 from H1 under a versioned bar
  definition), each tagged with the definition and the parent hash.

Synthetic data is never written into the store as if it were real: a
provenance flagged ``is_synthetic`` is kept under a separate namespace and the
real-data run refuses it.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pandas as pd

from .provenance import DatasetProvenance, FileProvider, ImportSpec, ProvenanceError, sha256_file, sha256_frame


class RawDataModified(RuntimeError):
    """A RAW file no longer matches the hash recorded at import."""


class DataStore:
    def __init__(self, root):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.norm = self.root / "normalised"
        self.derived = self.root / "derived"
        for p in (self.raw, self.norm, self.derived):
            p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ RAW
    def import_raw(self, source_path, spec: ImportSpec, imported_at=None) -> DatasetProvenance:
        """Copy a provider file into RAW unchanged (the source file itself is never touched)."""
        src = Path(source_path)
        if not src.is_file():
            raise FileNotFoundError(src)
        spec.provenance.validate()
        digest = sha256_file(src)
        prov = spec.provenance.with_import(source_file=src.name, source_sha256=digest,
                                           imported_at=pd.Timestamp(imported_at or pd.Timestamp.now(tz="UTC")).isoformat())
        ns = "synthetic" if prov.is_synthetic else "real"
        d = self.raw / ns / prov.dataset_id
        d.mkdir(parents=True, exist_ok=True)
        dst = d / src.name
        if dst.exists():
            if sha256_file(dst) != digest:
                raise RawDataModified(f"{dst} exists with different content - RAW files are immutable")
        else:
            shutil.copyfile(src, dst)
            os.chmod(dst, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        spec_doc = {"fmt": spec.fmt, "column_map": dict(spec.column_map), "csv_kwargs": dict(spec.csv_kwargs),
                    "spread_to_pips": spec.spread_to_pips}
        meta = d / "provenance.json"
        if not meta.exists():
            meta.write_text(json.dumps({"provenance": prov.to_dict(), "import_spec": spec_doc}, sort_keys=True, indent=2))
        return DatasetProvenance.from_dict(json.loads(meta.read_text())["provenance"])

    def _raw_dir(self, dataset_id: str) -> Path:
        for ns in ("real", "synthetic"):
            d = self.raw / ns / dataset_id
            if d.is_dir():
                return d
        raise KeyError(f"unknown dataset {dataset_id}")

    def provenance(self, dataset_id: str) -> DatasetProvenance:
        return DatasetProvenance.from_dict(json.loads((self._raw_dir(dataset_id) / "provenance.json").read_text())["provenance"])

    def verify_raw(self, dataset_id: str) -> str:
        prov = self.provenance(dataset_id)
        f = self._raw_dir(dataset_id) / prov.source_file
        got = sha256_file(f)
        if got != prov.source_sha256:
            raise RawDataModified(f"RAW file {f} changed since import ({got[:12]} != {prov.source_sha256[:12]})")
        return got

    def list_datasets(self, real_only: bool = True) -> list[DatasetProvenance]:
        out = []
        for ns in (("real",) if real_only else ("real", "synthetic")):
            base = self.raw / ns
            if base.is_dir():
                for d in sorted(base.iterdir()):
                    if (d / "provenance.json").exists():
                        out.append(DatasetProvenance.from_dict(json.loads((d / "provenance.json").read_text())["provenance"]))
        return out

    # ------------------------------------------------------------------ NORMALISED
    def normalise(self, dataset_id: str) -> tuple[pd.DataFrame, dict]:
        """RAW -> canonical UTC frame; the transformation log is stored next to it."""
        self.verify_raw(dataset_id)
        meta = json.loads((self._raw_dir(dataset_id) / "provenance.json").read_text())
        prov = DatasetProvenance.from_dict(meta["provenance"])
        s = meta["import_spec"]
        spec = ImportSpec(prov, fmt=s["fmt"], column_map=s["column_map"], csv_kwargs=s["csv_kwargs"],
                          spread_to_pips=s["spread_to_pips"])
        fp = FileProvider(spec)
        raw = fp.read_raw(self._raw_dir(dataset_id) / prov.source_file)
        df = fp.to_canonical_frame(raw)
        log = {"dataset_id": dataset_id, "parent_sha256": prov.source_sha256, "rows_in": int(len(raw)), "rows_out": int(len(df)),
               "steps": [f"column normalisation (lower-case, '<>' stripped, map={dict(spec.column_map)})",
                         f"naive timestamps localised as declared source timezone {prov.source_timezone!r} "
                         "(ambiguous/non-existent DST times raise); converted to UTC",
                         f"spread: availability={prov.spread_availability} unit={prov.spread_unit} -> pips",
                         "no sorting, de-duplication, filling or repair"],
               "price_type": prov.price_type, "normalised_sha256": sha256_frame(df)}
        prov2 = prov.with_import(start=df["timestamp"].min().isoformat() if len(df) else None,
                                 end=df["timestamp"].max().isoformat() if len(df) else None, rows=int(len(df)))
        out = self.norm / f"{dataset_id}.parquet"
        _plain(df).to_parquet(out, index=False)
        (self.norm / f"{dataset_id}.transform.json").write_text(json.dumps({"log": log, "provenance": prov2.to_dict()},
                                                                         sort_keys=True, indent=2))
        df.attrs["provenance"] = prov2
        df.attrs["transform_log"] = log
        return df, log

    def load_normalised(self, dataset_id: str) -> pd.DataFrame:
        p = self.norm / f"{dataset_id}.parquet"
        if not p.exists():
            return self.normalise(dataset_id)[0]
        doc = json.loads((self.norm / f"{dataset_id}.transform.json").read_text())
        if doc["log"]["parent_sha256"] != self.verify_raw(dataset_id):
            raise RawDataModified("normalised layer is stale relative to RAW")
        df = pd.read_parquet(p)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).astype("datetime64[ns, UTC]")
        if sha256_frame(df) != doc["log"]["normalised_sha256"]:
            raise RawDataModified("normalised layer content changed since it was produced")
        prov = DatasetProvenance.from_dict(doc["provenance"])
        df.attrs["provenance"] = prov
        df.attrs["transform_log"] = doc["log"]
        df.attrs["columns_present"] = {"volume": bool(df["volume"].notna().any()), "spread": bool(df["spread"].notna().any())}
        return df

    # ------------------------------------------------------------------ DERIVED
    def save_derived(self, name: str, df: pd.DataFrame, meta: dict) -> Path:
        p = self.derived / f"{name}.parquet"
        _plain(df).to_parquet(p, index=False)
        (self.derived / f"{name}.meta.json").write_text(json.dumps(dict(meta, sha256=sha256_frame(df)), sort_keys=True,
                                                                     indent=2, default=str))
        return p


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs = {}
    return out


def require_known_provenance(df: pd.DataFrame) -> DatasetProvenance:
    prov = df.attrs.get("provenance")
    if not isinstance(prov, DatasetProvenance):
        raise ProvenanceError("dataset has unknown provenance - backtests refuse it")
    prov.validate()
    if not prov.source_sha256:
        raise ProvenanceError("dataset has no content hash")
    return prov
