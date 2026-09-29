"""Repository / strategy / configuration freeze (Phase 1H).

``strategy_snapshot`` identifies exactly what was tested: code commit and
working-tree state, package version, the hash of every strategy source file,
all six configuration objects, the environment and a timestamp.
``configuration_freeze`` lists every parameter with its label:

* UNFITTED_BASELINE - engineering default, never fitted to data (all strategy
  thresholds, weights, bands, lookbacks, expiries, cut-offs, buffers);
* EXTERNAL_CONSTRAINT - fixed by the instrument, platform or regulation
  (symbol, pip size, digits, timeframe, contract size, units);
* EMPIRICALLY_VALIDATED - none at the start of Phase 1H.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import __version__
from ..config import describe_config

LABELS = ("UNFITTED_BASELINE", "EMPIRICALLY_VALIDATED", "EXTERNAL_CONSTRAINT")
EXTERNAL = {("data", "symbol"), ("data", "pip_size"), ("data", "timeframe_minutes"), ("execution", "points_per_pip"),
            ("execution", "spread_unit"), ("execution", "price_basis"), ("news", "currencies"),
            ("sessions", "sessions"), ("instrument", "*")}
PKG = Path(__file__).resolve().parents[1]


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return str(v)


def config_rows(name: str, cfg) -> list[dict]:
    rows = []
    for r in describe_config(cfg):
        sec, key = r["section"], r["name"]
        label = "EXTERNAL_CONSTRAINT" if (sec, key) in EXTERNAL or (sec, "*") in EXTERNAL else "UNFITTED_BASELINE"
        kind = ("weight" if key.startswith("w_") or key.endswith("_weight") else
                "lookback" if "lookback" in key or key.endswith("_bars") or "period" in key or "window" in key else
                "expiry" if "expir" in key or "stale" in key or "valid" in key else
                "buffer" if "buffer" in key else
                "atr_multiplier" if key.endswith("_atr") or "atr_" in key else "threshold")
        rows.append({"config": name, "section": sec, "name": key, "value": _jsonable(r["value"]), "kind": kind,
                     "label": label, "doc": r["doc"]})
    return rows


def configuration_freeze(configs) -> dict:
    parts = {"h4": configs.h4, "h1": configs.h1, "entry": configs.entry, "trade": configs.trade, "risk": configs.risk,
             "execution": configs.execution}
    rows = []
    for n, c in parts.items():
        rows += config_rows(n, c)
    blob = json.dumps([{k: r[k] for k in ("config", "section", "name", "value")} for r in rows], sort_keys=True)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
    return {"configuration_hash": hashlib.sha256(blob.encode()).hexdigest(), "parameters": rows, "label_counts": counts,
            "n_parameters": len(rows), "n_tunable": counts["UNFITTED_BASELINE"]}


def _git(args) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=PKG, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 - not a git checkout
        return None


def source_hash() -> dict:
    files = sorted(p for p in PKG.rglob("*.py") if "validation" not in p.parts and "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(PKG)).encode())
        h.update(p.read_bytes())
    return {"strategy_source_sha256": h.hexdigest(), "files": len(files)}


def strategy_snapshot(configs, execution_assumptions: dict | None = None, now=None) -> dict:
    frz = configuration_freeze(configs)
    src = source_hash()
    status = _git(["status", "--porcelain"])
    snap = {"package_version": __version__, "git_commit": _git(["rev-parse", "HEAD"]),
            "git_dirty": bool(status) if status is not None else None, **src,
            "configuration_hash": frz["configuration_hash"],
            "risk_policy_hash": configs.risk.policy_hash() if hasattr(configs.risk, "policy_hash") else None,
            "execution_policy_hash": configs.execution.policy_hash(),
            "execution_assumptions": execution_assumptions or {},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                            "pandas": pd.__version__}}
    ident = hashlib.sha256(json.dumps({k: snap[k] for k in ("strategy_source_sha256", "configuration_hash",
                                                            "execution_assumptions")}, sort_keys=True,
                                      default=str).encode()).hexdigest()
    snap["snapshot_id"] = f"SNAP-{ident[:16]}"
    snap["created_at"] = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).isoformat()
    return snap

