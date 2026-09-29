"""Versioned, deterministic H1 -> H4 resampling and H1/H4 consistency checks (Phase 1H).

The H4 bar definition materially changes strategy behaviour, so it is an
explicit, versioned object.  Two families are supported:

* a fixed UTC grid (optionally offset, e.g. 21:00-anchored buckets), and
* a provider/server-clock grid (buckets aligned in ``grid_timezone``, e.g. a
  broker's server clock), whose UTC boundaries move with that clock's DST.

Only COMPLETE buckets are emitted (``need`` constituent H1 bars).  On a DST
transition day a server-clock bucket can contain 3 or 5 H1 bars; it is dropped
and reported, never stretched or filled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..data.model import CANONICAL_COLUMNS
from ..data.resample import resample_complete


@dataclass(frozen=True)
class H4BarDefinition:
    name: str = "H4_UTC_GRID_00"
    version: int = 1
    grid_timezone: str = "UTC"  # timezone whose wall clock defines bucket boundaries
    offset_hours: int = 0  # bucket start offset within that clock (0 -> 00/04/08/12/16/20)
    week_boundary: str = "provider week (Sunday open to Friday close); buckets never span a closure"
    dst_policy: str = "UTC grid: no DST adjustment; server-clock grids follow the declared timezone's DST"

    @property
    def definition_id(self) -> str:
        h = hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]
        return f"{self.name}_v{self.version}_{h}"


UTC_GRID = H4BarDefinition()


def resample_h1_to_h4(h1: pd.DataFrame, definition: H4BarDefinition = UTC_GRID) -> tuple[pd.DataFrame, dict]:
    """Deterministic H1 -> H4 under ``definition``.  Returns (H4 bars, report)."""
    if definition.grid_timezone == "UTC":
        out, dropped = resample_complete(h1, 60, 240, pd.Timedelta(hours=definition.offset_hours))
    else:
        local = h1["timestamp"].dt.tz_convert(definition.grid_timezone).dt.tz_localize(None)
        key = (local - pd.Timedelta(hours=definition.offset_hours)).dt.floor("240min")
        g = h1.assign(_k=key.to_numpy()).groupby("_k", sort=True)
        agg = g.agg(timestamp=("timestamp", "min"), open=("open", "first"), high=("high", "max"), low=("low", "min"),
                    close=("close", "last"), volume=("volume", lambda v: v.sum(min_count=1)), spread=("spread", "mean"),
                    source=("source", "first"), n=("close", "size"),
                    span=("timestamp", lambda t: (t.max() - t.min()) / pd.Timedelta(hours=1)))
        ok = (agg["n"] == 4) & (agg["span"] == 3)
        dropped = list(agg.loc[~ok, "timestamp"])
        out = agg.loc[ok].drop(columns=["n", "span"]).reset_index(drop=True)
        out["timestamp"] = out["timestamp"].astype("datetime64[ns, UTC]")
        out["source"] = out["source"].astype(str) + ":agg240"
        out = out[CANONICAL_COLUMNS]
        out.attrs["columns_present"] = h1.attrs.get("columns_present", {"volume": True, "spread": True})
    report = {"definition": asdict(definition), "definition_id": definition.definition_id, "h1_bars": int(len(h1)),
              "h4_bars": int(len(out)), "dropped_incomplete_buckets": len(dropped),
              "dropped_examples": [pd.Timestamp(x).isoformat() for x in dropped[:10]]}
    out.attrs["h4_definition_id"] = definition.definition_id
    return out, report


def compare_h4(provider_h4: pd.DataFrame, derived_h4: pd.DataFrame, tolerance_pips: float = 0.5,
               pip_size: float = 0.01) -> dict:
    """Compare a provider's own H4 file with H4 derived from its H1 (never silently mixed)."""
    a = provider_h4.set_index("timestamp")[["open", "high", "low", "close"]]
    b = derived_h4.set_index("timestamp")[["open", "high", "low", "close"]]
    only_p = a.index.difference(b.index)
    only_d = b.index.difference(a.index)
    both = a.index.intersection(b.index)
    diff = (a.loc[both] - b.loc[both]).abs() / pip_size
    bad = diff.max(axis=1) > tolerance_pips
    grid_p = sorted({int(t.hour % 4) for t in a.index})
    grid_d = sorted({int(t.hour % 4) for t in b.index})
    return {"provider_bars": int(len(a)), "derived_bars": int(len(b)), "common_bars": int(len(both)),
            "only_in_provider": int(len(only_p)), "only_in_derived": int(len(only_d)),
            "ohlc_mismatches": int(bad.sum()), "max_abs_diff_pips": float(diff.to_numpy().max()) if len(both) else None,
            "mismatch_examples": [t.isoformat() for t in diff.index[bad.to_numpy()][:10]],
            "provider_grid_hour_mod4": grid_p, "derived_grid_hour_mod4": grid_d,
            "grids_consistent": grid_p == grid_d,
            "verdict": "CONSISTENT" if not bad.any() and not len(only_p) and not len(only_d) and grid_p == grid_d
            else "DISCREPANCIES_REPORTED"}


def verify_aggregation(h1: pd.DataFrame, h4: pd.DataFrame) -> dict:
    """Exact check that every H4 bar equals the aggregate of its four H1 bars (UTC grid)."""
    idx = h1.set_index("timestamp")
    bad = []
    for _, r in h4.iterrows():
        sub = idx.loc[r["timestamp"]: r["timestamp"] + pd.Timedelta(hours=3)]
        exp = (sub["open"].iloc[0], sub["high"].max(), sub["low"].min(), sub["close"].iloc[-1]) if len(sub) == 4 else None
        got = (r["open"], r["high"], r["low"], r["close"])
        if exp is None or not np.allclose(exp, got, rtol=0, atol=1e-9):
            bad.append(r["timestamp"].isoformat())
    return {"h4_bars": int(len(h4)), "exact": not bad, "mismatches": bad[:10], "n_mismatches": len(bad)}
