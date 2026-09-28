"""Point-in-time H4 -> H1 alignment.

Rule: an H1 bar that CLOSES at time ``t`` may only see the most recent H4 bar
whose CLOSE (``available_at``) is ``<= t``.  The join is a backward as-of join
on close times, so an H4 candle that is still forming at ``t`` (its close is
later than ``t``) is unreachable by construction.  At the instant an H4 candle
closes, the H1 candle closing at the same instant may see it - both are then
complete.

Because the join is on close times it is independent of the H4 grid anchor
(e.g. brokers whose H4 candles open at 21:00 UTC after a DST shift) and of
missing bars.  Safeguards on top of the join:

* a post-join assertion that no attached H4 close is later than the H1 close
  (``AlignmentError`` -> the H1 engine blocks everything, never guesses)
* ``h4_context_status``: NONE (no completed H4 yet), STALE (older than
  ``max_context_age_hours`` - e.g. missing H4 bars), OK
* the H4 bar index is exposed so H4 objects (zones, context details) are
  looked up by that index only - never by the H1 timestamp
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import H1AlignmentConfig

H4_FIELDS_FROM_FEATURES = ("regime", "atr", "atr_pct", "last_swing_high", "last_swing_low", "structure_state",
                           "volatility_regime", "warmup_complete", "data_quality_status")
H4_FIELDS_FROM_CONTEXT = ("directional_permission", "permission_confidence", "long_context_score", "short_context_score",
                          "context_conflict_score", "context_quality_score", "primary_structure",
                          "primary_structure_confidence", "intermediate_structure", "immediate_structure",
                          "structural_direction", "long_room_score", "short_room_score", "structural_range_percentile",
                          "trend_maturity", "context_state", "hard_blockers")


class AlignmentError(RuntimeError):
    """Raised if an H1 bar would be joined to H4 information not yet available."""


def align_h4_to_h1(h1_close: pd.Series, h4_result, cfg: H1AlignmentConfig) -> pd.DataFrame:
    h4f = h4_result.features
    h4c = h4_result.context.frame if h4_result.context is not None else None
    right = pd.DataFrame({"h4_available_at": h4f["available_at"].to_numpy(), "h4_index": np.arange(len(h4f))})
    left = pd.DataFrame({"h1_available_at": h1_close.to_numpy(), "_row": np.arange(len(h1_close))})
    if len(right):
        joined = pd.merge_asof(left, right, left_on="h1_available_at", right_on="h4_available_at",
                               direction="backward", allow_exact_matches=True)
    else:
        joined = left.assign(h4_available_at=pd.NaT, h4_index=np.nan)
    joined = joined.sort_values("_row").reset_index(drop=True)
    has = joined["h4_index"].notna()
    if (joined.loc[has, "h4_available_at"] > joined.loc[has, "h1_available_at"]).any():
        raise AlignmentError("H4 information later than the H1 close would be attached")
    idx = joined["h4_index"].fillna(-1).astype(int).to_numpy()
    age = (joined["h1_available_at"] - joined["h4_available_at"]).dt.total_seconds() / 3600.0
    out = pd.DataFrame(index=h1_close.index)
    out["h4_index"] = idx
    out["h4_timestamp"] = [h4f["timestamp"].iloc[i] if i >= 0 else pd.NaT for i in idx]
    out["h4_available_at"] = joined["h4_available_at"].to_numpy()
    out["h4_context_age_hours"] = age.to_numpy()
    status = np.where(idx < 0, "NONE", np.where(age.to_numpy() > cfg.max_context_age_hours, "STALE", "OK"))
    out["h4_context_status"] = status
    for col in H4_FIELDS_FROM_FEATURES:
        vals = h4f[col].to_numpy(object) if col in h4f else np.full(len(h4f), None, dtype=object)
        out[f"h4_{col}"] = [vals[i] if i >= 0 else None for i in idx]
    for col in H4_FIELDS_FROM_CONTEXT:
        vals = h4c[col].to_numpy(object) if h4c is not None and col in h4c else np.full(len(h4f), None, dtype=object)
        out[f"h4_{col}"] = [vals[i] if i >= 0 else None for i in idx]
    return out
