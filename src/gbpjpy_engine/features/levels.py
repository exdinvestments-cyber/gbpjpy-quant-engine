"""Objective H4 support/resistance zone map.

At each bar ``c`` zones are rebuilt from information known at the close of
``c`` only:

* confirmed swing highs/lows that are alive at ``c`` (or were superseded
  after their confirmation, still within the lookback)
* structural break levels whose break bar is ``<= c``
* the trailing range high/low (bars ``c - range_extreme_lookback + 1 .. c``)

Candidate prices are clustered with single linkage using a volatility-aware
tolerance (``cluster_atr_mult * ATR[c]``).  Interactions, rejections and
breaks are counted on bars inside the trailing lookback up to ``c``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import LevelConfig
from .structure import StructureResult


LEVEL_COLUMNS = [
    "nearest_support", "support_lower", "support_upper", "support_strength",
    "distance_to_support", "distance_to_support_atr",
    "nearest_resistance", "resistance_lower", "resistance_upper", "resistance_strength",
    "distance_to_resistance", "distance_to_resistance_atr", "n_zones",
]


@dataclass
class Zone:
    lower: float
    upper: float
    midpoint: float
    zone_type: str  # support | resistance | (inside: price within zone)
    sources: list
    n_sources: int
    interactions: int
    touch_bars: int
    last_interaction_index: int | None
    last_interaction_time: str | None
    bars_since_interaction: int | None
    age_bars: int
    rejections: int
    rejection_strength: float
    breaks: int
    distance: float
    distance_atr: float
    strength: float

    def to_dict(self) -> dict:
        return asdict(self)


def _cluster(points: list[tuple[float, str, int]], tol: float) -> list[list[tuple[float, str, int]]]:
    if not points:
        return []
    pts = sorted(points, key=lambda p: (p[0], p[1], p[2]))
    clusters = [[pts[0]]]
    for p in pts[1:]:
        if p[0] - clusters[-1][-1][0] <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return clusters


def compute_levels(
    df: pd.DataFrame,
    atr_s: pd.Series,
    structure: StructureResult,
    cfg: LevelConfig,
) -> tuple[pd.DataFrame, list[list[Zone]]]:
    n = len(df)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    atr = atr_s.to_numpy(float)
    ts = list(df["timestamp"])
    LB = cfg.lookback_bars
    RX = cfg.range_extreme_lookback

    swings = structure.swings
    breaks = structure.breaks
    sw_acc = np.array([s.accepted_index for s in swings], dtype=int)
    sw_piv = np.array([s.pivot_index for s in swings], dtype=int)
    br_idx = np.array([b.break_index for b in breaks], dtype=int)
    rows = []
    zones_by_bar: list[list[Zone]] = []

    for c in range(n):
        a = atr[c]
        if np.isnan(a) or a <= 0:
            zones_by_bar.append([])
            rows.append({})
            continue
        start = max(0, c - LB + 1)
        pts: list[tuple[float, str, int]] = []
        # swings known at c (accepted by c) with pivot inside the lookback;
        # superseded swings still mark a price that was once structural
        for k in np.flatnonzero((sw_acc <= c) & (sw_piv >= start)):
            s = swings[k]
            pts.append((s.price, f"swing_{s.kind}", s.pivot_index))
        for k in np.flatnonzero((br_idx <= c) & (br_idx >= start)):
            b = breaks[k]
            pts.append((b.level, "break_level", b.break_index))
        rs = max(0, c - RX + 1)
        if c - rs + 1 >= min(RX, 20):
            hi_idx = rs + int(np.argmax(h[rs:c + 1]))
            lo_idx = rs + int(np.argmin(l[rs:c + 1]))
            pts.append((float(h[hi_idx]), "range_high", hi_idx))
            pts.append((float(l[lo_idx]), "range_low", lo_idx))

        tol = cfg.cluster_atr_mult * a
        hw = cfg.min_half_width_atr * a
        hs, ls, cs = h[start:c + 1], l[start:c + 1], cl[start:c + 1]
        zones: list[Zone] = []
        for cluster in _cluster(pts, tol):
            prices = [p[0] for p in cluster]
            lo, hi = min(prices), max(prices)
            mid = (lo + hi) / 2
            lo, hi = min(lo, mid - hw), max(hi, mid + hw)
            touched = (hs >= lo) & (ls <= hi)
            touch_bars = int(touched.sum())
            # interaction episodes = runs of consecutive touching bars
            episodes = int(touched[0]) + int(np.sum(touched[1:] & ~touched[:-1])) if touch_bars else 0
            last_touch = int(np.flatnonzero(touched)[-1]) + start if touch_bars else None
            support_rej = touched & (ls <= hi) & (cs > hi)
            resist_rej = touched & (hs >= lo) & (cs < lo)
            rejections = int(support_rej.sum() + resist_rej.sum())
            side = np.where(cs > hi, 1, np.where(cs < lo, -1, 0))
            nz = side[side != 0]
            zone_breaks = int(np.sum(nz[1:] != nz[:-1])) if len(nz) > 1 else 0
            first_idx = min(p[2] for p in cluster)
            sources = sorted({p[1] for p in cluster})
            n_swing_src = sum(p[1].startswith("swing") for p in cluster)
            rej_strength = 100.0 * rejections / touch_bars if touch_bars else 0.0

            price = cl[c]
            if price > hi:
                ztype, dist = "support", price - hi
            elif price < lo:
                ztype, dist = "resistance", lo - price
            else:
                ztype, dist = ("support" if price >= mid else "resistance"), 0.0

            recency = 1.0 - ((c - last_touch) / LB) if last_touch is not None else 0.0
            strength = (
                30.0 * min(episodes / 4.0, 1.0)
                + 25.0 * (rej_strength / 100.0)
                + 15.0 * min(len(sources) / 3.0, 1.0)
                + 10.0 * min(n_swing_src / 3.0, 1.0)
                + 10.0 * max(recency, 0.0)
                + 10.0 * (1.0 - min(zone_breaks / 3.0, 1.0))
            )
            zones.append(
                Zone(
                    lower=float(lo), upper=float(hi), midpoint=float(mid), zone_type=ztype, sources=sources,
                    n_sources=len(cluster), interactions=episodes, touch_bars=touch_bars,
                    last_interaction_index=last_touch,
                    last_interaction_time=ts[last_touch].isoformat() if last_touch is not None else None,
                    bars_since_interaction=(c - last_touch) if last_touch is not None else None,
                    age_bars=int(c - first_idx), rejections=rejections, rejection_strength=round(rej_strength, 2),
                    breaks=zone_breaks, distance=float(dist), distance_atr=float(dist / a),
                    strength=round(min(max(float(strength), 0.0), 100.0), 2),
                )
            )
        zones.sort(key=lambda z: (-z.strength, z.distance))
        zones = zones[: cfg.max_zones]
        zones.sort(key=lambda z: z.midpoint)
        zones_by_bar.append(zones)

        sup = [z for z in zones if z.zone_type == "support"]
        res = [z for z in zones if z.zone_type == "resistance"]
        ns = min(sup, key=lambda z: (z.distance, -z.midpoint)) if sup else None
        nr = min(res, key=lambda z: (z.distance, z.midpoint)) if res else None
        rows.append(
            {
                "nearest_support": ns.midpoint if ns else np.nan,
                "support_lower": ns.lower if ns else np.nan,
                "support_upper": ns.upper if ns else np.nan,
                "support_strength": ns.strength if ns else np.nan,
                "distance_to_support": ns.distance if ns else np.nan,
                "distance_to_support_atr": ns.distance_atr if ns else np.nan,
                "nearest_resistance": nr.midpoint if nr else np.nan,
                "resistance_lower": nr.lower if nr else np.nan,
                "resistance_upper": nr.upper if nr else np.nan,
                "resistance_strength": nr.strength if nr else np.nan,
                "distance_to_resistance": nr.distance if nr else np.nan,
                "distance_to_resistance_atr": nr.distance_atr if nr else np.nan,
                "n_zones": len(zones),
            }
        )
    frame = pd.DataFrame(rows, index=df.index, columns=LEVEL_COLUMNS)
    return frame, zones_by_bar
