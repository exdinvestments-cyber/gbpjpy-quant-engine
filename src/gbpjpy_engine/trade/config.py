"""Phase 1E trade-construction configuration.

Every value is an untested engineering baseline, documented and NOT fitted to
profit.  There is no account, lot, currency-risk or leverage parameter here:
trade geometry is account-independent (sizing belongs to a future account-level
risk engine).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..config import _p

STOP_TYPES = ("STRUCTURAL_INVALIDATION", "SWING_INVALIDATION", "ZONE_INVALIDATION", "RECLAIM_FAILURE",
              "BREAK_RETEST_FAILURE", "VOLATILITY_ADJUSTED_STRUCTURE")


@dataclass(frozen=True)
class StopConfig:
    policies: tuple = _p(
        (
            ("TREND_PULLBACK_CONTINUATION", ("STRUCTURAL_INVALIDATION", "SWING_INVALIDATION", "ZONE_INVALIDATION")),
            ("BREAK_RETEST_CONTINUATION", ("BREAK_RETEST_FAILURE", "STRUCTURAL_INVALIDATION", "SWING_INVALIDATION")),
            ("LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION", ("RECLAIM_FAILURE", "STRUCTURAL_INVALIDATION", "SWING_INVALIDATION")),
            ("COMPRESSION_EXPANSION_IN_H4_DIRECTION", ("STRUCTURAL_INVALIDATION", "SWING_INVALIDATION", "ZONE_INVALIDATION")),
        ),
        "Per setup family: ordered structural invalidation references. The FIRST reference that exists on the "
        "correct side of the entry is the thesis-invalidation level. Targets and R never influence this choice.",
    )
    min_structure_atr: float = _p(0.5, "A preferred reference closer than this (H1 ATR) to the entry sits in ordinary noise: the "
                                       "next structural reference FARTHER away is used (VOLATILITY_ADJUSTED_STRUCTURE). "
                                       "A stop is never moved closer.")
    buffer_atr: float = _p(0.10, "Minimum noise buffer beyond the structural level, in H1 ATR (untested).")
    wick_lookback: int = _p(50, "Completed H1 bars whose adverse wicks form the wick-noise distribution.")
    wick_quantile: float = _p(0.75, "Quantile of adverse wick length used as the wick-noise buffer (untested).")
    zone_min_strength: float = _p(50.0, "Minimum level strength for an H1 zone to be a ZONE_INVALIDATION reference.")
    min_stop_pips: float = _p(5.0, "Stops closer than this (pips) are rejected as nonsensical (never widened to comply).")
    tight_atr: float = _p(0.5, "Stop distance below this many ATR = TOO_TIGHT.")
    normal_atr: float = _p(2.5, "Stop distance up to this many ATR = NORMAL.")
    wide_atr: float = _p(4.0, "Stop distance up to this many ATR = WIDE; beyond = EXTREME (rejected: STOP_TOO_WIDE).")
    percentile_lookback: int = _p(200, "Bars forming the trailing 5-bar-range distribution for stop_distance_percentile.")
    noise_reject_score: float = _p(70.0, "stop_noise_risk_score at/above which the trade is rejected (STOP_INSIDE_NOISE).")
    noise_warn_score: float = _p(50.0, "stop_noise_risk_score at/above which STOP_INSIDE_NOISE is flagged as a conflict.")


@dataclass(frozen=True)
class TargetConfig:
    max_targets: int = _p(3, "Structural targets kept in the ladder (T1..Tn).")
    cluster_tolerance_atr: float = _p(0.5, "Target/barrier levels within this many ATR are one cluster (counted once).")
    zone_min_strength: float = _p(50.0, "Minimum level strength for an H1 zone to count as a target/barrier.")
    min_target_atr: float = _p(0.5, "Clusters closer than this to the entry cannot be targets (they remain path barriers).")
    front_run_atr: float = _p(0.05, "Targets sit this many ATR BEFORE the structural level (never beyond it).")
    min_reachability: float = _p(35.0, "Targets with reachability below this are TARGET_UNREALISTIC and cannot be primary.")
    reach_full_atr: float = _p(1.0, "Distance (ATR) at/below which the distance component of reachability is 1.")
    reach_zero_atr: float = _p(8.0, "Distance (ATR) at/above which the distance component of reachability is 0; targets beyond "
                                         "it are never realistic.")
    strong_barrier: float = _p(60.0, "A path barrier at/above this strength counts as strong.")
    congested_strong_barriers: int = _p(2, "Strong barriers between entry and the primary target at/above which the path is congested.")
    congested_density: float = _p(60.0, "Path barrier density at/above which the path is congested.")
    density_window_atr: float = _p(3.0, "Path clusters are weighted by proximity over this distance (ATR).")
    density_norm: float = _p(1.5, "Weighted path-cluster sum mapping to density 100.")


@dataclass(frozen=True)
class CostConfig:
    assumed_slippage_pips_per_side: float = _p(
        0.5, "When no slippage model exists, this CONSERVATIVE assumption (pips per fill) is applied and flagged UNKNOWN.")
    assumed_commission_pips_round_turn: float = _p(
        0.5, "When no commission is known, this CONSERVATIVE assumption (pip-equivalent) is applied and flagged UNKNOWN.")


@dataclass(frozen=True)
class AsymmetryConfig:
    min_net_r: tuple = _p(
        (("TREND_PULLBACK_CONTINUATION", 1.5), ("BREAK_RETEST_CONTINUATION", 1.5),
         ("LIQUIDITY_SWEEP_REVERSAL_IN_H4_DIRECTION", 1.5), ("COMPRESSION_EXPANSION_IN_H4_DIRECTION", 1.5)),
        "Minimum estimated net R of the PRIMARY target per setup family. Unfitted baseline - NOT assumed optimal.",
    )


@dataclass(frozen=True)
class TradeScoringConfig:
    w_entry: float = _p(0.20, "Trade-quality family weight: ENTRY_QUALITY (Phase 1D entry_quality_score).")
    w_stop: float = _p(0.20, "Family weight: STOP_QUALITY.")
    w_target: float = _p(0.20, "Family weight: TARGET_QUALITY (structural legitimacy + reachability of the primary).")
    w_asymmetry: float = _p(0.15, "Family weight: ASYMMETRY (estimated net R).")
    w_path: float = _p(0.10, "Family weight: PATH_QUALITY (100 - barrier density on the path to the primary).")
    w_context: float = _p(0.15, "Family weight: MARKET_CONTEXT (H4 context score and permission confidence).")
    conflict_penalty: float = _p(0.5, "quality *= 1 - conflict_penalty x conflict/100.")
    min_quality: float = _p(40.0, "trade_construction_quality_score below which the trade is rejected.")
    max_conflict: float = _p(60.0, "trade_construction_conflict_score above which the trade is rejected.")


@dataclass(frozen=True)
class ManagementConfig:
    """Architecture only: every management feature is DISABLED by default (no assumed edge)."""

    exit_plan: str = _p("SINGLE_TARGET", "Default exit representation: SINGLE_TARGET (100% at the primary target). "
                                         "Multi-target splits exist for later research only.")
    break_even_enabled: bool = _p(False, "Break-even policy (disabled: not assumed beneficial).")
    trailing_enabled: bool = _p(False, "Trailing-stop policy (disabled: not assumed beneficial).")
    time_exit_enabled: bool = _p(False, "Time-based exit policy (disabled: not assumed beneficial).")


@dataclass(frozen=True)
class TradeConfig:
    stop: StopConfig = field(default_factory=StopConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    asymmetry: AsymmetryConfig = field(default_factory=AsymmetryConfig)
    scoring: TradeScoringConfig = field(default_factory=TradeScoringConfig)
    management: ManagementConfig = field(default_factory=ManagementConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:16]

    def stop_policy(self, family: str) -> tuple:
        for fam, refs in self.stop.policies:
            if fam == family:
                return tuple(refs)
        raise KeyError(f"no stop policy for {family!r}")

    def min_net_r(self, family: str) -> float:
        for fam, v in self.asymmetry.min_net_r:
            if fam == family:
                return float(v)
        raise KeyError(f"no minimum asymmetry for {family!r}")

    def validate(self) -> None:
        from ..h1.setups import FAMILIES

        if {f for f, _ in self.stop.policies} != set(FAMILIES) or {f for f, _ in self.asymmetry.min_net_r} != set(FAMILIES):
            raise ValueError("every Phase 1C setup family needs a stop policy and a minimum asymmetry")
        for _, refs in self.stop.policies:
            if not refs or not set(refs) <= set(STOP_TYPES):
                raise ValueError("stop policies must name known stop reference types")
        if any(v <= 0 for _, v in self.asymmetry.min_net_r):
            raise ValueError("minimum net R must be positive")
        s = self.stop
        if not (0 < s.tight_atr < s.normal_atr < s.wide_atr) or s.buffer_atr < 0 or s.min_stop_pips <= 0:
            raise ValueError("invalid stop thresholds")
        if self.cost.assumed_slippage_pips_per_side <= 0 or self.cost.assumed_commission_pips_round_turn <= 0:
            raise ValueError("unknown costs must use positive conservative assumptions (never zero)")
        if self.management.exit_plan not in ("SINGLE_TARGET", "MULTI_TARGET"):
            raise ValueError("management.exit_plan must be SINGLE_TARGET or MULTI_TARGET")
        w = [getattr(self.scoring, f.name) for f in fields(self.scoring) if f.name.startswith("w_")]
        if min(w) < 0 or sum(w) <= 0:
            raise ValueError("trade scoring weights must be non-negative and not all zero")


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, (list, tuple)) else v


def trade_config_from_dict(values) -> TradeConfig:
    base = TradeConfig()
    unknown = set(values or {}) - {f.name for f in fields(TradeConfig)}
    if unknown:
        raise KeyError(f"unknown trade config sections: {sorted(unknown)}")
    sections = {}
    for name, v in (values or {}).items():
        cur = getattr(base, name)
        bad = set(v) - {f.name for f in fields(cur)}
        if bad:
            raise KeyError(f"unknown config keys for {name}: {sorted(bad)}")
        sections[name] = replace(cur, **{k: _tuples(x) for k, x in v.items()})
    cfg = replace(base, **sections)
    cfg.validate()
    return cfg


def load_trade_config(path=None) -> TradeConfig:
    if path is None:
        cfg = TradeConfig()
        cfg.validate()
        return cfg
    from pathlib import Path

    p = Path(path)
    text = p.read_text()
    if p.suffix.lower() in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    return trade_config_from_dict(data)
