"""Phase 1G execution-safety policy.  Every value is CONFIGURATION, NOT VALIDATED EDGE.

Execution may only ACCEPT, DEFER, REJECT, CANCEL, RECONCILE or HALT; nothing
here can improve a price, tighten a stop, extend a target or raise a volume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace

from ..config import _p

NOTE = " [CONFIGURATION - NOT VALIDATED EDGE]"


def _d(v, doc):
    return _p(v, doc + NOTE)


@dataclass(frozen=True)
class FreshnessPolicy:
    quote_max_age_seconds: float = _d(10.0, "A quote older than this is STALE_QUOTE (never submitted on).")
    account_max_age_seconds: float = _d(60.0, "Account snapshots older than this block submission.")
    capabilities_max_age_seconds: float = _d(3600.0, "Symbol/capability data older than this blocks submission.")
    rate_max_age_seconds: float = _d(7200.0, "Conversion rates older than this block the risk recheck.")


@dataclass(frozen=True)
class IntentPolicy:
    validity_minutes: float = _d(30.0, "Finite validity of an OrderIntent after creation.")
    aging_fraction: float = _d(0.5, "Share of the validity after which the intent is AGING.")
    max_spread_pips: float = _d(4.0, "Maximum spread (pips) at submission.")
    max_spread_atr_fraction: float = _d(0.15, "Maximum spread as a fraction of H1 ATR at submission.")
    spread_action: str = _d("DEFER", "DEFER or REJECT when the final spread check fails.")
    max_deterioration_pips: float = _d(8.0, "Maximum adverse executable-price move vs the approved reference (pips).")
    max_deterioration_atr: float = _d(0.25, "Maximum adverse move in H1 ATR.")
    deterioration_action: str = _d("REJECT", "REJECT or DEFER on excessive deterioration. Never chase.")
    max_slippage_pips: float = _d(3.0, "Maximum slippage an adapter may accept on a market order (sent with the request).")
    allow_volume_reduction: bool = _d(True, "Reduce volume (round DOWN) when a price move would exceed the Phase 1F cap.")
    min_room_atr: float = _d(0.5, "Minimum distance (ATR) from the executable price to the first opposing structure.")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = _d(3, "Maximum submission attempts per intent (each attempt re-runs the full gate).")
    max_requotes: int = _d(2, "Maximum price-change/requote attempts per intent.")
    backoff_base_seconds: float = _d(1.0, "Deterministic backoff base for transient infrastructure failures.")
    backoff_factor: float = _d(2.0, "Backoff multiplier per attempt.")
    backoff_max_seconds: float = _d(8.0, "Backoff ceiling.")
    submission_timeout_seconds: float = _d(10.0, "No acknowledgement within this (monotonic) time = UNKNOWN.")


@dataclass(frozen=True)
class FillPolicy:
    partial_fill_policy: str = _d("ACCEPT_PARTIAL", "ACCEPT_PARTIAL, CANCEL_REMAINDER or CONTINUE_WITHIN_VALIDITY_WINDOW.")
    excessive_slippage_pips: float = _d(10.0, "A fill slipping adversely by more than this raises an incident.")


@dataclass(frozen=True)
class ProtectionPolicy:
    require_stop: bool = _d(True, "Every filled position must carry the structural stop.")
    require_target: bool = _d(True, "Every filled position must carry the primary target.")
    unprotected_action: str = _d("EMERGENCY_CLOSE_INTENT", "EMERGENCY_CLOSE_INTENT (recorded for the future adapter) "
                                                           "or ALERT_ONLY. Nothing is closed in Phase 1G.")
    price_tolerance_points: float = _d(0.0, "Allowed difference (points) between requested and confirmed protective prices.")


@dataclass(frozen=True)
class BreakerPolicy:
    max_consecutive_rejections: int = _d(3, "Consecutive rejections that HALT execution.")
    max_reconciliation_mismatches: int = _d(2, "Critical reconciliation mismatches that HALT execution.")
    max_stale_quote_events: int = _d(5, "Consecutive stale-quote gates that PAUSE execution.")
    max_disconnects: int = _d(3, "Disconnects within the session that PAUSE execution.")
    allow_degraded_connection: bool = _d(False, "Permit submission while the connection is DEGRADED.")
    allow_unknown_session: bool = _d(False, "Permit submission while the trading session status is UNKNOWN.")


@dataclass(frozen=True)
class ExecutionPolicy:
    freshness: FreshnessPolicy = field(default_factory=FreshnessPolicy)
    intent: IntentPolicy = field(default_factory=IntentPolicy)
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    fills: FillPolicy = field(default_factory=FillPolicy)
    protection: ProtectionPolicy = field(default_factory=ProtectionPolicy)
    breaker: BreakerPolicy = field(default_factory=BreakerPolicy)

    def to_dict(self) -> dict:
        return asdict(self)

    def policy_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def validate(self) -> None:
        i = self.intent
        if i.spread_action not in ("DEFER", "REJECT") or i.deterioration_action not in ("DEFER", "REJECT"):
            raise ValueError("spread/deterioration actions must be DEFER or REJECT")
        if self.fills.partial_fill_policy not in ("ACCEPT_PARTIAL", "CANCEL_REMAINDER", "CONTINUE_WITHIN_VALIDITY_WINDOW"):
            raise ValueError("invalid partial-fill policy")
        if self.protection.unprotected_action not in ("EMERGENCY_CLOSE_INTENT", "ALERT_ONLY"):
            raise ValueError("invalid unprotected-position action")
        if self.retry.max_attempts < 1 or self.retry.max_attempts > 10 or self.retry.max_requotes > 10:
            raise ValueError("retries must be bounded (1..10)")
        if i.validity_minutes <= 0 or not (0 < i.aging_fraction < 1):
            raise ValueError("intent validity must be finite and positive")


def execution_policy_from_dict(values) -> ExecutionPolicy:
    base = ExecutionPolicy()
    unknown = set(values or {}) - {f.name for f in fields(ExecutionPolicy)}
    if unknown:
        raise KeyError(f"unknown execution policy sections: {sorted(unknown)}")
    sections = {}
    for name, v in (values or {}).items():
        cur = getattr(base, name)
        bad = set(v) - {f.name for f in fields(cur)}
        if bad:
            raise KeyError(f"unknown execution policy keys for {name}: {sorted(bad)}")
        sections[name] = replace(cur, **v)
    pol = replace(base, **sections)
    pol.validate()
    return pol


def load_execution_policy(path=None) -> ExecutionPolicy:
    if path is None:
        pol = ExecutionPolicy()
        pol.validate()
        return pol
    from pathlib import Path

    import yaml

    return execution_policy_from_dict(yaml.safe_load(Path(path).read_text()) or {})
