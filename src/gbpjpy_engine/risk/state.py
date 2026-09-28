"""Persistent risk state and state stores (Phase 1F).

Everything that protects capital across restarts lives here: global
ENABLED/PAUSED/HALTED state, day/week/month reference capital, high-water
marks, consecutive losses, reservations, processed approvals (idempotency),
the closed-result ledger and balance adjustments.

Persistence: JSON with every Decimal stored as an exact string, a SHA-256
checksum over the canonical payload, and atomic replace-on-write.  A checksum
mismatch or unreadable file raises ``CorruptedRiskState``; the engine then
fails closed (HALTED) and keeps the corrupted file for inspection.

Concurrency: ``transaction()`` holds an in-process lock and, for the file
store, an exclusive lock file, so load -> evaluate -> reserve -> save is atomic
across threads and cooperating processes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

GLOBAL_STATES = ("ENABLED", "PAUSED", "HALTED")


class CorruptedRiskState(RuntimeError):
    pass


@dataclass
class RiskState:
    version: int = 0
    global_state: str = "ENABLED"
    halt_reason: str | None = None
    halt_source: str | None = None  # AUTO | MANUAL
    halted_at: str | None = None
    pause_reason: str | None = None
    pause_until_day: str | None = None  # trading-day key after which a NEXT_TRADING_DAY pause lifts
    periods: dict = field(default_factory=dict)  # {"day": {"key", "reference", "set_at"}, "week": ..., "month": ...}
    hwm_balance: str | None = None
    hwm_equity: str | None = None
    consecutive_losses: int = 0
    pause_trigger_count: int | None = None  # consecutive-loss count that last triggered a pause (no re-trigger loop)
    ledger: list = field(default_factory=list)  # closed results (trading P&L only)
    adjustments: list = field(default_factory=list)  # deposits / withdrawals / corrections
    reservations: dict = field(default_factory=dict)  # risk_approval_id -> reservation
    approvals: dict = field(default_factory=dict)  # trade_proposal_id -> approved decision (idempotency)
    last_account_timestamp: str | None = None
    audit: list = field(default_factory=list)  # bounded audit of state-changing events

    def note(self, at: str, event: str, detail) -> None:
        self.audit.append({"at": at, "event": event, "detail": detail})
        if len(self.audit) > 500:
            del self.audit[: len(self.audit) - 500]

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "RiskState":
        known = {f for f in RiskState.__dataclass_fields__}
        if not isinstance(d, dict) or set(d) - known:
            raise CorruptedRiskState("unexpected risk-state structure")
        st = RiskState(**d)
        if st.global_state not in GLOBAL_STATES:
            raise CorruptedRiskState(f"invalid global state {st.global_state!r}")
        return st


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def checksum(payload: dict) -> str:
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


class InMemoryRiskStateStore:
    """Process-local store (tests / research). Same transaction semantics as the file store."""

    def __init__(self):
        self._blob: str | None = None
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self._lock:
            yield

    def load(self) -> RiskState:
        if self._blob is None:
            return RiskState()
        d = json.loads(self._blob)
        if d.get("checksum") != checksum(d.get("state", {})):
            raise CorruptedRiskState("checksum mismatch")
        return RiskState.from_json(d["state"])

    def save(self, st: RiskState) -> None:
        payload = st.to_json()
        self._blob = _canonical({"state": payload, "checksum": checksum(payload)})

    def raw(self) -> str | None:  # tests: simulate corruption
        return self._blob

    def set_raw(self, blob: str) -> None:
        self._blob = blob


class JsonFileRiskStateStore:
    """Durable store: checksummed JSON, atomic os.replace, exclusive lock file for cross-process safety."""

    _locks: dict = {}

    def __init__(self, path, lock_timeout: float = 10.0):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = lock_timeout
        self._tlock = JsonFileRiskStateStore._locks.setdefault(str(self.path.resolve()), threading.RLock())
        self._depth = 0

    @contextmanager
    def transaction(self):
        with self._tlock:
            if self._depth == 0:
                deadline = time.monotonic() + self.lock_timeout
                while True:
                    try:
                        fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.write(fd, str(os.getpid()).encode())
                        os.close(fd)
                        break
                    except FileExistsError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"risk-state lock {self.lock_path} is held")
                        time.sleep(0.005)
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0:
                    try:
                        os.remove(self.lock_path)
                    except FileNotFoundError:
                        pass

    def load(self) -> RiskState:
        if not self.path.exists():
            return RiskState()
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise CorruptedRiskState(f"unreadable risk state: {exc}") from exc
        if not isinstance(d, dict) or d.get("checksum") != checksum(d.get("state", {})):
            raise CorruptedRiskState("checksum mismatch")
        return RiskState.from_json(d["state"])

    def save(self, st: RiskState) -> None:
        payload = st.to_json()
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(_canonical({"state": payload, "checksum": checksum(payload)}))
        os.replace(tmp, self.path)

    def quarantine(self) -> Path | None:
        if self.path.exists():
            dst = self.path.with_suffix(self.path.suffix + f".corrupt{int(time.time())}")
            os.replace(self.path, dst)
            return dst
        return None


def snapshot(st: RiskState) -> RiskState:
    return copy.deepcopy(st)
