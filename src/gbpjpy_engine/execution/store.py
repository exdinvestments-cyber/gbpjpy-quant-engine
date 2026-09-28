"""Persistent execution state and append-only event store (Phase 1G).

Persisted: order intents, lifecycle state and history, submission attempts,
fills (deduplicated by fill id), positions, protection state, reconciliation
issues, incidents, execution halt state, heartbeat, latency marks and the
append-only event log.  Storage mirrors the Phase 1F risk store: checksummed
canonical JSON, atomic replace, in-process lock plus an exclusive lock file.
A checksum failure raises ``CorruptedExecutionState`` and execution fails
closed (HALTED).

The event log is append-only: there is no API that edits or deletes an event;
current state is derived data and never overwrites history.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..risk.state import checksum


class CorruptedExecutionState(RuntimeError):
    pass


@dataclass
class ExecutionState:
    execution_state: str = "EXECUTION_ENABLED"
    halt_reason: str | None = None
    halt_source: str | None = None
    intents: dict = field(default_factory=dict)  # order_intent_id -> intent dict (immutable content)
    orders: dict = field(default_factory=dict)  # order_intent_id -> lifecycle record
    fills: dict = field(default_factory=dict)  # fill_id -> fill record
    positions: dict = field(default_factory=dict)  # position_id -> position record
    incidents: list = field(default_factory=list)
    reconciliation: dict = field(default_factory=lambda: {"startup_done": False, "open_issues": [], "last_run": None})
    control_actions: dict = field(default_factory=dict)  # idempotency_key -> close/modify/cancel/emergency request record
    heartbeat: dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {"consecutive_rejections": 0, "stale_quotes": 0, "disconnects": 0,
                                                    "mismatches": 0})
    events: list = field(default_factory=list)  # append-only

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(d: dict) -> "ExecutionState":
        if not isinstance(d, dict) or set(d) - set(ExecutionState.__dataclass_fields__):
            raise CorruptedExecutionState("unexpected execution-state structure")
        return ExecutionState(**d)


def _canon(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), default=str)


class InMemoryExecutionStore:
    def __init__(self):
        self._blob = None
        self._lock = threading.RLock()

    @contextmanager
    def transaction(self):
        with self._lock:
            yield

    def load(self) -> ExecutionState:
        if self._blob is None:
            return ExecutionState()
        d = json.loads(self._blob)
        if d.get("checksum") != checksum(d.get("state", {})):
            raise CorruptedExecutionState("checksum mismatch")
        return ExecutionState.from_json(d["state"])

    def save(self, st: ExecutionState) -> None:
        p = json.loads(_canon(st.to_json()))
        self._blob = _canon({"state": p, "checksum": checksum(p)})

    def raw(self):
        return self._blob

    def set_raw(self, blob) -> None:
        self._blob = blob


class JsonFileExecutionStore:
    _locks: dict = {}

    def __init__(self, path, lock_timeout: float = 10.0):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.lock_timeout = lock_timeout
        self._t = JsonFileExecutionStore._locks.setdefault(str(self.path.resolve()), threading.RLock())
        self._depth = 0

    @contextmanager
    def transaction(self):
        with self._t:
            if self._depth == 0:
                end = time.monotonic() + self.lock_timeout
                while True:
                    try:
                        fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        os.close(fd)
                        break
                    except FileExistsError:
                        if time.monotonic() > end:
                            raise TimeoutError(f"execution-state lock {self.lock_path} is held")
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

    def load(self) -> ExecutionState:
        if not self.path.exists():
            return ExecutionState()
        try:
            d = json.loads(self.path.read_text())
        except (OSError, ValueError) as exc:
            raise CorruptedExecutionState(str(exc)) from exc
        if not isinstance(d, dict) or d.get("checksum") != checksum(d.get("state", {})):
            raise CorruptedExecutionState("checksum mismatch")
        return ExecutionState.from_json(d["state"])

    def save(self, st: ExecutionState) -> None:
        p = json.loads(_canon(st.to_json()))
        tmp = self.path.with_suffix(self.path.suffix + f".tmp{os.getpid()}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(_canon({"state": p, "checksum": checksum(p)}))
        os.replace(tmp, self.path)

    def quarantine(self):
        if self.path.exists():
            os.replace(self.path, self.path.with_suffix(self.path.suffix + f".corrupt{int(time.time())}"))
