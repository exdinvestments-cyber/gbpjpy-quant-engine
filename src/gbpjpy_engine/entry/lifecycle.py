"""Entry-candidate lifecycle (Phase 1D).

States::

    NO_ENTRY -> WAITING_FOR_CONFIRMATION -> CONFIRMING -> ENTRY_CANDIDATE
                                         \\-> DEFERRED -> (next opportunity) ...
    terminal: REJECTED, EXPIRED, INVALIDATED
    (an ENTRY_CANDIDATE may later become EXPIRED or INVALIDATED when its
     availability window lapses or its structure fails; the accepted record
     itself is never rewritten)

Every transition is appended with the bar index, the phase in which it was
decided (``OPEN`` = at the first executable price after a bar close,
``CLOSE`` = at a bar close), the UTC time and a reason.  Nothing already
recorded is ever modified, so ``state_at(i)`` reproduces the state exactly as
it was known after bar ``i``.

An ENTRY_CANDIDATE is NOT an order: there is no size, risk amount, stop loss,
take profit or ticket anywhere in this package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

ACTIVE = ("WAITING_FOR_CONFIRMATION", "CONFIRMING", "DEFERRED", "ENTRY_CANDIDATE")
TERMINAL = ("REJECTED", "EXPIRED", "INVALIDATED")
DECISIONS = ("ACCEPT_ENTRY_CANDIDATE", "DEFER", "REJECT", "EXPIRE", "INVALIDATE")


@dataclass
class EntryCandidate:
    entry_candidate_id: str
    setup_id: str
    side: str
    setup_family: str
    qualified_index: int
    qualified_at: str
    qualified_close: float
    policy: dict
    setup_payload: dict
    state: str = "NO_ENTRY"
    transitions: list = field(default_factory=list)  # (index, phase, at, from, to, reason)
    audit: list = field(default_factory=list)  # (index, phase, at, step, detail)
    confirmation: dict | None = None
    confirmation_index: int | None = None
    confirmation_history: list = field(default_factory=list)  # discarded confirmations (e.g. across a market closure)
    opportunities: list = field(default_factory=list)  # every executable-price evaluation (decision records)
    accepted: dict | None = None
    accepted_index: int | None = None
    window_status: str | None = None
    decision: str | None = None
    end_index: int | None = None
    end_reason: str | None = None
    end_category: str | None = None
    reconfirm_after: int | None = None  # confirmations must come from bars >= this index (post-reopen)
    weekend_carryover: bool = False
    last_scores: dict | None = None  # last confirmation-family scores seen while waiting (research)
    ran_atr: float = 0.0  # directional move since qualification while unconfirmed (ATR)

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE

    @property
    def awaiting_price(self) -> bool:
        return (self.state == "CONFIRMING" and self.confirmation is not None) or \
               (self.state == "DEFERRED" and self.reconfirm_after is None)

    def move(self, i: int, phase: str, at: str, to: str, reason: str) -> None:
        self.transitions.append((i, phase, at, self.state, to, reason))
        self.state = to

    def note(self, i: int, phase: str, at: str, step: str, detail=None) -> None:
        self.audit.append((i, phase, at, step, detail))

    def end(self, i: int, phase: str, at: str, to: str, reason: str, category: str, decision: str | None = None) -> None:
        self.move(i, phase, at, to, reason)
        self.end_index, self.end_reason, self.end_category = i, reason, category
        if decision is not None:
            self.decision = decision
        self.note(i, phase, at, f"ENTRY {to}", reason)

    def state_at(self, i: int) -> str:
        st = "NO_ENTRY"
        for idx, _, _, _, to, _ in self.transitions:
            if idx <= i:
                st = to
        return st

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transitions"] = [{"index": t[0], "phase": t[1], "at": t[2], "from": t[3], "to": t[4], "reason": t[5]}
                            for t in self.transitions]
        d["audit"] = [{"index": t[0], "phase": t[1], "at": t[2], "step": t[3], "detail": t[4]} for t in self.audit]
        return d


__all__ = ["ACTIVE", "DECISIONS", "TERMINAL", "EntryCandidate"]
