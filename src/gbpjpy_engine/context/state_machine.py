"""Explicit H4 context state machine with transition validation and temporal
stability tracking.

The OBSERVED state for a bar is derived from that bar's context (rule order
below).  The machine records the current state, the previous state, bars in
state, when the state last changed, and whether the transition is one of the
EXPECTED transitions.  Unexpected transitions are still applied (the engine
never pretends the market did something else) but are logged with a reason.

No artificial confirmation lag is added: stability comes from the inputs being
swing-based (confirmed structure) rather than single-candle based.  The number
of state changes in the trailing ``flip_window_bars`` is exposed so noisy
periods are visible.

Rule order (first match): BLOCKED (hard blocker) -> COMPRESSION -> EXPANSION ->
FAILURE (latest break failed/invalidated within 6 bars) -> ACCEPTANCE
(latest break ACCEPTED within 6 bars, or accepting) -> BREAK (break within 3 bars)
-> REVERSAL_ATTEMPT (intermediate opposes primary and immediate agrees with
intermediate) -> TRANSITION (primary/intermediate transitional) -> DEEP_PULLBACK
/ PULLBACK (active counter-structure correction) -> CONTINUATION_ATTEMPT
(with-structure active leg not yet beyond the previous extreme) -> TREND ->
RANGE (ranging/neutral primary) -> UNDEFINED.
"""

from __future__ import annotations

from collections import deque

STATES = ("UNDEFINED", "BLOCKED", "TREND", "PULLBACK", "DEEP_PULLBACK", "CONTINUATION_ATTEMPT", "BREAK",
          "ACCEPTANCE", "FAILURE", "TRANSITION", "REVERSAL_ATTEMPT", "COMPRESSION", "EXPANSION", "RANGE")

EXPECTED = {
    "TREND": {"PULLBACK", "DEEP_PULLBACK", "BREAK", "ACCEPTANCE", "CONTINUATION_ATTEMPT", "TRANSITION", "COMPRESSION", "RANGE"},
    "PULLBACK": {"TREND", "DEEP_PULLBACK", "CONTINUATION_ATTEMPT", "BREAK", "TRANSITION", "COMPRESSION", "RANGE"},
    "DEEP_PULLBACK": {"PULLBACK", "TRANSITION", "REVERSAL_ATTEMPT", "CONTINUATION_ATTEMPT", "BREAK", "RANGE", "TREND"},
    "CONTINUATION_ATTEMPT": {"BREAK", "TREND", "PULLBACK", "DEEP_PULLBACK", "FAILURE", "ACCEPTANCE"},
    "BREAK": {"ACCEPTANCE", "FAILURE", "TREND", "PULLBACK", "CONTINUATION_ATTEMPT", "EXPANSION", "TRANSITION", "RANGE"},
    "ACCEPTANCE": {"TREND", "PULLBACK", "BREAK", "FAILURE", "CONTINUATION_ATTEMPT", "DEEP_PULLBACK"},
    "FAILURE": {"TRANSITION", "REVERSAL_ATTEMPT", "RANGE", "BREAK", "PULLBACK", "DEEP_PULLBACK", "TREND"},
    "TRANSITION": {"REVERSAL_ATTEMPT", "TREND", "RANGE", "BREAK", "PULLBACK", "DEEP_PULLBACK", "COMPRESSION", "FAILURE"},
    "REVERSAL_ATTEMPT": {"TRANSITION", "TREND", "BREAK", "FAILURE", "RANGE", "PULLBACK", "DEEP_PULLBACK"},
    "COMPRESSION": {"EXPANSION", "BREAK", "RANGE", "TREND", "TRANSITION"},
    "EXPANSION": {"ACCEPTANCE", "FAILURE", "BREAK", "TREND", "PULLBACK", "COMPRESSION", "TRANSITION"},
    "RANGE": {"COMPRESSION", "BREAK", "TREND", "TRANSITION", "PULLBACK", "DEEP_PULLBACK", "REVERSAL_ATTEMPT"},
}
FREE = {"UNDEFINED", "BLOCKED"}  # transitions to/from these are always expected


def observe(ctx: dict) -> tuple[str, str]:
    p, i, im = ctx["primary"], ctx["intermediate"], ctx["immediate"]
    bc = ctx["break_ctx"]
    if ctx["blockers"]:
        return "BLOCKED", "hard blocker: " + ",".join(ctx["blockers"])
    if ctx["expansion"]["expansion_state"] == "COMPRESSING":
        return "COMPRESSION", "compression episode active"
    if ctx["expansion"]["expansion_state"].startswith("EXPANDING"):
        return "EXPANSION", "expansion out of compression"
    if bc is not None and bc.bars_since_break <= 6:
        if bc.state in ("FAILED", "INVALIDATED"):
            return "FAILURE", f"{bc.direction} break {bc.state.lower()}"
        if bc.state == "ACCEPTED" or bc.acceptance_state == "ACCEPTING":
            return "ACCEPTANCE", f"{bc.direction} break accepted"
        if bc.bars_since_break <= 3:
            return "BREAK", f"{bc.direction} break {bc.state.lower()}"
    if p.direction and i.direction == -p.direction and im.direction == i.direction:
        return "REVERSAL_ATTEMPT", "intermediate and immediate oppose primary"
    if p.classification == "TRANSITIONAL" or i.classification == "TRANSITIONAL":
        return "TRANSITION", "transitional structure"
    D = p.direction or i.direction
    if D:
        if ctx["correction_active"]:
            band = ctx["retracement"]["retracement_band"]
            if band in ("DEEP", "VERY_DEEP"):
                return "DEEP_PULLBACK", f"{band.lower()} retracement against structure"
            return "PULLBACK", "counter-structure correction"
        if ctx["active_leg"] is not None and ctx["active_leg_with_structure"] and not ctx["active_leg_beyond_extreme"]:
            return "CONTINUATION_ATTEMPT", "with-structure leg below the prior extreme"
        return "TREND", "directional structure"
    if p.classification in ("RANGING", "NEUTRAL"):
        return "RANGE", f"primary {p.classification.lower()}"
    return "UNDEFINED", "no classifiable structure"


class ContextStateMachine:
    def __init__(self, flip_window: int):
        self.state = None
        self.previous = None
        self.since = None
        self.changed_at = None
        self.changes = deque()
        self.flip_window = flip_window
        self.unexpected: list[dict] = []

    def step(self, c: int, observed: str, reason: str, time_iso: str) -> dict:
        valid = True
        if self.state is None:
            self.state, self.since, self.changed_at = observed, c, time_iso
        elif observed != self.state:
            valid = observed in FREE or self.state in FREE or observed in EXPECTED.get(self.state, set())
            if not valid:
                self.unexpected.append({"index": c, "at": time_iso, "from": self.state, "to": observed, "reason": reason})
            self.previous, self.state, self.since, self.changed_at = self.state, observed, c, time_iso
            self.changes.append(c)
        while self.changes and self.changes[0] <= c - self.flip_window:
            self.changes.popleft()
        return {"context_state": self.state, "previous_context_state": self.previous,
                "bars_in_state": c - self.since, "state_changed_at": self.changed_at,
                "state_transition_expected": valid, "state_reason": reason,
                "state_changes_in_window": len(self.changes)}
