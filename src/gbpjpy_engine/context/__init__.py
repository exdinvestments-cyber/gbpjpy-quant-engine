"""Phase 1B: H4 context and directional permission (market analysis only).

Consumes Phase 1A/1A.1 outputs; produces context, LONG/SHORT context scores
and a directional permission (ALLOW_LONG / ALLOW_SHORT / ALLOW_BOTH /
BLOCK_ALL) describing what a future H1 engine may SEARCH for.  No trade
signals, no sizing, no execution.
"""

from .engine import ContextEngine, ContextResult

__all__ = ["ContextEngine", "ContextResult"]
