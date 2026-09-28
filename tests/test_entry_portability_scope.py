"""Phase 1D: MT4/MT5 portability and scope (no orders, no sizing, no stops/targets, no broker)."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import gbpjpy_engine
from gbpjpy_engine.entry import (EntryIntelligenceEngine, IntrabarConfirmationProvider, LowerTimeframeProvider,
                                 NewsProvider, QuoteSource, SlippageModel)

PKG = Path(gbpjpy_engine.__file__).parent / "entry"
ROOT = Path(gbpjpy_engine.__file__).parents[2]


def test_entry_package_imports_are_core_only():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    for py in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= allowed, py.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed, (py.name, node.module)


FORBIDDEN_NAMES = {"stop_loss", "take_profit", "position_size", "lot_size", "lots", "risk_amount", "order_send",
                   "ordersend", "place_order", "submit_order", "send_order", "order_ticket", "ticket", "broker_order_id",
                   "account_info", "positions_get", "symbol_info_tick", "copy_rates", "metatrader5", "mql4", "mql5"}


def _names(py: Path) -> set[str]:
    """Identifiers, attributes, arguments, keywords and exact string keys used in the code (prose is ignored)."""
    out = set()
    for node in ast.walk(ast.parse(py.read_text())):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r"[a-z_0-9]+", node.value):
            out.add(node.value)
    return {n.lower() for n in out}


def test_no_orders_sizing_stops_targets_or_broker_calls():
    for py in PKG.rglob("*.py"):
        hits = _names(py) & FORBIDDEN_NAMES
        assert not hits, (py.name, hits)
    assert {n for n in dir(EntryIntelligenceEngine) if not n.startswith("_")} == {"run"}


def test_extension_interfaces_are_protocols():
    for proto in (QuoteSource, SlippageModel, NewsProvider, LowerTimeframeProvider, IntrabarConfirmationProvider):
        assert getattr(proto, "_is_protocol", False), proto


def test_phase1d_documentation():
    doc = (ROOT / "docs" / "PHASE_1D_ENTRY_INTELLIGENCE.md").read_text().lower()
    for token in ("not an order", "signal_price", "executable_reference_price", "ask", "bid", "never treated as zero",
                  "intrabar", "slippage", "unknown", "known limitations", "counterfactual", "weekend", "one active"):
        assert token in doc, token
    iface = (ROOT / "docs" / "BROKER_NEUTRAL_INTERFACES.md").read_text()
    assert "## Quotes and executable prices" in iface and "## News calendar" in iface
