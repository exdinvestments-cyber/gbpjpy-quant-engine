"""Phase 1E: MT4/MT5 portability and scope (no orders, no volume/lot sizing, no monetary risk, no broker)."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import gbpjpy_engine
from gbpjpy_engine.trade import (BreakEvenPolicy, BrokerConstraintSource, SymbolSpecSource, TimeExitPolicy, TrailingPolicy,
                                 TransactionCostModel)

PKG = Path(gbpjpy_engine.__file__).parent / "trade"
ROOT = Path(gbpjpy_engine.__file__).parents[2]
FORBIDDEN = {"lot", "lots", "lot_size", "volume", "position_size", "risk_amount", "risk_percent", "currency_risk",
             "account_balance", "balance", "equity", "leverage", "margin", "order_send", "ordersend", "place_order",
             "submit_order", "send_order", "ticket", "order_ticket", "account_info", "positions_get", "symbol_info_tick",
             "copy_rates", "metatrader5", "mql4", "mql5", "martingale", "grid"}


def _names(py: Path) -> set[str]:
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


def test_trade_package_imports_are_core_only():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    for py in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= allowed, py.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed, (py.name, node.module)


def test_no_sizing_monetary_risk_orders_or_platform_calls():
    for py in PKG.rglob("*.py"):
        hits = _names(py) & FORBIDDEN
        assert not hits, (py.name, hits)


def test_adapter_facing_interfaces_are_protocols():
    for proto in (SymbolSpecSource, BrokerConstraintSource, TransactionCostModel, BreakEvenPolicy, TrailingPolicy,
                  TimeExitPolicy):
        assert getattr(proto, "_is_protocol", False), proto


def test_phase1e_documentation():
    doc = (ROOT / "docs" / "PHASE_1E_TRADE_CONSTRUCTION.md").read_text().lower()
    for token in ("not an order", "risk before reward", "1r", "never moved closer", "never moved farther", "unknown",
                  "mae", "account", "known limitations", "partial", "break-even", "trailing", "time"):
        assert token in doc, token
    iface = (ROOT / "docs" / "BROKER_NEUTRAL_INTERFACES.md").read_text()
    assert "## Trade proposal" in iface and "## Broker stop constraints" in iface
