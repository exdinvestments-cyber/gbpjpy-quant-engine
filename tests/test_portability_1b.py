"""Phase 1B architectural review: the strategy core stays MT4/MT5/MQL/broker agnostic."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import gbpjpy_engine
from gbpjpy_engine.data.interfaces import BarSource, IntrabarProvider, TickSource

PKG = Path(gbpjpy_engine.__file__).parent
ROOT = PKG.parents[1]
FORBIDDEN_TEXT = [r"metatrader", r"\bmql4\b", r"\bmql5\b", r"\.mq4\b", r"\.mq5\b", r"\bex4\b", r"\bex5\b",
                  r"\bMT4\b(?!/)", r"\bMT5\b(?!/)", r"order_send", r"ordersend", r"positions_get", r"account_info",
                  r"symbol_info_tick", r"copy_rates"]


def _code_only(py: Path) -> str:
    """Source without comments/docstrings (documentation may mention the future adapters)."""
    tree = ast.parse(py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)) and node.body and \
                isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant):
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_no_platform_or_execution_dependency_in_code():
    offenders = []
    for py in PKG.rglob("*.py"):
        code = _code_only(py)
        for pat in FORBIDDEN_TEXT:
            if re.search(pat, code, flags=re.IGNORECASE):
                offenders.append((str(py.relative_to(PKG)), pat))
    assert not offenders, offenders


def test_no_mql_or_platform_files_in_repository():
    bad = [p for ext in ("*.mq4", "*.mq5", "*.ex4", "*.ex5", "*.mqh", "*.set") for p in ROOT.rglob(ext)]
    assert not bad


def test_context_layer_imports_only_core_modules():
    for py in (PKG / "context").rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                assert node.module is None or node.module.split(".")[0] in (
                    "config", "features", "reason_codes", "breakouts", "displacement", "hierarchy", "legs", "liquidity",
                    "location", "maturity", "permission", "room", "state_machine", "zones", "engine"), (py.name, node.module)
                assert not (node.module or "").startswith(("engine", "cli", "logging_utils")) or py.name == "__init__.py"


def test_market_data_interfaces_are_protocols_only():
    for proto in (BarSource, TickSource, IntrabarProvider):
        assert getattr(proto, "_is_protocol", False)


def test_broker_neutral_interface_document():
    doc = (ROOT / "docs" / "BROKER_NEUTRAL_INTERFACES.md").read_text()
    for section in ("Bars", "Ticks", "Spread", "Symbol specification", "Account state", "Order request",
                    "Order result", "Position state"):
        assert f"## {section}" in doc, section
    assert "not implemented" in doc.lower()
