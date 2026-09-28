"""Phase 1G portability and scope: platform-neutral port, no network, no platform names, documented contracts."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

import gbpjpy_engine
from gbpjpy_engine.execution import (ExecutionPolicy, ExecutionPort, PaperBroker, SimClock, SymbolMapping, canonical_rejection,
                                     execution_policy_from_dict, load_execution_policy)
from gbpjpy_engine.execution.port import Clock, SymbolMapper
from gbpjpy_engine.reason_codes import REASON_DESCRIPTIONS, ReasonCode

PKG = Path(gbpjpy_engine.__file__).parent / "execution"
ROOT = Path(gbpjpy_engine.__file__).parents[2]
FORBIDDEN = {"order_send", "ordersend", "place_order", "submit_order", "send_order", "account_info", "positions_get",
             "symbol_info_tick", "copy_rates", "metatrader5", "mql4", "mql5", "martingale", "grid", "lot", "lots", "lot_size",
             "leverage", "api_key", "password", "token", "secret", "login", "server"}


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
    return {n.lower() for n in out}


def test_execution_imports_are_core_only_and_no_network():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    banned = {"socket", "http", "urllib", "ssl", "asyncio", "subprocess", "ftplib", "smtplib", "xmlrpc"}
    for py in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods = {a.name.split(".")[0] for a in node.names}
                assert mods <= allowed and not mods & banned, (py.name, mods)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed and node.module.split(".")[0] not in banned, (py.name, node.module)


def test_no_platform_order_functions_credentials_or_forbidden_sizing_in_code():
    for py in PKG.rglob("*.py"):
        hits = _names(py) & FORBIDDEN
        assert not hits, (py.name, hits)


def test_no_wall_clock_in_decision_code():
    for name in ("engine.py", "paper.py", "model.py", "port.py"):
        text = (PKG / name).read_text()
        assert not re.search(r"datetime\.now|Timestamp\.now|time\.time\(|utcnow", text), name


def test_adapter_facing_interfaces_are_protocols_and_paper_conforms():
    for proto in (ExecutionPort, Clock, SymbolMapper):
        assert getattr(proto, "_is_protocol", False), proto
    import pandas as pd

    assert isinstance(PaperBroker(), ExecutionPort)
    assert isinstance(SimClock(pd.Timestamp("2024-01-01", tz="UTC")), Clock)
    assert isinstance(SymbolMapping({"GBPJPY": "GBPJPY.x"}), SymbolMapper)


def test_symbol_mapping_and_canonical_rejections():
    m = SymbolMapping({"GBPJPY": "GBPJPYm"})
    assert m.to_broker("GBPJPY") == "GBPJPYm" and m.to_canonical("GBPJPYm") == "GBPJPY"
    with pytest.raises(KeyError):
        m.to_broker("EURJPY")
    with pytest.raises(KeyError):
        m.to_canonical("GBPJPY")
    with pytest.raises(ValueError):
        SymbolMapping({"GBPJPY": "X", "EURJPY": "X"})
    table = {134: "INSUFFICIENT_MARGIN", 138: "REQUOTE"}
    assert canonical_rejection(138, table) == "REQUOTE" and canonical_rejection(9999, table) == "UNKNOWN_BROKER_REJECTION"


def test_policy_file_matches_defaults_and_every_value_is_labelled():
    assert load_execution_policy(ROOT / "config" / "execution_default.yaml") == ExecutionPolicy()
    from gbpjpy_engine.config import describe_config

    rows = describe_config(ExecutionPolicy())
    assert len(rows) > 30 and all("NOT VALIDATED EDGE" in r["doc"] for r in rows)
    with pytest.raises(KeyError):
        execution_policy_from_dict({"intent": {"chase_price": True}})
    with pytest.raises(ValueError):
        execution_policy_from_dict({"retry": {"max_attempts": 1000}})
    with pytest.raises(ValueError):
        execution_policy_from_dict({"intent": {"spread_action": "WIDEN"}})


def test_every_execution_reason_code_is_registered_and_described():
    valid = {c.value for c in ReasonCode}
    engine_src = (PKG / "engine.py").read_text()
    used = set(re.findall(r'"([A-Z][A-Z0-9_]{5,})"', engine_src))
    codes = {c for c in used if c in valid}
    for c in ("SUBMIT_READY", "QUOTE_STALE", "SPREAD_TOO_HIGH", "PRICE_DETERIORATED", "RISK_EXCEEDED_AFTER_PRICE_MOVE",
              "VOLUME_REDUCED", "RR_NO_LONGER_VALID", "MARKET_CLOSED", "CONNECTION_LOST", "INTENT_EXPIRED",
              "DUPLICATE_INTENT_BLOCKED", "SUBMISSION_TIMEOUT", "RECONCILIATION_REQUIRED", "BROKER_REJECTED",
              "PROTECTION_VALID", "UNPROTECTED_POSITION", "EXECUTION_PAUSED", "EXECUTION_HALTED", "PARTIAL_FILL", "FULL_FILL"):
        assert c in valid and c in REASON_DESCRIPTIONS, c
    assert codes and all(c in REASON_DESCRIPTIONS for c in codes)


def test_phase1g_documentation():
    doc = (ROOT / "docs" / "PHASE_1G_EXECUTION_SAFETY.md").read_text()
    low = doc.lower()
    for token in ("not validated edge", "write-ahead", "reconciliation", "unprotected_position", "testing only", "ask",
                  "bid", "round", "known limitations", "mt4 adapter requirements", "mt5 adapter requirements",
                  "request_submission", "submit_order", "netting", "hedging", "magic number", "source of truth",
                  "favourable slippage", "no real order"):
        assert token in low, token
    iface = (ROOT / "docs" / "BROKER_NEUTRAL_INTERFACES.md").read_text()
    assert "## Execution port (Phase 1G)" in iface
    assert "PHASE_1G_EXECUTION_SAFETY.md" in (ROOT / "README.md").read_text()


def test_paper_port_is_documented_as_testing_only():
    doc = (PKG / "paper.py").read_text()
    assert "TESTING ONLY" in doc and "NOT REALISM" in doc


def test_execution_does_not_touch_upstream_phases():
    for py in PKG.rglob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 2:
                assert node.module.split(".")[0] in ("risk", "trade", "reason_codes", "config"), (py.name, node.module)
