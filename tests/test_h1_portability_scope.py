"""Phase 1C: configuration, portability and scope checks (no trading functionality)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine.config import describe_config
from gbpjpy_engine.data import DataIntegrityError
from gbpjpy_engine.h1 import H1Config, H1SetupEngine, load_h1_config

ROOT = Path(gbpjpy_engine.__file__).parents[2]
H1_PKG = Path(gbpjpy_engine.__file__).parent / "h1"


def test_h1_config_documented_and_yaml_matches_defaults():
    rows = describe_config(H1Config())
    assert all(r["doc"] and len(r["doc"]) > 10 for r in rows)
    assert load_h1_config(ROOT / "config" / "h1_default.yaml").to_dict() == H1Config().to_dict()
    assert load_h1_config().config_hash() == H1Config().config_hash()


def test_h1_consumes_canonical_bars_only(h1_engine):
    bad = pd.DataFrame({"time": [1, 2], "o": [1, 2]})
    with pytest.raises(DataIntegrityError):
        h1_engine.run(bad, None)


def test_h1_package_imports_are_core_only():
    import sys

    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    for py in H1_PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                assert {a.name.split(".")[0] for a in node.names} <= allowed, py.name
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed, (py.name, node.module)


def test_no_execution_objects_in_h1_code():
    pat = re.compile(r"\b(entry_price|stop_loss|take_profit|position_size|lot_size|order_send|place_order|"
                     r"submit_order|broker)\b\s*[=(]", re.IGNORECASE)
    for py in H1_PKG.rglob("*.py"):
        assert not pat.search(py.read_text()), py.name
    assert {n for n in dir(H1SetupEngine) if not n.startswith("_")} == {"run"}


def test_h1_harness_detects_leakage(h1_results):
    from test_h1_lookahead import same_rows

    _, _, res = h1_results["h1_range"]
    full = res.frame[["long_setup_score"]].copy()
    full["future"] = full["long_setup_score"].shift(-1)
    part = full.iloc[:1500].copy()
    part["future"] = part["long_setup_score"].shift(-1)
    with pytest.raises(AssertionError, match="H1 look-ahead"):
        same_rows(full, part, 1499)


def test_phase1c_documentation():
    doc = (ROOT / "docs" / "PHASE_1C_H1_SETUP_INTELLIGENCE.md").read_text()
    for token in ("H4 -> H1 alignment", "NOT a trade", "not a probability", "Known limitations", "counterfactual"):
        assert token.lower() in doc.lower(), token
