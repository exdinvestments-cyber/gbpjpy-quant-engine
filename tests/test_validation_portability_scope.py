"""Phase 1H portability and scope: research only, no network, no platform, no write-back, documented import spec."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import gbpjpy_engine
from gbpjpy_engine.cli import main as cli_main
from gbpjpy_engine.validation import HistoricalDataProvider, describe_accepted_formats, real_data_required

PKG = Path(gbpjpy_engine.__file__).parent / "validation"
ROOT = Path(gbpjpy_engine.__file__).parents[2]
FORBIDDEN_NAMES = {"order_send", "ordersend", "place_order", "submit_order", "send_order", "account_info", "positions_get",
                   "symbol_info_tick", "copy_rates", "metatrader5", "mql4", "mql5", "martingale", "grid_step", "api_key",
                   "password", "token", "secret", "login"}


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
    return {n.lower() for n in out}


def test_validation_imports_are_core_only_and_no_network():
    allowed = set(sys.stdlib_module_names) | {"numpy", "pandas", "yaml"}
    banned = {"socket", "http", "urllib", "ssl", "asyncio", "ftplib", "smtplib", "xmlrpc", "requests"}
    for py in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(py.read_text())):
            if isinstance(node, ast.Import):
                mods = {a.name.split(".")[0] for a in node.names}
                assert mods <= allowed and not mods & banned, (py.name, mods)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module.split(".")[0] in allowed and node.module.split(".")[0] not in banned, (py.name, node.module)


def test_no_platform_order_or_credential_names():
    for py in PKG.rglob("*.py"):
        hits = _names(py) & FORBIDDEN_NAMES
        assert not hits, (py.name, hits)


def test_no_auto_tuning_or_write_back_to_production_configuration():
    for py in PKG.rglob("*.py"):
        src = py.read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("setattr", "exec", "eval"):
                raise AssertionError(f"{py.name}: {node.func.id}")
        assert "config/" not in src and "_default.yaml" not in src, py.name  # never writes the production configs


def test_research_layer_is_not_imported_by_the_strategy_risk_or_execution_layers():
    core = Path(gbpjpy_engine.__file__).parent
    for sub in ("context", "h1", "entry", "trade", "risk", "execution", "features", "classification", "data"):
        for py in (core / sub).rglob("*.py"):
            for n in ast.walk(ast.parse(py.read_text())):
                if isinstance(n, ast.ImportFrom):
                    m = n.module or ""
                    research = (n.level >= 2 and m.startswith("validation")) or m.startswith("gbpjpy_engine.validation")
                    assert not research, py  # (data/validation.py is the data-quality module, not the research layer)


def test_provider_interface_is_a_protocol():
    assert getattr(HistoricalDataProvider, "_is_protocol", False)


def test_real_data_required_specification_and_cli(tmp_path, capsys):
    spec = real_data_required(tmp_path)
    assert spec["status"] == "REAL_DATA_REQUIRED" and "no synthetic data has been substituted" in spec["message"]
    fmt = describe_accepted_formats()
    assert set(fmt["formats"]) == {"CSV", "PARQUET", "TERMINAL_TAB_EXPORT"} and "price_type (BID/ASK/MID/UNKNOWN)" in \
        fmt["required_provenance"]
    assert cli_main(["validate", "status", "--store", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "REAL_DATA_REQUIRED"
    assert cli_main(["validate", "run", "--store", str(tmp_path / "s")]) == 2


def test_cli_import_creates_raw_and_normalised_layers(tmp_path, capsys):
    f = tmp_path / "h1.csv"
    f.write_text("timestamp,open,high,low,close,volume,spread\n2024-01-08 00:00,190,190.1,189.9,190.05,10,20\n"
                 "2024-01-08 01:00,190.05,190.2,190.0,190.1,12,18\n")
    rc = cli_main(["validate", "import", "--store", str(tmp_path / "s"), "--file", str(f), "--timeframe", "H1",
                   "--provider", "broker_export", "--source-tz", "Etc/GMT-2", "--price-type", "BID", "--spread", "PER_BAR",
                   "--spread-unit", "POINTS", "--retrieved-at", "2024-06-01T00:00:00Z"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["rows"] == 2 and any("Etc/GMT-2" in s for s in out["transform"]["steps"])
    assert (tmp_path / "s" / "raw" / "real" / out["dataset_id"] / "h1.csv").exists()


def test_phase1h_documentation():
    doc = (ROOT / "docs" / "PHASE_1H_REAL_DATA_VALIDATION.md").read_text()
    low = doc.lower()
    for token in ("real_data_required", "falsify", "provenance", "raw", "normalised", "derived", "utc", "bid", "ask", "mid",
                  "spread", "commission", "slippage", "swap", "conservative", "lower_timeframe_required", "warm-up",
                  "holdout", "walk-forward", "post_hoc", "experiment manifest", "research budget", "hypothesis",
                  "overfitting", "promotion gates", "known limitations", "terminal_tab_export", "mt5", "news_data_unavailable",
                  "500"):
        assert token in low, token
    assert "PHASE_1H_REAL_DATA_VALIDATION.md" in (ROOT / "README.md").read_text()
