"""Platform isolation (MT4/MT5), canonical data and volume/spread handling."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine.data import BarSource, DataIntegrityError, assert_canonical

ALLOWED_THIRD_PARTY = {"numpy", "pandas", "yaml"}
FORBIDDEN_FRAGMENTS = ("metatrader", "mt4", "mt5", "zmq", "win32", "pywintypes", "ctypes", "socket", "requests",
                       "urllib", "http", "websocket", "grpc", "oanda", "ccxt", "fix")


def _imports(py: Path) -> set[str]:
    tree = ast.parse(py.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return mods


def test_core_imports_only_stdlib_and_numeric_stack():
    pkg = Path(gbpjpy_engine.__file__).parent
    stdlib = set(sys.stdlib_module_names)
    for py in pkg.rglob("*.py"):
        for mod in _imports(py):
            assert mod in stdlib or mod in ALLOWED_THIRD_PARTY, f"{py.name} imports non-portable module {mod!r}"
            assert not any(mod.lower().startswith(f) for f in FORBIDDEN_FRAGMENTS), f"{py.name}: {mod}"


def test_declared_dependencies_are_platform_neutral():
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text().lower()
    for frag in ("metatrader", "mt5", "mt4", "zmq", "pywin32"):
        assert frag not in text


def test_bar_source_is_protocol_only():
    class Replay:
        def closed_bars(self, as_of):
            return pd.DataFrame()

    assert isinstance(Replay(), BarSource)
    import gbpjpy_engine.data.interfaces as itf

    assert not [n for n in dir(itf) if "adapter" in n.lower() or "mt" == n.lower()[:2]]


def test_canonical_timestamps_are_timezone_aware(rw_result, rw_bars):
    assert str(rw_result.features["timestamp"].dt.tz) == "UTC"
    assert str(rw_result.features["available_at"].dt.tz) == "UTC"
    naive = rw_bars.copy()
    naive["timestamp"] = naive["timestamp"].dt.tz_localize(None)
    with pytest.raises(DataIntegrityError):
        assert_canonical(naive)


def test_volume_and_spread_preserved_but_not_consumed(engine, rw_bars, rw_result):
    alt = rw_bars.copy()
    rng = np.random.default_rng(0)
    alt["volume"] = rng.integers(1, 100000, len(alt)).astype(float)
    alt["spread"] = rng.uniform(0.5, 25.0, len(alt))
    res = engine.run(alt)
    # carried through untouched for later execution / market-quality modules
    np.testing.assert_array_equal(res.features["volume"].to_numpy(), alt["volume"].to_numpy())
    np.testing.assert_array_equal(res.features["spread"].to_numpy(), alt["spread"].to_numpy())
    snap = res.snapshot_at(700)
    assert snap.market_data["volume"] == alt["volume"].iloc[700]
    assert snap.market_data["spread"] == pytest.approx(alt["spread"].iloc[700])
    # ... and no strategy feature depends on them
    skip = {"volume", "spread", "data_quality_flags", "data_quality_status", "reason_codes"}
    for col in rw_result.features.columns:
        if col in skip:
            continue
        a, b = rw_result.features[col], res.features[col]
        assert [repr(x) for x in a] == [repr(x) for x in b], col
