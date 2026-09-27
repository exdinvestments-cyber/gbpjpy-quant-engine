from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import gbpjpy_engine
from gbpjpy_engine import H4MarketIntelligenceEngine, describe_config, load_config
from gbpjpy_engine.cli import main as cli_main
from gbpjpy_engine.config import H4Config, config_from_dict
from gbpjpy_engine.logging_utils import read_evaluation_log
from gbpjpy_engine.reason_codes import REASON_DESCRIPTIONS, ReasonCode
from gbpjpy_engine.research import explain, inspect

SPEC_FIELDS = [
    "timestamp", "symbol", "close", "market_structure", "structure_quality_score", "trend_state", "trend_strength",
    "adx", "plus_di", "minus_di", "atr", "atr_percentile", "volatility_regime", "volatility_shock", "shock_severity",
    "momentum_score", "momentum_state", "chop_score", "market_quality", "directional_efficiency", "nearest_support",
    "nearest_resistance", "support_strength", "resistance_strength", "distance_to_support_atr",
    "distance_to_resistance_atr", "range_location_short", "range_location_medium", "range_location_long",
    "extension_score", "extension_state", "session_context", "regime", "bullish_evidence_score",
    "bearish_evidence_score", "h4_bias", "bias_confidence", "data_quality_status", "reason_codes",
]


def test_snapshot_contains_spec_fields_and_is_json(rw_result):
    snap = rw_result.latest()
    d = snap.to_dict()
    for k in SPEC_FIELDS:
        assert k in d, k
    json.loads(snap.to_json())
    assert d["h4_bias"] in ("LONG", "SHORT", "NEUTRAL")
    assert 0 <= d["bullish_evidence_score"] <= 100 and 0 <= d["bearish_evidence_score"] <= 100
    assert d["available_at"] > d["timestamp"]
    assert set(d["reason_codes"]) <= {c.value for c in ReasonCode}


def test_every_bar_has_reason_codes(rw_result):
    assert all(len(c) > 0 for c in rw_result.features["reason_codes"])
    valid = {c.value for c in ReasonCode}
    assert all(set(c) <= valid for c in rw_result.features["reason_codes"])


def test_scores_within_bounds(rw_result):
    f = rw_result.features.iloc[300:]
    for col in ("structure_quality_score", "trend_strength", "chop_score", "shock_severity", "extension_score",
                "bias_confidence", "atr_percentile", "directional_efficiency_percentile"):
        s = f[col].dropna()
        assert ((s >= 0) & (s <= 100)).all(), col
    assert f["directional_efficiency"].between(0, 1).all()


def test_warmup_outputs(rw_result):
    f = rw_result.features
    w = rw_result.config.engine.warmup_bars
    assert (f["regime"].iloc[:w] == "UNCLEAR").all()
    assert (f["h4_bias"].iloc[:w] == "NEUTRAL").all()
    assert all("INSUFFICIENT_HISTORY" in c for c in f["reason_codes"].iloc[:w])


def test_neutral_is_common_on_random_walk(rw_result):
    # the engine must be comfortable returning NEUTRAL frequently
    f = rw_result.features.iloc[rw_result.config.engine.warmup_bars :]
    assert (f["h4_bias"] == "NEUTRAL").mean() > 0.4


def test_determinism(engine, rw_bars, rw_result):
    again = engine.run(rw_bars)
    pd.testing.assert_frame_equal(again.features.drop(columns=["reason_codes", "data_quality_flags"]),
                                  rw_result.features.drop(columns=["reason_codes", "data_quality_flags"]))
    assert list(again.features["reason_codes"]) == list(rw_result.features["reason_codes"])


def test_structured_log_and_exports(tmp_path, rw_result):
    log = tmp_path / "eval.jsonl"
    n = rw_result.write_evaluation_log(log)
    recs = read_evaluation_log(log)
    assert n == len(recs) == len(rw_result.features)
    r = recs[-1]
    for k in ("timestamp", "input_data_status", "features", "regime", "bias", "confidence", "reason_codes",
              "warnings", "config_hash", "engine_version"):
        assert k in r
    assert r["features"]["atr"] is not None
    csv = rw_result.export_features(tmp_path / "f.csv")
    assert len(pd.read_csv(csv)) == len(rw_result.features)
    pq = rw_result.export_features(tmp_path / "f.parquet")
    assert len(pd.read_parquet(pq)) == len(rw_result.features)


def test_research_view(rw_result):
    ts = rw_result.features["timestamp"].iloc[600]
    text = inspect(rw_result, ts)
    for token in ("REGIME", "H4 BIAS", "STRUCTURE", "TREND", "VOLATILITY", "MOMENTUM", "CHOP", "LEVELS", "REASON CODES"):
        assert token in text
    assert "NOT an entry signal" in text
    assert explain(rw_result.latest())
    with pytest.raises(KeyError):
        rw_result.snapshot(pd.Timestamp("1999-01-01", tz="UTC"))


def test_every_parameter_documented():
    rows = describe_config()
    assert len(rows) > 80
    for r in rows:
        assert r["doc"] and len(r["doc"]) > 10, f"{r['section']}.{r['name']} lacks documentation"


def test_yaml_config_matches_defaults():
    path = Path(__file__).resolve().parents[1] / "config" / "h4_default.yaml"
    cfg = load_config(path)
    assert cfg.to_dict() == H4Config().to_dict()


def test_config_overrides_and_validation():
    cfg = config_from_dict({"swing": {"right_bars": 5}, "trend": {"ema_fast": 21}})
    assert cfg.swing.right_bars == 5 and cfg.trend.ema_fast == 21
    assert cfg.config_hash() != H4Config().config_hash()
    with pytest.raises(KeyError):
        config_from_dict({"swing": {"nonexistent": 1}})
    with pytest.raises(ValueError):
        config_from_dict({"trend": {"ema_fast": 60, "ema_mid": 50}})


def test_reason_codes_all_described():
    for c in ReasonCode:
        assert c.value in REASON_DESCRIPTIONS


def test_cli_smoke(tmp_path, capsys):
    out = tmp_path / "up.csv"
    assert cli_main(["synthetic", "--scenario", "clean_uptrend", "--out", str(out)]) == 0
    assert cli_main(["inspect", "--csv", str(out)]) == 0
    assert "REGIME" in capsys.readouterr().out
    assert cli_main(["run", "--csv", str(out), "--out", str(tmp_path / "f.csv"), "--log", str(tmp_path / "l.jsonl")]) == 0
    assert (tmp_path / "l.jsonl").exists()


# ---------------------------------------------------------------------------
# Phase 1A safety: no trading functionality
# ---------------------------------------------------------------------------

FORBIDDEN = [
    r"order_send", r"place_order", r"submit_order", r"market_order", r"limit_order", r"position_size",
    r"lot_size", r"\blots?\b\s*=", r"leverage\s*=", r"stop_loss", r"take_profit", r"\bbroker\s*\.",
    r"import\s+MetaTrader5", r"oandapyV20", r"ccxt", r"requests\.", r"urllib\.request", r"socket\.",
    r"api_key", r"password",
]


def test_no_trading_or_network_functionality_in_package():
    pkg = Path(gbpjpy_engine.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        text = py.read_text()
        for pat in FORBIDDEN:
            if re.search(pat, text, flags=re.IGNORECASE):
                offenders.append((py.name, pat))
    assert not offenders, offenders


def test_engine_public_surface_is_descriptive_only():
    public = {n for n in dir(H4MarketIntelligenceEngine) if not n.startswith("_")}
    assert public == {"run"}
