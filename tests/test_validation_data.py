"""Phase 1H data layer: provenance, raw immutability, normalisation, timezones, quality, gaps, resampling."""

from __future__ import annotations

import os
import stat

import numpy as np
import pandas as pd
import pytest

from gbpjpy_engine.data.model import DataIntegrityError
from gbpjpy_engine.validation import (DataQualityBlocked, DataStore, H4BarDefinition, ProvenanceError, RawDataModified,
                                      compare_h4, quality_gate, require_known_provenance, research_quality_report,
                                      resample_h1_to_h4, verify_aggregation)
from gbpjpy_engine.validation.provenance import FileProvider, ImportSpec, sha256_file
from gbpjpy_engine.validation.quality import QualityPolicy, classify_gaps
from validation_helpers import T0, bars, prov, random_walk_h1, spec


# --------------------------------------------------------------------------- provenance
def test_provenance_requires_every_field_and_valid_vocabularies():
    prov().validate()
    for bad in ({"provider": ""}, {"source_timezone": ""}, {"price_type": "LAST"}, {"timeframe": "H2"},
                {"spread_availability": "PER_BAR", "spread_unit": "UNKNOWN"}, {"provider": "synthetic_gen"}):
        with pytest.raises(ProvenanceError):
            prov(**bad).validate()
    prov(provider="synthetic_gen", is_synthetic=True).validate()
    with pytest.raises(ProvenanceError):
        _ = prov().dataset_id  # no content hash before import


def test_backtests_refuse_unknown_provenance():
    with pytest.raises(ProvenanceError):
        require_known_provenance(random_walk_h1(10))


# --------------------------------------------------------------------------- raw immutability
def test_raw_import_is_byte_exact_read_only_and_verified(tmp_path):
    src = tmp_path / "vendor.csv"
    random_walk_h1(50).drop(columns=["source"]).to_csv(src, index=False)
    before = sha256_file(src)
    store = DataStore(tmp_path / "store")
    p = store.import_raw(src, spec())
    assert sha256_file(src) == before  # the provider file itself is never touched
    raw = store.raw / "real" / p.dataset_id / src.name
    assert sha256_file(raw) == p.source_sha256 == before
    assert not (os.stat(raw).st_mode & stat.S_IWUSR)
    assert store.verify_raw(p.dataset_id) == before
    assert store.import_raw(src, spec()).dataset_id == p.dataset_id  # idempotent
    os.chmod(raw, stat.S_IWUSR | stat.S_IRUSR)
    raw.write_text(raw.read_text().replace("190", "191", 1))
    with pytest.raises(RawDataModified):
        store.verify_raw(p.dataset_id)
    with pytest.raises(RawDataModified):
        store.normalise(p.dataset_id)


def test_layers_are_separate_and_normalised_layer_is_checked(tmp_path):
    src = tmp_path / "v.csv"
    random_walk_h1(40).drop(columns=["source"]).to_csv(src, index=False)
    store = DataStore(tmp_path / "s")
    p = store.import_raw(src, spec())
    df, log = store.normalise(p.dataset_id)
    assert (store.norm / f"{p.dataset_id}.parquet").exists() and "no sorting" in " ".join(log["steps"])
    again = store.load_normalised(p.dataset_id)
    assert again.attrs["provenance"].rows == 40 and require_known_provenance(again).dataset_id == p.dataset_id
    tampered = pd.read_parquet(store.norm / f"{p.dataset_id}.parquet")
    tampered.loc[3, "close"] += 1
    tampered.to_parquet(store.norm / f"{p.dataset_id}.parquet", index=False)
    with pytest.raises(RawDataModified):
        store.load_normalised(p.dataset_id)
    store.save_derived("h4_test", df.head(4), {"parent": p.dataset_id})
    assert (store.derived / "h4_test.meta.json").exists()


def test_synthetic_datasets_live_in_a_separate_namespace(tmp_path):
    src = tmp_path / "syn.csv"
    random_walk_h1(20).drop(columns=["source"]).to_csv(src, index=False)
    store = DataStore(tmp_path / "s")
    store.import_raw(src, spec(provider="synthetic_fixture", is_synthetic=True))
    assert store.list_datasets(real_only=True) == [] and len(store.list_datasets(real_only=False)) == 1


# --------------------------------------------------------------------------- normalisation / timezones
def test_naive_timestamps_use_the_declared_source_timezone_only(tmp_path):
    raw = pd.DataFrame({"time": ["2024-07-01 03:00", "2024-07-01 04:00"], "open": [190.0, 190.1], "high": [190.2] * 2,
                        "low": [189.9] * 2, "close": [190.1, 190.0], "spread": [15, 20]})
    fp = FileProvider(ImportSpec(prov(source_timezone="Europe/Athens", spread_unit="POINTS")))
    out = fp.to_canonical_frame(raw)
    assert str(out["timestamp"].dt.tz) == "UTC" and out["timestamp"].iloc[0] == pd.Timestamp("2024-07-01 00:00", tz="UTC")
    assert list(out["spread"]) == [1.5, 2.0]  # POINTS -> pips
    aware = raw.assign(time=["2024-07-01T03:00:00+00:00", "2024-07-01T04:00:00+00:00"])
    assert fp.to_canonical_frame(aware)["timestamp"].iloc[0] == pd.Timestamp("2024-07-01 03:00", tz="UTC")


def test_ambiguous_dst_time_is_not_silently_resolved():
    raw = pd.DataFrame({"time": ["2024-10-27 03:30"], "open": [190.0], "high": [190.2], "low": [189.9], "close": [190.1]})
    fp = FileProvider(ImportSpec(prov(source_timezone="Europe/Athens", spread_availability="NONE")))
    with pytest.raises(Exception):
        fp.to_canonical_frame(raw)


def test_terminal_tab_export_and_missing_spread(tmp_path):
    f = tmp_path / "export.csv"
    f.write_text("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>\n"
                 "2024.01.08\t00:00:00\t190.000\t190.100\t189.900\t190.050\t500\t0\t18\n"
                 "2024.01.08\t01:00:00\t190.050\t190.150\t189.950\t190.100\t400\t0\t22\n")
    fp = FileProvider(ImportSpec(prov(source_timezone="Etc/GMT-2", spread_unit="POINTS"), fmt="TERMINAL_TAB_EXPORT"))
    out = fp.to_canonical_frame(fp.read_raw(f))
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-07 22:00", tz="UTC")
    assert list(out["volume"]) == [500.0, 400.0] and list(out["spread"]) == [1.8, 2.2]
    fp2 = FileProvider(ImportSpec(prov(source_timezone="UTC", spread_availability="NONE"), fmt="TERMINAL_TAB_EXPORT"))
    assert fp2.to_canonical_frame(fp2.read_raw(f))["spread"].isna().all()  # never invented, never zero


def test_naive_without_timezone_is_rejected():
    from gbpjpy_engine.data.model import to_canonical

    with pytest.raises(DataIntegrityError):
        to_canonical(pd.DataFrame({"timestamp": ["2024-01-01 00:00"], "open": [1], "high": [1], "low": [1], "close": [1]}))


# --------------------------------------------------------------------------- quality
def test_quality_report_detects_defects_and_blocks_corrupt_data():
    df = random_walk_h1(300)
    df.loc[10, "high"] = df.loc[10, "low"] - 0.1  # OHLC inconsistency
    df.loc[20, "timestamp"] = df.loc[19, "timestamp"]  # duplicate
    df.loc[30, "close"] = -1.0  # impossible
    rep = research_quality_report(df, "H1")
    c = rep.counts
    assert rep.status == "BLOCKED" and c.get("INVALID_OHLC") and c.get("DUPLICATE_TIMESTAMP") and c.get("NONPOSITIVE_PRICE")
    with pytest.raises(DataQualityBlocked):
        quality_gate(rep)


def test_quality_warnings_are_reported_not_repaired():
    df = random_walk_h1(400)
    df.loc[100:106, ["open", "high", "low", "close"]] = 191.0  # stale + repeated + flat
    df.loc[200, "spread"] = 45.0
    df.loc[201, "spread"] = 0.0
    df.loc[250, "high"] = df.loc[250, "high"] + 20  # spike (and implausible? no: < 400)
    sat = pd.Timestamp("2024-01-13 10:00", tz="UTC")
    df = pd.concat([df, pd.DataFrame([{**df.iloc[-1].to_dict(), "timestamp": sat + pd.Timedelta(days=21)}])], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    before = df.copy()
    rep = research_quality_report(df, "H1")
    c = rep.counts
    for k in ("STALE_SEQUENCE", "REPEATED_BAR", "FLAT_BAR", "ABNORMAL_SPREAD", "ZERO_SPREAD", "PRICE_SPIKE", "WEEKEND_BAR"):
        assert c.get(k), k
    pd.testing.assert_frame_equal(df, before)
    assert rep.status == "WARNINGS" and quality_gate(rep) is rep


def test_missing_volume_and_spread_flagged():
    df = random_walk_h1(100)
    df["volume"] = np.nan
    df["spread"] = np.nan
    rep = research_quality_report(df, "H1", spread_expected=True)
    assert rep.counts.get("MISSING_VOLUME") and rep.counts.get("MISSING_SPREAD")


def test_gap_classification_and_blackout():
    df = random_walk_h1(500)
    # weekend gaps exist naturally; add a 30h weekday hole and a 2-bar hole and a declared outage
    ts = df["timestamp"]
    drop = list(range(150, 180)) + [300, 301]
    df = df.drop(index=drop).reset_index(drop=True)
    out_start = df["timestamp"].iloc[398] + pd.Timedelta(minutes=1)
    df = df.drop(index=range(399, 404)).reset_index(drop=True)
    pol = QualityPolicy(blackout_bars_after_gap=10)
    gaps = classify_gaps(df, 60, pol, known_outages=[(out_start.isoformat(), (out_start + pd.Timedelta(hours=8)).isoformat())])
    kinds = {g.classification for g in gaps}
    assert {"EXPECTED_MARKET_CLOSURE", "MAJOR_MISSING_INTERVAL", "SHORT_UNEXPLAINED_GAP", "PROVIDER_OUTAGE"} <= kinds
    rep = research_quality_report(df, "H1", pol, known_outages=[(out_start.isoformat(),
                                                                (out_start + pd.Timedelta(hours=8)).isoformat())])
    major = next(g for g in rep.gaps if g.classification == "MAJOR_MISSING_INTERVAL")
    assert major.index in rep.blackout and major.index + 9 in rep.blackout and major.index + 10 not in rep.blackout
    assert len(ts) == 500


def test_dst_irregularity_in_weekly_open_is_reported_not_corrected():
    a = random_walk_h1(120, start=pd.Timestamp("2024-03-04 22:00", tz="UTC"))
    b = random_walk_h1(120, start=pd.Timestamp("2024-04-07 21:00", tz="UTC"))
    rep = research_quality_report(pd.concat([a, b], ignore_index=True), "H1")
    assert len(rep.dst_summary["week_open_hour_utc_distribution"]) > 1 and rep.counts.get("DST_IRREGULARITY")


# --------------------------------------------------------------------------- resampling / H1-H4 consistency
def test_resampling_is_deterministic_versioned_and_exact():
    h1 = random_walk_h1(400)
    a, ra = resample_h1_to_h4(h1)
    b, rb = resample_h1_to_h4(h1.copy())
    pd.testing.assert_frame_equal(a, b)
    assert ra["definition_id"] == rb["definition_id"] and ra["definition_id"].startswith("H4_UTC_GRID_00_v1")
    assert verify_aggregation(h1, a)["exact"]
    assert set(a["timestamp"].dt.hour % 4) == {0}
    off, r2 = resample_h1_to_h4(h1, H4BarDefinition(name="H4_UTC_GRID_01", offset_hours=1))
    assert r2["definition_id"] != ra["definition_id"] and set(off["timestamp"].dt.hour % 4) == {1}


def test_server_clock_grid_follows_declared_dst_and_drops_irregular_buckets():
    h1 = random_walk_h1(24 * 20, start=pd.Timestamp("2024-03-18 00:00", tz="UTC"))
    h4, rep = resample_h1_to_h4(h1, H4BarDefinition(name="H4_SERVER", grid_timezone="Europe/Athens"))
    hours = h4["timestamp"].dt.hour % 4
    assert set(hours) == {1, 2}  # UTC+2 before the 31 March DST change, UTC+3 after
    assert rep["dropped_incomplete_buckets"] >= 1


def test_provider_h4_comparison_reports_discrepancies():
    h1 = random_walk_h1(200)
    derived, _ = resample_h1_to_h4(h1)
    assert compare_h4(derived, derived)["verdict"] == "CONSISTENT"
    prov_h4 = derived.copy()
    prov_h4.loc[5, "high"] += 0.05
    prov_h4 = prov_h4.drop(index=[7])
    rep = compare_h4(prov_h4, derived)
    assert rep["verdict"] == "DISCREPANCIES_REPORTED" and rep["ohlc_mismatches"] == 1 and rep["only_in_derived"] == 1
    shifted = derived.assign(timestamp=derived["timestamp"] + pd.Timedelta(hours=1))
    assert not compare_h4(shifted, derived)["grids_consistent"]


def test_incomplete_h4_buckets_are_dropped_not_filled():
    h1 = bars([(190, 190.1, 189.9, 190.05)] * 7, T0)
    h4, rep = resample_h1_to_h4(h1)
    assert len(h4) == 1 and rep["dropped_incomplete_buckets"] == 1
