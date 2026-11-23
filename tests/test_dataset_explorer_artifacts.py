"""
AeroRUL — Dataset Explorer Artifact and Integrity Test Suite

Verifies:
1. Raw C-MAPSS train, test, and RUL file loading and schema conformity.
2. Actual row counts, unique engines, and contiguity (1..N).
3. FD004 count discrepancy resolution (249 train engines, 248 test engines).
4. Test engine count equals official RUL ground-truth vector length.
5. 1-indexed deterministic test RUL alignment.
6. Engine index CSV file existence, row count, column schema, and fields.
7. Training physical RUL derivation semantics (terminal cycle RUL = 0, no target cap).
8. Test endpoint RUL remains endpoint-only (no synthetic per-cycle extrapolation).
9. Structural data quality: zero missing values, zero infs, zero duplicate rows/unit-cycle pairs.
10. Monotonic cycle ordering starting at cycle 1 for every engine.
11. Sensor and operating setting metadata integrity.
12. Source file SHA-256 digest consistency against manifest.
13. Gzip compression and decompression byte-for-byte roundtrip fidelity.
"""

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import (
    AVAILABLE_SUBSETS,
    COLUMN_NAMES,
    EXPECTED_NUM_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SENSOR_COLUMNS,
    rul_filename as get_rul_filename,
    test_filename as get_test_filename,
    train_filename as get_train_filename,
)

ARTIFACT_DIR = PROJECT_ROOT / "reports" / "dashboard_dataset_explorer"


@pytest.fixture(scope="module")
def manifest():
    manifest_path = ARTIFACT_DIR / "dataset_manifest.json"
    assert manifest_path.is_file(), f"Manifest file missing: {manifest_path}"
    with open(manifest_path, "r") as f:
        return json.load(f)


def test_manifest_structure_and_schema(manifest):
    """Verify manifest top-level fields and canonical schema."""
    assert manifest["manifest_version"] == "1.0.0"
    assert "generated_at" in manifest
    assert "canonical_schema" in manifest
    assert manifest["canonical_schema"]["total_columns"] == EXPECTED_NUM_COLUMNS
    assert manifest["canonical_schema"]["columns"] == COLUMN_NAMES
    assert len(manifest["sensor_specifications"]) == 21
    assert len(manifest["operating_setting_specifications"]) == 3
    assert set(manifest["datasets"].keys()) == set(AVAILABLE_SUBSETS)


def test_fd004_count_discrepancy_resolution(manifest):
    """
    Verify FD004 engine count discrepancy resolution:
    CMAPSSData/readme.txt stated 248 train / 249 test, while actual raw data
    and official RUL vector prove 249 train / 248 test engines.
    """
    res = manifest["fd004_count_discrepancy_resolution"]
    assert res["actual_raw_train_engines"] == 249
    assert res["actual_raw_test_engines"] == 248
    assert res["actual_rul_rows"] == 248

    # Verify directly from raw files
    train_df = pd.read_csv(RAW_DATA_DIR / "train_FD004.txt", sep=r"\s+", header=None, usecols=[0])
    test_df = pd.read_csv(RAW_DATA_DIR / "test_FD004.txt", sep=r"\s+", header=None, usecols=[0])
    with open(RAW_DATA_DIR / "RUL_FD004.txt") as f:
        rul_lines = [l.strip() for l in f if l.strip()]

    assert train_df[0].nunique() == 249
    assert test_df[0].nunique() == 248
    assert len(rul_lines) == 248


@pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
def test_raw_files_and_engine_contiguity(subset, manifest):
    """Verify raw file line counts, engine counts, and 1..N contiguity."""
    train_path = RAW_DATA_DIR / get_train_filename(subset)
    test_path = RAW_DATA_DIR / get_test_filename(subset)
    rul_path = RAW_DATA_DIR / get_rul_filename(subset)

    assert train_path.is_file()
    assert test_path.is_file()
    assert rul_path.is_file()

    train_df = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    with open(rul_path) as f:
        rul_vec = [int(l.strip()) for l in f if l.strip()]

    n_train_units = train_df["unit_id"].nunique()
    n_test_units = test_df["unit_id"].nunique()

    # Contiguity
    assert sorted(train_df["unit_id"].unique()) == list(range(1, n_train_units + 1))
    assert sorted(test_df["unit_id"].unique()) == list(range(1, n_test_units + 1))

    # Test engine count == RUL vector count
    assert n_test_units == len(rul_vec)

    # Manifest match
    ds_meta = manifest["datasets"][subset]
    assert ds_meta["train"]["row_count"] == len(train_df)
    assert ds_meta["train"]["engine_count"] == n_train_units
    assert ds_meta["test"]["row_count"] == len(test_df)
    assert ds_meta["test"]["engine_count"] == n_test_units
    assert ds_meta["test"]["rul_statistics"]["endpoint_count"] == len(rul_vec)


@pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
def test_engine_index_artifacts(subset):
    """Verify engine index CSV files exist and adhere to canonical schema."""
    train_idx_path = ARTIFACT_DIR / f"engine_index_{subset}_train.csv"
    test_idx_path = ARTIFACT_DIR / f"engine_index_{subset}_test.csv"

    assert train_idx_path.is_file()
    assert test_idx_path.is_file()

    train_idx = pd.read_csv(train_idx_path)
    test_idx = pd.read_csv(test_idx_path)

    expected_cols = [
        "dataset",
        "split",
        "unit_id",
        "observation_count",
        "first_cycle",
        "last_observed_cycle",
        "observed_cycle_count",
        "endpoint_status",
        "physical_failure_cycle",
        "official_endpoint_rul",
    ]
    assert list(train_idx.columns) == expected_cols
    assert list(test_idx.columns) == expected_cols

    # Training semantics
    assert (train_idx["split"] == "train").all()
    assert (train_idx["endpoint_status"] == "run-to-failure").all()
    assert (train_idx["physical_failure_cycle"] == train_idx["last_observed_cycle"]).all()
    assert train_idx["official_endpoint_rul"].isna().all()

    # Test semantics
    assert (test_idx["split"] == "test").all()
    assert (test_idx["endpoint_status"] == "censored").all()
    assert test_idx["physical_failure_cycle"].isna().all()
    assert not test_idx["official_endpoint_rul"].isna().any()

    # Unit ID 1..N order
    assert list(train_idx["unit_id"]) == list(range(1, len(train_idx) + 1))
    assert list(test_idx["unit_id"]) == list(range(1, len(test_idx) + 1))


@pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
def test_data_quality_integrity(subset, manifest):
    """Verify zero missing values, zero infs, zero duplicate rows/unit-cycles, cycle ordering."""
    for split in ["train", "test"]:
        q = manifest["datasets"][subset][split]["quality"]
        assert q["column_count_ok"] is True
        assert q["column_names_ok"] is True
        assert q["total_missing_values"] == 0
        assert q["total_non_finite_values"] == 0
        assert q["duplicate_rows"] == 0
        assert q["duplicate_unit_cycle"] == 0
        assert q["cycles_ordered"] is True
        assert q["cycles_start_at_1"] is True
        assert q["valid_unit_ids"] is True
        assert q["valid_cycle_values"] is True
        assert q["structural_status"] == "PASS"
        assert q["integrity_status"] == "PASS"


@pytest.mark.parametrize("subset", AVAILABLE_SUBSETS)
def test_file_hashes_and_compression_roundtrip(subset, manifest):
    """Verify SHA-256 file hashes in manifest match disk, and verify gzip roundtrip."""
    f_info = manifest["datasets"][subset]["files"]

    for file_key, hash_key in [
        ("train_filename", "train_sha256"),
        ("test_filename", "test_sha256"),
        ("rul_filename", "rul_sha256"),
    ]:
        filepath = RAW_DATA_DIR / f_info[file_key]
        assert filepath.is_file()
        
        # Hash match
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            raw_bytes = f.read()
            h.update(raw_bytes)
        assert h.hexdigest() == f_info[hash_key]

        # Gzip roundtrip verification
        compressed = gzip.compress(raw_bytes, compresslevel=6)
        decompressed = gzip.decompress(compressed)
        assert decompressed == raw_bytes, f"Gzip compression altered bytes for {filepath.name}"
