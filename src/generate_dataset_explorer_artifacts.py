"""
AeroRUL — Dataset Explorer Artifact Generator

Generates validated metadata, engine indexes, and dataset manifests directly from
the raw NASA C-MAPSS source files in CMAPSSData/.

Outputs saved to reports/dashboard_dataset_explorer/:
  - dataset_manifest.json
  - engine_index_FD001_train.csv
  - engine_index_FD001_test.csv
  - engine_index_FD002_train.csv
  - engine_index_FD002_test.csv
  - engine_index_FD003_train.csv
  - engine_index_FD003_test.csv
  - engine_index_FD004_train.csv
  - engine_index_FD004_test.csv
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_DIR))

import numpy as np
import pandas as pd

from src.config import (
    AVAILABLE_SUBSETS,
    COLUMN_NAMES,
    EXPECTED_NUM_COLUMNS,
    NEAR_CONSTANT_STD_THRESHOLD,
    OP_SETTING_COLUMNS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SENSOR_COLUMNS,
    SUBSET_METADATA,
    rul_filename,
    test_filename,
    train_filename,
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "dashboard_dataset_explorer"

SENSOR_SPECIFICATIONS = {
    "sensor_1": {"name": "Total temp at fan inlet", "unit": "°R", "type": "Temperature"},
    "sensor_2": {"name": "Total temp at LPC outlet", "unit": "°R", "type": "Temperature"},
    "sensor_3": {"name": "Total temp at HPC outlet", "unit": "°R", "type": "Temperature"},
    "sensor_4": {"name": "Total temp at LPT outlet", "unit": "°R", "type": "Temperature"},
    "sensor_5": {"name": "Pressure at fan inlet", "unit": "psia", "type": "Pressure"},
    "sensor_6": {"name": "Total pressure in bypass duct", "unit": "psia", "type": "Pressure"},
    "sensor_7": {"name": "Total pressure at HPC outlet", "unit": "psia", "type": "Pressure"},
    "sensor_8": {"name": "Physical fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_9": {"name": "Physical core speed", "unit": "rpm", "type": "Speed"},
    "sensor_10": {"name": "Engine pressure ratio (P50/P2)", "unit": "—", "type": "Ratio"},
    "sensor_11": {"name": "Static pressure at HPC outlet", "unit": "psia", "type": "Pressure"},
    "sensor_12": {"name": "Ratio of fuel flow to Ps30", "unit": "pps/psia", "type": "Ratio"},
    "sensor_13": {"name": "Corrected fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_14": {"name": "Corrected core speed", "unit": "rpm", "type": "Speed"},
    "sensor_15": {"name": "Bypass ratio", "unit": "—", "type": "Ratio"},
    "sensor_16": {"name": "Burner fuel-air ratio", "unit": "—", "type": "Ratio"},
    "sensor_17": {"name": "Bleed enthalpy", "unit": "—", "type": "Enthalpy"},
    "sensor_18": {"name": "Demanded fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_19": {"name": "Demanded corrected fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_20": {"name": "HPT coolant bleed", "unit": "lbm/s", "type": "Flow"},
    "sensor_21": {"name": "LPT coolant bleed", "unit": "lbm/s", "type": "Flow"},
}

OP_SETTING_SPECIFICATIONS = {
    "op_setting_1": {"name": "Altitude / Setting 1", "unit": "kft", "description": "Flight altitude (0 to ~42 kft)"},
    "op_setting_2": {"name": "Mach Number / Setting 2", "unit": "Mach", "description": "Flight Mach number (0 to ~0.84)"},
    "op_setting_3": {"name": "TRA / Setting 3", "unit": "TRA", "description": "Throttle resolver angle / sea level temp modifier (60 to 100)"},
}


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_raw_table(filepath: Path) -> pd.DataFrame:
    """Load space-separated raw file with whitespace handling."""
    return pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)


def load_rul_vector(filepath: Path) -> List[int]:
    """Load official RUL ground-truth vector."""
    with open(filepath, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def generate_engine_index(
    subset: str,
    split: str,
    df: pd.DataFrame,
    rul_vector: List[int] = None,
) -> pd.DataFrame:
    """
    Generate an engine index DataFrame for a subset split.
    Training: physical_failure_cycle is max cycle, official_endpoint_rul is None.
    Test: physical_failure_cycle is None, official_endpoint_rul is the corresponding RUL value.
    """
    rows = []
    engine_groups = df.groupby("unit_id")
    unique_units = sorted(df["unit_id"].unique())

    for unit in unique_units:
        engine_data = engine_groups.get_group(unit)
        obs_count = len(engine_data)
        first_cycle = int(engine_data["cycle"].min())
        last_cycle = int(engine_data["cycle"].max())

        if split == "train":
            endpoint_status = "run-to-failure"
            phys_fail = last_cycle
            endpoint_rul = None
        else:
            endpoint_status = "censored"
            phys_fail = None
            # 1-indexed mapping
            endpoint_rul = rul_vector[unit - 1] if rul_vector and (unit - 1) < len(rul_vector) else None

        rows.append({
            "dataset": subset,
            "split": split,
            "unit_id": int(unit),
            "observation_count": obs_count,
            "first_cycle": first_cycle,
            "last_observed_cycle": last_cycle,
            "observed_cycle_count": obs_count,
            "endpoint_status": endpoint_status,
            "physical_failure_cycle": phys_fail,
            "official_endpoint_rul": endpoint_rul,
        })

    index_df = pd.DataFrame(rows)
    return index_df


def analyze_split_quality(df: pd.DataFrame, split: str) -> Dict[str, Any]:
    """Perform comprehensive data quality and validation checks on a split DataFrame."""
    n_rows, n_cols = df.shape
    col_names = list(df.columns)

    col_count_ok = n_cols == EXPECTED_NUM_COLUMNS
    col_names_ok = col_names == COLUMN_NAMES

    # Missing & non-finite
    missing_by_col = df.isna().sum().to_dict()
    total_missing = int(df.isna().sum().sum())
    
    # Non-finite (inf)
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = int(np.isinf(numeric_df).sum().sum())

    # Duplicate rows
    n_dup_rows = int(df.duplicated().sum())

    # Duplicate (unit_id, cycle)
    n_dup_uc = int(df.duplicated(subset=["unit_id", "cycle"]).sum())

    # Cycle monotonicity & start at 1
    def _is_monotonic(g):
        return g["cycle"].is_monotonic_increasing

    cycles_ordered = bool(df.groupby("unit_id").apply(_is_monotonic, include_groups=False).all())
    first_cycles = df.groupby("unit_id")["cycle"].min()
    cycles_start_at_1 = bool((first_cycles == 1).all())

    # Valid unit ids: positive integers 1..N contiguous
    units = sorted(df["unit_id"].unique())
    n_units = len(units)
    valid_unit_ids = bool(units == list(range(1, n_units + 1)))

    # Valid cycle values: positive integers >= 1
    valid_cycles = bool((df["cycle"] >= 1).all() and (df["cycle"] % 1 == 0).all())

    # Status assessment
    structural_status = "PASS" if (col_count_ok and col_names_ok and n_dup_rows == 0 and n_dup_uc == 0) else "FAIL"
    integrity_status = "PASS" if (total_missing == 0 and inf_count == 0 and cycles_ordered and cycles_start_at_1 and valid_unit_ids and valid_cycles) else "FAIL"

    return {
        "column_count_ok": col_count_ok,
        "column_names_ok": col_names_ok,
        "total_missing_values": total_missing,
        "total_non_finite_values": inf_count,
        "duplicate_rows": n_dup_rows,
        "duplicate_unit_cycle": n_dup_uc,
        "cycles_ordered": cycles_ordered,
        "cycles_start_at_1": cycles_start_at_1,
        "valid_unit_ids": valid_unit_ids,
        "valid_cycle_values": valid_cycles,
        "structural_status": structural_status,
        "integrity_status": integrity_status,
    }


def compute_split_summary(df: pd.DataFrame, split: str, rul_vector: List[int] = None) -> Dict[str, Any]:
    """Compute statistics, trajectory lengths, sensors and op settings for a split."""
    n_rows = len(df)
    unique_units = df["unit_id"].nunique()
    traj_lengths = df.groupby("unit_id")["cycle"].count()

    quality = analyze_split_quality(df, split)

    # Trajectory statistics
    traj_stats = {
        "min": int(traj_lengths.min()),
        "max": int(traj_lengths.max()),
        "mean": round(float(traj_lengths.mean()), 2),
        "median": float(traj_lengths.median()),
        "p25": float(traj_lengths.quantile(0.25)),
        "p75": float(traj_lengths.quantile(0.75)),
        "std": round(float(traj_lengths.std()), 2),
    }

    # Op setting summaries
    op_summaries = {}
    for col in OP_SETTING_COLUMNS:
        s = df[col]
        op_summaries[col] = {
            "name": OP_SETTING_SPECIFICATIONS[col]["name"],
            "unit": OP_SETTING_SPECIFICATIONS[col]["unit"],
            "description": OP_SETTING_SPECIFICATIONS[col]["description"],
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "n_unique": int(s.nunique()),
        }

    # Sensor summaries
    sensor_summaries = {}
    for col in SENSOR_COLUMNS:
        s = df[col]
        col_mean = float(s.mean())
        col_std = float(s.std())
        if col_mean != 0:
            is_near_constant = bool(abs(col_std / col_mean) < NEAR_CONSTANT_STD_THRESHOLD)
        else:
            is_near_constant = bool(col_std < 1e-10)

        sensor_summaries[col] = {
            "name": SENSOR_SPECIFICATIONS[col]["name"],
            "unit": SENSOR_SPECIFICATIONS[col]["unit"],
            "type": SENSOR_SPECIFICATIONS[col]["type"],
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "mean": round(col_mean, 4),
            "std": round(col_std, 4),
            "n_unique": int(s.nunique()),
            "is_constant": bool(s.nunique() <= 1),
            "is_near_constant": is_near_constant,
            "missing_count": int(s.isna().sum()),
        }

    # RUL metrics
    rul_metrics = {}
    if split == "train":
        # Derived physical training RUL: failure_cycle - cycle
        # Calculate max cycle per unit
        max_cycles = df.groupby("unit_id")["cycle"].transform("max")
        derived_rul = max_cycles - df["cycle"]
        rul_metrics = {
            "type": "derived_physical_training_rul",
            "description": "Per-cycle remaining cycles until terminal failure (max_cycle - cycle) across all observations.",
            "observation_count": len(derived_rul),
            "min": int(derived_rul.min()),
            "max": int(derived_rul.max()),
            "mean": round(float(derived_rul.mean()), 2),
            "median": float(derived_rul.median()),
            "std": round(float(derived_rul.std()), 2),
            "p25": float(derived_rul.quantile(0.25)),
            "p75": float(derived_rul.quantile(0.75)),
        }
    else:
        # Official test endpoint RUL
        if rul_vector:
            s_rul = pd.Series(rul_vector)
            rul_metrics = {
                "type": "official_test_endpoint_rul",
                "description": "Single official ground-truth Remaining Useful Life value corresponding to the final observed cycle of each test engine.",
                "endpoint_count": len(s_rul),
                "min": int(s_rul.min()),
                "max": int(s_rul.max()),
                "mean": round(float(s_rul.mean()), 2),
                "median": float(s_rul.median()),
                "std": round(float(s_rul.std()), 2),
                "p25": float(s_rul.quantile(0.25)),
                "p75": float(s_rul.quantile(0.75)),
            }

    return {
        "row_count": n_rows,
        "engine_count": unique_units,
        "column_count": len(df.columns),
        "trajectory_statistics": traj_stats,
        "operating_settings": op_summaries,
        "sensors": sensor_summaries,
        "quality": quality,
        "rul_statistics": rul_metrics,
    }


def main():
    print("=" * 70)
    print("AeroRUL — Dataset Explorer Artifact Generation")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "canonical_schema": {
            "total_columns": EXPECTED_NUM_COLUMNS,
            "columns": COLUMN_NAMES,
            "id_columns": ["unit_id", "cycle"],
            "op_setting_columns": OP_SETTING_COLUMNS,
            "sensor_columns": SENSOR_COLUMNS,
        },
        "sensor_specifications": SENSOR_SPECIFICATIONS,
        "operating_setting_specifications": OP_SETTING_SPECIFICATIONS,
        "datasets": {},
        "fd004_count_discrepancy_resolution": {
            "readme_claimed_train_engines": 248,
            "readme_claimed_test_engines": 249,
            "actual_raw_train_engines": 249,
            "actual_raw_test_engines": 248,
            "actual_rul_rows": 248,
            "resolution": "CMAPSSData/readme.txt inverted the train and test counts for FD004. Actual raw file analysis confirms 249 training engines (unit_id 1..249) and 248 test engines (unit_id 1..248), matching the 248 rows in RUL_FD004.txt.",
        },
    }

    for subset in AVAILABLE_SUBSETS:
        print(f"\nProcessing {subset}...")
        meta = SUBSET_METADATA[subset]
        train_path = RAW_DATA_DIR / train_filename(subset)
        test_path = RAW_DATA_DIR / test_filename(subset)
        rul_path = RAW_DATA_DIR / rul_filename(subset)

        # File digests
        train_hash = compute_sha256(train_path)
        test_hash = compute_sha256(test_path)
        rul_hash = compute_sha256(rul_path)

        # Load raw data
        train_df = load_raw_table(train_path)
        test_df = load_raw_table(test_path)
        rul_vec = load_rul_vector(rul_path)

        # Validate engine contiguity
        train_units = sorted(train_df["unit_id"].unique())
        test_units = sorted(test_df["unit_id"].unique())
        assert train_units == list(range(1, len(train_units) + 1)), f"{subset} train units non-contiguous"
        assert test_units == list(range(1, len(test_units) + 1)), f"{subset} test units non-contiguous"
        assert len(test_units) == len(rul_vec), f"{subset} test engine count != RUL count"

        # Generate engine indexes
        train_index = generate_engine_index(subset, "train", train_df)
        test_index = generate_engine_index(subset, "test", test_df, rul_vec)

        # Save engine index CSVs
        train_index_path = OUTPUT_DIR / f"engine_index_{subset}_train.csv"
        test_index_path = OUTPUT_DIR / f"engine_index_{subset}_test.csv"
        train_index.to_csv(train_index_path, index=False)
        test_index.to_csv(test_index_path, index=False)
        print(f"  Saved {train_index_path.name} ({len(train_index)} engines)")
        print(f"  Saved {test_index_path.name} ({len(test_index)} engines)")

        # Summaries
        train_summary = compute_split_summary(train_df, "train")
        test_summary = compute_split_summary(test_df, "test", rul_vec)

        manifest["datasets"][subset] = {
            "dataset": subset,
            "conditions": meta["conditions"],
            "fault_modes": meta["fault_modes"],
            "description": meta["description"],
            "files": {
                "train_filename": train_filename(subset),
                "train_sha256": train_hash,
                "train_size_bytes": train_path.stat().st_size,
                "test_filename": test_filename(subset),
                "test_sha256": test_hash,
                "test_size_bytes": test_path.stat().st_size,
                "rul_filename": rul_filename(subset),
                "rul_sha256": rul_hash,
                "rul_size_bytes": rul_path.stat().st_size,
            },
            "train": train_summary,
            "test": test_summary,
        }

    # Save manifest
    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest to {manifest_path} ({manifest_path.stat().st_size} bytes)")
    print("=" * 70)
    print("Generation complete!")


if __name__ == "__main__":
    main()
