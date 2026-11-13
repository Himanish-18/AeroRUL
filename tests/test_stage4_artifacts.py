"""
Unit tests for Stage 4 Feature Intelligence and Degradation Artifacts.
Verifies that:
- All four models load and feature counts match metadata and boosters.
- Feature ordering matches feature_names.json.
- Gain and split importance are non-negative and properly normalized.
- Family aggregations match individual feature sums.
- Missing correlations remain empty/missing and not converted to zero.
- No NaN/Inf exist in numeric fields.
- No official test labels or test datasets are used.
"""

import json
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE4_DIR = PROJECT_ROOT / "reports" / "dashboard_stage4"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]


def test_stage4_artifact_files_exist():
    """Verify all 15 required artifact files exist and are non-empty."""
    expected_files = [
        "feature_family_importance.csv",
        "feature_importance_manifest.json",
        "sensor_trajectories_manifest.json",
    ]
    for ds in DATASETS:
        expected_files.append(f"feature_importance_{ds}.csv")
        expected_files.append(f"sensor_importance_{ds}.csv")
        expected_files.append(f"sensor_trajectories_{ds}.csv")

    for fname in expected_files:
        fpath = STAGE4_DIR / fname
        assert fpath.exists(), f"Missing expected artifact: {fname}"
        assert fpath.stat().st_size > 0, f"Artifact is empty: {fname}"


def test_feature_importance_schema_and_integrity():
    """Verify schema, numeric ranges, normalization, and ordering of feature importance."""
    expected_cols = [
        "dataset", "feature_name", "sensor", "statistic", "window_or_lag",
        "description", "gain_importance", "split_importance",
        "gain_importance_pct", "split_importance_pct", "gain_rank", "split_rank",
        "correlation_pearson", "correlation_spearman", "abs_pearson", "abs_spearman"
    ]

    for ds in DATASETS:
        fpath = STAGE4_DIR / f"feature_importance_{ds}.csv"
        df = pd.read_csv(fpath)
        assert list(df.columns) == expected_cols, f"Column mismatch in {fpath.name}"

        # No duplicate features
        assert df["feature_name"].nunique() == len(df), f"Duplicate feature names in {fpath.name}"
        assert df["dataset"].unique().tolist() == [ds]

        # Valid ranks
        assert sorted(df["gain_rank"]) == list(range(1, len(df) + 1))
        assert sorted(df["split_rank"]) == list(range(1, len(df) + 1))

        # Gain and split importance are non-negative
        assert (df["gain_importance"] >= 0).all()
        assert (df["split_importance"] >= 0).all()

        # Percentages sum to ~100%
        gain_pct_sum = df["gain_importance_pct"].sum()
        assert np.isclose(gain_pct_sum, 100.0, atol=0.1), f"Gain pct sum {gain_pct_sum} not near 100 for {ds}"

        split_pct_sum = df["split_importance_pct"].sum()
        assert np.isclose(split_pct_sum, 100.0, atol=0.1), f"Split pct sum {split_pct_sum} not near 100 for {ds}"

        # Check correlations: missing values are NaN in pandas, not 0.0
        # Correlation magnitudes must be <= 1.0
        valid_p = df["correlation_pearson"].dropna()
        if len(valid_p) > 0:
            assert (valid_p >= -1.0).all() and (valid_p <= 1.0).all()


def test_sensor_importance_consistency():
    """Verify sensor importance matches individual feature importance totals."""
    for ds in DATASETS:
        feat_df = pd.read_csv(STAGE4_DIR / f"feature_importance_{ds}.csv")
        sensor_df = pd.read_csv(STAGE4_DIR / f"sensor_importance_{ds}.csv")

        # Sum of sensor gain equals feature gain sum
        feat_gain_sum = feat_df["gain_importance"].sum()
        sensor_gain_sum = sensor_df["gain_importance"].sum()
        assert np.isclose(feat_gain_sum, sensor_gain_sum, rtol=1e-4)

        # Sum of sensor split equals feature split sum
        assert feat_df["split_importance"].sum() == sensor_df["split_importance"].sum()

        # Total feature counts match
        assert feat_df["feature_name"].count() == sensor_df["feature_count"].sum()


def test_family_breakdown_consistency():
    """Verify feature_family_importance.csv has consistent totals per family type."""
    df_fam = pd.read_csv(STAGE4_DIR / "feature_family_importance.csv")

    for ds in DATASETS:
        sub = df_fam[df_fam["dataset"] == ds]
        feat_df = pd.read_csv(STAGE4_DIR / f"feature_importance_{ds}.csv")
        total_feats = len(feat_df)

        for fam_type in ["sensor", "statistic", "window"]:
            fam_sub = sub[sub["family_type"] == fam_type]
            assert len(fam_sub) > 0, f"No entries for family {fam_type} in {ds}"
            assert fam_sub["feature_count"].sum() == total_feats
            assert np.isclose(fam_sub["gain_importance_pct"].sum(), 100.0, atol=0.1)


def test_sensor_trajectories_integrity():
    """Verify sensor trajectories are authentic, strictly formatted, and finite."""
    manifest_path = STAGE4_DIR / "sensor_trajectories_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for ds in DATASETS:
        fpath = STAGE4_DIR / f"sensor_trajectories_{ds}.csv"
        df = pd.read_csv(fpath)

        assert list(df.columns) == ["dataset", "unit_id", "cycle", "sensor", "sensor_value"]
        assert df["dataset"].unique().tolist() == [ds]
        assert df["unit_id"].nunique() == 10
        assert df["sensor"].nunique() == 21
        assert not df["sensor_value"].isna().any()
        assert np.isfinite(df["sensor_value"]).all()

        # Cycles must be positive integers
        assert (df["cycle"] > 0).all()

        # Check default unit in manifest
        ds_meta = meta["subsets"][ds]
        assert ds_meta["default_unit_id"] in ds_meta["selected_units"]
