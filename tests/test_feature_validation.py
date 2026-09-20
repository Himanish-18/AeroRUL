"""
Tests for src.feature_validation — Feature quality and integrity checks.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_validation import (
    data_retention_report,
    feature_target_correlations,
    validate_engine_separation,
    validate_feature_quality,
    validate_target_integrity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_feature_df():
    """Clean feature dataset with no issues."""
    return pd.DataFrame({
        "unit_id": [1, 1, 1, 2, 2, 2],
        "cycle": [10, 11, 12, 10, 11, 12],
        "rul": [40, 39, 38, 20, 19, 18],
        "sensor_1_mean_10": [100.0, 101.0, 102.0, 200.0, 201.0, 202.0],
        "sensor_1_slope_10": [1.0, 1.1, 1.2, 2.0, 2.1, 2.2],
        "sensor_1_delta_1": [1.0, 1.5, 0.8, 2.0, 1.3, 0.9],
    })


@pytest.fixture
def nan_feature_df():
    """Feature dataset with NaN values."""
    return pd.DataFrame({
        "unit_id": [1, 1],
        "cycle": [10, 11],
        "rul": [5, 4],
        "sensor_1_mean_10": [100.0, np.nan],
        "sensor_1_slope_10": [np.nan, 1.0],
    })


@pytest.fixture
def inf_feature_df():
    """Feature dataset with Inf values."""
    return pd.DataFrame({
        "unit_id": [1, 1],
        "cycle": [10, 11],
        "rul": [5, 4],
        "sensor_1_mean_10": [100.0, np.inf],
        "sensor_1_relchange_5": [-np.inf, 1.0],
    })


# ---------------------------------------------------------------------------
# Feature quality tests
# ---------------------------------------------------------------------------

class TestFeatureQuality:

    def test_clean_data(self, clean_feature_df):
        result = validate_feature_quality(clean_feature_df)
        assert result["all_clean"] is True
        assert result["total_nan"] == 0
        assert result["total_inf"] == 0
        assert result["n_constant_features"] == 0

    def test_detects_nan(self, nan_feature_df):
        result = validate_feature_quality(nan_feature_df)
        assert result["total_nan"] == 2
        assert result["all_clean"] is False

    def test_detects_inf(self, inf_feature_df):
        result = validate_feature_quality(inf_feature_df)
        assert result["total_inf"] == 2
        assert result["all_clean"] is False

    def test_detects_constant_features(self):
        df = pd.DataFrame({
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "rul": [2, 1, 0],
            "sensor_1_mean_5": [42.0, 42.0, 42.0],  # constant
            "sensor_2_mean_5": [1.0, 2.0, 3.0],  # varying
        })
        result = validate_feature_quality(df)
        assert result["n_constant_features"] == 1
        assert "sensor_1_mean_5" in result["constant_features"]


# ---------------------------------------------------------------------------
# Target integrity tests
# ---------------------------------------------------------------------------

class TestTargetIntegrity:

    def test_valid_target(self, clean_feature_df):
        result = validate_target_integrity(clean_feature_df)
        assert result["all_valid"] is True
        assert result["rul_non_negative"] is True
        assert result["no_failure_cycle_leakage"] is True

    def test_detects_negative_rul(self):
        df = pd.DataFrame({
            "unit_id": [1],
            "cycle": [10],
            "rul": [-5],
            "sensor_1_mean_10": [100.0],
        })
        result = validate_target_integrity(df)
        assert result["rul_non_negative"] is False
        assert result["all_valid"] is False

    def test_detects_failure_cycle_leakage(self):
        df = pd.DataFrame({
            "unit_id": [1],
            "cycle": [10],
            "rul": [5],
            "failure_cycle": [15],  # LEAKAGE!
            "sensor_1_mean_10": [100.0],
        })
        result = validate_target_integrity(df)
        assert result["no_failure_cycle_leakage"] is False

    def test_matches_original_rul(self, clean_feature_df):
        original = pd.DataFrame({
            "unit_id": [1, 1, 1, 2, 2, 2],
            "cycle": [10, 11, 12, 10, 11, 12],
            "rul": [40, 39, 38, 20, 19, 18],
        })
        result = validate_target_integrity(clean_feature_df, original)
        assert result["rul_matches_original"] is True


# ---------------------------------------------------------------------------
# Engine separation tests
# ---------------------------------------------------------------------------

class TestEngineSeparation:

    def test_no_overlap(self):
        train = pd.DataFrame({"unit_id": [1, 1, 2, 2]})
        val = pd.DataFrame({"unit_id": [3, 3, 4, 4]})
        result = validate_engine_separation(train, val)
        assert result["no_overlap"] is True
        assert result["n_train_engines"] == 2
        assert result["n_val_engines"] == 2

    def test_detects_overlap(self):
        train = pd.DataFrame({"unit_id": [1, 1, 2, 2]})
        val = pd.DataFrame({"unit_id": [2, 2, 3, 3]})
        result = validate_engine_separation(train, val)
        assert result["no_overlap"] is False
        assert 2 in result["overlap_engines"]


# ---------------------------------------------------------------------------
# Data retention tests
# ---------------------------------------------------------------------------

class TestDataRetention:

    def test_retention_report(self):
        report = data_retention_report(
            original_rows=1000, feature_rows=900,
            original_engines=10, feature_engines=10,
            window_size=10,
        )
        assert report["discarded_rows"] == 100
        assert report["retention_pct"] == 90.0
        assert report["engines_preserved"] is True


# ---------------------------------------------------------------------------
# Feature-target correlation tests
# ---------------------------------------------------------------------------

class TestFeatureTargetCorrelations:

    def test_returns_dataframe(self, clean_feature_df):
        result = feature_target_correlations(clean_feature_df)
        assert isinstance(result, pd.DataFrame)
        assert "feature" in result.columns
        assert "correlation" in result.columns
        assert "abs_correlation" in result.columns

    def test_sorted_by_abs_correlation(self, clean_feature_df):
        result = feature_target_correlations(clean_feature_df)
        abs_corrs = result["abs_correlation"].values
        assert all(abs_corrs[i] >= abs_corrs[i + 1] for i in range(len(abs_corrs) - 1))
