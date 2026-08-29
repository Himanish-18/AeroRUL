"""
Tests for src.data_validation — C-MAPSS data validation functions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLUMN_NAMES, EXPECTED_NUM_COLUMNS, SENSOR_COLUMNS
from src.data_loader import load_cmapss
from src.data_validation import (
    analyze_operating_conditions,
    get_engine_final_cycles,
    sensor_correlation_matrix,
    sensor_cycle_correlation,
    trajectory_summary,
    validate_engines,
    validate_missing_values,
    validate_sensors,
    validate_structure,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd001_data():
    """Load FD001 once for the entire test module."""
    return load_cmapss("FD001")


@pytest.fixture
def synthetic_df():
    """Create a small synthetic DataFrame matching C-MAPSS schema."""
    n_rows = 20
    data = {
        "unit_id": [1] * 10 + [2] * 10,
        "cycle": list(range(1, 11)) + list(range(1, 11)),
    }
    for col in COLUMN_NAMES[2:]:
        data[col] = np.random.default_rng(42).random(n_rows)
    return pd.DataFrame(data)


@pytest.fixture
def df_with_nans(synthetic_df):
    """Synthetic DataFrame with some NaN values injected."""
    df = synthetic_df.copy()
    df.loc[0, "sensor_1"] = np.nan
    df.loc[1, "sensor_1"] = np.nan
    df.loc[5, "sensor_5"] = np.nan
    return df


@pytest.fixture
def df_with_duplicates(synthetic_df):
    """Synthetic DataFrame with duplicate (unit_id, cycle) rows."""
    df = synthetic_df.copy()
    dup_row = df.iloc[[0]].copy()
    df = pd.concat([df, dup_row], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Structural validation tests
# ---------------------------------------------------------------------------

class TestValidateStructure:
    """Tests for validate_structure."""

    def test_column_count_ok(self, fd001_data):
        result = validate_structure(fd001_data["train"], "FD001_train")
        assert result["column_count_ok"] is True

    def test_column_names_ok(self, fd001_data):
        result = validate_structure(fd001_data["train"], "FD001_train")
        assert result["column_names_ok"] is True

    def test_returns_row_count(self, fd001_data):
        result = validate_structure(fd001_data["train"], "FD001_train")
        assert result["n_rows"] > 0

    def test_no_duplicates_in_fd001(self, fd001_data):
        result = validate_structure(fd001_data["train"], "FD001_train")
        assert result["n_duplicate_unit_cycle"] == 0


# ---------------------------------------------------------------------------
# Missing value validation tests
# ---------------------------------------------------------------------------

class TestValidateMissingValues:
    """Tests for validate_missing_values."""

    def test_no_missing_in_fd001(self, fd001_data):
        result = validate_missing_values(fd001_data["train"])
        assert result["n_total_missing"].sum() == 0

    def test_detects_nans(self, df_with_nans):
        result = validate_missing_values(df_with_nans)
        sensor1_row = result[result["column"] == "sensor_1"].iloc[0]
        assert sensor1_row["n_nan"] == 2

    def test_reports_percentages(self, df_with_nans):
        result = validate_missing_values(df_with_nans)
        sensor1_row = result[result["column"] == "sensor_1"].iloc[0]
        # 2 NaN out of 20 rows = 10%
        assert sensor1_row["pct_nan"] == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Engine validation tests
# ---------------------------------------------------------------------------

class TestValidateEngines:
    """Tests for validate_engines."""

    def test_fd001_has_100_engines(self, fd001_data):
        result = validate_engines(fd001_data["train"])
        assert result["n_engines"] == 100

    def test_cycles_ordered(self, fd001_data):
        result = validate_engines(fd001_data["train"])
        assert result["cycles_ordered"] is True

    def test_cycles_start_at_1(self, fd001_data):
        result = validate_engines(fd001_data["train"])
        assert result["cycles_start_at_1"] is True

    def test_synthetic_engine_count(self, synthetic_df):
        result = validate_engines(synthetic_df)
        assert result["n_engines"] == 2

    def test_synthetic_max_cycles(self, synthetic_df):
        result = validate_engines(synthetic_df)
        assert result["max_cycles"] == 10


# ---------------------------------------------------------------------------
# Duplicate detection tests
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    """Tests for duplicate detection in validate_structure."""

    def test_detects_duplicate_unit_cycle(self, df_with_duplicates):
        result = validate_structure(df_with_duplicates, "test")
        assert result["n_duplicate_unit_cycle"] > 0


# ---------------------------------------------------------------------------
# Engine final cycles
# ---------------------------------------------------------------------------

class TestEngineFinalCycles:
    """Tests for get_engine_final_cycles."""

    def test_returns_all_engines(self, fd001_data):
        final = get_engine_final_cycles(fd001_data["train"])
        assert len(final) == 100

    def test_has_correct_columns(self, fd001_data):
        final = get_engine_final_cycles(fd001_data["train"])
        assert list(final.columns) == ["unit_id", "final_cycle"]


# ---------------------------------------------------------------------------
# Trajectory summary tests
# ---------------------------------------------------------------------------

class TestTrajectorySummary:
    """Tests for trajectory_summary."""

    def test_returns_correct_keys(self, fd001_data):
        result = trajectory_summary(fd001_data["train"], "FD001_train")
        expected_keys = {
            "label", "n_engines", "n_observations",
            "avg_trajectory_length", "min_trajectory_length",
            "max_trajectory_length", "median_trajectory_length",
        }
        assert set(result.keys()) == expected_keys

    def test_fd001_engine_count(self, fd001_data):
        result = trajectory_summary(fd001_data["train"], "FD001_train")
        assert result["n_engines"] == 100


# ---------------------------------------------------------------------------
# Sensor validation tests
# ---------------------------------------------------------------------------

class TestValidateSensors:
    """Tests for validate_sensors."""

    def test_returns_21_rows(self, fd001_data):
        result = validate_sensors(fd001_data["train"])
        assert len(result) == 21

    def test_has_expected_columns(self, fd001_data):
        result = validate_sensors(fd001_data["train"])
        expected = {"sensor", "min", "max", "mean", "std", "var", "n_unique", "pct_missing", "is_constant", "is_near_constant"}
        assert set(result.columns) == expected


# ---------------------------------------------------------------------------
# Sensor correlation tests
# ---------------------------------------------------------------------------

class TestSensorCorrelation:
    """Tests for sensor correlation functions."""

    def test_cycle_correlation_returns_21_rows(self, fd001_data):
        result = sensor_cycle_correlation(fd001_data["train"])
        assert len(result) == 21

    def test_correlation_matrix_is_square(self, fd001_data):
        result = sensor_correlation_matrix(fd001_data["train"])
        assert result.shape[0] == result.shape[1]
        assert result.shape[0] == 21


# ---------------------------------------------------------------------------
# Operating conditions tests
# ---------------------------------------------------------------------------

class TestOperatingConditions:
    """Tests for analyze_operating_conditions."""

    def test_returns_3_rows(self, fd001_data):
        result = analyze_operating_conditions(fd001_data["train"])
        assert len(result) == 3

    def test_has_expected_columns(self, fd001_data):
        result = analyze_operating_conditions(fd001_data["train"])
        expected = {"setting", "n_unique", "mean", "std", "min", "max"}
        assert set(result.columns) == expected
