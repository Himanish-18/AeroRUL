"""
Tests for src.data_loader — C-MAPSS data loading and discovery.
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

from src.config import AVAILABLE_SUBSETS, COLUMN_NAMES, EXPECTED_NUM_COLUMNS, RAW_DATA_DIR
from src.data_loader import _load_rul_file, _load_tabular_file, discover_datasets, load_cmapss


# ---------------------------------------------------------------------------
# Dataset discovery tests
# ---------------------------------------------------------------------------

class TestDiscoverDatasets:
    """Tests for the discover_datasets function."""

    def test_discover_returns_all_subsets(self):
        """discover_datasets should report on all four subsets."""
        result = discover_datasets()
        for subset in AVAILABLE_SUBSETS:
            assert subset in result, f"Missing subset: {subset}"

    def test_discover_reports_file_existence(self):
        """Each subset entry should have train, test, rul keys."""
        result = discover_datasets()
        for subset, info in result.items():
            assert "train" in info
            assert "test" in info
            assert "rul" in info

    def test_discover_nonexistent_dir_raises(self):
        """Passing a nonexistent directory should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_datasets(Path("/nonexistent/path/does/not/exist"))


# ---------------------------------------------------------------------------
# Schema / column tests
# ---------------------------------------------------------------------------

class TestSchema:
    """Tests for schema consistency."""

    def test_expected_column_count(self):
        """EXPECTED_NUM_COLUMNS should be 26."""
        assert EXPECTED_NUM_COLUMNS == 26

    def test_column_names_length(self):
        """COLUMN_NAMES should have exactly EXPECTED_NUM_COLUMNS entries."""
        assert len(COLUMN_NAMES) == EXPECTED_NUM_COLUMNS

    def test_column_names_start_with_ids(self):
        """First two columns should be unit_id and cycle."""
        assert COLUMN_NAMES[0] == "unit_id"
        assert COLUMN_NAMES[1] == "cycle"


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------

class TestLoadCmapss:
    """Tests for the load_cmapss function."""

    def test_invalid_subset_raises(self):
        """Invalid subset name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid subset"):
            load_cmapss("FD999")

    def test_load_fd001_returns_dict(self):
        """Loading FD001 should return a dict with train, test, rul keys."""
        result = load_cmapss("FD001")
        assert isinstance(result, dict)
        assert "train" in result
        assert "test" in result
        assert "rul" in result

    def test_load_fd001_train_columns(self):
        """FD001 train DataFrame should have correct column names."""
        result = load_cmapss("FD001")
        assert list(result["train"].columns) == COLUMN_NAMES

    def test_load_fd001_train_column_count(self):
        """FD001 train DataFrame should have 26 columns."""
        result = load_cmapss("FD001")
        assert result["train"].shape[1] == EXPECTED_NUM_COLUMNS

    def test_load_fd001_test_columns(self):
        """FD001 test DataFrame should have correct column names."""
        result = load_cmapss("FD001")
        assert list(result["test"].columns) == COLUMN_NAMES

    def test_load_fd001_nonempty(self):
        """FD001 should have non-empty train and test DataFrames."""
        result = load_cmapss("FD001")
        assert len(result["train"]) > 0
        assert len(result["test"]) > 0

    def test_load_fd001_rul_has_unit_id(self):
        """FD001 RUL should have unit_id and rul columns."""
        result = load_cmapss("FD001")
        assert result["rul"] is not None
        assert "unit_id" in result["rul"].columns
        assert "rul" in result["rul"].columns

    def test_unit_id_is_integer(self):
        """unit_id should be int64 dtype."""
        result = load_cmapss("FD001")
        assert result["train"]["unit_id"].dtype == np.int64

    def test_cycle_is_integer(self):
        """cycle should be int64 dtype."""
        result = load_cmapss("FD001")
        assert result["train"]["cycle"].dtype == np.int64

    def test_case_insensitive_subset(self):
        """load_cmapss should accept lowercase subset identifiers."""
        result = load_cmapss("fd001")
        assert isinstance(result, dict)
        assert len(result["train"]) > 0

    def test_missing_file_raises(self):
        """Loading from an empty directory should raise FileNotFoundError."""
        empty_dir = Path(__file__).parent / "_empty_test_dir_"
        empty_dir.mkdir(exist_ok=True)
        try:
            with pytest.raises(FileNotFoundError):
                load_cmapss("FD001", data_dir=empty_dir)
        finally:
            empty_dir.rmdir()


# ---------------------------------------------------------------------------
# Schema mismatch test with synthetic data
# ---------------------------------------------------------------------------

class TestSchemaMismatch:
    """Test schema validation catches malformed files."""

    def test_wrong_column_count_raises(self, tmp_path):
        """A file with wrong column count should raise ValueError."""
        # Create a fake file with only 10 columns
        fake_data = "1 1 0.1 0.2 0.3 100 200 300 400 500\n"
        fake_file = tmp_path / "fake.txt"
        fake_file.write_text(fake_data)

        with pytest.raises(ValueError, match="Schema mismatch"):
            _load_tabular_file(fake_file, COLUMN_NAMES)
