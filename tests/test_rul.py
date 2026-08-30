"""
Tests for src.rul — RUL target generation, validation, and clipping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLUMN_NAMES
from src.data_loader import load_cmapss
from src.rul import (
    clip_rul,
    generate_rul_targets,
    rul_clipping_comparison,
    rul_statistics,
    validate_rul_targets,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd001_train():
    """Load FD001 training data."""
    data = load_cmapss("FD001")
    return data["train"]


@pytest.fixture(scope="module")
def fd001_with_rul(fd001_train):
    """FD001 training data with RUL targets."""
    return generate_rul_targets(fd001_train)


@pytest.fixture
def synthetic_train():
    """Small synthetic training DataFrame: 2 engines, 10 cycles each."""
    data = {
        "unit_id": [1] * 10 + [2] * 5,
        "cycle": list(range(1, 11)) + list(range(1, 6)),
    }
    # Fill remaining columns
    for col in COLUMN_NAMES[2:]:
        data[col] = np.random.default_rng(42).random(15)
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_with_rul(synthetic_train):
    return generate_rul_targets(synthetic_train)


# ---------------------------------------------------------------------------
# RUL generation tests
# ---------------------------------------------------------------------------

class TestGenerateRUL:
    """Tests for generate_rul_targets."""

    def test_adds_rul_column(self, fd001_with_rul):
        assert "rul" in fd001_with_rul.columns

    def test_preserves_original_columns(self, fd001_train, fd001_with_rul):
        for col in fd001_train.columns:
            assert col in fd001_with_rul.columns

    def test_preserves_row_count(self, fd001_train, fd001_with_rul):
        assert len(fd001_with_rul) == len(fd001_train)

    def test_does_not_modify_original(self, fd001_train):
        """generate_rul_targets should return a copy."""
        assert "rul" not in fd001_train.columns

    def test_synthetic_rul_values(self, synthetic_with_rul):
        """Engine 1: cycles 1-10, max=10, so RUL at cycle 1 = 9."""
        engine1 = synthetic_with_rul[synthetic_with_rul["unit_id"] == 1]
        assert engine1.iloc[0]["rul"] == 9  # cycle 1: 10 - 1 = 9
        assert engine1.iloc[-1]["rul"] == 0  # cycle 10: 10 - 10 = 0

    def test_synthetic_engine2_rul(self, synthetic_with_rul):
        """Engine 2: cycles 1-5, max=5, so RUL at cycle 1 = 4."""
        engine2 = synthetic_with_rul[synthetic_with_rul["unit_id"] == 2]
        assert engine2.iloc[0]["rul"] == 4
        assert engine2.iloc[-1]["rul"] == 0


# ---------------------------------------------------------------------------
# RUL validation tests
# ---------------------------------------------------------------------------

class TestValidateRUL:
    """Tests for validate_rul_targets."""

    def test_fd001_all_valid(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["all_valid"] is True

    def test_rul_non_negative(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["rul_non_negative"] is True
        assert result["n_negative_rul"] == 0

    def test_min_rul_zero_per_engine(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["min_rul_zero_per_engine"] is True

    def test_max_rul_correct(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["max_rul_correct"] is True

    def test_rul_decreases_by_one(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["rul_decreases_by_one"] is True

    def test_rul_plus_cycle_invariant(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["rul_plus_cycle_invariant"] is True

    def test_n_engines(self, fd001_with_rul):
        result = validate_rul_targets(fd001_with_rul)
        assert result["n_engines"] == 100

    def test_missing_rul_column_raises(self, fd001_train):
        with pytest.raises(ValueError, match="rul"):
            validate_rul_targets(fd001_train)


# ---------------------------------------------------------------------------
# RUL statistics tests
# ---------------------------------------------------------------------------

class TestRULStatistics:
    """Tests for rul_statistics."""

    def test_returns_expected_keys(self, fd001_with_rul):
        stats = rul_statistics(fd001_with_rul)
        expected = {"min_rul", "max_rul", "mean_rul", "median_rul",
                    "std_rul", "n_observations", "n_engines"}
        assert set(stats.keys()) == expected

    def test_min_rul_is_zero(self, fd001_with_rul):
        stats = rul_statistics(fd001_with_rul)
        assert stats["min_rul"] == 0

    def test_n_engines(self, fd001_with_rul):
        stats = rul_statistics(fd001_with_rul)
        assert stats["n_engines"] == 100


# ---------------------------------------------------------------------------
# RUL clipping tests
# ---------------------------------------------------------------------------

class TestClipping:
    """Tests for clip_rul and rul_clipping_comparison."""

    def test_clip_adds_column(self, fd001_with_rul):
        clipped = clip_rul(fd001_with_rul, rul_cap=125)
        assert "rul_clipped" in clipped.columns

    def test_clip_preserves_original_rul(self, fd001_with_rul):
        clipped = clip_rul(fd001_with_rul, rul_cap=125)
        assert "rul" in clipped.columns

    def test_clip_upper_bound(self, fd001_with_rul):
        clipped = clip_rul(fd001_with_rul, rul_cap=125)
        assert clipped["rul_clipped"].max() <= 125

    def test_clip_does_not_affect_low_rul(self, fd001_with_rul):
        clipped = clip_rul(fd001_with_rul, rul_cap=125)
        low_mask = fd001_with_rul["rul"] <= 125
        pd.testing.assert_series_equal(
            clipped.loc[low_mask, "rul_clipped"].reset_index(drop=True),
            fd001_with_rul.loc[low_mask, "rul"].reset_index(drop=True),
            check_names=False,
        )

    def test_clip_row_count_preserved(self, fd001_with_rul):
        clipped = clip_rul(fd001_with_rul, rul_cap=50)
        assert len(clipped) == len(fd001_with_rul)

    def test_comparison_table_shape(self, fd001_with_rul):
        comp = rul_clipping_comparison(fd001_with_rul)
        # Default: 6 cap values (None, 50, 75, 100, 125, 150)
        assert len(comp) == 6

    def test_comparison_none_has_zero_clipped(self, fd001_with_rul):
        comp = rul_clipping_comparison(fd001_with_rul)
        none_row = comp[comp["rul_cap"] == "None"].iloc[0]
        assert none_row["observations_clipped"] == 0

    def test_comparison_custom_caps(self, fd001_with_rul):
        comp = rul_clipping_comparison(fd001_with_rul, caps=[50, 100])
        assert len(comp) == 2
