"""
Tests for src.splitting — Engine-level train/validation splitting.
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
from src.rul import generate_rul_targets
from src.splitting import split_by_engine, validate_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd001_with_rul():
    """FD001 training data with RUL targets."""
    data = load_cmapss("FD001")
    return generate_rul_targets(data["train"])


@pytest.fixture(scope="module")
def fd001_split(fd001_with_rul):
    """Pre-computed split of FD001."""
    return split_by_engine(fd001_with_rul, validation_fraction=0.2, random_state=42)


@pytest.fixture
def synthetic_df():
    """Small synthetic DataFrame: 10 engines, varying lengths."""
    rows = []
    for uid in range(1, 11):
        n_cycles = 10 + uid * 5  # engine 1: 15 cycles, engine 10: 60 cycles
        for c in range(1, n_cycles + 1):
            row = {"unit_id": uid, "cycle": c}
            for col in COLUMN_NAMES[2:]:
                row[col] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Split tests
# ---------------------------------------------------------------------------

class TestSplitByEngine:
    """Tests for split_by_engine."""

    def test_no_engine_overlap(self, fd001_split):
        train_df, val_df = fd001_split
        train_engines = set(train_df["unit_id"].unique())
        val_engines = set(val_df["unit_id"].unique())
        assert len(train_engines & val_engines) == 0

    def test_all_rows_preserved(self, fd001_with_rul, fd001_split):
        train_df, val_df = fd001_split
        assert len(train_df) + len(val_df) == len(fd001_with_rul)

    def test_correct_engine_counts(self, fd001_split):
        train_df, val_df = fd001_split
        n_train = train_df["unit_id"].nunique()
        n_val = val_df["unit_id"].nunique()
        assert n_train + n_val == 100

    def test_approximately_20pct_validation(self, fd001_split):
        train_df, val_df = fd001_split
        n_val = val_df["unit_id"].nunique()
        # 20% of 100 = 20 engines
        assert n_val == 20

    def test_deterministic_split(self, fd001_with_rul):
        """Same seed produces identical splits."""
        t1, v1 = split_by_engine(fd001_with_rul, random_state=42)
        t2, v2 = split_by_engine(fd001_with_rul, random_state=42)
        assert set(t1["unit_id"].unique()) == set(t2["unit_id"].unique())
        assert set(v1["unit_id"].unique()) == set(v2["unit_id"].unique())

    def test_different_seed_different_split(self, fd001_with_rul):
        """Different seeds produce different splits."""
        _, v1 = split_by_engine(fd001_with_rul, random_state=42)
        _, v2 = split_by_engine(fd001_with_rul, random_state=99)
        assert set(v1["unit_id"].unique()) != set(v2["unit_id"].unique())

    def test_rul_column_preserved(self, fd001_split):
        train_df, val_df = fd001_split
        assert "rul" in train_df.columns
        assert "rul" in val_df.columns

    def test_no_rows_duplicated(self, fd001_with_rul, fd001_split):
        train_df, val_df = fd001_split
        combined = pd.concat([train_df, val_df], ignore_index=True)
        # No exact duplicate rows that weren't in original
        assert len(combined) == len(fd001_with_rul)

    def test_invalid_fraction_raises(self, fd001_with_rul):
        with pytest.raises(ValueError):
            split_by_engine(fd001_with_rul, validation_fraction=0.0)
        with pytest.raises(ValueError):
            split_by_engine(fd001_with_rul, validation_fraction=1.0)

    def test_synthetic_split(self, synthetic_df):
        train_df, val_df = split_by_engine(synthetic_df, validation_fraction=0.3)
        train_engines = set(train_df["unit_id"].unique())
        val_engines = set(val_df["unit_id"].unique())
        assert len(train_engines & val_engines) == 0
        assert len(train_df) + len(val_df) == len(synthetic_df)


# ---------------------------------------------------------------------------
# Split validation tests
# ---------------------------------------------------------------------------

class TestValidateSplit:
    """Tests for validate_split."""

    def test_fd001_split_valid(self, fd001_with_rul, fd001_split):
        train_df, val_df = fd001_split
        result = validate_split(train_df, val_df, fd001_with_rul)
        assert result["all_valid"] is True

    def test_no_overlap_reported(self, fd001_split):
        train_df, val_df = fd001_split
        result = validate_split(train_df, val_df)
        assert result["no_engine_overlap"] is True
        assert len(result["overlap_engines"]) == 0

    def test_rows_preserved(self, fd001_with_rul, fd001_split):
        train_df, val_df = fd001_split
        result = validate_split(train_df, val_df, fd001_with_rul)
        assert result["rows_preserved"] is True

    def test_trajectory_stats_present(self, fd001_split):
        train_df, val_df = fd001_split
        result = validate_split(train_df, val_df)
        assert "train_trajectory_stats" in result
        assert "val_trajectory_stats" in result
        assert "mean" in result["train_trajectory_stats"]
