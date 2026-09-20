"""
Tests for src.features — Sliding-window feature engineering.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLUMN_NAMES, SENSOR_COLUMNS
from src.data_loader import load_cmapss
from src.features import (
    _compute_deltas,
    _compute_relative_changes,
    _compute_rolling_slope_r2,
    _compute_rolling_stats,
    generate_feature_metadata,
    generate_multiscale_features,
    generate_window_features,
    get_feature_columns,
)
from src.rul import generate_rul_targets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_engine():
    """Synthetic engine with known values for manual verification."""
    n = 50
    data = {
        "unit_id": [1] * n,
        "cycle": list(range(1, n + 1)),
        "op_setting_1": [0.0] * n,
        "op_setting_2": [0.0] * n,
        "op_setting_3": [100.0] * n,
    }
    # sensor_1: linear ramp 0,1,2,...,49
    data["sensor_1"] = list(range(n))
    # sensor_2: constant at 500
    data["sensor_2"] = [500.0] * n
    # sensor_3: known pattern for testing
    data["sensor_3"] = [float(i ** 2) for i in range(n)]
    # Fill remaining sensors
    for i in range(4, 22):
        data[f"sensor_{i}"] = [float(i) * 10.0] * n
    df = pd.DataFrame(data)
    return generate_rul_targets(df)


@pytest.fixture
def two_engine_df():
    """Two engines with different lengths."""
    rows = []
    # Engine 1: 30 cycles
    for c in range(1, 31):
        row = {"unit_id": 1, "cycle": c}
        for col in COLUMN_NAMES[2:]:
            if col == "sensor_1":
                row[col] = float(c)
            else:
                row[col] = 0.0
        rows.append(row)
    # Engine 2: 20 cycles
    for c in range(1, 21):
        row = {"unit_id": 2, "cycle": c}
        for col in COLUMN_NAMES[2:]:
            if col == "sensor_1":
                row[col] = float(c * 2)
            else:
                row[col] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    return generate_rul_targets(df)


@pytest.fixture(scope="module")
def fd001_with_rul():
    """FD001 training data with RUL targets."""
    data = load_cmapss("FD001")
    return generate_rul_targets(data["train"])


# ---------------------------------------------------------------------------
# Window construction tests
# ---------------------------------------------------------------------------

class TestWindowConstruction:
    """Tests for window construction correctness."""

    def test_first_complete_window_at_cycle_w(self, synthetic_engine):
        """First complete window should appear at the W-th cycle position."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # First row should be at cycle W (10th cycle)
        assert features["cycle"].min() == W

    def test_no_future_cycles_in_window(self, two_engine_df):
        """Features at cycle t must not include data from cycle > t."""
        W = 5
        features = generate_window_features(
            two_engine_df, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # For engine 1 at cycle 10, mean of sensor_1 should be mean(6,7,8,9,10)
        e1_c10 = features[(features["unit_id"] == 1) & (features["cycle"] == 10)]
        expected_mean = np.mean([6, 7, 8, 9, 10])
        assert abs(e1_c10["sensor_1_mean_5"].values[0] - expected_mean) < 1e-6

    def test_correct_window_length(self, synthetic_engine):
        """Window should contain exactly W observations."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean", "min", "max"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # At cycle 20 (0-indexed: 19), mean should be mean(10,11,...,19)
        c20 = features[features["cycle"] == 20]
        expected_mean = np.mean(list(range(10, 20)))
        assert abs(c20["sensor_1_mean_10"].values[0] - expected_mean) < 1e-6
        assert c20["sensor_1_min_10"].values[0] == 10.0
        assert c20["sensor_1_max_10"].values[0] == 19.0

    def test_row_count_after_windowing(self, synthetic_engine):
        """Should drop W-1 initial rows per engine."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # 50 cycles, W=10 → first valid at cycle 10 → 41 rows
        assert len(features) == 50 - W + 1

    def test_two_engines_independent(self, two_engine_df):
        """Windows must not cross engine boundaries."""
        W = 5
        features = generate_window_features(
            two_engine_df, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # Engine 2 first valid cycle should be 5
        e2 = features[features["unit_id"] == 2]
        assert e2["cycle"].min() == W
        # Engine 2 at cycle 5, mean should be mean(2,4,6,8,10) = mean of sensor_1 which is c*2
        e2_c5 = e2[e2["cycle"] == 5]
        expected = np.mean([1 * 2, 2 * 2, 3 * 2, 4 * 2, 5 * 2])
        assert abs(e2_c5["sensor_1_mean_5"].values[0] - expected) < 1e-6


# ---------------------------------------------------------------------------
# Feature correctness tests
# ---------------------------------------------------------------------------

class TestRollingStats:
    """Tests for rolling statistics correctness."""

    def test_mean_known_values(self, synthetic_engine):
        W = 5
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # At cycle 5 (values 0,1,2,3,4): mean = 2.0
        c5 = features[features["cycle"] == 5]
        assert abs(c5["sensor_1_mean_5"].values[0] - 2.0) < 1e-6

    def test_std_known_values(self, synthetic_engine):
        W = 5
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["std"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # At cycle 5 (values 0,1,2,3,4): population std
        expected_std = np.std([0, 1, 2, 3, 4], ddof=0)
        c5 = features[features["cycle"] == 5]
        assert abs(c5["sensor_1_std_5"].values[0] - expected_std) < 1e-6

    def test_constant_sensor_std_zero(self, synthetic_engine):
        W = 5
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_2"],  # constant at 500
            statistics=["std"],
            compute_deltas=[], compute_relative_changes=[],
        )
        assert (features["sensor_2_std_5"] == 0.0).all()

    def test_range_equals_max_minus_min(self, synthetic_engine):
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["min", "max", "range"],
            compute_deltas=[], compute_relative_changes=[],
        )
        diff = (features["sensor_1_max_10"] - features["sensor_1_min_10"]).values
        assert np.allclose(diff, features["sensor_1_range_10"].values)


# ---------------------------------------------------------------------------
# Trend feature tests
# ---------------------------------------------------------------------------

class TestTrendFeatures:
    """Tests for slope and R² correctness."""

    def test_linear_slope(self, synthetic_engine):
        """Perfectly linear sensor_1 should have slope ≈ 1.0."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["slope"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # sensor_1 increases by 1 per cycle → slope = 1.0
        slopes = features["sensor_1_slope_10"].dropna()
        assert np.allclose(slopes, 1.0, atol=1e-6)

    def test_linear_r2(self, synthetic_engine):
        """Perfectly linear sensor_1 should have R² = 1.0."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["r2"],
            compute_deltas=[], compute_relative_changes=[],
        )
        r2_vals = features["sensor_1_r2_10"].dropna()
        assert np.allclose(r2_vals, 1.0, atol=1e-4)

    def test_constant_sensor_slope_zero(self, synthetic_engine):
        """Constant sensor_2 should have slope = 0."""
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_2"],
            statistics=["slope"],
            compute_deltas=[], compute_relative_changes=[],
        )
        slopes = features["sensor_2_slope_10"].dropna()
        assert np.allclose(slopes, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Delta feature tests
# ---------------------------------------------------------------------------

class TestDeltaFeatures:
    """Tests for temporal difference features."""

    def test_delta_1_linear(self, synthetic_engine):
        """delta_1 of linear sensor_1 (step=1) should be 1.0."""
        features = generate_window_features(
            synthetic_engine, window_size=5,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[1], compute_relative_changes=[],
        )
        deltas = features["sensor_1_delta_1"].dropna()
        assert np.allclose(deltas, 1.0)

    def test_delta_5_linear(self, synthetic_engine):
        """delta_5 of linear sensor_1 should be 5.0."""
        features = generate_window_features(
            synthetic_engine, window_size=5,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[5], compute_relative_changes=[],
        )
        deltas = features["sensor_1_delta_5"].dropna()
        assert np.allclose(deltas, 5.0)


# ---------------------------------------------------------------------------
# Relative change feature tests
# ---------------------------------------------------------------------------

class TestRelativeChangeFeatures:
    """Tests for relative change features."""

    def test_no_infinite_values(self, synthetic_engine):
        """Relative changes should never produce infinities."""
        features = generate_window_features(
            synthetic_engine, window_size=5,
            sensor_columns=["sensor_1", "sensor_2"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[5],
        )
        for col in features.columns:
            if "relchange" in col:
                vals = features[col].dropna()
                assert not np.any(np.isinf(vals)), f"Inf in {col}"


# ---------------------------------------------------------------------------
# Multi-scale feature tests
# ---------------------------------------------------------------------------

class TestMultiscaleFeatures:
    """Tests for multi-scale feature generation."""

    def test_combines_multiple_windows(self, synthetic_engine):
        features = generate_multiscale_features(
            synthetic_engine,
            windows=[5, 10],
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            deltas=[], relative_changes=[],
        )
        assert "sensor_1_mean_5" in features.columns
        assert "sensor_1_mean_10" in features.columns

    def test_rows_match_largest_window(self, synthetic_engine):
        """Output should only contain rows valid for the largest window."""
        features = generate_multiscale_features(
            synthetic_engine,
            windows=[5, 10, 20],
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            deltas=[], relative_changes=[],
        )
        # Largest window = 20, so first valid at cycle 20
        # 50 - 20 + 1 = 31 rows
        assert len(features) == 50 - 20 + 1
        assert features["cycle"].min() == 20

    def test_metadata_columns_preserved(self, synthetic_engine):
        features = generate_multiscale_features(
            synthetic_engine,
            windows=[10],
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            deltas=[], relative_changes=[],
        )
        assert "unit_id" in features.columns
        assert "cycle" in features.columns
        assert "rul" in features.columns


# ---------------------------------------------------------------------------
# LEAKAGE TEST (CRITICAL)
# ---------------------------------------------------------------------------

class TestCausalityNoLeakage:
    """
    Critical test: features at cycle t must NOT change when future cycles
    are modified. This is the fundamental causal invariant.
    """

    def test_future_modification_does_not_affect_past_features(self):
        """
        1. Create engine with 50 cycles
        2. Generate features at cycle 30
        3. Modify cycles 31-50
        4. Regenerate features at cycle 30
        5. Assert features at cycle 30 are IDENTICAL
        """
        n = 50
        # Original data
        data_orig = {
            "unit_id": [1] * n,
            "cycle": list(range(1, n + 1)),
        }
        for col in COLUMN_NAMES[2:]:
            if col == "sensor_1":
                data_orig[col] = [float(i) for i in range(n)]
            else:
                data_orig[col] = [100.0] * n
        df_orig = generate_rul_targets(pd.DataFrame(data_orig))

        # Modified data: change cycles 31-50
        df_modified = df_orig.copy()
        mask = df_modified["cycle"] > 30
        df_modified.loc[mask, "sensor_1"] = 9999.0
        df_modified.loc[mask, "sensor_2"] = -9999.0
        # Recalculate RUL (failure cycle is still 50)
        df_modified["rul"] = df_modified.groupby("unit_id")["cycle"].transform("max") - df_modified["cycle"]

        # Generate features for both
        W = 10
        sensors = ["sensor_1", "sensor_2"]
        feat_orig = generate_window_features(
            df_orig, window_size=W, sensor_columns=sensors,
            statistics=["mean", "std", "slope"],
            compute_deltas=[1, 5], compute_relative_changes=[5],
        )
        feat_mod = generate_window_features(
            df_modified, window_size=W, sensor_columns=sensors,
            statistics=["mean", "std", "slope"],
            compute_deltas=[1, 5], compute_relative_changes=[5],
        )

        # Compare features at cycle 30 — should be IDENTICAL
        orig_c30 = feat_orig[feat_orig["cycle"] == 30].reset_index(drop=True)
        mod_c30 = feat_mod[feat_mod["cycle"] == 30].reset_index(drop=True)

        feature_cols = get_feature_columns(orig_c30)
        for col in feature_cols:
            orig_val = orig_c30[col].values[0]
            mod_val = mod_c30[col].values[0]
            if np.isnan(orig_val) and np.isnan(mod_val):
                continue
            assert abs(orig_val - mod_val) < 1e-10, (
                f"LEAKAGE DETECTED: {col} changed when future was modified. "
                f"Original={orig_val}, Modified={mod_val}"
            )


# ---------------------------------------------------------------------------
# Target preservation tests
# ---------------------------------------------------------------------------

class TestTargetPreservation:
    """Tests that RUL is correctly preserved in feature output."""

    def test_rul_preserved(self, synthetic_engine):
        features = generate_window_features(
            synthetic_engine, window_size=10,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        assert "rul" in features.columns
        assert (features["rul"] >= 0).all()

    def test_rul_matches_original(self, synthetic_engine):
        W = 10
        features = generate_window_features(
            synthetic_engine, window_size=W,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        # Merge on unit_id, cycle to verify RUL
        merged = features.merge(
            synthetic_engine[["unit_id", "cycle", "rul"]],
            on=["unit_id", "cycle"],
            suffixes=("_feat", "_orig"),
        )
        assert (merged["rul_feat"] == merged["rul_orig"]).all()

    def test_no_failure_cycle_in_features(self, synthetic_engine):
        features = generate_window_features(
            synthetic_engine, window_size=10,
            sensor_columns=["sensor_1"],
            statistics=["mean"],
            compute_deltas=[], compute_relative_changes=[],
        )
        assert "failure_cycle" not in features.columns


# ---------------------------------------------------------------------------
# Feature metadata tests
# ---------------------------------------------------------------------------

class TestFeatureMetadata:
    """Tests for feature metadata generation."""

    def test_metadata_parses_correctly(self):
        cols = ["sensor_1_mean_10", "sensor_2_slope_30", "sensor_21_r2_50"]
        meta = generate_feature_metadata(cols)
        assert len(meta) == 3
        assert meta.iloc[0]["sensor"] == "sensor_1"
        assert meta.iloc[0]["statistic"] == "mean"
        assert meta.iloc[0]["window_or_lag"] == 10
        assert meta.iloc[1]["sensor"] == "sensor_2"
        assert meta.iloc[1]["statistic"] == "slope"
        assert meta.iloc[2]["sensor"] == "sensor_21"
        assert meta.iloc[2]["window_or_lag"] == 50

    def test_metadata_handles_deltas(self):
        cols = ["sensor_1_delta_1", "sensor_2_relchange_5"]
        meta = generate_feature_metadata(cols)
        assert meta.iloc[0]["statistic"] == "delta"
        assert meta.iloc[1]["statistic"] == "relchange"
