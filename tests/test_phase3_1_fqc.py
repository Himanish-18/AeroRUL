"""
Tests for Phase 3.1 — Feature Quality Control.

Covers:
    - NaN handling and root-cause analysis
    - Constant feature detection
    - Schema consistency
    - Metadata/target exclusion from feature matrix
    - Temporal causality
    - Engine separation
    - Normalization isolation
    - Duplicate feature detection
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    COLUMN_NAMES,
    METADATA_COLUMNS,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
)
from src.feature_quality_control import (
    constant_feature_audit,
    detect_duplicate_features,
    determine_nan_handling,
    distribution_shift_analysis,
    generate_feature_usability_inventory,
    nan_root_cause_analysis,
    verify_engine_separation,
    verify_schema_consistency,
    verify_target_isolation,
    verify_temporal_causality,
)
from src.features import (
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
def synthetic_train_val():
    """Create synthetic train/validation feature datasets."""
    np.random.seed(42)
    n_train, n_val = 100, 30

    train = pd.DataFrame({
        "unit_id": [1] * 50 + [2] * 50,
        "cycle": list(range(75, 125)) + list(range(75, 125)),
        "rul": list(range(49, -1, -1)) + list(range(49, -1, -1)),
        "sensor_1_mean_10": np.random.randn(n_train),
        "sensor_1_std_10": np.abs(np.random.randn(n_train)),
        "sensor_1_slope_10": np.random.randn(n_train) * 0.1,
        "sensor_1_r2_10": np.random.uniform(0, 1, n_train),
        "sensor_1_delta_1": np.random.randn(n_train),
        "sensor_1_relchange_5": np.random.randn(n_train) * 0.01,
        "sensor_5_mean_10": [42.0] * n_train,  # constant
        "sensor_5_std_10": [0.0] * n_train,  # constant (zero)
        "sensor_5_range_10": [0.0] * n_train,  # constant (zero)
    })

    val = pd.DataFrame({
        "unit_id": [3] * 15 + [4] * 15,
        "cycle": list(range(75, 90)) + list(range(75, 90)),
        "rul": list(range(14, -1, -1)) + list(range(14, -1, -1)),
        "sensor_1_mean_10": np.random.randn(n_val),
        "sensor_1_std_10": np.abs(np.random.randn(n_val)),
        "sensor_1_slope_10": np.random.randn(n_val) * 0.1,
        "sensor_1_r2_10": np.random.uniform(0, 1, n_val),
        "sensor_1_delta_1": np.random.randn(n_val),
        "sensor_1_relchange_5": np.random.randn(n_val) * 0.01,
        "sensor_5_mean_10": [42.0] * n_val,
        "sensor_5_std_10": [0.0] * n_val,
        "sensor_5_range_10": [0.0] * n_val,
    })

    return train, val


@pytest.fixture
def feature_metadata_fixture():
    """Feature metadata for synthetic data."""
    cols = [
        "sensor_1_mean_10", "sensor_1_std_10", "sensor_1_slope_10",
        "sensor_1_r2_10", "sensor_1_delta_1", "sensor_1_relchange_5",
        "sensor_5_mean_10", "sensor_5_std_10", "sensor_5_range_10",
    ]
    return generate_feature_metadata(cols)


@pytest.fixture
def nan_train_val():
    """Train/val with intentional NaN features."""
    np.random.seed(42)
    n = 50

    # R² NaN: simulate constant-window edge case
    r2_vals = np.random.uniform(0, 1, n)
    r2_vals[0:5] = np.nan  # First 5 are NaN (constant window)

    train = pd.DataFrame({
        "unit_id": [1] * n,
        "cycle": list(range(75, 75 + n)),
        "rul": list(range(n - 1, -1, -1)),
        "sensor_1_mean_10": np.random.randn(n),
        "sensor_1_r2_10": r2_vals,
        "sensor_1_slope_10": np.random.randn(n),
    })

    val = pd.DataFrame({
        "unit_id": [2] * 20,
        "cycle": list(range(75, 95)),
        "rul": list(range(19, -1, -1)),
        "sensor_1_mean_10": np.random.randn(20),
        "sensor_1_r2_10": np.random.uniform(0, 1, 20),
        "sensor_1_slope_10": np.random.randn(20),
    })

    return train, val


@pytest.fixture
def full_synthetic_engine():
    """Full engine data for temporal causality testing."""
    n = 100
    data = {
        "unit_id": [1] * n,
        "cycle": list(range(1, n + 1)),
    }
    for col in COLUMN_NAMES[2:]:
        if col == "sensor_1":
            data[col] = [float(i) for i in range(n)]
        elif col == "sensor_2":
            data[col] = [500.0] * n  # constant
        else:
            data[col] = [float(np.sin(i * 0.1)) for i in range(n)]
    return generate_rul_targets(pd.DataFrame(data))


# ---------------------------------------------------------------------------
# Test: NaN Feature Behavior
# ---------------------------------------------------------------------------

class TestNaNFeatureBehavior:
    """Test NaN detection and root-cause classification."""

    def test_nan_detection(self, nan_train_val, feature_metadata_fixture):
        """NaN features should be correctly identified."""
        train, val = nan_train_val
        # Use a minimal metadata
        meta = generate_feature_metadata(get_feature_columns(train))
        audit = nan_root_cause_analysis(train, val, meta)
        # sensor_1_r2_10 has NaN in train
        r2_row = audit[audit["feature_name"] == "sensor_1_r2_10"]
        assert len(r2_row) == 1
        assert r2_row.iloc[0]["nan_count_train"] == 5
        assert r2_row.iloc[0]["in_train"] == True

    def test_r2_nan_classified_as_expected(self, nan_train_val):
        """R² NaN should be classified as EXPECTED_NUMERICAL_EDGE_CASE."""
        train, val = nan_train_val
        meta = generate_feature_metadata(get_feature_columns(train))
        audit = nan_root_cause_analysis(train, val, meta)
        r2_row = audit[audit["feature_name"] == "sensor_1_r2_10"]
        assert r2_row.iloc[0]["root_cause_category"] == "EXPECTED_NUMERICAL_EDGE_CASE"
        assert r2_row.iloc[0]["expected"] == True

    def test_nan_handling_r2_convert(self, nan_train_val):
        """R² NaN handling should recommend CONVERT_TO_VALUE with 0.0."""
        train, val = nan_train_val
        meta = generate_feature_metadata(get_feature_columns(train))
        audit = nan_root_cause_analysis(train, val, meta)
        audit = determine_nan_handling(audit)
        r2_row = audit[audit["feature_name"] == "sensor_1_r2_10"]
        assert r2_row.iloc[0]["action"] == "CONVERT_TO_VALUE"
        assert r2_row.iloc[0]["replacement_value"] == 0.0

    def test_no_blanket_fillna(self, nan_train_val):
        """Handling should NOT use blanket fillna(0) for all features."""
        train, val = nan_train_val
        meta = generate_feature_metadata(get_feature_columns(train))
        audit = nan_root_cause_analysis(train, val, meta)
        audit = determine_nan_handling(audit)
        # Each feature should have a specific action with justification
        for _, row in audit.iterrows():
            assert row["action"] in (
                "KEEP_NAN", "CONVERT_TO_VALUE", "EXCLUDE_FEATURE",
                "FIX_IMPLEMENTATION",
            )
            assert row["justification"] is not None
            assert len(str(row["justification"])) > 10


# ---------------------------------------------------------------------------
# Test: Constant Feature Detection
# ---------------------------------------------------------------------------

class TestConstantFeatureDetection:
    """Test constant feature identification and classification."""

    def test_detects_constant_features(self, synthetic_train_val, feature_metadata_fixture):
        """Constant features should be detected."""
        train, val = synthetic_train_val
        audit = constant_feature_audit(train, val, feature_metadata_fixture)
        # sensor_5_mean_10, sensor_5_std_10, sensor_5_range_10 are constant
        const_names = set(audit["feature_name"].tolist())
        assert "sensor_5_mean_10" in const_names
        assert "sensor_5_std_10" in const_names
        assert "sensor_5_range_10" in const_names

    def test_constant_classification(self, synthetic_train_val, feature_metadata_fixture):
        """Constant features from constant sensors should be classified correctly."""
        train, val = synthetic_train_val
        audit = constant_feature_audit(
            train, val, feature_metadata_fixture,
            constant_sensors=["sensor_5"],
        )
        for _, row in audit.iterrows():
            if row["sensor"] == "sensor_5":
                assert row["original_sensor_constant"] is True
                assert row["category"] == "ORIGINAL_SENSOR_CONSTANT"


# ---------------------------------------------------------------------------
# Test: Train/Validation Feature Schema
# ---------------------------------------------------------------------------

class TestTrainValidationFeatureSchema:
    """Test schema consistency between train and validation."""

    def test_identical_schema(self, synthetic_train_val):
        """Train and val should have identical column schema."""
        train, val = synthetic_train_val
        result = verify_schema_consistency(train, val)
        assert result["same_columns"] is True
        assert result["same_feature_count"] is True
        assert result["same_feature_names"] is True
        assert result["same_feature_order"] is True
        assert result["schema_valid"] is True

    def test_detects_missing_column(self, synthetic_train_val):
        """Should detect when val is missing a column."""
        train, val = synthetic_train_val
        val_bad = val.drop(columns=["sensor_1_slope_10"])
        result = verify_schema_consistency(train, val_bad)
        assert result["same_columns"] is False
        assert result["schema_valid"] is False
        assert "sensor_1_slope_10" in result["train_only_features"]

    def test_detects_extra_column(self, synthetic_train_val):
        """Should detect when val has an extra column."""
        train, val = synthetic_train_val
        val_bad = val.copy()
        val_bad["extra_feature"] = 0.0
        result = verify_schema_consistency(train, val_bad)
        assert result["same_columns"] is False
        assert "extra_feature" in result["val_only_features"]


# ---------------------------------------------------------------------------
# Test: No Target or Metadata in Feature Matrix
# ---------------------------------------------------------------------------

class TestNoTargetOrMetadataInFeatureMatrix:
    """Verify target/metadata columns are excluded from model features."""

    def test_rul_not_in_features(self, synthetic_train_val):
        """RUL should not appear in feature columns."""
        train, _ = synthetic_train_val
        feat_cols = get_feature_columns(train)
        assert "rul" not in feat_cols

    def test_unit_id_not_in_features(self, synthetic_train_val):
        """unit_id should not appear in feature columns."""
        train, _ = synthetic_train_val
        feat_cols = get_feature_columns(train)
        assert "unit_id" not in feat_cols

    def test_cycle_not_in_features(self, synthetic_train_val):
        """cycle should not appear in feature columns."""
        train, _ = synthetic_train_val
        feat_cols = get_feature_columns(train)
        assert "cycle" not in feat_cols

    def test_failure_cycle_not_in_features(self, synthetic_train_val):
        """failure_cycle should not appear anywhere."""
        train, _ = synthetic_train_val
        assert "failure_cycle" not in train.columns

    def test_target_isolation_verification(self, synthetic_train_val):
        """Target isolation check should pass for clean data."""
        train, val = synthetic_train_val
        result = verify_target_isolation(train, val)
        assert result["verified"] is True

    def test_target_isolation_detects_leakage(self, synthetic_train_val):
        """Should detect when failure_cycle is present."""
        train, val = synthetic_train_val
        train_bad = train.copy()
        train_bad["failure_cycle"] = 200
        result = verify_target_isolation(train_bad, val)
        assert result["verified"] is False


# ---------------------------------------------------------------------------
# Test: Feature Temporal Causality
# ---------------------------------------------------------------------------

class TestFeatureTemporalCausality:
    """Features at cycle t must depend only on cycles <= t."""

    def test_temporal_causality_passes(self, full_synthetic_engine):
        """Temporal causality should pass for correctly implemented features."""
        result = verify_temporal_causality(
            full_synthetic_engine,
            sensor_columns=["sensor_1", "sensor_3"],
            window=10,
            test_cycle=30,
        )
        assert result["verified"] is True
        assert result["n_violations"] == 0

    def test_future_modification_invariance(self):
        """
        Modify future cycles and verify features at earlier cycles
        remain unchanged.
        """
        n = 60
        data = {
            "unit_id": [1] * n,
            "cycle": list(range(1, n + 1)),
        }
        for col in COLUMN_NAMES[2:]:
            if col == "sensor_1":
                data[col] = [float(i) for i in range(n)]
            else:
                data[col] = [100.0] * n
        df = generate_rul_targets(pd.DataFrame(data))

        W = 10
        sensors = ["sensor_1"]

        feat_orig = generate_window_features(
            df, window_size=W, sensor_columns=sensors,
            statistics=["mean", "std", "slope"],
            compute_deltas=[1], compute_relative_changes=[5],
        )

        df_mod = df.copy()
        df_mod.loc[df_mod["cycle"] > 30, "sensor_1"] = 99999.0
        df_mod["rul"] = df_mod.groupby("unit_id")["cycle"].transform("max") - df_mod["cycle"]

        feat_mod = generate_window_features(
            df_mod, window_size=W, sensor_columns=sensors,
            statistics=["mean", "std", "slope"],
            compute_deltas=[1], compute_relative_changes=[5],
        )

        orig_c25 = feat_orig[feat_orig["cycle"] == 25].reset_index(drop=True)
        mod_c25 = feat_mod[feat_mod["cycle"] == 25].reset_index(drop=True)

        for col in get_feature_columns(orig_c25):
            o = orig_c25[col].values[0]
            m = mod_c25[col].values[0]
            if np.isnan(o) and np.isnan(m):
                continue
            assert abs(o - m) < 1e-10, f"LEAKAGE: {col}"


# ---------------------------------------------------------------------------
# Test: No Engine Overlap
# ---------------------------------------------------------------------------

class TestNoEngineOverlap:
    """No engine should appear in both train and validation."""

    def test_no_overlap(self, synthetic_train_val):
        """Engines should be disjoint between train and val."""
        train, val = synthetic_train_val
        result = verify_engine_separation(train, val)
        assert result["verified"] is True
        assert len(result["overlap"]) == 0

    def test_detects_overlap(self, synthetic_train_val):
        """Should detect when an engine appears in both splits."""
        train, val = synthetic_train_val
        # Inject engine 1 into val
        val_bad = pd.concat([val, train[train["unit_id"] == 1].head(5)])
        result = verify_engine_separation(train, val_bad)
        assert result["verified"] is False
        assert 1 in result["overlap"]


# ---------------------------------------------------------------------------
# Test: Normalization Uses Training Data Only
# ---------------------------------------------------------------------------

class TestNormalizationUsesTrainingDataOnly:
    """Normalization parameters must be derived from training data only."""

    def test_normalization_isolation(self):
        """Verify norm params come from train only."""
        from src.normalization import fit_regime_normalization, identify_operating_regimes

        # Create train and val with different means
        train = pd.DataFrame({
            "unit_id": [1] * 50,
            "cycle": list(range(1, 51)),
            "op_setting_1": [0.0] * 50,
            "op_setting_2": [0.0] * 50,
            "op_setting_3": [100.0] * 50,
            "sensor_1": [100.0] * 50,
            "sensor_2": [200.0] * 50,
        })
        val = pd.DataFrame({
            "unit_id": [2] * 30,
            "cycle": list(range(1, 31)),
            "op_setting_1": [0.0] * 30,
            "op_setting_2": [0.0] * 30,
            "op_setting_3": [100.0] * 30,
            "sensor_1": [500.0] * 30,  # Very different from train
            "sensor_2": [600.0] * 30,
        })

        regimes = identify_operating_regimes(train)
        params = fit_regime_normalization(
            train, regimes, sensor_columns=["sensor_1", "sensor_2"],
        )

        # Verify params match training data
        for regime, p in params.items():
            assert abs(p["mean"]["sensor_1"] - 100.0) < 1e-6
            assert abs(p["mean"]["sensor_2"] - 200.0) < 1e-6
            # Val data (500, 600) should NOT influence these params
            assert abs(p["mean"]["sensor_1"] - 500.0) > 100
            assert abs(p["mean"]["sensor_2"] - 600.0) > 100


# ---------------------------------------------------------------------------
# Test: Duplicate Feature Detection
# ---------------------------------------------------------------------------

class TestDuplicateFeatureDetection:
    """Test detection of exactly duplicate feature columns."""

    def test_no_duplicates_on_varying_features(self):
        """Features with different values should not be detected as duplicates."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
            "rul": [4, 3, 2, 1, 0],
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [5.0, 4.0, 3.0, 2.0, 1.0],
            "feat_c": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = detect_duplicate_features(df)
        assert len(result) == 0

    def test_detects_exact_duplicates(self):
        """Should detect when two columns have identical values."""
        df = pd.DataFrame({
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "rul": [2, 1, 0],
            "feat_a": [1.0, 2.0, 3.0],
            "feat_b": [1.0, 2.0, 3.0],  # exact copy of feat_a
            "feat_c": [4.0, 5.0, 6.0],
        })
        result = detect_duplicate_features(df)
        assert len(result) > 0
        dup_names = set(result["feature_name"].tolist())
        assert "feat_a" in dup_names or "feat_b" in dup_names


# ---------------------------------------------------------------------------
# Test: Distribution Shift Analysis
# ---------------------------------------------------------------------------

class TestDistributionShift:
    """Test distribution shift diagnostics."""

    def test_shift_analysis_runs(self, synthetic_train_val):
        """Distribution shift analysis should produce results."""
        train, val = synthetic_train_val
        result = distribution_shift_analysis(train, val)
        assert len(result) > 0
        assert "standardized_difference" in result.columns
        assert "shift_flag" in result.columns

    def test_large_shift_flagged(self):
        """Features with large mean differences should be flagged."""
        np.random.seed(123)
        train = pd.DataFrame({
            "unit_id": [1] * 100,
            "cycle": list(range(100)),
            "rul": list(range(99, -1, -1)),
            "feat_normal": np.random.randn(100),
            "feat_shifted": np.random.randn(100) * 0.5,  # mean ~0, std ~0.5
        })
        val = pd.DataFrame({
            "unit_id": [2] * 50,
            "cycle": list(range(50)),
            "rul": list(range(49, -1, -1)),
            "feat_normal": np.random.randn(50),
            "feat_shifted": np.random.randn(50) * 0.5 + 100,  # mean ~100, same scale
        })
        result = distribution_shift_analysis(train, val, shift_threshold=0.5)
        shifted_row = result[result["feature"] == "feat_shifted"]
        assert len(shifted_row) == 1
        assert shifted_row.iloc[0]["shift_flag"] == True


# ---------------------------------------------------------------------------
# Test: Feature Usability Inventory
# ---------------------------------------------------------------------------

class TestFeatureUsabilityInventory:
    """Test comprehensive feature usability assessment."""

    def test_inventory_produced(self, synthetic_train_val, feature_metadata_fixture):
        """Should produce an inventory for all features."""
        train, val = synthetic_train_val
        meta = feature_metadata_fixture
        nan_audit = nan_root_cause_analysis(train, val, meta)
        nan_audit = determine_nan_handling(nan_audit)
        const_audit = constant_feature_audit(train, val, meta)
        dup = detect_duplicate_features(train)

        inventory = generate_feature_usability_inventory(
            train, val, meta, nan_audit, const_audit, dup,
        )
        feat_cols = get_feature_columns(train)
        assert len(inventory) == len(feat_cols)
        assert "usable_for_phase4" in inventory.columns

    def test_constant_both_marked_no(self, synthetic_train_val, feature_metadata_fixture):
        """Features constant in both splits should be marked NO."""
        train, val = synthetic_train_val
        meta = feature_metadata_fixture
        nan_audit = nan_root_cause_analysis(train, val, meta)
        nan_audit = determine_nan_handling(nan_audit)
        const_audit = constant_feature_audit(train, val, meta)
        dup = detect_duplicate_features(train)

        inventory = generate_feature_usability_inventory(
            train, val, meta, nan_audit, const_audit, dup,
        )
        # sensor_5_mean_10 is constant in both
        s5_mean = inventory[inventory["feature_name"] == "sensor_5_mean_10"]
        assert len(s5_mean) == 1
        assert s5_mean.iloc[0]["usable_for_phase4"] == "NO"
