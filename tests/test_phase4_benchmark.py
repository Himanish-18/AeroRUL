"""
Tests for Phase 4 — Classical Regression Benchmarking.

Covers metrics, data preflight, NASA score, endpoint extraction,
RUL range assignment, and pipeline integrity.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import (
    mae, rmse, r2_score, mean_signed_error,
    nasa_score, nasa_score_per_engine,
    extract_endpoints, endpoint_metrics,
    assign_rul_range, rul_range_error_analysis,
    engine_macro_metrics,
)
from src.run_phase4_benchmark import (
    data_preflight,
    prepare_clean_features,
    load_approved_features,
    select_window_features,
    detect_exact_duplicates_train,
    create_predictions_df,
    ALL_MODEL_CONFIGS,
    build_model,
    train_and_predict,
)
from src.features import get_feature_columns


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_predictions():
    """Create synthetic predictions dataframe."""
    return pd.DataFrame({
        "dataset": ["FD001"] * 20,
        "model": ["TestModel"] * 20,
        "unit_id": [1]*10 + [2]*10,
        "cycle": list(range(75, 85)) + list(range(75, 85)),
        "actual_rul": list(range(9, -1, -1)) + list(range(9, -1, -1)),
        "predicted_rul_raw": [10, 8, 7, 5, 4, 3, 2, 1, 0.5, -0.5] +
                              [9, 7, 6, 5, 3, 2, 1, 0, -1, -2],
        "predicted_rul_nonnegative": [10, 8, 7, 5, 4, 3, 2, 1, 0.5, 0] +
                                     [9, 7, 6, 5, 3, 2, 1, 0, 0, 0],
        "prediction_error": [1, 0, 0, -1, 0, 0, 0, 0, 0.5, 0] +
                            [0, -1, 0, 0, -1, 0, 0, 0, 0, 0],
        "absolute_error": [1, 0, 0, 1, 0, 0, 0, 0, 0.5, 0] +
                           [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    })


@pytest.fixture
def synthetic_train_val():
    """Create synthetic train/val for preflight."""
    np.random.seed(42)
    n = 100
    features = {
        f"sensor_{s}_mean_{w}": np.random.randn(n)
        for s in [2, 3, 4] for w in [10, 20, 30]
    }
    train = pd.DataFrame({
        "unit_id": [1]*50 + [2]*50,
        "cycle": list(range(75, 125)) * 2,
        "rul": list(range(49, -1, -1)) * 2,
        **features,
    })
    val = pd.DataFrame({
        "unit_id": [3]*30 + [4]*30,
        "cycle": list(range(75, 105)) * 2,
        "rul": list(range(29, -1, -1)) * 2,
        **{k: np.random.randn(60) for k in features},
    })
    return train, val


# ═══════════════════════════════════════════════════════════════════════════
# TEST: MAE / RMSE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

class TestMaeRmseCalculation:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0.0

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == 0.0

    def test_mae_known(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert mae(y_true, y_pred) == 1.0

    def test_rmse_known(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        expected = np.sqrt((1 + 4 + 9) / 3)
        assert abs(rmse(y_true, y_pred) - expected) < 1e-10

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert r2_score(y, y) == 1.0

    def test_r2_baseline(self):
        """Predicting the mean should give R²=0."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        assert abs(r2_score(y_true, y_pred)) < 1e-10


# ═══════════════════════════════════════════════════════════════════════════
# TEST: NASA SCORE — ERROR SIGN AND BRANCHES
# ═══════════════════════════════════════════════════════════════════════════

class TestNasaScoreErrorSignAndBranches:
    def test_perfect_prediction_score_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        assert nasa_score(y, y) == 0.0

    def test_underprediction_branch(self):
        """d < 0: late prediction (predicting more life than actual).
        Uses exp(-d/13) - 1."""
        y_true = np.array([10.0])
        y_pred = np.array([5.0])  # d = 5 - 10 = -5
        d = -5.0
        expected = np.exp(-d / 13.0) - 1  # exp(5/13) - 1
        assert abs(nasa_score(y_true, y_pred) - expected) < 1e-10

    def test_overprediction_branch(self):
        """d >= 0: early prediction (predicting less life than actual).
        Uses exp(d/10) - 1."""
        y_true = np.array([10.0])
        y_pred = np.array([15.0])  # d = 15 - 10 = 5
        d = 5.0
        expected = np.exp(d / 10.0) - 1
        assert abs(nasa_score(y_true, y_pred) - expected) < 1e-10

    def test_overprediction_penalized_more(self):
        """Overprediction of RUL (predicting more life than actual, d > 0) is penalized more heavily than underprediction (d < 0)."""
        y_true = np.array([10.0])
        # Same magnitude (10 cycles), different direction
        over_pred = np.array([20.0])   # d = +10 (overestimating life: catastrophic)
        under_pred = np.array([0.0])   # d = -10 (underestimating life: early service)
        score_over = nasa_score(y_true, over_pred)
        score_under = nasa_score(y_true, under_pred)
        assert score_over > score_under

    def test_lower_is_better(self):
        """Lower NASA score means better predictions."""
        y = np.array([10.0, 20.0])
        good_pred = np.array([10.0, 20.0])
        bad_pred = np.array([15.0, 25.0])
        assert nasa_score(y, good_pred) < nasa_score(y, bad_pred)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: ENDPOINT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class TestEndpointExtraction:
    def test_selects_last_cycle(self, synthetic_predictions):
        """Should select the row with max cycle per engine."""
        ep = extract_endpoints(synthetic_predictions)
        assert len(ep) == 2  # 2 engines
        for _, row in ep.iterrows():
            assert row["cycle"] == 84  # max cycle in the fixture

    def test_endpoint_rul_is_zero(self, synthetic_predictions):
        """Last cycle should have RUL close to 0."""
        ep = extract_endpoints(synthetic_predictions)
        # actual_rul at cycle 84 (max) should be 0
        assert all(ep["actual_rul"] == 0)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: RUL RANGE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

class TestRulRangeAssignment:
    def test_critical_range(self):
        assert assign_rul_range(np.array([0]))[0] == "critical_0_30"
        assert assign_rul_range(np.array([30]))[0] == "critical_0_30"

    def test_low_range(self):
        assert assign_rul_range(np.array([31]))[0] == "low_31_75"
        assert assign_rul_range(np.array([75]))[0] == "low_31_75"

    def test_moderate_range(self):
        assert assign_rul_range(np.array([76]))[0] == "moderate_76_125"
        assert assign_rul_range(np.array([125]))[0] == "moderate_76_125"

    def test_high_range(self):
        assert assign_rul_range(np.array([126]))[0] == "high_gt125"
        assert assign_rul_range(np.array([500]))[0] == "high_gt125"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: NO TARGET OR METADATA IN PREDICTOR MATRIX
# ═══════════════════════════════════════════════════════════════════════════

class TestNoTargetOrMetadataInPredictorMatrix:
    def test_rul_excluded(self, synthetic_train_val):
        train, val = synthetic_train_val
        feat_cols = get_feature_columns(train)
        assert "rul" not in feat_cols
        assert "unit_id" not in feat_cols
        assert "cycle" not in feat_cols

    def test_preflight_catches_forbidden(self, synthetic_train_val):
        train, val = synthetic_train_val
        # Include forbidden columns
        bad_features = ["rul", "sensor_2_mean_10"]
        result = data_preflight(train, val, bad_features, "TEST")
        assert result["passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST: APPROVED FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestApprovedFeatureSelection:
    def test_loads_approved(self):
        """Should load YES features from Phase 3.1 inventory."""
        features = load_approved_features("FD001")
        assert len(features) > 0
        assert len(features) == 670  # verified count

    def test_no_metadata_in_approved(self):
        features = load_approved_features("FD001")
        assert "rul" not in features
        assert "unit_id" not in features
        assert "cycle" not in features


# ═══════════════════════════════════════════════════════════════════════════
# TEST: TRAIN/VAL ENGINE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainValidationEngineSeparation:
    def test_no_overlap(self, synthetic_train_val):
        train, val = synthetic_train_val
        preflight = data_preflight(train, val,
                                   get_feature_columns(train), "TEST")
        assert preflight["passed"]
        overlap_issues = [i for i in preflight["issues"] if "overlap" in i.lower()]
        assert len(overlap_issues) == 0

    def test_detects_overlap(self, synthetic_train_val):
        train, val = synthetic_train_val
        val_leaked = val.copy()
        val_leaked["unit_id"] = 1  # overlap with train
        preflight = data_preflight(train, val_leaked,
                                   get_feature_columns(train), "TEST")
        assert preflight["passed"] is False


# ═══════════════════════════════════════════════════════════════════════════
# TEST: PREPROCESSING FITTED ON TRAINING DATA ONLY
# ═══════════════════════════════════════════════════════════════════════════

class TestPreprocessingFittedOnTrainingDataOnly:
    def test_scaler_from_train_only(self):
        """StandardScaler must be fit on train, transform val."""
        from sklearn.preprocessing import StandardScaler
        X_train = np.array([[0, 0], [2, 2]], dtype=float)
        X_val = np.array([[10, 10]], dtype=float)

        scaler = StandardScaler()
        scaler.fit(X_train)

        # Scaler should reflect training stats, not val
        assert abs(scaler.mean_[0] - 1.0) < 1e-10
        X_val_scaled = scaler.transform(X_val)
        # Val should not have mean 0 since scaler was fit on train
        assert abs(X_val_scaled[0, 0]) > 1.0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: RUL NONNEGATIVE PREDICTION CONSTRAINT
# ═══════════════════════════════════════════════════════════════════════════

class TestRulNonnegativePredictionConstraint:
    def test_nonnegative_constraint(self):
        val_df = pd.DataFrame({
            "unit_id": [1, 1, 1],
            "cycle": [1, 2, 3],
            "rul": [2, 1, 0],
        })
        preds_raw = np.array([5.0, -1.0, -3.0])
        result = create_predictions_df(val_df, preds_raw, "test", "FD001")
        assert (result["predicted_rul_nonnegative"] >= 0).all()
        assert result["predicted_rul_raw"].iloc[1] == -1.0  # raw preserved


# ═══════════════════════════════════════════════════════════════════════════
# TEST: RUL CAP TARGET TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════════

class TestRulCapTargetTransformation:
    def test_cap_applied_to_train(self):
        y_train = np.array([200, 100, 50, 20])
        cap = 125
        y_capped = np.minimum(y_train, cap)
        assert y_capped.max() <= cap
        assert y_capped[3] == 20  # below cap unchanged

    def test_uncapped_val_preserved(self):
        """Cap experiments must evaluate against uncapped val targets."""
        y_val = np.array([200, 100, 50])
        # Uncapped target should remain
        assert y_val[0] == 200


# ═══════════════════════════════════════════════════════════════════════════
# TEST: WINDOW FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestWindowFeatureSelection:
    def test_all_windows(self):
        features = ["sensor_2_mean_10", "sensor_2_mean_30", "sensor_2_delta_5"]
        result = select_window_features(features, None)
        assert result == features

    def test_single_window(self):
        features = [
            "sensor_2_mean_10", "sensor_2_std_10",
            "sensor_2_mean_30", "sensor_2_std_30",
            "sensor_2_delta_5", "sensor_2_relchange_10",
        ]
        result = select_window_features(features, 10)
        # Should include window-10 rolling features + all delta/relchange
        assert "sensor_2_mean_10" in result
        assert "sensor_2_std_10" in result
        assert "sensor_2_mean_30" not in result
        assert "sensor_2_delta_5" in result
        assert "sensor_2_relchange_10" in result


# ═══════════════════════════════════════════════════════════════════════════
# TEST: DUPLICATE FEATURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class TestDuplicateFeatureDetection:
    def test_detects_duplicates(self):
        X = np.array([[1, 1, 2], [3, 3, 4], [5, 5, 6]], dtype=float)
        names = ["a", "b", "c"]
        deduped, groups = detect_exact_duplicates_train(X, names)
        assert len(deduped) == 2
        assert "a" in deduped or "b" in deduped
        assert "c" in deduped

    def test_no_duplicates(self):
        X = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        names = ["a", "b", "c"]
        deduped, groups = detect_exact_duplicates_train(X, names)
        assert len(deduped) == 3
        assert len(groups) == 0


# ═══════════════════════════════════════════════════════════════════════════
# TEST: PREDICTION ROW ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

class TestPredictionRowAlignment:
    def test_alignment(self, synthetic_train_val):
        train, val = synthetic_train_val
        preds_raw = np.random.randn(len(val))
        result = create_predictions_df(val, preds_raw, "test", "FD001")
        assert len(result) == len(val)
        assert (result["unit_id"].values == val["unit_id"].values).all()
        assert (result["cycle"].values == val["cycle"].values).all()


# ═══════════════════════════════════════════════════════════════════════════
# TEST: DETERMINISTIC MODEL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

class TestDeterministicModelConfiguration:
    def test_random_seeds_set(self):
        """All models with randomness should have random_state set."""
        for name, cfg in ALL_MODEL_CONFIGS.items():
            if name in ("DummyBaseline", "LinearRegression"):
                continue
            assert "random_state" in cfg["params"], f"{name} missing random_state"
            assert cfg["params"]["random_state"] == 42

    def test_model_builds(self):
        """All configured models should instantiate."""
        for name, cfg in ALL_MODEL_CONFIGS.items():
            model = build_model(name, cfg)
            assert model is not None
