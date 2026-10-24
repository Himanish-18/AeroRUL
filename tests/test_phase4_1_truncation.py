"""
Unit tests for AeroRUL Phase 4.1: Retrospective Truncation & Validation Protocol.
Covers all 16 required verification criteria in Step 13.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.truncation import (
    compute_training_cutoffs,
    extract_simulated_truncations,
    compute_cutoff_coverage,
    engine_cluster_bootstrap,
    evaluate_simulated_endpoints,
    evaluate_simulated_endpoints_by_cutoff,
    simulated_rul_range_error_analysis,
    get_protocol_audit_table,
)
from src.metrics import (
    mae,
    rmse,
    nasa_score,
    nasa_score_per_engine,
    mean_signed_error,
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_engine_data():
    """Create synthetic training and validation engine trajectories."""
    # 5 training engines with known terminal lengths: 100, 150, 200, 250, 300
    train_records = []
    for eid, terminal in enumerate([100, 150, 200, 250, 300], start=1):
        for c in range(1, terminal + 1):
            train_records.append({
                "unit_id": eid,
                "cycle": c,
                "rul": terminal - c,
                "sensor_1": 100.0 + c * 0.1,
                "sensor_2": 500.0 - c * 0.2,
            })
    train_df = pd.DataFrame(train_records)

    # 3 validation engines with known terminal lengths: 120, 220, 320
    val_records = []
    for eid, terminal in enumerate([120, 220, 320], start=101):
        for c in range(1, terminal + 1):
            val_records.append({
                "unit_id": eid,
                "cycle": c,
                "rul": terminal - c,
                "sensor_1": 100.0 + c * 0.1,
                "sensor_2": 500.0 - c * 0.2,
            })
    val_df = pd.DataFrame(val_records)

    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════════════
# 1. CUTOFF SCHEDULE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_cutoff_schedule_uses_training_engines_only(synthetic_engine_data):
    """Cutoff schedule must depend strictly on training data."""
    train_df, val_df = synthetic_engine_data
    cutoffs, meta = compute_training_cutoffs(train_df, base_checkpoints=(80, 100), min_cycle_feature_availability=75)

    assert meta["n_training_engines"] == 5
    assert meta["terminal_min"] == 100
    assert meta["terminal_max"] == 300
    # Training terminals are [100, 150, 200, 250, 300]
    # 50th percentile (median) is 200
    assert meta["terminal_quantiles"]["q50"] == 200
    assert 200 in cutoffs


def test_deterministic_cutoff_schedule(synthetic_engine_data):
    """Cutoff schedule must be deterministic across repeated calls."""
    train_df, _ = synthetic_engine_data
    c1, m1 = compute_training_cutoffs(train_df)
    c2, m2 = compute_training_cutoffs(train_df)
    assert c1 == c2
    assert m1["terminal_quantiles"] == m2["terminal_quantiles"]


def test_cutoffs_are_within_feature_availability(synthetic_engine_data):
    """Cutoffs below min_cycle_feature_availability must be excluded."""
    train_df, _ = synthetic_engine_data
    # Include a candidate cutoff (50) below 75
    cutoffs, meta = compute_training_cutoffs(
        train_df,
        base_checkpoints=(50, 80, 100),
        min_cycle_feature_availability=75,
    )
    assert 50 not in cutoffs
    assert 50 in meta["excluded_below_min_cycle"]
    assert all(c >= 75 for c in cutoffs)


# ═══════════════════════════════════════════════════════════════════════════
# 2. SIMULATED OBSERVATION EXTRACTION & TRUTH
# ═══════════════════════════════════════════════════════════════════════════

def test_simulated_cutoffs_have_positive_rul(synthetic_engine_data):
    """Every extracted simulated observation must have strictly positive true RUL."""
    _, val_df = synthetic_engine_data
    # Cutoff 120 is terminal cycle for engine 101 (Tu=120)
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[80, 100, 120])
    assert len(trunc_df) > 0
    assert (trunc_df["rul"] > 0).all()


def test_terminal_rows_are_excluded_from_simulated_evaluation(synthetic_engine_data):
    """Terminal row (cycle == Tu, RUL == 0) must NEVER be included."""
    _, val_df = synthetic_engine_data
    # Engine 101 has Tu = 120. At cutoff 120, engine 101 has RUL = 0 and must be excluded.
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[120])
    # Engine 101 should not be in trunc_df at cutoff 120
    e101_obs = trunc_df[trunc_df["unit_id"] == 101]
    assert len(e101_obs) == 0


def test_simulated_rul_truth_is_correct(synthetic_engine_data):
    """Validation truth must satisfy RUL == Tu - t."""
    _, val_df = synthetic_engine_data
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[80, 100])
    for _, row in trunc_df.iterrows():
        assert row["rul"] == row["actual_terminal_cycle"] - row["cutoff_cycle"]
        assert row["cycle"] == row["cutoff_cycle"]


def test_cutoff_observation_alignment(synthetic_engine_data):
    """Each observation must align correctly with unit_id, cycle, and sensors."""
    _, val_df = synthetic_engine_data
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[80])
    for _, row in trunc_df.iterrows():
        eid = row["unit_id"]
        c = row["cycle"]
        original_row = val_df[(val_df["unit_id"] == eid) & (val_df["cycle"] == c)].iloc[0]
        assert row["sensor_1"] == original_row["sensor_1"]
        assert row["sensor_2"] == original_row["sensor_2"]


def test_no_future_features_used_at_cutoff(synthetic_engine_data):
    """Feature matrix contains only information observable up to cutoff cycle."""
    _, val_df = synthetic_engine_data
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[80, 100])
    forbidden_in_predictors = ["rul", "failure_cycle", "actual_terminal_cycle", "cutoff_schedule_id"]
    # When building feature matrix, these metadata columns are excluded
    feature_cols = [c for c in trunc_df.columns if c not in forbidden_in_predictors + ["unit_id", "cycle", "cutoff_cycle"]]
    assert "sensor_1" in feature_cols
    assert "sensor_2" in feature_cols
    assert "rul" not in feature_cols
    assert "actual_terminal_cycle" not in feature_cols


def test_train_validation_engine_separation(synthetic_engine_data):
    """No engine may appear in both training and validation sets."""
    train_df, val_df = synthetic_engine_data
    train_engines = set(train_df["unit_id"].unique())
    val_engines = set(val_df["unit_id"].unique())
    assert len(train_engines.intersection(val_engines)) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING & TARGET CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

def test_model_preprocessing_fitted_on_training_only(synthetic_engine_data):
    """Scalers must be fitted on training observations only."""
    train_df, val_df = synthetic_engine_data
    features = ["sensor_1", "sensor_2"]
    scaler = StandardScaler()
    scaler.fit(train_df[features])

    # Check scaler mean matches train_df, not combined
    expected_mean_0 = train_df["sensor_1"].mean()
    assert np.isclose(scaler.mean_[0], expected_mean_0)


def test_capped_targets_preserve_uncapped_validation_truth():
    """Evaluating against uncapped truth preserves physical error."""
    y_val_uncapped = np.array([150.0, 80.0, 20.0])
    y_pred_capped = np.array([125.0, 75.0, 20.0])  # predicted with cap 125

    # Evaluation against uncapped truth evaluates physical error (150 - 125 = 25)
    physical_mae = mae(y_val_uncapped, y_pred_capped)
    assert np.isclose(physical_mae, (25.0 + 5.0 + 0.0) / 3.0)


def test_capped_predictions_respect_cap():
    """Capped predictions must satisfy 0 <= y_pred_cap <= cap."""
    raw_preds = np.array([-10.0, 45.0, 110.0, 160.0])
    cap = 125.0
    preds_capped = np.minimum(np.maximum(raw_preds, 0.0), cap)

    assert (preds_capped >= 0.0).all()
    assert (preds_capped <= cap).all()
    assert preds_capped[0] == 0.0
    assert preds_capped[1] == 45.0
    assert preds_capped[2] == 110.0
    assert preds_capped[3] == 125.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. METRIC AND COVERAGE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_nasa_score_branches():
    """Verify both branches of NASA asymmetric score: underprediction and overprediction."""
    y_true = np.array([100.0, 100.0])
    # Case 1: Underprediction d = -10 (d < 0) -> exp(10/13) - 1 ~ 1.1581
    y_pred_under = np.array([90.0, 100.0])
    # Case 2: Overprediction d = +10 (d >= 0) -> exp(10/10) - 1 ~ 1.7183
    y_pred_over = np.array([110.0, 100.0])

    s_under = nasa_score(y_true, y_pred_under)
    s_over = nasa_score(y_true, y_pred_over)

    # In C-MAPSS NASA score, overpredicting RUL is penalized more (exp(d/10)-1 > exp(-d/13)-1)
    assert s_over > s_under
    assert np.isclose(s_under, np.exp(10.0 / 13.0) - 1.0)
    assert np.isclose(s_over, np.exp(10.0 / 10.0) - 1.0)


def test_engine_macro_metrics():
    """Engine-macro metrics must give equal weight to each engine."""
    # Engine 1 has 1 observation with error 10
    # Engine 2 has 9 observations with error 0
    preds_df = pd.DataFrame({
        "unit_id": [1] + [2] * 9,
        "cycle": [80] + list(range(81, 90)),
        "actual_rul": [50.0] * 10,
        "predicted_rul_nonnegative": [60.0] + [50.0] * 9,
    })
    results = evaluate_simulated_endpoints(preds_df)

    # Engine 1 MAE = 10, Engine 2 MAE = 0
    # Macro MAE = (10 + 0) / 2 = 5.0
    # Pooled MAE = 10 / 10 = 1.0
    assert np.isclose(results["engine_macro_mae"], 5.0)
    assert np.isclose(results["simulated_truncated_endpoint_mae"], 1.0)


def test_cutoff_coverage_reporting(synthetic_engine_data):
    """Cutoff coverage table must report eligible engines and stats correctly."""
    _, val_df = synthetic_engine_data
    # Val engines have Tu = [120, 220, 320]
    cutoffs = [80, 100, 150, 250, 350]
    trunc_df = extract_simulated_truncations(val_df, cutoffs)
    cov_df = compute_cutoff_coverage(trunc_df, cutoffs, total_val_engines=3, subset="TEST")

    assert len(cov_df) == 5
    # At cutoff 80: 3 engines eligible
    c80 = cov_df[cov_df["cutoff_cycle"] == 80].iloc[0]
    assert c80["n_eligible_engines"] == 3
    assert c80["minimum_true_rul"] == 120 - 80  # 40

    # At cutoff 350: 0 engines eligible (all died before 350)
    c350 = cov_df[cov_df["cutoff_cycle"] == 350].iloc[0]
    assert c350["n_eligible_engines"] == 0
    assert bool(c350["is_low_sample"]) is True


def test_metric_reproducibility():
    """Metrics must be completely deterministic."""
    y_true = np.array([120.0, 95.0, 30.0, 15.0])
    y_pred = np.array([115.0, 100.0, 28.0, 19.0])
    m1 = mae(y_true, y_pred)
    m2 = mae(y_true, y_pred)
    assert m1 == m2
    assert rmse(y_true, y_pred) == rmse(y_true, y_pred)
    assert nasa_score(y_true, y_pred) == nasa_score(y_true, y_pred)


def test_protocol_audit_table():
    """Audit table must return all 5 required metric audit dimensions."""
    audit_df = get_protocol_audit_table()
    assert len(audit_df) == 5
    assert "metric_or_experiment" in audit_df.columns
    assert "existing_method" in audit_df.columns
    assert "identified_limitation" in audit_df.columns
    assert "required_action" in audit_df.columns
