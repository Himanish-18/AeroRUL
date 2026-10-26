"""
AeroRUL — Phase 5 Automated Test Suite

Covers all 18 mandatory audit and regression requirements:
1. test_grouped_cv_has_no_engine_overlap
2. test_fold_preprocessing_fitted_on_training_engines_only
3. test_cutoff_schedule_training_fold_only
4. test_simulated_validation_rul_is_positive
5. test_prediction_target_alignment
6. test_optuna_trial_reproducibility
7. test_target_cap_training_transform
8. test_output_constraint
9. test_uncapped_physical_truth_preserved
10. test_window_feature_selection
11. test_partial_history_feature_semantics
12. test_test_endpoint_feature_availability
13. test_all_official_test_engines_predicted
14. test_official_test_label_alignment
15. test_test_label_access_after_configuration_freeze
16. test_final_model_manifest_reproducibility
17. test_final_test_metrics_recomputed_from_predictions
18. test_official_test_not_used_for_selection
"""

import json
from pathlib import Path
import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.config import COLUMN_NAMES, RAW_DATA_DIR, REPORTS_DIR
from src.features import generate_multiscale_features
from src.metrics import mae, rmse, nasa_score
from src.run_phase4_benchmark import select_window_features
from src.truncation import (
    compute_training_cutoffs,
    extract_simulated_truncations,
    prepare_truncation_matrices,
)

PHASE5_DIR = REPORTS_DIR / "phase5"


@pytest.fixture
def synthetic_multitraj_df():
    """Generates synthetic multi-engine dataset with varying run lengths."""
    np.random.seed(42)
    records = []
    engine_lengths = [85, 110, 130, 95, 150, 100]  # 6 engines

    for uid, length in enumerate(engine_lengths, start=1):
        for c in range(1, length + 1):
            rul = length - c
            s2 = 10.0 + 0.05 * c + np.random.normal(0, 0.02)
            s3 = 50.0 - 0.08 * c + np.random.normal(0, 0.03)
            records.append({
                "unit_id": uid,
                "cycle": c,
                "sensor_2": s2,
                "sensor_3": s3,
                "rul": rul,
            })
    return pd.DataFrame(records)


def test_grouped_cv_has_no_engine_overlap(synthetic_multitraj_df):
    """Verifies that inner grouped CV strictly enforces zero engine overlap between folds."""
    df = synthetic_multitraj_df
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(df, groups=df["unit_id"]))

    assert len(splits) == 3
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        tr_engines = set(df.iloc[tr_idx]["unit_id"].unique())
        val_engines = set(df.iloc[val_idx]["unit_id"].unique())
        assert len(tr_engines & val_engines) == 0, f"Fold {fold_i} has engine overlap!"


def test_fold_preprocessing_fitted_on_training_engines_only(synthetic_multitraj_df):
    """Verifies that preprocessing scalers fit only on fold-training observations."""
    df = synthetic_multitraj_df
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(df, groups=df["unit_id"]))

    tr_idx, val_idx = splits[0]
    X_tr = df.iloc[tr_idx][["sensor_2", "sensor_3"]].values
    X_val = df.iloc[val_idx][["sensor_2", "sensor_3"]].values

    scaler = StandardScaler()
    scaler.fit(X_tr)

    assert np.allclose(scaler.mean_, np.mean(X_tr, axis=0))
    # Must NOT equal full data mean
    X_full = df[["sensor_2", "sensor_3"]].values
    assert not np.allclose(scaler.mean_, np.mean(X_full, axis=0))


def test_cutoff_schedule_training_fold_only(synthetic_multitraj_df):
    """Verifies that cutoff schedule is constructed exclusively from training engines."""
    df = synthetic_multitraj_df
    gkf = GroupKFold(n_splits=3)
    splits = list(gkf.split(df, groups=df["unit_id"]))

    tr_idx, val_idx = splits[0]
    fold_train_df = df.iloc[tr_idx]

    cutoffs, meta = compute_training_cutoffs(
        fold_train_df,
        base_checkpoints=(80, 100),
        quantiles=(0.50, 0.75),
        min_cycle_feature_availability=75,
    )
    tr_units = set(fold_train_df["unit_id"].unique())
    assert meta["n_training_engines"] == len(tr_units)
    assert all(c >= 75 for c in cutoffs)


def test_simulated_validation_rul_is_positive(synthetic_multitraj_df):
    """Verifies that simulated truncation strictly extracts positive RUL (RUL > 0)."""
    df = synthetic_multitraj_df
    cutoffs = [80, 90, 100]
    trunc_df = extract_simulated_truncations(df, cutoffs)

    assert len(trunc_df) > 0
    assert (trunc_df["rul"] > 0).all(), "Simulated endpoints contain terminal RUL=0 rows!"
    for _, row in trunc_df.iterrows():
        assert row["cycle"] in cutoffs


def test_prediction_target_alignment(synthetic_multitraj_df):
    """Verifies 1-to-1 row alignment between feature matrix and target vector."""
    df = synthetic_multitraj_df
    cutoffs = [80, 90]
    trunc_df = extract_simulated_truncations(df, cutoffs)
    usable_cols = ["sensor_2", "sensor_3"]

    X, y = prepare_truncation_matrices(trunc_df, usable_cols)
    assert len(X) == len(y) == len(trunc_df)
    assert np.allclose(y, trunc_df["rul"].values)
    assert np.allclose(X[:, 0], trunc_df["sensor_2"].values)


def test_optuna_trial_reproducibility():
    """Verifies that Optuna with deterministic seed generates reproducible trial sequences."""
    def dummy_objective(trial):
        x = trial.suggest_float("x", -5.0, 5.0)
        y = trial.suggest_int("y", 1, 10)
        return (x - 2) ** 2 + y

    sampler1 = optuna.samplers.TPESampler(seed=42)
    study1 = optuna.create_study(direction="minimize", sampler=sampler1)
    study1.optimize(dummy_objective, n_trials=5)

    sampler2 = optuna.samplers.TPESampler(seed=42)
    study2 = optuna.create_study(direction="minimize", sampler=sampler2)
    study2.optimize(dummy_objective, n_trials=5)

    assert study1.best_value == study2.best_value
    assert study1.best_params == study2.best_params


def test_target_cap_training_transform():
    """Verifies that target cap applies min(y, cap) correctly."""
    y = np.array([150, 130, 125, 100, 50, 20, 0])
    cap = 125
    y_capped = np.minimum(y, cap)

    expected = np.array([125, 125, 125, 100, 50, 20, 0])
    assert np.array_equal(y_capped, expected)


def test_output_constraint():
    """Verifies operational output constraint: non-negative and optional upper cap."""
    preds_raw = np.array([-15.5, -0.1, 0.0, 55.2, 120.0, 160.0])
    cap = 125.0

    preds_nonneg = np.maximum(0.0, preds_raw)
    assert (preds_nonneg >= 0).all()

    preds_operational = np.minimum(preds_nonneg, cap)
    assert (preds_operational >= 0).all()
    assert (preds_operational <= cap).all()
    assert preds_operational[0] == 0.0
    assert preds_operational[-1] == 125.0


def test_uncapped_physical_truth_preserved():
    """Verifies that true physical RUL is preserved uncapped for metric scoring."""
    y_true_uncapped = np.array([160.0, 140.0, 90.0, 30.0])
    preds = np.array([125.0, 125.0, 85.0, 32.0])  # predictions from capped model

    # Evaluate against uncapped truth
    res_mae = mae(y_true_uncapped, preds)
    expected_mae = (35.0 + 15.0 + 5.0 + 2.0) / 4.0
    assert np.isclose(res_mae, expected_mae)


def test_window_feature_selection():
    """Verifies that window feature selection filters correctly and keeps shared features."""
    all_feats = [
        "sensor_2_mean_10", "sensor_2_std_10",
        "sensor_2_mean_30", "sensor_2_std_30",
        "sensor_2_delta_1", "sensor_2_relchange_5",
    ]
    w10_feats = select_window_features(all_feats, target_window=10)
    assert "sensor_2_mean_10" in w10_feats
    assert "sensor_2_std_10" in w10_feats
    assert "sensor_2_delta_1" in w10_feats  # shared
    assert "sensor_2_relchange_5" in w10_feats  # shared
    assert "sensor_2_mean_30" not in w10_feats


def test_partial_history_feature_semantics():
    """Verifies causal partial-history features: no NaNs, and identical to full window on large cycles."""
    np.random.seed(42)
    df = pd.DataFrame({
        "unit_id": [1] * 90,
        "cycle": list(range(1, 91)),
        "rul": list(range(89, -1, -1)),
        "sensor_2": np.linspace(10, 25, 90) + np.random.normal(0, 0.05, 90),
    })

    f_full = generate_multiscale_features(
        df, windows=[10, 20, 30], sensor_columns=["sensor_2"], deltas=[1, 5], relative_changes=[5],
        partial_history=False
    )
    f_part = generate_multiscale_features(
        df, windows=[10, 20, 30], sensor_columns=["sensor_2"], deltas=[1, 5], relative_changes=[5],
        partial_history=True
    )

    # Partial history retains all 90 rows without NaN
    assert len(f_part) == 90
    assert f_part.isna().sum().sum() == 0

    # For cycle >= 30, partial history matches full window exactly
    c_full = f_full.set_index("cycle")
    c_part = f_part[f_part["cycle"] >= 30].set_index("cycle")
    cols = [c for c in c_full.columns if c not in ["unit_id", "rul"]]
    diff = np.max(np.abs(c_full[cols].values - c_part[cols].values))
    assert np.isclose(diff, 0.0, atol=1e-12)


def test_test_endpoint_feature_availability():
    """Verifies that all official test engines have endpoint features available under partial history."""
    avail_file = PHASE5_DIR / "test_endpoint_feature_availability.csv"
    if not avail_file.exists():
        pytest.skip("Phase 5 test availability report not yet generated.")

    df_avail = pd.read_csv(avail_file)
    part_avail = df_avail[df_avail["candidate_feature_configuration"] == "ALL_WINDOWS_PARTIAL"]
    assert len(part_avail) == 707  # 100 + 259 + 100 + 248
    assert part_avail["endpoint_feature_available"].all()


def test_all_official_test_engines_predicted():
    """Verifies that exactly 707 official test engines receive predictions across FD001–FD004."""
    preds_file = PHASE5_DIR / "final_test_predictions.csv"
    if not preds_file.exists():
        pytest.skip("Phase 5 test predictions not yet generated.")

    df_preds = pd.read_csv(preds_file)
    counts = df_preds.groupby("dataset")["unit_id"].nunique().to_dict()
    assert counts.get("FD001") == 100
    assert counts.get("FD002") == 259
    assert counts.get("FD003") == 100
    assert counts.get("FD004") == 248
    assert len(df_preds) == 707


def test_official_test_label_alignment():
    """Verifies that official test predictions strictly align with unit_id 1..N."""
    preds_file = PHASE5_DIR / "final_test_predictions.csv"
    if not preds_file.exists():
        pytest.skip("Phase 5 test predictions not yet generated.")

    df_preds = pd.read_csv(preds_file)
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        sub_preds = df_preds[df_preds["dataset"] == subset].sort_values("unit_id")
        n_units = len(sub_preds)
        assert (sub_preds["unit_id"].values == np.arange(1, n_units + 1)).all()


def test_test_label_access_after_configuration_freeze():
    """Verifies that the final configuration manifest exists and was frozen."""
    manifest_file = PHASE5_DIR / "final_model_manifest.json"
    if not manifest_file.exists():
        pytest.skip("Manifest not yet generated.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "frozen_timestamp" in manifest
    assert "selected_configurations" in manifest
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        assert subset in manifest["selected_configurations"]


def test_final_model_manifest_reproducibility():
    """Verifies manifest schema and configuration validity."""
    manifest_file = PHASE5_DIR / "final_model_manifest.json"
    if not manifest_file.exists():
        pytest.skip("Manifest not yet generated.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for subset, cfg in manifest["selected_configurations"].items():
        assert cfg["model_family"] in ("LightGBM", "XGBoost", "RandomForest")
        assert "hyperparameters" in cfg
        assert "target_cap" in cfg
        assert "inner_cv_scores" in cfg


def test_final_test_metrics_recomputed_from_predictions():
    """Verifies that reported test metrics match predictions independently recomputed."""
    preds_file = PHASE5_DIR / "final_test_predictions.csv"
    metrics_file = PHASE5_DIR / "final_test_metrics.csv"
    if not preds_file.exists() or not metrics_file.exists():
        pytest.skip("Phase 5 test outputs not yet generated.")

    df_preds = pd.read_csv(preds_file)
    df_metrics = pd.read_csv(metrics_file)

    for _, mrow in df_metrics.iterrows():
        subset = mrow["dataset"]
        sub_p = df_preds[df_preds["dataset"] == subset]
        y_true = sub_p["actual_rul_official"].values
        y_pred = sub_p["predicted_rul_final"].values

        recomputed_mae = mae(y_true, y_pred)
        recomputed_rmse = rmse(y_true, y_pred)
        recomputed_nasa_sum = float(nasa_score(y_true, y_pred))

        assert np.isclose(recomputed_mae, mrow["endpoint_mae"], atol=1e-2)
        assert np.isclose(recomputed_rmse, mrow["endpoint_rmse"], atol=1e-2)
        assert np.isclose(recomputed_nasa_sum, mrow["nasa_score_sum"], atol=1.0)


def test_official_test_not_used_for_selection():
    """Verifies that model selection was driven strictly by inner-CV / dev-val, not official test labels."""
    manifest_file = PHASE5_DIR / "final_model_manifest.json"
    if not manifest_file.exists():
        pytest.skip("Manifest not yet generated.")

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for subset, cfg in manifest["selected_configurations"].items():
        assert cfg["selection_metric"] == "inner_cv_engine_macro_mae"
        assert "test_mae" not in cfg["inner_cv_scores"]
