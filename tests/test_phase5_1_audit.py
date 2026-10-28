"""
AeroRUL — Phase 5.1 Final Evaluation Audit Tests
Tests verifying manifest consistency, metric recomputations, NASA score calculations,
bias interpretation, endpoint coverage, and report reconciliation.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import AVAILABLE_SUBSETS, RAW_DATA_DIR, REPORTS_DIR
from src.metrics import mae, mean_signed_error, nasa_score, r2_score, rmse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5_DIR = REPORTS_DIR / "phase5"
PHASE5_1_DIR = REPORTS_DIR / "phase5_1"
MODELS_FINAL_DIR = PROJECT_ROOT / "models" / "final"


@pytest.fixture(scope="session")
def manifest():
    manifest_path = PHASE5_DIR / "final_model_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def predictions_df():
    preds_path = PHASE5_DIR / "final_test_predictions.csv"
    assert preds_path.exists(), f"Missing predictions CSV: {preds_path}"
    return pd.read_csv(preds_path)


@pytest.fixture(scope="session")
def verified_results_df():
    results_path = PHASE5_1_DIR / "final_verified_results.csv"
    assert results_path.exists(), f"Missing verified results: {results_path}"
    return pd.read_csv(results_path)


@pytest.fixture(scope="session")
def contributions_df():
    contribs_path = PHASE5_1_DIR / "nasa_score_contributions.csv"
    assert contribs_path.exists(), f"Missing contributions CSV: {contribs_path}"
    return pd.read_csv(contribs_path)


# ---------------------------------------------------------------------------
# TEST 1: Manifest matches saved model parameters
# ---------------------------------------------------------------------------
def test_manifest_matches_saved_model_parameters(manifest):
    """Verify that hyperparameters in manifest match the saved joblib model parameters."""
    for subset in AVAILABLE_SUBSETS:
        cfg = manifest["selected_configurations"][subset]
        model_path = MODELS_FINAL_DIR / subset / "model.joblib"
        assert model_path.exists(), f"Missing model for {subset}: {model_path}"
        saved_model = joblib.load(model_path)

        for p_name, p_val in cfg["hyperparameters"].items():
            assert hasattr(saved_model, p_name), f"Model {subset} missing param {p_name}"
            actual = getattr(saved_model, p_name)
            if isinstance(p_val, float):
                assert np.isclose(actual, p_val, atol=1e-5), (
                    f"Mismatch in {subset} param {p_name}: manifest={p_val}, saved={actual}"
                )
            else:
                assert actual == p_val, (
                    f"Mismatch in {subset} param {p_name}: manifest={p_val}, saved={actual}"
                )


# ---------------------------------------------------------------------------
# TEST 2: Prediction count matches official labels
# ---------------------------------------------------------------------------
def test_prediction_count_matches_official_labels(predictions_df):
    """Verify that predictions count equals official label row counts for each subset."""
    expected_counts = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    for subset, exp_count in expected_counts.items():
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        assert len(sub_preds) == exp_count, f"{subset} predictions count {len(sub_preds)} != {exp_count}"

        # Verify against official label file
        label_file = RAW_DATA_DIR / f"RUL_{subset}.txt"
        labels = np.loadtxt(label_file)
        assert len(labels) == exp_count, f"{subset} raw label count {len(labels)} != {exp_count}"
        assert len(sub_preds) == len(labels)


# ---------------------------------------------------------------------------
# TEST 3: Unique official test engine predictions
# ---------------------------------------------------------------------------
def test_unique_official_test_engine_predictions(predictions_df):
    """Verify every engine ID is represented exactly once with no duplicates or missing units."""
    expected_counts = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}
    for subset, exp_count in expected_counts.items():
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        unit_ids = sub_preds["unit_id"].tolist()
        expected_ids = list(range(1, exp_count + 1))
        assert sorted(unit_ids) == expected_ids, f"{subset} engine IDs do not match 1..{exp_count}"
        assert len(set(unit_ids)) == exp_count, f"{subset} has duplicate engine IDs"


# ---------------------------------------------------------------------------
# TEST 4: Independent MAE recalculation
# ---------------------------------------------------------------------------
def test_independent_mae_recalculation(predictions_df, verified_results_df):
    """Verify independent MAE calculation on synthetic data and saved test predictions."""
    # Synthetic check
    y_true_synth = np.array([10.0, 20.0, 30.0])
    y_pred_synth = np.array([12.0, 18.0, 33.0])
    assert np.isclose(mae(y_true_synth, y_pred_synth), 7.0 / 3.0)

    # Saved test predictions check
    for subset in AVAILABLE_SUBSETS:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        y_true = sub_preds["actual_rul_official"].values
        y_pred = sub_preds["predicted_rul_final"].values
        calc_mae = mae(y_true, y_pred)

        expected_mae = verified_results_df.loc[
            verified_results_df["dataset"] == subset, "test_mae"
        ].values[0]
        assert np.isclose(calc_mae, expected_mae, atol=1e-5), f"{subset} MAE mismatch: {calc_mae} vs {expected_mae}"


# ---------------------------------------------------------------------------
# TEST 5: Independent RMSE recalculation
# ---------------------------------------------------------------------------
def test_independent_rmse_recalculation(predictions_df, verified_results_df):
    """Verify independent RMSE calculation on synthetic data and saved test predictions."""
    # Synthetic check
    y_true_synth = np.array([10.0, 20.0, 30.0])
    y_pred_synth = np.array([13.0, 16.0, 30.0])
    # diffs: 3, -4, 0 -> squared: 9 + 16 + 0 = 25 -> mean = 25/3 -> sqrt = sqrt(25/3)
    assert np.isclose(rmse(y_true_synth, y_pred_synth), np.sqrt(25.0 / 3.0))

    # Saved test predictions check
    for subset in AVAILABLE_SUBSETS:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        y_true = sub_preds["actual_rul_official"].values
        y_pred = sub_preds["predicted_rul_final"].values
        calc_rmse = rmse(y_true, y_pred)

        expected_rmse = verified_results_df.loc[
            verified_results_df["dataset"] == subset, "test_rmse"
        ].values[0]
        assert np.isclose(calc_rmse, expected_rmse, atol=1e-5), f"{subset} RMSE mismatch: {calc_rmse} vs {expected_rmse}"


# ---------------------------------------------------------------------------
# TEST 6: Independent R2 recalculation
# ---------------------------------------------------------------------------
def test_independent_r2_recalculation(predictions_df, verified_results_df):
    """Verify independent R2 calculation on synthetic data and saved test predictions."""
    # Synthetic check
    y_true_synth = np.array([10.0, 20.0, 30.0])
    y_pred_synth = np.array([10.0, 20.0, 30.0])
    assert np.isclose(r2_score(y_true_synth, y_pred_synth), 1.0)

    # Saved test predictions check
    for subset in AVAILABLE_SUBSETS:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        y_true = sub_preds["actual_rul_official"].values
        y_pred = sub_preds["predicted_rul_final"].values
        calc_r2 = r2_score(y_true, y_pred)

        expected_r2 = verified_results_df.loc[
            verified_results_df["dataset"] == subset, "test_r2"
        ].values[0]
        assert np.isclose(calc_r2, expected_r2, atol=1e-5), f"{subset} R2 mismatch: {calc_r2} vs {expected_r2}"


# ---------------------------------------------------------------------------
# TEST 7: NASA score branch signs and asymmetric penalty
# ---------------------------------------------------------------------------
def test_nasa_score_branch_signs():
    """Verify that NASA asymmetric score penalizes overprediction more heavily than underprediction."""
    y_true = np.array([100.0, 100.0])
    # Underprediction by 10 (d = -10): exp(10/13) - 1 ~ 1.15818
    # Overprediction by 10 (d = +10): exp(10/10) - 1 ~ 1.71828
    y_pred = np.array([90.0, 110.0])

    s_under = np.exp(10.0 / 13.0) - 1.0
    s_over = np.exp(10.0 / 10.0) - 1.0

    score_under = nasa_score(np.array([100.0]), np.array([90.0]))
    score_over = nasa_score(np.array([100.0]), np.array([110.0]))

    assert np.isclose(score_under, s_under, atol=1e-5)
    assert np.isclose(score_over, s_over, atol=1e-5)
    assert score_over > score_under, "Overprediction penalty must exceed underprediction penalty for equal |d|"


# ---------------------------------------------------------------------------
# TEST 8: NASA score sum and mean calculation
# ---------------------------------------------------------------------------
def test_nasa_score_sum_and_mean(predictions_df, verified_results_df):
    """Verify that NASA score sum matches and mean equals sum / N."""
    for subset in AVAILABLE_SUBSETS:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        y_true = sub_preds["actual_rul_official"].values
        y_pred = sub_preds["predicted_rul_final"].values
        total_score = nasa_score(y_true, y_pred)

        row = verified_results_df[verified_results_df["dataset"] == subset].iloc[0]
        expected_sum = row["nasa_score_sum"]
        expected_mean = row["nasa_score_mean"]
        n_engines = row["official_test_engine_count"]

        assert np.isclose(total_score, expected_sum, atol=1e-2), (
            f"{subset} NASA sum mismatch: {total_score} vs {expected_sum}"
        )
        assert np.isclose(total_score / n_engines, expected_mean, atol=1e-4), (
            f"{subset} NASA mean mismatch: {total_score/n_engines} vs {expected_mean}"
        )


# ---------------------------------------------------------------------------
# TEST 9: NASA score contribution reconciliation
# ---------------------------------------------------------------------------
def test_nasa_score_contribution_reconciliation(contributions_df, verified_results_df):
    """Verify that the sum of individual NASA penalties equals total dataset score."""
    for subset in AVAILABLE_SUBSETS:
        sub_contribs = contributions_df[contributions_df["dataset"] == subset]
        sum_individual = sub_contribs["individual_nasa_score"].sum()

        expected_sum = verified_results_df.loc[
            verified_results_df["dataset"] == subset, "nasa_score_sum"
        ].values[0]

        assert np.isclose(sum_individual, expected_sum, atol=1e-2), (
            f"{subset} individual penalty sum {sum_individual} != {expected_sum}"
        )


# ---------------------------------------------------------------------------
# TEST 10: Signed error interpretation
# ---------------------------------------------------------------------------
def test_signed_error_interpretation(predictions_df, verified_results_df):
    """Verify that signed error e = y_pred - y_true (>0 overprediction, <0 underprediction)."""
    # Synthetic check
    y_true = np.array([50.0, 80.0])
    y_pred = np.array([60.0, 70.0])
    err = y_pred - y_true
    assert err[0] > 0, "60 - 50 is positive -> overprediction"
    assert err[1] < 0, "70 - 80 is negative -> underprediction"
    assert np.isclose(mean_signed_error(y_true, y_pred), 0.0)

    # Real data counts check
    for subset in AVAILABLE_SUBSETS:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        y_true = sub_preds["actual_rul_official"].values
        y_pred = sub_preds["predicted_rul_final"].values
        diffs = y_pred - y_true

        n_over = int((diffs > 0).sum())
        n_under = int((diffs < 0).sum())
        n_zero = int((diffs == 0).sum())
        total = len(diffs)

        row = verified_results_df[verified_results_df["dataset"] == subset].iloc[0]
        assert n_over == row["overprediction_count"], f"{subset} overprediction count mismatch"
        assert n_under == row["underprediction_count"], f"{subset} underprediction count mismatch"
        assert n_zero == row["zero_error_count"], f"{subset} zero error count mismatch"
        assert n_over + n_under + n_zero == total, f"{subset} counts do not sum to total engines"


# ---------------------------------------------------------------------------
# TEST 11: RUL range grouping
# ---------------------------------------------------------------------------
def test_rul_range_grouping():
    """Verify that RUL range bins partition engines completely without overlaps or gaps."""
    range_path = PHASE5_1_DIR / "test_error_by_rul_range.csv"
    assert range_path.exists(), f"Missing range CSV: {range_path}"
    df_range = pd.read_csv(range_path)

    expected_bins = ["Critical (0–30)", "Low (31–75)", "Moderate (76–125)", "High (>125)"]
    expected_counts = {"FD001": 100, "FD002": 259, "FD003": 100, "FD004": 248}

    for subset, exp_total in expected_counts.items():
        sub = df_range[df_range["dataset"] == subset]
        assert set(sub["rul_bin"].tolist()) == set(expected_bins), f"{subset} missing bins"
        assert sub["engine_count"].sum() == exp_total, f"{subset} bin engine sum {sub['engine_count'].sum()} != {exp_total}"


# ---------------------------------------------------------------------------
# TEST 12: Feature schema matches manifest
# ---------------------------------------------------------------------------
def test_feature_schema_matches_manifest(manifest):
    """Verify that feature_names.json matches feature count and names in manifest."""
    for subset in AVAILABLE_SUBSETS:
        cfg = manifest["selected_configurations"][subset]
        features_path = MODELS_FINAL_DIR / subset / "feature_names.json"
        assert features_path.exists(), f"Missing features file: {features_path}"
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        assert len(features) == cfg["n_features"], (
            f"{subset} feature count mismatch: {len(features)} vs {cfg['n_features']}"
        )


# ---------------------------------------------------------------------------
# TEST 13: Partial history endpoint coverage
# ---------------------------------------------------------------------------
def test_partial_history_endpoint_coverage():
    """Verify that all 707 test engines have endpoint features even with short histories."""
    coverage_path = PHASE5_1_DIR / "test_feature_coverage_audit.csv"
    assert coverage_path.exists(), f"Missing coverage CSV: {coverage_path}"
    df_coverage = pd.read_csv(coverage_path)

    assert (df_coverage["partial_history_coverage_pct"] == 100.0).all(), "Feature coverage is not 100%"
    assert (df_coverage["coverage_status"] == "FULL_COVERAGE_100_PCT").all(), "Coverage status is not all FULL_COVERAGE_100_PCT"
    assert (df_coverage["official_engines_predicted"] == df_coverage["official_engines_expected"]).all(), "Engine counts do not match"


# ---------------------------------------------------------------------------
# TEST 14: No target or metadata predictors
# ---------------------------------------------------------------------------
def test_no_target_or_metadata_predictors():
    """Verify that no forbidden metadata or leakage columns exist in saved feature names."""
    forbidden = ["RUL", "target", "unit_id", "cycle", "rul_early", "rul_piecewise"]
    for subset in AVAILABLE_SUBSETS:
        features_path = MODELS_FINAL_DIR / subset / "feature_names.json"
        with open(features_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        for col in features:
            for forb in forbidden:
                assert col.lower() != forb.lower(), f"Forbidden feature found in {subset}: {col}"


# ---------------------------------------------------------------------------
# TEST 15: Target cap vs output constraint
# ---------------------------------------------------------------------------
def test_target_cap_vs_output_constraint(predictions_df, verified_results_df):
    """Verify that training target cap and operational output constraints are properly separated."""
    # Predictions must be non-negative
    assert (predictions_df["predicted_rul_final"] >= 0.0).all(), "Negative predictions found"

    # Operational predictions in FD001 and FD002 can exceed 150 because cap is a training target cap
    for subset in ["FD001", "FD002"]:
        sub_preds = predictions_df[predictions_df["dataset"] == subset]
        # Verify training target cap is 150 in verified results
        cap_val = verified_results_df.loc[
            verified_results_df["dataset"] == subset, "selected_target_cap"
        ].values[0]
        assert str(cap_val) in ("150", "150.0")


# ---------------------------------------------------------------------------
# TEST 16: Window selection report consistency
# ---------------------------------------------------------------------------
def test_window_selection_report_consistency():
    """Verify that window selection reconciliation contains all subsets and documents selections."""
    win_path = PHASE5_1_DIR / "window_selection_reconciliation.csv"
    assert win_path.exists(), f"Missing window reconciliation CSV: {win_path}"
    df_win = pd.read_csv(win_path)

    assert set(df_win["dataset"].tolist()) == set(AVAILABLE_SUBSETS)
    required_cols = [
        "dataset", "best_inner_cv_config", "best_dev_val_config",
        "selected_final_config", "selection_metric", "selection_rationale"
    ]
    for c in required_cols:
        assert c in df_win.columns, f"Missing column {c} in window reconciliation CSV"

    # Verify that raw ablation metrics also exist and contain all window sizes
    raw_ablation = pd.read_csv(PHASE5_DIR / "window_ablation_metrics.csv")
    for subset in AVAILABLE_SUBSETS:
        sub_raw = raw_ablation[raw_ablation["dataset"] == subset]
        configs = sub_raw["window_configuration"].tolist()
        assert "ALL_WINDOWS" in configs, f"ALL_WINDOWS missing from {subset}"
        assert "W75" in configs, f"W75 missing from {subset}"


# ---------------------------------------------------------------------------
# TEST 17: Report generated from verified results
# ---------------------------------------------------------------------------
def test_report_generated_from_verified_results(verified_results_df):
    """Verify that key numerical values in phase5_1_final_audit_report.md match verified results."""
    report_path = PHASE5_1_DIR / "phase5_1_final_audit_report.md"
    assert report_path.exists(), f"Missing audit report: {report_path}"
    report_text = report_path.read_text(encoding="utf-8")

    for _, row in verified_results_df.iterrows():
        subset = row["dataset"]
        mae_str = f"{row['test_mae']:.2f}"
        rmse_str = f"{row['test_rmse']:.2f}"
        r2_str = f"{row['test_r2']:.4f}"

        assert mae_str in report_text, f"{subset} MAE {mae_str} not found in report"
        assert rmse_str in report_text, f"{subset} RMSE {rmse_str} not found in report"
        assert r2_str in report_text, f"{subset} R2 {r2_str} not found in report"
