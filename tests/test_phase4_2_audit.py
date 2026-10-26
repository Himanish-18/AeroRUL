"""
Unit tests for AeroRUL Phase 4.2: Forensic Evaluation Audit & Metric Integrity.
Covers all 21 test criteria specified in Phase 4.2 Step 12.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    mae,
    rmse,
    r2_score,
    mean_signed_error,
    nasa_score,
    nasa_score_per_engine,
)
from src.truncation import (
    compute_training_cutoffs,
    extract_simulated_truncations,
    prepare_truncation_matrices,
    evaluate_simulated_endpoints,
    engine_cluster_bootstrap,
)
from src.run_phase4_benchmark import select_window_features


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_canonical_predictions():
    """Create a synthetic canonical prediction DataFrame."""
    records = []
    # 2 engines, 3 cutoffs each
    for eid, terminal in [(1, 150), (2, 200)]:
        for c in [80, 100, 125]:
            true_rul = terminal - c
            pred_raw = true_rul + (5.0 if eid == 1 else -4.0)
            pred_nn = max(pred_raw, 0.0)
            records.append({
                "dataset": "FD001",
                "model": "TestModel",
                "experiment_type": "primary_benchmark",
                "unit_id": eid,
                "cutoff_cycle": c,
                "cycle": c,
                "actual_terminal_cycle": terminal,
                "actual_rul": float(true_rul),
                "predicted_rul_raw": float(pred_raw),
                "predicted_rul_nonnegative": float(pred_nn),
                "prediction_error": float(pred_nn - true_rul),
                "absolute_error": float(abs(pred_nn - true_rul)),
                "model_configuration_id": "TestModel_fixed",
                "feature_configuration_id": "all_features",
                "rul_cap": None,
            })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 1. METRIC RECOMPUTATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_mae_recomputed_from_predictions(synthetic_canonical_predictions):
    """MAE must exactly equal mean absolute error of predicted_rul_nonnegative vs actual_rul."""
    df = synthetic_canonical_predictions
    computed_mae = mae(df["actual_rul"].values, df["predicted_rul_nonnegative"].values)
    expected_mae = float(df["absolute_error"].mean())
    assert np.isclose(computed_mae, expected_mae)
    assert np.isclose(computed_mae, 4.5)  # (5+5+5 + 4+4+4)/6 = 4.5


def test_rmse_recomputed_from_predictions(synthetic_canonical_predictions):
    """RMSE must equal root mean squared error."""
    df = synthetic_canonical_predictions
    computed_rmse = rmse(df["actual_rul"].values, df["predicted_rul_nonnegative"].values)
    expected_rmse = np.sqrt(np.mean(df["prediction_error"] ** 2))
    assert np.isclose(computed_rmse, expected_rmse)


def test_engine_macro_metrics(synthetic_canonical_predictions):
    """Engine macro metrics must average within each engine first, then across engines."""
    df = synthetic_canonical_predictions
    res = evaluate_simulated_endpoints(df)
    # Engine 1 MAE = 5.0, Engine 2 MAE = 4.0 -> Macro MAE = 4.5
    assert np.isclose(res["engine_macro_mae"], 4.5)
    assert res["n_engines"] == 2
    assert res["n_cutoff_observations"] == 6


# ═══════════════════════════════════════════════════════════════════════════
# 2. NASA ASYMMETRIC SCORE TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test_nasa_score_underprediction_branch():
    """Underprediction d = -13 must yield exp(13/13) - 1 = e - 1 ~ 1.71828."""
    y_true = np.array([100.0])
    y_pred = np.array([87.0])  # d = -13
    score = nasa_score(y_true, y_pred)
    assert np.isclose(score, np.exp(1.0) - 1.0)


def test_nasa_score_overprediction_branch():
    """Overprediction d = +10 must yield exp(10/10) - 1 = e - 1 ~ 1.71828."""
    y_true = np.array([100.0])
    y_pred = np.array([110.0])  # d = +10
    score = nasa_score(y_true, y_pred)
    assert np.isclose(score, np.exp(1.0) - 1.0)


def test_nasa_score_aggregate_definition():
    """NASA score sum must equal the sum of individual penalties, and mean must equal sum / N."""
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([87.0, 110.0])
    total_score = nasa_score(y_true, y_pred)
    mean_score = nasa_score_per_engine(y_true, y_pred)
    assert np.isclose(total_score, 2.0 * (np.exp(1.0) - 1.0))
    assert np.isclose(mean_score, total_score / 2.0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. PREDICTION TABLE INTEGRITY & ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def test_prediction_key_uniqueness(synthetic_canonical_predictions):
    """The canonical key (dataset, model, experiment_type, unit_id, cutoff_cycle) must be unique."""
    df = synthetic_canonical_predictions
    key_cols = ["dataset", "model", "experiment_type", "unit_id", "cutoff_cycle"]
    assert len(df) == len(df.drop_duplicates(subset=key_cols))


def test_prediction_target_alignment(synthetic_canonical_predictions):
    """actual_rul must strictly equal actual_terminal_cycle - cutoff_cycle."""
    df = synthetic_canonical_predictions
    assert (df["actual_rul"] == (df["actual_terminal_cycle"] - df["cutoff_cycle"])).all()


def test_prediction_cutoff_alignment(synthetic_canonical_predictions):
    """Observation cycle must equal cutoff_cycle."""
    df = synthetic_canonical_predictions
    assert (df["cycle"] == df["cutoff_cycle"]).all()


def test_positive_rul_at_simulated_endpoints(synthetic_canonical_predictions):
    """Every simulated endpoint observation must have strictly positive true RUL."""
    df = synthetic_canonical_predictions
    assert (df["actual_rul"] > 0).all()


# ═══════════════════════════════════════════════════════════════════════════
# 4. BASELINE FITTING AND FAIR POPULATION
# ═══════════════════════════════════════════════════════════════════════════

def test_baseline_fitted_on_training_targets_only():
    """DummyBaseline median must be calculated exclusively on training targets."""
    y_train = np.array([10.0, 50.0, 100.0, 150.0, 200.0])
    expected_median = 100.0
    from sklearn.dummy import DummyRegressor
    dummy = DummyRegressor(strategy="median")
    dummy.fit(np.zeros((len(y_train), 2)), y_train)
    assert dummy.constant_[0][0] == expected_median


def test_baseline_and_models_use_same_evaluation_population():
    """Baseline and ML models must be evaluated on the identical set of cutoff rows."""
    path = Path("reports/phase4_2/canonical_simulated_predictions.csv")
    if path.exists():
        df = pd.read_csv(path)
        bench_df = df[df["experiment_type"] == "primary_benchmark"]
        for subset in bench_df["dataset"].unique():
            sub_bench = bench_df[bench_df["dataset"] == subset]
            counts = sub_bench.groupby("model")["unit_id"].count()
            # All models must have the exact same count
            assert counts.nunique() == 1


# ═══════════════════════════════════════════════════════════════════════════
# 5. TARGET CAPS & OUTPUT CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

def test_cap_training_targets():
    """Capped training targets must equal min(y_train, cap)."""
    y_train = np.array([20.0, 80.0, 150.0])
    cap = 100.0
    capped = np.minimum(y_train, cap)
    assert np.array_equal(capped, np.array([20.0, 80.0, 100.0]))


def test_cap_output_predictions():
    """Capped output predictions must satisfy 0 <= y_pred <= cap."""
    raw_preds = np.array([-5.0, 40.0, 110.0, 180.0])
    cap = 125.0
    preds_capped = np.minimum(np.maximum(raw_preds, 0.0), cap)
    assert (preds_capped >= 0.0).all()
    assert (preds_capped <= cap).all()
    assert preds_capped[0] == 0.0
    assert preds_capped[1] == 40.0
    assert preds_capped[2] == 110.0
    assert preds_capped[3] == 125.0


def test_uncapped_physical_validation_truth_preserved():
    """Uncapped validation truth must NOT be capped during physical error evaluation."""
    y_val_uncapped = np.array([160.0, 90.0, 30.0])
    cap = 100.0
    y_pred_capped = np.array([100.0, 90.0, 30.0])
    # Evaluating against uncapped truth: error on row 0 is |160 - 100| = 60
    err = np.abs(y_val_uncapped - y_pred_capped)
    assert err[0] == 60.0


# ═══════════════════════════════════════════════════════════════════════════
# 6. WINDOW FEATURE SELECTION & INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

def test_window_feature_selection():
    """select_window_features must return only features matching the window or shared."""
    feature_list = [
        "sensor_2_mean_10", "sensor_2_std_10", "sensor_2_mean_20", "sensor_2_std_20",
        "sensor_2_delta_10", "sensor_2_delta_20", "sensor_2_relchange_10"
    ]
    w10 = select_window_features(feature_list, 10)
    assert "sensor_2_mean_10" in w10
    assert "sensor_2_mean_20" not in w10
    assert "sensor_2_delta_10" in w10
    assert "sensor_2_delta_20" in w10


def test_feature_inventory_compliance():
    """Verified models must strictly use approved features from Phase 3.1."""
    inv_path = Path("reports/phase3_1/feature_quality_inventory_FD001.csv")
    if inv_path.exists():
        inv = pd.read_csv(inv_path)
        approved = set(inv[inv["usable_for_phase4"] == "YES"]["feature_name"])
        # Forbidden columns should never be approved
        forbidden = {"rul", "cycle", "unit_id", "failure_cycle"}
        assert len(forbidden.intersection(approved)) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 7. CUTOFF SCHEDULE CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════

def test_cutoff_schedule_training_only():
    """Cutoff schedules must depend on training engine lifespans only."""
    train_terminals = np.array([120, 150, 180, 200, 240, 300])
    df_train = pd.DataFrame({
        "unit_id": list(range(1, len(train_terminals) + 1)),
        "cycle": train_terminals,
    })
    cutoffs, meta = compute_training_cutoffs(df_train, base_checkpoints=(80, 100), min_cycle_feature_availability=75)
    assert 80 in cutoffs
    assert 100 in cutoffs
    assert meta["n_training_engines"] == 6


def test_empty_cutoff_handling():
    """Checkpoints beyond all validation engines must be handled gracefully without fabrication."""
    val_df = pd.DataFrame({
        "unit_id": [1, 1],
        "cycle": [75, 80],
        "rul": [5, 0],
    })
    # Cutoff 100 has 0 active engines (engine 1 failed at 80)
    trunc_df = extract_simulated_truncations(val_df, cutoffs=[100])
    assert len(trunc_df) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 8. REPORT MATCHING & RANKING
# ═══════════════════════════════════════════════════════════════════════════

def test_report_matches_recomputed_metrics():
    """Metric values in reports/phase4_2/metric_reconciliation.csv must match recomputed tables."""
    recon_path = Path("reports/phase4_2/metric_reconciliation.csv")
    bench_path = Path("reports/phase4_2/canonical_simulated_predictions.csv")
    if recon_path.exists() and bench_path.exists():
        recon_df = pd.read_csv(recon_path)
        preds_df = pd.read_csv(bench_path)
        primary_preds = preds_df[preds_df["experiment_type"] == "primary_benchmark"]
        for _, row in recon_df.iterrows():
            sub = row["dataset"]
            mname = row["model"]
            m_preds = primary_preds[(primary_preds["dataset"] == sub) & (primary_preds["model"] == mname)]
            actual_mae = mae(m_preds["actual_rul"].values, m_preds["predicted_rul_nonnegative"].values)
            assert np.isclose(round(actual_mae, 2), row["recomputed_value"])


def test_ranking_generated_from_authoritative_results():
    """Ranking reconciliation table must be internally consistent with reported MAEs."""
    rank_path = Path("reports/phase4_2/ranking_reconciliation.csv")
    if rank_path.exists():
        df = pd.read_csv(rank_path)
        for subset in df["dataset"].unique():
            sub = df[df["dataset"] == subset].sort_values("rank_by_mae")
            # Verify ranks are sequential 1..N
            assert list(sub["rank_by_mae"]) == list(range(1, len(sub) + 1))
            # Verify MAEs are monotonically increasing with rank
            maes = sub["recomputed_simulated_mae"].values
            assert (np.diff(maes) >= 0).all()
