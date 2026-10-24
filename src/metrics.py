"""
AeroRUL — Phase 4: Metrics Module

Evaluation metrics for RUL prediction benchmarking.
Includes row-level, endpoint, engine-macro, and NASA asymmetric score.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
# 1. ROW-LEVEL METRICS
# ═══════════════════════════════════════════════════════════════════════════

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mean_signed_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean signed error (bias). Positive = overprediction."""
    return float(np.mean(y_pred - y_true))


# ═══════════════════════════════════════════════════════════════════════════
# 2. NASA ASYMMETRIC SCORE
# ═══════════════════════════════════════════════════════════════════════════

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA C-MAPSS asymmetric scoring function.

    d_i = y_pred_i - y_true_i

    s_i = exp(-d_i / 13) - 1    if d_i < 0  (underprediction — more costly)
    s_i = exp( d_i / 10) - 1    if d_i >= 0 (overprediction)

    S = sum(s_i)

    Lower is better. Late predictions (underpredicting RUL / predicting
    failure later than actual) are penalized more heavily.
    """
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp(d / 10.0) - 1,
    )
    return float(np.sum(scores))


def nasa_score_per_engine(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Average NASA score (for comparisons across different-sized subsets)."""
    d = y_pred - y_true
    scores = np.where(
        d < 0,
        np.exp(-d / 13.0) - 1,
        np.exp(d / 10.0) - 1,
    )
    return float(np.mean(scores))


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENGINE-MACRO METRICS
# ═══════════════════════════════════════════════════════════════════════════

def engine_macro_metrics(
    df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> dict:
    """
    Compute per-engine MAE and RMSE, then macro-average.

    Each engine gets equal weight regardless of trajectory length.
    """
    per_engine = []
    for eid, edf in df.groupby("unit_id"):
        y_true = edf[actual_col].values
        y_pred = edf[pred_col].values
        per_engine.append({
            "unit_id": eid,
            "engine_mae": mae(y_true, y_pred),
            "engine_rmse": rmse(y_true, y_pred),
            "n_observations": len(edf),
        })

    per_engine_df = pd.DataFrame(per_engine)
    return {
        "macro_mae": float(per_engine_df["engine_mae"].mean()),
        "macro_rmse": float(per_engine_df["engine_rmse"].mean()),
        "n_engines": len(per_engine_df),
        "per_engine": per_engine_df,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. ENDPOINT METRICS
# ═══════════════════════════════════════════════════════════════════════════

def extract_endpoints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the last observed feature row for each validation engine.

    The endpoint is determined by the maximum cycle within each engine's
    available feature rows. This mirrors the C-MAPSS official test setup.
    """
    idx = df.groupby("unit_id")["cycle"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def endpoint_metrics(
    endpoints: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> dict:
    """Compute all endpoint-level metrics."""
    y_true = endpoints[actual_col].values
    y_pred = endpoints[pred_col].values
    d = y_pred - y_true

    n_under = int(np.sum(d < 0))
    n_over = int(np.sum(d >= 0))

    return {
        "endpoint_mae": mae(y_true, y_pred),
        "endpoint_rmse": rmse(y_true, y_pred),
        "endpoint_r2": r2_score(y_true, y_pred),
        "endpoint_nasa_score": nasa_score(y_true, y_pred),
        "endpoint_nasa_avg": nasa_score_per_engine(y_true, y_pred),
        "n_engines": len(endpoints),
        "mean_signed_error": mean_signed_error(y_true, y_pred),
        "n_underpredictions": n_under,
        "n_overpredictions": n_over,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. RUL-RANGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

RUL_RANGE_BINS = [
    ("critical_0_30", 0, 30),
    ("low_31_75", 31, 75),
    ("moderate_76_125", 76, 125),
    ("high_gt125", 126, float("inf")),
]


def assign_rul_range(rul_values: np.ndarray) -> np.ndarray:
    """Assign each RUL value to a named range bin."""
    labels = np.empty(len(rul_values), dtype=object)
    for label, low, high in RUL_RANGE_BINS:
        mask = (rul_values >= low) & (rul_values <= high)
        labels[mask] = label
    return labels


def rul_range_error_analysis(
    df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> pd.DataFrame:
    """Compute error metrics stratified by true RUL range."""
    df = df.copy()
    df["rul_range"] = assign_rul_range(df[actual_col].values)

    rows = []
    for label, _, _ in RUL_RANGE_BINS:
        subset = df[df["rul_range"] == label]
        if len(subset) == 0:
            continue
        y_true = subset[actual_col].values
        y_pred = subset[pred_col].values
        rows.append({
            "rul_range": label,
            "n_observations": len(subset),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mean_signed_error": mean_signed_error(y_true, y_pred),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPREHENSIVE EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(
    preds_df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> dict:
    """Run all evaluation metrics on a predictions DataFrame."""
    y_true = preds_df[actual_col].values
    y_pred = preds_df[pred_col].values

    # Row-level
    row_metrics = {
        "row_mae": mae(y_true, y_pred),
        "row_rmse": rmse(y_true, y_pred),
        "row_r2": r2_score(y_true, y_pred),
        "n_observations": len(preds_df),
    }

    # Engine-macro
    eng_macro = engine_macro_metrics(preds_df, pred_col, actual_col)
    row_metrics["macro_mae"] = eng_macro["macro_mae"]
    row_metrics["macro_rmse"] = eng_macro["macro_rmse"]
    row_metrics["n_engines"] = eng_macro["n_engines"]

    # Endpoint
    endpoints = extract_endpoints(preds_df)
    ep = endpoint_metrics(endpoints, pred_col, actual_col)
    row_metrics.update(ep)

    # RUL range
    rul_range = rul_range_error_analysis(preds_df, pred_col, actual_col)
    row_metrics["rul_range_analysis"] = rul_range

    return row_metrics
