"""
AeroRUL — Phase 4.1: Retrospective Truncation & Validation Module

Implements simulated pre-failure truncation protocol for NASA C-MAPSS engines:
- Training-derived cutoff schedule generation
- Positive-RUL validation observation extraction
- Cutoff coverage and reliability reporting
- Engine-cluster bootstrap confidence intervals
- Validation protocol auditing
"""

import numpy as np
import pandas as pd
from typing import Callable, Optional

from src.metrics import (
    mae,
    rmse,
    r2_score,
    mean_signed_error,
    nasa_score,
    nasa_score_per_engine,
    assign_rul_range,
    RUL_RANGE_BINS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. CUTOFF SCHEDULE GENERATION (TRAINING ENGINES ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def compute_training_cutoffs(
    train_df: pd.DataFrame,
    base_checkpoints: tuple[int, ...] = (80, 100, 125),
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75, 0.90),
    min_cycle_feature_availability: int = 75,
) -> tuple[list[int], dict]:
    """
    Construct deterministic cycle cutoff schedule derived STRICTLY from training engines.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training observations containing 'unit_id' and 'cycle'.
    base_checkpoints : tuple of int
        Predeclared early absolute cycle checkpoints.
    quantiles : tuple of float
        Percentiles of training engine terminal cycles (Tu).
    min_cycle_feature_availability : int
        Minimum cycle supported by the feature extraction pipeline (e.g. 75 for max window 75).

    Returns
    -------
    cutoffs : list of int
        Sorted, deduplicated list of valid cutoff cycles.
    metadata : dict
        Details on terminal cycle distributions and quantiles.
    """
    # Terminal cycle per training engine
    terminals = train_df.groupby("unit_id")["cycle"].max().values
    if len(terminals) == 0:
        raise ValueError("train_df contains no engines")

    q_dict = {}
    q_values = []
    for q in quantiles:
        val = int(np.round(np.percentile(terminals, q * 100)))
        q_dict[f"q{int(q*100)}"] = val
        q_values.append(val)

    # Combine base checkpoints + training quantiles
    candidate_cutoffs = set(base_checkpoints) | set(q_values)

    # Filter out any below feature availability
    valid_cutoffs = sorted([c for c in candidate_cutoffs if c >= min_cycle_feature_availability])
    excluded_cutoffs = sorted([c for c in candidate_cutoffs if c < min_cycle_feature_availability])

    metadata = {
        "n_training_engines": int(len(terminals)),
        "terminal_min": int(np.min(terminals)),
        "terminal_median": float(np.median(terminals)),
        "terminal_mean": float(np.mean(terminals)),
        "terminal_max": int(np.max(terminals)),
        "terminal_quantiles": q_dict,
        "base_checkpoints": list(base_checkpoints),
        "excluded_below_min_cycle": excluded_cutoffs,
        "valid_cutoffs": valid_cutoffs,
    }

    return valid_cutoffs, metadata


# ═══════════════════════════════════════════════════════════════════════════
# 2. SIMULATED TRUNCATION EXTRACTION (POSITIVE RUL ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def extract_simulated_truncations(
    val_df: pd.DataFrame,
    cutoffs: list[int],
    schedule_id: str = "train_quantiles_v1",
) -> pd.DataFrame:
    """
    Extract validation feature rows at pre-declared cutoff checkpoints.

    Strict Evaluation Criteria:
    1. Observation cycle must equal cutoff cycle: cycle == t
    2. True RUL must be STRICTLY POSITIVE: actual_rul > 0 (i.e. t < Tu).
       Terminal rows (actual_rul == 0) are strictly excluded from simulated pre-failure evaluation.
    3. Retains all features already engineered up to cycle t.
    4. Future features, failure cycles, and metadata are NOT present in predictors.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only eligible simulated truncation points with
        metadata columns 'cutoff_cycle', 'cutoff_schedule_id', and 'actual_terminal_cycle'.
    """
    # Compute true terminal cycle per engine for audit & verification
    engine_terminals = val_df.groupby("unit_id")["cycle"].max().to_dict()

    eligible_rows = []
    for c in cutoffs:
        rows_at_c = val_df[val_df["cycle"] == c].copy()
        if len(rows_at_c) == 0:
            continue

        for idx, row in rows_at_c.iterrows():
            eid = row["unit_id"]
            terminal_cycle = engine_terminals[eid]
            actual_rul = row["rul"]

            # Strict pre-failure condition: t < Tu (actual_rul > 0)
            if row["cycle"] < terminal_cycle and actual_rul > 0:
                row_dict = row.to_dict()
                row_dict["cutoff_cycle"] = int(c)
                row_dict["actual_terminal_cycle"] = int(terminal_cycle)
                row_dict["cutoff_schedule_id"] = schedule_id
                eligible_rows.append(row_dict)

    if not eligible_rows:
        return pd.DataFrame()

    trunc_df = pd.DataFrame(eligible_rows)
    return trunc_df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. CUTOFF COVERAGE & RELIABILITY
# ═══════════════════════════════════════════════════════════════════════════

def compute_cutoff_coverage(
    trunc_df: pd.DataFrame,
    cutoffs: list[int],
    total_val_engines: int,
    subset: str,
) -> pd.DataFrame:
    """
    Calculate engine coverage and true-RUL distribution for each cutoff checkpoint.
    """
    records = []
    for c in cutoffs:
        if len(trunc_df) > 0:
            at_c = trunc_df[trunc_df["cutoff_cycle"] == c]
        else:
            at_c = pd.DataFrame()

        n_eligible = len(at_c)
        n_engines = at_c["unit_id"].nunique() if n_eligible > 0 else 0

        if n_eligible > 0:
            ruls = at_c["rul"].values
            min_rul = float(np.min(ruls))
            med_rul = float(np.median(ruls))
            mean_rul = float(np.mean(ruls))
            max_rul = float(np.max(ruls))
            std_rul = float(np.std(ruls))
        else:
            min_rul = None
            med_rul = None
            mean_rul = None
            max_rul = None
            std_rul = None

        records.append({
            "dataset": subset,
            "cutoff_cycle": int(c),
            "n_eligible_engines": int(n_engines),
            "n_predictions": int(n_eligible),
            "engine_coverage_pct": round(float(n_engines / total_val_engines * 100), 1) if total_val_engines > 0 else 0.0,
            "minimum_true_rul": min_rul,
            "median_true_rul": med_rul,
            "mean_true_rul": round(mean_rul, 2) if mean_rul is not None else None,
            "maximum_true_rul": max_rul,
            "std_true_rul": round(std_rul, 2) if std_rul is not None else None,
            "is_low_sample": bool(n_engines < 5),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ENGINE-CLUSTER BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════

def engine_cluster_bootstrap(
    df: pd.DataFrame,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
    n_bootstraps: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Compute cluster bootstrap confidence intervals by resampling whole engines with replacement.
    Prevents artificially narrow intervals caused by treating multiple cutoff observations
    from the same engine as independent.
    """
    unique_engines = df["unit_id"].unique()
    n_engines = len(unique_engines)
    if n_engines == 0:
        return 0.0, 0.0, 0.0

    y_true_all = df[actual_col].values
    y_pred_all = df[pred_col].values
    point_est = float(metric_fn(y_true_all, y_pred_all))

    if n_engines < 3:
        return point_est, point_est, point_est

    rng = np.random.RandomState(seed)
    # Pre-index observations per engine for fast resampling
    engine_groups = {eid: edf.index.values for eid, edf in df.groupby("unit_id")}

    boot_metrics = np.empty(n_bootstraps, dtype=float)
    for b in range(n_bootstraps):
        sampled_eids = rng.choice(unique_engines, size=n_engines, replace=True)
        sampled_indices = np.concatenate([engine_groups[eid] for eid in sampled_eids])
        b_true = df.loc[sampled_indices, actual_col].values
        b_pred = df.loc[sampled_indices, pred_col].values
        boot_metrics[b] = metric_fn(b_true, b_pred)

    ci_lower = float(np.percentile(boot_metrics, (alpha / 2.0) * 100))
    ci_upper = float(np.percentile(boot_metrics, (1.0 - alpha / 2.0) * 100))

    return point_est, ci_lower, ci_upper


# ═══════════════════════════════════════════════════════════════════════════
# 5. SIMULATED ENDPOINT EVALUATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_simulated_endpoints(
    preds_df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> dict:
    """
    Evaluate simulated pre-failure endpoints:
    - Pooled metrics across all eligible cutoff observations
    - Engine-macro metrics (equal weight per engine)
    - NASA asymmetric score (sum and mean per observation)
    - Bias (mean signed error)
    """
    if len(preds_df) == 0:
        return {}

    y_true = preds_df[actual_col].values
    y_pred = preds_df[pred_col].values
    d = y_pred - y_true

    # Pooled metrics
    pooled_mae = mae(y_true, y_pred)
    pooled_rmse = rmse(y_true, y_pred)
    pooled_r2 = r2_score(y_true, y_pred)
    bias = mean_signed_error(y_true, y_pred)
    nasa_sum = nasa_score(y_true, y_pred)
    nasa_mean = float(nasa_sum / len(preds_df))

    # Engine-macro metrics: compute within-engine mean error, then macro-average
    engine_stats = []
    for eid, edf in preds_df.groupby("unit_id"):
        ey_true = edf[actual_col].values
        ey_pred = edf[pred_col].values
        engine_stats.append({
            "unit_id": eid,
            "engine_mae": mae(ey_true, ey_pred),
            "engine_rmse": rmse(ey_true, ey_pred),
            "n_cutoffs": len(edf),
        })
    engine_stats_df = pd.DataFrame(engine_stats)
    macro_mae = float(engine_stats_df["engine_mae"].mean())
    macro_rmse = float(engine_stats_df["engine_rmse"].mean())

    return {
        "n_engines": int(preds_df["unit_id"].nunique()),
        "n_cutoff_observations": int(len(preds_df)),
        "simulated_truncated_endpoint_mae": pooled_mae,
        "simulated_truncated_endpoint_rmse": pooled_rmse,
        "simulated_truncated_endpoint_r2": pooled_r2,
        "engine_macro_mae": macro_mae,
        "engine_macro_rmse": macro_rmse,
        "mean_signed_error": bias,
        "nasa_score_sum": nasa_sum,
        "nasa_score_mean": nasa_mean,
        "n_underpredictions": int(np.sum(d < 0)),
        "n_overpredictions": int(np.sum(d >= 0)),
    }


def evaluate_simulated_endpoints_by_cutoff(
    preds_df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> pd.DataFrame:
    """Compute performance metrics broken down by individual cutoff checkpoint."""
    records = []
    for cutoff, cdf in preds_df.groupby("cutoff_cycle"):
        y_true = cdf[actual_col].values
        y_pred = cdf[pred_col].values
        d = y_pred - y_true
        ns = nasa_score(y_true, y_pred)

        records.append({
            "cutoff_cycle": int(cutoff),
            "n_engines": int(cdf["unit_id"].nunique()),
            "n_observations": int(len(cdf)),
            "true_rul_mean": float(np.mean(y_true)),
            "true_rul_median": float(np.median(y_true)),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mean_signed_error": mean_signed_error(y_true, y_pred),
            "nasa_score_sum": ns,
            "nasa_score_mean": float(ns / len(cdf)),
            "n_under": int(np.sum(d < 0)),
            "n_over": int(np.sum(d >= 0)),
        })

    return pd.DataFrame(records).sort_values("cutoff_cycle").reset_index(drop=True)


def simulated_rul_range_error_analysis(
    preds_df: pd.DataFrame,
    pred_col: str = "predicted_rul_nonnegative",
    actual_col: str = "actual_rul",
) -> pd.DataFrame:
    """Analyze errors across standard operational RUL range bins at simulated cutoffs."""
    df = preds_df.copy()
    df["rul_range"] = assign_rul_range(df[actual_col].values)

    records = []
    for label, low, high in RUL_RANGE_BINS:
        subset = df[df["rul_range"] == label]
        if len(subset) == 0:
            continue
        y_true = subset[actual_col].values
        y_pred = subset[pred_col].values
        d = y_pred - y_true
        ns = nasa_score(y_true, y_pred)

        records.append({
            "rul_range": label,
            "range_min": low,
            "range_max": high if high != float("inf") else ">125",
            "n_observations": int(len(subset)),
            "n_unique_engines": int(subset["unit_id"].nunique()),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mean_signed_error": mean_signed_error(y_true, y_pred),
            "nasa_score_sum": ns,
            "nasa_score_mean": float(ns / len(subset)),
            "underpredict_pct": round(float(np.sum(d < 0) / len(subset) * 100), 1),
            "overpredict_pct": round(float(np.sum(d >= 0) / len(subset) * 100), 1),
        })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PROTOCOL AUDIT DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_protocol_audit_table() -> pd.DataFrame:
    """Return structured audit table contrasting Phase 4 and Phase 4.1 validation protocols."""
    rows = [
        {
            "metric_or_experiment": "Endpoint MAE & RMSE",
            "existing_method": "Evaluated at final row of complete run-to-failure validation trajectories: idxmax(cycle), where actual RUL = 0.",
            "identified_limitation": "Measures terminal-state failure diagnostic accuracy (RUL=0) rather than operational remaining life prediction when life remains (RUL>0).",
            "required_action": "Evaluate at predeclared retrospective cycle cutoffs (t < Tu, RUL = Tu - t > 0) derived from training engine quantiles.",
        },
        {
            "metric_or_experiment": "NASA Asymmetric Score",
            "existing_method": "Sum of asymmetric penalties evaluated exclusively at terminal cycle Tu where true RUL = 0.",
            "identified_limitation": "Since true RUL=0, any positive prediction is an overprediction (d >= 0, exp(d/10)-1). Underprediction penalty branch (d < 0, exp(-d/13)-1) is almost never triggered.",
            "required_action": "Evaluate NASA score on positive-RUL cutoff checkpoints, exercising both over- and under-prediction penalty branches, and report both sum and mean per observation.",
        },
        {
            "metric_or_experiment": "RUL-Cap Sensitivity (Capped vs Uncapped)",
            "existing_method": "Compared endpoint MAE vs uncapped truth (0) and vs capped truth (min(0, cap)).",
            "identified_limitation": "Since min(0, cap) = 0 for any cap >= 0, capped-target and uncapped-target endpoint MAEs were numerically identical in Phase 4.",
            "required_action": "Evaluate cap models at positive-RUL cutoffs against uncapped physical RUL truth to measure genuine impact of early-life plateauing on operating engines.",
        },
        {
            "metric_or_experiment": "Window Configuration Comparison",
            "existing_method": "Compared single windows (W=10..75) against combined 'All' windows using row-level pooled MAE and terminal endpoint MAE.",
            "identified_limitation": "Terminal-cycle evaluation does not test whether multi-scale window features preserve degradation trend tracking at mid-life operational cycles.",
            "required_action": "Re-evaluate window configurations under identical positive-RUL cutoffs and consistent validation populations.",
        },
        {
            "metric_or_experiment": "Row-Level Pooled Validation Metrics",
            "existing_method": "Aggregated MAE, RMSE, and R² across all available validation cycles.",
            "identified_limitation": "Mixes all degradation stages without standardization; longer-lived engines contribute more cycles, disproportionately weighting slow-degrading units.",
            "required_action": "Preserve row-level metrics as baseline diagnostic while introducing standardized cutoff checkpoints and engine-macro averaging.",
        },
    ]
    return pd.DataFrame(rows)
