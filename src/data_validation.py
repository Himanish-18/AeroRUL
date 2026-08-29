"""
AeroRUL — C-MAPSS Data Validation

Comprehensive validation and exploratory statistics for C-MAPSS data.
All functions accept DataFrames and return structured results — no side effects.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    COLUMN_NAMES,
    EXPECTED_NUM_COLUMNS,
    ID_COLUMNS,
    NEAR_CONSTANT_STD_THRESHOLD,
    OP_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. STRUCTURAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_structure(df: pd.DataFrame, label: str = "dataset") -> dict:
    """
    Validate the structural properties of a loaded C-MAPSS DataFrame.

    Returns a dict with keys:
        n_rows, n_cols, column_names, dtypes, column_count_ok,
        column_names_ok, n_duplicate_rows, n_duplicate_unit_cycle,
        duplicate_rows (DataFrame), duplicate_unit_cycle (DataFrame)
    """
    n_rows, n_cols = df.shape
    col_names = list(df.columns)

    # Column count check
    column_count_ok = n_cols == EXPECTED_NUM_COLUMNS

    # Column names check
    column_names_ok = col_names == COLUMN_NAMES

    # Duplicate rows
    dup_rows = df[df.duplicated(keep=False)]
    n_dup_rows = df.duplicated().sum()

    # Duplicate (unit_id, cycle) combinations
    dup_uc = df[df.duplicated(subset=["unit_id", "cycle"], keep=False)]
    n_dup_uc = df.duplicated(subset=["unit_id", "cycle"]).sum()

    return {
        "label": label,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "column_names": col_names,
        "dtypes": df.dtypes.to_dict(),
        "column_count_ok": column_count_ok,
        "column_names_ok": column_names_ok,
        "n_duplicate_rows": int(n_dup_rows),
        "n_duplicate_unit_cycle": int(n_dup_uc),
        "duplicate_rows": dup_rows,
        "duplicate_unit_cycle": dup_uc,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. MISSING VALUE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check every column for NaN, null, and infinite values.

    Returns a DataFrame with columns:
        column, n_nan, pct_nan, n_inf, pct_inf, n_total_missing, pct_total_missing
    """
    results = []
    n = len(df)

    for col in df.columns:
        n_nan = int(df[col].isna().sum())
        pct_nan = round(100.0 * n_nan / n, 4) if n > 0 else 0.0

        # Infinite check only for numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            n_inf = int(np.isinf(df[col]).sum())
        else:
            n_inf = 0
        pct_inf = round(100.0 * n_inf / n, 4) if n > 0 else 0.0

        n_total = n_nan + n_inf
        pct_total = round(100.0 * n_total / n, 4) if n > 0 else 0.0

        results.append({
            "column": col,
            "n_nan": n_nan,
            "pct_nan": pct_nan,
            "n_inf": n_inf,
            "pct_inf": pct_inf,
            "n_total_missing": n_total,
            "pct_total_missing": pct_total,
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENGINE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_engines(df: pd.DataFrame) -> dict:
    """
    Calculate engine-level statistics.

    Returns a dict with:
        n_engines, cycles_per_engine (Series), min_cycles, max_cycles,
        mean_cycles, median_cycles, cycles_ordered, cycles_start_at_1,
        anomalous_start_engines
    """
    cycles_per_engine = df.groupby("unit_id")["cycle"].max()

    # Check if cycles are ordered within each engine
    def _is_monotonic(group):
        return group["cycle"].is_monotonic_increasing

    ordered_check = df.groupby("unit_id").apply(_is_monotonic, include_groups=False)
    all_ordered = bool(ordered_check.all())

    # Check if cycles start at 1 for each engine
    first_cycles = df.groupby("unit_id")["cycle"].min()
    starts_at_1 = bool((first_cycles == 1).all())
    anomalous_starts = first_cycles[first_cycles != 1]

    return {
        "n_engines": int(df["unit_id"].nunique()),
        "cycles_per_engine": cycles_per_engine,
        "min_cycles": int(cycles_per_engine.min()),
        "max_cycles": int(cycles_per_engine.max()),
        "mean_cycles": round(float(cycles_per_engine.mean()), 2),
        "median_cycles": float(cycles_per_engine.median()),
        "cycles_ordered": all_ordered,
        "cycles_start_at_1": starts_at_1,
        "anomalous_start_engines": anomalous_starts,
    }


def get_engine_final_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the final (maximum) cycle for every engine in the dataset.

    Returns a DataFrame with columns: unit_id, final_cycle
    """
    final = (
        df.groupby("unit_id")["cycle"]
        .max()
        .reset_index()
        .rename(columns={"cycle": "final_cycle"})
    )
    return final


# ═══════════════════════════════════════════════════════════════════════════
# 4. SENSOR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_sensors(
    df: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute descriptive statistics for each sensor column.

    Returns a DataFrame with one row per sensor and columns:
        sensor, min, max, mean, std, var, n_unique,
        pct_missing, is_constant, is_near_constant
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    n = len(df)
    results = []

    for col in sensor_columns:
        if col not in df.columns:
            continue

        series = df[col]
        col_std = float(series.std())
        col_mean = float(series.mean())

        # Near-constant: std is negligible relative to mean
        if col_mean != 0:
            is_near_constant = abs(col_std / col_mean) < NEAR_CONSTANT_STD_THRESHOLD
        else:
            is_near_constant = col_std < 1e-10

        n_missing = int(series.isna().sum())

        results.append({
            "sensor": col,
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(col_mean, 6),
            "std": round(col_std, 6),
            "var": round(float(series.var()), 6),
            "n_unique": int(series.nunique()),
            "pct_missing": round(100.0 * n_missing / n, 4) if n > 0 else 0.0,
            "is_constant": bool(series.nunique() <= 1),
            "is_near_constant": bool(is_near_constant),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 5. OPERATING CONDITION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyze_operating_conditions(
    df: pd.DataFrame,
    op_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute statistics for operational setting columns.

    Returns a DataFrame with columns:
        setting, n_unique, mean, std, min, max
    """
    if op_columns is None:
        op_columns = OP_SETTING_COLUMNS

    results = []
    for col in op_columns:
        if col not in df.columns:
            continue
        series = df[col]
        results.append({
            "setting": col,
            "n_unique": int(series.nunique()),
            "mean": round(float(series.mean()), 6),
            "std": round(float(series.std()), 6),
            "min": float(series.min()),
            "max": float(series.max()),
        })

    return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SENSOR CORRELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def sensor_cycle_correlation(
    df: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute Pearson correlation between each sensor and the cycle column.

    Returns a DataFrame with columns: sensor, pearson_r
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    results = []
    for col in sensor_columns:
        if col not in df.columns:
            continue
        r = df[col].corr(df["cycle"])
        results.append({"sensor": col, "pearson_r": round(float(r), 6)})

    return pd.DataFrame(results)


def sensor_correlation_matrix(
    df: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Compute the sensor × sensor Pearson correlation matrix.

    Returns a square DataFrame.
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    cols = [c for c in sensor_columns if c in df.columns]
    return df[cols].corr(method="pearson")


# ═══════════════════════════════════════════════════════════════════════════
# 7. TRAJECTORY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def trajectory_summary(df: pd.DataFrame, label: str = "dataset") -> dict:
    """
    Compute trajectory-level summary statistics.

    Returns a dict:
        label, n_engines, n_observations, avg_trajectory_length,
        min_trajectory_length, max_trajectory_length, median_trajectory_length
    """
    traj_lengths = df.groupby("unit_id")["cycle"].count()

    return {
        "label": label,
        "n_engines": int(df["unit_id"].nunique()),
        "n_observations": int(len(df)),
        "avg_trajectory_length": round(float(traj_lengths.mean()), 2),
        "min_trajectory_length": int(traj_lengths.min()),
        "max_trajectory_length": int(traj_lengths.max()),
        "median_trajectory_length": float(traj_lengths.median()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 8. FULL VALIDATION PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_full_validation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: Optional[pd.DataFrame] = None,
    subset: str = "unknown",
) -> dict:
    """
    Run the complete validation pipeline on a single subset.

    Returns a structured dict containing all validation results.
    """
    results = {
        "subset": subset,
        # Structural
        "train_structure": validate_structure(train_df, f"{subset}_train"),
        "test_structure": validate_structure(test_df, f"{subset}_test"),
        # Missing values
        "train_missing": validate_missing_values(train_df),
        "test_missing": validate_missing_values(test_df),
        # Engine validation
        "train_engines": validate_engines(train_df),
        "test_engines": validate_engines(test_df),
        # Trajectory summaries
        "train_trajectory": trajectory_summary(train_df, f"{subset}_train"),
        "test_trajectory": trajectory_summary(test_df, f"{subset}_test"),
        # Engine final cycles
        "train_final_cycles": get_engine_final_cycles(train_df),
        # Sensor validation
        "train_sensors": validate_sensors(train_df),
        "test_sensors": validate_sensors(test_df),
        # Operating conditions
        "train_op_conditions": analyze_operating_conditions(train_df),
        "test_op_conditions": analyze_operating_conditions(test_df),
        # Sensor correlations (training only — no test leakage)
        "sensor_cycle_corr": sensor_cycle_correlation(train_df),
        "sensor_corr_matrix": sensor_correlation_matrix(train_df),
    }

    # RUL file info
    if rul_df is not None:
        results["rul_n_engines"] = len(rul_df)
        results["rul_min"] = int(rul_df["rul"].min())
        results["rul_max"] = int(rul_df["rul"].max())
        results["rul_mean"] = round(float(rul_df["rul"].mean()), 2)

    return results
