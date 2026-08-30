"""
AeroRUL — Operating-Condition Normalization

Identifies operating regimes from C-MAPSS operational settings and provides
leakage-safe condition-aware normalization.

All normalization parameters are fitted on TRAINING data only.
Validation/test data is transformed using the training-fitted parameters.

Usage:
    from src.normalization import (
        identify_operating_regimes,
        fit_regime_normalization,
        apply_regime_normalization,
    )

    regimes = identify_operating_regimes(train_df)
    norm_params = fit_regime_normalization(train_df, regimes)
    train_norm = apply_regime_normalization(train_df, norm_params, regimes)
    val_norm   = apply_regime_normalization(val_df, norm_params, regimes)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    NORMALIZATION_STD_FLOOR,
    OP_SETTING_COLUMNS,
    OP_SETTING_ROUNDING_DECIMALS,
    SENSOR_COLUMNS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. OPERATING REGIME IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def identify_operating_regimes(
    df: pd.DataFrame,
    op_columns: Optional[list[str]] = None,
    rounding_decimals: int = OP_SETTING_ROUNDING_DECIMALS,
) -> pd.Series:
    """
    Identify operating regimes by rounding operational settings.

    Rounds each operational setting to the specified number of decimals,
    then creates a tuple-based regime identifier.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with operational setting columns.
    op_columns : list[str], optional
        Columns to use for regime identification.
        Defaults to config.OP_SETTING_COLUMNS.
    rounding_decimals : int
        Number of decimal places to round to.

    Returns
    -------
    pd.Series
        A Series of string regime labels, one per row.
        Format: "regime_(val1, val2, val3)"
    """
    if op_columns is None:
        op_columns = OP_SETTING_COLUMNS

    rounded = df[op_columns].round(rounding_decimals) + 0.0  # eliminate -0.0
    regime_labels = rounded.apply(
        lambda row: "regime_" + str(tuple(row.values)), axis=1
    )

    return regime_labels


def regime_summary(
    df: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Summarize the number of observations per operating regime.

    Returns
    -------
    pd.DataFrame
        Columns: regime, n_observations, pct_observations
    """
    counts = regimes.value_counts().reset_index()
    counts.columns = ["regime", "n_observations"]
    counts["pct_observations"] = round(
        100.0 * counts["n_observations"] / len(df), 2
    )
    counts = counts.sort_values("regime").reset_index(drop=True)
    return counts


# ═══════════════════════════════════════════════════════════════════════════
# 2. NORMALIZATION FITTING (TRAINING DATA ONLY)
# ═══════════════════════════════════════════════════════════════════════════

def fit_regime_normalization(
    train_df: pd.DataFrame,
    regimes: pd.Series,
    sensor_columns: Optional[list[str]] = None,
    std_floor: float = NORMALIZATION_STD_FLOOR,
) -> dict:
    """
    Fit per-regime normalization parameters using TRAINING DATA ONLY.

    For each regime, calculates the mean and std of each sensor.
    Standard deviations below std_floor are clamped to avoid division by zero.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data (DO NOT pass validation data).
    regimes : pd.Series
        Regime labels aligned with train_df.
    sensor_columns : list[str], optional
        Sensor columns to normalize.
    std_floor : float
        Minimum allowed standard deviation.

    Returns
    -------
    dict
        {
            regime_label: {
                "mean": {sensor: float, ...},
                "std": {sensor: float, ...},
                "n_observations": int,
            },
            ...
        }
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    cols = [c for c in sensor_columns if c in train_df.columns]
    params = {}

    for regime_label in regimes.unique():
        mask = regimes == regime_label
        regime_data = train_df.loc[mask, cols]

        means = regime_data.mean().to_dict()
        stds = regime_data.std().to_dict()

        # Clamp small stds to avoid division by zero
        stds = {k: max(v, std_floor) for k, v in stds.items()}

        params[regime_label] = {
            "mean": means,
            "std": stds,
            "n_observations": int(mask.sum()),
        }

    return params


# ═══════════════════════════════════════════════════════════════════════════
# 3. NORMALIZATION APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def apply_regime_normalization(
    df: pd.DataFrame,
    norm_params: dict,
    regimes: pd.Series,
    sensor_columns: Optional[list[str]] = None,
    suffix: str = "_norm",
) -> pd.DataFrame:
    """
    Apply per-regime normalization using pre-fitted parameters.

    For each row, looks up its regime, retrieves the training-fitted
    mean and std, and computes:
        normalized = (value - regime_mean) / regime_std

    Unknown regimes (not seen during training) are left unnormalized.

    Parameters
    ----------
    df : pd.DataFrame
        Data to normalize (can be train, validation, or test).
    norm_params : dict
        Parameters from fit_regime_normalization.
    regimes : pd.Series
        Regime labels aligned with df.
    sensor_columns : list[str], optional
        Sensor columns to normalize.
    suffix : str
        Suffix for normalized column names. Set to "" to overwrite in-place.

    Returns
    -------
    pd.DataFrame
        Copy of df with normalized sensor columns added.
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    cols = [c for c in sensor_columns if c in df.columns]
    df = df.copy()

    # Initialize normalized columns with original values (as float64)
    for col in cols:
        target_col = col + suffix if suffix else col
        df[target_col] = df[col].astype(np.float64).values.copy()

    # Apply per-regime normalization
    for regime_label, params in norm_params.items():
        mask = regimes == regime_label
        if not mask.any():
            continue

        for col in cols:
            target_col = col + suffix if suffix else col
            mean = params["mean"].get(col, 0.0)
            std = params["std"].get(col, 1.0)
            df.loc[mask, target_col] = (df.loc[mask, col] - mean) / std

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def regime_sensor_analysis(
    df: pd.DataFrame,
    regimes: pd.Series,
    sensor_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Analyze sensor statistics within each operating regime.

    Returns a DataFrame with one row per (regime, sensor) combination.
    Columns: regime, sensor, mean, std, min, max, n_observations
    """
    if sensor_columns is None:
        sensor_columns = SENSOR_COLUMNS

    cols = [c for c in sensor_columns if c in df.columns]
    rows = []

    for regime_label in sorted(regimes.unique()):
        mask = regimes == regime_label
        regime_data = df.loc[mask]

        for col in cols:
            series = regime_data[col]
            rows.append({
                "regime": regime_label,
                "sensor": col,
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": float(series.min()),
                "max": float(series.max()),
                "n_observations": int(mask.sum()),
            })

    return pd.DataFrame(rows)
