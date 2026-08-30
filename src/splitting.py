"""
AeroRUL — Engine-Level Train/Validation Splitting

Leakage-safe splitting that operates on engine IDs, not individual
observations. No engine may appear in both training and validation.

Usage:
    from src.splitting import split_by_engine

    train_df, val_df = split_by_engine(df, validation_fraction=0.2, random_state=42)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import DEFAULT_RANDOM_STATE, DEFAULT_VALIDATION_FRACTION


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENGINE-LEVEL SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def split_by_engine(
    df: pd.DataFrame,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train and validation at the engine level.

    Every row belonging to an engine stays in the same partition.
    No engine appears in both partitions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'unit_id' column.
    validation_fraction : float
        Fraction of engines to hold out for validation (0.0–1.0).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, val_df) — disjoint DataFrames.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            f"validation_fraction must be between 0 and 1 (exclusive), "
            f"got {validation_fraction}"
        )

    engine_ids = np.array(sorted(df["unit_id"].unique()))
    n_engines = len(engine_ids)
    n_val = max(1, int(round(n_engines * validation_fraction)))

    # Deterministic shuffle
    rng = np.random.RandomState(random_state)
    shuffled = rng.permutation(engine_ids)

    val_engines = set(shuffled[:n_val])
    train_engines = set(shuffled[n_val:])

    val_mask = df["unit_id"].isin(val_engines)
    train_df = df[~val_mask].copy().reset_index(drop=True)
    val_df = df[val_mask].copy().reset_index(drop=True)

    return train_df, val_df


# ═══════════════════════════════════════════════════════════════════════════
# 2. SPLIT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Validate that a train/validation split is leakage-free.

    Checks:
        1. No engine in both partitions (empty intersection)
        2. All rows preserved (no loss or duplication)
        3. Correct engine counts

    Parameters
    ----------
    train_df : pd.DataFrame
        Training partition.
    val_df : pd.DataFrame
        Validation partition.
    original_df : pd.DataFrame, optional
        The original DataFrame before splitting. If provided, verifies
        total row count is preserved.

    Returns
    -------
    dict
        Validation results.
    """
    train_engines = set(train_df["unit_id"].unique())
    val_engines = set(val_df["unit_id"].unique())
    overlap = train_engines & val_engines

    results = {
        "no_engine_overlap": len(overlap) == 0,
        "overlap_engines": overlap,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "n_total_engines": len(train_engines) + len(val_engines),
        "n_train_rows": len(train_df),
        "n_val_rows": len(val_df),
        "n_total_rows": len(train_df) + len(val_df),
        "val_engine_fraction": round(
            len(val_engines) / (len(train_engines) + len(val_engines)), 4
        ),
    }

    # Row preservation check
    if original_df is not None:
        expected_rows = len(original_df)
        actual_rows = len(train_df) + len(val_df)
        results["rows_preserved"] = actual_rows == expected_rows
        results["expected_rows"] = expected_rows
        results["actual_rows"] = actual_rows
    else:
        results["rows_preserved"] = None

    # Distribution comparison
    train_lengths = train_df.groupby("unit_id")["cycle"].max()
    val_lengths = val_df.groupby("unit_id")["cycle"].max()

    results["train_trajectory_stats"] = {
        "mean": round(float(train_lengths.mean()), 2),
        "median": float(train_lengths.median()),
        "min": int(train_lengths.min()),
        "max": int(train_lengths.max()),
    }
    results["val_trajectory_stats"] = {
        "mean": round(float(val_lengths.mean()), 2),
        "median": float(val_lengths.median()),
        "min": int(val_lengths.min()),
        "max": int(val_lengths.max()),
    }

    results["all_valid"] = (
        results["no_engine_overlap"]
        and (results["rows_preserved"] is None or results["rows_preserved"])
    )

    return results
