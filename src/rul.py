"""
AeroRUL — RUL Target Generation, Validation, and Clipping

Constructs Remaining Useful Life targets for C-MAPSS training data.
RUL = failure_cycle - current_cycle, where failure_cycle is the maximum
observed cycle for each engine in the training set.

Usage:
    from src.rul import generate_rul_targets, clip_rul

    df_with_rul = generate_rul_targets(train_df)
    df_clipped  = clip_rul(df_with_rul, rul_cap=125)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import RUL_CLIP_VALUES


# ═══════════════════════════════════════════════════════════════════════════
# 1. RUL TARGET GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_rul_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate RUL targets for training data.

    For each engine, RUL = max_cycle - current_cycle.
    The final observed cycle has RUL = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with at least 'unit_id' and 'cycle' columns.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with an added 'rul' column.
    """
    df = df.copy()

    # Calculate failure cycle (max cycle) per engine
    failure_cycles = df.groupby("unit_id")["cycle"].transform("max")

    # RUL = failure_cycle - current_cycle
    df["rul"] = failure_cycles - df["cycle"]

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2. RUL VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_rul_targets(df: pd.DataFrame) -> dict:
    """
    Validate that RUL targets satisfy all required properties.

    Checks:
        1. RUL >= 0 everywhere
        2. min(RUL) == 0 for every engine (final cycle = failure)
        3. max(RUL) == failure_cycle - 1 for every engine
        4. RUL decreases by exactly 1 per consecutive cycle within each engine
        5. RUL + cycle == failure_cycle for every row

    Returns
    -------
    dict
        Validation results with keys:
            all_valid, rul_non_negative, min_rul_zero_per_engine,
            max_rul_correct, rul_decreases_by_one, rul_plus_cycle_invariant,
            n_engines, n_violations (per check)
    """
    if "rul" not in df.columns:
        raise ValueError("DataFrame must contain a 'rul' column. "
                         "Run generate_rul_targets first.")

    results = {}

    # 1. RUL >= 0
    n_negative = int((df["rul"] < 0).sum())
    results["rul_non_negative"] = n_negative == 0
    results["n_negative_rul"] = n_negative

    # 2. min(RUL) == 0 per engine
    min_rul_per_engine = df.groupby("unit_id")["rul"].min()
    engines_not_zero = min_rul_per_engine[min_rul_per_engine != 0]
    results["min_rul_zero_per_engine"] = len(engines_not_zero) == 0
    results["n_engines_min_rul_not_zero"] = len(engines_not_zero)

    # 3. max(RUL) == failure_cycle - 1
    max_rul_per_engine = df.groupby("unit_id")["rul"].max()
    failure_cycles = df.groupby("unit_id")["cycle"].max()
    expected_max_rul = failure_cycles - 1
    max_rul_mismatches = (max_rul_per_engine != expected_max_rul).sum()
    results["max_rul_correct"] = int(max_rul_mismatches) == 0
    results["n_max_rul_mismatches"] = int(max_rul_mismatches)

    # 4. RUL decreases by exactly 1 per consecutive cycle
    n_violations = 0
    for uid, group in df.groupby("unit_id"):
        group_sorted = group.sort_values("cycle")
        rul_diff = group_sorted["rul"].diff().dropna()
        cycle_diff = group_sorted["cycle"].diff().dropna()
        # For consecutive cycles (diff=1), RUL should decrease by 1 (diff=-1)
        mask = cycle_diff == 1
        bad = (rul_diff[mask] != -1).sum()
        n_violations += int(bad)
    results["rul_decreases_by_one"] = n_violations == 0
    results["n_decrease_violations"] = n_violations

    # 5. RUL + cycle == failure_cycle
    failure_cycle_per_row = df.groupby("unit_id")["cycle"].transform("max")
    invariant_check = (df["rul"] + df["cycle"]) == failure_cycle_per_row
    n_invariant_fail = int((~invariant_check).sum())
    results["rul_plus_cycle_invariant"] = n_invariant_fail == 0
    results["n_invariant_failures"] = n_invariant_fail

    # Summary
    results["n_engines"] = int(df["unit_id"].nunique())
    results["all_valid"] = all([
        results["rul_non_negative"],
        results["min_rul_zero_per_engine"],
        results["max_rul_correct"],
        results["rul_decreases_by_one"],
        results["rul_plus_cycle_invariant"],
    ])

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. RUL STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def rul_statistics(df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for the RUL column.

    Returns
    -------
    dict
        min_rul, max_rul, mean_rul, median_rul, std_rul, n_observations
    """
    if "rul" not in df.columns:
        raise ValueError("DataFrame must contain a 'rul' column.")

    return {
        "min_rul": int(df["rul"].min()),
        "max_rul": int(df["rul"].max()),
        "mean_rul": round(float(df["rul"].mean()), 2),
        "median_rul": float(df["rul"].median()),
        "std_rul": round(float(df["rul"].std()), 2),
        "n_observations": len(df),
        "n_engines": int(df["unit_id"].nunique()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. RUL CLIPPING
# ═══════════════════════════════════════════════════════════════════════════

def clip_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """
    Apply RUL clipping: RUL_clipped = min(RUL, rul_cap).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'rul' column.
    rul_cap : int
        Maximum allowed RUL value.

    Returns
    -------
    pd.DataFrame
        Copy with 'rul_clipped' column added (original 'rul' preserved).
    """
    if "rul" not in df.columns:
        raise ValueError("DataFrame must contain a 'rul' column.")

    df = df.copy()
    df["rul_clipped"] = df["rul"].clip(upper=rul_cap)
    return df


def rul_clipping_comparison(
    df: pd.DataFrame,
    caps: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compare the statistical effect of different RUL clipping thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Training DataFrame with 'rul' column.
    caps : list, optional
        List of cap values to evaluate. None entries mean no clipping.
        Defaults to config.RUL_CLIP_VALUES.

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
            rul_cap, observations_clipped, pct_clipped,
            engines_affected, mean_rul, median_rul, std_rul
    """
    if "rul" not in df.columns:
        raise ValueError("DataFrame must contain a 'rul' column.")

    if caps is None:
        caps = RUL_CLIP_VALUES

    n_total = len(df)
    n_engines_total = df["unit_id"].nunique()
    rows = []

    for cap in caps:
        if cap is None:
            rows.append({
                "rul_cap": "None",
                "observations_clipped": 0,
                "pct_clipped": 0.0,
                "engines_affected": 0,
                "mean_rul": round(float(df["rul"].mean()), 2),
                "median_rul": float(df["rul"].median()),
                "std_rul": round(float(df["rul"].std()), 2),
            })
        else:
            clipped = df["rul"].clip(upper=cap)
            mask_clipped = df["rul"] > cap
            n_clipped = int(mask_clipped.sum())
            engines_affected = int(df.loc[mask_clipped, "unit_id"].nunique())

            rows.append({
                "rul_cap": int(cap),
                "observations_clipped": n_clipped,
                "pct_clipped": round(100.0 * n_clipped / n_total, 2),
                "engines_affected": engines_affected,
                "mean_rul": round(float(clipped.mean()), 2),
                "median_rul": float(clipped.median()),
                "std_rul": round(float(clipped.std()), 2),
            })

    return pd.DataFrame(rows)
