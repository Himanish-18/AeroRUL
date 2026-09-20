"""
AeroRUL — Feature Quality Validation

Comprehensive validation for generated temporal features:
- NaN/Inf detection
- Constant feature detection
- Duplicate detection
- Target integrity checks
- Engine separation verification
- Feature-target correlation analysis
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import METADATA_COLUMNS, TARGET_COLUMN
from src.features import get_feature_columns


# ═══════════════════════════════════════════════════════════════════════════
# 1. FEATURE QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_feature_quality(df: pd.DataFrame) -> dict:
    """
    Comprehensive quality check on a feature dataset.

    Checks:
        1. NaN count per column and total
        2. Inf count per column and total
        3. Constant features (zero variance)
        4. Duplicate columns (exact copies)
        5. Duplicate rows

    Returns
    -------
    dict
        Quality metrics with detailed breakdowns.
    """
    feat_cols = get_feature_columns(df)

    # NaN analysis
    nan_per_col = df[feat_cols].isna().sum()
    total_nan = int(nan_per_col.sum())
    cols_with_nan = nan_per_col[nan_per_col > 0]

    # Inf analysis
    inf_per_col = df[feat_cols].apply(lambda s: np.isinf(s).sum() if np.issubdtype(s.dtype, np.floating) else 0)
    total_inf = int(inf_per_col.sum())
    cols_with_inf = inf_per_col[inf_per_col > 0]

    # Constant features (zero variance)
    variances = df[feat_cols].var()
    constant_cols = variances[variances == 0].index.tolist()

    # High variance features (top 10 by variance)
    high_var_cols = variances.nlargest(10)

    # Duplicate columns
    seen = {}
    dup_col_groups = []
    for col in feat_cols:
        col_hash = hash(tuple(df[col].values))
        if col_hash in seen:
            dup_col_groups.append((seen[col_hash], col))
        else:
            seen[col_hash] = col

    # Duplicate rows
    n_dup_rows = int(df[feat_cols].duplicated().sum())

    return {
        "n_features": len(feat_cols),
        "n_rows": len(df),
        "total_nan": total_nan,
        "n_cols_with_nan": len(cols_with_nan),
        "cols_with_nan": cols_with_nan.to_dict() if len(cols_with_nan) > 0 else {},
        "total_inf": total_inf,
        "n_cols_with_inf": len(cols_with_inf),
        "cols_with_inf": cols_with_inf.to_dict() if len(cols_with_inf) > 0 else {},
        "n_constant_features": len(constant_cols),
        "constant_features": constant_cols,
        "n_duplicate_column_pairs": len(dup_col_groups),
        "duplicate_column_pairs": dup_col_groups,
        "n_duplicate_rows": n_dup_rows,
        "high_variance_features": high_var_cols.to_dict(),
        "all_clean": total_nan == 0 and total_inf == 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. TARGET INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════

def validate_target_integrity(
    feature_df: pd.DataFrame,
    original_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Verify that the target (RUL) in the feature dataset is correct.

    Checks:
        1. RUL >= 0
        2. RUL column exists
        3. No failure_cycle column present (leakage)
        4. If original_df provided, RUL values match for matching (unit_id, cycle)

    Returns
    -------
    dict
        Target validation results.
    """
    results = {}

    # RUL column exists
    results["rul_column_exists"] = TARGET_COLUMN in feature_df.columns

    if not results["rul_column_exists"]:
        results["all_valid"] = False
        return results

    # RUL >= 0
    n_negative = int((feature_df[TARGET_COLUMN] < 0).sum())
    results["rul_non_negative"] = n_negative == 0
    results["n_negative_rul"] = n_negative

    # No failure_cycle leakage
    results["no_failure_cycle_leakage"] = "failure_cycle" not in feature_df.columns

    # RUL consistency with original
    if original_df is not None and TARGET_COLUMN in original_df.columns:
        merged = feature_df[["unit_id", "cycle", TARGET_COLUMN]].merge(
            original_df[["unit_id", "cycle", TARGET_COLUMN]],
            on=["unit_id", "cycle"],
            suffixes=("_feat", "_orig"),
        )
        n_mismatch = int(
            (merged[f"{TARGET_COLUMN}_feat"] != merged[f"{TARGET_COLUMN}_orig"]).sum()
        )
        results["rul_matches_original"] = n_mismatch == 0
        results["n_rul_mismatches"] = n_mismatch
    else:
        results["rul_matches_original"] = None

    results["all_valid"] = (
        results["rul_non_negative"]
        and results["no_failure_cycle_leakage"]
        and (results["rul_matches_original"] is None or results["rul_matches_original"])
    )

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 3. ENGINE SEPARATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_engine_separation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Verify no engine appears in both train and validation feature datasets.
    """
    train_engines = set(train_df["unit_id"].unique())
    val_engines = set(val_df["unit_id"].unique())
    overlap = train_engines & val_engines

    return {
        "no_overlap": len(overlap) == 0,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "overlap_engines": overlap,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE-TARGET CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════

def feature_target_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Compute correlation between each feature and the RUL target.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset with RUL column.
    method : str
        'pearson' or 'spearman'.
    top_n : int
        Return top N features by absolute correlation.

    Returns
    -------
    pd.DataFrame
        Columns: feature, correlation, abs_correlation
    """
    feat_cols = get_feature_columns(df)

    # Drop columns with all NaN
    valid_cols = [c for c in feat_cols if df[c].notna().any()]

    corrs = df[valid_cols].corrwith(df[TARGET_COLUMN], method=method)
    result = pd.DataFrame({
        "feature": corrs.index,
        "correlation": corrs.values,
        "abs_correlation": np.abs(corrs.values),
    })
    result = result.dropna(subset=["correlation"])
    result = result.sort_values("abs_correlation", ascending=False).head(top_n)
    result = result.reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. DATA RETENTION REPORT
# ═══════════════════════════════════════════════════════════════════════════

def data_retention_report(
    original_rows: int,
    feature_rows: int,
    original_engines: int,
    feature_engines: int,
    window_size: int,
) -> dict:
    """
    Report how many observations were discarded due to insufficient history.
    """
    discarded = original_rows - feature_rows
    return {
        "window_size": window_size,
        "original_rows": original_rows,
        "feature_rows": feature_rows,
        "discarded_rows": discarded,
        "retention_pct": round(100.0 * feature_rows / original_rows, 2) if original_rows > 0 else 0,
        "original_engines": original_engines,
        "feature_engines": feature_engines,
        "engines_preserved": original_engines == feature_engines,
    }
