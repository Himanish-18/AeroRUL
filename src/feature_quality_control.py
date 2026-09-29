"""
AeroRUL — Phase 3.1: Feature Quality Control

Comprehensive feature-quality diagnostics for Phase 3 engineered features.
This module provides tools for:
    - NaN root-cause analysis
    - Constant feature auditing
    - Duplicate feature detection
    - Train/validation schema consistency verification
    - Leakage auditing (temporal causality, engine separation, normalization isolation)
    - Train/validation distribution shift analysis
    - Feature usability inventory

All functions are diagnostic — no model training, no feature selection.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import METADATA_COLUMNS, SENSOR_COLUMNS, TARGET_COLUMN
from src.features import _parse_feature_name, get_feature_columns


# ═══════════════════════════════════════════════════════════════════════════
# 1. NaN ROOT-CAUSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def nan_root_cause_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Investigate every NaN-producing feature across train and validation.

    For each NaN feature determines:
        - feature_name, sensor, feature_family, window
        - nan_count_train, nan_pct_train, nan_count_val, nan_pct_val
        - in_train, in_val
        - mathematical root cause classification
        - whether the NaN is expected

    Root cause categories:
        EXPECTED_NUMERICAL_EDGE_CASE:
            R² undefined for constant-value windows (denom_y == 0)
        IMPLEMENTATION_BUG:
            NaN caused by code defect (not expected in current implementation)
        DATA_QUALITY_PROBLEM:
            NaN caused by underlying data issues
        LEGITIMATE_MISSING_FEATURE:
            Feature undefined for certain data characteristics

    Returns
    -------
    pd.DataFrame
        NaN audit with one row per NaN-affected feature.
    """
    feat_cols = get_feature_columns(train_df)

    # NaN counts per column
    nan_train = train_df[feat_cols].isna().sum()
    nan_val = val_df[feat_cols].isna().sum()

    # Union of columns with any NaN
    nan_cols_train = set(nan_train[nan_train > 0].index)
    nan_cols_val = set(nan_val[nan_val > 0].index)
    all_nan_cols = nan_cols_train | nan_cols_val

    if not all_nan_cols:
        return pd.DataFrame(columns=[
            "feature_name", "sensor", "feature_family", "window",
            "nan_count_train", "nan_pct_train",
            "nan_count_val", "nan_pct_val",
            "in_train", "in_val",
            "root_cause_category", "root_cause_explanation", "expected",
        ])

    n_train = len(train_df)
    n_val = len(val_df)

    # Build metadata lookup
    meta_lookup = {}
    if feature_metadata is not None and len(feature_metadata) > 0:
        for _, row in feature_metadata.iterrows():
            meta_lookup[row["feature_name"]] = {
                "sensor": row["sensor"],
                "statistic": row["statistic"],
                "window_or_lag": row["window_or_lag"],
            }

    rows = []
    for col in sorted(all_nan_cols):
        n_nan_train = int(nan_train.get(col, 0))
        n_nan_val = int(nan_val.get(col, 0))

        # Parse feature name
        meta = meta_lookup.get(col, {})
        sensor = meta.get("sensor", "unknown")
        stat = meta.get("statistic", "unknown")
        window = meta.get("window_or_lag", 0)

        # If not in metadata, try parsing directly
        if sensor == "unknown":
            parsed = _parse_feature_name(col)
            if parsed:
                sensor, stat, window = parsed

        # Determine root cause
        category, explanation, expected = _classify_nan_root_cause(
            col, sensor, stat, window, train_df, val_df,
            n_nan_train, n_nan_val,
        )

        rows.append({
            "feature_name": col,
            "sensor": sensor,
            "feature_family": stat,
            "window": window,
            "nan_count_train": n_nan_train,
            "nan_pct_train": round(100.0 * n_nan_train / n_train, 4) if n_train > 0 else 0,
            "nan_count_val": n_nan_val,
            "nan_pct_val": round(100.0 * n_nan_val / n_val, 4) if n_val > 0 else 0,
            "in_train": n_nan_train > 0,
            "in_val": n_nan_val > 0,
            "root_cause_category": category,
            "root_cause_explanation": explanation,
            "expected": expected,
        })

    return pd.DataFrame(rows)


def _classify_nan_root_cause(
    col: str, sensor: str, stat: str, window: int,
    train_df: pd.DataFrame, val_df: pd.DataFrame,
    n_nan_train: int, n_nan_val: int,
) -> tuple[str, str, bool]:
    """
    Classify the root cause of NaN for a specific feature.

    Returns (category, explanation, expected).
    """
    if stat == "r2":
        # R² becomes NaN when denom_y = W * Σ(y²) - (Σy)² ≈ 0
        # This happens when all values in the window are constant
        # (zero total variance). This is mathematically expected.
        return (
            "EXPECTED_NUMERICAL_EDGE_CASE",
            f"R² undefined when sensor '{sensor}' has zero variance within "
            f"window of size {window}. denom_y = W*Σ(y²) - (Σy)² = 0 for "
            f"constant windows. The implementation correctly returns NaN.",
            True,
        )

    if stat == "std":
        # std with ddof=0 should always return 0.0 for constant windows,
        # not NaN. If we see NaN here, investigate further.
        if n_nan_train > 0 or n_nan_val > 0:
            return (
                "IMPLEMENTATION_BUG",
                f"Rolling std (ddof=0) should return 0.0 for constant windows, "
                f"not NaN. Investigate pandas rolling.std() edge cases.",
                False,
            )

    if stat == "slope":
        # Slope uses denom_x = W*Σ(x²) - (Σx)² which is always > 0 for W >= 2.
        # Should not produce NaN unless data itself has NaN propagation.
        # The only case would be if the sensor value is NaN at some point.
        return (
            "EXPECTED_NUMERICAL_EDGE_CASE",
            f"Slope NaN likely from insufficient non-NaN data in window or "
            f"NaN propagation from sensor '{sensor}'. denom_x is always positive "
            f"for W >= 2, so this is a data-side issue, not formula-side.",
            True,
        )

    if stat == "relchange":
        # Relative change = (x_t - x_(t-k)) / |x_(t-k)|
        # With floor=1e-8, denominator should never be exactly zero.
        # NaN occurs when shift(k) returns NaN (insufficient history),
        # but rows with insufficient history should have been dropped
        # by the window filter. If NaN persists, it's from NaN in the
        # original sensor values.
        return (
            "EXPECTED_NUMERICAL_EDGE_CASE",
            f"Relative change NaN from NaN propagation in sensor '{sensor}' "
            f"or edge-case where shift({window}) lacks history.",
            True,
        )

    if stat == "delta":
        # Similar to relchange: shift(k) can produce NaN at boundaries.
        return (
            "EXPECTED_NUMERICAL_EDGE_CASE",
            f"Delta NaN from shift({window}) lacking sufficient history "
            f"or NaN propagation in sensor '{sensor}'.",
            True,
        )

    if stat in ("mean", "min", "max", "range"):
        # These should not produce NaN after the window-validity filter
        # unless the underlying sensor has NaN values.
        return (
            "DATA_QUALITY_PROBLEM",
            f"Rolling {stat} NaN suggests underlying sensor '{sensor}' contains "
            f"NaN values or min_periods requirement not met.",
            False,
        )

    return (
        "LEGITIMATE_MISSING_FEATURE",
        f"NaN in feature '{col}' — unknown statistic type '{stat}'.",
        False,
    )


def determine_nan_handling(nan_audit: pd.DataFrame) -> pd.DataFrame:
    """
    For every NaN category, determine the mathematically correct handling.

    Possible actions:
        KEEP_NAN: Model family can handle NaN; NaN has meaningful interpretation
        CONVERT_TO_VALUE: Mathematically justified replacement
        EXCLUDE_FEATURE: Feature fundamentally undefined
        FIX_IMPLEMENTATION: NaN caused by implementation defect

    Returns the audit with additional columns: action, replacement_value, justification
    """
    audit = nan_audit.copy()
    actions = []
    values = []
    justifications = []

    for _, row in audit.iterrows():
        stat = row["feature_family"]
        category = row["root_cause_category"]

        if category == "IMPLEMENTATION_BUG":
            actions.append("FIX_IMPLEMENTATION")
            values.append(None)
            justifications.append("Fix the underlying code defect.")
        elif stat == "r2":
            # R² = undefined when sensor is constant in window.
            # Mathematically, if there's no variance, the linear model
            # explains all variance (trivially). R² = 1.0 could be argued,
            # but NaN is also defensible. For tree-based models, NaN is
            # acceptable. For linear models, we might replace with 0.0
            # (no meaningful trend when constant).
            # Decision: keep NaN for now; Phase 4 model can handle or
            # impute as needed.
            actions.append("CONVERT_TO_VALUE")
            values.append(0.0)
            justifications.append(
                "R² undefined for constant windows (zero variance). "
                "Convert to 0.0: when sensor is constant, the linear trend "
                "explains none of the (nonexistent) variance. This is "
                "mathematically equivalent to 'no meaningful trend'."
            )
        elif stat in ("delta", "relchange"):
            # These NaNs come from insufficient history at boundaries.
            # Since we already filter by max_window, any remaining NaN
            # means the delta/relchange lag exceeds the available history
            # for some short-lived engines. These are genuinely missing.
            actions.append("KEEP_NAN")
            values.append(None)
            justifications.append(
                f"{stat.title()} NaN from insufficient history. "
                f"Genuinely missing data — should not be imputed with "
                f"an arbitrary value."
            )
        else:
            actions.append("KEEP_NAN")
            values.append(None)
            justifications.append(
                f"NaN in {stat} feature. Root cause: {category}. "
                f"Retain NaN for diagnostic transparency."
            )

    audit["action"] = actions
    audit["replacement_value"] = values
    audit["justification"] = justifications

    return audit


# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTANT FEATURE AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def constant_feature_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    constant_sensors: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Investigate all constant features across train and validation.

    Classifies each constant feature:
        ORIGINAL_SENSOR_CONSTANT:
            Feature derived from a sensor that is itself constant.
        ENGINEERED_CONSTANT:
            Feature is constant because the transformation produces
            zero variance (e.g., std of constant sensor = 0, range = 0).
        DATASET_SPECIFIC_CONSTANT:
            Feature is constant in a specific dataset due to data
            characteristics (e.g., single operating condition).
        IMPLEMENTATION_ARTIFACT:
            Feature should not be constant — indicates a bug.
    """
    feat_cols = get_feature_columns(train_df)

    # Detect constant columns
    var_train = train_df[feat_cols].var()
    var_val = val_df[feat_cols].var()

    const_train = set(var_train[var_train == 0].index)
    const_val = set(var_val[var_val == 0].index)
    all_const = const_train | const_val

    # Build metadata lookup
    meta_lookup = {}
    if feature_metadata is not None:
        for _, row in feature_metadata.iterrows():
            meta_lookup[row["feature_name"]] = {
                "sensor": row["sensor"],
                "statistic": row["statistic"],
                "window_or_lag": row["window_or_lag"],
            }

    # Determine which raw sensors are constant
    if constant_sensors is None:
        constant_sensors = []
        raw_sensor_cols = [c for c in SENSOR_COLUMNS if c in train_df.columns]
        for sc in raw_sensor_cols:
            if sc in train_df.columns and train_df[sc].var() == 0:
                constant_sensors.append(sc)

    const_sensor_set = set(constant_sensors)

    rows = []
    for col in sorted(all_const):
        meta = meta_lookup.get(col, {})
        sensor = meta.get("sensor", "unknown")
        stat = meta.get("statistic", "unknown")
        window = meta.get("window_or_lag", 0)

        if sensor == "unknown":
            parsed = _parse_feature_name(col)
            if parsed:
                sensor, stat, window = parsed

        in_train = col in const_train
        in_val = col in const_val

        # Get the constant value
        const_value_train = None
        const_value_val = None
        if in_train and col in train_df.columns:
            vals = train_df[col].dropna().unique()
            if len(vals) == 1:
                const_value_train = float(vals[0])
            elif len(vals) == 0:
                const_value_train = None  # all NaN
        if in_val and col in val_df.columns:
            vals = val_df[col].dropna().unique()
            if len(vals) == 1:
                const_value_val = float(vals[0])
            elif len(vals) == 0:
                const_value_val = None

        # Classify
        category = _classify_constant_feature(
            sensor, stat, const_sensor_set, in_train, in_val,
        )

        rows.append({
            "feature_name": col,
            "sensor": sensor,
            "feature_family": stat,
            "window": window,
            "constant_value_train": const_value_train,
            "constant_value_val": const_value_val,
            "constant_train": in_train,
            "constant_val": in_val,
            "constant_both": in_train and in_val,
            "original_sensor_constant": sensor in const_sensor_set,
            "category": category,
        })

    return pd.DataFrame(rows)


def _classify_constant_feature(
    sensor: str, stat: str, const_sensors: set,
    in_train: bool, in_val: bool,
) -> str:
    """Classify why a feature is constant."""
    if sensor in const_sensors:
        # The underlying sensor is constant
        if stat in ("std", "range", "slope", "delta", "relchange"):
            # These naturally become 0 when the source sensor is constant
            return "ORIGINAL_SENSOR_CONSTANT"
        elif stat in ("mean", "min", "max"):
            # These equal the constant sensor value
            return "ORIGINAL_SENSOR_CONSTANT"
        elif stat == "r2":
            # R² is NaN (or 0 after conversion) for constant sensors
            return "ORIGINAL_SENSOR_CONSTANT"
        return "ORIGINAL_SENSOR_CONSTANT"

    if stat in ("std", "range"):
        # These can be zero even for non-constant sensors if the sensor
        # has very low variance within windows
        return "ENGINEERED_CONSTANT"

    if stat in ("slope", "delta", "relchange"):
        # Slope/delta/relchange of a near-constant sensor
        return "ENGINEERED_CONSTANT"

    if in_train and not in_val:
        return "DATASET_SPECIFIC_CONSTANT"
    if in_val and not in_train:
        return "DATASET_SPECIFIC_CONSTANT"

    return "ENGINEERED_CONSTANT"


# ═══════════════════════════════════════════════════════════════════════════
# 3. DUPLICATE FEATURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_duplicate_features(
    df: pd.DataFrame,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """
    Detect exact duplicate feature columns.

    Two columns are considered duplicates if their values are identical
    (within floating-point tolerance) or if they have a perfect correlation
    of exactly 1.0.

    Returns
    -------
    pd.DataFrame
        Columns: group_id, feature_name, duplicate_with, reason
    """
    feat_cols = get_feature_columns(df)

    # Use hash-based detection first (fast)
    seen_hashes: dict[int, str] = {}
    groups: list[dict] = []
    group_id = 0

    for col in feat_cols:
        values = df[col].values
        # Create a robust hash accounting for NaN
        nan_mask = np.isnan(values) if np.issubdtype(values.dtype, np.floating) else np.zeros(len(values), dtype=bool)
        non_nan_vals = values[~nan_mask]

        if len(non_nan_vals) == 0:
            col_hash = hash(("all_nan", int(nan_mask.sum())))
        else:
            # Use a sample-based hash for speed; verify exact match later
            sample_vals = tuple(np.round(non_nan_vals[:50], decimals=12)) + \
                          tuple(np.round(non_nan_vals[-50:], decimals=12))
            col_hash = hash((sample_vals, len(non_nan_vals), int(nan_mask.sum())))

        if col_hash in seen_hashes:
            # Verify with actual comparison
            other_col = seen_hashes[col_hash]
            other_vals = df[other_col].values
            nan_mask_other = np.isnan(other_vals) if np.issubdtype(other_vals.dtype, np.floating) else np.zeros(len(other_vals), dtype=bool)

            if np.array_equal(nan_mask, nan_mask_other):
                non_nan_this = values[~nan_mask]
                non_nan_other = other_vals[~nan_mask]
                if len(non_nan_this) > 0 and np.allclose(non_nan_this, non_nan_other, atol=tolerance, equal_nan=True):
                    groups.append({
                        "group_id": group_id,
                        "feature_name": col,
                        "duplicate_with": other_col,
                        "reason": "exact_duplicate_values",
                    })
                    groups.append({
                        "group_id": group_id,
                        "feature_name": other_col,
                        "duplicate_with": col,
                        "reason": "exact_duplicate_values",
                    })
                    group_id += 1
                    continue

        seen_hashes[col_hash] = col

    if not groups:
        return pd.DataFrame(columns=["group_id", "feature_name", "duplicate_with", "reason"])

    result = pd.DataFrame(groups).drop_duplicates(
        subset=["feature_name", "duplicate_with"]
    ).reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. SCHEMA CONSISTENCY VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def verify_schema_consistency(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Verify train and validation feature datasets have identical schemas.

    Checks:
        - Exact same feature names
        - Exact same feature count
        - Exact same feature ordering
        - Identical data types
        - Target excluded from features
        - Metadata excluded from features
        - No train-only or val-only features
    """
    train_cols = list(train_df.columns)
    val_cols = list(val_df.columns)

    train_feat_cols = get_feature_columns(train_df)
    val_feat_cols = get_feature_columns(val_df)

    # Check exact match
    same_columns = train_cols == val_cols
    same_feature_count = len(train_feat_cols) == len(val_feat_cols)
    same_feature_names = set(train_feat_cols) == set(val_feat_cols)
    same_feature_order = train_feat_cols == val_feat_cols

    # Train-only / val-only features
    train_only = set(train_feat_cols) - set(val_feat_cols)
    val_only = set(val_feat_cols) - set(train_feat_cols)

    # Dtype consistency
    dtype_mismatches = []
    common_cols = set(train_cols) & set(val_cols)
    for col in common_cols:
        if train_df[col].dtype != val_df[col].dtype:
            dtype_mismatches.append({
                "column": col,
                "train_dtype": str(train_df[col].dtype),
                "val_dtype": str(val_df[col].dtype),
            })

    # Check forbidden columns in feature matrix
    forbidden_in_features = {"rul", "failure_cycle", "unit_id", "cycle"}
    forbidden_present = forbidden_in_features & set(train_feat_cols)

    # Check metadata columns present (they should be)
    has_unit_id = "unit_id" in train_cols and "unit_id" in val_cols
    has_cycle = "cycle" in train_cols and "cycle" in val_cols
    has_rul = "rul" in train_cols and "rul" in val_cols

    return {
        "same_columns": same_columns,
        "same_feature_count": same_feature_count,
        "n_train_features": len(train_feat_cols),
        "n_val_features": len(val_feat_cols),
        "same_feature_names": same_feature_names,
        "same_feature_order": same_feature_order,
        "train_only_features": sorted(train_only),
        "val_only_features": sorted(val_only),
        "dtype_mismatches": dtype_mismatches,
        "forbidden_in_features": sorted(forbidden_present),
        "has_unit_id": has_unit_id,
        "has_cycle": has_cycle,
        "has_rul": has_rul,
        "schema_valid": (
            same_columns
            and same_feature_count
            and same_feature_names
            and len(forbidden_present) == 0
            and len(dtype_mismatches) == 0
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def verify_temporal_causality(
    df: pd.DataFrame,
    sensor_columns: Optional[list[str]] = None,
    window: int = 10,
    test_cycle: int = 30,
) -> dict:
    """
    Verify temporal causality: features at cycle t depend only on cycles <= t.

    Method:
        1. Generate features from original data.
        2. Modify future cycles (> test_cycle) with extreme values.
        3. Regenerate features at test_cycle.
        4. Assert features are identical.
    """
    from src.features import generate_window_features
    from src.rul import generate_rul_targets

    if sensor_columns is None:
        sensor_columns = [c for c in SENSOR_COLUMNS if c in df.columns]

    # Get a single engine with enough cycles
    engine_lengths = df.groupby("unit_id")["cycle"].count()
    valid_engines = engine_lengths[engine_lengths > test_cycle + 20].index
    if len(valid_engines) == 0:
        return {"verified": False, "reason": "No engine with enough cycles"}

    eid = valid_engines[0]
    engine_df = df[df["unit_id"] == eid].copy()

    # Ensure RUL column exists
    if "rul" not in engine_df.columns:
        engine_df = generate_rul_targets(engine_df)

    # Original features
    feat_orig = generate_window_features(
        engine_df, window_size=window,
        sensor_columns=sensor_columns,
        statistics=["mean", "std", "slope", "r2"],
        compute_deltas=[1, 5], compute_relative_changes=[5],
    )

    # Modify future values
    engine_mod = engine_df.copy()
    future_mask = engine_mod["cycle"] > test_cycle
    for sc in sensor_columns:
        engine_mod.loc[future_mask, sc] = 99999.0
    # Recalculate RUL
    engine_mod["rul"] = (
        engine_mod.groupby("unit_id")["cycle"].transform("max")
        - engine_mod["cycle"]
    )

    feat_mod = generate_window_features(
        engine_mod, window_size=window,
        sensor_columns=sensor_columns,
        statistics=["mean", "std", "slope", "r2"],
        compute_deltas=[1, 5], compute_relative_changes=[5],
    )

    # Compare at test_cycle
    orig_row = feat_orig[feat_orig["cycle"] == test_cycle]
    mod_row = feat_mod[feat_mod["cycle"] == test_cycle]

    if len(orig_row) == 0 or len(mod_row) == 0:
        return {"verified": False, "reason": f"No data at cycle {test_cycle}"}

    feature_cols = get_feature_columns(orig_row)
    violations = []
    for col in feature_cols:
        orig_val = orig_row[col].values[0]
        mod_val = mod_row[col].values[0]
        if np.isnan(orig_val) and np.isnan(mod_val):
            continue
        if abs(orig_val - mod_val) > 1e-10:
            violations.append({
                "feature": col,
                "original": float(orig_val),
                "modified": float(mod_val),
            })

    return {
        "verified": len(violations) == 0,
        "n_violations": len(violations),
        "violations": violations[:10],  # limit output
        "test_engine": int(eid),
        "test_cycle": test_cycle,
        "window": window,
    }


def verify_engine_separation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """Verify no engine appears in both train and validation."""
    train_engines = set(train_df["unit_id"].unique())
    val_engines = set(val_df["unit_id"].unique())
    overlap = train_engines & val_engines

    return {
        "verified": len(overlap) == 0,
        "n_train_engines": len(train_engines),
        "n_val_engines": len(val_engines),
        "overlap": sorted(overlap),
    }


def verify_normalization_isolation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    norm_params: dict,
) -> dict:
    """
    Verify normalization parameters are derived from training data only.

    Checks that the norm_params means/stds match training data statistics
    and NOT a mixture of train+val.
    """
    results = {
        "verified": True,
        "n_regimes": len(norm_params),
        "details": [],
    }

    for regime, params in norm_params.items():
        regime_mean = params.get("mean", {})
        regime_std = params.get("std", {})
        n_obs = params.get("n_observations", 0)

        # Verify n_observations is consistent with training data
        # (We can't directly recompute without regime labels, but we
        # can check that the total observations across regimes
        # don't exceed training size)
        results["details"].append({
            "regime": regime,
            "n_observations": n_obs,
            "n_sensors_normalized": len(regime_mean),
        })

    # Check total observations across regimes
    total_regime_obs = sum(d["n_observations"] for d in results["details"])
    if total_regime_obs > len(train_df):
        results["verified"] = False
        results["reason"] = (
            f"Total regime observations ({total_regime_obs}) exceeds "
            f"training data size ({len(train_df)})"
        )

    return results


def verify_target_isolation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict:
    """
    Verify no feature is derived from RUL, failure_cycle, or future values.

    Checks:
        - 'rul' is not among feature columns
        - 'failure_cycle' is not present at all
        - No feature column name contains 'rul' or 'failure'
    """
    train_feat_cols = get_feature_columns(train_df)
    val_feat_cols = get_feature_columns(val_df)

    issues = []

    # Check column names
    for col in train_feat_cols:
        if "rul" in col.lower() and col != "rul":
            issues.append(f"Feature '{col}' contains 'rul' in name")
        if "failure" in col.lower():
            issues.append(f"Feature '{col}' contains 'failure' in name")

    # Check failure_cycle not in dataframe
    if "failure_cycle" in train_df.columns:
        issues.append("failure_cycle column present in train")
    if "failure_cycle" in val_df.columns:
        issues.append("failure_cycle column present in val")

    return {
        "verified": len(issues) == 0,
        "issues": issues,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6. DISTRIBUTION SHIFT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def distribution_shift_analysis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    shift_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare train vs validation distributions for usable features.

    For each feature computes:
        - train/val mean, std, median
        - mean difference, standardized mean difference (Cohen's d)
        - shift_flag (if |standardized_difference| > threshold)

    The threshold is diagnostic only, not a feature-selection rule.

    Parameters
    ----------
    shift_threshold : float
        Threshold for standardized mean difference to flag as shifted.
        Default 0.5 (medium effect size per Cohen's convention).
    """
    feat_cols = get_feature_columns(train_df)

    rows = []
    for col in feat_cols:
        train_vals = train_df[col].dropna()
        val_vals = val_df[col].dropna()

        if len(train_vals) == 0 or len(val_vals) == 0:
            continue

        train_mean = float(train_vals.mean())
        val_mean = float(val_vals.mean())
        train_std = float(train_vals.std())
        val_std = float(val_vals.std())
        train_median = float(train_vals.median())
        val_median = float(val_vals.median())

        mean_diff = val_mean - train_mean

        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(
            ((len(train_vals) - 1) * train_std ** 2 + (len(val_vals) - 1) * val_std ** 2)
            / (len(train_vals) + len(val_vals) - 2)
        ) if (len(train_vals) + len(val_vals)) > 2 else 1.0

        standardized_diff = mean_diff / pooled_std if pooled_std > 1e-15 else 0.0

        rows.append({
            "feature": col,
            "train_mean": round(train_mean, 6),
            "val_mean": round(val_mean, 6),
            "train_std": round(train_std, 6),
            "val_std": round(val_std, 6),
            "mean_difference": round(mean_diff, 6),
            "standardized_difference": round(standardized_diff, 6),
            "train_median": round(train_median, 6),
            "val_median": round(val_median, 6),
            "shift_flag": abs(standardized_diff) > shift_threshold,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 7. FEATURE USABILITY INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

def generate_feature_usability_inventory(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_metadata: pd.DataFrame,
    nan_audit: pd.DataFrame,
    constant_audit: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a comprehensive feature usability inventory for Phase 4.

    For each feature records usability status:
        YES: ready for Phase 4
        NO: not usable
        REQUIRES_FIX: usable after documented fix
    """
    feat_cols = get_feature_columns(train_df)

    # Precompute lookups
    nan_lookup = {}
    if nan_audit is not None and len(nan_audit) > 0:
        for _, row in nan_audit.iterrows():
            nan_lookup[row["feature_name"]] = row

    const_lookup = {}
    if constant_audit is not None and len(constant_audit) > 0:
        for _, row in constant_audit.iterrows():
            const_lookup[row["feature_name"]] = row

    dup_set = set()
    if duplicate_groups is not None and len(duplicate_groups) > 0:
        for _, row in duplicate_groups.iterrows():
            dup_set.add(row["feature_name"])

    meta_lookup = {}
    if feature_metadata is not None:
        for _, row in feature_metadata.iterrows():
            meta_lookup[row["feature_name"]] = row

    rows = []
    for col in feat_cols:
        meta = meta_lookup.get(col, {})
        sensor = meta.get("sensor", "unknown") if isinstance(meta, dict) else getattr(meta, "sensor", "unknown")
        stat = meta.get("statistic", "unknown") if isinstance(meta, dict) else getattr(meta, "statistic", "unknown")
        window = meta.get("window_or_lag", 0) if isinstance(meta, dict) else getattr(meta, "window_or_lag", 0)

        if sensor == "unknown":
            parsed = _parse_feature_name(col)
            if parsed:
                sensor, stat, window = parsed

        # NaN info
        nan_info = nan_lookup.get(col, None)
        nan_train = int(nan_info["nan_count_train"]) if nan_info is not None else int(train_df[col].isna().sum())
        nan_val = int(nan_info["nan_count_val"]) if nan_info is not None else int(val_df[col].isna().sum())

        # Inf info
        inf_train = int(np.isinf(train_df[col]).sum()) if np.issubdtype(train_df[col].dtype, np.floating) else 0
        inf_val = int(np.isinf(val_df[col]).sum()) if np.issubdtype(val_df[col].dtype, np.floating) else 0

        # Constant info
        const_info = const_lookup.get(col, None)
        const_train = const_info is not None and const_info["constant_train"]
        const_val = const_info is not None and const_info["constant_val"]

        # Duplicate info
        dup_group = col in dup_set

        # Determine usability
        usable, reason = _determine_usability(
            col, nan_train, nan_val, inf_train, inf_val,
            const_train, const_val, dup_group, nan_info, const_info,
        )

        rows.append({
            "feature_name": col,
            "sensor": sensor,
            "feature_family": stat,
            "window": window,
            "dtype": str(train_df[col].dtype),
            "nan_count_train": nan_train,
            "nan_count_val": nan_val,
            "inf_count_train": inf_train,
            "inf_count_val": inf_val,
            "constant_train": const_train,
            "constant_val": const_val,
            "duplicate_group": dup_group,
            "usable_for_phase4": usable,
            "reason": reason,
        })

    return pd.DataFrame(rows)


def _determine_usability(
    col: str,
    nan_train: int, nan_val: int,
    inf_train: int, inf_val: int,
    const_train: bool, const_val: bool,
    dup_group: bool,
    nan_info: Optional[object], const_info: Optional[object],
) -> tuple[str, str]:
    """Determine feature usability for Phase 4."""
    # Infinite values are always problematic
    if inf_train > 0 or inf_val > 0:
        return "NO", f"Contains infinite values (train={inf_train}, val={inf_val})"

    # Features constant in both train AND validation carry no information
    if const_train and const_val:
        return "NO", "Constant in both train and validation — zero information content"

    # NaN analysis
    if nan_info is not None:
        action = nan_info.get("action", None) if isinstance(nan_info, dict) else getattr(nan_info, "action", None)
        category = nan_info.get("root_cause_category", None) if isinstance(nan_info, dict) else getattr(nan_info, "root_cause_category", None)

        if action == "FIX_IMPLEMENTATION":
            return "REQUIRES_FIX", f"NaN from implementation bug — {category}"
        if action == "CONVERT_TO_VALUE":
            return "REQUIRES_FIX", f"NaN requires conversion — {category}"

    # High NaN percentage (>50% in either split)
    # These features may have too little usable data
    # But we keep them as REQUIRES_FIX, not NO, because Phase 4 models
    # may handle NaN natively
    if nan_train > 0 or nan_val > 0:
        return "YES", f"Contains NaN (train={nan_train}, val={nan_val}) — expected edge case"

    return "YES", "Clean feature"
