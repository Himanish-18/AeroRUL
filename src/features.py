"""
AeroRUL — Sliding-Window Feature Engineering

Causal temporal feature generation for C-MAPSS degradation data.
All features at cycle t use ONLY information from cycles <= t.

Feature families:
    - Rolling statistics: mean, std, min, max, range
    - Trend features: slope (linear OLS), R² (goodness of fit)
    - Delta features: x_t - x_(t-k)
    - Relative change features: (x_t - x_(t-k)) / |x_(t-k)|

Usage:
    from src.features import generate_multiscale_features

    features_df = generate_multiscale_features(train_df, sensor_columns=SENSOR_COLUMNS)
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_CONFIG,
    METADATA_COLUMNS,
    RELATIVE_CHANGE_FLOOR,
    SENSOR_COLUMNS,
    TARGET_COLUMN,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ROLLING STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rolling_stats(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window: int,
    statistics: list[str],
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Compute rolling-window statistics for each sensor within each engine.

    Uses pd.DataFrame.rolling() which is inherently causal (backward-looking).

    Parameters
    ----------
    df : pd.DataFrame
        Sorted by (unit_id, cycle) with sensor columns.
    sensor_columns : list[str]
        Sensor columns to compute statistics for.
    window : int
        Window size.
    statistics : list[str]
        Which statistics to compute. Supports: mean, std, min, max, range.
    partial_history : bool
        If True, use min_periods=1 to support shorter histories causally.

    Returns
    -------
    pd.DataFrame
        New columns named {sensor}_{stat}_{window}.
    """
    result_dict = {}

    stat_basic = {"mean", "std", "min", "max"}
    requested_basic = [s for s in statistics if s in stat_basic]
    needs_range = "range" in statistics
    min_periods = 1 if partial_history else window

    for sensor in sensor_columns:
        series = df[sensor]

        # Group-aware rolling: we apply rolling per engine group
        rolling = series.groupby(df["unit_id"]).rolling(
            window=window, min_periods=min_periods
        )

        for stat in requested_basic:
            col_name = f"{sensor}_{stat}_{window}"
            if stat == "mean":
                result_dict[col_name] = rolling.mean().droplevel(0)
            elif stat == "std":
                s = rolling.std(ddof=0).droplevel(0)
                if partial_history:
                    s = s.fillna(0.0)
                result_dict[col_name] = s
            elif stat == "min":
                result_dict[col_name] = rolling.min().droplevel(0)
            elif stat == "max":
                result_dict[col_name] = rolling.max().droplevel(0)

        if needs_range:
            col_min = f"{sensor}_min_{window}"
            col_max = f"{sensor}_max_{window}"
            col_range = f"{sensor}_range_{window}"
            if col_min in result_dict and col_max in result_dict:
                result_dict[col_range] = result_dict[col_max] - result_dict[col_min]
            else:
                r_min = rolling.min().droplevel(0)
                r_max = rolling.max().droplevel(0)
                result_dict[col_range] = r_max - r_min

    return pd.DataFrame(result_dict, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════
# 2. TREND FEATURES (SLOPE + R²)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_rolling_slope_r2(
    df: pd.DataFrame,
    sensor_columns: list[str],
    window: int,
    compute_slope: bool = True,
    compute_r2: bool = True,
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Compute rolling linear trend slope and R² using vectorized OLS.

    For a window of size W with values y_1..y_W and positions x=0..W-1:
        β₁ = (W·Σ(xy) - Σx·Σy) / (W·Σ(x²) - (Σx)²)
        R² = [W·Σ(xy) - Σx·Σy]² / [(W·Σ(x²) - (Σx)²) · (W·Σ(y²) - (Σy)²)]

    Since x = 0,1,...,W-1 is fixed:
        Σx = W(W-1)/2
        Σx² = W(W-1)(2W-1)/6

    When partial_history=True, points with cycle < W use the available history
    causally (w_eff = t + 1) if w_eff >= 2, or 0.0 if w_eff < 2.
    """
    result = pd.DataFrame(index=df.index)

    if not compute_slope and not compute_r2:
        return result

    W = window
    sum_x = W * (W - 1) / 2.0
    sum_x2 = W * (W - 1) * (2 * W - 1) / 6.0
    denom_x = W * sum_x2 - sum_x * sum_x  # always > 0 for W >= 2

    if denom_x == 0:
        for sensor in sensor_columns:
            if compute_slope:
                result[f"{sensor}_slope_{W}"] = 0.0 if partial_history else np.nan
            if compute_r2:
                result[f"{sensor}_r2_{W}"] = 0.0 if partial_history else np.nan
        return result

    for sensor in sensor_columns:
        series = df[sensor].astype(np.float64)
        series_sq = series ** 2

        sum_xy_vals = np.full(len(df), np.nan)
        slope_partial = np.full(len(df), np.nan) if partial_history else None
        r2_partial = np.full(len(df), np.nan) if partial_history else None

        for uid, group_idx in df.groupby("unit_id").groups.items():
            g = series.loc[group_idx].values
            g_sq = series_sq.loc[group_idx].values
            n = len(g)

            positions = np.arange(n, dtype=np.float64)
            jy = positions * g
            cum_jy = np.cumsum(jy)
            cum_y = np.cumsum(g)
            cum_y2 = np.cumsum(g_sq)

            if partial_history:
                limit = min(W - 1, n)
                for t in range(limit):
                    w_eff = t + 1
                    if w_eff < 2:
                        slope_partial[group_idx[t]] = 0.0
                        r2_partial[group_idx[t]] = 0.0
                        continue
                    s_jy_p = cum_jy[t]
                    s_y_p = cum_y[t]
                    s_y2_p = cum_y2[t]
                    sx_p = w_eff * (w_eff - 1) / 2.0
                    sx2_p = w_eff * (w_eff - 1) * (2 * w_eff - 1) / 6.0
                    den_x_p = w_eff * sx2_p - sx_p * sx_p
                    num_p = w_eff * s_jy_p - sx_p * s_y_p
                    den_y_p = w_eff * s_y2_p - s_y_p * s_y_p
                    den_tot_p = den_x_p * den_y_p

                    sl_p = num_p / den_x_p if abs(den_x_p) > 1e-15 else 0.0
                    r2_p = np.clip((num_p ** 2) / den_tot_p, 0.0, 1.0) if abs(den_tot_p) > 1e-15 else 0.0
                    slope_partial[group_idx[t]] = sl_p
                    r2_partial[group_idx[t]] = r2_p

            if n >= W:
                for t in range(W - 1, n):
                    start = t - W + 1
                    s_jy = cum_jy[t] - (cum_jy[start - 1] if start > 0 else 0)
                    s_y = cum_y[t] - (cum_y[start - 1] if start > 0 else 0)
                    sum_xy_vals[group_idx[t]] = s_jy - start * s_y

        sum_xy = pd.Series(sum_xy_vals, index=df.index)

        grouped = series.groupby(df["unit_id"])
        sum_y = grouped.rolling(window=W, min_periods=W).sum().droplevel(0)
        sum_y2 = series_sq.groupby(df["unit_id"]).rolling(window=W, min_periods=W).sum().droplevel(0)

        if compute_slope:
            numerator = W * sum_xy - sum_x * sum_y
            slope = numerator / denom_x
            if partial_history:
                sp_s = pd.Series(slope_partial, index=df.index)
                slope = slope.combine_first(sp_s).fillna(0.0)
            result[f"{sensor}_slope_{W}"] = slope

        if compute_r2:
            numerator = W * sum_xy - sum_x * sum_y
            denom_y = W * sum_y2 - sum_y * sum_y
            denom_total = denom_x * denom_y
            r2 = np.where(
                np.abs(denom_total) > 1e-15,
                (numerator ** 2) / denom_total,
                np.nan,
            )
            r2 = np.clip(r2, 0.0, 1.0)
            r2_series = pd.Series(r2, index=df.index)
            if partial_history:
                r2p_s = pd.Series(r2_partial, index=df.index)
                r2_series = r2_series.combine_first(r2p_s).fillna(0.0)
            result[f"{sensor}_r2_{W}"] = r2_series

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. DELTA FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_deltas(
    df: pd.DataFrame,
    sensor_columns: list[str],
    deltas: list[int],
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Compute temporal difference features: delta_k = x_t - x_(t-k).

    Causal: only uses past values. Returns NaN if insufficient history.
    When partial_history=True, points with cycle <= k use difference from initial cycle (cycle 0).
    Delta features are per-engine (no cross-engine contamination).
    """
    result = pd.DataFrame(index=df.index)

    for sensor in sensor_columns:
        first_val = df.groupby("unit_id")[sensor].transform("first") if partial_history else None
        for k in deltas:
            col_name = f"{sensor}_delta_{k}"
            shifted = df.groupby("unit_id")[sensor].shift(k)
            diff = df[sensor] - shifted
            if partial_history:
                diff = diff.fillna(df[sensor] - first_val)
            result[col_name] = diff

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. RELATIVE CHANGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_relative_changes(
    df: pd.DataFrame,
    sensor_columns: list[str],
    deltas: list[int],
    floor: float = RELATIVE_CHANGE_FLOOR,
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Compute relative change: (x_t - x_(t-k)) / |x_(t-k)|.

    Zero/near-zero denominators are clamped to `floor` to avoid infinity.
    When partial_history=True, points with cycle <= k use relative change from initial cycle.
    """
    result = pd.DataFrame(index=df.index)

    for sensor in sensor_columns:
        if partial_history:
            first_val = df.groupby("unit_id")[sensor].transform("first")
            first_denom = first_val.abs().clip(lower=floor)
        for k in deltas:
            col_name = f"{sensor}_relchange_{k}"
            shifted = df.groupby("unit_id")[sensor].shift(k)
            denom = shifted.abs().clip(lower=floor)
            rc = (df[sensor] - shifted) / denom
            if partial_history:
                rc = rc.fillna((df[sensor] - first_val) / first_denom)
            result[col_name] = rc

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 5. SINGLE-WINDOW FEATURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_window_features(
    df: pd.DataFrame,
    window_size: int,
    sensor_columns: Optional[list[str]] = None,
    statistics: Optional[list[str]] = None,
    compute_deltas: Optional[list[int]] = None,
    compute_relative_changes: Optional[list[int]] = None,
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Generate temporal features for a single window size.

    The DataFrame must be sorted by (unit_id, cycle) and contain
    'unit_id', 'cycle', and sensor columns (and optional 'rul').
    """
    if sensor_columns is None:
        sensor_columns = [c for c in SENSOR_COLUMNS if c in df.columns]
    if statistics is None:
        statistics = FEATURE_CONFIG["statistics"]
    if compute_deltas is None:
        compute_deltas = FEATURE_CONFIG["deltas"]
    if compute_relative_changes is None:
        compute_relative_changes = FEATURE_CONFIG["relative_changes"]

    # Ensure sorted
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Separate stat types
    basic_stats = [s for s in statistics if s in {"mean", "std", "min", "max", "range"}]
    need_slope = "slope" in statistics
    need_r2 = "r2" in statistics

    parts = []

    # Rolling statistics
    if basic_stats:
        rolling_stats = _compute_rolling_stats(
            df, sensor_columns, window_size, basic_stats, partial_history=partial_history
        )
        parts.append(rolling_stats)

    # Trend features
    if need_slope or need_r2:
        trend = _compute_rolling_slope_r2(
            df, sensor_columns, window_size,
            compute_slope=need_slope, compute_r2=need_r2,
            partial_history=partial_history,
        )
        parts.append(trend)

    # Delta features
    if compute_deltas:
        delta_feats = _compute_deltas(
            df, sensor_columns, compute_deltas, partial_history=partial_history
        )
        parts.append(delta_feats)

    # Relative change features
    if compute_relative_changes:
        relchange_feats = _compute_relative_changes(
            df, sensor_columns, compute_relative_changes, partial_history=partial_history
        )
        parts.append(relchange_feats)

    # Combine: metadata + target + features
    meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns:
        meta_cols.append(TARGET_COLUMN)
    meta_target = df[meta_cols].copy()
    features = pd.concat([meta_target] + parts, axis=1)

    if not partial_history:
        cycle_within_engine = df.groupby("unit_id").cumcount() + 1
        valid_mask = cycle_within_engine >= window_size
        features = features.loc[valid_mask].reset_index(drop=True)
    else:
        features = features.fillna(0.0).reset_index(drop=True)

    return features


# ═══════════════════════════════════════════════════════════════════════════
# 6. MULTI-SCALE FEATURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_multiscale_features(
    df: pd.DataFrame,
    windows: Optional[list[int]] = None,
    sensor_columns: Optional[list[str]] = None,
    statistics: Optional[list[str]] = None,
    deltas: Optional[list[int]] = None,
    relative_changes: Optional[list[int]] = None,
    partial_history: bool = False,
) -> pd.DataFrame:
    """
    Generate multi-scale temporal features across multiple window sizes.

    For each window W, generates rolling statistics. Delta and relative-change
    features are computed once (they are not window-dependent).

    When partial_history=False (default):
        The output retains only rows where ALL windows are complete (i.e., cycle
        position >= max(windows) within each engine).
    When partial_history=True:
        All rows are retained and shorter windows are computed causally from
        available history.
    """
    if windows is None:
        windows = FEATURE_CONFIG["windows"]
    if sensor_columns is None:
        sensor_columns = [c for c in SENSOR_COLUMNS if c in df.columns]
    if statistics is None:
        statistics = FEATURE_CONFIG["statistics"]
    if deltas is None:
        deltas = FEATURE_CONFIG["deltas"]
    if relative_changes is None:
        relative_changes = FEATURE_CONFIG["relative_changes"]

    # Ensure sorted
    df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    basic_stats = [s for s in statistics if s in {"mean", "std", "min", "max", "range"}]
    need_slope = "slope" in statistics
    need_r2 = "r2" in statistics

    # Start with metadata + optional target
    meta_cols = [c for c in METADATA_COLUMNS if c in df.columns]
    if TARGET_COLUMN in df.columns:
        meta_cols.append(TARGET_COLUMN)
    result = df[meta_cols].copy()

    # Generate rolling stats + trend for each window
    for W in sorted(windows):
        if basic_stats:
            rolling = _compute_rolling_stats(
                df, sensor_columns, W, basic_stats, partial_history=partial_history
            )
            result = pd.concat([result, rolling], axis=1)

        if need_slope or need_r2:
            trend = _compute_rolling_slope_r2(
                df, sensor_columns, W,
                compute_slope=need_slope, compute_r2=need_r2,
                partial_history=partial_history,
            )
            result = pd.concat([result, trend], axis=1)

    # Delta features (computed once — not window-dependent)
    if deltas:
        delta_feats = _compute_deltas(
            df, sensor_columns, deltas, partial_history=partial_history
        )
        result = pd.concat([result, delta_feats], axis=1)

    # Relative change features (computed once)
    if relative_changes:
        relchange_feats = _compute_relative_changes(
            df, sensor_columns, relative_changes, partial_history=partial_history
        )
        result = pd.concat([result, relchange_feats], axis=1)

    if not partial_history:
        # Drop rows where the largest window is incomplete
        max_window = max(windows)
        cycle_within_engine = df.groupby("unit_id").cumcount() + 1
        valid_mask = cycle_within_engine >= max_window
        result = result.loc[valid_mask].reset_index(drop=True)
    else:
        result = result.fillna(0.0).reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7. FEATURE METADATA
# ═══════════════════════════════════════════════════════════════════════════

def generate_feature_metadata(feature_columns: list[str]) -> pd.DataFrame:
    """
    Parse feature column names to produce a metadata table.

    Expected naming: {sensor}_{statistic}_{window} or {sensor}_{type}_{lag}

    Returns
    -------
    pd.DataFrame
        Columns: feature_name, sensor, statistic, window_or_lag, description
    """
    rows = []
    stat_descriptions = {
        "mean": "Rolling mean over previous {w} cycles",
        "std": "Rolling standard deviation over previous {w} cycles",
        "min": "Rolling minimum over previous {w} cycles",
        "max": "Rolling maximum over previous {w} cycles",
        "range": "Rolling range (max-min) over previous {w} cycles",
        "slope": "Linear degradation slope over previous {w} cycles",
        "r2": "R-squared of linear fit over previous {w} cycles",
        "delta": "Temporal difference: x_t - x_(t-{w})",
        "relchange": "Relative change: (x_t - x_(t-{w})) / |x_(t-{w})|",
    }

    for col in feature_columns:
        parts = col.rsplit("_", 2)
        if len(parts) == 3:
            # e.g., sensor_2_mean_30 → need to handle sensor names with underscores
            # Try rsplit with maxsplit=2 on the full name
            pass

        # More robust parsing: find the statistic type
        parsed = _parse_feature_name(col)
        if parsed:
            sensor, stat, window_lag = parsed
            desc_template = stat_descriptions.get(stat, f"{stat} feature")
            desc = desc_template.format(w=window_lag)
            rows.append({
                "feature_name": col,
                "sensor": sensor,
                "statistic": stat,
                "window_or_lag": int(window_lag),
                "description": desc,
            })
        else:
            rows.append({
                "feature_name": col,
                "sensor": "unknown",
                "statistic": "unknown",
                "window_or_lag": 0,
                "description": col,
            })

    return pd.DataFrame(rows)


def _parse_feature_name(name: str) -> tuple | None:
    """
    Parse a feature name like 'sensor_2_mean_30' into (sensor, stat, window).

    Handles sensor names with underscores (e.g., 'sensor_21').
    """
    known_stats = [
        "mean", "std", "min", "max", "range",
        "slope", "r2", "delta", "relchange",
    ]

    for stat in known_stats:
        marker = f"_{stat}_"
        idx = name.rfind(marker)
        if idx != -1:
            sensor = name[:idx]
            remainder = name[idx + len(marker):]
            try:
                window = int(remainder)
                return (sensor, stat, window)
            except ValueError:
                continue

    return None


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return only model feature columns (excluding metadata and target)."""
    exclude = set(METADATA_COLUMNS + [TARGET_COLUMN, "failure_cycle"])
    return [c for c in df.columns if c not in exclude]
