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

    Returns
    -------
    pd.DataFrame
        New columns named {sensor}_{stat}_{window}.
    """
    result_dict = {}

    # Map of stat name -> pandas rolling method
    stat_basic = {"mean", "std", "min", "max"}
    requested_basic = [s for s in statistics if s in stat_basic]
    needs_range = "range" in statistics

    for sensor in sensor_columns:
        series = df[sensor]

        # Group-aware rolling: we apply rolling per engine group
        rolling = series.groupby(df["unit_id"]).rolling(
            window=window, min_periods=window
        )

        for stat in requested_basic:
            col_name = f"{sensor}_{stat}_{window}"
            if stat == "mean":
                result_dict[col_name] = rolling.mean().droplevel(0)
            elif stat == "std":
                # ddof=0 for population std within window
                result_dict[col_name] = rolling.std(ddof=0).droplevel(0)
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
                # Need to compute min/max just for range
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
) -> pd.DataFrame:
    """
    Compute rolling linear trend slope and R² using vectorized OLS.

    For a window of size W with values y_1..y_W and positions x=0..W-1:
        β₁ = (W·Σ(xy) - Σx·Σy) / (W·Σ(x²) - (Σx)²)
        R² = [W·Σ(xy) - Σx·Σy]² / [(W·Σ(x²) - (Σx)²) · (W·Σ(y²) - (Σy)²)]

    Since x = 0,1,...,W-1 is fixed:
        Σx = W(W-1)/2
        Σx² = W(W-1)(2W-1)/6

    We only need rolling Σy, Σy², Σ(xy) for each sensor.
    """
    result = pd.DataFrame(index=df.index)

    if not compute_slope and not compute_r2:
        return result

    W = window
    # Precompute fixed x-sums
    sum_x = W * (W - 1) / 2.0
    sum_x2 = W * (W - 1) * (2 * W - 1) / 6.0
    denom_x = W * sum_x2 - sum_x * sum_x  # always > 0 for W >= 2

    if denom_x == 0:
        # Window of 1 — no slope/r2 possible
        for sensor in sensor_columns:
            if compute_slope:
                result[f"{sensor}_slope_{W}"] = np.nan
            if compute_r2:
                result[f"{sensor}_r2_{W}"] = np.nan
        return result

    # For Σ(xy), we need x_i * y_i where x_i = i (0-indexed within window)
    # Using the identity: Σ(i * y_{t-W+1+i}) for i=0..W-1
    # = Σ(i * y_{t-W+1+i})
    # We compute this via: rolling_sum(x * y) where x is a position weight

    for sensor in sensor_columns:
        series = df[sensor].astype(np.float64)

        # Group-aware rolling sums
        grouped = series.groupby(df["unit_id"])

        # Σy via rolling sum
        sum_y = grouped.rolling(window=W, min_periods=W).sum().droplevel(0)

        # Σ(y²) via rolling sum of squared values
        series_sq = (series ** 2)
        sum_y2 = series_sq.groupby(df["unit_id"]).rolling(
            window=W, min_periods=W
        ).sum().droplevel(0)

        # Σ(xy): x_i = 0,1,...,W-1 within each window
        # For position-weighted rolling sum, we use a trick:
        # Σ(i * y_{t-W+1+i}) = (W-1)*y_t + (W-2)*y_{t-1} + ... + 0*y_{t-W+1}
        # = Σ_{k=0}^{W-1} k * y_{t-W+1+k}
        # This equals: W * rolling_mean_of_cumweighted - correction
        # Simpler: compute directly using rolling apply
        # For efficiency, use the identity:
        # sum_xy = sum(i * y_i for i in 0..W-1) where y_i are the W most recent
        # = (W-1)*y_t + (W-2)*y_{t-1} + ... + 0*y_{t-W+1}
        # We can compute this as a weighted rolling sum using convolution-style

        # Efficient approach: use cumulative sums
        # Σ(i * y) = Σ(y_{t-k} * (W-1-k)) for k=0..W-1
        #          = (W-1)*Σ(y) - Σ(k * y_{t-k})
        # And Σ(k * y_{t-k}) = running_sum that can be computed iteratively

        # Most practical: use rolling apply with weights
        # But for large datasets, let's use the cumsum trick:
        # cumsum_y = cumsum(y), so sum_y_window = cumsum[t] - cumsum[t-W]
        # For sum_xy, we need: sum_{j=t-W+1}^{t} (j - (t-W+1)) * y_j
        # = sum_{j} j*y_j - (t-W+1)*sum_y_window
        # And sum_{j} j*y_j can be tracked with cumsum(j*y_j)

        # Use per-group computation for correctness
        sum_xy_vals = np.full(len(df), np.nan)

        for uid, group_idx in df.groupby("unit_id").groups.items():
            g = series.loc[group_idx].values
            n = len(g)
            if n < W:
                continue
            # Compute cumulative j*y_j where j is the position in the group
            positions = np.arange(n, dtype=np.float64)
            jy = positions * g
            cum_jy = np.cumsum(jy)
            cum_y = np.cumsum(g)

            for t in range(W - 1, n):
                start = t - W + 1
                # sum of position-weighted y: sum_{j=start}^{t} j * y_j
                s_jy = cum_jy[t] - (cum_jy[start - 1] if start > 0 else 0)
                s_y = cum_y[t] - (cum_y[start - 1] if start > 0 else 0)
                # Convert absolute positions to relative (0..W-1)
                # sum_xy_relative = s_jy - start * s_y
                sum_xy_vals[group_idx[t]] = s_jy - start * s_y

        sum_xy = pd.Series(sum_xy_vals, index=df.index)

        if compute_slope:
            numerator = W * sum_xy - sum_x * sum_y
            slope = numerator / denom_x
            result[f"{sensor}_slope_{W}"] = slope

        if compute_r2:
            numerator = W * sum_xy - sum_x * sum_y
            denom_y = W * sum_y2 - sum_y * sum_y
            # R² = numerator² / (denom_x * denom_y)
            denom_total = denom_x * denom_y
            # Handle edge cases: constant sensor in window → denom_y = 0 → R² = NaN
            r2 = np.where(
                np.abs(denom_total) > 1e-15,
                (numerator ** 2) / denom_total,
                np.nan,
            )
            # Clamp to [0, 1] for numerical stability
            r2 = np.clip(r2, 0.0, 1.0)
            result[f"{sensor}_r2_{W}"] = r2

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3. DELTA FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_deltas(
    df: pd.DataFrame,
    sensor_columns: list[str],
    deltas: list[int],
) -> pd.DataFrame:
    """
    Compute temporal difference features: delta_k = x_t - x_(t-k).

    Causal: only uses past values. Returns NaN if insufficient history.
    Delta features are per-engine (no cross-engine contamination).
    """
    result = pd.DataFrame(index=df.index)

    for sensor in sensor_columns:
        for k in deltas:
            col_name = f"{sensor}_delta_{k}"
            shifted = df.groupby("unit_id")[sensor].shift(k)
            result[col_name] = df[sensor] - shifted

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 4. RELATIVE CHANGE FEATURES
# ═══════════════════════════════════════════════════════════════════════════

def _compute_relative_changes(
    df: pd.DataFrame,
    sensor_columns: list[str],
    deltas: list[int],
    floor: float = RELATIVE_CHANGE_FLOOR,
) -> pd.DataFrame:
    """
    Compute relative change: (x_t - x_(t-k)) / |x_(t-k)|.

    Zero/near-zero denominators are clamped to `floor` to avoid infinity.
    Returns NaN where insufficient history exists.
    """
    result = pd.DataFrame(index=df.index)

    for sensor in sensor_columns:
        for k in deltas:
            col_name = f"{sensor}_relchange_{k}"
            shifted = df.groupby("unit_id")[sensor].shift(k)
            denom = shifted.abs().clip(lower=floor)
            result[col_name] = (df[sensor] - shifted) / denom

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
) -> pd.DataFrame:
    """
    Generate temporal features for a single window size.

    The DataFrame must be sorted by (unit_id, cycle) and contain
    'unit_id', 'cycle', 'rul', and sensor columns.

    Only complete windows are retained (rows with cycle < window_size
    within their engine are dropped).

    Parameters
    ----------
    df : pd.DataFrame
        Input with unit_id, cycle, rul, and sensor columns.
    window_size : int
        Number of historical cycles in each window.
    sensor_columns : list[str], optional
        Sensors to compute features for.
    statistics : list[str], optional
        Rolling statistics to compute.
    compute_deltas : list[int], optional
        Delta lags to compute. Pass empty list to skip.
    compute_relative_changes : list[int], optional
        Relative change lags to compute. Pass empty list to skip.

    Returns
    -------
    pd.DataFrame
        Feature DataFrame with metadata (unit_id, cycle), target (rul),
        and all generated feature columns. Only complete-window rows.
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
        rolling_stats = _compute_rolling_stats(df, sensor_columns, window_size, basic_stats)
        parts.append(rolling_stats)

    # Trend features
    if need_slope or need_r2:
        trend = _compute_rolling_slope_r2(
            df, sensor_columns, window_size,
            compute_slope=need_slope, compute_r2=need_r2,
        )
        parts.append(trend)

    # Delta features (not window-dependent, but we compute them here)
    if compute_deltas:
        delta_feats = _compute_deltas(df, sensor_columns, compute_deltas)
        parts.append(delta_feats)

    # Relative change features
    if compute_relative_changes:
        relchange_feats = _compute_relative_changes(
            df, sensor_columns, compute_relative_changes,
        )
        parts.append(relchange_feats)

    # Combine: metadata + target + features
    meta_target = df[METADATA_COLUMNS + [TARGET_COLUMN]].copy()
    features = pd.concat([meta_target] + parts, axis=1)

    # Drop rows with incomplete windows
    # A complete window requires at least `window_size` cycles of history
    # For each engine, the first valid row is at the W-th cycle
    cycle_within_engine = df.groupby("unit_id").cumcount() + 1
    valid_mask = cycle_within_engine >= window_size

    # Also require that delta/relchange NaNs are handled:
    # Delta/relchange rows that lack history get NaN, which is fine —
    # they'll be dropped if the window mask already excludes them.
    # The window_size requirement is the binding constraint.
    features = features.loc[valid_mask].reset_index(drop=True)

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
) -> pd.DataFrame:
    """
    Generate multi-scale temporal features across multiple window sizes.

    For each window W, generates rolling statistics. Delta and relative-change
    features are computed once (they are not window-dependent).

    The output retains only rows where ALL windows are complete (i.e., cycle
    position >= max(windows) within each engine).

    Parameters
    ----------
    df : pd.DataFrame
        Input with unit_id, cycle, rul, and sensor columns.
    windows : list[int], optional
        Window sizes to use.

    Returns
    -------
    pd.DataFrame
        Combined feature DataFrame.
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

    # Start with metadata + target
    result = df[METADATA_COLUMNS + [TARGET_COLUMN]].copy()

    # Generate rolling stats + trend for each window
    for W in sorted(windows):
        if basic_stats:
            rolling = _compute_rolling_stats(df, sensor_columns, W, basic_stats)
            result = pd.concat([result, rolling], axis=1)

        if need_slope or need_r2:
            trend = _compute_rolling_slope_r2(
                df, sensor_columns, W,
                compute_slope=need_slope, compute_r2=need_r2,
            )
            result = pd.concat([result, trend], axis=1)

    # Delta features (computed once — not window-dependent)
    if deltas:
        delta_feats = _compute_deltas(df, sensor_columns, deltas)
        result = pd.concat([result, delta_feats], axis=1)

    # Relative change features (computed once)
    if relative_changes:
        relchange_feats = _compute_relative_changes(
            df, sensor_columns, relative_changes,
        )
        result = pd.concat([result, relchange_feats], axis=1)

    # Drop rows where the largest window is incomplete
    max_window = max(windows)
    cycle_within_engine = df.groupby("unit_id").cumcount() + 1
    valid_mask = cycle_within_engine >= max_window

    result = result.loc[valid_mask].reset_index(drop=True)

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
