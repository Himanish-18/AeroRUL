"""
AeroRUL — Stage 4 Artifact Generation Script

Extracts native LightGBM feature importance (gain and split) from the frozen models,
aggregates feature importance by sensor, statistic, and window families, joins
feature-target correlations from Phase 3, and generates compact, authentic sensor
trajectories from original C-MAPSS training data.

Outputs are written to: reports/dashboard_stage4/
Strict constraints:
- Does NOT retrain or alter any model.
- Does NOT use any official test labels or test predictions.
- Does NOT fabricate any importance scores or sensor values.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import COLUMN_NAMES, PROJECT_ROOT, RAW_DATA_DIR, SENSOR_COLUMNS

# Input locations
MODELS_FINAL_DIR = PROJECT_ROOT / "models" / "final"
PHASE3_REPORTS_DIR = PROJECT_ROOT / "reports" / "phase3"
FEATURE_METADATA_PATH = PHASE3_REPORTS_DIR / "feature_metadata.csv"

# Output location
OUTPUT_DIR = PROJECT_ROOT / "reports" / "dashboard_stage4"

DATASETS = ["FD001", "FD002", "FD003", "FD004"]

# Sensor descriptions and physical units from NASA C-MAPSS specification
SENSOR_INFO = {
    "sensor_1": {"name": "Total temp at fan inlet", "unit": "°R", "type": "Temperature"},
    "sensor_2": {"name": "Total temp at LPC outlet", "unit": "°R", "type": "Temperature"},
    "sensor_3": {"name": "Total temp at HPC outlet", "unit": "°R", "type": "Temperature"},
    "sensor_4": {"name": "Total temp at LPT outlet", "unit": "°R", "type": "Temperature"},
    "sensor_5": {"name": "Pressure at fan inlet", "unit": "psia", "type": "Pressure"},
    "sensor_6": {"name": "Total pressure in bypass duct", "unit": "psia", "type": "Pressure"},
    "sensor_7": {"name": "Total pressure at HPC outlet", "unit": "psia", "type": "Pressure"},
    "sensor_8": {"name": "Physical fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_9": {"name": "Physical core speed", "unit": "rpm", "type": "Speed"},
    "sensor_10": {"name": "Engine pressure ratio (P50/P2)", "unit": "—", "type": "Ratio"},
    "sensor_11": {"name": "Static pressure at HPC outlet", "unit": "psia", "type": "Pressure"},
    "sensor_12": {"name": "Ratio of fuel flow to Ps30", "unit": "pps/psia", "type": "Ratio"},
    "sensor_13": {"name": "Corrected fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_14": {"name": "Corrected core speed", "unit": "rpm", "type": "Speed"},
    "sensor_15": {"name": "Bypass ratio", "unit": "—", "type": "Ratio"},
    "sensor_16": {"name": "Burner fuel-air ratio", "unit": "—", "type": "Ratio"},
    "sensor_17": {"name": "Bleed enthalpy", "unit": "—", "type": "Enthalpy"},
    "sensor_18": {"name": "Demanded fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_19": {"name": "Demanded corrected fan speed", "unit": "rpm", "type": "Speed"},
    "sensor_20": {"name": "HPT coolant bleed", "unit": "lbm/s", "type": "Flow"},
    "sensor_21": {"name": "LPT coolant bleed", "unit": "lbm/s", "type": "Flow"},
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file for cryptographic provenance."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def fallback_feature_parser(feature_name: str) -> Dict[str, Any]:
    """
    Defensive parser to extract sensor, statistic, and window/lag from
    feature names if metadata is missing.
    Format: sensor_{id}_{statistic}_{window_or_lag}
    """
    match = re.match(r"^(sensor_\d+)_([a-zA-Z0-9]+)_(\d+)$", feature_name)
    if match:
        sensor, stat, win = match.groups()
        return {
            "sensor": sensor,
            "statistic": stat,
            "window_or_lag": win,
            "description": f"{stat} over window/lag {win} for {sensor}",
        }
    return {
        "sensor": "unknown",
        "statistic": "unknown",
        "window_or_lag": "unknown",
        "description": feature_name,
    }


def load_feature_metadata() -> pd.DataFrame:
    """Load Phase 3 feature metadata table."""
    if not FEATURE_METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing required feature metadata: {FEATURE_METADATA_PATH}")
    df = pd.read_csv(FEATURE_METADATA_PATH)
    return df.set_index("feature_name")


def load_correlations(dataset: str) -> pd.DataFrame:
    """Load Phase 3 feature-target correlation records for a dataset."""
    corr_path = PHASE3_REPORTS_DIR / f"feature_target_correlations_{dataset}.csv"
    if not corr_path.exists():
        print(f"  Warning: No correlation file found at {corr_path}")
        return pd.DataFrame(columns=["feature", "correlation_pearson", "correlation_spearman"])
    df = pd.read_csv(corr_path)
    return df.set_index("feature")


def format_window_family(window_or_lag: Any, statistic: str) -> str:
    """Format window or lag into standard analytical groups (e.g. W10, Lag1)."""
    win_str = str(window_or_lag).strip()
    if statistic in ("delta", "relchange"):
        return f"Lag {win_str}"
    return f"W{win_str}"


# ═══════════════════════════════════════════════════════════════════════════
# PART A: FEATURE IMPORTANCE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def generate_feature_importance_artifacts() -> Dict[str, Any]:
    """Extract model importance, join metadata and correlations, and aggregate."""
    print("\n--- Part A: Generating Feature Importance Artifacts ---")
    metadata_df = load_feature_metadata()
    manifest_datasets: Dict[str, Any] = {}
    family_rows: List[Dict[str, Any]] = []

    for dataset in DATASETS:
        print(f"\nProcessing dataset: {dataset}")
        model_dir = MODELS_FINAL_DIR / dataset
        model_path = model_dir / "model.joblib"
        feat_path = model_dir / "feature_names.json"
        config_path = model_dir / "config_metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Frozen model missing: {model_path}")
        if not feat_path.exists():
            raise FileNotFoundError(f"Feature names missing: {feat_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config metadata missing: {config_path}")

        # 1. Load model and verify configuration
        model = joblib.load(model_path)
        with open(feat_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)
        with open(config_path, "r", encoding="utf-8") as f:
            config_meta = json.load(f)

        n_features = len(feature_names)
        booster = model.booster_
        num_booster_features = booster.num_feature()

        if num_booster_features != n_features:
            raise ValueError(
                f"Mismatch in feature counts for {dataset}: "
                f"Booster has {num_booster_features}, feature_names.json has {n_features}"
            )
        if config_meta.get("n_features") != n_features:
            raise ValueError(
                f"Mismatch in config metadata n_features for {dataset}: "
                f"config has {config_meta.get('n_features')}, expected {n_features}"
            )

        print(f"  Verified {n_features} features against booster and config.")

        # 2. Extract native gain and split importance
        gain_importance = booster.feature_importance(importance_type="gain").astype(float)
        split_importance = booster.feature_importance(importance_type="split").astype(int)

        total_gain = float(gain_importance.sum())
        total_split = int(split_importance.sum())

        gain_pct = (gain_importance / total_gain * 100.0) if total_gain > 0 else np.zeros_like(gain_importance)
        split_pct = (split_importance / total_split * 100.0) if total_split > 0 else np.zeros_like(split_importance)

        # Compute ranks (1 = highest importance)
        # Using method='min' or argsort descending
        gain_order = np.argsort(-gain_importance)
        gain_ranks = np.empty_like(gain_order)
        gain_ranks[gain_order] = np.arange(1, n_features + 1)

        split_order = np.argsort(-split_importance)
        split_ranks = np.empty_like(split_order)
        split_ranks[split_order] = np.arange(1, n_features + 1)

        # 3. Load correlations
        corr_df = load_correlations(dataset)

        # 4. Assemble feature rows
        feature_rows: List[Dict[str, Any]] = []
        for idx, feat_name in enumerate(feature_names):
            if feat_name in metadata_df.index:
                meta_row = metadata_df.loc[feat_name]
                sensor = str(meta_row["sensor"])
                stat = str(meta_row["statistic"])
                win_lag = str(meta_row["window_or_lag"])
                desc = str(meta_row["description"])
            else:
                fallback = fallback_feature_parser(feat_name)
                sensor = fallback["sensor"]
                stat = fallback["statistic"]
                win_lag = fallback["window_or_lag"]
                desc = fallback["description"]

            # Correlations (strictly None if not present)
            corr_pearson = None
            corr_spearman = None
            abs_pearson = None
            abs_spearman = None
            if feat_name in corr_df.index:
                row_corr = corr_df.loc[feat_name]
                val_p = row_corr.get("correlation_pearson")
                val_s = row_corr.get("correlation_spearman")
                if pd.notna(val_p):
                    corr_pearson = round(float(val_p), 6)
                    abs_pearson = round(abs(float(val_p)), 6)
                if pd.notna(val_s):
                    corr_spearman = round(float(val_s), 6)
                    abs_spearman = round(abs(float(val_s)), 6)

            feature_rows.append({
                "dataset": dataset,
                "feature_name": feat_name,
                "sensor": sensor,
                "statistic": stat,
                "window_or_lag": win_lag,
                "description": desc,
                "gain_importance": round(gain_importance[idx], 4),
                "split_importance": int(split_importance[idx]),
                "gain_importance_pct": round(gain_pct[idx], 4),
                "split_importance_pct": round(split_pct[idx], 4),
                "gain_rank": int(gain_ranks[idx]),
                "split_rank": int(split_ranks[idx]),
                "correlation_pearson": "" if corr_pearson is None else corr_pearson,
                "correlation_spearman": "" if corr_spearman is None else corr_spearman,
                "abs_pearson": "" if abs_pearson is None else abs_pearson,
                "abs_spearman": "" if abs_spearman is None else abs_spearman,
            })

        df_feat = pd.DataFrame(feature_rows)
        # Sort by gain rank
        df_feat = df_feat.sort_values("gain_rank", ascending=True)

        feat_out_path = OUTPUT_DIR / f"feature_importance_{dataset}.csv"
        df_feat.to_csv(feat_out_path, index=False)
        print(f"  Saved: {feat_out_path} ({len(df_feat)} rows)")

        # 5. Aggregate by sensor
        sensor_agg = df_feat.groupby("sensor").agg(
            gain_importance=("gain_importance", "sum"),
            split_importance=("split_importance", "sum"),
            gain_importance_pct=("gain_importance_pct", "sum"),
            split_importance_pct=("split_importance_pct", "sum"),
            feature_count=("feature_name", "count"),
        ).reset_index()

        sensor_agg["dataset"] = dataset
        # Find top feature for each sensor
        top_feats = (
            df_feat.sort_values("gain_importance", ascending=False)
            .groupby("sensor")
            .first()["feature_name"]
            .to_dict()
        )
        sensor_agg["top_feature"] = sensor_agg["sensor"].map(top_feats)

        # Sensor ranks
        sensor_agg = sensor_agg.sort_values("gain_importance", ascending=False).reset_index(drop=True)
        sensor_agg["gain_rank"] = np.arange(1, len(sensor_agg) + 1)

        sensor_split_order = np.argsort(-sensor_agg["split_importance"].values)
        sensor_split_ranks = np.empty_like(sensor_split_order)
        sensor_split_ranks[sensor_split_order] = np.arange(1, len(sensor_agg) + 1)
        sensor_agg["split_rank"] = sensor_split_ranks

        # Reorder columns
        sensor_cols = [
            "dataset", "sensor", "gain_importance", "split_importance",
            "gain_importance_pct", "split_importance_pct", "gain_rank",
            "split_rank", "feature_count", "top_feature"
        ]
        sensor_agg = sensor_agg[sensor_cols]
        sensor_out_path = OUTPUT_DIR / f"sensor_importance_{dataset}.csv"
        sensor_agg.round({
            "gain_importance": 4,
            "gain_importance_pct": 4,
            "split_importance_pct": 4,
        }).to_csv(sensor_out_path, index=False)
        print(f"  Saved: {sensor_out_path} ({len(sensor_agg)} sensors)")

        # 6. Accumulate Family Breakdown Rows
        # Group 1: Sensor family
        for _, s_row in sensor_agg.iterrows():
            family_rows.append({
                "dataset": dataset,
                "family_type": "sensor",
                "family_key": s_row["sensor"],
                "gain_importance": round(s_row["gain_importance"], 4),
                "split_importance": int(s_row["split_importance"]),
                "gain_importance_pct": round(s_row["gain_importance_pct"], 4),
                "split_importance_pct": round(s_row["split_importance_pct"], 4),
                "feature_count": int(s_row["feature_count"]),
                "rank_in_family": int(s_row["gain_rank"]),
            })

        # Group 2: Statistic family
        stat_agg = df_feat.groupby("statistic").agg(
            gain_importance=("gain_importance", "sum"),
            split_importance=("split_importance", "sum"),
            gain_importance_pct=("gain_importance_pct", "sum"),
            split_importance_pct=("split_importance_pct", "sum"),
            feature_count=("feature_name", "count"),
        ).reset_index().sort_values("gain_importance", ascending=False).reset_index(drop=True)
        stat_agg["rank_in_family"] = np.arange(1, len(stat_agg) + 1)

        for _, st_row in stat_agg.iterrows():
            family_rows.append({
                "dataset": dataset,
                "family_type": "statistic",
                "family_key": st_row["statistic"],
                "gain_importance": round(st_row["gain_importance"], 4),
                "split_importance": int(st_row["split_importance"]),
                "gain_importance_pct": round(st_row["gain_importance_pct"], 4),
                "split_importance_pct": round(st_row["split_importance_pct"], 4),
                "feature_count": int(st_row["feature_count"]),
                "rank_in_family": int(st_row["rank_in_family"]),
            })

        # Group 3: Window family
        df_feat["window_family"] = df_feat.apply(
            lambda r: format_window_family(r["window_or_lag"], r["statistic"]), axis=1
        )
        win_agg = df_feat.groupby("window_family").agg(
            gain_importance=("gain_importance", "sum"),
            split_importance=("split_importance", "sum"),
            gain_importance_pct=("gain_importance_pct", "sum"),
            split_importance_pct=("split_importance_pct", "sum"),
            feature_count=("feature_name", "count"),
        ).reset_index().sort_values("gain_importance", ascending=False).reset_index(drop=True)
        win_agg["rank_in_family"] = np.arange(1, len(win_agg) + 1)

        for _, w_row in win_agg.iterrows():
            family_rows.append({
                "dataset": dataset,
                "family_type": "window",
                "family_key": w_row["window_family"],
                "gain_importance": round(w_row["gain_importance"], 4),
                "split_importance": int(w_row["split_importance"]),
                "gain_importance_pct": round(w_row["gain_importance_pct"], 4),
                "split_importance_pct": round(w_row["split_importance_pct"], 4),
                "feature_count": int(w_row["feature_count"]),
                "rank_in_family": int(w_row["rank_in_family"]),
            })

        # Manifest metadata for dataset
        manifest_datasets[dataset] = {
            "dataset": dataset,
            "source_model_path": str(model_path.relative_to(PROJECT_ROOT)),
            "model_file_sha256": compute_file_sha256(model_path),
            "model_family": config_meta.get("model_family", "LightGBM"),
            "feature_configuration": config_meta.get("feature_configuration", "ALL_WINDOWS"),
            "n_features": n_features,
            "total_gain_importance": round(total_gain, 4),
            "total_split_importance": total_split,
            "importance_methods": ["gain", "split"],
            "normalization_method": "percentage_of_total",
            "top_feature_by_gain": df_feat.iloc[0]["feature_name"],
            "top_feature_gain_pct": df_feat.iloc[0]["gain_importance_pct"],
            "top_sensor_by_gain": sensor_agg.iloc[0]["sensor"],
            "top_sensor_gain_pct": sensor_agg.iloc[0]["gain_importance_pct"],
        }

    # Save feature_family_importance.csv
    df_families = pd.DataFrame(family_rows)
    family_out_path = OUTPUT_DIR / "feature_family_importance.csv"
    df_families.to_csv(family_out_path, index=False)
    print(f"\n  Saved combined: {family_out_path} ({len(df_families)} rows)")

    # Save feature_importance_manifest.json
    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": manifest_datasets,
        "interpretability_notice": (
            "Gain measures the reduction in split impurity attributable to a feature across trees. "
            "Split importance counts feature usage in decision nodes. Neither metric represents SHAP "
            "values or individual prediction causality."
        ),
    }
    manifest_path = OUTPUT_DIR / "feature_importance_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest: {manifest_path}")

    return manifest


# ═══════════════════════════════════════════════════════════════════════════
# PART B: SENSOR DEGRADATION TRAJECTORIES PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def select_representative_engines(df_train: pd.DataFrame) -> List[int]:
    """
    Select 10 diverse engines per dataset:
    - Standard initial fleet engines 1, 2, 3, 4, 5
    - Shortest-lived engine (minimum cycles)
    - 25th percentile lifetime engine
    - Median lifetime engine
    - 75th percentile lifetime engine
    - Longest-lived engine (maximum cycles)
    """
    lens = df_train.groupby("unit_id")["cycle"].max()
    min_unit = int(lens.idxmin())
    max_unit = int(lens.idxmax())

    p25_val = lens.quantile(0.25)
    p50_val = lens.median()
    p75_val = lens.quantile(0.75)

    unit_p25 = int((lens - p25_val).abs().idxmin())
    unit_p50 = int((lens - p50_val).abs().idxmin())
    unit_p75 = int((lens - p75_val).abs().idxmin())

    selected = sorted(set([1, 2, 3, 4, 5, min_unit, unit_p25, unit_p50, unit_p75, max_unit]))
    return selected


def generate_sensor_trajectories_artifacts() -> Dict[str, Any]:
    """
    Generate compact sensor trajectory artifacts from authentic training data.
    Downsampling strategy:
    - Deterministic stride 2: cycle % 2 == 1 OR cycle == max_cycle
    - Retains 100% of cycle 1 and exact terminal failure cycles
    - Uses only authentic training telemetry; no smoothed/invented points
    """
    print("\n--- Part B: Generating Sensor Degradation Trajectory Artifacts ---")
    manifest_subsets: Dict[str, Any] = {}

    for dataset in DATASETS:
        print(f"\nProcessing trajectories for dataset: {dataset}")
        train_file = RAW_DATA_DIR / f"train_{dataset}.txt"
        if not train_file.exists():
            raise FileNotFoundError(f"Missing training file: {train_file}")

        df_raw = pd.read_csv(train_file, sep=r"\s+", header=None, names=COLUMN_NAMES)
        selected_units = select_representative_engines(df_raw)
        print(f"  Selected {len(selected_units)} representative engines: {selected_units}")

        # Filter to selected engines
        df_sub = df_raw[df_raw["unit_id"].isin(selected_units)].copy()

        # Deterministic downsampling per engine
        unit_meta: Dict[str, Any] = {}
        kept_rows = []

        for unit in selected_units:
            unit_df = df_sub[df_sub["unit_id"] == unit].sort_values("cycle")
            max_c = int(unit_df["cycle"].max())
            min_c = int(unit_df["cycle"].min())

            # Stride 2 sampling while guaranteeing boundary points
            sampled = unit_df[(unit_df["cycle"] % 2 == 1) | (unit_df["cycle"] == max_c)]
            kept_rows.append(sampled)

            unit_meta[str(unit)] = {
                "unit_id": unit,
                "lifetime_cycles": max_c,
                "sampled_points": len(sampled),
                "is_fleet_standard": unit in [1, 2, 3, 4, 5],
            }

        df_sampled = pd.concat(kept_rows, ignore_index=True)

        # Convert to structured trajectory records: dataset, unit_id, cycle, sensor, sensor_value
        trajectory_records: List[Dict[str, Any]] = []
        for _, row in df_sampled.iterrows():
            u_id = int(row["unit_id"])
            cyc = int(row["cycle"])
            for sensor_col in SENSOR_COLUMNS:
                val = float(row[sensor_col])
                trajectory_records.append({
                    "dataset": dataset,
                    "unit_id": u_id,
                    "cycle": cyc,
                    "sensor": sensor_col,
                    "sensor_value": round(val, 4),
                })

        df_traj = pd.DataFrame(trajectory_records)
        traj_out_path = OUTPUT_DIR / f"sensor_trajectories_{dataset}.csv"
        df_traj.to_csv(traj_out_path, index=False)
        print(f"  Saved: {traj_out_path} ({len(df_traj)} points, {len(df_sampled)} engine-cycles)")

        manifest_subsets[dataset] = {
            "dataset": dataset,
            "default_unit_id": 1 if 1 in selected_units else selected_units[0],
            "selected_units": selected_units,
            "unit_metadata": unit_meta,
            "available_sensors": SENSOR_COLUMNS,
            "total_sampled_cycles": len(df_sampled),
            "total_measurement_records": len(df_traj),
        }

    # Save sensor_trajectories_manifest.json
    traj_manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sampling_strategy": (
            "Deterministic stride-2 subsampling (cycle % 2 == 1 or cycle == terminal_cycle) "
            "across 10 representative engines spanning lifetime percentiles (min, 25%, median, 75%, max) "
            "and standard fleet engines 1-5. Authentic C-MAPSS training measurements only."
        ),
        "sensor_specifications": SENSOR_INFO,
        "subsets": manifest_subsets,
    }

    traj_manifest_path = OUTPUT_DIR / "sensor_trajectories_manifest.json"
    with open(traj_manifest_path, "w", encoding="utf-8") as f:
        json.dump(traj_manifest, f, indent=2)
    print(f"\n  Saved trajectories manifest: {traj_manifest_path}")

    return traj_manifest


def main():
    print(f"Starting Stage 4 artifact generation...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_feature_importance_artifacts()
    generate_sensor_trajectories_artifacts()
    print("\nStage 4 artifact generation complete!")


if __name__ == "__main__":
    main()
