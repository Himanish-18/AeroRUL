"""
AeroRUL — Phase 3 Analysis Pipeline

Run the complete Phase 3 analysis: multi-scale sliding-window feature
engineering, feature quality validation, feature-target correlation analysis,
and report/figure generation.

Usage:
    python -m src.run_phase3_analysis
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    AVAILABLE_SUBSETS,
    FEATURE_CONFIG,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_STYLE,
    METADATA_COLUMNS,
    OP_SETTING_COLUMNS,
    PHASE3_DATA_DIR,
    PHASE3_FIGURES_DIR,
    PHASE3_REPORTS_DIR,
    SENSOR_COLUMNS,
    SUBSET_METADATA,
    TARGET_COLUMN,
)
from src.data_loader import load_all_subsets
from src.feature_validation import (
    data_retention_report,
    feature_target_correlations,
    validate_engine_separation,
    validate_feature_quality,
    validate_target_integrity,
)
from src.features import (
    generate_feature_metadata,
    generate_multiscale_features,
    generate_window_features,
    get_feature_columns,
)
from src.normalization import (
    apply_regime_normalization,
    fit_regime_normalization,
    identify_operating_regimes,
)
from src.rul import generate_rul_targets
from src.splitting import split_by_engine


def _apply_style():
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_rolling_mean_example(train_df, features_df, subset, output_dir):
    """Show raw sensor trajectory vs rolling mean."""
    _apply_style()
    engines = sorted(train_df["unit_id"].unique())[:2]
    sensor = "sensor_2"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for eid in engines:
        raw = train_df[train_df["unit_id"] == eid].sort_values("cycle")
        feat = features_df[features_df["unit_id"] == eid].sort_values("cycle")

        axes[0].plot(raw["cycle"], raw[sensor], linewidth=0.6, alpha=0.6, label=f"Engine {eid} (raw)")
        if f"{sensor}_mean_30" in feat.columns:
            axes[0].plot(feat["cycle"], feat[f"{sensor}_mean_30"], linewidth=1.5, label=f"Engine {eid} (mean W=30)")

    axes[0].set_title(f"{subset} -- {sensor}: Raw vs Rolling Mean (W=30)", fontweight="bold")
    axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("Value"); axes[0].legend(fontsize=8)

    # Rolling std
    for eid in engines:
        feat = features_df[features_df["unit_id"] == eid].sort_values("cycle")
        if f"{sensor}_std_30" in feat.columns:
            axes[1].plot(feat["cycle"], feat[f"{sensor}_std_30"], linewidth=1.2, label=f"Engine {eid}")

    axes[1].set_title(f"{subset} -- {sensor}: Rolling Std (W=30)", fontweight="bold")
    axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("Std"); axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / f"rolling_mean_std_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_slope(train_df, features_df, subset, output_dir):
    """Show sensor trajectory with fitted trend slope."""
    _apply_style()
    engines = sorted(train_df["unit_id"].unique())[:2]
    sensor = "sensor_2"

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for i, eid in enumerate(engines):
        feat = features_df[features_df["unit_id"] == eid].sort_values("cycle")
        raw = train_df[train_df["unit_id"] == eid].sort_values("cycle")

        axes[i].plot(raw["cycle"], raw[sensor], linewidth=0.6, alpha=0.5, label="Raw")
        if f"{sensor}_slope_30" in feat.columns:
            ax2 = axes[i].twinx()
            ax2.plot(feat["cycle"], feat[f"{sensor}_slope_30"], color="red", linewidth=1.2, label="Slope (W=30)")
            ax2.set_ylabel("Slope", color="red")
            ax2.legend(loc="upper left", fontsize=8)

        axes[i].set_title(f"Engine {eid}", fontweight="bold")
        axes[i].set_xlabel("Cycle"); axes[i].set_ylabel("Sensor Value")
        axes[i].legend(loc="lower left", fontsize=8)

    fig.suptitle(f"{subset} -- {sensor}: Degradation Slope (W=30)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"degradation_slope_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_multiscale_slopes(features_df, subset, output_dir):
    """Show multi-scale slopes for a representative engine."""
    _apply_style()
    engines = sorted(features_df["unit_id"].unique())
    eid = engines[0]
    sensor = "sensor_2"

    feat = features_df[features_df["unit_id"] == eid].sort_values("cycle")

    fig, ax = plt.subplots(figsize=(14, 5))
    for W in FEATURE_CONFIG["windows"]:
        col = f"{sensor}_slope_{W}"
        if col in feat.columns:
            ax.plot(feat["cycle"], feat[col], linewidth=1.0, label=f"W={W}")

    ax.set_title(f"{subset} Engine {eid} -- {sensor}: Multi-Scale Slopes", fontweight="bold")
    ax.set_xlabel("Cycle"); ax.set_ylabel("Slope"); ax.legend()
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_dir / f"multiscale_slopes_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_target_scatter(features_df, subset, output_dir):
    """Plot representative feature-target scatterplots."""
    _apply_style()
    feat_cols = get_feature_columns(features_df)

    # Pick top 4 features by absolute correlation with RUL
    corrs = features_df[feat_cols].corrwith(features_df[TARGET_COLUMN]).abs()
    top_feats = corrs.nlargest(4).index.tolist()

    if len(top_feats) < 2:
        return

    n = min(len(top_feats), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, feat in enumerate(top_feats):
        # Subsample for readability
        sample = features_df.sample(n=min(2000, len(features_df)), random_state=42)
        axes[i].scatter(sample[feat], sample[TARGET_COLUMN], alpha=0.2, s=5, color="steelblue")
        axes[i].set_xlabel(feat, fontsize=8)
        axes[i].set_ylabel("RUL" if i == 0 else "")
        r = features_df[[feat, TARGET_COLUMN]].dropna().corr().iloc[0, 1]
        axes[i].set_title(f"r={r:.3f}", fontweight="bold")

    fig.suptitle(f"{subset} -- Top Features vs RUL", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"feature_target_scatter_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation_heatmap(features_df, subset, output_dir):
    """Correlation heatmap for top correlated features (manageable size)."""
    _apply_style()
    feat_cols = get_feature_columns(features_df)
    corrs = features_df[feat_cols].corrwith(features_df[TARGET_COLUMN]).abs()
    top_20 = corrs.nlargest(20).index.tolist()

    if len(top_20) < 3:
        return

    corr_matrix = features_df[top_20].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap="RdBu_r", center=0,
                xticklabels=True, yticklabels=True, ax=ax)
    ax.set_title(f"{subset} -- Top-20 Feature Correlation", fontweight="bold")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"feature_correlation_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase3_summary(
    all_retention, all_quality, all_target, all_engine_sep,
    all_top_corrs, feature_metadata, output_path,
):
    """Generate the human-readable phase3_summary.md report."""
    lines = []
    lines.append("# AeroRUL -- Phase 3 Summary Report\n")

    # Feature methodology
    lines.append("## 1. Feature Methodology\n")
    lines.append("### Feature Families\n")
    lines.append("| Family | Features | Description |")
    lines.append("|--------|----------|-------------|")
    lines.append("| Rolling Mean | sensor_{s}_mean_{W} | Mean of sensor over previous W cycles |")
    lines.append("| Rolling Std | sensor_{s}_std_{W} | Population std over previous W cycles |")
    lines.append("| Rolling Min | sensor_{s}_min_{W} | Minimum over previous W cycles |")
    lines.append("| Rolling Max | sensor_{s}_max_{W} | Maximum over previous W cycles |")
    lines.append("| Rolling Range | sensor_{s}_range_{W} | max - min over previous W cycles |")
    lines.append("| Degradation Slope | sensor_{s}_slope_{W} | Linear OLS trend slope over W cycles |")
    lines.append("| Trend Strength | sensor_{s}_r2_{W} | R² of linear fit over W cycles |")
    lines.append("| Temporal Delta | sensor_{s}_delta_{k} | x_t - x_(t-k) |")
    lines.append("| Relative Change | sensor_{s}_relchange_{k} | (x_t - x_(t-k)) / |x_(t-k)| |\n")

    # Window configurations
    lines.append("## 2. Window Configurations\n")
    lines.append("| Window | Captures |")
    lines.append("|--------|----------|")
    lines.append("| W=10 | Short-term / recent change |")
    lines.append("| W=20 | Short-to-medium term |")
    lines.append("| W=30 | Medium-term degradation |")
    lines.append("| W=50 | Medium-to-long term |")
    lines.append("| W=75 | Long-term degradation trend |\n")

    # Feature counts
    if feature_metadata is not None:
        n_feats = len(feature_metadata)
        lines.append(f"## 3. Feature Count\n")
        lines.append(f"Total features per observation: **{n_feats}**\n")
        stat_counts = feature_metadata.groupby("statistic").size().to_dict()
        lines.append("| Statistic | Count |")
        lines.append("|-----------|-------|")
        for stat, count in sorted(stat_counts.items()):
            lines.append(f"| {stat} | {count} |")
        lines.append("")

    # Data retention
    lines.append("## 4. Data Retention\n")
    lines.append("Initial cycles are discarded when insufficient history exists for the largest window.\n")
    lines.append("| Subset | Split | Original | Retained | Discarded | Retention % |")
    lines.append("|--------|-------|----------|----------|-----------|------------|")
    for key, r in sorted(all_retention.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {r['original_rows']} | {r['feature_rows']} "
            f"| {r['discarded_rows']} | {r['retention_pct']}% |"
        )

    # Feature quality
    lines.append("\n## 5. Feature Quality\n")
    lines.append("| Subset | Split | Features | Rows | NaN Cols | Inf Cols | Constant | Dup Rows |")
    lines.append("|--------|-------|----------|------|----------|----------|----------|----------|")
    for key, q in sorted(all_quality.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {q['n_features']} | {q['n_rows']} "
            f"| {q['n_cols_with_nan']} | {q['n_cols_with_inf']} "
            f"| {q['n_constant_features']} | {q['n_duplicate_rows']} |"
        )

    # Target integrity
    lines.append("\n## 6. Target Integrity\n")
    lines.append("| Subset | Split | RUL≥0 | No Leakage | Matches Original |")
    lines.append("|--------|-------|-------|------------|------------------|")
    for key, t in sorted(all_target.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {'PASS' if t['rul_non_negative'] else 'FAIL'} "
            f"| {'PASS' if t['no_failure_cycle_leakage'] else 'FAIL'} "
            f"| {'PASS' if t.get('rul_matches_original') else 'N/A'} |"
        )

    # Leakage verification
    lines.append("\n## 7. Leakage Verification\n")
    lines.append("### Temporal Causality")
    lines.append("All features at cycle t depend only on cycles ≤ t. Verified by automated test")
    lines.append("that modifies future cycles and checks features at earlier cycles remain identical.\n")
    lines.append("### Engine Separation")
    for key, es in sorted(all_engine_sep.items()):
        lines.append(f"- **{key}**: train={es['n_train_engines']} engines, val={es['n_val_engines']} engines, overlap={'NONE' if es['no_overlap'] else 'DETECTED!'}")

    # Normalization
    lines.append("\n## 8. Normalization\n")
    lines.append("- **FD001/FD003**: Raw sensor values used (single operating regime)")
    lines.append("- **FD002/FD004**: Phase 2 condition-aware normalization applied BEFORE feature generation")
    lines.append("- Normalization parameters fitted on training data only\n")

    # Top correlations
    lines.append("## 9. Top Feature-Target Correlations (Pearson)\n")
    for subset, corr_df in sorted(all_top_corrs.items()):
        lines.append(f"### {subset} (top 10)\n")
        lines.append("| Feature | Correlation |")
        lines.append("|---------|------------|")
        for _, row in corr_df.head(10).iterrows():
            lines.append(f"| {row['feature']} | {row['correlation']:.4f} |")
        lines.append("")

    # Methodological note
    lines.append("## 10. Methodological Note\n")
    lines.append("> The optimal window size and feature set will be determined in Phase 4")
    lines.append("> through validation performance with regression models. Phase 3 makes")
    lines.append("> no claims about which window is 'best'.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AeroRUL -- Phase 3 Analysis Pipeline")
    print("=" * 60)

    PHASE3_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    windows = FEATURE_CONFIG["windows"]
    max_window = max(windows)

    # ── Step 1: Load and prepare data ──────────────────────────────────
    print("\n[1/9] Loading all datasets...")
    all_data = load_all_subsets()

    # ── Step 2: RUL + split + normalization ────────────────────────────
    print("\n[2/9] Applying Phase 2 pipeline (RUL → split → normalize)...")
    all_splits = {}
    all_rul = {}

    for subset in sorted(AVAILABLE_SUBSETS):
        data = all_data[subset]
        df_rul = generate_rul_targets(data["train"])
        train_df, val_df = split_by_engine(df_rul)
        all_rul[subset] = df_rul

        # Apply normalization for multi-condition subsets
        n_conditions = SUBSET_METADATA[subset]["conditions"]
        if n_conditions > 1:
            regimes_train = identify_operating_regimes(train_df)
            norm_params = fit_regime_normalization(train_df, regimes_train)
            train_norm = apply_regime_normalization(train_df, norm_params, regimes_train)
            # Replace sensor columns with normalized versions
            norm_cols = [c for c in train_norm.columns if c.endswith("_norm")]
            for nc in norm_cols:
                orig = nc.replace("_norm", "")
                if orig in SENSOR_COLUMNS:
                    train_df[orig] = train_norm[nc].values

            regimes_val = identify_operating_regimes(val_df)
            val_norm = apply_regime_normalization(val_df, norm_params, regimes_val)
            for nc in norm_cols:
                orig = nc.replace("_norm", "")
                if orig in SENSOR_COLUMNS:
                    val_df[orig] = val_norm[nc].values

            print(f"  {subset}: normalized ({n_conditions} conditions)")
        else:
            print(f"  {subset}: raw (single condition)")

        all_splits[subset] = (train_df, val_df)
        print(f"    train={len(train_df)} rows ({train_df['unit_id'].nunique()} engines), "
              f"val={len(val_df)} rows ({val_df['unit_id'].nunique()} engines)")

    # ── Step 3: Generate multi-scale features ──────────────────────────
    print(f"\n[3/9] Generating multi-scale features (windows={windows})...")
    all_train_features = {}
    all_val_features = {}
    all_retention = {}

    for subset in sorted(AVAILABLE_SUBSETS):
        train_df, val_df = all_splits[subset]

        train_feats = generate_multiscale_features(train_df, windows=windows)
        val_feats = generate_multiscale_features(val_df, windows=windows)

        all_train_features[subset] = train_feats
        all_val_features[subset] = val_feats

        # Retention report
        all_retention[(subset, "train")] = data_retention_report(
            len(train_df), len(train_feats),
            train_df["unit_id"].nunique(), train_feats["unit_id"].nunique(),
            max_window,
        )
        all_retention[(subset, "val")] = data_retention_report(
            len(val_df), len(val_feats),
            val_df["unit_id"].nunique(), val_feats["unit_id"].nunique(),
            max_window,
        )

        feat_cols = get_feature_columns(train_feats)
        print(f"  {subset}: {len(feat_cols)} features, "
              f"train={len(train_feats)} rows, val={len(val_feats)} rows")

    # ── Step 4: Feature quality validation ─────────────────────────────
    print("\n[4/9] Validating feature quality...")
    all_quality = {}
    all_target = {}
    all_engine_sep = {}

    for subset in sorted(AVAILABLE_SUBSETS):
        train_feats = all_train_features[subset]
        val_feats = all_val_features[subset]
        train_df, val_df = all_splits[subset]

        # Quality
        q_train = validate_feature_quality(train_feats)
        q_val = validate_feature_quality(val_feats)
        all_quality[(subset, "train")] = q_train
        all_quality[(subset, "val")] = q_val

        # Target integrity
        t_train = validate_target_integrity(train_feats, train_df)
        t_val = validate_target_integrity(val_feats, val_df)
        all_target[(subset, "train")] = t_train
        all_target[(subset, "val")] = t_val

        # Engine separation
        es = validate_engine_separation(train_feats, val_feats)
        all_engine_sep[subset] = es

        print(f"  {subset}: quality={'CLEAN' if q_train['all_clean'] and q_val['all_clean'] else 'ISSUES'}, "
              f"target={'VALID' if t_train['all_valid'] and t_val['all_valid'] else 'INVALID'}, "
              f"overlap={'NONE' if es['no_overlap'] else 'DETECTED!'}")

    # ── Step 5: Save feature datasets ──────────────────────────────────
    print("\n[5/9] Saving feature datasets (Parquet)...")
    for subset in sorted(AVAILABLE_SUBSETS):
        subset_dir = PHASE3_DATA_DIR / subset
        subset_dir.mkdir(parents=True, exist_ok=True)

        all_train_features[subset].to_parquet(
            subset_dir / "train_features.parquet", index=False,
        )
        all_val_features[subset].to_parquet(
            subset_dir / "val_features.parquet", index=False,
        )
        print(f"  {subset}: saved train_features.parquet, val_features.parquet")

    # ── Step 6: Feature metadata ───────────────────────────────────────
    print("\n[6/9] Generating feature metadata...")
    sample_feats = all_train_features["FD001"]
    feat_cols = get_feature_columns(sample_feats)
    feature_metadata = generate_feature_metadata(feat_cols)
    feature_metadata.to_csv(PHASE3_REPORTS_DIR / "feature_metadata.csv", index=False)
    print(f"  {len(feature_metadata)} features documented")

    # Feature-target correlations
    all_top_corrs = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        pearson = feature_target_correlations(all_train_features[subset], method="pearson", top_n=50)
        spearman = feature_target_correlations(all_train_features[subset], method="spearman", top_n=50)
        all_top_corrs[subset] = pearson

        corr_combined = pearson.merge(
            spearman[["feature", "correlation"]],
            on="feature", suffixes=("_pearson", "_spearman"),
            how="outer",
        )
        corr_combined.to_csv(
            PHASE3_REPORTS_DIR / f"feature_target_correlations_{subset}.csv", index=False,
        )
    print(f"  Saved correlation reports for all subsets")

    # Feature summary per window
    summary_rows = []
    for W in windows:
        # Count features for this window
        w_feats = [c for c in feat_cols if c.endswith(f"_{W}")]
        summary_rows.append({
            "window": W,
            "n_window_features": len(w_feats),
        })
    # Add delta/relchange counts
    delta_feats = [c for c in feat_cols if "_delta_" in c]
    relchange_feats = [c for c in feat_cols if "_relchange_" in c]
    summary_rows.append({
        "window": "delta",
        "n_window_features": len(delta_feats),
    })
    summary_rows.append({
        "window": "relchange",
        "n_window_features": len(relchange_feats),
    })
    feat_summary = pd.DataFrame(summary_rows)
    feat_summary.to_csv(PHASE3_REPORTS_DIR / "feature_summary.csv", index=False)

    # Feature quality CSV
    quality_rows = []
    for key, q in sorted(all_quality.items()):
        quality_rows.append({
            "subset": key[0], "split": key[1],
            "n_features": q["n_features"], "n_rows": q["n_rows"],
            "total_nan": q["total_nan"], "n_cols_with_nan": q["n_cols_with_nan"],
            "total_inf": q["total_inf"], "n_cols_with_inf": q["n_cols_with_inf"],
            "n_constant": q["n_constant_features"],
            "n_dup_rows": q["n_duplicate_rows"],
        })
    pd.DataFrame(quality_rows).to_csv(PHASE3_REPORTS_DIR / "feature_quality.csv", index=False)

    # ── Step 7: Visualizations ─────────────────────────────────────────
    print("\n[7/9] Generating Phase 3 visualizations...")
    for subset in sorted(AVAILABLE_SUBSETS):
        train_df, _ = all_splits[subset]
        train_feats = all_train_features[subset]

        plot_rolling_mean_example(train_df, train_feats, subset, PHASE3_FIGURES_DIR)
        plot_degradation_slope(train_df, train_feats, subset, PHASE3_FIGURES_DIR)
        plot_multiscale_slopes(train_feats, subset, PHASE3_FIGURES_DIR)
        plot_feature_target_scatter(train_feats, subset, PHASE3_FIGURES_DIR)
        plot_feature_correlation_heatmap(train_feats, subset, PHASE3_FIGURES_DIR)
        print(f"  {subset}: all plots saved")

    # ── Step 8: Generate report ────────────────────────────────────────
    print("\n[8/9] Generating Phase 3 summary report...")
    generate_phase3_summary(
        all_retention, all_quality, all_target, all_engine_sep,
        all_top_corrs, feature_metadata,
        PHASE3_REPORTS_DIR / "phase3_summary.md",
    )

    # ── Step 9: Final integrity check ──────────────────────────────────
    print("\n[9/9] Final integrity checks...")
    all_ok = True
    for subset in sorted(AVAILABLE_SUBSETS):
        es = all_engine_sep[subset]
        q_t = all_quality[(subset, "train")]
        q_v = all_quality[(subset, "val")]
        t_t = all_target[(subset, "train")]
        t_v = all_target[(subset, "val")]

        ok = es["no_overlap"] and t_t["all_valid"] and t_v["all_valid"]
        if not ok:
            all_ok = False
        print(f"  {subset}: engine_sep={'PASS' if es['no_overlap'] else 'FAIL'}, "
              f"target={'PASS' if t_t['all_valid'] and t_v['all_valid'] else 'FAIL'}, "
              f"nan_train={q_t['total_nan']}, nan_val={q_v['total_nan']}")

    print("\n" + "=" * 60)
    print(f"Phase 3 analysis {'COMPLETE' if all_ok else 'COMPLETE WITH WARNINGS'}!")
    print(f"Features: {PHASE3_DATA_DIR}")
    print(f"Reports:  {PHASE3_REPORTS_DIR}")
    print(f"Figures:  {PHASE3_FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
