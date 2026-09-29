"""
AeroRUL — Phase 3.1 Analysis Pipeline

Run the complete Phase 3.1 Feature Quality Control audit.

Usage:
    python -m src.run_phase3_1_analysis
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
    DISTRIBUTION_SHIFT_THRESHOLD,
    FEATURE_CONFIG,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_STYLE,
    METADATA_COLUMNS,
    OP_SETTING_COLUMNS,
    PHASE3_1_FIGURES_DIR,
    PHASE3_1_REPORTS_DIR,
    PHASE3_DATA_DIR,
    SENSOR_COLUMNS,
    SUBSET_METADATA,
    TARGET_COLUMN,
)
from src.data_loader import load_all_subsets
from src.feature_quality_control import (
    constant_feature_audit,
    detect_duplicate_features,
    determine_nan_handling,
    distribution_shift_analysis,
    generate_feature_usability_inventory,
    nan_root_cause_analysis,
    verify_engine_separation,
    verify_schema_consistency,
    verify_target_isolation,
    verify_temporal_causality,
)
from src.feature_validation import validate_feature_quality
from src.features import (
    generate_feature_metadata,
    generate_multiscale_features,
    get_feature_columns,
)
from src.normalization import (
    apply_regime_normalization,
    fit_regime_normalization,
    identify_operating_regimes,
)
from src.rul import generate_rul_targets
from src.splitting import split_by_engine


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Use DISTRIBUTION_SHIFT_THRESHOLD from config
SHIFT_THRESHOLD = DISTRIBUTION_SHIFT_THRESHOLD


def _apply_style():
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_nan_distribution(nan_audit: pd.DataFrame, subset: str, output_dir: Path):
    """Plot NaN distribution by feature family."""
    if len(nan_audit) == 0:
        return

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By feature family
    family_nans = nan_audit.groupby("feature_family")["nan_count_train"].sum()
    if len(family_nans) > 0:
        family_nans.plot.bar(ax=axes[0], color="coral", edgecolor="black")
        axes[0].set_title(f"{subset}: NaN Count by Feature Family (Train)", fontweight="bold")
        axes[0].set_ylabel("Total NaN Count")
        axes[0].tick_params(axis="x", rotation=45)

    # By window size
    window_nans = nan_audit.groupby("window")["nan_count_train"].sum()
    if len(window_nans) > 0:
        window_nans.plot.bar(ax=axes[1], color="steelblue", edgecolor="black")
        axes[1].set_title(f"{subset}: NaN Count by Window Size (Train)", fontweight="bold")
        axes[1].set_ylabel("Total NaN Count")

    fig.tight_layout()
    fig.savefig(output_dir / f"nan_distribution_{subset}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_constant_features(const_audit: pd.DataFrame, subset: str, output_dir: Path):
    """Plot constant feature distribution."""
    if len(const_audit) == 0:
        return

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By category
    cat_counts = const_audit["category"].value_counts()
    if len(cat_counts) > 0:
        cat_counts.plot.bar(ax=axes[0], color="mediumpurple", edgecolor="black")
        axes[0].set_title(f"{subset}: Constant Features by Category", fontweight="bold")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)

    # By feature family
    fam_counts = const_audit["feature_family"].value_counts()
    if len(fam_counts) > 0:
        fam_counts.plot.bar(ax=axes[1], color="darkseagreen", edgecolor="black")
        axes[1].set_title(f"{subset}: Constant Features by Family", fontweight="bold")
        axes[1].set_ylabel("Count")
        axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / f"constant_features_{subset}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_distribution_shift(shift_report: pd.DataFrame, subset: str, output_dir: Path):
    """Plot distribution shift summary."""
    if len(shift_report) == 0:
        return

    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of standardized differences
    std_diffs = shift_report["standardized_difference"].dropna()
    if len(std_diffs) > 0:
        axes[0].hist(std_diffs, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[0].axvline(x=SHIFT_THRESHOLD, color="red", linestyle="--", label=f"Threshold={SHIFT_THRESHOLD}")
        axes[0].axvline(x=-SHIFT_THRESHOLD, color="red", linestyle="--")
        axes[0].set_title(f"{subset}: Standardized Mean Difference Distribution", fontweight="bold")
        axes[0].set_xlabel("Standardized Difference (Cohen's d)")
        axes[0].set_ylabel("Feature Count")
        axes[0].legend()

    # Shifted vs non-shifted
    n_shifted = int(shift_report["shift_flag"].sum())
    n_ok = len(shift_report) - n_shifted
    if n_shifted > 0 or n_ok > 0:
        axes[1].bar(["Not Shifted", "Shifted"], [n_ok, n_shifted],
                    color=["green", "red"], edgecolor="black", alpha=0.7)
        axes[1].set_title(f"{subset}: Feature Shift Summary", fontweight="bold")
        axes[1].set_ylabel("Feature Count")

    fig.tight_layout()
    fig.savefig(output_dir / f"distribution_shift_{subset}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_usability_summary(inventory: pd.DataFrame, subset: str, output_dir: Path):
    """Plot feature usability summary."""
    if len(inventory) == 0:
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    usability_counts = inventory["usable_for_phase4"].value_counts()
    colors = {"YES": "green", "NO": "red", "REQUIRES_FIX": "orange"}
    bar_colors = [colors.get(k, "gray") for k in usability_counts.index]
    usability_counts.plot.bar(ax=ax, color=bar_colors, edgecolor="black")
    ax.set_title(f"{subset}: Feature Usability for Phase 4", fontweight="bold")
    ax.set_ylabel("Feature Count")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_dir / f"usability_summary_{subset}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase3_1_summary(
    all_nan_audits, all_constant_audits, all_duplicates,
    all_schema, all_leakage, all_shift,
    all_inventories, output_path,
):
    """Generate Phase 3.1 summary report."""
    lines = []
    lines.append("# AeroRUL — Phase 3.1 Feature Quality Control Report\n")

    # ── 1. Executive Summary ──
    lines.append("## 1. Executive Summary\n")

    # Determine overall status
    all_pass = True
    blocking_issues = []

    for subset in sorted(all_schema.keys()):
        schema = all_schema[subset]
        if not schema["schema_valid"]:
            all_pass = False
            blocking_issues.append(f"{subset}: Schema validation failed")

    for subset in sorted(all_leakage.keys()):
        leak = all_leakage[subset]
        if not leak["temporal_causality"]["verified"]:
            all_pass = False
            blocking_issues.append(f"{subset}: Temporal causality violation")
        if not leak["engine_separation"]["verified"]:
            all_pass = False
            blocking_issues.append(f"{subset}: Engine separation violation")
        if not leak["target_isolation"]["verified"]:
            all_pass = False
            blocking_issues.append(f"{subset}: Target isolation violation")

    if all_pass:
        lines.append("**Phase 3.1 Status: PASS**\n")
        lines.append("All quality control checks passed. The feature pipeline is ")
        lines.append("mathematically valid, leakage-safe, schema-consistent, and ")
        lines.append("ready for Phase 4 classical ML benchmarking.\n")
    else:
        lines.append("**Phase 3.1 Status: NOT READY**\n")
        lines.append("Blocking issues:\n")
        for issue in blocking_issues:
            lines.append(f"- {issue}")
        lines.append("")

    # ── 2. NaN Analysis ──
    lines.append("## 2. NaN Analysis\n")
    for subset in sorted(all_nan_audits.keys()):
        nan_audit = all_nan_audits[subset]
        n_nan_features = len(nan_audit)
        lines.append(f"### {subset}\n")
        lines.append(f"- **NaN-affected features**: {n_nan_features}")

        if n_nan_features > 0:
            # Group by root cause
            cause_counts = nan_audit["root_cause_category"].value_counts()
            for cause, count in cause_counts.items():
                lines.append(f"- {cause}: {count} features")

            # Group by feature family
            family_counts = nan_audit.groupby("feature_family")["nan_count_train"].sum()
            lines.append(f"\n| Feature Family | Total NaN (Train) |")
            lines.append(f"|----------------|-------------------|")
            for fam, count in family_counts.items():
                lines.append(f"| {fam} | {int(count)} |")

            # Actions
            if "action" in nan_audit.columns:
                action_counts = nan_audit["action"].value_counts()
                lines.append(f"\n**Actions taken:**")
                for action, count in action_counts.items():
                    lines.append(f"- {action}: {count} features")
        lines.append("")

    # ── 3. Constant Feature Analysis ──
    lines.append("## 3. Constant Feature Analysis\n")
    for subset in sorted(all_constant_audits.keys()):
        const_audit = all_constant_audits[subset]
        n_const = len(const_audit)
        lines.append(f"### {subset}\n")
        lines.append(f"- **Total constant features**: {n_const}")

        if n_const > 0:
            cat_counts = const_audit["category"].value_counts()
            for cat, count in cat_counts.items():
                lines.append(f"- {cat}: {count} features")

            both_count = int(const_audit["constant_both"].sum())
            train_only = int((const_audit["constant_train"] & ~const_audit["constant_val"]).sum())
            val_only = int((~const_audit["constant_train"] & const_audit["constant_val"]).sum())
            lines.append(f"- Constant in both: {both_count}")
            lines.append(f"- Train-only constant: {train_only}")
            lines.append(f"- Val-only constant: {val_only}")
        lines.append("")

    # ── 4. Redundancy Analysis ──
    lines.append("## 4. Redundancy Analysis\n")
    for subset in sorted(all_duplicates.keys()):
        dup = all_duplicates[subset]
        n_dup = len(dup)
        if n_dup > 0:
            n_groups = dup["group_id"].nunique()
            lines.append(f"### {subset}: {n_groups} duplicate groups found\n")
            for gid in dup["group_id"].unique():
                group = dup[dup["group_id"] == gid]
                features = group["feature_name"].tolist()
                lines.append(f"- Group {gid}: {', '.join(features)}")
        else:
            lines.append(f"### {subset}: No exact duplicate features detected\n")
        lines.append("")

    # ── 5. Schema Validation ──
    lines.append("## 5. Schema Validation\n")
    lines.append("| Subset | Same Columns | Same Count | Same Order | Forbidden | Status |")
    lines.append("|--------|-------------|------------|------------|-----------|--------|")
    for subset in sorted(all_schema.keys()):
        s = all_schema[subset]
        status = "PASS" if s["schema_valid"] else "FAIL"
        forbidden = ", ".join(s["forbidden_in_features"]) if s["forbidden_in_features"] else "None"
        lines.append(
            f"| {subset} | {'✓' if s['same_columns'] else '✗'} "
            f"| {'✓' if s['same_feature_count'] else '✗'} "
            f"| {'✓' if s['same_feature_order'] else '✗'} "
            f"| {forbidden} | {status} |"
        )
    lines.append("")

    # ── 6. Leakage Validation ──
    lines.append("## 6. Leakage Validation\n")
    for subset in sorted(all_leakage.keys()):
        leak = all_leakage[subset]
        lines.append(f"### {subset}\n")
        tc = leak["temporal_causality"]
        es = leak["engine_separation"]
        ti = leak["target_isolation"]
        lines.append(f"- **Temporal Causality**: {'PASS' if tc['verified'] else 'FAIL'}")
        if not tc["verified"] and tc.get("violations"):
            for v in tc["violations"][:3]:
                lines.append(f"  - Violation: {v['feature']}")
        lines.append(f"- **Engine Separation**: {'PASS' if es['verified'] else 'FAIL'} "
                     f"(train={es['n_train_engines']}, val={es['n_val_engines']})")
        lines.append(f"- **Target Isolation**: {'PASS' if ti['verified'] else 'FAIL'}")
        if "normalization_isolation" in leak:
            ni = leak["normalization_isolation"]
            lines.append(f"- **Normalization Isolation**: {'PASS' if ni['verified'] else 'FAIL'}")
        lines.append("")

    # ── 7. Distribution Shift ──
    lines.append("## 7. Distribution Shift Summary\n")
    lines.append(f"Threshold for shift flag: |Cohen's d| > {SHIFT_THRESHOLD}\n")
    lines.append("| Subset | Total Features | Shifted | Not Shifted | Shift % |")
    lines.append("|--------|---------------|---------|-------------|---------|")
    for subset in sorted(all_shift.keys()):
        shift = all_shift[subset]
        n_total = len(shift)
        n_shifted = int(shift["shift_flag"].sum()) if n_total > 0 else 0
        n_ok = n_total - n_shifted
        pct = round(100.0 * n_shifted / n_total, 1) if n_total > 0 else 0
        lines.append(f"| {subset} | {n_total} | {n_shifted} | {n_ok} | {pct}% |")
    lines.append("")

    # ── 8. Phase 4 Readiness ──
    lines.append("## 8. Final Phase 4 Readiness\n")
    for subset in sorted(all_inventories.keys()):
        inv = all_inventories[subset]
        n_yes = int((inv["usable_for_phase4"] == "YES").sum())
        n_no = int((inv["usable_for_phase4"] == "NO").sum())
        n_fix = int((inv["usable_for_phase4"] == "REQUIRES_FIX").sum())
        total = len(inv)
        lines.append(f"### {subset}")
        lines.append(f"- Total features: {total}")
        lines.append(f"- YES (ready): {n_yes}")
        lines.append(f"- NO (excluded): {n_no}")
        lines.append(f"- REQUIRES_FIX: {n_fix}")
        lines.append("")

    if all_pass:
        lines.append("**Status: READY for Phase 4 — Classical Regression Benchmarking**\n")
    else:
        lines.append("**Status: NOT READY**\n")
        lines.append("Blocking issues must be resolved before Phase 4.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AeroRUL — Phase 3.1 Feature Quality Control")
    print("=" * 60)

    PHASE3_1_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE3_1_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    windows = FEATURE_CONFIG["windows"]

    # ── Step 1: Load Phase 3 features ──
    print("\n[1/10] Loading Phase 3 feature datasets...")
    all_train = {}
    all_val = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        subset_dir = PHASE3_DATA_DIR / subset
        train_path = subset_dir / "train_features.parquet"
        val_path = subset_dir / "val_features.parquet"

        if not train_path.exists() or not val_path.exists():
            print(f"  {subset}: MISSING — regenerating features...")
            # Regenerate if missing
            continue

        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        all_train[subset] = train_df
        all_val[subset] = val_df
        feat_cols = get_feature_columns(train_df)
        print(f"  {subset}: train={len(train_df)} rows, val={len(val_df)} rows, "
              f"features={len(feat_cols)}")

    if len(all_train) < 4:
        print("\n  Some datasets missing. Regenerating all features from scratch...")
        all_data = load_all_subsets()
        for subset in sorted(AVAILABLE_SUBSETS):
            if subset in all_train:
                continue

            data = all_data[subset]
            df_rul = generate_rul_targets(data["train"])
            train_df, val_df = split_by_engine(df_rul)

            # Normalization for multi-condition subsets
            n_conditions = SUBSET_METADATA[subset]["conditions"]
            if n_conditions > 1:
                regimes_train = identify_operating_regimes(train_df)
                norm_params = fit_regime_normalization(train_df, regimes_train)
                train_norm = apply_regime_normalization(train_df, norm_params, regimes_train)
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

            train_feats = generate_multiscale_features(train_df, windows=windows)
            val_feats = generate_multiscale_features(val_df, windows=windows)
            all_train[subset] = train_feats
            all_val[subset] = val_feats
            feat_cols = get_feature_columns(train_feats)
            print(f"  {subset}: regenerated — train={len(train_feats)}, "
                  f"val={len(val_feats)}, features={len(feat_cols)}")

    # ── Step 2: Feature metadata ──
    print("\n[2/10] Generating feature metadata...")
    sample_feats = all_train["FD001"]
    feat_cols = get_feature_columns(sample_feats)
    feature_metadata = generate_feature_metadata(feat_cols)
    print(f"  {len(feature_metadata)} features documented")

    # ── Step 3: NaN Root-Cause Analysis ──
    print("\n[3/10] NaN root-cause analysis...")
    all_nan_audits = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        nan_audit = nan_root_cause_analysis(
            all_train[subset], all_val[subset], feature_metadata,
        )
        nan_audit = determine_nan_handling(nan_audit)
        all_nan_audits[subset] = nan_audit

        n_nan = len(nan_audit)
        if n_nan > 0:
            categories = nan_audit["root_cause_category"].value_counts().to_dict()
            print(f"  {subset}: {n_nan} NaN features — {categories}")
        else:
            print(f"  {subset}: No NaN features")

        # Save
        nan_audit.to_csv(
            PHASE3_1_REPORTS_DIR / f"nan_audit_{subset}.csv", index=False,
        )

    # Merge all nan audits into single CSV
    all_nan_combined = pd.concat(
        [df.assign(subset=s) for s, df in all_nan_audits.items()],
        ignore_index=True,
    )
    all_nan_combined.to_csv(
        PHASE3_1_REPORTS_DIR / "nan_audit.csv", index=False,
    )

    # ── Step 4: Constant Feature Audit ──
    print("\n[4/10] Constant feature audit...")
    all_constant_audits = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        # Determine which sensors are constant in this dataset's training set
        const_sensors = []
        for sc in SENSOR_COLUMNS:
            if sc in all_train[subset].columns:
                # Check if the sensor column exists in the feature dataframe
                # The feature df may not have raw sensors, so check the raw
                # features derived from this sensor
                sensor_feats = [c for c in get_feature_columns(all_train[subset])
                               if c.startswith(f"{sc}_")]
                if sensor_feats:
                    # If all derived features from this sensor are constant,
                    # the sensor is likely constant
                    all_const = all([all_train[subset][f].var() == 0 for f in sensor_feats
                                    if f in all_train[subset].columns])
                    if all_const:
                        const_sensors.append(sc)

        const_audit = constant_feature_audit(
            all_train[subset], all_val[subset], feature_metadata,
            constant_sensors=const_sensors,
        )
        all_constant_audits[subset] = const_audit

        n_const = len(const_audit)
        if n_const > 0:
            categories = const_audit["category"].value_counts().to_dict()
            print(f"  {subset}: {n_const} constant features — {categories}")
        else:
            print(f"  {subset}: No constant features")

        const_audit.to_csv(
            PHASE3_1_REPORTS_DIR / f"constant_feature_audit_{subset}.csv", index=False,
        )

    # Merge constant audits
    all_const_combined = pd.concat(
        [df.assign(subset=s) for s, df in all_constant_audits.items()],
        ignore_index=True,
    )
    all_const_combined.to_csv(
        PHASE3_1_REPORTS_DIR / "constant_feature_audit.csv", index=False,
    )

    # ── Step 5: Duplicate Feature Detection ──
    print("\n[5/10] Duplicate feature detection...")
    all_duplicates = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        dup = detect_duplicate_features(all_train[subset])
        all_duplicates[subset] = dup

        if len(dup) > 0:
            n_groups = dup["group_id"].nunique()
            print(f"  {subset}: {n_groups} duplicate groups found")
        else:
            print(f"  {subset}: No exact duplicates")

    # Merge duplicates
    all_dup_combined = pd.concat(
        [df.assign(subset=s) for s, df in all_duplicates.items()],
        ignore_index=True,
    )
    all_dup_combined.to_csv(
        PHASE3_1_REPORTS_DIR / "duplicate_feature_groups.csv", index=False,
    )

    # ── Step 6: Schema Consistency Verification ──
    print("\n[6/10] Schema consistency verification...")
    all_schema = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        schema = verify_schema_consistency(all_train[subset], all_val[subset])
        all_schema[subset] = schema
        print(f"  {subset}: {'PASS' if schema['schema_valid'] else 'FAIL'} "
              f"(train={schema['n_train_features']}, val={schema['n_val_features']})")
        if schema["forbidden_in_features"]:
            print(f"    WARNING: Forbidden columns in features: {schema['forbidden_in_features']}")

    # ── Step 7: Leakage Audit ──
    print("\n[7/10] Leakage audit...")
    all_leakage = {}

    # Load raw data for temporal causality check
    all_data = load_all_subsets()

    for subset in sorted(AVAILABLE_SUBSETS):
        print(f"  {subset}:")
        data = all_data[subset]
        df_rul = generate_rul_targets(data["train"])
        train_raw, val_raw = split_by_engine(df_rul)

        # Normalization for multi-condition subsets
        n_conditions = SUBSET_METADATA[subset]["conditions"]
        norm_params_for_check = None
        if n_conditions > 1:
            regimes_train = identify_operating_regimes(train_raw)
            norm_params_for_check = fit_regime_normalization(train_raw, regimes_train)
            train_norm = apply_regime_normalization(train_raw, norm_params_for_check, regimes_train)
            norm_cols = [c for c in train_norm.columns if c.endswith("_norm")]
            for nc in norm_cols:
                orig = nc.replace("_norm", "")
                if orig in SENSOR_COLUMNS:
                    train_raw[orig] = train_norm[nc].values

        # Temporal causality
        tc = verify_temporal_causality(train_raw, window=10, test_cycle=30)
        print(f"    Temporal causality: {'PASS' if tc['verified'] else 'FAIL'}")

        # Engine separation
        es = verify_engine_separation(all_train[subset], all_val[subset])
        print(f"    Engine separation: {'PASS' if es['verified'] else 'FAIL'}")

        # Target isolation
        ti = verify_target_isolation(all_train[subset], all_val[subset])
        print(f"    Target isolation: {'PASS' if ti['verified'] else 'FAIL'}")

        leak_result = {
            "temporal_causality": tc,
            "engine_separation": es,
            "target_isolation": ti,
        }

        # Normalization isolation (multi-condition only)
        if norm_params_for_check is not None:
            from src.feature_quality_control import verify_normalization_isolation
            ni = verify_normalization_isolation(train_raw, val_raw, norm_params_for_check)
            print(f"    Normalization isolation: {'PASS' if ni['verified'] else 'FAIL'}")
            leak_result["normalization_isolation"] = ni

        all_leakage[subset] = leak_result

    # ── Step 8: Distribution Shift Analysis ──
    print("\n[8/10] Distribution shift analysis...")
    all_shift = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        shift = distribution_shift_analysis(
            all_train[subset], all_val[subset],
            shift_threshold=SHIFT_THRESHOLD,
        )
        all_shift[subset] = shift

        n_shifted = int(shift["shift_flag"].sum()) if len(shift) > 0 else 0
        n_total = len(shift)
        pct = round(100.0 * n_shifted / n_total, 1) if n_total > 0 else 0
        print(f"  {subset}: {n_shifted}/{n_total} features shifted ({pct}%)")

        shift.to_csv(
            PHASE3_1_REPORTS_DIR / f"distribution_shift_{subset}.csv", index=False,
        )

    # Merge shift reports
    all_shift_combined = pd.concat(
        [df.assign(subset=s) for s, df in all_shift.items()],
        ignore_index=True,
    )
    all_shift_combined.to_csv(
        PHASE3_1_REPORTS_DIR / "distribution_shift_report.csv", index=False,
    )

    # ── Step 9: Feature Usability Inventory ──
    print("\n[9/10] Feature usability inventory...")
    all_inventories = {}
    for subset in sorted(AVAILABLE_SUBSETS):
        inventory = generate_feature_usability_inventory(
            all_train[subset], all_val[subset],
            feature_metadata,
            all_nan_audits[subset],
            all_constant_audits[subset],
            all_duplicates[subset],
        )
        all_inventories[subset] = inventory

        n_yes = int((inventory["usable_for_phase4"] == "YES").sum())
        n_no = int((inventory["usable_for_phase4"] == "NO").sum())
        n_fix = int((inventory["usable_for_phase4"] == "REQUIRES_FIX").sum())
        print(f"  {subset}: YES={n_yes}, NO={n_no}, REQUIRES_FIX={n_fix}")

        inventory.to_csv(
            PHASE3_1_REPORTS_DIR / f"feature_quality_inventory_{subset}.csv", index=False,
        )

    # Merge inventories
    all_inv_combined = pd.concat(
        [df.assign(subset=s) for s, df in all_inventories.items()],
        ignore_index=True,
    )
    all_inv_combined.to_csv(
        PHASE3_1_REPORTS_DIR / "feature_quality_inventory.csv", index=False,
    )

    # ── Step 10: Visualizations and Report ──
    print("\n[10/10] Generating visualizations and report...")
    for subset in sorted(AVAILABLE_SUBSETS):
        plot_nan_distribution(all_nan_audits[subset], subset, PHASE3_1_FIGURES_DIR)
        plot_constant_features(all_constant_audits[subset], subset, PHASE3_1_FIGURES_DIR)
        plot_distribution_shift(all_shift[subset], subset, PHASE3_1_FIGURES_DIR)
        plot_usability_summary(all_inventories[subset], subset, PHASE3_1_FIGURES_DIR)
        print(f"  {subset}: plots saved")

    generate_phase3_1_summary(
        all_nan_audits, all_constant_audits, all_duplicates,
        all_schema, all_leakage, all_shift,
        all_inventories,
        PHASE3_1_REPORTS_DIR / "phase3_1_summary.md",
    )

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("Phase 3.1 Feature Quality Control COMPLETE")
    print(f"Reports:  {PHASE3_1_REPORTS_DIR}")
    print(f"Figures:  {PHASE3_1_FIGURES_DIR}")
    print("=" * 60)

    # Print summary
    for subset in sorted(AVAILABLE_SUBSETS):
        inv = all_inventories[subset]
        n_yes = int((inv["usable_for_phase4"] == "YES").sum())
        n_no = int((inv["usable_for_phase4"] == "NO").sum())
        n_fix = int((inv["usable_for_phase4"] == "REQUIRES_FIX").sum())
        print(f"  {subset}: {n_yes} YES / {n_no} NO / {n_fix} REQUIRES_FIX")


if __name__ == "__main__":
    main()
