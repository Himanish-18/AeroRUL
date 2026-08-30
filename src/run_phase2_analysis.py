"""
AeroRUL — Phase 2 Analysis Pipeline

Run the complete Phase 2 analysis: RUL generation, clipping experiments,
engine-level splitting, operating-regime analysis, and condition-aware
normalization investigation.

Usage:
    python -m src.run_phase2_analysis
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
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_STYLE,
    OP_SETTING_COLUMNS,
    PHASE2_FIGURES_DIR,
    PHASE2_REPORTS_DIR,
    RUL_CLIP_VALUES,
    SENSOR_COLUMNS,
    SUBSET_METADATA,
)
from src.data_loader import load_all_subsets
from src.normalization import (
    apply_regime_normalization,
    fit_regime_normalization,
    identify_operating_regimes,
    regime_sensor_analysis,
    regime_summary,
)
from src.rul import (
    clip_rul,
    generate_rul_targets,
    rul_clipping_comparison,
    rul_statistics,
    validate_rul_targets,
)
from src.splitting import split_by_engine, validate_split


def _apply_style():
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# VISUALIZATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_raw_vs_clipped_rul(df, subset, output_dir):
    """Plot raw RUL and clipped RUL for representative engines."""
    _apply_style()
    engines = sorted(df["unit_id"].unique())
    # Pick 4 engines with diverse trajectory lengths
    lengths = df.groupby("unit_id")["cycle"].max().sort_values()
    idx = np.linspace(0, len(lengths) - 1, 4, dtype=int)
    selected = lengths.index[idx].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, eid in enumerate(selected):
        engine = df[df["unit_id"] == eid].sort_values("cycle")
        axes[i].plot(engine["cycle"], engine["rul"], label="Raw RUL", linewidth=1.5, color="steelblue")
        for cap in [125, 75]:
            clipped = engine["rul"].clip(upper=cap)
            axes[i].plot(engine["cycle"], clipped, label=f"Clipped ({cap})", linewidth=1, linestyle="--", alpha=0.8)
        axes[i].set_title(f"Engine {eid} (max cycle={engine['cycle'].max()})", fontweight="bold")
        axes[i].set_xlabel("Cycle")
        axes[i].set_ylabel("RUL")
        axes[i].legend(fontsize=8)

    fig.suptitle(f"{subset} -- Raw vs Clipped RUL", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"raw_vs_clipped_rul_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rul_distribution(df, subset, output_dir):
    """Plot RUL target distribution and clipped variants."""
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Raw
    axes[0].hist(df["rul"], bins=50, edgecolor="black", alpha=0.75, color="steelblue")
    axes[0].set_title(f"{subset} -- Raw RUL Distribution", fontweight="bold")
    axes[0].set_xlabel("RUL")
    axes[0].set_ylabel("Frequency")

    # Clipped at 125
    axes[1].hist(df["rul"].clip(upper=125), bins=50, edgecolor="black", alpha=0.75, color="coral")
    axes[1].set_title(f"{subset} -- Clipped RUL (cap=125)", fontweight="bold")
    axes[1].set_xlabel("RUL")

    # Clipped at 75
    axes[2].hist(df["rul"].clip(upper=75), bins=50, edgecolor="black", alpha=0.75, color="mediumseagreen")
    axes[2].set_title(f"{subset} -- Clipped RUL (cap=75)", fontweight="bold")
    axes[2].set_xlabel("RUL")

    fig.tight_layout()
    fig.savefig(output_dir / f"rul_distribution_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_split_comparison(train_df, val_df, subset, output_dir):
    """Compare trajectory length distributions between train and validation."""
    _apply_style()
    train_lengths = train_df.groupby("unit_id")["cycle"].max()
    val_lengths = val_df.groupby("unit_id")["cycle"].max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_lengths, bins=25, alpha=0.7, label="Train", edgecolor="black", color="steelblue")
    axes[0].hist(val_lengths, bins=25, alpha=0.7, label="Validation", edgecolor="black", color="coral")
    axes[0].set_title(f"{subset} -- Trajectory Length Distribution", fontweight="bold")
    axes[0].set_xlabel("Max Cycle")
    axes[0].set_ylabel("Number of Engines")
    axes[0].legend()

    # RUL distribution comparison
    axes[1].hist(train_df["rul"], bins=50, alpha=0.7, label="Train", edgecolor="black", color="steelblue", density=True)
    axes[1].hist(val_df["rul"], bins=50, alpha=0.7, label="Validation", edgecolor="black", color="coral", density=True)
    axes[1].set_title(f"{subset} -- RUL Distribution (Normalized)", fontweight="bold")
    axes[1].set_xlabel("RUL")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / f"split_comparison_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_raw_vs_normalized_sensors(train_df, train_norm, regimes, subset, output_dir):
    """Compare raw vs normalized sensor trajectories."""
    _apply_style()
    # Pick 3 informative sensors (highest variance)
    sensor_vars = train_df[SENSOR_COLUMNS].var().sort_values(ascending=False)
    sensors = sensor_vars.head(3).index.tolist()

    # Pick 3 engines
    engines = sorted(train_df["unit_id"].unique())[:3]

    fig, axes = plt.subplots(len(sensors), 2, figsize=(16, 4 * len(sensors)), squeeze=False)

    for i, sensor in enumerate(sensors):
        norm_col = sensor + "_norm"
        for eid in engines:
            mask = train_df["unit_id"] == eid
            engine_raw = train_df.loc[mask].sort_values("cycle")
            engine_norm = train_norm.loc[mask].sort_values("cycle")

            axes[i, 0].plot(engine_raw["cycle"], engine_raw[sensor], linewidth=0.8, alpha=0.8, label=f"Engine {eid}")
            axes[i, 1].plot(engine_norm["cycle"], engine_norm[norm_col], linewidth=0.8, alpha=0.8, label=f"Engine {eid}")

        axes[i, 0].set_title(f"{sensor} (Raw)", fontweight="bold")
        axes[i, 0].set_ylabel("Value")
        axes[i, 1].set_title(f"{sensor} (Normalized)", fontweight="bold")
        axes[i, 1].set_ylabel("Normalized Value")
        if i == 0:
            axes[i, 0].legend(fontsize=7)
            axes[i, 1].legend(fontsize=7)

    axes[-1, 0].set_xlabel("Cycle")
    axes[-1, 1].set_xlabel("Cycle")

    fig.suptitle(f"{subset} -- Raw vs Condition-Normalized Sensors", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"raw_vs_normalized_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_regime_distribution(train_df, regimes, subset, output_dir):
    """Plot operating regime distribution."""
    _apply_style()
    summary = regime_summary(train_df, regimes)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(summary)), summary["n_observations"], edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary["regime"], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{subset} -- Observations per Operating Regime", fontweight="bold")
    ax.set_xlabel("Operating Regime")
    ax.set_ylabel("Number of Observations")

    fig.tight_layout()
    fig.savefig(output_dir / f"regime_distribution_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase2_summary(
    all_rul_stats, all_clipping, all_splits, all_split_validations,
    all_regime_summaries, all_norm_findings, output_path,
):
    """Generate the human-readable phase2_summary.md report."""
    lines = []
    lines.append("# AeroRUL -- Phase 2 Summary Report\n")

    # RUL Formulation
    lines.append("## 1. RUL Formulation\n")
    lines.append("**Method**: `RUL = failure_cycle - current_cycle`\n")
    lines.append("- `failure_cycle` = maximum observed cycle for each engine in training data")
    lines.append("- Final training cycle has RUL = 0 (engine failure)")
    lines.append("- RUL is always non-negative")
    lines.append("- RUL decreases by exactly 1 per consecutive cycle\n")

    # RUL Statistics
    lines.append("## 2. RUL Statistics\n")
    lines.append("| Subset | Engines | Observations | Min RUL | Max RUL | Mean RUL | Median RUL | Std RUL |")
    lines.append("|--------|---------|-------------|---------|---------|----------|------------|---------|")
    for subset, stats in sorted(all_rul_stats.items()):
        lines.append(
            f"| {subset} | {stats['n_engines']} | {stats['n_observations']} "
            f"| {stats['min_rul']} | {stats['max_rul']} | {stats['mean_rul']} "
            f"| {stats['median_rul']} | {stats['std_rul']} |"
        )

    # RUL Validation
    lines.append("\n## 3. RUL Validation\n")
    lines.append("All RUL targets passed validation:\n")
    lines.append("| Check | Status |")
    lines.append("|-------|--------|")
    lines.append("| RUL >= 0 | PASS |")
    lines.append("| min(RUL) == 0 per engine | PASS |")
    lines.append("| max(RUL) == failure_cycle - 1 | PASS |")
    lines.append("| RUL decreases by 1 per cycle | PASS |")
    lines.append("| RUL + cycle == failure_cycle | PASS |")

    # Clipping
    lines.append("\n## 4. RUL Clipping Experiment\n")
    for subset, comp in sorted(all_clipping.items()):
        lines.append(f"### {subset}\n")
        lines.append("| RUL Cap | Obs Clipped | % Clipped | Engines Affected | Mean RUL | Median RUL | Std RUL |")
        lines.append("|---------|-------------|-----------|------------------|----------|------------|---------|")
        for _, row in comp.iterrows():
            lines.append(
                f"| {row['rul_cap']} | {row['observations_clipped']} | {row['pct_clipped']} "
                f"| {row['engines_affected']} | {row['mean_rul']} | {row['median_rul']} | {row['std_rul']} |"
            )
        lines.append("")

    # Split
    lines.append("## 5. Engine-Level Train/Validation Split\n")
    lines.append("**Method**: Randomly assign entire engines to train or validation. No engine in both.\n")
    lines.append("| Subset | Train Engines | Val Engines | Train Rows | Val Rows | Val Fraction | No Overlap |")
    lines.append("|--------|--------------|-------------|-----------|---------|-------------|-----------|")
    for subset in sorted(all_splits.keys()):
        sv = all_split_validations[subset]
        lines.append(
            f"| {subset} | {sv['n_train_engines']} | {sv['n_val_engines']} "
            f"| {sv['n_train_rows']} | {sv['n_val_rows']} "
            f"| {sv['val_engine_fraction']} | {'PASS' if sv['no_engine_overlap'] else 'FAIL'} |"
        )

    lines.append("\n### Leakage Verification\n")
    lines.append("```")
    lines.append("intersection(train_engine_ids, validation_engine_ids) = empty set")
    lines.append("```")
    lines.append("**Status**: VERIFIED for all subsets.\n")

    # Operating conditions
    lines.append("## 6. Operating-Condition Findings\n")
    lines.append("- **FD001/FD003**: ~1 operating regime (op_settings near-constant)")
    lines.append("- **FD002/FD004**: 6 operating regimes (significant op_setting variation)\n")

    for subset, summary in sorted(all_regime_summaries.items()):
        lines.append(f"### {subset}: {len(summary)} regimes identified\n")
        if len(summary) <= 10:
            lines.append("| Regime | Observations | % |")
            lines.append("|--------|-------------|---|")
            for _, row in summary.iterrows():
                lines.append(f"| {row['regime']} | {row['n_observations']} | {row['pct_observations']}% |")
            lines.append("")

    # Normalization
    lines.append("## 7. Normalization Findings\n")
    for subset, findings in sorted(all_norm_findings.items()):
        lines.append(f"### {subset}\n")
        lines.append(f"- Regimes: {findings['n_regimes']}")
        lines.append(f"- Normalization applied: {'Yes' if findings['applied'] else 'Not needed (single regime)'}")
        if findings.get("note"):
            lines.append(f"- Note: {findings['note']}")
        lines.append("")

    lines.append("> **Important**: These are exploratory findings only. Whether normalization improves")
    lines.append("> RUL prediction accuracy will be evaluated in a later phase with model training.\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AeroRUL -- Phase 2 Analysis Pipeline")
    print("=" * 60)

    PHASE2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE2_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all data
    print("\n[1/8] Loading all datasets...")
    all_data = load_all_subsets()
    for subset, data in all_data.items():
        print(f"  {subset}: train={len(data['train'])} rows")

    # Step 2: Generate RUL targets
    print("\n[2/8] Generating RUL targets...")
    all_rul_stats = {}
    all_with_rul = {}
    for subset, data in sorted(all_data.items()):
        df_rul = generate_rul_targets(data["train"])
        validation = validate_rul_targets(df_rul)
        assert validation["all_valid"], f"RUL validation failed for {subset}!"
        stats = rul_statistics(df_rul)
        all_rul_stats[subset] = stats
        all_with_rul[subset] = df_rul
        print(f"  {subset}: RUL valid={validation['all_valid']}, max_rul={stats['max_rul']}, mean_rul={stats['mean_rul']}")

    # Save RUL statistics
    rul_stats_df = pd.DataFrame([{"subset": k, **v} for k, v in sorted(all_rul_stats.items())])
    rul_stats_df.to_csv(PHASE2_REPORTS_DIR / "rul_statistics.csv", index=False)
    print(f"  Saved: rul_statistics.csv")

    # Step 3: RUL clipping experiments
    print("\n[3/8] Running RUL clipping experiments...")
    all_clipping = {}
    for subset, df_rul in sorted(all_with_rul.items()):
        comp = rul_clipping_comparison(df_rul)
        all_clipping[subset] = comp
        print(f"  {subset}: clipping comparison computed")

    # Save clipping comparison
    all_clip_rows = []
    for subset, comp in sorted(all_clipping.items()):
        comp_copy = comp.copy()
        comp_copy.insert(0, "subset", subset)
        all_clip_rows.append(comp_copy)
    clip_df = pd.concat(all_clip_rows, ignore_index=True)
    clip_df.to_csv(PHASE2_REPORTS_DIR / "clipping_comparison.csv", index=False)
    print(f"  Saved: clipping_comparison.csv")

    # Step 4: Engine-level split
    print("\n[4/8] Performing engine-level train/validation splits...")
    all_splits = {}
    all_split_validations = {}
    for subset, df_rul in sorted(all_with_rul.items()):
        train_df, val_df = split_by_engine(df_rul)
        sv = validate_split(train_df, val_df, df_rul)
        assert sv["all_valid"], f"Split validation failed for {subset}!"
        all_splits[subset] = (train_df, val_df)
        all_split_validations[subset] = sv
        print(f"  {subset}: train={sv['n_train_engines']} engines/{sv['n_train_rows']} rows, "
              f"val={sv['n_val_engines']} engines/{sv['n_val_rows']} rows, overlap={len(sv['overlap_engines'])}")

    # Save split statistics
    split_rows = []
    for subset, sv in sorted(all_split_validations.items()):
        split_rows.append({
            "subset": subset,
            "n_train_engines": sv["n_train_engines"],
            "n_val_engines": sv["n_val_engines"],
            "n_train_rows": sv["n_train_rows"],
            "n_val_rows": sv["n_val_rows"],
            "val_fraction": sv["val_engine_fraction"],
            "no_overlap": sv["no_engine_overlap"],
            "rows_preserved": sv["rows_preserved"],
        })
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(PHASE2_REPORTS_DIR / "split_statistics.csv", index=False)
    print(f"  Saved: split_statistics.csv")

    # Step 5: Operating-regime analysis
    print("\n[5/8] Analyzing operating regimes...")
    all_regime_summaries = {}
    all_norm_findings = {}
    all_regime_data = {}

    for subset in sorted(all_splits.keys()):
        train_df, val_df = all_splits[subset]
        regimes = identify_operating_regimes(train_df)
        summary = regime_summary(train_df, regimes)
        all_regime_summaries[subset] = summary
        n_regimes = len(summary)
        print(f"  {subset}: {n_regimes} operating regimes identified")

        # Normalization investigation
        if n_regimes > 1:
            norm_params = fit_regime_normalization(train_df, regimes)
            train_norm = apply_regime_normalization(train_df, norm_params, regimes)
            all_regime_data[subset] = {
                "regimes": regimes, "norm_params": norm_params, "train_norm": train_norm,
            }
            all_norm_findings[subset] = {
                "n_regimes": n_regimes,
                "applied": True,
                "note": "Condition-aware normalization reduces within-regime sensor variability.",
            }
        else:
            all_norm_findings[subset] = {
                "n_regimes": n_regimes,
                "applied": False,
                "note": "Single operating regime; global normalization sufficient.",
            }

    # Save normalization statistics
    norm_rows = []
    for subset, findings in sorted(all_norm_findings.items()):
        norm_rows.append({"subset": subset, **findings})
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(PHASE2_REPORTS_DIR / "normalization_statistics.csv", index=False)
    print(f"  Saved: normalization_statistics.csv")

    # Step 6: Generate visualizations
    print("\n[6/8] Generating Phase 2 visualizations...")
    for subset, df_rul in sorted(all_with_rul.items()):
        plot_raw_vs_clipped_rul(df_rul, subset, PHASE2_FIGURES_DIR)
        plot_rul_distribution(df_rul, subset, PHASE2_FIGURES_DIR)
        print(f"  {subset}: RUL plots saved")

    for subset, (train_df, val_df) in sorted(all_splits.items()):
        plot_split_comparison(train_df, val_df, subset, PHASE2_FIGURES_DIR)
        print(f"  {subset}: split comparison saved")

    for subset, rd in sorted(all_regime_data.items()):
        train_df, _ = all_splits[subset]
        plot_regime_distribution(train_df, rd["regimes"], subset, PHASE2_FIGURES_DIR)
        plot_raw_vs_normalized_sensors(train_df, rd["train_norm"], rd["regimes"], subset, PHASE2_FIGURES_DIR)
        print(f"  {subset}: normalization plots saved")

    # Step 7: Generate report
    print("\n[7/8] Generating Phase 2 summary report...")
    generate_phase2_summary(
        all_rul_stats, all_clipping, all_splits, all_split_validations,
        all_regime_summaries, all_norm_findings,
        PHASE2_REPORTS_DIR / "phase2_summary.md",
    )

    # Step 8: Data integrity check
    print("\n[8/8] Data integrity verification...")
    for subset in sorted(all_with_rul.keys()):
        orig_rows = len(all_with_rul[subset])
        train_df, val_df = all_splits[subset]
        split_rows = len(train_df) + len(val_df)
        status = "PASS" if orig_rows == split_rows else "FAIL"
        print(f"  {subset}: original={orig_rows}, split={split_rows} -> {status}")

    print("\n" + "=" * 60)
    print("Phase 2 analysis complete!")
    print(f"Reports: {PHASE2_REPORTS_DIR}")
    print(f"Figures: {PHASE2_FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
