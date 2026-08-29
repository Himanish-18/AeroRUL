"""
AeroRUL — Phase 1 Analysis Pipeline

Run the complete Phase 1 analysis: load all data, validate, generate
statistics, create visualizations, and produce reports.

Usage:
    python -m src.run_phase1_analysis
"""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.config import FIGURES_DIR, PHASE1_REPORTS_DIR, SENSOR_COLUMNS, SUBSET_METADATA
from src.data_loader import discover_datasets, load_all_subsets
from src.data_validation import run_full_validation
from src.visualization import generate_all_phase1_plots


def generate_dataset_statistics_csv(all_results: dict, output_path: Path):
    """Generate dataset_statistics.csv with per-subset summary."""
    rows = []
    for subset, vr in sorted(all_results.items()):
        train_traj = vr["train_trajectory"]
        test_traj = vr["test_trajectory"]
        meta = SUBSET_METADATA.get(subset, {})
        row = {
            "subset": subset,
            "conditions": meta.get("conditions", ""),
            "fault_modes": meta.get("fault_modes", ""),
            "train_engines": train_traj["n_engines"],
            "test_engines": test_traj["n_engines"],
            "train_rows": train_traj["n_observations"],
            "test_rows": test_traj["n_observations"],
            "avg_train_traj_length": train_traj["avg_trajectory_length"],
            "min_train_traj_length": train_traj["min_trajectory_length"],
            "max_train_traj_length": train_traj["max_trajectory_length"],
            "median_train_traj_length": train_traj["median_trajectory_length"],
            "avg_test_traj_length": test_traj["avg_trajectory_length"],
            "min_test_traj_length": test_traj["min_trajectory_length"],
            "max_test_traj_length": test_traj["max_trajectory_length"],
            "median_test_traj_length": test_traj["median_trajectory_length"],
            "train_dup_rows": vr["train_structure"]["n_duplicate_rows"],
            "train_dup_unit_cycle": vr["train_structure"]["n_duplicate_unit_cycle"],
            "test_dup_rows": vr["test_structure"]["n_duplicate_rows"],
            "test_dup_unit_cycle": vr["test_structure"]["n_duplicate_unit_cycle"],
            "cycles_ordered": vr["train_engines"]["cycles_ordered"],
            "cycles_start_at_1": vr["train_engines"]["cycles_start_at_1"],
        }
        if "rul_n_engines" in vr:
            row["rul_engines"] = vr["rul_n_engines"]
            row["rul_min"] = vr["rul_min"]
            row["rul_max"] = vr["rul_max"]
            row["rul_mean"] = vr["rul_mean"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


def generate_sensor_statistics_csv(all_results: dict, output_path: Path):
    """Generate sensor_statistics.csv with per-sensor, per-subset stats."""
    rows = []
    for subset, vr in sorted(all_results.items()):
        sensors = vr["train_sensors"]
        corr = vr["sensor_cycle_corr"]
        merged = sensors.merge(corr, on="sensor", how="left")
        merged.insert(0, "subset", subset)
        rows.append(merged)

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    return df


def generate_dataset_summary_md(
    discovery: dict,
    all_results: dict,
    stats_df: pd.DataFrame,
    sensor_df: pd.DataFrame,
    output_path: Path,
):
    """Generate the human-readable dataset_summary.md report."""

    lines = []
    lines.append("# AeroRUL — Phase 1 Dataset Summary Report\n")
    lines.append("## 1. Dataset Overview\n")
    lines.append("**Dataset**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)\n")
    lines.append("**Reference**: Saxena et al., PHM08 Conference\n")

    # Availability
    lines.append("### Available Subsets\n")
    lines.append("| Subset | Train | Test | RUL | Conditions | Fault Modes | Description |")
    lines.append("|--------|-------|------|-----|------------|-------------|-------------|")
    for subset, avail in sorted(discovery.items()):
        meta = SUBSET_METADATA.get(subset, {})
        lines.append(
            f"| {subset} | {'Yes' if avail['train'] else 'No'} "
            f"| {'Yes' if avail['test'] else 'No'} "
            f"| {'Yes' if avail['rul'] else 'No'} "
            f"| {meta.get('conditions', '?')} "
            f"| {meta.get('fault_modes', '?')} "
            f"| {meta.get('description', '')} |"
        )

    lines.append("\n### Schema\n")
    lines.append("- **Total columns**: 26")
    lines.append("- **ID columns**: unit_id, cycle")
    lines.append("- **Operational settings**: op_setting_1, op_setting_2, op_setting_3")
    lines.append("- **Sensors**: sensor_1 through sensor_21\n")

    # Per-subset statistics
    lines.append("## 2. Per-Subset Statistics\n")
    lines.append("| Subset | Train Engines | Test Engines | Train Rows | Test Rows | Avg Train Length | Min | Max | Median |")
    lines.append("|--------|--------------|-------------|-----------|----------|-----------------|-----|-----|--------|")
    for _, row in stats_df.iterrows():
        lines.append(
            f"| {row['subset']} | {row['train_engines']} | {row['test_engines']} "
            f"| {row['train_rows']} | {row['test_rows']} "
            f"| {row['avg_train_traj_length']} | {row['min_train_traj_length']} "
            f"| {row['max_train_traj_length']} | {row['median_train_traj_length']} |"
        )

    # RUL summary
    lines.append("\n### RUL Ground Truth Summary\n")
    lines.append("| Subset | Engines | Min RUL | Max RUL | Mean RUL |")
    lines.append("|--------|---------|---------|---------|----------|")
    for _, row in stats_df.iterrows():
        if "rul_engines" in row and pd.notna(row.get("rul_engines")):
            lines.append(
                f"| {row['subset']} | {int(row['rul_engines'])} | {int(row['rul_min'])} "
                f"| {int(row['rul_max'])} | {row['rul_mean']:.1f} |"
            )

    # Data quality
    lines.append("\n## 3. Data Quality Findings\n")

    all_clean = True
    for subset, vr in sorted(all_results.items()):
        issues = []
        if vr["train_structure"]["n_duplicate_rows"] > 0:
            issues.append(f"  - {vr['train_structure']['n_duplicate_rows']} duplicate rows in training data")
            all_clean = False
        if vr["train_structure"]["n_duplicate_unit_cycle"] > 0:
            issues.append(f"  - {vr['train_structure']['n_duplicate_unit_cycle']} duplicate (unit_id, cycle) in training data")
            all_clean = False
        if vr["test_structure"]["n_duplicate_rows"] > 0:
            issues.append(f"  - {vr['test_structure']['n_duplicate_rows']} duplicate rows in test data")
            all_clean = False

        # Missing values
        train_missing = vr["train_missing"]["n_total_missing"].sum()
        test_missing = vr["test_missing"]["n_total_missing"].sum()
        if train_missing > 0:
            issues.append(f"  - {train_missing} missing values in training data")
            all_clean = False
        if test_missing > 0:
            issues.append(f"  - {test_missing} missing values in test data")
            all_clean = False

        # Cycle ordering
        if not vr["train_engines"]["cycles_ordered"]:
            issues.append("  - Cycles NOT monotonically increasing for all engines")
            all_clean = False
        if not vr["train_engines"]["cycles_start_at_1"]:
            anomalous = vr["train_engines"]["anomalous_start_engines"]
            issues.append(f"  - {len(anomalous)} engines do not start at cycle 1")
            all_clean = False

        if issues:
            lines.append(f"### {subset}\n")
            for issue in issues:
                lines.append(issue)
            lines.append("")
        else:
            lines.append(f"### {subset}\n")
            lines.append(f"- [OK] No missing values, no duplicates, cycles ordered, all start at 1\n")

    if all_clean:
        lines.append("\n> **Overall**: All subsets are clean — no missing values, no duplicates, correct cycle ordering.\n")

    # Constant/near-constant sensors
    lines.append("## 4. Sensor Findings\n")

    lines.append("### Constant and Near-Constant Sensors\n")
    for subset, vr in sorted(all_results.items()):
        sensors = vr["train_sensors"]
        constant = sensors[sensors["is_constant"]]["sensor"].tolist()
        near_const = sensors[sensors["is_near_constant"] & ~sensors["is_constant"]]["sensor"].tolist()

        lines.append(f"**{subset}**:")
        if constant:
            lines.append(f"- Constant sensors: {', '.join(constant)}")
        else:
            lines.append("- No constant sensors")
        if near_const:
            lines.append(f"- Near-constant sensors: {', '.join(near_const)}")
        else:
            lines.append("- No near-constant sensors")
        lines.append("")

    # Sensor-cycle correlations
    lines.append("### Sensor–Cycle Correlation (Training Data)\n")
    lines.append("| Sensor | " + " | ".join(sorted(all_results.keys())) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(all_results)) + "|")

    # Build correlation table
    corr_data = {}
    for subset, vr in sorted(all_results.items()):
        corr = vr["sensor_cycle_corr"].set_index("sensor")["pearson_r"]
        corr_data[subset] = corr

    for sensor in SENSOR_COLUMNS:
        row_vals = []
        for subset in sorted(all_results.keys()):
            val = corr_data[subset].get(sensor, float("nan"))
            if pd.isna(val):
                row_vals.append("N/A")
            else:
                row_vals.append(f"{val:+.4f}")
        lines.append(f"| {sensor} | " + " | ".join(row_vals) + " |")

    # Operating conditions
    lines.append("\n## 5. Operating Condition Findings\n")
    lines.append(
        "The C-MAPSS subsets differ primarily in the number of operating conditions:\n"
    )
    lines.append("- **FD001/FD003**: Single operating condition (Sea Level) — op_settings are nearly constant")
    lines.append("- **FD002/FD004**: Six operating conditions — op_settings show significant variation\n")
    lines.append("This makes FD002/FD004 substantially harder because sensor readings are confounded by operating regime changes.\n")

    for subset, vr in sorted(all_results.items()):
        oc = vr["train_op_conditions"]
        lines.append(f"### {subset} Operating Conditions\n")
        lines.append("| Setting | Unique | Mean | Std | Min | Max |")
        lines.append("|---------|--------|------|-----|-----|-----|")
        for _, row in oc.iterrows():
            lines.append(
                f"| {row['setting']} | {row['n_unique']} | {row['mean']:.4f} "
                f"| {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
            )
        lines.append("")

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Saved: {output_path}")


def main():
    """Run the complete Phase 1 analysis pipeline."""
    print("=" * 60)
    print("AeroRUL — Phase 1 Analysis Pipeline")
    print("=" * 60)

    # Ensure output directories exist
    PHASE1_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Discover datasets
    print("\n[1/6] Discovering datasets...")
    discovery = discover_datasets()
    for subset, avail in discovery.items():
        status = "[OK]" if all(avail.values()) else "[!!]"
        print(f"  {status} {subset}: train={avail['train']}, test={avail['test']}, rul={avail['rul']}")

    # Step 2: Load all data
    print("\n[2/6] Loading all datasets...")
    all_data = load_all_subsets()
    for subset, data in all_data.items():
        print(f"  {subset}: train={len(data['train'])} rows, test={len(data['test'])} rows")

    # Step 3: Run validation on all subsets
    print("\n[3/6] Running full validation...")
    all_results = {}
    for subset, data in all_data.items():
        print(f"  Validating {subset}...")
        all_results[subset] = run_full_validation(
            data["train"], data["test"], data.get("rul"), subset
        )

    # Step 4: Generate statistics CSVs
    print("\n[4/6] Generating statistics...")
    stats_df = generate_dataset_statistics_csv(
        all_results, PHASE1_REPORTS_DIR / "dataset_statistics.csv"
    )
    sensor_df = generate_sensor_statistics_csv(
        all_results, PHASE1_REPORTS_DIR / "sensor_statistics.csv"
    )

    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")
    saved_figs = generate_all_phase1_plots(all_data, all_results, FIGURES_DIR)
    for fig_path in saved_figs:
        print(f"  Saved: {fig_path.name}")

    # Step 6: Generate summary report
    print("\n[6/6] Generating dataset summary report...")
    generate_dataset_summary_md(
        discovery, all_results, stats_df, sensor_df,
        PHASE1_REPORTS_DIR / "dataset_summary.md"
    )

    print("\n" + "=" * 60)
    print("Phase 1 analysis complete!")
    print(f"Reports: {PHASE1_REPORTS_DIR}")
    print(f"Figures: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
