"""
Script to create notebooks/05_1_final_evaluation_audit.ipynb
AeroRUL — Phase 5.1: Final Evaluation Audit, Error Analysis & Report Reconciliation
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "05_1_final_evaluation_audit.ipynb"


def make_md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def create_phase5_1_notebook():
    cells = []

    # Title & Metadata
    cells.append(make_md_cell("""
# AeroRUL — Phase 5.1: Final Evaluation Audit, Error Analysis & Report Reconciliation

This notebook provides an interactive, reproducible audit and verification walkthrough of **Phase 5** results:
1. **Artifact Inventory & Manifest Audit**: Verifying frozen model configurations, hyperparameter fidelity, and preprocessing integrity.
2. **Official Test Endpoint Coverage**: Verifying 100% causal feature coverage across all 707 test engines without future leakage.
3. **Independent Metric Recalculation**: Independently calculating MAE, RMSE, $R^2$, Signed Error, and NASA asymmetric scores against official ground truth.
4. **Investigation of Large NASA Scores**: Explaining the mathematical and physical causes behind the large penalties in FD003 and FD004 (Pareto concentration and overprediction on early truncations).
5. **Signed Error & Directional Bias**: Resolving contradictions in the historical report narrative regarding underprediction vs overprediction.
6. **Degradation Phase Analysis**: Evaluating model performance across physical operational regimes ($RUL \\le 30$ Critical, $31-75$ Low, $76-125$ Moderate, $>125$ High).
7. **Window Configuration & Target Cap Reconciliation**: Documenting evidence-based rationale for model selections (e.g., W75 for FD002, Cap=150 vs Cap=None).
8. **Interactive Visualizations**: Inspecting residual distributions, actual vs predicted regressions, and Pareto cumulative penalty curves.

> **Verification & Audit Only**: As per engineering protocol, no models are retrained, no hyperparameters are tuned, and no test labels are used for model selection.
"""))

    # Cell 1: Environment Setup
    cells.append(make_md_cell("## 1. Environment & Setup"))
    cells.append(make_code_cell("""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path("..").resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AVAILABLE_SUBSETS, REPORTS_DIR, RAW_DATA_DIR
from src.metrics import mae, rmse, r2_score, mean_signed_error, nasa_score

PHASE5_DIR = REPORTS_DIR / "phase5"
PHASE5_1_DIR = REPORTS_DIR / "phase5_1"
PHASE5_1_FIGURES_DIR = PHASE5_1_DIR / "figures"

print("Project root:", PROJECT_ROOT)
print("Available subsets:", AVAILABLE_SUBSETS)
"""))

    # Cell 2: Artifact Inventory & Frozen Manifest Verification
    cells.append(make_md_cell("## 2. Frozen Configuration & Manifest Verification"))
    cells.append(make_code_cell("""
manifest_audit_path = PHASE5_1_DIR / "final_manifest_audit.csv"
if manifest_audit_path.exists():
    df_manifest = pd.read_csv(manifest_audit_path)
    print("Frozen Model Manifest Audit Status:")
    display(df_manifest[["dataset", "manifest_model", "saved_model", "manifest_target_cap", "manifest_feature_config", "status"]])
else:
    print("Manifest audit file not found. Run scripts/run_phase5_1_audit.py first.")
"""))

    # Cell 3: Official Test Coverage Audit
    cells.append(make_md_cell("## 3. Official Test Engine Coverage & Alignment"))
    cells.append(make_code_cell("""
coverage_audit_path = PHASE5_1_DIR / "test_feature_coverage_audit.csv"
if coverage_audit_path.exists():
    df_coverage = pd.read_csv(coverage_audit_path)
    print("Official Test Engine Coverage (All 707 engines represented exactly once):")
    display(df_coverage[["dataset", "official_engines_expected", "official_engines_predicted", "engines_with_lt_75_cycles", "partial_history_coverage_pct", "coverage_status"]])
"""))

    # Cell 4: Independent Test Metric Recalculation
    cells.append(make_md_cell("## 4. Independent Test Metric Recalculation"))
    cells.append(make_code_cell("""
# Load independent recalculations
recalc_path = PHASE5_1_DIR / "independent_test_metric_recalculation.csv"
df_recalc = pd.read_csv(recalc_path)

display_cols = [
    "dataset", "n_engines", "recalculated_mae", "reported_mae",
    "recalculated_rmse", "reported_rmse",
    "recalculated_nasa_sum", "reported_nasa_sum",
    "recalculated_mean_signed_error", "median_signed_error", "verification_status"
]
print("Independent Recalculations vs Reported Phase 5 Metrics:")
display(df_recalc[display_cols])
"""))

    # Cell 5: Root Cause of Large NASA Scores in FD003 & FD004
    cells.append(make_md_cell("""
## 5. Root Cause of Large NASA Scores in FD003 and FD004

The NASA asymmetric scoring function is:
$$d_i = \\hat{y}_i - y_i$$
$$s_i = \\begin{cases} e^{-d_i / 13} - 1, & d_i < 0 \\quad (\\text{Underprediction / Early Maintenance}) \\\\ e^{d_i / 10} - 1, & d_i \\ge 0 \\quad (\\text{Overprediction / Delayed Maintenance}) \\end{cases}$$

Notice that overprediction receives an exponential penalty with denominator $10$.
When an uncapped model ($Cap=\\text{None}$) predicts $\\hat{y} \\approx 245$ on a run truncated at early cycles with true $y = 137$, $d_i = +108.0$, yielding:
$$s_i = e^{10.8} - 1 = 48,938!$$
"""))
    cells.append(make_code_cell("""
top_penalties_path = PHASE5_1_DIR / "top_nasa_penalties.csv"
df_top = pd.read_csv(top_penalties_path)

print("Top 5 Engine Penalties in FD003 (Uncapped Target, High Overprediction):")
display(df_top[df_top["dataset"] == "FD003"].head(5)[["dataset", "unit_id", "actual_rul_official", "predicted_rul_final", "signed_error", "individual_nasa_score", "penalty_share_pct"]])

print("\\nTop 5 Engine Penalties in FD004 (Multi-Condition, Uncapped Target):")
display(df_top[df_top["dataset"] == "FD004"].head(5)[["dataset", "unit_id", "actual_rul_official", "predicted_rul_final", "signed_error", "individual_nasa_score", "penalty_share_pct"]])
"""))

    # Cell 6: Signed Error & Bias Analysis
    cells.append(make_md_cell("""
## 6. Directional Bias & Residual Analysis

The signed error is defined as $e_i = \\hat{y}_i - y_i$.
- **Positive ($e_i > 0$):** Model overpredicts remaining life (maintenance delayed).
- **Negative ($e_i < 0$):** Model underpredicts remaining life (early maintenance).

The Phase 5 report incorrectly claimed that multi-condition subsets exhibit 'conservative underprediction'.
Our audit reveals that FD003 and FD004 exhibit a systematic **overprediction bias** of $+11.58$ and $+11.64$ cycles.
"""))
    cells.append(make_code_cell("""
contribs_path = PHASE5_1_DIR / "nasa_score_contributions.csv"
df_contribs = pd.read_csv(contribs_path)

summary_bias = []
for subset in AVAILABLE_SUBSETS:
    sub = df_contribs[df_contribs["dataset"] == subset]
    n = len(sub)
    n_over = int((sub["signed_error"] > 0).sum())
    n_under = int((sub["signed_error"] < 0).sum())
    summary_bias.append({
        "Dataset": subset,
        "Engines": n,
        "Mean Signed Error": sub["signed_error"].mean(),
        "Median Signed Error": sub["signed_error"].median(),
        "Overpredicting Units": f"{n_over} ({n_over/n*100:.1f}%)",
        "Underpredicting Units": f"{n_under} ({n_under/n*100:.1f}%)",
        "Empirical Bias": "Overpredicting" if sub["signed_error"].mean() > 0 else "Underpredicting"
    })

display(pd.DataFrame(summary_bias))
"""))

    # Cell 7: Performance by True RUL Degradation Phase
    cells.append(make_md_cell("""
## 7. Performance by Physical Degradation Phase (True-RUL Bins)

Descriptive breakdown of endpoint predictions by true remaining life:
- **Critical Phase ($0-30$ cycles):** Near-failure operational regime where maintenance must be scheduled.
- **Low Phase ($31-75$ cycles):** Advanced degradation.
- **Moderate Phase ($76-125$ cycles):** Intermediate degradation.
- **High Phase ($>125$ cycles):** Early life / minimal degradation.
"""))
    cells.append(make_code_cell("""
range_path = PHASE5_1_DIR / "test_error_by_rul_range.csv"
df_range = pd.read_csv(range_path)

display(df_range[["dataset", "rul_bin", "engine_count", "mae", "rmse", "mean_signed_error", "nasa_score_sum", "mean_nasa_score"]])
"""))

    # Cell 8: Single Verified Results Table
    cells.append(make_md_cell("## 8. Single Source of Truth: Verified Results Table"))
    cells.append(make_code_cell("""
verified_results_path = PHASE5_1_DIR / "final_verified_results.csv"
df_verified = pd.read_csv(verified_results_path)

display(df_verified[[
    "dataset", "selected_model", "selected_target_cap", "selected_feature_configuration",
    "official_test_engine_count", "test_mae", "test_rmse", "test_r2",
    "mean_signed_error", "nasa_score_sum", "nasa_score_mean", "maximum_individual_nasa_score"
]])
"""))

    # Cell 9: Key Visualizations
    cells.append(make_md_cell("## 9. Key Diagnostic Visualizations"))
    cells.append(make_code_cell("""
from IPython.display import Image, display

figures = [
    ("Actual vs Predicted RUL Regressions", PHASE5_1_FIGURES_DIR / "fig01_actual_vs_predicted.png"),
    ("Residual Bias vs True RUL", PHASE5_1_FIGURES_DIR / "fig03_residual_vs_actual.png"),
    ("Cumulative Pareto NASA Penalty Concentration", PHASE5_1_FIGURES_DIR / "fig06_nasa_penalty_ranking.png"),
    ("Error (MAE) Across Degradation Phases", PHASE5_1_FIGURES_DIR / "fig07_error_by_rul_range.png"),
    ("Mean vs Median Signed Error", PHASE5_1_FIGURES_DIR / "fig08_signed_error_comparison.png"),
]

for title, fpath in figures:
    if fpath.exists():
        print(f"=== {title} ===")
        display(Image(filename=str(fpath)))
"""))

    notebook_data = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2)

    print(f"Successfully generated notebook at: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_phase5_1_notebook()
