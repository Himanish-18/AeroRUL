"""
Script to create notebooks/05_hyperparameter_tuning_and_final_evaluation.ipynb
AeroRUL — Phase 5: Hyperparameter Tuning & Final Model Selection
"""

import json
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH = NOTEBOOKS_DIR / "05_hyperparameter_tuning_and_final_evaluation.ipynb"


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


def create_phase5_notebook():
    cells = []

    # Title & Metadata
    cells.append(make_md_cell("""
# AeroRUL — Phase 5: Hyperparameter Tuning & Final Model Selection

This notebook provides an interactive, reproducible walkthrough of **Phase 5**:
1. **Critical Preflight**: Official test endpoint feature availability audit (addressing truncated test trajectories).
2. **Stage A — Inner Grouped Cross-Validation**: 3-fold engine-grouped CV optimizing Engine-Macro MAE.
3. **Hyperparameter Optimization**: Bayesian optimization of LightGBM and XGBoost via Optuna.
4. **Baseline & Classical Benchmarks**: Comparing against DummyBaseline, LinearRegression, Ridge, and RandomForest.
5. **RUL Target Cap Sensitivity**: Evaluating caps $\\in [\\text{None}, 50, 75, 100, 125, 150]$ against uncapped physical truth.
6. **Window Feature Configuration Ablation**: Evaluating single windows ($W=10..75$) vs multi-scale combined windows.
7. **Stage B — Development Validation**: Validating finalists on designated Phase 2 validation partition.
8. **Configuration Freeze**: Saving `final_model_manifest.json` before touching official test labels.
9. **Stage C — Official Test Evaluation**: Evaluating frozen models against official `RUL_FD*.txt` ground truth.
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

from src.config import AVAILABLE_SUBSETS, REPORTS_DIR, FIGURES_DIR
from src.metrics import mae, rmse, r2_score, mean_signed_error, nasa_score

PHASE5_DIR = REPORTS_DIR / "phase5"
PHASE5_FIGURES_DIR = PHASE5_DIR / "figures"

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
print(f"Project root: {PROJECT_ROOT}")
print(f"Phase 5 reports directory: {PHASE5_DIR}")
"""))

    # Cell 2: Test Feature Availability Preflight
    cells.append(make_md_cell("""
## 2. Critical Preflight: Official Test Feature Availability Audit

In C-MAPSS, test engine trajectories are truncated before failure. Many test engines have fewer than 75 observed cycles at their final recorded point (e.g. min cycle 19 in FD004, 21 in FD002, 31 in FD001). Under standard 75-cycle windowing, these engines would lack feature representation.
AeroRUL implements a **causal partial-history-aware feature mode** that represents 100% of official test engines without looking ahead.
"""))
    cells.append(make_code_cell("""
avail_df = pd.read_csv(PHASE5_DIR / "test_endpoint_feature_availability.csv")
print(f"Total test engine audit records: {len(avail_df)}")

# Summary by configuration and dataset
summary = avail_df.groupby(["dataset", "candidate_feature_configuration"])["endpoint_feature_available"].value_counts().unstack().fillna(0)
print("\\nEndpoint Availability Summary (True vs False):")
display(summary)

# Feature Configuration Inventory
inv_df = pd.read_csv(PHASE5_DIR / "feature_configuration_inventory.csv")
print("\\nCandidate Feature Configurations:")
display(inv_df[["dataset", "feature_configuration", "feature_count", "earliest_usable_cycle_full", "earliest_usable_cycle_partial"]])
"""))

    # Cell 3: Stage A - Inner Grouped Cross-Validation Benchmark
    cells.append(make_md_cell("""
## 3. Stage A: Inner Grouped Cross-Validation Benchmark & Optuna Tuning

The training engines were partitioned using 3-fold `GroupKFold` by engine ID. All preprocessing scalers and cutoff schedules were fitted exclusively on fold-training engines.
The primary optimization metric is **Engine-Macro MAE**.
"""))
    cells.append(make_code_cell("""
cv_df = pd.read_csv(PHASE5_DIR / "cross_validation_metrics.csv")
display(cv_df[["dataset", "model", "engine_macro_mae", "pooled_mae", "pooled_rmse", "nasa_score_mean"]])

# Display tuning curves
fig_path = PHASE5_FIGURES_DIR / "cv_model_comparison.png"
if fig_path.exists():
    from IPython.display import Image
    display(Image(filename=str(fig_path)))
"""))

    # Cell 4: Optuna Tuning Convergence
    cells.append(make_md_cell("## 4. Optuna Hyperparameter Search Progression"))
    cells.append(make_code_cell("""
opt_df = pd.read_csv(PHASE5_DIR / "hyperparameter_search_results.csv")
print(f"Total Optuna search trials: {len(opt_df)}")

# Display best trial per model family
best_trials = opt_df.sort_values("engine_macro_mae").groupby(["dataset", "model_family"]).first().reset_index()
display(best_trials[["dataset", "model_family", "trial_id", "engine_macro_mae", "pooled_mae", "pooled_rmse", "runtime_sec"]])

fig_path = PHASE5_FIGURES_DIR / "optuna_tuning_history.png"
if fig_path.exists():
    from IPython.display import Image
    display(Image(filename=str(fig_path)))
"""))

    # Cell 5: RUL Target Cap Sensitivity
    cells.append(make_md_cell("""
## 5. RUL Target Cap Sensitivity Analysis

Piecewise linear degradation modeling caps healthy initial RUL. Target caps $\\in [\\text{None}, 50, 75, 100, 125, 150]$ were evaluated against **UNCAPPED physical ground truth** to quantify generalization fidelity.
"""))
    cells.append(make_code_cell("""
cap_df = pd.read_csv(PHASE5_DIR / "target_cap_sensitivity.csv")
display(cap_df[["dataset", "target_cap", "cv_engine_macro_mae", "val_engine_macro_mae", "val_pooled_rmse", "val_nasa_score_sum"]])

fig_path = PHASE5_FIGURES_DIR / "target_cap_sensitivity.png"
if fig_path.exists():
    from IPython.display import Image
    display(Image(filename=str(fig_path)))
"""))

    # Cell 6: Window Ablation
    cells.append(make_md_cell("""
## 6. Window Feature Configuration Ablation

Comparing single-scale rolling representations against multi-scale combinations (`ALL_WINDOWS`).
"""))
    cells.append(make_code_cell("""
win_df = pd.read_csv(PHASE5_DIR / "window_ablation_metrics.csv")
display(win_df[["dataset", "window_configuration", "feature_count", "cv_engine_macro_mae", "val_engine_macro_mae", "val_pooled_rmse"]])

fig_path = PHASE5_FIGURES_DIR / "window_ablation_comparison.png"
if fig_path.exists():
    from IPython.display import Image
    display(Image(filename=str(fig_path)))
"""))

    # Cell 7: Configuration Freeze & Model Manifest
    cells.append(make_md_cell("""
## 7. Configuration Freeze & Final Model Manifest

Before loading official C-MAPSS test labels, the winning configuration for each dataset was frozen in `final_model_manifest.json`.
"""))
    cells.append(make_code_cell("""
manifest_file = PHASE5_DIR / "final_model_manifest.json"
with open(manifest_file, "r", encoding="utf-8") as f:
    manifest = json.load(f)

print(f"Manifest Version: {manifest['manifest_version']}")
print(f"Frozen Timestamp: {manifest['frozen_timestamp']}")
print(f"Python: {manifest['system_metadata']['python_version']}")
print(f"Platform: {manifest['system_metadata']['platform']}")

print("\\nSelected Configurations per Dataset:")
for subset, cfg in manifest["selected_configurations"].items():
    print(f"\\n--- {subset} ---")
    print(f"  Model Family: {cfg['model_family']}")
    print(f"  Target Cap: {cfg['target_cap']}")
    print(f"  Feature Config: {cfg['feature_configuration']} ({cfg['n_features']} features)")
    print(f"  Inner-CV Engine-Macro MAE: {cfg['inner_cv_scores']['engine_macro_mae']:.2f}")
    print(f"  Dev-Val Macro MAE: {cfg['development_validation_scores']['engine_macro_mae']:.2f}")
"""))

    # Cell 8: Official Test Set Evaluation
    cells.append(make_md_cell("""
## 8. Stage C: Official C-MAPSS Test Set Evaluation

Final frozen models were evaluated once on each test engine's last recorded cycle against official ground truth remaining life (`RUL_FD*.txt`).
"""))
    cells.append(make_code_cell("""
test_metrics_df = pd.read_csv(PHASE5_DIR / "final_test_metrics.csv")
test_preds_df = pd.read_csv(PHASE5_DIR / "final_test_predictions.csv")

print("Official C-MAPSS Test Metrics:")
display(test_metrics_df)

print(f"\\nTotal Test Predictions: {len(test_preds_df)}")
display(test_preds_df.head(10))

# Display diagnostic figures
for fig_name in [
    "final_test_predicted_vs_actual.png",
    "final_test_residuals.png",
    "final_test_error_by_rul_range.png",
]:
    fig_path = PHASE5_FIGURES_DIR / fig_name
    if fig_path.exists():
        from IPython.display import Image
        display(Image(filename=str(fig_path)))
"""))

    # Cell 9: Metric Integrity Verification
    cells.append(make_md_cell("""
## 9. Metric Integrity & Verification Checks

Independently recomputing all official test metrics directly from saved prediction rows to guarantee 100% mathematical integrity.
"""))
    cells.append(make_code_cell("""
for _, row in test_metrics_df.iterrows():
    subset = row["dataset"]
    sub_p = test_preds_df[test_preds_df["dataset"] == subset]
    y_true = sub_p["actual_rul_official"].values
    y_pred = sub_p["predicted_rul_final"].values

    calc_mae = mae(y_true, y_pred)
    calc_rmse = rmse(y_true, y_pred)
    calc_nasa_sum = nasa_score(y_true, y_pred)

    assert np.isclose(calc_mae, row["endpoint_mae"], atol=1e-2), f"{subset} MAE mismatch!"
    assert np.isclose(calc_rmse, row["endpoint_rmse"], atol=1e-2), f"{subset} RMSE mismatch!"
    assert np.isclose(calc_nasa_sum, row["nasa_score_sum"], atol=1.0), f"{subset} NASA mismatch!"
    print(f"✓ {subset} verified: MAE={calc_mae:.2f}, RMSE={calc_rmse:.2f}, NASA Sum={calc_nasa_sum:.1f}")

print("\\nAll official test results independently verified and reconciled!")
"""))

    notebook_dict = {
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
                "version": "3.11.2",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=2)
    print(f"Created notebook: {NOTEBOOK_PATH}")


if __name__ == "__main__":
    create_phase5_notebook()
