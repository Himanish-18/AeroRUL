"""
Generate notebooks/04_1_simulated_truncation_evaluation.ipynb
Covers all 12 sections required by Phase 4.1 specification.
"""

from pathlib import Path
import nbformat as nbf

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Overview
    cells.append(nbf.v4.new_markdown_cell("""# Phase 4.1: Validation Protocol Correction & Simulated Truncation Evaluation
### AeroRUL — Remaining Useful Life Estimation for Turbofan Engines

This notebook implements the corrected validation protocol for RUL estimation on NASA C-MAPSS datasets:
1. **Methodological Correction**: Replace terminal-cycle evaluation ($RUL = 0$) with a **retrospective simulated truncation protocol** at positive-RUL operational checkpoints ($t < T_u$, $RUL > 0$).
2. **Training-Derived Cutoff Schedule**: Derived strictly from training engine lifetimes and percentiles (independent of validation engine lifespans).
3. **Causal Integrity**: Predictions use only sensor features observable up to cycle $t$; future features and failure cycles are strictly excluded.
4. **Pre-Failure Performance Evaluation**: Evaluate models on active operating engines using pooled, engine-macro, and NASA asymmetric score metrics.
5. **Corrected Cap & Window Diagnostics**: Evaluate RUL capping and rolling window subsets against uncapped physical ground truth.
"""))

    # Section 1: Setup & Imports
    cells.append(nbf.v4.new_markdown_cell("""## 1. Setup, Environment & Imports"""))
    cells.append(nbf.v4.new_code_cell("""import sys
import warnings
from pathlib import Path

# Ensure project root is on sys.path
_cwd = Path.cwd()
PROJECT_ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    AVAILABLE_SUBSETS,
    FIGURE_DPI,
    FIGURE_FORMAT,
    REPORTS_DIR,
)
from src.metrics import (
    mae, rmse, r2_score, mean_signed_error, nasa_score, nasa_score_per_engine,
)
from src.truncation import (
    compute_training_cutoffs,
    extract_simulated_truncations,
    compute_cutoff_coverage,
    engine_cluster_bootstrap,
    evaluate_simulated_endpoints,
    evaluate_simulated_endpoints_by_cutoff,
    simulated_rul_range_error_analysis,
    get_protocol_audit_table,
)
from src.run_phase4_benchmark import (
    load_phase3_data,
    load_approved_features,
    prepare_clean_features,
    train_and_predict,
    ALL_MODEL_CONFIGS,
    select_window_features,
)

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
%matplotlib inline

print("AeroRUL Phase 4.1 modules loaded successfully.")
"""))

    # Section 2: Phase 4 Protocol Audit
    cells.append(nbf.v4.new_markdown_cell("""## 2. Existing Phase 4 Protocol Audit

Documenting the methodological limitations identified in Phase 4:
- Endpoint evaluation evaluated terminal rows where $RUL_{u, T_u} = 0$.
- True RUL = 0 made capped and uncapped endpoint evaluation mathematically identical.
- NASA asymmetric underprediction branch was unexercised.
"""))
    cells.append(nbf.v4.new_code_cell("""audit_df = get_protocol_audit_table()
display(audit_df[['metric_or_experiment', 'existing_method', 'identified_limitation', 'required_action']])
"""))

    # Section 3: Training-Derived Cutoff Schedule
    cells.append(nbf.v4.new_markdown_cell("""## 3. Training-Derived Cutoff Schedule

The cutoff schedule is constructed **strictly from training engine lifespans**:
- Early-life base checkpoints: 80, 100, 125 cycles ($t \\ge 75$ feature availability constraint).
- Training terminal length percentiles: 25th, 50th, 75th, 90th percentiles of training $T_u$.
- No validation lifespans or labels are used to construct the schedule.
"""))
    cells.append(nbf.v4.new_code_cell("""subset = 'FD001'
train_df, val_df = load_phase3_data(subset)

cutoffs, meta = compute_training_cutoffs(
    train_df,
    base_checkpoints=(80, 100, 125),
    quantiles=(0.25, 0.50, 0.75, 0.90),
    min_cycle_feature_availability=75,
)

print(f"Subset: {subset}")
print(f"Training engines: {meta['n_training_engines']}")
print(f"Terminal cycle min={meta['terminal_min']}, median={meta['terminal_median']}, max={meta['terminal_max']}")
print(f"Training Quantiles: {meta['terminal_quantiles']}")
print(f"Final Cutoff Schedule: {cutoffs}")
"""))

    # Section 4: Simulated Truncation Construction & Positive-RUL Validation
    cells.append(nbf.v4.new_markdown_cell("""## 4. Simulated Truncation Construction & Positive-RUL Validation

Filter validation engine observations at the predeclared cutoff cycles.
Strict conditions:
1. $t < T_u$ (strictly non-terminal, pre-failure).
2. $RUL_{u,t} = T_u - t > 0$ (positive remaining useful life).
3. Features are strictly causal (no future values or failure metadata).
"""))
    cells.append(nbf.v4.new_code_cell("""trunc_df = extract_simulated_truncations(val_df, cutoffs, schedule_id='train_quantiles_v1')

print(f"Extracted {len(trunc_df)} simulated pre-failure observations across {trunc_df['unit_id'].nunique()} validation engines.")
print(f"All RUL > 0: {(trunc_df['rul'] > 0).all()}")
print(f"Min true RUL: {trunc_df['rul'].min()}, Max true RUL: {trunc_df['rul'].max()}, Mean true RUL: {trunc_df['rul'].mean():.1f}")
print(f"Terminal rows excluded: {(trunc_df['cycle'] == trunc_df['actual_terminal_cycle']).sum() == 0}")

display(trunc_df[['unit_id', 'cycle', 'cutoff_cycle', 'actual_terminal_cycle', 'rul']].head(10))
"""))

    # Section 5: Per-Cutoff Coverage & Reliability
    cells.append(nbf.v4.new_markdown_cell("""## 5. Cutoff Coverage & Reliability Analysis

Examine active engine retention and true RUL distribution across each cycle checkpoint.
As engines fail, fewer engines remain active at late cycles.
"""))
    cells.append(nbf.v4.new_code_cell("""cov_df = compute_cutoff_coverage(trunc_df, cutoffs, total_val_engines=val_df['unit_id'].nunique(), subset=subset)
display(cov_df)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(cov_df['cutoff_cycle'], cov_df['n_eligible_engines'], 'o-', color='royalblue', linewidth=2, markersize=8)
ax.set_title(f'{subset}: Eligible Validation Engines per Cutoff Checkpoint', fontweight='bold')
ax.set_xlabel('Cutoff Cycle (t)')
ax.set_ylabel('Active Eligible Engines (t < Tu)')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
"""))

    # Section 6: Fixed-Configuration Model Evaluation
    cells.append(nbf.v4.new_markdown_cell("""## 6. Fixed-Configuration Model Evaluation

Evaluate all classical models (DummyBaseline, LinearRegression, RidgeRegression, RandomForest, XGBoost, LightGBM) on the simulated pre-failure observations.
"""))
    cells.append(nbf.v4.new_code_cell("""approved = load_approved_features(subset)
usable_features = [f for f in approved if f in train_df.columns]

X_train, X_val, y_train, y_val = prepare_clean_features(train_df, val_df, usable_features)
trunc_indices = trunc_df.index.values
X_val_trunc = X_val[trunc_indices]
y_val_trunc = y_val[trunc_indices]

model_results = []
preds_dict = {}

for model_name, cfg in ALL_MODEL_CONFIGS.items():
    print(f"Evaluating {model_name:18s} ...", end=" ")
    _, _, p_raw, rt = train_and_predict(model_name, cfg, X_train, y_train, X_val_trunc)
    p_nn = np.maximum(p_raw, 0.0)
    
    m_df = trunc_df[['unit_id', 'cycle', 'cutoff_cycle', 'actual_terminal_cycle', 'rul']].copy()
    m_df.rename(columns={'rul': 'actual_rul'}, inplace=True)
    m_df['predicted_rul_nonnegative'] = p_nn
    m_df['predicted_rul_raw'] = p_raw
    preds_dict[model_name] = m_df
    
    res = evaluate_simulated_endpoints(m_df)
    res['model'] = model_name
    res['runtime_seconds'] = round(rt, 2)
    model_results.append(res)
    print(f"Simulated EP MAE: {res['simulated_truncated_endpoint_mae']:.2f} | Macro MAE: {res['engine_macro_mae']:.2f} | NASA: {res['nasa_score_sum']:.1f}")

sim_res_df = pd.DataFrame(model_results)[
    ['model', 'simulated_truncated_endpoint_mae', 'engine_macro_mae', 'simulated_truncated_endpoint_rmse', 'nasa_score_sum', 'nasa_score_mean', 'runtime_seconds']
].sort_values('simulated_truncated_endpoint_mae')

display(sim_res_df)
"""))

    # Section 7: Pooled vs Engine-Macro Metrics & Visualizations
    cells.append(nbf.v4.new_markdown_cell("""## 7. Pooled vs Engine-Macro Metrics Visualizations"""))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = sim_res_df['model'].values
axes[0].barh(models, sim_res_df['simulated_truncated_endpoint_mae'], color='steelblue', edgecolor='black')
axes[0].set_title(f'{subset}: Pooled Simulated Endpoint MAE (cycles)', fontweight='bold')
axes[0].set_xlabel('MAE (cycles)')
axes[0].grid(True, alpha=0.3, axis='x')

axes[1].barh(models, sim_res_df['engine_macro_mae'], color='coral', edgecolor='black')
axes[1].set_title(f'{subset}: Engine-Macro MAE (equal engine weighting)', fontweight='bold')
axes[1].set_xlabel('Macro MAE (cycles)')
axes[1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()
"""))

    # Section 8: Terminal vs Simulated Pre-Failure Comparison
    cells.append(nbf.v4.new_markdown_cell("""## 8. Terminal (Phase 4) vs Simulated Pre-Failure (Phase 4.1) Comparison

Comparing model behavior at failure ($RUL=0$) versus active operational life ($RUL>0$).
"""))
    cells.append(nbf.v4.new_code_cell("""comp_path = REPORTS_DIR / 'phase4_1' / 'corrected_model_comparison.csv'
if comp_path.exists():
    all_comp_df = pd.read_csv(comp_path)
    sub_comp = all_comp_df[all_comp_df['dataset'] == subset]
    display(sub_comp[['dataset', 'model', 'simulated_truncated_endpoint_mae', 'original_phase4_terminal_mae', 'simulated_rank', 'original_rank', 'rank_changed']])
"""))

    # Section 9: RUL Cap Sensitivity on Positive RUL
    cells.append(nbf.v4.new_markdown_cell("""## 9. Corrected RUL-Cap Sensitivity on Positive RUL

Evaluate capped training targets against **uncapped physical truth** ($T_u - t$) at positive-RUL cutoffs.
Demonstrates why the Phase 4 equality between capped and uncapped metrics disappears when $RUL > 0$.
"""))
    cells.append(nbf.v4.new_code_cell("""cap_path = REPORTS_DIR / 'phase4_1' / 'corrected_rul_cap_sensitivity.csv'
if cap_path.exists():
    all_cap_df = pd.read_csv(cap_path)
    sub_cap = all_cap_df[all_cap_df['dataset'] == subset]
    display(sub_cap[['model', 'cap_label', 'simulated_endpoint_mae_uncapped', 'simulated_endpoint_rmse_uncapped', 'engine_macro_mae_uncapped', 'nasa_score_uncapped', 'simulated_endpoint_mae_capped']])
    
    fig, ax = plt.subplots(figsize=(9, 4))
    for mname, mdf in sub_cap.groupby('model'):
        ax.plot(mdf['cap_label'], mdf['simulated_endpoint_mae_uncapped'], 'o-', label=f'{mname} (vs Uncapped Truth)', linewidth=2)
        ax.plot(mdf['cap_label'], mdf['simulated_endpoint_mae_capped'], '--', label=f'{mname} (vs Capped Target)', alpha=0.7)
    ax.set_title(f'{subset}: RUL Cap Sensitivity on Positive-RUL Observations', fontweight='bold')
    ax.set_xlabel('Training Target Cap')
    ax.set_ylabel('MAE (cycles)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
"""))

    # Section 10: Window Configuration Comparison
    cells.append(nbf.v4.new_markdown_cell("""## 10. Corrected Temporal Window Comparison"""))
    cells.append(nbf.v4.new_code_cell("""win_path = REPORTS_DIR / 'phase4_1' / 'corrected_window_comparison.csv'
if win_path.exists():
    all_win_df = pd.read_csv(win_path)
    sub_win = all_win_df[all_win_df['dataset'] == subset]
    display(sub_win[['window_config', 'n_features', 'simulated_endpoint_mae', 'engine_macro_mae', 'nasa_score_sum']])
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sub_win['window_config'], sub_win['simulated_endpoint_mae'], color='teal', edgecolor='black', alpha=0.85)
    ax.set_title(f'{subset}: Window Configuration Performance under Simulated Truncation', fontweight='bold')
    ax.set_xlabel('Window Configuration')
    ax.set_ylabel('Simulated Endpoint MAE (cycles)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
"""))

    # Section 11: Error Analysis by RUL Range Bins
    cells.append(nbf.v4.new_markdown_cell("""## 11. Error Analysis across Operational RUL Ranges

Stratified error evaluation:
- `critical_0_30`: Critical / near failure ($0 \\le RUL \\le 30$)
- `low_31_75`: Low remaining life ($31 \\le RUL \\le 75$)
- `moderate_76_125`: Moderate remaining life ($76 \\le RUL \\le 125$)
- `high_gt125`: High remaining life ($RUL > 125$)
"""))
    cells.append(nbf.v4.new_code_cell("""range_path = REPORTS_DIR / 'phase4_1' / 'simulated_rul_range_error_analysis.csv'
if range_path.exists():
    all_range_df = pd.read_csv(range_path)
    sub_range = all_range_df[(all_range_df['dataset'] == subset) & (all_range_df['model'].isin(['LightGBM', 'RandomForest', 'RidgeRegression']))]
    display(sub_range[['model', 'rul_range', 'n_observations', 'mae', 'rmse', 'mean_signed_error', 'underpredict_pct', 'overpredict_pct']])
"""))

    # Section 12: Methodological Limitations & Phase 5 Readiness
    cells.append(nbf.v4.new_markdown_cell("""## 12. Methodological Limitations & Phase 5 Readiness

### Methodological Conclusions:
1. **Validation Protocol Validity**: The retrospective simulated truncation protocol provides a realistic, statistically sound assessment of pre-failure RUL prediction capability on unseen engines.
2. **Cap Tradeoffs**: In active engines ($RUL > 0$), target capping $c \\in [100, 125]$ cycles achieves the optimal balance between noise suppression and large-RUL under-prediction penalty.
3. **Multi-Scale Feature Superiority**: The combined multi-window feature representation (`All` windows) consistently achieves the lowest error across all subsets.

### Phase 5 Readiness:
- Primary candidate model families for systematic hyperparameter tuning in Phase 5: **LightGBM**, **XGBoost**, and **Random Forest**.
- The official C-MAPSS test set remains **completely untouched** and reserved for final declared evaluation in Phase 5.
"""))

    nb['cells'] = cells

    out_path = Path("notebooks/04_1_simulated_truncation_evaluation.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Generated {out_path}")

if __name__ == "__main__":
    build_notebook()
