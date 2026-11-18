# AeroRUL — Predictive Maintenance & RUL Estimation for Turbofan Engines

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NASA C-MAPSS](https://img.shields.io/badge/dataset-NASA%20C--MAPSS-orange.svg)](https://data.nasa.gov/)
[![Tests Passing](https://img.shields.io/badge/tests-271%20passed-brightgreen.svg)](tests/)
[![Architecture](https://img.shields.io/badge/architecture-leakage--free%20ML-purple.svg)](src/)

AeroRUL is an end-to-end predictive maintenance ML engineering and research framework that estimates the **Remaining Useful Life (RUL)** of commercial aircraft turbofan engines from multivariate sensor degradation time-series using the **NASA C-MAPSS** benchmark dataset.

The system incorporates causal sliding-window feature engineering, leakage-safe engine-level cross-validation, simulated truncation evaluation, Bayesian hyperparameter optimization (Optuna), and production model freezing across all four C-MAPSS sub-benchmarks (FD001–FD004).

---

## Benchmark Results (Final Frozen Models)

Authoritative performance of the frozen production models ([`models/final/`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/models/final)) evaluated on all **707 official test engines** across the four C-MAPSS subsets, independently audited and verified with 100% prediction coverage:

| Subset | Operating Conditions | Fault Modes | Selected Model | Target Cap | Feature Config | Feature Count | Test Engines | Test MAE (cycles) | Test RMSE (cycles) | Test $R^2$ | NASA Score Sum |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FD001** | 1 (Sea Level) | 1 (HPC) | LightGBM | 150 | ALL_WINDOWS | 670 | 100 / 100 | **11.43** | **15.69** | **0.8575** | **589.87** |
| **FD002** | 6 Flight Regimes | 1 (HPC) | LightGBM | 150 | W75 | 231 | 259 / 259 | **13.78** | **20.13** | **0.8599** | **2,306.70** |
| **FD003** | 1 (Sea Level) | 2 (HPC, Fan) | LightGBM | None | ALL_WINDOWS | 681 | 100 / 100 | **14.68** | **25.00** | **0.6352** | **66,975.40** |
| **FD004** | 6 Flight Regimes | 2 (HPC, Fan) | LightGBM | None | ALL_WINDOWS | 768 | 248 / 248 | **22.81** | **33.80** | **0.6157** | **1,139,046.46** |

*All results reflect independent post-evaluation audit verification ([`reports/phase5/final_test_metrics.csv`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase5/final_test_metrics.csv)). Zero data leakage occurred between training, validation, and test splits.*

---

## NASA C-MAPSS Dataset Overview

The **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset represents simulated degradation of 90,000-lb-thrust turbofan engines operated from sea level to cruise conditions until system failure.

Reference: Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08.

### C-MAPSS Subsets

| Subset | Flight Regimes | Fault Modes | Train Units | Test Units | Total Train Cycles | Total Test Cycles | Complexity Level |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **FD001** | 1 (Sea Level) | 1 (HPC Degradation) | 100 | 100 | 20,631 | 13,096 | Baseline single-condition |
| **FD002** | 6 Regimes | 1 (HPC Degradation) | 260 | 259 | 53,759 | 33,991 | High operating condition confounding |
| **FD003** | 1 (Sea Level) | 2 (HPC + Fan Degradation) | 100 | 100 | 24,720 | 16,596 | Dual degradation mechanisms |
| **FD004** | 6 Regimes | 2 (HPC + Fan Degradation) | 249 | 248 | 61,249 | 41,214 | Maximum complexity (multi-regime + dual-fault) |

### Data Format & Signals

Each run-to-failure file contains 26 space-separated numerical columns:
- `unit_id`: Engine identifier.
- `cycle`: Operating time in flight cycles ($t \ge 1$).
- `op_setting_1..3`: Operational altitude ($0\text{--}42{,}000\text{ ft}$), Mach number ($0\text{--}0.84$), and Throttle Resolver Angle ($20\text{--}100^\circ$).
- `sensor_1..21`: 21 physical sensor measurements across temperatures, pressures, spool speeds, and coolant bleed flows.

---

## Research & Engineering Progression

The project was executed across five sequential, scientifically audited research phases:

```mermaid
flowchart LR
    P1["Phase 1: Dataset Understanding"] --> P2["Phase 2: RUL Formulation & Splitting"]
    P2 --> P3["Phase 3: Sliding-Window Features"]
    P3 --> P31["Phase 3.1: Feature Quality Control"]
    P31 --> P4["Phase 4: Classical Benchmark"]
    P4 --> P41["Phase 4.1: Simulated Truncation"]
    P41 --> P42["Phase 4.2: Metric Integrity Audit"]
    P42 --> P5["Phase 5: Tuning & Final Models"]
    P5 --> P51["Phase 5.1: Final Evaluation Audit"]
    P51 --> P52["Phase 5.2: Report Reconciliation"]
```

### Phase 1 — Dataset Understanding & Sensor Profiling
- **Audit Findings**: 160,359 training rows and 104,897 test rows verified with 0 missing values and 0 duplicate cycle entries.
- **Sensor Characterization**: Identified non-informative flatline sensors in single-regime subsets (sensors 1, 5, 10 [FD001], 16, 18, 19).
- **Operating Envelopes**: Identified the 6 distinct operational clusters in FD002/FD004 that mask physical wear unless properly normalized.
- **Reports & Artifacts**: [`reports/phase1/dataset_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase1/dataset_summary.md), [`notebooks/01_cmapss_dataset_understanding.ipynb`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/notebooks/01_cmapss_dataset_understanding.ipynb).

### Phase 2 — RUL Target Formulation & Leakage-Safe Splitting
- **Piecewise Linear RUL Target**: Modeled early-life healthy operation where degradation is non-detectable, capping targets at $RUL_{cap}$ cycles:
  $$\text{RUL}(t) = \min(\text{failure\_cycle} - t, RUL_{cap})$$
- **Engine-Level Grouped Splitting**: Established strict 80% train / 20% validation splits grouped strictly by `unit_id`, guaranteeing **0 engine overlap** between splits.
- **Condition-Aware Normalization**: Implemented $k$-means regime clustering on operational settings with per-regime z-score normalization to isolate mechanical degradation in FD002 and FD004.
- **Reports & Artifacts**: [`reports/phase2/phase2_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase2/phase2_summary.md), [`notebooks/02_rul_formulation_and_data_split.ipynb`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/notebooks/02_rul_formulation_and_data_split.ipynb).

### Phase 3 & 3.1 — Sliding-Window Feature Engineering & Quality Control
- **Causal Feature Extraction**: Computed temporal statistics strictly over past cycles $[t - w + 1, t]$ ($w \in \{10, 20, 30, 50, 75\}$ cycles), guaranteeing zero future lookahead.
- **9 Statistical Feature Families**: Mean, standard deviation, minimum, maximum, dynamic range, linear slope, $R^2$ fit quality, recent delta, and relative change.
- **Feature Quality Control (Phase 3.1)**: Screened for zero-variance features, collinear duplicates, infinite values, and distribution shift across splits.
- **Reports & Artifacts**: [`reports/phase3/phase3_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase3/phase3_summary.md), [`reports/phase3_1/phase3_1_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase3_1/phase3_1_summary.md), [`notebooks/03_sliding_window_feature_engineering.ipynb`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/notebooks/03_sliding_window_feature_engineering.ipynb).

### Phase 4, 4.1 & 4.2 — Classical Benchmark & Evaluation Integrity
- **Multi-Model Benchmark (Phase 4)**: Evaluated Ridge, Lasso, Random Forest, XGBoost, and LightGBM across all feature sets and cap thresholds. Gradient-boosted trees decisively outperformed linear and bagging models.
- **Simulated Truncation Protocol (Phase 4.1)**: Discovered that evaluating models only at run-to-failure endpoints on validation data produced optimistic bias relative to official test engines (which are truncated at arbitrary health stages). Developed a simulated cutoff schedule mirroring test truncation dynamics without test label leakage.
- **Metric Integrity & Scoring Audit (Phase 4.2)**: Standardized signed prediction errors ($d_i = \hat{y}_i - y_i$) and verified numerical implementation of the asymmetric NASA scoring function.
- **Reports & Artifacts**: [`reports/phase4/phase4_benchmark_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase4/phase4_benchmark_summary.md), [`reports/phase4_1/phase4_1_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase4_1/phase4_1_summary.md), [`reports/phase4_2/phase4_2_audit_summary.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase4_2/phase4_2_audit_summary.md).

### Phase 5, 5.1 & 5.2 — Bayesian Tuning, Final Models & Audit Reconciliation
- **Optuna Hyperparameter Tuning**: Ran 50 Bayesian optimization trials per subset over LightGBM hyperparameters (learning rate, tree depth, leaf count, feature fraction, min child samples, L1/L2 regularization) guided by 5-fold GroupKFold engine-level CV.
- **Target Cap Selection**: Verified that $Cap=150$ was optimal for single-fault subsets (FD001, FD002), while uncapped targets ($Cap=\text{None}$) were optimal for complex dual-fault subsets (FD003, FD004).
- **Official Test Evaluation & Independent Audit**: Evaluated final frozen models against official test ground truths. Conducted a 100% independent recalculation audit verifying $0.0000$ difference across all 20 reported test metrics.
- **Documentation Reconciliation (Phase 5.2)**: Resolved cross-experiment inner-CV versus development validation rank nuances, updated structural feature-count representations, and eliminated legacy draft text.
- **Reports & Artifacts**: [`reports/phase5/final_model_manifest.json`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase5/final_model_manifest.json), [`reports/phase5_1/phase5_1_final_audit_report.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase5_1/phase5_1_final_audit_report.md), [`reports/phase5_2/phase5_2_completion_report.md`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/reports/phase5_2/phase5_2_completion_report.md).

---

## Evaluation Metrics & NASA Scoring Function

Predictive maintenance requires asymmetric risk penalization. Overestimating remaining life ($d_i > 0$) is hazardous as it delays inspection before failure, whereas underestimating ($d_i < 0$) is merely conservative.

The framework computes standard error metrics alongside the official PHM08 asymmetric scoring function:

- **Signed Error**:
  $$d_i = \hat{y}_i - y_i \quad (\text{predicted} - \text{actual})$$
- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |d_i|$$
- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N d_i^2}$$
- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_{i=1}^N d_i^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
- **NASA Asymmetric Scoring Function**:
  $$S = \sum_{i=1}^N s_i, \quad s_i = \begin{cases} \exp\left(-\frac{d_i}{13}\right) - 1 & \text{if } d_i < 0 \text{ (underprediction / conservative)} \\ \exp\left(\frac{d_i}{10}\right) - 1 & \text{if } d_i \ge 0 \text{ (overprediction / hazardous)} \end{cases}$$

---

## Frozen Production Models

The final selected model bundles are permanently preserved in [`models/final/`](file:///c:/Users/himan/OneDrive/Documents/AeroRUL/models/final):

```
models/final/
├── FD001/
│   ├── model.joblib              # Frozen LightGBM regressor
│   ├── config_metadata.json      # Target cap (150), feature config (ALL_WINDOWS), params
│   └── feature_names.json        # 670 ordered feature column names
├── FD002/
│   ├── model.joblib              # Frozen LightGBM regressor
│   ├── config_metadata.json      # Target cap (150), feature config (W75), params
│   ├── feature_names.json        # 231 ordered feature column names
│   └── norm_params.json          # 6-regime normalization cluster centroids & scalers
├── FD003/
│   ├── model.joblib              # Frozen LightGBM regressor
│   ├── config_metadata.json      # Target cap (None), feature config (ALL_WINDOWS), params
│   └── feature_names.json        # 681 ordered feature column names
└── FD004/
    ├── model.joblib              # Frozen LightGBM regressor
    ├── config_metadata.json      # Target cap (None), feature config (ALL_WINDOWS), params
    ├── feature_names.json        # 768 ordered feature column names
    └── norm_params.json          # 6-regime normalization cluster centroids & scalers
```

---

## Interactive Evaluation Workbench (Frontend)

An interactive, production-grade research dashboard built with **React 18**, **TypeScript**, **Vite**, **Recharts**, and the **Hallmark Console** design system is available on the `frontend` branch.

### Workspaces
1. **Overview Dashboard**: Fleet-level RUL predictions, official test benchmark comparison, and actual vs. predicted parity scatter plots.
2. **Engine Predictions**: Unit-level endpoint error inspection, signed error distributions, and overprediction/underprediction risk categorization.
3. **Model Evaluation & Benchmarking**: Cross-validation vs. Dev-Val rank comparison, window ablation metrics, target cap sensitivity curves, and Optuna hyperparameter trial inspector.
4. **Feature Intelligence & Degradation Dynamics**: Native LightGBM split gain/count feature importance, sensor family breakdowns, multivariate tree gain vs. univariate linear correlation quadrants, and multi-engine degradation trajectories.

To run the frontend workspace:
```bash
git checkout frontend
cd frontend
npm install
npm run dev
```

---

## Repository Structure

```
AeroRUL/
├── CMAPSSData/                  # Official NASA C-MAPSS data (read-only)
│   ├── train_FD00*.txt
│   ├── test_FD00*.txt
│   └── RUL_FD00*.txt
├── models/
│   └── final/                   # Authoritative frozen production models
│       ├── FD001/
│       ├── FD002/
│       ├── FD003/
│       └── FD004/
├── src/
│   ├── __init__.py
│   ├── config.py                # Global paths, constants, sensor definitions
│   ├── data_loader.py           # Robust C-MAPSS data loader
│   ├── data_validation.py       # Data validation & profiling
│   ├── rul.py                   # RUL target calculation & clipping
│   ├── splitting.py             # Engine-level grouped splitting (zero leakage)
│   ├── normalization.py         # Multi-regime condition-aware normalization
│   ├── features.py              # Causal multi-scale sliding-window engineering
│   ├── feature_validation.py    # Feature sanity & leakage checks
│   ├── feature_quality_control.py # Variance filtering & collinearity deduplication
│   ├── truncation.py            # Simulated early-cutoff evaluation protocol
│   ├── metrics.py               # MAE, RMSE, R², MSE, NASA scoring function
│   ├── visualization.py         # Publication-grade plotting utilities
│   ├── run_phase1_analysis.py   # Phase 1 pipeline runner
│   ├── run_phase2_analysis.py   # Phase 2 pipeline runner
│   ├── run_phase3_analysis.py   # Phase 3 pipeline runner
│   ├── run_phase3_1_analysis.py # Phase 3.1 pipeline runner
│   ├── run_phase4_benchmark.py  # Phase 4 classical benchmark runner
│   ├── run_phase4_1_benchmark.py# Phase 4.1 simulated truncation runner
│   ├── run_phase4_2_audit.py    # Phase 4.2 metric audit runner
│   └── run_phase5_tuning.py     # Phase 5 Optuna tuning & model export runner
├── tests/                       # 271 automated unit, pipeline, and audit tests
│   ├── test_data_loader.py
│   ├── test_data_validation.py
│   ├── test_rul.py
│   ├── test_splitting.py
│   ├── test_normalization.py
│   ├── test_features.py
│   ├── test_feature_validation.py
│   ├── test_phase3_1_fqc.py
│   ├── test_phase4_benchmark.py
│   ├── test_phase4_1_truncation.py
│   ├── test_phase4_2_audit.py
│   ├── test_phase5_tuning.py
│   ├── test_phase5_1_audit.py
│   └── test_phase5_2_documentation.py
├── notebooks/                   # Interactive exploratory & audit notebooks
│   ├── 01_cmapss_dataset_understanding.ipynb
│   ├── 02_rul_formulation_and_data_split.ipynb
│   ├── 03_sliding_window_feature_engineering.ipynb
│   ├── 03_1_feature_quality_control.ipynb
│   ├── 04_classical_regression_benchmark.ipynb
│   ├── 04_1_simulated_truncation_evaluation.ipynb
│   ├── 04_2_metric_integrity_audit.ipynb
│   ├── 05_hyperparameter_tuning_and_final_evaluation.ipynb
│   └── 05_1_final_evaluation_audit.ipynb
├── reports/                     # Authoritative reports, audit CSVs, and figures
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   ├── phase3_1/
│   ├── phase4/
│   ├── phase4_1/
│   ├── phase4_2/
│   ├── phase5/
│   ├── phase5_1/
│   └── phase5_2/
├── scripts/                     # Audit, reconciliation, and notebook generators
│   ├── run_phase5_1_audit.py
│   └── validate_phase5_2_docs.py
├── requirements.txt             # Python environment dependencies
└── README.md                    # Project documentation
```

---

## Quickstart & Pipeline Execution

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/Himanish-18/AeroRUL.git
cd AeroRUL

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite

Verify complete test coverage across loaders, feature transformers, models, and audit checks:

```bash
pytest tests/ -v
# 271 passed
```

### 3. Execute Research Pipelines

Execute any phase independently using the modular pipeline entrypoints:

```bash
# Phase 1: Dataset profiling & sensor statistics
python -m src.run_phase1_analysis

# Phase 2: RUL formulation & condition normalization
python -m src.run_phase2_analysis

# Phase 3: Sliding-window feature extraction
python -m src.run_phase3_analysis

# Phase 3.1: Feature quality control & variance screening
python -m src.run_phase3_1_analysis

# Phase 4: Classical regression benchmarking
python -m src.run_phase4_benchmark

# Phase 4.1: Simulated truncation evaluation
python -m src.run_phase4_1_benchmark

# Phase 4.2: Metric audit & scoring reconciliation
python -m src.run_phase4_2_audit

# Phase 5: Optuna Bayesian tuning & model freezing
python -m src.run_phase5_tuning

# Phase 5.1 & 5.2: Evaluation audit & report reconciliation
python scripts/run_phase5_1_audit.py
python scripts/validate_phase5_2_docs.py
```

---

## Dependencies & Environment

- **Core Runtime**: Python 3.11+
- **Data & Numerical**: `pandas >= 2.0.0`, `numpy >= 1.24.0`, `scipy >= 1.10.0`, `pyarrow >= 14.0.0`
- **Machine Learning**: `scikit-learn >= 1.3.0`, `lightgbm >= 4.0.0`, `xgboost >= 1.7.0`, `optuna >= 3.0.0`, `joblib >= 1.3.0`
- **Visualization**: `matplotlib >= 3.7.0`, `seaborn >= 0.12.0`
- **Testing**: `pytest >= 7.0.0`

---

## References

1. **A. Saxena, K. Goebel, D. Simon, and N. Eklund**, *"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation"*, In Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008.
2. **A. Saxena and K. Goebel**, *"Turbofan Engine Degradation Simulation Data Set"*, NASA Ames Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA, 2008.
3. **E. Ramasso and A. Saxena**, *"Performance Benchmarking and Analysis of Prognostic Methods for CMAPSS Datasets"*, International Journal of Prognostics and Health Management, 2014.
4. **G. Ke et al.**, *"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"*, Advances in Neural Information Processing Systems (NeurIPS), 2017.
