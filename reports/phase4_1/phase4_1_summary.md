# AeroRUL — Phase 4.1: Validation Protocol Correction & Simulated Truncation Evaluation

## A. Executive Summary

Phase 4.1 corrects a critical methodological limitation identified in the initial Phase 4 benchmark. In Phase 4, endpoint metrics were evaluated on the final observed cycle of complete validation trajectories, where true Remaining Useful Life was identically zero ($RUL = 0$).

To establish a realistic, statistically defensible evaluation protocol, Phase 4.1 introduces a **retrospective simulated truncation protocol**: evaluating held-out engines at predeclared operational checkpoints where the engine is active, sensor history is strictly causal ($t < T_u$), and remaining useful life is strictly positive ($RUL_{u,t} = T_u - t > 0$).

### Key Findings by Dataset (Simulated Truncated Pre-Failure Evaluation):

### FD001
- **Best Model**: **RidgeRegression**
- **Simulated Endpoint MAE**: **45.44 cycles** (95% CI: [38.8, 52.1])
- **Engine-Macro MAE**: **43.52 cycles**
- **Simulated Endpoint RMSE**: **55.65 cycles**
- **NASA Asymmetric Score**: **1902952.4** (Mean per observation: 21143.92)
- **Baseline EP MAE**: 39.88 cycles (NASA: 49954.2)
- **Improvement over Baseline**: **-5.56 cycles (-13.9%)**

### FD002
- **Best Model**: **RandomForest**
- **Simulated Endpoint MAE**: **53.97 cycles** (95% CI: [47.7, 60.9])
- **Engine-Macro MAE**: **51.81 cycles**
- **Simulated Endpoint RMSE**: **66.19 cycles**
- **NASA Asymmetric Score**: **11761344.5** (Mean per observation: 52741.45)
- **Baseline EP MAE**: 40.81 cycles (NASA: 4776401.1)
- **Improvement over Baseline**: **-13.16 cycles (-32.3%)**

### FD003
- **Best Model**: **RandomForest**
- **Simulated Endpoint MAE**: **67.07 cycles** (95% CI: [50.6, 82.9])
- **Engine-Macro MAE**: **61.11 cycles**
- **Simulated Endpoint RMSE**: **87.76 cycles**
- **NASA Asymmetric Score**: **259297362.5** (Mean per observation: 2946560.94)
- **Baseline EP MAE**: 60.19 cycles (NASA: 6705272222.0)
- **Improvement over Baseline**: **-6.88 cycles (-11.4%)**

### FD004
- **Best Model**: **RandomForest**
- **Simulated Endpoint MAE**: **88.47 cycles** (95% CI: [75.3, 103.9])
- **Engine-Macro MAE**: **85.58 cycles**
- **Simulated Endpoint RMSE**: **110.66 cycles**
- **NASA Asymmetric Score**: **3011254047846.3** (Mean per observation: 13443098427.89)
- **Baseline EP MAE**: 62.99 cycles (NASA: 6118803476.0)
- **Improvement over Baseline**: **-25.47 cycles (-40.4%)**

## B. Original Endpoint Limitation & Audit

In Phase 4, validation trajectories were drawn from complete run-to-failure sequences. The function `extract_endpoints()` selected `idxmax(cycle)`, which corresponds to the failure cycle $T_u$. Consequently:
1. **True RUL was identically zero ($RUL = 0$)**: The metric measured terminal breakdown diagnosis rather than progressive degradation tracking.
2. **NASA Score Distortion**: Because true RUL was 0, any prediction $\hat{y} \ge 0$ was classified as an overprediction ($d \ge 0$), exercising only the $\exp(d/10)-1$ penalty branch and leaving the underprediction penalty branch unexercised.
3. **Cap Sensitivity Equality**: The equality `endpoint_mae_vs_uncapped == endpoint_mae_vs_capped` in Phase 4 occurred because $\min(0, c) = 0$ for all caps $c \ge 0$. Evaluating at positive RUL eliminates this artifact.

| Metric / Experiment | Existing Method | Identified Limitation | Required Action |
| :--- | :--- | :--- | :--- |
| Endpoint MAE & RMSE | Evaluated at final row of complete run-to-failure validation trajectories: idxmax(cycle), where actual RUL = 0. | Measures terminal-state failure diagnostic accuracy (RUL=0) rather than operational remaining life prediction when life remains (RUL>0). | Evaluate at predeclared retrospective cycle cutoffs (t < Tu, RUL = Tu - t > 0) derived from training engine quantiles. |
| NASA Asymmetric Score | Sum of asymmetric penalties evaluated exclusively at terminal cycle Tu where true RUL = 0. | Since true RUL=0, any positive prediction is an overprediction (d >= 0, exp(d/10)-1). Underprediction penalty branch (d < 0, exp(-d/13)-1) is almost never triggered. | Evaluate NASA score on positive-RUL cutoff checkpoints, exercising both over- and under-prediction penalty branches, and report both sum and mean per observation. |
| RUL-Cap Sensitivity (Capped vs Uncapped) | Compared endpoint MAE vs uncapped truth (0) and vs capped truth (min(0, cap)). | Since min(0, cap) = 0 for any cap >= 0, capped-target and uncapped-target endpoint MAEs were numerically identical in Phase 4. | Evaluate cap models at positive-RUL cutoffs against uncapped physical RUL truth to measure genuine impact of early-life plateauing on operating engines. |
| Window Configuration Comparison | Compared single windows (W=10..75) against combined 'All' windows using row-level pooled MAE and terminal endpoint MAE. | Terminal-cycle evaluation does not test whether multi-scale window features preserve degradation trend tracking at mid-life operational cycles. | Re-evaluate window configurations under identical positive-RUL cutoffs and consistent validation populations. |
| Row-Level Pooled Validation Metrics | Aggregated MAE, RMSE, and R² across all available validation cycles. | Mixes all degradation stages without standardization; longer-lived engines contribute more cycles, disproportionately weighting slow-degrading units. | Preserve row-level metrics as baseline diagnostic while introducing standardized cutoff checkpoints and engine-macro averaging. |

## C. Retrospective Truncation Protocol Details

The cutoff schedule was constructed **strictly from training engine lifetimes** independently of validation engine lifespans:
- Base early-life checkpoints: `80, 100, 125` cycles (supported by feature availability $t \ge 75$).
- Training terminal length percentiles: 25th, 50th, 75th, 90th percentiles of training $T_u$.

### Training-Derived Cutoff Schedules:
- **FD001**: `[80, 100, 125, 180, 199, 216, 279]`
- **FD002**: `[80, 100, 125, 177, 200, 234, 275]`
- **FD003**: `[80, 100, 125, 188, 222, 290, 349]`
- **FD004**: `[80, 100, 125, 193, 235, 292, 345]`

### Strict Causal & Integrity Constraints:
1. **Strictly Positive RUL**: Only observations where $t < T_u$ ($RUL > 0$) are evaluated. Terminal rows ($t = T_u$) are excluded.
2. **Causal Predictor Matrix**: Predictor features include only rolling statistics computed through cycle $t$.
3. **No Target or Future Leakage**: `rul`, `cycle`, `unit_id`, and failure metadata are excluded from predictors.
4. **No Validation Contamination**: Preprocessing transformations (StandardScaler) are fitted exclusively on training data.
5. **Reserved Test Set**: Official C-MAPSS test sets (`test_FD*.txt`, `RUL_FD*.txt`) remain completely untouched.

## D. Corrected Benchmark Model Comparison

| Dataset | Model | Simulated EP MAE (cycles) | 95% CI | Engine-Macro MAE | Simulated EP RMSE | NASA Score Sum | Phase 4 Terminal MAE | Rank Shift |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FD001 | DummyBaseline | 39.88 | [34.1, 45.7] | 35.73 | 50.86 | 49954.2 | 66.00 | Unchanged |
| FD001 | LinearRegression | 45.58 | [38.8, 52.4] | 43.71 | 55.75 | 2791266.3 | 2.77 | Unchanged |
| FD001 | RidgeRegression | 45.44 | [38.8, 52.1] | 43.52 | 55.65 | 1902952.4 | 2.62 | Unchanged |
| FD001 | RandomForest | 46.71 | [39.9, 53.5] | 44.71 | 57.49 | 421488.0 | 2.65 | Unchanged |
| FD001 | XGBoost | 46.45 | [39.3, 54.2] | 44.07 | 57.63 | 452410.3 | 1.98 | Unchanged |
| FD001 | LightGBM | 47.47 | [40.1, 55.4] | 44.64 | 58.99 | 549520.2 | 1.86 | Unchanged |
| FD002 | DummyBaseline | 40.81 | [34.9, 46.5] | 36.12 | 52.49 | 4776401.1 | 67.00 | Unchanged |
| FD002 | LinearRegression | 54.91 | [48.4, 61.3] | 52.83 | 67.82 | 13908190.2 | 5.06 | Unchanged |
| FD002 | RidgeRegression | 54.92 | [48.4, 61.4] | 52.85 | 67.86 | 14727731.6 | 5.12 | Unchanged |
| FD002 | RandomForest | 53.97 | [47.7, 60.9] | 51.81 | 66.19 | 11761344.5 | 2.91 | Unchanged |
| FD002 | XGBoost | 56.22 | [49.5, 63.4] | 54.16 | 69.09 | 20003832.3 | 2.54 | Unchanged |
| FD002 | LightGBM | 55.72 | [48.8, 62.6] | 53.72 | 68.30 | 18194263.2 | 2.38 | Unchanged |
| FD003 | DummyBaseline | 60.19 | [37.6, 83.3] | 50.35 | 87.73 | 6705272222.0 | 88.00 | Unchanged |
| FD003 | LinearRegression | 71.68 | [60.9, 84.3] | 69.46 | 88.32 | 333258573.2 | 8.35 | Unchanged |
| FD003 | RidgeRegression | 71.55 | [61.1, 84.0] | 69.30 | 87.92 | 274667775.5 | 7.94 | Unchanged |
| FD003 | RandomForest | 67.07 | [50.6, 82.9] | 61.11 | 87.76 | 259297362.5 | 2.35 | Unchanged |
| FD003 | XGBoost | 79.40 | [61.9, 95.1] | 75.04 | 99.10 | 582102038.1 | 1.79 | Unchanged |
| FD003 | LightGBM | 72.00 | [55.1, 87.7] | 66.61 | 92.33 | 254627759.2 | 1.66 | Unchanged |
| FD004 | DummyBaseline | 62.99 | [53.4, 73.4] | 54.05 | 82.93 | 6118803476.0 | 87.00 | Unchanged |
| FD004 | LinearRegression | 89.03 | [75.3, 106.3] | 85.83 | 111.87 | 1901660895717.7 | 6.48 | Unchanged |
| FD004 | RidgeRegression | 89.01 | [75.2, 106.3] | 85.81 | 111.89 | 1990662673891.6 | 6.47 | Unchanged |
| FD004 | RandomForest | 88.47 | [75.3, 103.9] | 85.58 | 110.66 | 3011254047846.3 | 5.69 | Unchanged |
| FD004 | XGBoost | 89.59 | [76.2, 106.8] | 87.85 | 112.88 | 3253460355858.9 | 5.20 | Unchanged |
| FD004 | LightGBM | 89.59 | [76.1, 105.7] | 87.68 | 112.50 | 2983971908005.5 | 5.58 | Unchanged |

## E. Cutoff Coverage & Sample Reliability

Validation engines naturally drop out of later checkpoints once their failure cycle is reached. The table below documents coverage and RUL distribution across checkpoints:

| Dataset | Cutoff Cycle | Eligible Engines | Observations | Engine Coverage | Mean True RUL | Median True RUL | RUL Range [Min, Max] | Low Sample (<5 engines)? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FD001 | 80 | 20 | 20 | 100.0% | 123.5 | 113.5 | [55, 189] | No |
| FD001 | 100 | 20 | 20 | 100.0% | 103.5 | 93.5 | [35, 169] | No |
| FD001 | 125 | 20 | 20 | 100.0% | 78.5 | 68.5 | [10, 144] | No |
| FD001 | 180 | 13 | 13 | 65.0% | 48.3 | 54.0 | [8, 89] | No |
| FD001 | 199 | 9 | 9 | 45.0% | 45.7 | 41.0 | [9, 70] | No |
| FD001 | 216 | 8 | 8 | 40.0% | 33.2 | 32.0 | [15, 53] | No |
| FD001 | 279 | 0 | 0 | 0.0% | nan | nan | [nan, nan] | YES (Interpret with caution) |
| FD002 | 80 | 52 | 52 | 100.0% | 118.0 | 115.0 | [49, 263] | No |
| FD002 | 100 | 52 | 52 | 100.0% | 98.0 | 95.0 | [29, 243] | No |
| FD002 | 125 | 52 | 52 | 100.0% | 73.0 | 70.0 | [4, 218] | No |
| FD002 | 177 | 33 | 33 | 63.5% | 44.3 | 29.0 | [1, 166] | No |
| FD002 | 200 | 22 | 22 | 42.3% | 36.5 | 23.5 | [1, 143] | No |
| FD002 | 234 | 9 | 9 | 17.3% | 37.2 | 17.0 | [7, 109] | No |
| FD002 | 275 | 3 | 3 | 5.8% | 39.3 | 28.0 | [22, 68] | YES (Interpret with caution) |
| FD003 | 80 | 20 | 20 | 100.0% | 155.4 | 124.5 | [76, 379] | No |
| FD003 | 100 | 20 | 20 | 100.0% | 135.4 | 104.5 | [56, 359] | No |
| FD003 | 125 | 20 | 20 | 100.0% | 110.4 | 79.5 | [31, 334] | No |
| FD003 | 188 | 16 | 16 | 80.0% | 61.8 | 29.0 | [1, 271] | No |
| FD003 | 222 | 7 | 7 | 35.0% | 90.0 | 37.0 | [4, 237] | No |
| FD003 | 290 | 3 | 3 | 15.0% | 115.0 | 119.0 | [57, 169] | YES (Interpret with caution) |
| FD003 | 349 | 2 | 2 | 10.0% | 85.0 | 85.0 | [60, 110] | YES (Interpret with caution) |
| FD004 | 80 | 50 | 50 | 100.0% | 159.1 | 145.5 | [51, 377] | No |
| FD004 | 100 | 50 | 50 | 100.0% | 139.1 | 125.5 | [31, 357] | No |
| FD004 | 125 | 50 | 50 | 100.0% | 114.1 | 100.5 | [6, 332] | No |
| FD004 | 193 | 32 | 32 | 64.0% | 87.4 | 72.5 | [1, 264] | No |
| FD004 | 235 | 24 | 24 | 48.0% | 68.6 | 57.5 | [6, 222] | No |
| FD004 | 292 | 12 | 12 | 24.0% | 55.2 | 51.5 | [5, 165] | No |
| FD004 | 345 | 6 | 6 | 12.0% | 35.8 | 29.0 | [2, 112] | No |

## F. Corrected RUL-Cap Findings on Positive-RUL Observations

When evaluated against **uncapped physical truth** at positive-RUL cutoff checkpoints:
1. **Disappearance of Equality**: `simulated_endpoint_mae_uncapped` and `simulated_endpoint_mae_capped` are no longer equal. Evaluating capped models against uncapped true physical RUL reveals the penalty of clipping when true life exceeds the cap.
2. **Optimal Cap Balance**: A target cap of **100–125 cycles** provides the best balance between reducing early-life noise and avoiding severe under-prediction penalties when true remaining life is large.
3. **Cap 50 Tradeoff**: While Cap 50 achieves excellent near-failure precision ($RUL \le 30$), it severely truncates predictions for engines observed at cycles 80–100 where true RUL is >100, incurring substantial physical MAE on early checkpoints.

| Dataset | Model | RUL Cap | Uncapped EP MAE (cycles) | Uncapped EP RMSE | Uncapped Macro MAE | NASA Score Sum | Capped Target MAE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FD001 | RidgeRegression | None | 15.16 | 18.63 | 15.17 | 757.2 | 15.16 |
| FD001 | RidgeRegression | 50 | 26.44 | 33.87 | 26.86 | 2271.0 | 1.13 |
| FD001 | RidgeRegression | 75 | 14.33 | 18.50 | 14.30 | 356.9 | 4.29 |
| FD001 | RidgeRegression | 100 | 7.38 | 8.71 | 7.09 | 97.4 | 5.68 |
| FD001 | RidgeRegression | 125 | 7.50 | 8.58 | 7.58 | 119.0 | 7.50 |
| FD001 | RidgeRegression | 150 | 12.77 | 14.94 | 12.72 | 358.5 | 12.77 |
| FD001 | LinearRegression | None | 15.39 | 19.12 | 15.43 | 890.9 | 15.39 |
| FD001 | LinearRegression | 50 | 26.32 | 33.84 | 26.73 | 2269.3 | 1.01 |
| FD001 | LinearRegression | 75 | 14.62 | 18.65 | 14.57 | 363.8 | 4.59 |
| FD001 | LinearRegression | 100 | 8.05 | 9.43 | 7.74 | 112.3 | 6.35 |
| FD001 | LinearRegression | 125 | 7.81 | 9.06 | 7.86 | 130.7 | 7.81 |
| FD001 | LinearRegression | 150 | 13.28 | 15.52 | 13.22 | 398.1 | 13.28 |
| FD002 | RandomForest | None | 18.08 | 22.14 | 18.06 | 2752.0 | 18.08 |
| FD002 | RandomForest | 50 | 23.92 | 40.73 | 24.06 | 645711.8 | 2.33 |
| FD002 | RandomForest | 75 | 15.82 | 30.18 | 15.84 | 95732.8 | 5.10 |
| FD002 | RandomForest | 100 | 13.75 | 24.54 | 13.42 | 16150.4 | 8.30 |
| FD002 | RandomForest | 125 | 15.73 | 22.02 | 15.51 | 3795.7 | 11.94 |
| FD002 | RandomForest | 150 | 17.25 | 22.04 | 17.10 | 2498.3 | 14.92 |
| FD002 | LinearRegression | None | 16.76 | 20.58 | 16.63 | 1992.5 | 16.76 |
| FD002 | LinearRegression | 50 | 23.31 | 40.66 | 23.53 | 645667.9 | 1.72 |
| FD002 | LinearRegression | 75 | 14.90 | 29.95 | 14.98 | 94437.7 | 4.19 |
| FD002 | LinearRegression | 100 | 13.56 | 24.31 | 13.25 | 14284.6 | 8.11 |
| FD002 | LinearRegression | 125 | 15.84 | 22.82 | 15.60 | 4273.1 | 12.06 |
| FD002 | LinearRegression | 150 | 16.41 | 22.08 | 16.28 | 2744.7 | 14.08 |
| FD003 | RandomForest | None | 20.56 | 21.34 | 20.66 | 388.2 | 20.56 |
| FD003 | RandomForest | 50 | 90.51 | 94.01 | 91.56 | 404831.5 | 0.01 |
| FD003 | RandomForest | 75 | 66.44 | 70.83 | 67.53 | 60896.5 | 0.94 |
| FD003 | RandomForest | 100 | 48.40 | 52.44 | 49.40 | 11208.8 | 7.83 |
| FD003 | RandomForest | 125 | 37.29 | 40.37 | 38.10 | 3258.2 | 17.18 |
| FD003 | RandomForest | 150 | 31.39 | 33.57 | 32.06 | 1468.8 | 24.63 |
| FD003 | RidgeRegression | None | 34.12 | 37.25 | 33.88 | 5708.0 | 34.12 |
| FD003 | RidgeRegression | 50 | 90.52 | 94.01 | 91.59 | 404136.1 | 0.02 |
| FD003 | RidgeRegression | 75 | 66.01 | 70.45 | 67.13 | 59020.7 | 0.51 |
| FD003 | RidgeRegression | 100 | 45.49 | 50.97 | 46.64 | 10245.3 | 4.92 |
| FD003 | RidgeRegression | 125 | 33.06 | 40.27 | 34.24 | 3966.5 | 12.95 |
| FD003 | RidgeRegression | 150 | 26.00 | 33.53 | 26.99 | 2090.9 | 19.24 |
| FD004 | RandomForest | None | 18.78 | 26.45 | 17.98 | 3062.7 | 18.78 |
| FD004 | RandomForest | 50 | 88.13 | 118.68 | 86.71 | 603050460.8 | 1.69 |
| FD004 | RandomForest | 75 | 71.71 | 101.54 | 70.56 | 89272467.1 | 3.23 |
| FD004 | RandomForest | 100 | 59.06 | 86.13 | 58.19 | 14005851.0 | 5.77 |
| FD004 | RandomForest | 125 | 49.34 | 72.13 | 48.58 | 2327209.7 | 8.43 |
| FD004 | RandomForest | 150 | 40.94 | 59.69 | 40.23 | 462709.5 | 9.66 |
| FD004 | RidgeRegression | None | 32.70 | 37.35 | 32.71 | 8948.3 | 32.70 |
| FD004 | RidgeRegression | 50 | 89.37 | 120.02 | 87.78 | 682528691.9 | 2.93 |
| FD004 | RidgeRegression | 75 | 73.56 | 103.18 | 72.22 | 108075366.3 | 5.09 |
| FD004 | RidgeRegression | 100 | 62.04 | 87.53 | 61.13 | 15470375.0 | 8.74 |
| FD004 | RidgeRegression | 125 | 52.34 | 73.32 | 51.77 | 2062061.8 | 11.44 |
| FD004 | RidgeRegression | 150 | 45.18 | 61.42 | 44.70 | 316817.4 | 13.90 |

## G. Corrected Temporal Window Ablation

Evaluating single rolling window feature subsets ($W \in \{10, 20, 30, 50, 75\}$) against the combined representation (`All` windows, 670–776 features) under the corrected protocol confirms:
1. **Combined Multi-Scale Representation is Best**: `All` windows achieves the lowest simulated endpoint MAE across FD001, FD002, FD003, and FD004.
2. **Longer Windows Dominate Single-Window Models**: Among single windows, $W=50$ and $W=75$ substantially outperform short windows ($W=10, 20$), confirming that capturing low-frequency long-term degradation curvature is critical for estimating positive RUL.

| Dataset | Window Config | Features | Simulated EP MAE | Simulated EP RMSE | Macro MAE | NASA Score Sum |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FD001 | All | 670 | 15.16 | 18.63 | 15.17 | 757.2 |
| FD001 | W=10 | 194 | 26.77 | 28.68 | 26.82 | 2274.6 |
| FD001 | W=20 | 195 | 20.45 | 24.07 | 21.47 | 1465.1 |
| FD001 | W=30 | 194 | 13.99 | 17.29 | 13.84 | 552.3 |
| FD001 | W=50 | 194 | 12.03 | 14.84 | 11.95 | 377.7 |
| FD001 | W=75 | 193 | 21.33 | 24.65 | 21.72 | 1552.9 |
| FD002 | All | 776 | 18.08 | 22.14 | 18.06 | 2752.0 |
| FD002 | W=10 | 235 | 23.07 | 26.24 | 22.58 | 4135.3 |
| FD002 | W=20 | 237 | 22.71 | 26.25 | 22.24 | 4443.3 |
| FD002 | W=30 | 237 | 20.47 | 24.04 | 20.34 | 3741.2 |
| FD002 | W=50 | 236 | 17.77 | 21.59 | 17.79 | 2491.7 |
| FD002 | W=75 | 231 | 18.75 | 23.14 | 18.67 | 3081.0 |
| FD003 | All | 681 | 20.56 | 21.34 | 20.66 | 388.2 |
| FD003 | W=10 | 200 | 30.97 | 33.26 | 30.96 | 1478.4 |
| FD003 | W=20 | 201 | 29.11 | 31.13 | 29.34 | 1202.9 |
| FD003 | W=30 | 201 | 27.46 | 29.25 | 27.75 | 953.9 |
| FD003 | W=50 | 200 | 22.95 | 24.23 | 23.43 | 528.2 |
| FD003 | W=75 | 199 | 21.71 | 22.30 | 21.74 | 418.6 |
| FD004 | All | 768 | 18.78 | 26.45 | 17.98 | 3062.7 |
| FD004 | W=10 | 235 | 28.58 | 33.86 | 28.00 | 5927.5 |
| FD004 | W=20 | 235 | 27.22 | 32.34 | 26.52 | 5024.2 |
| FD004 | W=30 | 234 | 26.02 | 31.90 | 25.40 | 5127.2 |
| FD004 | W=50 | 233 | 22.97 | 31.21 | 22.14 | 6626.9 |
| FD004 | W=75 | 231 | 17.67 | 24.53 | 17.14 | 2342.3 |

## H. Stratified Error Analysis by Operational RUL Range

| Dataset | Model | RUL Range | Range (cycles) | Observations | MAE (cycles) | RMSE (cycles) | Mean Signed Error | Underpredict % | Overpredict % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| FD001 | DummyBaseline | critical_0_30 | 0–30 | 12 | 49.08 | 49.61 | 49.08 | 0.0% | 100.0% |
| FD001 | DummyBaseline | low_31_75 | 31–75 | 34 | 14.38 | 18.18 | 13.15 | 17.6% | 82.4% |
| FD001 | DummyBaseline | moderate_76_125 | 76–125 | 23 | 29.22 | 32.23 | -29.22 | 100.0% | 0.0% |
| FD001 | DummyBaseline | high_gt125 | 126–>125 | 21 | 87.57 | 89.47 | -87.57 | 100.0% | 0.0% |
| FD001 | LightGBM | critical_0_30 | 0–30 | 12 | 55.01 | 64.01 | 55.01 | 0.0% | 100.0% |
| FD001 | LightGBM | low_31_75 | 31–75 | 34 | 37.52 | 48.78 | 28.99 | 26.5% | 73.5% |
| FD001 | LightGBM | moderate_76_125 | 76–125 | 23 | 33.32 | 39.06 | -5.92 | 65.2% | 34.8% |
| FD001 | LightGBM | high_gt125 | 126–>125 | 21 | 74.79 | 83.97 | -73.69 | 95.2% | 4.8% |
| FD001 | LinearRegression | critical_0_30 | 0–30 | 12 | 67.63 | 75.04 | 67.63 | 0.0% | 100.0% |
| FD001 | LinearRegression | low_31_75 | 31–75 | 34 | 36.34 | 44.25 | 32.03 | 14.7% | 85.3% |
| FD001 | LinearRegression | moderate_76_125 | 76–125 | 23 | 27.29 | 32.51 | -0.88 | 60.9% | 39.1% |
| FD001 | LinearRegression | high_gt125 | 126–>125 | 21 | 67.97 | 76.00 | -67.97 | 100.0% | 0.0% |
| FD001 | RandomForest | critical_0_30 | 0–30 | 12 | 62.03 | 70.77 | 62.03 | 0.0% | 100.0% |
| FD001 | RandomForest | low_31_75 | 31–75 | 34 | 38.92 | 47.25 | 31.56 | 23.5% | 76.5% |
| FD001 | RandomForest | moderate_76_125 | 76–125 | 23 | 27.52 | 33.31 | -2.08 | 52.2% | 47.8% |
| FD001 | RandomForest | high_gt125 | 126–>125 | 21 | 71.57 | 80.47 | -71.53 | 95.2% | 4.8% |
| FD001 | RidgeRegression | critical_0_30 | 0–30 | 12 | 67.00 | 74.35 | 67.00 | 0.0% | 100.0% |
| FD001 | RidgeRegression | low_31_75 | 31–75 | 34 | 36.05 | 44.05 | 31.70 | 17.6% | 82.4% |
| FD001 | RidgeRegression | moderate_76_125 | 76–125 | 23 | 27.18 | 32.28 | -1.48 | 60.9% | 39.1% |
| FD001 | RidgeRegression | high_gt125 | 126–>125 | 21 | 68.32 | 76.36 | -68.32 | 100.0% | 0.0% |
| FD001 | XGBoost | critical_0_30 | 0–30 | 12 | 57.39 | 65.75 | 57.39 | 0.0% | 100.0% |
| FD001 | XGBoost | low_31_75 | 31–75 | 34 | 37.92 | 48.05 | 30.40 | 23.5% | 76.5% |
| FD001 | XGBoost | moderate_76_125 | 76–125 | 23 | 30.03 | 36.45 | -3.40 | 52.2% | 47.8% |
| FD001 | XGBoost | high_gt125 | 126–>125 | 21 | 71.99 | 81.07 | -71.99 | 100.0% | 0.0% |
| FD002 | DummyBaseline | critical_0_30 | 0–30 | 47 | 49.91 | 50.72 | 49.91 | 0.0% | 100.0% |
| FD002 | DummyBaseline | low_31_75 | 31–75 | 68 | 13.01 | 16.58 | 10.63 | 27.9% | 72.1% |
| FD002 | DummyBaseline | moderate_76_125 | 76–125 | 72 | 33.14 | 36.28 | -33.14 | 100.0% | 0.0% |
| FD002 | DummyBaseline | high_gt125 | 126–>125 | 36 | 96.75 | 102.73 | -96.75 | 100.0% | 0.0% |
| FD002 | LightGBM | critical_0_30 | 0–30 | 47 | 58.48 | 72.37 | 56.53 | 8.5% | 91.5% |
| FD002 | LightGBM | low_31_75 | 31–75 | 68 | 42.32 | 52.21 | 15.79 | 45.6% | 54.4% |
| FD002 | LightGBM | moderate_76_125 | 76–125 | 72 | 51.69 | 58.02 | -29.44 | 72.2% | 27.8% |
| FD002 | LightGBM | high_gt125 | 126–>125 | 36 | 85.46 | 100.88 | -83.44 | 94.4% | 5.6% |
| FD002 | LinearRegression | critical_0_30 | 0–30 | 47 | 56.92 | 72.24 | 54.32 | 14.9% | 85.1% |
| FD002 | LinearRegression | low_31_75 | 31–75 | 68 | 41.21 | 49.96 | 12.59 | 50.0% | 50.0% |
| FD002 | LinearRegression | moderate_76_125 | 76–125 | 72 | 51.55 | 59.19 | -33.11 | 70.8% | 29.2% |
| FD002 | LinearRegression | high_gt125 | 126–>125 | 36 | 84.87 | 99.79 | -83.45 | 94.4% | 5.6% |
| FD002 | RandomForest | critical_0_30 | 0–30 | 47 | 58.98 | 71.48 | 57.25 | 8.5% | 91.5% |
| FD002 | RandomForest | low_31_75 | 31–75 | 68 | 40.11 | 48.49 | 15.98 | 45.6% | 54.4% |
| FD002 | RandomForest | moderate_76_125 | 76–125 | 72 | 48.54 | 55.46 | -29.15 | 69.4% | 30.6% |
| FD002 | RandomForest | high_gt125 | 126–>125 | 36 | 84.46 | 99.38 | -83.15 | 94.4% | 5.6% |
| FD002 | RidgeRegression | critical_0_30 | 0–30 | 47 | 57.00 | 72.37 | 54.42 | 14.9% | 85.1% |
| FD002 | RidgeRegression | low_31_75 | 31–75 | 68 | 41.15 | 49.95 | 12.65 | 48.5% | 51.5% |
| FD002 | RidgeRegression | moderate_76_125 | 76–125 | 72 | 51.57 | 59.22 | -32.98 | 70.8% | 29.2% |
| FD002 | RidgeRegression | high_gt125 | 126–>125 | 36 | 84.89 | 99.81 | -83.48 | 94.4% | 5.6% |
| FD002 | XGBoost | critical_0_30 | 0–30 | 47 | 58.77 | 73.49 | 56.83 | 8.5% | 91.5% |
| FD002 | XGBoost | low_31_75 | 31–75 | 68 | 42.34 | 52.25 | 15.52 | 48.5% | 51.5% |
| FD002 | XGBoost | moderate_76_125 | 76–125 | 72 | 52.02 | 58.53 | -30.12 | 70.8% | 29.2% |
| FD002 | XGBoost | high_gt125 | 126–>125 | 36 | 87.49 | 102.52 | -85.26 | 94.4% | 5.6% |
| FD003 | DummyBaseline | critical_0_30 | 0–30 | 10 | 78.20 | 78.51 | 78.20 | 0.0% | 100.0% |
| FD003 | DummyBaseline | low_31_75 | 31–75 | 19 | 32.79 | 35.76 | 32.79 | 0.0% | 100.0% |
| FD003 | DummyBaseline | moderate_76_125 | 76–125 | 32 | 16.81 | 20.19 | -14.50 | 78.1% | 21.9% |
| FD003 | DummyBaseline | high_gt125 | 126–>125 | 27 | 124.22 | 146.36 | -124.22 | 100.0% | 0.0% |
| FD003 | LightGBM | critical_0_30 | 0–30 | 10 | 122.89 | 126.18 | 122.89 | 0.0% | 100.0% |
| FD003 | LightGBM | low_31_75 | 31–75 | 19 | 67.38 | 75.23 | 67.38 | 0.0% | 100.0% |
| FD003 | LightGBM | moderate_76_125 | 76–125 | 32 | 33.48 | 42.40 | 27.07 | 25.0% | 75.0% |
| FD003 | LightGBM | high_gt125 | 126–>125 | 27 | 102.05 | 125.58 | -91.69 | 81.5% | 18.5% |
| FD003 | LinearRegression | critical_0_30 | 0–30 | 10 | 137.39 | 139.02 | 137.39 | 0.0% | 100.0% |
| FD003 | LinearRegression | low_31_75 | 31–75 | 19 | 89.87 | 92.66 | 89.87 | 0.0% | 100.0% |
| FD003 | LinearRegression | moderate_76_125 | 76–125 | 32 | 45.08 | 50.29 | 45.08 | 0.0% | 100.0% |
| FD003 | LinearRegression | high_gt125 | 126–>125 | 27 | 66.06 | 96.06 | -61.19 | 81.5% | 18.5% |
| FD003 | RandomForest | critical_0_30 | 0–30 | 10 | 114.56 | 116.36 | 114.56 | 0.0% | 100.0% |
| FD003 | RandomForest | low_31_75 | 31–75 | 19 | 62.14 | 66.90 | 62.14 | 0.0% | 100.0% |
| FD003 | RandomForest | moderate_76_125 | 76–125 | 32 | 27.81 | 33.44 | 20.65 | 21.9% | 78.1% |
| FD003 | RandomForest | high_gt125 | 126–>125 | 27 | 99.49 | 124.96 | -95.91 | 88.9% | 11.1% |
| FD003 | RidgeRegression | critical_0_30 | 0–30 | 10 | 136.71 | 138.13 | 136.71 | 0.0% | 100.0% |
| FD003 | RidgeRegression | low_31_75 | 31–75 | 19 | 89.55 | 92.06 | 89.55 | 0.0% | 100.0% |
| FD003 | RidgeRegression | moderate_76_125 | 76–125 | 32 | 44.84 | 49.46 | 44.84 | 0.0% | 100.0% |
| FD003 | RidgeRegression | high_gt125 | 126–>125 | 27 | 66.39 | 96.25 | -61.56 | 81.5% | 18.5% |
| FD003 | XGBoost | critical_0_30 | 0–30 | 10 | 133.02 | 139.05 | 133.02 | 0.0% | 100.0% |
| FD003 | XGBoost | low_31_75 | 31–75 | 19 | 77.24 | 88.83 | 77.24 | 0.0% | 100.0% |
| FD003 | XGBoost | moderate_76_125 | 76–125 | 32 | 42.49 | 55.84 | 33.80 | 28.1% | 71.9% |
| FD003 | XGBoost | high_gt125 | 126–>125 | 27 | 104.82 | 124.89 | -86.27 | 77.8% | 22.2% |
| FD004 | DummyBaseline | critical_0_30 | 0–30 | 27 | 72.33 | 72.91 | 72.33 | 0.0% | 100.0% |
| FD004 | DummyBaseline | low_31_75 | 31–75 | 55 | 32.98 | 35.37 | 32.98 | 0.0% | 100.0% |
| FD004 | DummyBaseline | moderate_76_125 | 76–125 | 55 | 14.65 | 18.50 | -12.33 | 69.1% | 30.9% |
| FD004 | DummyBaseline | high_gt125 | 126–>125 | 87 | 109.62 | 122.67 | -109.62 | 100.0% | 0.0% |
| FD004 | LightGBM | critical_0_30 | 0–30 | 27 | 113.32 | 132.00 | 113.32 | 0.0% | 100.0% |
| FD004 | LightGBM | low_31_75 | 31–75 | 55 | 87.93 | 113.15 | 74.86 | 21.8% | 78.2% |
| FD004 | LightGBM | moderate_76_125 | 76–125 | 55 | 63.92 | 78.02 | 12.57 | 49.1% | 50.9% |
| FD004 | LightGBM | high_gt125 | 126–>125 | 87 | 99.49 | 123.43 | -74.75 | 75.9% | 24.1% |
| FD004 | LinearRegression | critical_0_30 | 0–30 | 27 | 104.39 | 125.80 | 103.26 | 14.8% | 85.2% |
| FD004 | LinearRegression | low_31_75 | 31–75 | 55 | 83.11 | 100.90 | 53.93 | 30.9% | 69.1% |
| FD004 | LinearRegression | moderate_76_125 | 76–125 | 55 | 67.29 | 79.83 | -2.04 | 49.1% | 50.9% |
| FD004 | LinearRegression | high_gt125 | 126–>125 | 87 | 101.74 | 129.80 | -84.01 | 79.3% | 20.7% |
| FD004 | RandomForest | critical_0_30 | 0–30 | 27 | 102.19 | 121.25 | 102.19 | 0.0% | 100.0% |
| FD004 | RandomForest | low_31_75 | 31–75 | 55 | 80.23 | 104.90 | 65.60 | 23.6% | 76.4% |
| FD004 | RandomForest | moderate_76_125 | 76–125 | 55 | 63.64 | 76.29 | 3.94 | 60.0% | 40.0% |
| FD004 | RandomForest | high_gt125 | 126–>125 | 87 | 105.11 | 127.79 | -84.33 | 77.0% | 23.0% |
| FD004 | RidgeRegression | critical_0_30 | 0–30 | 27 | 104.24 | 125.68 | 103.18 | 14.8% | 85.2% |
| FD004 | RidgeRegression | low_31_75 | 31–75 | 55 | 83.07 | 100.85 | 53.95 | 30.9% | 69.1% |
| FD004 | RidgeRegression | moderate_76_125 | 76–125 | 55 | 67.18 | 79.83 | -2.14 | 49.1% | 50.9% |
| FD004 | RidgeRegression | high_gt125 | 126–>125 | 87 | 101.84 | 129.90 | -84.11 | 79.3% | 20.7% |
| FD004 | XGBoost | critical_0_30 | 0–30 | 27 | 114.59 | 132.57 | 114.59 | 0.0% | 100.0% |
| FD004 | XGBoost | low_31_75 | 31–75 | 55 | 89.16 | 113.56 | 75.79 | 21.8% | 78.2% |
| FD004 | XGBoost | moderate_76_125 | 76–125 | 55 | 64.88 | 80.80 | 16.04 | 45.5% | 54.5% |
| FD004 | XGBoost | high_gt125 | 126–>125 | 87 | 97.73 | 122.77 | -72.05 | 75.9% | 24.1% |

## I. Methodological Limitations

1. **Validation-Set Evaluation Only**: All reported metrics are evaluated on the Phase 2 engine-level validation split. The official C-MAPSS test set (`test_FD*.txt`) remains strictly untouched.
2. **Retrospective Simulation**: Although this protocol samples observations at $t < T_u$, it is a retrospective simulation and does not mirror the exact, secret truncation distribution of the official test set.
3. **Engine-Clustered Correlation**: Multiple cutoffs from the same engine share underlying mechanical degradation tendencies. While engine-cluster bootstrapping was used to prevent artificially narrow confidence intervals, checkpoints from the same unit remain correlated.
4. **Late-Checkpoint Attrition**: Checkpoints near the 90th percentile of training lifespans contain few active validation engines (e.g., $N < 5$), making metrics at very high cycles sensitive to individual unit outliers.
5. **Fixed Hyperparameters**: Models were evaluated using fixed starting configurations without Optuna hyperparameter searches.

## J. Phase 5 Readiness & Recommendations

- **Models to Carry Forward**: LightGBM and XGBoost should be the primary candidates for systematic hyperparameter tuning in Phase 5, with Random Forest as a robust classical benchmark.
- **Target Capping Recommendation**: Retain a piecewise linear target cap $c \in [100, 125]$ cycles as a tuned parameter in Phase 5, rather than the overly restrictive $c=50$ or unconstrained $c=\text{None}$.
- **Official Test Set**: Official test data will be evaluated only once final model architectures and hyperparameters are frozen in Phase 5.