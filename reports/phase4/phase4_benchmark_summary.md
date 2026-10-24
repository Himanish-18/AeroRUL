# AeroRUL — Phase 4: Classical Regression Benchmark Report

## A. Executive Summary

### FD001
- **Best model (endpoint MAE)**: LightGBM — 1.86 cycles
- **Endpoint NASA score**: 4.36
- **Row MAE**: 9.81, Row RMSE: 15.40
- **Improvement over baseline**: 64.14 cycles (97.2%)

### FD002
- **Best model (endpoint MAE)**: LightGBM — 2.38 cycles
- **Endpoint NASA score**: 14.41
- **Row MAE**: 13.55, Row RMSE: 20.76
- **Improvement over baseline**: 64.62 cycles (96.4%)

### FD003
- **Best model (endpoint MAE)**: LightGBM — 1.66 cycles
- **Endpoint NASA score**: 3.77
- **Row MAE**: 18.44, Row RMSE: 33.55
- **Improvement over baseline**: 86.34 cycles (98.1%)

### FD004
- **Best model (endpoint MAE)**: XGBoost — 5.20 cycles
- **Endpoint NASA score**: 85.34
- **Row MAE**: 24.94, Row RMSE: 37.93
- **Improvement over baseline**: 81.80 cycles (94.0%)

### Limitations
- Validation-set results; official test set NOT used for selection
- Hyperparameters are fixed starting points, not optimized
- Early cycles excluded by 75-cycle feature window
- Row-level metrics may differ from endpoint metrics

## B. Experimental Methodology

- **Split**: Engine-level train/val (Phase 2, seed=42, 80/20)
- **Features**: Phase 3.1 approved (`YES`) features only
- **Preprocessing**: StandardScaler (train-fit) for linear models
- **Random seed**: 42
- **Prediction constraint**: max(pred, 0) for nonnegative RUL

### Model Configurations

**DummyBaseline**: `DummyRegressor` — {'strategy': 'median'}
**LinearRegression**: `LinearRegression` — {}
**RidgeRegression**: `Ridge` — {'alpha': 1.0, 'random_state': 42}
**RandomForest**: `RandomForestRegressor` — {'n_estimators': 300, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'random_state': 42, 'n_jobs': -1}
**XGBoost**: `XGBRegressor` — {'objective': 'reg:squarederror', 'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42, 'tree_method': 'hist', 'n_jobs': -1, 'verbosity': 0}
**LightGBM**: `LGBMRegressor` — {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31, 'max_depth': 8, 'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1, 'verbose': -1}

## C. Primary Benchmark Results

### FD001

| Model | Row MAE | Row RMSE | Row R² | Endpoint MAE | Endpoint RMSE | NASA Score | Runtime (s) |
|-------|---------|----------|--------|--------------|---------------|------------|-------------|
| LightGBM | 9.81 | 15.40 | 0.8941 | 1.86 | 2.32 | 4.36 | 16.5 |
| XGBoost | 10.12 | 15.17 | 0.8972 | 1.98 | 2.78 | 4.96 | 46.1 |
| RidgeRegression | 14.26 | 18.03 | 0.8547 | 2.62 | 5.54 | 10.38 | 0.3 |
| RandomForest | 10.15 | 14.86 | 0.9014 | 2.65 | 2.92 | 6.28 | 34.0 |
| LinearRegression | 14.32 | 18.14 | 0.8530 | 2.77 | 5.42 | 10.11 | 2.7 |
| DummyBaseline | 39.05 | 47.57 | -0.0108 | 66.00 | 66.00 | 14681.90 | 0.0 |

### FD002

| Model | Row MAE | Row RMSE | Row R² | Endpoint MAE | Endpoint RMSE | NASA Score | Runtime (s) |
|-------|---------|----------|--------|--------------|---------------|------------|-------------|
| LightGBM | 13.55 | 20.76 | 0.8201 | 2.38 | 2.64 | 14.41 | 48.4 |
| XGBoost | 13.78 | 20.94 | 0.8170 | 2.54 | 2.77 | 15.48 | 97.3 |
| RandomForest | 14.07 | 21.06 | 0.8148 | 2.91 | 3.15 | 18.06 | 121.2 |
| LinearRegression | 15.78 | 21.79 | 0.8018 | 5.06 | 10.15 | 109.90 | 12.8 |
| RidgeRegression | 15.74 | 21.75 | 0.8026 | 5.12 | 10.21 | 109.89 | 1.1 |
| DummyBaseline | 38.78 | 48.99 | -0.0018 | 67.00 | 67.00 | 42193.10 | 0.0 |

### FD003

| Model | Row MAE | Row RMSE | Row R² | Endpoint MAE | Endpoint RMSE | NASA Score | Runtime (s) |
|-------|---------|----------|--------|--------------|---------------|------------|-------------|
| LightGBM | 18.44 | 33.55 | 0.8270 | 1.66 | 2.01 | 3.77 | 35.0 |
| XGBoost | 18.52 | 33.81 | 0.8243 | 1.79 | 2.21 | 4.14 | 48.0 |
| RandomForest | 19.99 | 35.18 | 0.8098 | 2.35 | 2.53 | 5.42 | 36.0 |
| RidgeRegression | 26.40 | 38.31 | 0.7744 | 7.94 | 15.72 | 134.80 | 0.6 |
| LinearRegression | 26.80 | 38.70 | 0.7698 | 8.35 | 16.18 | 144.73 | 2.4 |
| DummyBaseline | 59.11 | 81.36 | -0.0173 | 88.00 | 88.00 | 132664.88 | 0.0 |

### FD004

| Model | Row MAE | Row RMSE | Row R² | Endpoint MAE | Endpoint RMSE | NASA Score | Runtime (s) |
|-------|---------|----------|--------|--------------|---------------|------------|-------------|
| XGBoost | 24.94 | 37.93 | 0.7341 | 5.20 | 8.56 | 85.34 | 83.3 |
| LightGBM | 25.02 | 38.37 | 0.7280 | 5.58 | 8.83 | 105.76 | 76.1 |
| RandomForest | 25.76 | 39.43 | 0.7127 | 5.69 | 7.44 | 55.23 | 126.3 |
| RidgeRegression | 32.39 | 45.45 | 0.6183 | 6.47 | 15.95 | 4257.38 | 1.3 |
| LinearRegression | 32.42 | 45.48 | 0.6177 | 6.48 | 15.93 | 4152.52 | 6.9 |
| DummyBaseline | 58.21 | 74.56 | -0.0271 | 87.00 | 87.00 | 300095.61 | 0.0 |

## D. Window Configuration Comparison

### FD001

| Window | n_features | Row MAE | Endpoint MAE | NASA Score |
|--------|------------|---------|--------------|------------|
| All | 670 | 9.81 | 1.86 | 4.36 |
| W=10 | 194 | 17.24 | 2.41 | 5.99 |
| W=20 | 195 | 15.60 | 2.24 | 5.74 |
| W=30 | 194 | 13.60 | 1.84 | 4.29 |
| W=50 | 194 | 11.13 | 2.25 | 5.43 |
| W=75 | 193 | 9.86 | 2.06 | 4.71 |

### FD002

| Window | n_features | Row MAE | Endpoint MAE | NASA Score |
|--------|------------|---------|--------------|------------|
| All | 776 | 13.55 | 2.38 | 14.41 |
| W=10 | 235 | 21.00 | 4.50 | 31.72 |
| W=20 | 237 | 19.63 | 3.22 | 22.66 |
| W=30 | 237 | 17.94 | 3.00 | 19.46 |
| W=50 | 236 | 15.37 | 2.95 | 18.41 |
| W=75 | 231 | 13.81 | 3.35 | 21.35 |

### FD003

| Window | n_features | Row MAE | Endpoint MAE | NASA Score |
|--------|------------|---------|--------------|------------|
| All | 681 | 18.44 | 1.66 | 3.77 |
| W=10 | 200 | 29.22 | 2.36 | 6.14 |
| W=20 | 201 | 24.58 | 1.48 | 3.49 |
| W=30 | 201 | 20.56 | 1.48 | 3.44 |
| W=50 | 200 | 17.49 | 1.78 | 4.22 |
| W=75 | 199 | 19.39 | 1.64 | 3.71 |

### FD004

| Window | n_features | Row MAE | Endpoint MAE | NASA Score |
|--------|------------|---------|--------------|------------|
| All | 768 | 24.94 | 5.20 | 85.34 |
| W=10 | 235 | 33.74 | 5.57 | 75.13 |
| W=20 | 235 | 31.96 | 6.41 | 62.58 |
| W=30 | 234 | 29.60 | 7.03 | 96.91 |
| W=50 | 233 | 26.16 | 5.74 | 162.56 |
| W=75 | 231 | 25.71 | 7.20 | 3595.52 |

## E. RUL Cap Sensitivity

### FD001

| Model | Cap | EP MAE (uncapped) | EP RMSE (uncapped) | NASA | EP MAE (capped) |
|-------|-----|-------------------|--------------------| -----|-----------------|
| LightGBM | None | 1.86 | 2.32 | 4.36 | 1.86 |
| LightGBM | 50 | 1.71 | 2.29 | 4.05 | 1.71 |
| LightGBM | 75 | 1.81 | 2.65 | 4.51 | 1.81 |
| LightGBM | 100 | 2.01 | 2.74 | 4.97 | 2.01 |
| LightGBM | 125 | 1.90 | 2.53 | 4.58 | 1.9 |
| LightGBM | 150 | 1.96 | 2.58 | 4.74 | 1.96 |
| XGBoost | None | 1.98 | 2.78 | 4.96 | 1.98 |
| XGBoost | 50 | 1.68 | 2.40 | 4.08 | 1.68 |
| XGBoost | 75 | 2.04 | 2.86 | 5.14 | 2.04 |
| XGBoost | 100 | 1.88 | 2.58 | 4.61 | 1.88 |
| XGBoost | 125 | 1.90 | 2.65 | 4.69 | 1.9 |
| XGBoost | 150 | 1.97 | 2.66 | 4.84 | 1.97 |

### FD002

| Model | Cap | EP MAE (uncapped) | EP RMSE (uncapped) | NASA | EP MAE (capped) |
|-------|-----|-------------------|--------------------| -----|-----------------|
| LightGBM | None | 2.38 | 2.64 | 14.41 | 2.38 |
| LightGBM | 50 | 1.79 | 2.08 | 10.56 | 1.79 |
| LightGBM | 75 | 1.98 | 2.22 | 11.71 | 1.98 |
| LightGBM | 100 | 2.12 | 2.39 | 12.68 | 2.12 |
| LightGBM | 125 | 2.28 | 2.56 | 13.76 | 2.28 |
| LightGBM | 150 | 2.41 | 2.65 | 14.57 | 2.41 |
| XGBoost | None | 2.54 | 2.77 | 15.48 | 2.54 |
| XGBoost | 50 | 1.90 | 2.31 | 11.45 | 1.9 |
| XGBoost | 75 | 2.05 | 2.33 | 12.22 | 2.05 |
| XGBoost | 100 | 2.08 | 2.40 | 12.50 | 2.08 |
| XGBoost | 125 | 2.30 | 2.54 | 13.82 | 2.3 |
| XGBoost | 150 | 2.35 | 2.53 | 14.07 | 2.35 |

### FD003

| Model | Cap | EP MAE (uncapped) | EP RMSE (uncapped) | NASA | EP MAE (capped) |
|-------|-----|-------------------|--------------------| -----|-----------------|
| LightGBM | None | 1.66 | 2.01 | 3.77 | 1.66 |
| LightGBM | 50 | 1.26 | 1.65 | 2.83 | 1.26 |
| LightGBM | 75 | 1.36 | 1.90 | 3.14 | 1.36 |
| LightGBM | 100 | 1.33 | 1.86 | 3.05 | 1.33 |
| LightGBM | 125 | 1.34 | 1.78 | 3.04 | 1.34 |
| LightGBM | 150 | 1.57 | 2.00 | 3.58 | 1.57 |
| XGBoost | None | 1.79 | 2.21 | 4.14 | 1.79 |
| XGBoost | 50 | 1.52 | 2.02 | 3.51 | 1.52 |
| XGBoost | 75 | 1.51 | 2.00 | 3.47 | 1.51 |
| XGBoost | 100 | 1.38 | 1.86 | 3.16 | 1.38 |
| XGBoost | 125 | 1.65 | 2.02 | 3.75 | 1.65 |
| XGBoost | 150 | 1.57 | 2.03 | 3.61 | 1.57 |

### FD004

| Model | Cap | EP MAE (uncapped) | EP RMSE (uncapped) | NASA | EP MAE (capped) |
|-------|-----|-------------------|--------------------| -----|-----------------|
| XGBoost | None | 5.20 | 8.56 | 85.34 | 5.2 |
| XGBoost | 50 | 3.15 | 4.95 | 27.32 | 3.15 |
| XGBoost | 75 | 3.71 | 6.34 | 43.74 | 3.71 |
| XGBoost | 100 | 3.84 | 6.31 | 42.81 | 3.84 |
| XGBoost | 125 | 4.24 | 7.22 | 64.31 | 4.24 |
| XGBoost | 150 | 4.80 | 7.51 | 57.03 | 4.8 |
| LightGBM | None | 5.58 | 8.83 | 105.76 | 5.58 |
| LightGBM | 50 | 3.03 | 4.79 | 24.39 | 3.03 |
| LightGBM | 75 | 4.07 | 6.98 | 46.91 | 4.07 |
| LightGBM | 100 | 4.41 | 7.02 | 46.02 | 4.41 |
| LightGBM | 125 | 4.17 | 6.88 | 45.35 | 4.17 |
| LightGBM | 150 | 4.66 | 7.65 | 58.00 | 4.66 |

## F. Error Analysis by RUL Range

### FD001

**DummyBaseline**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 51.00 | 51.78 | 51.00 |
| low_31_75 | 885 | 15.19 | 18.52 | 13.25 |
| moderate_76_125 | 683 | 31.69 | 34.76 | -31.69 |
| high_gt125 | 402 | 85.64 | 87.38 | -85.64 |

**LinearRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 9.94 | 12.21 | -2.12 |
| low_31_75 | 885 | 14.40 | 17.59 | 6.06 |
| moderate_76_125 | 683 | 16.57 | 20.50 | 9.53 |
| high_gt125 | 402 | 17.09 | 22.25 | -8.52 |

**RidgeRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 9.70 | 12.01 | -2.16 |
| low_31_75 | 885 | 14.39 | 17.58 | 6.10 |
| moderate_76_125 | 683 | 16.51 | 20.31 | 9.50 |
| high_gt125 | 402 | 17.17 | 22.17 | -8.66 |

**RandomForest**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 3.26 | 4.13 | 2.10 |
| low_31_75 | 885 | 9.09 | 11.97 | 5.17 |
| moderate_76_125 | 683 | 10.61 | 13.97 | 3.00 |
| high_gt125 | 402 | 22.34 | 27.37 | -20.92 |

**XGBoost**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 2.98 | 4.03 | 1.77 |
| low_31_75 | 885 | 7.82 | 10.15 | 3.98 |
| moderate_76_125 | 683 | 11.99 | 16.27 | 2.02 |
| high_gt125 | 402 | 23.01 | 27.96 | -19.41 |

**LightGBM**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 2.79 | 4.03 | 1.46 |
| low_31_75 | 885 | 7.22 | 9.75 | 3.93 |
| moderate_76_125 | 683 | 11.28 | 15.63 | 1.15 |
| high_gt125 | 402 | 23.85 | 29.64 | -21.42 |

### FD002

**DummyBaseline**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 52.00 | 52.76 | 52.00 |
| low_31_75 | 2293 | 15.81 | 19.27 | 14.31 |
| moderate_76_125 | 1740 | 30.79 | 33.90 | -30.79 |
| high_gt125 | 802 | 95.24 | 100.72 | -95.24 |

**LinearRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 9.35 | 12.15 | 3.37 |
| low_31_75 | 2293 | 12.60 | 16.53 | 6.70 |
| moderate_76_125 | 1740 | 20.90 | 26.56 | 13.96 |
| high_gt125 | 802 | 26.70 | 34.78 | -16.57 |

**RidgeRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 9.33 | 12.11 | 3.36 |
| low_31_75 | 2293 | 12.56 | 16.49 | 6.75 |
| moderate_76_125 | 1740 | 20.85 | 26.50 | 13.97 |
| high_gt125 | 802 | 26.66 | 34.71 | -16.59 |

**RandomForest**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 3.74 | 5.03 | 3.02 |
| low_31_75 | 2293 | 12.23 | 16.04 | 9.70 |
| moderate_76_125 | 1740 | 17.31 | 21.05 | 11.67 |
| high_gt125 | 802 | 33.05 | 42.66 | -27.72 |

**XGBoost**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 3.44 | 4.92 | 2.40 |
| low_31_75 | 2293 | 11.04 | 15.27 | 7.33 |
| moderate_76_125 | 1740 | 19.09 | 24.15 | 13.22 |
| high_gt125 | 802 | 30.84 | 39.29 | -24.68 |

**LightGBM**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1612 | 3.44 | 4.77 | 2.48 |
| low_31_75 | 2293 | 10.83 | 14.80 | 7.16 |
| moderate_76_125 | 1740 | 18.03 | 23.07 | 12.02 |
| high_gt125 | 802 | 31.89 | 40.47 | -25.36 |

### FD003

**DummyBaseline**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 73.00 | 73.55 | 73.00 |
| low_31_75 | 900 | 35.00 | 37.33 | 35.00 |
| moderate_76_125 | 881 | 14.22 | 17.53 | -10.73 |
| high_gt125 | 827 | 122.78 | 141.20 | -122.78 |

**LinearRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 13.50 | 18.28 | 2.89 |
| low_31_75 | 900 | 17.13 | 21.60 | -0.44 |
| moderate_76_125 | 881 | 29.38 | 37.95 | 19.85 |
| high_gt125 | 827 | 44.55 | 59.61 | -1.45 |

**RidgeRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 13.13 | 17.74 | 2.32 |
| low_31_75 | 900 | 17.02 | 21.47 | -0.60 |
| moderate_76_125 | 881 | 28.85 | 37.38 | 19.72 |
| high_gt125 | 827 | 43.96 | 59.18 | -1.81 |

**RandomForest**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 2.48 | 3.42 | 1.37 |
| low_31_75 | 900 | 9.07 | 12.90 | 6.87 |
| moderate_76_125 | 881 | 21.91 | 34.12 | 14.14 |
| high_gt125 | 827 | 42.95 | 58.31 | -15.07 |

**XGBoost**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 2.66 | 3.61 | 0.43 |
| low_31_75 | 900 | 6.95 | 9.97 | 4.38 |
| moderate_76_125 | 881 | 18.61 | 30.59 | 11.28 |
| high_gt125 | 827 | 42.89 | 57.85 | -8.45 |

**LightGBM**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 620 | 2.76 | 3.87 | -0.02 |
| low_31_75 | 900 | 6.20 | 8.47 | 3.44 |
| moderate_76_125 | 881 | 18.51 | 29.54 | 10.20 |
| high_gt125 | 827 | 43.46 | 58.09 | -6.39 |

### FD004

**DummyBaseline**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 72.00 | 72.55 | 72.00 |
| low_31_75 | 2209 | 34.27 | 36.63 | 34.27 |
| moderate_76_125 | 1915 | 14.76 | 18.42 | -11.72 |
| high_gt125 | 2581 | 102.65 | 114.97 | -102.65 |

**LinearRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 16.42 | 25.19 | 9.48 |
| low_31_75 | 2209 | 24.71 | 32.72 | 12.76 |
| moderate_76_125 | 1915 | 30.93 | 40.01 | 14.45 |
| high_gt125 | 2581 | 49.74 | 64.28 | -29.95 |

**RidgeRegression**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 16.41 | 25.22 | 9.50 |
| low_31_75 | 2209 | 24.66 | 32.67 | 12.76 |
| moderate_76_125 | 1915 | 30.90 | 39.95 | 14.43 |
| high_gt125 | 2581 | 49.70 | 64.24 | -29.95 |

**RandomForest**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 5.78 | 8.98 | 4.36 |
| low_31_75 | 2209 | 15.43 | 21.73 | 10.13 |
| moderate_76_125 | 1915 | 22.69 | 30.33 | 6.03 |
| high_gt125 | 2581 | 48.87 | 61.95 | -34.59 |

**XGBoost**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 7.37 | 15.76 | 5.17 |
| low_31_75 | 2209 | 15.07 | 21.25 | 9.65 |
| moderate_76_125 | 1915 | 22.50 | 29.55 | 7.27 |
| high_gt125 | 2581 | 45.76 | 58.47 | -26.64 |

**LightGBM**

| RUL Range | n | MAE | RMSE | Mean Error |
|-----------|---|-----|------|------------|
| critical_0_30 | 1550 | 6.92 | 14.38 | 4.91 |
| low_31_75 | 2209 | 14.38 | 20.59 | 9.56 |
| moderate_76_125 | 1915 | 22.90 | 30.19 | 6.80 |
| high_gt125 | 2581 | 46.56 | 59.54 | -27.77 |

## G. Limitations

1. **Validation bias**: Model selection on validation set introduces optimistic estimates
2. **Row vs endpoint**: Row-level metrics weight longer-lived engines more heavily
3. **Window exclusion**: 75-cycle window excludes early observations (data retention ~64-70%)
4. **Dataset variation**: FD002/FD004 (multi-condition) may behave differently from FD001/FD003
5. **No hyperparameter tuning**: Fixed configurations, not optimized
6. **Official test set NOT used**: Final evaluation deferred to Phase 5

## H. Phase 4 Conclusion

### Findings
- Ensemble models (RF, XGBoost, LightGBM) substantially outperform the trivial baseline
- Linear models provide a useful intermediate reference
- Performance near failure (0-30 cycles) requires careful analysis

### Phase 5 Readiness
- Strongest model families identified for hyperparameter tuning
- Window and cap experiments provide starting configurations
- Official test set remains untouched
- **Status: READY for Phase 5**