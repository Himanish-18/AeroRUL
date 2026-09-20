# AeroRUL -- Phase 3 Summary Report

## 1. Feature Methodology

### Feature Families

| Family | Features | Description |
|--------|----------|-------------|
| Rolling Mean | sensor_{s}_mean_{W} | Mean of sensor over previous W cycles |
| Rolling Std | sensor_{s}_std_{W} | Population std over previous W cycles |
| Rolling Min | sensor_{s}_min_{W} | Minimum over previous W cycles |
| Rolling Max | sensor_{s}_max_{W} | Maximum over previous W cycles |
| Rolling Range | sensor_{s}_range_{W} | max - min over previous W cycles |
| Degradation Slope | sensor_{s}_slope_{W} | Linear OLS trend slope over W cycles |
| Trend Strength | sensor_{s}_r2_{W} | R² of linear fit over W cycles |
| Temporal Delta | sensor_{s}_delta_{k} | x_t - x_(t-k) |
| Relative Change | sensor_{s}_relchange_{k} | (x_t - x_(t-k)) / |x_(t-k)| |

## 2. Window Configurations

| Window | Captures |
|--------|----------|
| W=10 | Short-term / recent change |
| W=20 | Short-to-medium term |
| W=30 | Medium-term degradation |
| W=50 | Medium-to-long term |
| W=75 | Long-term degradation trend |

## 3. Feature Count

Total features per observation: **840**

| Statistic | Count |
|-----------|-------|
| delta | 63 |
| max | 105 |
| mean | 105 |
| min | 105 |
| r2 | 105 |
| range | 105 |
| relchange | 42 |
| slope | 105 |
| std | 105 |

## 4. Data Retention

Initial cycles are discarded when insufficient history exists for the largest window.

| Subset | Split | Original | Retained | Discarded | Retention % |
|--------|-------|----------|----------|-----------|------------|
| FD001 | train | 16561 | 10641 | 5920 | 64.25% |
| FD001 | val | 4070 | 2590 | 1480 | 63.64% |
| FD002 | train | 43464 | 28072 | 15392 | 64.59% |
| FD002 | val | 10295 | 6447 | 3848 | 62.62% |
| FD003 | train | 20012 | 14092 | 5920 | 70.42% |
| FD003 | val | 4708 | 3228 | 1480 | 68.56% |
| FD004 | train | 49294 | 34568 | 14726 | 70.13% |
| FD004 | val | 11955 | 8255 | 3700 | 69.05% |

## 5. Feature Quality

| Subset | Split | Features | Rows | NaN Cols | Inf Cols | Constant | Dup Rows |
|--------|-------|----------|------|----------|----------|----------|----------|
| FD001 | train | 840 | 10641 | 21 | 0 | 149 | 0 |
| FD001 | val | 840 | 2590 | 20 | 0 | 164 | 0 |
| FD002 | train | 840 | 28072 | 12 | 0 | 52 | 0 |
| FD002 | val | 840 | 6447 | 12 | 0 | 55 | 0 |
| FD003 | train | 840 | 14092 | 25 | 0 | 134 | 0 |
| FD003 | val | 840 | 3228 | 24 | 0 | 134 | 0 |
| FD004 | train | 840 | 34568 | 12 | 0 | 60 | 0 |
| FD004 | val | 840 | 8255 | 12 | 0 | 62 | 0 |

## 6. Target Integrity

| Subset | Split | RUL≥0 | No Leakage | Matches Original |
|--------|-------|-------|------------|------------------|
| FD001 | train | PASS | PASS | PASS |
| FD001 | val | PASS | PASS | PASS |
| FD002 | train | PASS | PASS | PASS |
| FD002 | val | PASS | PASS | PASS |
| FD003 | train | PASS | PASS | PASS |
| FD003 | val | PASS | PASS | PASS |
| FD004 | train | PASS | PASS | PASS |
| FD004 | val | PASS | PASS | PASS |

## 7. Leakage Verification

### Temporal Causality
All features at cycle t depend only on cycles ≤ t. Verified by automated test
that modifies future cycles and checks features at earlier cycles remain identical.

### Engine Separation
- **FD001**: train=80 engines, val=20 engines, overlap=NONE
- **FD002**: train=208 engines, val=52 engines, overlap=NONE
- **FD003**: train=80 engines, val=20 engines, overlap=NONE
- **FD004**: train=199 engines, val=50 engines, overlap=NONE

## 8. Normalization

- **FD001/FD003**: Raw sensor values used (single operating regime)
- **FD002/FD004**: Phase 2 condition-aware normalization applied BEFORE feature generation
- Normalization parameters fitted on training data only

## 9. Top Feature-Target Correlations (Pearson)

### FD001 (top 10)

| Feature | Correlation |
|---------|------------|
| sensor_11_r2_75 | -0.7897 |
| sensor_11_slope_75 | -0.7772 |
| sensor_4_slope_75 | -0.7689 |
| sensor_4_r2_75 | -0.7668 |
| sensor_15_slope_75 | -0.7600 |
| sensor_7_slope_75 | 0.7596 |
| sensor_20_slope_75 | 0.7591 |
| sensor_12_slope_75 | 0.7551 |
| sensor_12_r2_75 | -0.7539 |
| sensor_7_r2_75 | -0.7532 |

### FD002 (top 10)

| Feature | Correlation |
|---------|------------|
| sensor_11_r2_75 | -0.7891 |
| sensor_11_slope_75 | -0.7852 |
| sensor_4_mean_10 | -0.7745 |
| sensor_11_mean_10 | -0.7728 |
| sensor_15_slope_75 | -0.7718 |
| sensor_4_slope_75 | -0.7703 |
| sensor_15_mean_10 | -0.7675 |
| sensor_2_mean_20 | -0.7656 |
| sensor_2_mean_10 | -0.7639 |
| sensor_4_mean_20 | -0.7633 |

### FD003 (top 10)

| Feature | Correlation |
|---------|------------|
| sensor_3_mean_20 | -0.7762 |
| sensor_3_mean_10 | -0.7739 |
| sensor_17_mean_10 | -0.7726 |
| sensor_17_mean_20 | -0.7718 |
| sensor_3_mean_30 | -0.7714 |
| sensor_17_mean_30 | -0.7660 |
| sensor_3_mean_50 | -0.7558 |
| sensor_11_mean_10 | -0.7504 |
| sensor_17_mean_50 | -0.7493 |
| sensor_11_mean_20 | -0.7424 |

### FD004 (top 10)

| Feature | Correlation |
|---------|------------|
| sensor_11_mean_10 | -0.7624 |
| sensor_17_mean_10 | -0.7534 |
| sensor_11_mean_20 | -0.7522 |
| sensor_4_mean_10 | -0.7515 |
| sensor_17_mean_20 | -0.7498 |
| sensor_3_mean_10 | -0.7475 |
| sensor_3_mean_20 | -0.7463 |
| sensor_11_max_10 | -0.7449 |
| sensor_11_min_10 | -0.7440 |
| sensor_4_mean_20 | -0.7434 |

## 10. Methodological Note

> The optimal window size and feature set will be determined in Phase 4
> through validation performance with regression models. Phase 3 makes
> no claims about which window is 'best'.
