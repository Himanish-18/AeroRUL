# AeroRUL -- Phase 2 Summary Report

## 1. RUL Formulation

**Method**: `RUL = failure_cycle - current_cycle`

- `failure_cycle` = maximum observed cycle for each engine in training data
- Final training cycle has RUL = 0 (engine failure)
- RUL is always non-negative
- RUL decreases by exactly 1 per consecutive cycle

## 2. RUL Statistics

| Subset | Engines | Observations | Min RUL | Max RUL | Mean RUL | Median RUL | Std RUL |
|--------|---------|-------------|---------|---------|----------|------------|---------|
| FD001 | 100 | 20631 | 0 | 361 | 107.81 | 103.0 | 68.88 |
| FD002 | 260 | 53759 | 0 | 377 | 108.15 | 103.0 | 69.18 |
| FD003 | 100 | 24720 | 0 | 524 | 138.08 | 123.0 | 98.85 |
| FD004 | 249 | 61249 | 0 | 542 | 133.31 | 122.0 | 89.78 |

## 3. RUL Validation

All RUL targets passed validation:

| Check | Status |
|-------|--------|
| RUL >= 0 | PASS |
| min(RUL) == 0 per engine | PASS |
| max(RUL) == failure_cycle - 1 | PASS |
| RUL decreases by 1 per cycle | PASS |
| RUL + cycle == failure_cycle | PASS |

## 4. RUL Clipping Experiment

### FD001

| RUL Cap | Obs Clipped | % Clipped | Engines Affected | Mean RUL | Median RUL | Std RUL |
|---------|-------------|-----------|------------------|----------|------------|---------|
| None | 0 | 0.0 | 0 | 107.81 | 103.0 | 68.88 |
| 50 | 15531 | 75.28 | 100 | 43.82 | 50.0 | 13.03 |
| 75 | 13031 | 63.16 | 100 | 61.19 | 75.0 | 22.46 |
| 100 | 10531 | 51.04 | 100 | 75.52 | 100.0 | 32.26 |
| 125 | 8031 | 38.93 | 100 | 86.83 | 103.0 | 41.67 |
| 150 | 5607 | 27.18 | 93 | 95.13 | 103.0 | 49.97 |

### FD002

| RUL Cap | Obs Clipped | % Clipped | Engines Affected | Mean RUL | Median RUL | Std RUL |
|---------|-------------|-----------|------------------|----------|------------|---------|
| None | 0 | 0.0 | 0 | 108.15 | 103.0 | 69.18 |
| 50 | 40499 | 75.33 | 260 | 43.83 | 50.0 | 13.02 |
| 75 | 33999 | 63.24 | 260 | 61.22 | 75.0 | 22.45 |
| 100 | 27499 | 51.15 | 260 | 75.58 | 100.0 | 32.25 |
| 125 | 20999 | 39.06 | 260 | 86.91 | 103.0 | 41.67 |
| 150 | 14664 | 27.28 | 235 | 95.25 | 103.0 | 49.97 |

### FD003

| RUL Cap | Obs Clipped | % Clipped | Engines Affected | Mean RUL | Median RUL | Std RUL |
|---------|-------------|-----------|------------------|----------|------------|---------|
| None | 0 | 0.0 | 0 | 138.08 | 123.0 | 98.85 |
| 50 | 19620 | 79.37 | 100 | 44.84 | 50.0 | 12.13 |
| 75 | 17120 | 69.26 | 100 | 63.47 | 75.0 | 21.15 |
| 100 | 14620 | 59.14 | 100 | 79.57 | 100.0 | 30.85 |
| 125 | 12120 | 49.03 | 100 | 93.14 | 123.0 | 40.63 |
| 150 | 9630 | 38.96 | 98 | 104.19 | 123.0 | 49.98 |

### FD004

| RUL Cap | Obs Clipped | % Clipped | Engines Affected | Mean RUL | Median RUL | Std RUL |
|---------|-------------|-----------|------------------|----------|------------|---------|
| None | 0 | 0.0 | 0 | 133.31 | 122.0 | 89.78 |
| 50 | 48550 | 79.27 | 249 | 44.82 | 50.0 | 12.15 |
| 75 | 42325 | 69.1 | 249 | 63.41 | 75.0 | 21.19 |
| 100 | 36100 | 58.94 | 249 | 79.47 | 100.0 | 30.89 |
| 125 | 29875 | 48.78 | 249 | 92.99 | 122.0 | 40.67 |
| 150 | 23751 | 38.78 | 236 | 103.97 | 122.0 | 50.01 |

## 5. Engine-Level Train/Validation Split

**Method**: Randomly assign entire engines to train or validation. No engine in both.

| Subset | Train Engines | Val Engines | Train Rows | Val Rows | Val Fraction | No Overlap |
|--------|--------------|-------------|-----------|---------|-------------|-----------|
| FD001 | 80 | 20 | 16561 | 4070 | 0.2 | PASS |
| FD002 | 208 | 52 | 43464 | 10295 | 0.2 | PASS |
| FD003 | 80 | 20 | 20012 | 4708 | 0.2 | PASS |
| FD004 | 199 | 50 | 49294 | 11955 | 0.2008 | PASS |

### Leakage Verification

```
intersection(train_engine_ids, validation_engine_ids) = empty set
```
**Status**: VERIFIED for all subsets.

## 6. Operating-Condition Findings

- **FD001/FD003**: ~1 operating regime (op_settings near-constant)
- **FD002/FD004**: 6 operating regimes (significant op_setting variation)

### FD001: 1 regimes identified

| Regime | Observations | % |
|--------|-------------|---|
| regime_(0.0, 0.0, 100.0) | 16561 | 100.0% |

### FD002: 6 regimes identified

| Regime | Observations | % |
|--------|-------------|---|
| regime_(0.0, 0.0, 100.0) | 6507 | 14.97% |
| regime_(10.0, 0.0, 100.0) | 6572 | 15.12% |
| regime_(20.0, 1.0, 100.0) | 6535 | 15.04% |
| regime_(25.0, 1.0, 60.0) | 6450 | 14.84% |
| regime_(35.0, 1.0, 100.0) | 6543 | 15.05% |
| regime_(42.0, 1.0, 100.0) | 10857 | 24.98% |

### FD003: 1 regimes identified

| Regime | Observations | % |
|--------|-------------|---|
| regime_(0.0, 0.0, 100.0) | 20012 | 100.0% |

### FD004: 6 regimes identified

| Regime | Observations | % |
|--------|-------------|---|
| regime_(0.0, 0.0, 100.0) | 7447 | 15.11% |
| regime_(10.0, 0.0, 100.0) | 7385 | 14.98% |
| regime_(20.0, 1.0, 100.0) | 7307 | 14.82% |
| regime_(25.0, 1.0, 60.0) | 7371 | 14.95% |
| regime_(35.0, 1.0, 100.0) | 7375 | 14.96% |
| regime_(42.0, 1.0, 100.0) | 12409 | 25.17% |

## 7. Normalization Findings

### FD001

- Regimes: 1
- Normalization applied: Not needed (single regime)
- Note: Single operating regime; global normalization sufficient.

### FD002

- Regimes: 6
- Normalization applied: Yes
- Note: Condition-aware normalization reduces within-regime sensor variability.

### FD003

- Regimes: 1
- Normalization applied: Not needed (single regime)
- Note: Single operating regime; global normalization sufficient.

### FD004

- Regimes: 6
- Normalization applied: Yes
- Note: Condition-aware normalization reduces within-regime sensor variability.

> **Important**: These are exploratory findings only. Whether normalization improves
> RUL prediction accuracy will be evaluated in a later phase with model training.
