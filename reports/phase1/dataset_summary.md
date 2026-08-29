# AeroRUL — Phase 1 Dataset Summary Report

## 1. Dataset Overview

**Dataset**: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)

**Reference**: Saxena et al., PHM08 Conference

### Available Subsets

| Subset | Train | Test | RUL | Conditions | Fault Modes | Description |
|--------|-------|------|-----|------------|-------------|-------------|
| FD001 | Yes | Yes | Yes | 1 | 1 | ONE condition (Sea Level), ONE fault mode (HPC Degradation) |
| FD002 | Yes | Yes | Yes | 6 | 1 | SIX conditions, ONE fault mode (HPC Degradation) |
| FD003 | Yes | Yes | Yes | 1 | 2 | ONE condition (Sea Level), TWO fault modes (HPC + Fan Degradation) |
| FD004 | Yes | Yes | Yes | 6 | 2 | SIX conditions, TWO fault modes (HPC + Fan Degradation) |

### Schema

- **Total columns**: 26
- **ID columns**: unit_id, cycle
- **Operational settings**: op_setting_1, op_setting_2, op_setting_3
- **Sensors**: sensor_1 through sensor_21

## 2. Per-Subset Statistics

| Subset | Train Engines | Test Engines | Train Rows | Test Rows | Avg Train Length | Min | Max | Median |
|--------|--------------|-------------|-----------|----------|-----------------|-----|-----|--------|
| FD001 | 100 | 100 | 20631 | 13096 | 206.31 | 128 | 362 | 199.0 |
| FD002 | 260 | 259 | 53759 | 33991 | 206.77 | 128 | 378 | 199.0 |
| FD003 | 100 | 100 | 24720 | 16596 | 247.2 | 145 | 525 | 220.5 |
| FD004 | 249 | 248 | 61249 | 41214 | 245.98 | 128 | 543 | 234.0 |

### RUL Ground Truth Summary

| Subset | Engines | Min RUL | Max RUL | Mean RUL |
|--------|---------|---------|---------|----------|
| FD001 | 100 | 7 | 145 | 75.5 |
| FD002 | 259 | 6 | 194 | 81.2 |
| FD003 | 100 | 6 | 145 | 75.3 |
| FD004 | 248 | 6 | 195 | 86.5 |

## 3. Data Quality Findings

### FD001

- [OK] No missing values, no duplicates, cycles ordered, all start at 1

### FD002

- [OK] No missing values, no duplicates, cycles ordered, all start at 1

### FD003

- [OK] No missing values, no duplicates, cycles ordered, all start at 1

### FD004

- [OK] No missing values, no duplicates, cycles ordered, all start at 1


> **Overall**: All subsets are clean — no missing values, no duplicates, correct cycle ordering.

## 4. Sensor Findings

### Constant and Near-Constant Sensors

**FD001**:
- Constant sensors: sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19
- Near-constant sensors: sensor_2, sensor_6, sensor_8, sensor_13

**FD002**:
- No constant sensors
- No near-constant sensors

**FD003**:
- Constant sensors: sensor_1, sensor_5, sensor_16, sensor_18, sensor_19
- Near-constant sensors: sensor_2, sensor_6, sensor_8, sensor_13

**FD004**:
- No constant sensors
- No near-constant sensors

### Sensor–Cycle Correlation (Training Data)

| Sensor | FD001 | FD002 | FD003 | FD004 |
|--------|--------|--------|--------|--------|
| sensor_1 | N/A | -0.0010 | N/A | -0.0007 |
| sensor_2 | +0.5499 | +0.0038 | +0.3008 | +0.0025 |
| sensor_3 | +0.5439 | +0.0246 | +0.3735 | +0.0232 |
| sensor_4 | +0.6246 | +0.0371 | +0.3619 | +0.0293 |
| sensor_5 | -0.0000 | -0.0005 | -0.0000 | -0.0005 |
| sensor_6 | +0.1060 | -0.0007 | -0.4186 | -0.0013 |
| sensor_7 | -0.5959 | -0.0033 | +0.5310 | +0.0050 |
| sensor_8 | +0.4760 | -0.0040 | +0.4381 | -0.0011 |
| sensor_9 | +0.4440 | +0.0222 | +0.5078 | +0.0285 |
| sensor_10 | N/A | -0.0043 | +0.5282 | +0.0153 |
| sensor_11 | +0.6344 | +0.0422 | +0.4074 | +0.0369 |
| sensor_12 | -0.6114 | -0.0031 | +0.5382 | +0.0051 |
| sensor_13 | +0.4775 | -0.0043 | +0.4376 | -0.0009 |
| sensor_14 | +0.3703 | +0.0716 | +0.4886 | +0.0979 |
| sensor_15 | +0.5887 | +0.0351 | -0.4006 | -0.0278 |
| sensor_16 | -0.0000 | +0.0679 | +0.0000 | +0.0444 |
| sensor_17 | +0.5670 | +0.0251 | +0.3862 | +0.0229 |
| sensor_18 | N/A | -0.0046 | N/A | -0.0018 |
| sensor_19 | N/A | -0.0051 | N/A | -0.0018 |
| sensor_20 | -0.5836 | -0.0069 | +0.3046 | +0.0025 |
| sensor_21 | -0.5859 | -0.0068 | +0.3110 | +0.0025 |

## 5. Operating Condition Findings

The C-MAPSS subsets differ primarily in the number of operating conditions:

- **FD001/FD003**: Single operating condition (Sea Level) — op_settings are nearly constant
- **FD002/FD004**: Six operating conditions — op_settings show significant variation

This makes FD002/FD004 substantially harder because sensor readings are confounded by operating regime changes.

### FD001 Operating Conditions

| Setting | Unique | Mean | Std | Min | Max |
|---------|--------|------|-----|-----|-----|
| op_setting_1 | 158 | -0.0000 | 0.0022 | -0.0087 | 0.0087 |
| op_setting_2 | 13 | 0.0000 | 0.0003 | -0.0006 | 0.0006 |
| op_setting_3 | 1 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |

### FD002 Operating Conditions

| Setting | Unique | Mean | Std | Min | Max |
|---------|--------|------|-----|-----|-----|
| op_setting_1 | 536 | 23.9984 | 14.7474 | 0.0000 | 42.0080 |
| op_setting_2 | 105 | 0.5721 | 0.3100 | 0.0000 | 0.8420 |
| op_setting_3 | 2 | 94.0460 | 14.2377 | 60.0000 | 100.0000 |

### FD003 Operating Conditions

| Setting | Unique | Mean | Std | Min | Max |
|---------|--------|------|-----|-----|-----|
| op_setting_1 | 160 | -0.0000 | 0.0022 | -0.0086 | 0.0086 |
| op_setting_2 | 14 | 0.0000 | 0.0003 | -0.0006 | 0.0007 |
| op_setting_3 | 1 | 100.0000 | 0.0000 | 100.0000 | 100.0000 |

### FD004 Operating Conditions

| Setting | Unique | Mean | Std | Min | Max |
|---------|--------|------|-----|-----|-----|
| op_setting_1 | 536 | 23.9998 | 14.7807 | 0.0000 | 42.0080 |
| op_setting_2 | 105 | 0.5713 | 0.3107 | 0.0000 | 0.8420 |
| op_setting_3 | 2 | 94.0316 | 14.2520 | 60.0000 | 100.0000 |
