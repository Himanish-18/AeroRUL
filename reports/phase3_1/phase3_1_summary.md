# AeroRUL — Phase 3.1 Feature Quality Control Report

## 1. Executive Summary

**Phase 3.1 Status: PASS**

All quality control checks passed. The feature pipeline is 
mathematically valid, leakage-safe, schema-consistent, and 
ready for Phase 4 classical ML benchmarking.

## 2. NaN Analysis

### FD001

- **NaN-affected features**: 21
- EXPECTED_NUMERICAL_EDGE_CASE: 21 features

| Feature Family | Total NaN (Train) |
|----------------|-------------------|
| r2 | 202760 |

**Actions taken:**
- CONVERT_TO_VALUE: 21 features

### FD002

- **NaN-affected features**: 12
- EXPECTED_NUMERICAL_EDGE_CASE: 12 features

| Feature Family | Total NaN (Train) |
|----------------|-------------------|
| r2 | 153900 |

**Actions taken:**
- CONVERT_TO_VALUE: 12 features

### FD003

- **NaN-affected features**: 25
- EXPECTED_NUMERICAL_EDGE_CASE: 25 features

| Feature Family | Total NaN (Train) |
|----------------|-------------------|
| r2 | 250703 |

**Actions taken:**
- CONVERT_TO_VALUE: 25 features

### FD004

- **NaN-affected features**: 12
- EXPECTED_NUMERICAL_EDGE_CASE: 12 features

| Feature Family | Total NaN (Train) |
|----------------|-------------------|
| r2 | 182853 |

**Actions taken:**
- CONVERT_TO_VALUE: 12 features

## 3. Constant Feature Analysis

### FD001

- **Total constant features**: 164
- ENGINEERED_CONSTANT: 149 features
- DATASET_SPECIFIC_CONSTANT: 15 features
- Constant in both: 149
- Train-only constant: 0
- Val-only constant: 15

### FD002

- **Total constant features**: 55
- ENGINEERED_CONSTANT: 53 features
- DATASET_SPECIFIC_CONSTANT: 2 features
- Constant in both: 52
- Train-only constant: 0
- Val-only constant: 3

### FD003

- **Total constant features**: 134
- ENGINEERED_CONSTANT: 134 features
- Constant in both: 134
- Train-only constant: 0
- Val-only constant: 0

### FD004

- **Total constant features**: 62
- ENGINEERED_CONSTANT: 61 features
- DATASET_SPECIFIC_CONSTANT: 1 features
- Constant in both: 60
- Train-only constant: 0
- Val-only constant: 2

## 4. Redundancy Analysis

### FD001: 205 duplicate groups found

- Group 0: sensor_1_min_10, sensor_1_mean_10
- Group 1: sensor_1_max_10, sensor_1_mean_10
- Group 2: sensor_1_range_10, sensor_1_std_10
- Group 3: sensor_5_std_10, sensor_1_std_10
- Group 4: sensor_5_min_10, sensor_5_mean_10
- Group 5: sensor_5_max_10, sensor_5_mean_10
- Group 6: sensor_5_range_10, sensor_1_std_10
- Group 7: sensor_10_std_10, sensor_1_std_10
- Group 8: sensor_10_min_10, sensor_10_mean_10
- Group 9: sensor_10_max_10, sensor_10_mean_10
- Group 10: sensor_10_range_10, sensor_1_std_10
- Group 11: sensor_16_std_10, sensor_1_std_10
- Group 12: sensor_16_min_10, sensor_16_mean_10
- Group 13: sensor_16_max_10, sensor_16_mean_10
- Group 14: sensor_16_range_10, sensor_1_std_10
- Group 15: sensor_18_std_10, sensor_1_std_10
- Group 16: sensor_18_min_10, sensor_18_mean_10
- Group 17: sensor_18_max_10, sensor_18_mean_10
- Group 18: sensor_18_range_10, sensor_1_std_10
- Group 19: sensor_19_std_10, sensor_1_std_10
- Group 20: sensor_19_min_10, sensor_19_mean_10
- Group 21: sensor_19_max_10, sensor_19_mean_10
- Group 22: sensor_19_range_10, sensor_1_std_10
- Group 23: sensor_1_r2_10, sensor_1_std_10
- Group 24: sensor_10_slope_10, sensor_1_std_10
- Group 25: sensor_16_slope_10, sensor_1_std_10
- Group 26: sensor_18_slope_10, sensor_1_std_10
- Group 27: sensor_19_slope_10, sensor_1_std_10
- Group 28: sensor_1_mean_20, sensor_1_mean_10
- Group 29: sensor_1_std_20, sensor_1_std_10
- Group 30: sensor_1_min_20, sensor_1_mean_10
- Group 31: sensor_1_max_20, sensor_1_mean_10
- Group 32: sensor_1_range_20, sensor_1_std_10
- Group 33: sensor_5_mean_20, sensor_5_mean_10
- Group 34: sensor_5_std_20, sensor_1_std_10
- Group 35: sensor_5_min_20, sensor_5_mean_10
- Group 36: sensor_5_max_20, sensor_5_mean_10
- Group 37: sensor_5_range_20, sensor_1_std_10
- Group 38: sensor_6_max_20, sensor_6_max_10
- Group 39: sensor_10_mean_20, sensor_10_mean_10
- Group 40: sensor_10_std_20, sensor_1_std_10
- Group 41: sensor_10_min_20, sensor_10_mean_10
- Group 42: sensor_10_max_20, sensor_10_mean_10
- Group 43: sensor_10_range_20, sensor_1_std_10
- Group 44: sensor_16_mean_20, sensor_16_mean_10
- Group 45: sensor_16_std_20, sensor_1_std_10
- Group 46: sensor_16_min_20, sensor_16_mean_10
- Group 47: sensor_16_max_20, sensor_16_mean_10
- Group 48: sensor_16_range_20, sensor_1_std_10
- Group 49: sensor_18_mean_20, sensor_18_mean_10
- Group 50: sensor_18_std_20, sensor_1_std_10
- Group 51: sensor_18_min_20, sensor_18_mean_10
- Group 52: sensor_18_max_20, sensor_18_mean_10
- Group 53: sensor_18_range_20, sensor_1_std_10
- Group 54: sensor_19_mean_20, sensor_19_mean_10
- Group 55: sensor_19_std_20, sensor_1_std_10
- Group 56: sensor_19_min_20, sensor_19_mean_10
- Group 57: sensor_19_max_20, sensor_19_mean_10
- Group 58: sensor_19_range_20, sensor_1_std_10
- Group 59: sensor_1_r2_20, sensor_1_std_10
- Group 60: sensor_10_slope_20, sensor_1_std_10
- Group 61: sensor_16_slope_20, sensor_1_std_10
- Group 62: sensor_18_slope_20, sensor_1_std_10
- Group 63: sensor_19_slope_20, sensor_1_std_10
- Group 64: sensor_1_mean_30, sensor_1_mean_10
- Group 65: sensor_1_std_30, sensor_1_std_10
- Group 66: sensor_1_min_30, sensor_1_mean_10
- Group 67: sensor_1_max_30, sensor_1_mean_10
- Group 68: sensor_1_range_30, sensor_1_std_10
- Group 69: sensor_5_mean_30, sensor_5_mean_10
- Group 70: sensor_5_std_30, sensor_1_std_10
- Group 71: sensor_5_min_30, sensor_5_mean_10
- Group 72: sensor_5_max_30, sensor_5_mean_10
- Group 73: sensor_5_range_30, sensor_1_std_10
- Group 74: sensor_6_max_30, sensor_6_max_10
- Group 75: sensor_10_mean_30, sensor_10_mean_10
- Group 76: sensor_10_std_30, sensor_1_std_10
- Group 77: sensor_10_min_30, sensor_10_mean_10
- Group 78: sensor_10_max_30, sensor_10_mean_10
- Group 79: sensor_10_range_30, sensor_1_std_10
- Group 80: sensor_16_mean_30, sensor_16_mean_10
- Group 81: sensor_16_std_30, sensor_1_std_10
- Group 82: sensor_16_min_30, sensor_16_mean_10
- Group 83: sensor_16_max_30, sensor_16_mean_10
- Group 84: sensor_16_range_30, sensor_1_std_10
- Group 85: sensor_18_mean_30, sensor_18_mean_10
- Group 86: sensor_18_std_30, sensor_1_std_10
- Group 87: sensor_18_min_30, sensor_18_mean_10
- Group 88: sensor_18_max_30, sensor_18_mean_10
- Group 89: sensor_18_range_30, sensor_1_std_10
- Group 90: sensor_19_mean_30, sensor_19_mean_10
- Group 91: sensor_19_std_30, sensor_1_std_10
- Group 92: sensor_19_min_30, sensor_19_mean_10
- Group 93: sensor_19_max_30, sensor_19_mean_10
- Group 94: sensor_19_range_30, sensor_1_std_10
- Group 95: sensor_5_slope_30, sensor_1_std_10
- Group 96: sensor_10_slope_30, sensor_1_std_10
- Group 97: sensor_16_slope_30, sensor_1_std_10
- Group 98: sensor_18_slope_30, sensor_1_std_10
- Group 99: sensor_19_slope_30, sensor_1_std_10
- Group 100: sensor_1_mean_50, sensor_1_mean_10
- Group 101: sensor_1_std_50, sensor_1_std_10
- Group 102: sensor_1_min_50, sensor_1_mean_10
- Group 103: sensor_1_max_50, sensor_1_mean_10
- Group 104: sensor_1_range_50, sensor_1_std_10
- Group 105: sensor_5_mean_50, sensor_5_mean_10
- Group 106: sensor_5_std_50, sensor_1_std_10
- Group 107: sensor_5_min_50, sensor_5_mean_10
- Group 108: sensor_5_max_50, sensor_5_mean_10
- Group 109: sensor_5_range_50, sensor_1_std_10
- Group 110: sensor_6_max_50, sensor_6_max_10
- Group 111: sensor_10_mean_50, sensor_10_mean_10
- Group 112: sensor_10_std_50, sensor_1_std_10
- Group 113: sensor_10_min_50, sensor_10_mean_10
- Group 114: sensor_10_max_50, sensor_10_mean_10
- Group 115: sensor_10_range_50, sensor_1_std_10
- Group 116: sensor_16_mean_50, sensor_16_mean_10
- Group 117: sensor_16_std_50, sensor_1_std_10
- Group 118: sensor_16_min_50, sensor_16_mean_10
- Group 119: sensor_16_max_50, sensor_16_mean_10
- Group 120: sensor_16_range_50, sensor_1_std_10
- Group 121: sensor_18_mean_50, sensor_18_mean_10
- Group 122: sensor_18_std_50, sensor_1_std_10
- Group 123: sensor_18_min_50, sensor_18_mean_10
- Group 124: sensor_18_max_50, sensor_18_mean_10
- Group 125: sensor_18_range_50, sensor_1_std_10
- Group 126: sensor_19_mean_50, sensor_19_mean_10
- Group 127: sensor_19_std_50, sensor_1_std_10
- Group 128: sensor_19_min_50, sensor_19_mean_10
- Group 129: sensor_19_max_50, sensor_19_mean_10
- Group 130: sensor_19_range_50, sensor_1_std_10
- Group 131: sensor_5_slope_50, sensor_1_std_10
- Group 132: sensor_10_slope_50, sensor_1_std_10
- Group 133: sensor_16_slope_50, sensor_1_std_10
- Group 134: sensor_18_slope_50, sensor_1_std_10
- Group 135: sensor_19_slope_50, sensor_1_std_10
- Group 136: sensor_1_mean_75, sensor_1_mean_10
- Group 137: sensor_1_std_75, sensor_1_std_10
- Group 138: sensor_1_min_75, sensor_1_mean_10
- Group 139: sensor_1_max_75, sensor_1_mean_10
- Group 140: sensor_1_range_75, sensor_1_std_10
- Group 141: sensor_5_mean_75, sensor_5_mean_10
- Group 142: sensor_5_std_75, sensor_1_std_10
- Group 143: sensor_5_min_75, sensor_5_mean_10
- Group 144: sensor_5_max_75, sensor_5_mean_10
- Group 145: sensor_5_range_75, sensor_1_std_10
- Group 146: sensor_6_max_75, sensor_6_max_10
- Group 147: sensor_10_mean_75, sensor_10_mean_10
- Group 148: sensor_10_std_75, sensor_1_std_10
- Group 149: sensor_10_min_75, sensor_10_mean_10
- Group 150: sensor_10_max_75, sensor_10_mean_10
- Group 151: sensor_10_range_75, sensor_1_std_10
- Group 152: sensor_16_mean_75, sensor_16_mean_10
- Group 153: sensor_16_std_75, sensor_1_std_10
- Group 154: sensor_16_min_75, sensor_16_mean_10
- Group 155: sensor_16_max_75, sensor_16_mean_10
- Group 156: sensor_16_range_75, sensor_1_std_10
- Group 157: sensor_18_mean_75, sensor_18_mean_10
- Group 158: sensor_18_std_75, sensor_1_std_10
- Group 159: sensor_18_min_75, sensor_18_mean_10
- Group 160: sensor_18_max_75, sensor_18_mean_10
- Group 161: sensor_18_range_75, sensor_1_std_10
- Group 162: sensor_19_mean_75, sensor_19_mean_10
- Group 163: sensor_19_std_75, sensor_1_std_10
- Group 164: sensor_19_min_75, sensor_19_mean_10
- Group 165: sensor_19_max_75, sensor_19_mean_10
- Group 166: sensor_19_range_75, sensor_1_std_10
- Group 167: sensor_1_slope_75, sensor_1_std_10
- Group 168: sensor_1_r2_75, sensor_1_std_10
- Group 169: sensor_5_slope_75, sensor_1_std_10
- Group 170: sensor_5_r2_75, sensor_1_std_10
- Group 171: sensor_10_slope_75, sensor_1_std_10
- Group 172: sensor_16_slope_75, sensor_1_std_10
- Group 173: sensor_18_slope_75, sensor_1_std_10
- Group 174: sensor_19_slope_75, sensor_1_std_10
- Group 175: sensor_1_delta_1, sensor_1_std_10
- Group 176: sensor_1_delta_5, sensor_1_std_10
- Group 177: sensor_1_delta_10, sensor_1_std_10
- Group 178: sensor_5_delta_1, sensor_1_std_10
- Group 179: sensor_5_delta_5, sensor_1_std_10
- Group 180: sensor_5_delta_10, sensor_1_std_10
- Group 181: sensor_10_delta_1, sensor_1_std_10
- Group 182: sensor_10_delta_5, sensor_1_std_10
- Group 183: sensor_10_delta_10, sensor_1_std_10
- Group 184: sensor_16_delta_1, sensor_1_std_10
- Group 185: sensor_16_delta_5, sensor_1_std_10
- Group 186: sensor_16_delta_10, sensor_1_std_10
- Group 187: sensor_18_delta_1, sensor_1_std_10
- Group 188: sensor_18_delta_5, sensor_1_std_10
- Group 189: sensor_18_delta_10, sensor_1_std_10
- Group 190: sensor_19_delta_1, sensor_1_std_10
- Group 191: sensor_19_delta_5, sensor_1_std_10
- Group 192: sensor_19_delta_10, sensor_1_std_10
- Group 193: sensor_1_relchange_5, sensor_1_std_10
- Group 194: sensor_1_relchange_10, sensor_1_std_10
- Group 195: sensor_5_relchange_5, sensor_1_std_10
- Group 196: sensor_5_relchange_10, sensor_1_std_10
- Group 197: sensor_10_relchange_5, sensor_1_std_10
- Group 198: sensor_10_relchange_10, sensor_1_std_10
- Group 199: sensor_16_relchange_5, sensor_1_std_10
- Group 200: sensor_16_relchange_10, sensor_1_std_10
- Group 201: sensor_18_relchange_5, sensor_1_std_10
- Group 202: sensor_18_relchange_10, sensor_1_std_10
- Group 203: sensor_19_relchange_5, sensor_1_std_10
- Group 204: sensor_19_relchange_10, sensor_1_std_10

### FD002: 43 duplicate groups found

- Group 0: sensor_18_std_10, sensor_18_mean_10
- Group 1: sensor_18_min_10, sensor_18_mean_10
- Group 2: sensor_18_max_10, sensor_18_mean_10
- Group 3: sensor_18_range_10, sensor_18_mean_10
- Group 4: sensor_19_max_10, sensor_18_mean_10
- Group 5: sensor_18_slope_10, sensor_18_mean_10
- Group 6: sensor_18_mean_20, sensor_18_mean_10
- Group 7: sensor_18_std_20, sensor_18_mean_10
- Group 8: sensor_18_min_20, sensor_18_mean_10
- Group 9: sensor_18_max_20, sensor_18_mean_10
- Group 10: sensor_18_range_20, sensor_18_mean_10
- Group 11: sensor_19_max_20, sensor_18_mean_10
- Group 12: sensor_18_slope_20, sensor_18_mean_10
- Group 13: sensor_18_mean_30, sensor_18_mean_10
- Group 14: sensor_18_std_30, sensor_18_mean_10
- Group 15: sensor_18_min_30, sensor_18_mean_10
- Group 16: sensor_18_max_30, sensor_18_mean_10
- Group 17: sensor_18_range_30, sensor_18_mean_10
- Group 18: sensor_19_max_30, sensor_18_mean_10
- Group 19: sensor_18_slope_30, sensor_18_mean_10
- Group 20: sensor_1_max_50, sensor_1_max_30
- Group 21: sensor_18_mean_50, sensor_18_mean_10
- Group 22: sensor_18_std_50, sensor_18_mean_10
- Group 23: sensor_18_min_50, sensor_18_mean_10
- Group 24: sensor_18_max_50, sensor_18_mean_10
- Group 25: sensor_18_range_50, sensor_18_mean_10
- Group 26: sensor_19_max_50, sensor_18_mean_10
- Group 27: sensor_18_slope_50, sensor_18_mean_10
- Group 28: sensor_1_max_75, sensor_1_max_30
- Group 29: sensor_18_mean_75, sensor_18_mean_10
- Group 30: sensor_18_std_75, sensor_18_mean_10
- Group 31: sensor_18_min_75, sensor_18_mean_10
- Group 32: sensor_18_max_75, sensor_18_mean_10
- Group 33: sensor_18_range_75, sensor_18_mean_10
- Group 34: sensor_19_min_75, sensor_19_min_50
- Group 35: sensor_19_max_75, sensor_18_mean_10
- Group 36: sensor_19_range_75, sensor_19_range_50
- Group 37: sensor_18_slope_75, sensor_18_mean_10
- Group 38: sensor_18_delta_1, sensor_18_mean_10
- Group 39: sensor_18_delta_5, sensor_18_mean_10
- Group 40: sensor_18_delta_10, sensor_18_mean_10
- Group 41: sensor_18_relchange_5, sensor_18_mean_10
- Group 42: sensor_18_relchange_10, sensor_18_mean_10

### FD003: 156 duplicate groups found

- Group 0: sensor_1_min_10, sensor_1_mean_10
- Group 1: sensor_1_max_10, sensor_1_mean_10
- Group 2: sensor_1_range_10, sensor_1_std_10
- Group 3: sensor_5_std_10, sensor_1_std_10
- Group 4: sensor_5_min_10, sensor_5_mean_10
- Group 5: sensor_5_max_10, sensor_5_mean_10
- Group 6: sensor_5_range_10, sensor_1_std_10
- Group 7: sensor_16_min_10, sensor_16_mean_10
- Group 8: sensor_16_max_10, sensor_16_mean_10
- Group 9: sensor_16_range_10, sensor_16_std_10
- Group 10: sensor_18_std_10, sensor_16_std_10
- Group 11: sensor_18_min_10, sensor_18_mean_10
- Group 12: sensor_18_max_10, sensor_18_mean_10
- Group 13: sensor_18_range_10, sensor_16_std_10
- Group 14: sensor_19_std_10, sensor_16_std_10
- Group 15: sensor_19_min_10, sensor_19_mean_10
- Group 16: sensor_19_max_10, sensor_19_mean_10
- Group 17: sensor_19_range_10, sensor_16_std_10
- Group 18: sensor_1_r2_10, sensor_16_std_10
- Group 19: sensor_18_slope_10, sensor_16_slope_10
- Group 20: sensor_19_slope_10, sensor_16_slope_10
- Group 21: sensor_1_mean_20, sensor_1_mean_10
- Group 22: sensor_1_std_20, sensor_16_slope_10
- Group 23: sensor_1_min_20, sensor_1_mean_10
- Group 24: sensor_1_max_20, sensor_1_mean_10
- Group 25: sensor_1_range_20, sensor_16_slope_10
- Group 26: sensor_5_mean_20, sensor_5_mean_10
- Group 27: sensor_5_std_20, sensor_16_slope_10
- Group 28: sensor_5_min_20, sensor_5_mean_10
- Group 29: sensor_5_max_20, sensor_5_mean_10
- Group 30: sensor_5_range_20, sensor_16_slope_10
- Group 31: sensor_16_mean_20, sensor_16_mean_10
- Group 32: sensor_16_min_20, sensor_16_mean_10
- Group 33: sensor_16_max_20, sensor_16_mean_10
- Group 34: sensor_16_range_20, sensor_16_std_20
- Group 35: sensor_18_mean_20, sensor_18_mean_10
- Group 36: sensor_18_std_20, sensor_16_std_20
- Group 37: sensor_18_min_20, sensor_18_mean_10
- Group 38: sensor_18_max_20, sensor_18_mean_10
- Group 39: sensor_18_range_20, sensor_16_std_20
- Group 40: sensor_19_mean_20, sensor_19_mean_10
- Group 41: sensor_19_std_20, sensor_16_std_20
- Group 42: sensor_19_min_20, sensor_19_mean_10
- Group 43: sensor_19_max_20, sensor_19_mean_10
- Group 44: sensor_19_range_20, sensor_16_std_20
- Group 45: sensor_1_r2_20, sensor_16_std_20
- Group 46: sensor_5_slope_20, sensor_16_std_20
- Group 47: sensor_18_slope_20, sensor_16_slope_20
- Group 48: sensor_19_slope_20, sensor_16_slope_20
- Group 49: sensor_1_mean_30, sensor_1_mean_10
- Group 50: sensor_1_std_30, sensor_16_slope_20
- Group 51: sensor_1_min_30, sensor_1_mean_10
- Group 52: sensor_1_max_30, sensor_1_mean_10
- Group 53: sensor_1_range_30, sensor_16_slope_20
- Group 54: sensor_5_mean_30, sensor_5_mean_10
- Group 55: sensor_5_std_30, sensor_16_slope_20
- Group 56: sensor_5_min_30, sensor_5_mean_10
- Group 57: sensor_5_max_30, sensor_5_mean_10
- Group 58: sensor_5_range_30, sensor_16_slope_20
- Group 59: sensor_16_mean_30, sensor_16_mean_10
- Group 60: sensor_16_min_30, sensor_16_mean_10
- Group 61: sensor_16_max_30, sensor_16_mean_10
- Group 62: sensor_16_range_30, sensor_16_std_30
- Group 63: sensor_18_mean_30, sensor_18_mean_10
- Group 64: sensor_18_std_30, sensor_16_std_30
- Group 65: sensor_18_min_30, sensor_18_mean_10
- Group 66: sensor_18_max_30, sensor_18_mean_10
- Group 67: sensor_18_range_30, sensor_16_std_30
- Group 68: sensor_19_mean_30, sensor_19_mean_10
- Group 69: sensor_19_std_30, sensor_16_std_30
- Group 70: sensor_19_min_30, sensor_19_mean_10
- Group 71: sensor_19_max_30, sensor_19_mean_10
- Group 72: sensor_19_range_30, sensor_16_std_30
- Group 73: sensor_5_slope_30, sensor_16_std_30
- Group 74: sensor_18_slope_30, sensor_16_slope_30
- Group 75: sensor_19_slope_30, sensor_16_slope_30
- Group 76: sensor_1_mean_50, sensor_1_mean_10
- Group 77: sensor_1_std_50, sensor_16_slope_30
- Group 78: sensor_1_min_50, sensor_1_mean_10
- Group 79: sensor_1_max_50, sensor_1_mean_10
- Group 80: sensor_1_range_50, sensor_16_slope_30
- Group 81: sensor_5_mean_50, sensor_5_mean_10
- Group 82: sensor_5_std_50, sensor_16_slope_30
- Group 83: sensor_5_min_50, sensor_5_mean_10
- Group 84: sensor_5_max_50, sensor_5_mean_10
- Group 85: sensor_5_range_50, sensor_16_slope_30
- Group 86: sensor_16_mean_50, sensor_16_mean_10
- Group 87: sensor_16_min_50, sensor_16_mean_10
- Group 88: sensor_16_max_50, sensor_16_mean_10
- Group 89: sensor_16_range_50, sensor_16_std_50
- Group 90: sensor_18_mean_50, sensor_18_mean_10
- Group 91: sensor_18_std_50, sensor_16_std_50
- Group 92: sensor_18_min_50, sensor_18_mean_10
- Group 93: sensor_18_max_50, sensor_18_mean_10
- Group 94: sensor_18_range_50, sensor_16_std_50
- Group 95: sensor_19_mean_50, sensor_19_mean_10
- Group 96: sensor_19_std_50, sensor_16_std_50
- Group 97: sensor_19_min_50, sensor_19_mean_10
- Group 98: sensor_19_max_50, sensor_19_mean_10
- Group 99: sensor_19_range_50, sensor_16_std_50
- Group 100: sensor_5_slope_50, sensor_16_std_50
- Group 101: sensor_18_slope_50, sensor_16_slope_50
- Group 102: sensor_19_slope_50, sensor_16_slope_50
- Group 103: sensor_1_mean_75, sensor_1_mean_10
- Group 104: sensor_1_std_75, sensor_16_slope_50
- Group 105: sensor_1_min_75, sensor_1_mean_10
- Group 106: sensor_1_max_75, sensor_1_mean_10
- Group 107: sensor_1_range_75, sensor_16_slope_50
- Group 108: sensor_5_mean_75, sensor_5_mean_10
- Group 109: sensor_5_std_75, sensor_16_slope_50
- Group 110: sensor_5_min_75, sensor_5_mean_10
- Group 111: sensor_5_max_75, sensor_5_mean_10
- Group 112: sensor_5_range_75, sensor_16_slope_50
- Group 113: sensor_16_mean_75, sensor_16_mean_10
- Group 114: sensor_16_min_75, sensor_16_mean_10
- Group 115: sensor_16_max_75, sensor_16_mean_10
- Group 116: sensor_16_range_75, sensor_16_std_75
- Group 117: sensor_18_mean_75, sensor_18_mean_10
- Group 118: sensor_18_std_75, sensor_16_std_75
- Group 119: sensor_18_min_75, sensor_18_mean_10
- Group 120: sensor_18_max_75, sensor_18_mean_10
- Group 121: sensor_18_range_75, sensor_16_std_75
- Group 122: sensor_19_mean_75, sensor_19_mean_10
- Group 123: sensor_19_std_75, sensor_16_std_75
- Group 124: sensor_19_min_75, sensor_19_mean_10
- Group 125: sensor_19_max_75, sensor_19_mean_10
- Group 126: sensor_19_range_75, sensor_16_std_75
- Group 127: sensor_1_slope_75, sensor_16_std_75
- Group 128: sensor_1_r2_75, sensor_16_std_75
- Group 129: sensor_5_slope_75, sensor_16_std_75
- Group 130: sensor_5_r2_75, sensor_16_std_75
- Group 131: sensor_18_slope_75, sensor_16_slope_75
- Group 132: sensor_19_slope_75, sensor_16_slope_75
- Group 133: sensor_1_delta_1, sensor_16_slope_75
- Group 134: sensor_1_delta_5, sensor_16_slope_75
- Group 135: sensor_1_delta_10, sensor_16_slope_75
- Group 136: sensor_5_delta_1, sensor_16_slope_75
- Group 137: sensor_5_delta_5, sensor_16_slope_75
- Group 138: sensor_5_delta_10, sensor_16_slope_75
- Group 139: sensor_16_delta_5, sensor_16_delta_1
- Group 140: sensor_16_delta_10, sensor_16_delta_1
- Group 141: sensor_18_delta_1, sensor_16_delta_1
- Group 142: sensor_18_delta_5, sensor_16_delta_1
- Group 143: sensor_18_delta_10, sensor_16_delta_1
- Group 144: sensor_19_delta_1, sensor_16_delta_1
- Group 145: sensor_19_delta_5, sensor_16_delta_1
- Group 146: sensor_19_delta_10, sensor_16_delta_1
- Group 147: sensor_1_relchange_5, sensor_16_delta_1
- Group 148: sensor_1_relchange_10, sensor_16_delta_1
- Group 149: sensor_5_relchange_5, sensor_16_delta_1
- Group 150: sensor_5_relchange_10, sensor_16_delta_1
- Group 151: sensor_16_relchange_10, sensor_16_relchange_5
- Group 152: sensor_18_relchange_5, sensor_16_relchange_5
- Group 153: sensor_18_relchange_10, sensor_16_relchange_5
- Group 154: sensor_19_relchange_5, sensor_16_relchange_5
- Group 155: sensor_19_relchange_10, sensor_16_relchange_5

### FD004: 51 duplicate groups found

- Group 0: sensor_18_std_10, sensor_18_mean_10
- Group 1: sensor_18_min_10, sensor_18_mean_10
- Group 2: sensor_18_max_10, sensor_18_mean_10
- Group 3: sensor_18_range_10, sensor_18_mean_10
- Group 4: sensor_19_max_10, sensor_18_mean_10
- Group 5: sensor_18_slope_10, sensor_18_mean_10
- Group 6: sensor_5_max_20, sensor_5_max_10
- Group 7: sensor_18_mean_20, sensor_18_mean_10
- Group 8: sensor_18_std_20, sensor_18_mean_10
- Group 9: sensor_18_min_20, sensor_18_mean_10
- Group 10: sensor_18_max_20, sensor_18_mean_10
- Group 11: sensor_18_range_20, sensor_18_mean_10
- Group 12: sensor_19_max_20, sensor_18_mean_10
- Group 13: sensor_18_slope_20, sensor_18_mean_10
- Group 14: sensor_5_max_30, sensor_5_max_10
- Group 15: sensor_18_mean_30, sensor_18_mean_10
- Group 16: sensor_18_std_30, sensor_18_mean_10
- Group 17: sensor_18_min_30, sensor_18_mean_10
- Group 18: sensor_18_max_30, sensor_18_mean_10
- Group 19: sensor_18_range_30, sensor_18_mean_10
- Group 20: sensor_19_max_30, sensor_18_mean_10
- Group 21: sensor_18_slope_30, sensor_18_mean_10
- Group 22: sensor_1_min_50, sensor_1_min_30
- Group 23: sensor_5_min_50, sensor_5_min_30
- Group 24: sensor_5_max_50, sensor_5_max_10
- Group 25: sensor_5_range_50, sensor_5_range_30
- Group 26: sensor_18_mean_50, sensor_18_mean_10
- Group 27: sensor_18_std_50, sensor_18_mean_10
- Group 28: sensor_18_min_50, sensor_18_mean_10
- Group 29: sensor_18_max_50, sensor_18_mean_10
- Group 30: sensor_18_range_50, sensor_18_mean_10
- Group 31: sensor_19_max_50, sensor_18_mean_10
- Group 32: sensor_18_slope_50, sensor_18_mean_10
- Group 33: sensor_1_min_75, sensor_1_min_30
- Group 34: sensor_5_min_75, sensor_5_min_30
- Group 35: sensor_5_max_75, sensor_5_max_10
- Group 36: sensor_5_range_75, sensor_5_range_30
- Group 37: sensor_18_mean_75, sensor_18_mean_10
- Group 38: sensor_18_std_75, sensor_18_mean_10
- Group 39: sensor_18_min_75, sensor_18_mean_10
- Group 40: sensor_18_max_75, sensor_18_mean_10
- Group 41: sensor_18_range_75, sensor_18_mean_10
- Group 42: sensor_19_min_75, sensor_19_min_50
- Group 43: sensor_19_max_75, sensor_18_mean_10
- Group 44: sensor_19_range_75, sensor_19_range_50
- Group 45: sensor_18_slope_75, sensor_18_mean_10
- Group 46: sensor_18_delta_1, sensor_18_mean_10
- Group 47: sensor_18_delta_5, sensor_18_mean_10
- Group 48: sensor_18_delta_10, sensor_18_mean_10
- Group 49: sensor_18_relchange_5, sensor_18_mean_10
- Group 50: sensor_18_relchange_10, sensor_18_mean_10

## 5. Schema Validation

| Subset | Same Columns | Same Count | Same Order | Forbidden | Status |
|--------|-------------|------------|------------|-----------|--------|
| FD001 | ✓ | ✓ | ✓ | None | PASS |
| FD002 | ✓ | ✓ | ✓ | None | PASS |
| FD003 | ✓ | ✓ | ✓ | None | PASS |
| FD004 | ✓ | ✓ | ✓ | None | PASS |

## 6. Leakage Validation

### FD001

- **Temporal Causality**: PASS
- **Engine Separation**: PASS (train=80, val=20)
- **Target Isolation**: PASS

### FD002

- **Temporal Causality**: PASS
- **Engine Separation**: PASS (train=208, val=52)
- **Target Isolation**: PASS
- **Normalization Isolation**: PASS

### FD003

- **Temporal Causality**: PASS
- **Engine Separation**: PASS (train=80, val=20)
- **Target Isolation**: PASS

### FD004

- **Temporal Causality**: PASS
- **Engine Separation**: PASS (train=199, val=50)
- **Target Isolation**: PASS
- **Normalization Isolation**: PASS

## 7. Distribution Shift Summary

Threshold for shift flag: |Cohen's d| > 0.5

| Subset | Total Features | Shifted | Not Shifted | Shift % |
|--------|---------------|---------|-------------|---------|
| FD001 | 823 | 20 | 803 | 2.4% |
| FD002 | 835 | 0 | 835 | 0.0% |
| FD003 | 824 | 0 | 824 | 0.0% |
| FD004 | 835 | 0 | 835 | 0.0% |

## 8. Final Phase 4 Readiness

### FD001
- Total features: 840
- YES (ready): 670
- NO (excluded): 149
- REQUIRES_FIX: 21

### FD002
- Total features: 840
- YES (ready): 776
- NO (excluded): 52
- REQUIRES_FIX: 12

### FD003
- Total features: 840
- YES (ready): 681
- NO (excluded): 134
- REQUIRES_FIX: 25

### FD004
- Total features: 840
- YES (ready): 768
- NO (excluded): 60
- REQUIRES_FIX: 12

**Status: READY for Phase 4 — Classical Regression Benchmarking**
