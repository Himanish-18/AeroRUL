# AeroRUL — Predictive Maintenance for Turbofan Engines

Estimate the **Remaining Useful Life (RUL)** of turbofan aircraft engines from multivariate sensor degradation patterns using the NASA C-MAPSS dataset.

## Dataset

**NASA C-MAPSS** (Commercial Modular Aero-Propulsion System Simulation)

Reference: Saxena et al., *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM08

### C-MAPSS Subsets

| Subset | Conditions | Fault Modes | Train Engines | Test Engines |
|--------|-----------|-------------|---------------|-------------|
| FD001  | 1 (Sea Level) | 1 (HPC) | 100 | 100 |
| FD002  | 6 | 1 (HPC) | 260 | 259 |
| FD003  | 1 (Sea Level) | 2 (HPC+Fan) | 100 | 100 |
| FD004  | 6 | 2 (HPC+Fan) | 249 | 248 |

### Data Structure

Each file contains 26 space-separated columns:
- `unit_id` — Engine identifier
- `cycle` — Operating cycle number
- `op_setting_1..3` — Three operational settings
- `sensor_1..21` — Twenty-one sensor measurements

---

## Phase 1 — Dataset Understanding

**Status**: COMPLETE

Phase 1 performed comprehensive dataset understanding, validation, and exploratory analysis.

### Key Findings

1. **All four subsets available** with complete train/test/RUL files
2. **Data quality is excellent**: Zero missing values, zero duplicates, all cycles ordered and starting at 1
3. **Total observations**: 160,359 training + 104,897 test rows
4. **Constant sensors in FD001/FD003**: sensor_1, sensor_5, sensor_10 (FD001 only), sensor_16, sensor_18, sensor_19
5. **Near-constant sensors in FD001/FD003**: sensor_2, sensor_6, sensor_8, sensor_13
6. **FD002/FD004 are harder** due to 6 operating conditions confounding sensor readings

### How to Run Phase 1 Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full analysis pipeline
python -m src.run_phase1_analysis

# Run tests
python -m pytest tests/ -v
```

### Reports

After running the pipeline, find outputs in:
- `reports/phase1/dataset_summary.md` — Human-readable dataset report
- `reports/phase1/dataset_statistics.csv` — Per-subset statistics
- `reports/phase1/sensor_statistics.csv` — Per-sensor statistics
- `reports/phase1/figures/` — All visualizations

### Notebook

- `notebooks/01_cmapss_dataset_understanding.ipynb` — Interactive exploration notebook

---

## Project Structure

```
AeroRUL/
├── CMAPSSData/                  # Raw C-MAPSS data (DO NOT MODIFY)
│   ├── train_FD00*.txt
│   ├── test_FD00*.txt
│   └── RUL_FD00*.txt
├── src/
│   ├── __init__.py
│   ├── config.py                # Project configuration & schema
│   ├── data_loader.py           # Reusable C-MAPSS data loader
│   ├── data_validation.py       # Validation & statistics functions
│   ├── visualization.py         # Publication-quality plotting
│   └── run_phase1_analysis.py   # Phase 1 analysis pipeline
├── tests/
│   ├── test_data_loader.py
│   └── test_data_validation.py
├── notebooks/
│   └── 01_cmapss_dataset_understanding.ipynb
├── reports/
│   └── phase1/
│       ├── dataset_summary.md
│       ├── dataset_statistics.csv
│       ├── sensor_statistics.csv
│       └── figures/
├── requirements.txt
└── README.md
```

---

## Dependencies

- Python 3.11+
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.7
- seaborn >= 0.12
- scipy >= 1.10
- pytest >= 7.0
