"""
AeroRUL — Project Configuration

Central configuration for paths, schema definitions, and dataset metadata.
All paths are relative to the project root to avoid machine-specific hard-coding.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (two levels up from src/config.py → AeroRUL/)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
RAW_DATA_DIR = PROJECT_ROOT / "CMAPSSData"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PHASE1_REPORTS_DIR = REPORTS_DIR / "phase1"
FIGURES_DIR = PHASE1_REPORTS_DIR / "figures"
PHASE2_REPORTS_DIR = REPORTS_DIR / "phase2"
PHASE2_FIGURES_DIR = PHASE2_REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# ---------------------------------------------------------------------------
# C-MAPSS dataset metadata
# ---------------------------------------------------------------------------
AVAILABLE_SUBSETS = ("FD001", "FD002", "FD003", "FD004")

SUBSET_METADATA = {
    "FD001": {"conditions": 1, "fault_modes": 1, "description": "ONE condition (Sea Level), ONE fault mode (HPC Degradation)"},
    "FD002": {"conditions": 6, "fault_modes": 1, "description": "SIX conditions, ONE fault mode (HPC Degradation)"},
    "FD003": {"conditions": 1, "fault_modes": 2, "description": "ONE condition (Sea Level), TWO fault modes (HPC + Fan Degradation)"},
    "FD004": {"conditions": 6, "fault_modes": 2, "description": "SIX conditions, TWO fault modes (HPC + Fan Degradation)"},
}

# ---------------------------------------------------------------------------
# C-MAPSS schema
# ---------------------------------------------------------------------------
EXPECTED_NUM_COLUMNS = 26

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Column groups for convenient access
ID_COLUMNS = ["unit_id", "cycle"]
OP_SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]

# Data types expected after loading
COLUMN_DTYPES = {
    "unit_id": "int64",
    "cycle": "int64",
}
# Operational settings and sensors are float64 by default

# ---------------------------------------------------------------------------
# File-naming conventions
# ---------------------------------------------------------------------------

def train_filename(subset: str) -> str:
    """Return the expected training filename for a subset."""
    return f"train_{subset}.txt"


def test_filename(subset: str) -> str:
    """Return the expected test filename for a subset."""
    return f"test_{subset}.txt"


def rul_filename(subset: str) -> str:
    """Return the expected RUL ground-truth filename for a subset."""
    return f"RUL_{subset}.txt"


# ---------------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------------
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

# Near-constant sensor threshold: if std < this fraction of mean, flag it
NEAR_CONSTANT_STD_THRESHOLD = 0.001

# ---------------------------------------------------------------------------
# Phase 2 — RUL, Splitting, Normalization
# ---------------------------------------------------------------------------

# RUL clipping thresholds to investigate
RUL_CLIP_VALUES = [None, 50, 75, 100, 125, 150]

# Engine-level train/validation split defaults
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_RANDOM_STATE = 42

# Operating-regime identification: rounding decimals for op_settings
# Used to discretize continuous op_settings into regime clusters
OP_SETTING_ROUNDING_DECIMALS = 0

# Safety floor for standard deviation during normalization
NORMALIZATION_STD_FLOOR = 1e-6

# ---------------------------------------------------------------------------
# Phase 3 — Sliding-Window Feature Engineering
# ---------------------------------------------------------------------------

PHASE3_REPORTS_DIR = REPORTS_DIR / "phase3"
PHASE3_FIGURES_DIR = PHASE3_REPORTS_DIR / "figures"
PHASE3_DATA_DIR = PROCESSED_DATA_DIR / "phase3"

# Feature generation configuration
FEATURE_CONFIG = {
    "windows": [10, 20, 30, 50, 75],
    "statistics": ["mean", "std", "min", "max", "range", "slope", "r2"],
    "deltas": [1, 5, 10],
    "relative_changes": [5, 10],
}

DEFAULT_WINDOW_SIZE = 30

# Safety floor for relative change denominator
RELATIVE_CHANGE_FLOOR = 1e-8

# Metadata columns (not model features)
METADATA_COLUMNS = ["unit_id", "cycle"]
TARGET_COLUMN = "rul"

# ---------------------------------------------------------------------------
# Phase 3.1 — Feature Quality Control
# ---------------------------------------------------------------------------

PHASE3_1_REPORTS_DIR = REPORTS_DIR / "phase3_1"
PHASE3_1_FIGURES_DIR = PHASE3_1_REPORTS_DIR / "figures"

# Distribution shift threshold (Cohen's d = 0.5 = medium effect)
DISTRIBUTION_SHIFT_THRESHOLD = 0.5
