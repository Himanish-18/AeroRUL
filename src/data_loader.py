"""
AeroRUL — C-MAPSS Data Loader

Reusable module for discovering, loading, and validating C-MAPSS dataset files.
Returns clean pandas DataFrames with proper column names and basic schema checks.

Usage:
    from src.data_loader import load_cmapss, discover_datasets

    available = discover_datasets()
    data = load_cmapss("FD001")
    train_df = data["train"]
    test_df  = data["test"]
    rul_df   = data["rul"]
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    AVAILABLE_SUBSETS,
    COLUMN_NAMES,
    EXPECTED_NUM_COLUMNS,
    ID_COLUMNS,
    RAW_DATA_DIR,
    rul_filename,
    test_filename,
    train_filename,
)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_datasets(data_dir: Optional[Path] = None) -> dict:
    """
    Scan the data directory and report which C-MAPSS subsets are available.

    Returns a dict keyed by subset name, e.g.:
        {
            "FD001": {"train": True, "test": True, "rul": True},
            "FD002": {"train": True, "test": True, "rul": False},
            ...
        }
    """
    data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            "Check that the C-MAPSS data has been downloaded."
        )

    result = {}
    for subset in AVAILABLE_SUBSETS:
        result[subset] = {
            "train": (data_dir / train_filename(subset)).is_file(),
            "test": (data_dir / test_filename(subset)).is_file(),
            "rul": (data_dir / rul_filename(subset)).is_file(),
        }
    return result


# ---------------------------------------------------------------------------
# File loading helpers
# ---------------------------------------------------------------------------

def _load_tabular_file(filepath: Path, column_names: list[str]) -> pd.DataFrame:
    """
    Load a space-separated C-MAPSS data file (train or test).

    Parameters
    ----------
    filepath : Path
        Absolute path to the .txt file.
    column_names : list[str]
        Column names to assign.

    Returns
    -------
    pd.DataFrame
        Loaded and column-named DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the actual number of columns does not match the expected schema.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"File not found: {filepath}")

    # C-MAPSS files are space-separated with trailing whitespace
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    actual_cols = df.shape[1]
    expected_cols = len(column_names)

    if actual_cols != expected_cols:
        raise ValueError(
            f"Schema mismatch in {filepath.name}: "
            f"expected {expected_cols} columns, found {actual_cols}. "
            f"The file may have a non-standard format."
        )

    df.columns = column_names

    # Enforce integer types for ID columns
    for col in ID_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(np.int64)

    return df


def _load_rul_file(filepath: Path) -> pd.DataFrame:
    """
    Load a C-MAPSS RUL ground-truth file.

    Each line contains a single integer: the true remaining useful life
    for the corresponding test engine (1-indexed).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['unit_id', 'rul'].
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"RUL file not found: {filepath}")

    rul_values = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=["rul"],
        engine="python",
    )

    # Assign 1-indexed unit_id
    rul_values.insert(0, "unit_id", range(1, len(rul_values) + 1))
    rul_values["unit_id"] = rul_values["unit_id"].astype(np.int64)
    rul_values["rul"] = rul_values["rul"].astype(np.int64)

    return rul_values


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_cmapss(
    subset: str,
    data_dir: Optional[Path] = None,
) -> dict:
    """
    Load a C-MAPSS dataset subset.

    Parameters
    ----------
    subset : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir : Path, optional
        Override the default raw data directory.

    Returns
    -------
    dict
        {
            "train": pd.DataFrame,      # Training data
            "test":  pd.DataFrame,       # Test data
            "rul":   pd.DataFrame | None # Ground-truth RUL (None if file missing)
        }

    Raises
    ------
    ValueError
        If subset is not a valid C-MAPSS identifier.
    FileNotFoundError
        If required train/test files are missing.
    """
    subset = subset.upper()

    if subset not in AVAILABLE_SUBSETS:
        raise ValueError(
            f"Invalid subset '{subset}'. "
            f"Valid subsets: {AVAILABLE_SUBSETS}"
        )

    data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR

    # ── Load training data ──────────────────────────────────────────────
    train_path = data_dir / train_filename(subset)
    train_df = _load_tabular_file(train_path, COLUMN_NAMES)

    # ── Load test data ──────────────────────────────────────────────────
    test_path = data_dir / test_filename(subset)
    test_df = _load_tabular_file(test_path, COLUMN_NAMES)

    # ── Load RUL ground truth (optional) ────────────────────────────────
    rul_path = data_dir / rul_filename(subset)
    rul_df = None
    if rul_path.is_file():
        rul_df = _load_rul_file(rul_path)

    return {"train": train_df, "test": test_df, "rul": rul_df}


def load_all_subsets(data_dir: Optional[Path] = None) -> dict:
    """
    Load all available C-MAPSS subsets.

    Returns
    -------
    dict
        Keyed by subset name, e.g.:
        {
            "FD001": {"train": df, "test": df, "rul": df},
            "FD002": ...,
            ...
        }
        Only subsets with train files present are included.
    """
    available = discover_datasets(data_dir)
    all_data = {}

    for subset, files in available.items():
        if files["train"] and files["test"]:
            all_data[subset] = load_cmapss(subset, data_dir)

    return all_data
