"""
Tests for src.normalization — Operating-condition normalization.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import COLUMN_NAMES, OP_SETTING_COLUMNS, SENSOR_COLUMNS
from src.data_loader import load_cmapss
from src.normalization import (
    apply_regime_normalization,
    fit_regime_normalization,
    identify_operating_regimes,
    regime_sensor_analysis,
    regime_summary,
)
from src.rul import generate_rul_targets
from src.splitting import split_by_engine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fd002_data():
    """Load FD002 (multi-condition) training data with RUL."""
    data = load_cmapss("FD002")
    return generate_rul_targets(data["train"])


@pytest.fixture(scope="module")
def fd002_split(fd002_data):
    """Pre-computed split of FD002."""
    return split_by_engine(fd002_data, validation_fraction=0.2, random_state=42)


@pytest.fixture(scope="module")
def fd002_regimes(fd002_split):
    """Operating regimes for FD002 training split."""
    train_df, _ = fd002_split
    return identify_operating_regimes(train_df)


@pytest.fixture(scope="module")
def fd002_norm_params(fd002_split, fd002_regimes):
    """Normalization parameters fitted on FD002 training data."""
    train_df, _ = fd002_split
    return fit_regime_normalization(train_df, fd002_regimes)


@pytest.fixture
def synthetic_multi_regime():
    """Synthetic DataFrame with 2 clear operating regimes."""
    rng = np.random.default_rng(42)
    n = 100

    # Regime 1: op_setting_1 ~ 0
    r1 = pd.DataFrame({
        "unit_id": [1] * n,
        "cycle": list(range(1, n + 1)),
        "op_setting_1": rng.normal(0, 0.001, n),
        "op_setting_2": rng.normal(0, 0.001, n),
        "op_setting_3": np.full(n, 100.0),
        "sensor_1": rng.normal(500, 10, n),
        "sensor_2": rng.normal(600, 5, n),
    })

    # Regime 2: op_setting_1 ~ 42
    r2 = pd.DataFrame({
        "unit_id": [2] * n,
        "cycle": list(range(1, n + 1)),
        "op_setting_1": rng.normal(42, 0.001, n),
        "op_setting_2": rng.normal(0.84, 0.001, n),
        "op_setting_3": np.full(n, 100.0),
        "sensor_1": rng.normal(550, 8, n),
        "sensor_2": rng.normal(650, 6, n),
    })

    return pd.concat([r1, r2], ignore_index=True)


# ---------------------------------------------------------------------------
# Regime identification tests
# ---------------------------------------------------------------------------

class TestRegimeIdentification:
    """Tests for identify_operating_regimes."""

    def test_fd002_has_multiple_regimes(self, fd002_split):
        train_df, _ = fd002_split
        regimes = identify_operating_regimes(train_df)
        n_unique = regimes.nunique()
        # FD002 should have ~6 operating conditions
        assert n_unique >= 4  # Conservative lower bound

    def test_returns_series(self, fd002_split):
        train_df, _ = fd002_split
        regimes = identify_operating_regimes(train_df)
        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(train_df)

    def test_synthetic_two_regimes(self, synthetic_multi_regime):
        regimes = identify_operating_regimes(
            synthetic_multi_regime,
            op_columns=["op_setting_1", "op_setting_2", "op_setting_3"],
        )
        assert regimes.nunique() == 2

    def test_regime_summary_columns(self, fd002_split, fd002_regimes):
        train_df, _ = fd002_split
        summary = regime_summary(train_df, fd002_regimes)
        assert "regime" in summary.columns
        assert "n_observations" in summary.columns
        assert "pct_observations" in summary.columns


# ---------------------------------------------------------------------------
# Normalization fitting tests
# ---------------------------------------------------------------------------

class TestFitNormalization:
    """Tests for fit_regime_normalization."""

    def test_returns_dict(self, fd002_norm_params):
        assert isinstance(fd002_norm_params, dict)
        assert len(fd002_norm_params) > 0

    def test_params_have_mean_and_std(self, fd002_norm_params):
        for regime, params in fd002_norm_params.items():
            assert "mean" in params
            assert "std" in params
            assert "n_observations" in params

    def test_std_never_zero(self, fd002_norm_params):
        """Standard deviations should never be zero (clamped by floor)."""
        for regime, params in fd002_norm_params.items():
            for sensor, std_val in params["std"].items():
                assert std_val > 0, f"std=0 for {regime}/{sensor}"

    def test_fitted_on_training_only(self, fd002_split, fd002_regimes):
        """Parameters should reflect training data statistics."""
        train_df, val_df = fd002_split
        params = fit_regime_normalization(train_df, fd002_regimes)

        # Verify a regime's mean matches the training data
        first_regime = list(params.keys())[0]
        mask = fd002_regimes == first_regime
        regime_data = train_df.loc[mask]

        for sensor in ["sensor_2", "sensor_3"]:
            if sensor in regime_data.columns:
                expected_mean = float(regime_data[sensor].mean())
                actual_mean = params[first_regime]["mean"][sensor]
                assert abs(expected_mean - actual_mean) < 1e-4


# ---------------------------------------------------------------------------
# Normalization application tests
# ---------------------------------------------------------------------------

class TestApplyNormalization:
    """Tests for apply_regime_normalization."""

    def test_training_normalized_has_new_columns(self, fd002_split, fd002_norm_params, fd002_regimes):
        train_df, _ = fd002_split
        normalized = apply_regime_normalization(train_df, fd002_norm_params, fd002_regimes)
        assert "sensor_2_norm" in normalized.columns
        assert "sensor_3_norm" in normalized.columns

    def test_preserves_original_columns(self, fd002_split, fd002_norm_params, fd002_regimes):
        train_df, _ = fd002_split
        normalized = apply_regime_normalization(train_df, fd002_norm_params, fd002_regimes)
        assert "sensor_2" in normalized.columns  # Original preserved

    def test_preserves_row_count(self, fd002_split, fd002_norm_params, fd002_regimes):
        train_df, _ = fd002_split
        normalized = apply_regime_normalization(train_df, fd002_norm_params, fd002_regimes)
        assert len(normalized) == len(train_df)

    def test_training_regime_approx_zero_mean(self, fd002_split, fd002_norm_params, fd002_regimes):
        """After normalization, each regime should have approx zero mean for normalized sensors."""
        train_df, _ = fd002_split
        normalized = apply_regime_normalization(train_df, fd002_norm_params, fd002_regimes)

        for regime_label in fd002_regimes.unique():
            mask = fd002_regimes == regime_label
            if mask.sum() < 10:
                continue  # Skip tiny regimes
            regime_mean = normalized.loc[mask, "sensor_2_norm"].mean()
            assert abs(regime_mean) < 0.1, f"Regime {regime_label}: mean={regime_mean}"

    def test_validation_uses_training_params(self, fd002_split, fd002_norm_params):
        """Validation normalization should use training-fitted parameters."""
        _, val_df = fd002_split
        val_regimes = identify_operating_regimes(val_df)
        normalized_val = apply_regime_normalization(val_df, fd002_norm_params, val_regimes)

        # Validation mean won't be exactly zero (different data), but should be reasonable
        assert "sensor_2_norm" in normalized_val.columns
        assert len(normalized_val) == len(val_df)


# ---------------------------------------------------------------------------
# Regime sensor analysis tests
# ---------------------------------------------------------------------------

class TestRegimeSensorAnalysis:
    """Tests for regime_sensor_analysis."""

    def test_returns_dataframe(self, fd002_split, fd002_regimes):
        train_df, _ = fd002_split
        analysis = regime_sensor_analysis(train_df, fd002_regimes)
        assert isinstance(analysis, pd.DataFrame)
        assert "regime" in analysis.columns
        assert "sensor" in analysis.columns
        assert "mean" in analysis.columns
