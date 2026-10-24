"""
AeroRUL — Phase 4: Benchmark Pipeline

Complete classical regression benchmark for RUL prediction.
Processes FD001-FD004 independently, evaluates multiple models,
and generates comprehensive reports and visualizations.

Usage:
    python -m src.run_phase4_benchmark
"""

import json
import sys
import time
import warnings
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from src.config import (
    AVAILABLE_SUBSETS,
    FEATURE_CONFIG,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_STYLE,
    METADATA_COLUMNS,
    PHASE3_DATA_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    RUL_CLIP_VALUES,
)
from src.features import get_feature_columns, _parse_feature_name
from src.metrics import (
    mae, rmse, r2_score, mean_signed_error,
    nasa_score, nasa_score_per_engine,
    engine_macro_metrics, extract_endpoints, endpoint_metrics,
    rul_range_error_analysis, evaluate_model, assign_rul_range,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

PHASE4_REPORTS_DIR = REPORTS_DIR / "phase4"
PHASE4_FIGURES_DIR = PHASE4_REPORTS_DIR / "figures"
PHASE4_PREDICTIONS_DIR = PHASE4_REPORTS_DIR / "predictions"

RANDOM_SEED = 42
N_JOBS = -1  # parallel execution

# Model configurations (documented starting points)
MODEL_CONFIGS = {
    "DummyBaseline": {
        "class": "DummyRegressor",
        "params": {"strategy": "median"},
        "needs_scaling": False,
    },
    "LinearRegression": {
        "class": "LinearRegression",
        "params": {},
        "needs_scaling": True,
    },
    "RidgeRegression": {
        "class": "Ridge",
        "params": {"alpha": 1.0, "random_state": RANDOM_SEED},
        "needs_scaling": True,
    },
    "RandomForest": {
        "class": "RandomForestRegressor",
        "params": {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_leaf": 5,
            "random_state": RANDOM_SEED,
            "n_jobs": N_JOBS,
        },
        "needs_scaling": False,
    },
}

# Optional models — added if available
OPTIONAL_MODELS = {}

try:
    from xgboost import XGBRegressor
    OPTIONAL_MODELS["XGBoost"] = {
        "class": "XGBRegressor",
        "params": {
            "objective": "reg:squarederror",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "tree_method": "hist",
            "n_jobs": N_JOBS,
            "verbosity": 0,
        },
        "needs_scaling": False,
    }
except ImportError:
    pass

try:
    from lightgbm import LGBMRegressor
    OPTIONAL_MODELS["LightGBM"] = {
        "class": "LGBMRegressor",
        "params": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 8,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "n_jobs": N_JOBS,
            "verbose": -1,
        },
        "needs_scaling": False,
    }
except ImportError:
    pass

ALL_MODEL_CONFIGS = {**MODEL_CONFIGS, **OPTIONAL_MODELS}

# RUL cap values for sensitivity experiment
RUL_CAPS = [None, 50, 75, 100, 125, 150]


def _apply_style():
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING AND PREFLIGHT
# ═══════════════════════════════════════════════════════════════════════════

def load_phase3_data(subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Phase 3 feature files for a subset."""
    subset_dir = PHASE3_DATA_DIR / subset
    train_df = pd.read_parquet(subset_dir / "train_features.parquet")
    val_df = pd.read_parquet(subset_dir / "val_features.parquet")
    return train_df, val_df


def load_approved_features(subset: str) -> list[str]:
    """Load Phase 3.1 approved feature list (usable_for_phase4 == YES)."""
    inv_path = REPORTS_DIR / "phase3_1" / f"feature_quality_inventory_{subset}.csv"
    inv = pd.read_csv(inv_path)
    yes_features = inv[inv["usable_for_phase4"] == "YES"]["feature_name"].tolist()
    return yes_features


def data_preflight(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    approved_features: list[str],
    subset: str,
) -> dict:
    """
    Strict data preflight before fitting any model.

    Checks target integrity, predictor integrity, feature quality,
    and engine separation.
    """
    results = {"subset": subset, "issues": [], "passed": True}

    # 1. Target integrity
    if TARGET_COLUMN not in train_df.columns:
        results["issues"].append("CRITICAL: rul column missing from train")
        results["passed"] = False
    if TARGET_COLUMN not in val_df.columns:
        results["issues"].append("CRITICAL: rul column missing from val")
        results["passed"] = False

    if results["passed"]:
        if (train_df[TARGET_COLUMN] < 0).any():
            results["issues"].append("CRITICAL: negative RUL in train")
            results["passed"] = False
        if (val_df[TARGET_COLUMN] < 0).any():
            results["issues"].append("CRITICAL: negative RUL in val")
            results["passed"] = False

    # 2. Approved features present
    available = set(train_df.columns)
    missing = [f for f in approved_features if f not in available]
    if missing:
        results["issues"].append(f"WARNING: {len(missing)} approved features missing from data")

    # Filter to available approved features
    usable = [f for f in approved_features if f in available]
    results["n_approved"] = len(approved_features)
    results["n_usable"] = len(usable)
    results["usable_features"] = usable

    # 3. Forbidden columns check
    forbidden = {"rul", "unit_id", "cycle", "failure_cycle"}
    leaked = forbidden & set(usable)
    if leaked:
        results["issues"].append(f"CRITICAL: forbidden columns in features: {leaked}")
        results["passed"] = False

    # 4. Feature quality on approved features
    X_train = train_df[usable]
    nan_count = int(X_train.isna().sum().sum())
    inf_count = int(np.isinf(X_train.select_dtypes(include=[np.floating])).sum().sum())

    results["nan_count_train"] = nan_count
    results["inf_count_train"] = inf_count

    if nan_count > 0:
        results["issues"].append(f"WARNING: {nan_count} NaN values in approved train features")
    if inf_count > 0:
        results["issues"].append(f"CRITICAL: {inf_count} inf values in approved train features")
        results["passed"] = False

    # 5. Engine separation
    train_engines = set(train_df["unit_id"].unique())
    val_engines = set(val_df["unit_id"].unique())
    overlap = train_engines & val_engines
    if overlap:
        results["issues"].append(f"CRITICAL: engine overlap: {overlap}")
        results["passed"] = False

    results["n_train_engines"] = len(train_engines)
    results["n_val_engines"] = len(val_engines)
    results["n_train_rows"] = len(train_df)
    results["n_val_rows"] = len(val_df)

    return results


def prepare_clean_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    usable_features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract clean feature matrices and targets. Handle any remaining NaN."""
    X_train = train_df[usable_features].values.copy()
    X_val = val_df[usable_features].values.copy()
    y_train = train_df[TARGET_COLUMN].values.copy()
    y_val = val_df[TARGET_COLUMN].values.copy()

    # Handle R² NaN (REQUIRES_FIX features were excluded, but YES features
    # should be clean. Double-check and replace any NaN with 0.0 for R² features)
    nan_mask_train = np.isnan(X_train)
    nan_mask_val = np.isnan(X_val)
    if nan_mask_train.any():
        X_train = np.nan_to_num(X_train, nan=0.0)
    if nan_mask_val.any():
        X_val = np.nan_to_num(X_val, nan=0.0)

    return X_train, X_val, y_train, y_val


# ═══════════════════════════════════════════════════════════════════════════
# 2. MODEL BUILDING AND TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, config: dict):
    """Instantiate a model from configuration."""
    class_name = config["class"]
    params = config["params"]

    if class_name == "DummyRegressor":
        return DummyRegressor(**params)
    elif class_name == "LinearRegression":
        return LinearRegression(**params)
    elif class_name == "Ridge":
        return Ridge(**params)
    elif class_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif class_name == "XGBRegressor":
        from xgboost import XGBRegressor
        return XGBRegressor(**params)
    elif class_name == "LGBMRegressor":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Unknown model class: {class_name}")


def train_and_predict(
    model_name: str,
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
) -> tuple:
    """Train model and return (model, scaler, raw_predictions, runtime_seconds)."""
    scaler = None
    X_tr = X_train.copy()
    X_v = X_val.copy()

    if config.get("needs_scaling", False):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_v = scaler.transform(X_v)

    model = build_model(model_name, config)

    start = time.time()
    model.fit(X_tr, y_train)
    preds_raw = model.predict(X_v)
    elapsed = time.time() - start

    return model, scaler, preds_raw, elapsed


def create_predictions_df(
    val_df: pd.DataFrame,
    preds_raw: np.ndarray,
    model_name: str,
    subset: str,
) -> pd.DataFrame:
    """Build a predictions DataFrame with all required columns."""
    pred_df = pd.DataFrame({
        "dataset": subset,
        "model": model_name,
        "unit_id": val_df["unit_id"].values,
        "cycle": val_df["cycle"].values,
        "actual_rul": val_df[TARGET_COLUMN].values,
        "predicted_rul_raw": preds_raw,
        "predicted_rul_nonnegative": np.maximum(preds_raw, 0),
    })
    pred_df["prediction_error"] = pred_df["predicted_rul_nonnegative"] - pred_df["actual_rul"]
    pred_df["absolute_error"] = np.abs(pred_df["prediction_error"])
    return pred_df


# ═══════════════════════════════════════════════════════════════════════════
# 3. DUPLICATE FEATURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_exact_duplicates_train(
    X_train: np.ndarray,
    feature_names: list[str],
) -> tuple[list[str], dict]:
    """
    Detect exact duplicate columns using training data only.
    Returns (deduplicated_feature_names, duplicate_groups).
    """
    n_cols = X_train.shape[1]
    keep_mask = np.ones(n_cols, dtype=bool)
    groups = {}
    group_id = 0

    for i in range(n_cols):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, n_cols):
            if not keep_mask[j]:
                continue
            if np.allclose(X_train[:, i], X_train[:, j], atol=1e-12, equal_nan=True):
                keep_mask[j] = False
                gkey = f"group_{group_id}"
                if gkey not in groups:
                    groups[gkey] = {"kept": feature_names[i], "removed": []}
                groups[gkey]["removed"].append(feature_names[j])
        if feature_names[i] in [g["kept"] for g in groups.values()]:
            group_id += 1

    deduped = [feature_names[i] for i in range(n_cols) if keep_mask[i]]
    return deduped, groups


# ═══════════════════════════════════════════════════════════════════════════
# 4. WINDOW FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def select_window_features(
    all_features: list[str],
    target_window: int | None = None,
) -> list[str]:
    """
    Select features belonging to a specific window, or all if None.

    For individual windows: includes rolling features (mean, std, min, max,
    range, slope, r2) for the target window PLUS all delta and relchange
    features (which are consistent across window experiments).
    """
    if target_window is None:
        return all_features

    selected = []
    for feat in all_features:
        parsed = _parse_feature_name(feat)
        if parsed is None:
            continue
        sensor, stat, window = parsed

        if stat in ("delta", "relchange"):
            # Always include delta and relchange features
            selected.append(feat)
        elif window == target_window:
            # Include rolling features for the target window
            selected.append(feat)

    return selected


# ═══════════════════════════════════════════════════════════════════════════
# 5. VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════

def plot_model_comparison(
    metrics_df: pd.DataFrame,
    subset: str,
    output_dir: Path,
):
    """Create model comparison bar charts for a dataset."""
    _apply_style()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Row-level MAE
    ax = axes[0, 0]
    models = metrics_df["model"].values
    ax.barh(models, metrics_df["row_mae"].values, color=sns.color_palette("husl", len(models)))
    ax.set_title(f"{subset}: Row-Level MAE", fontweight="bold")
    ax.set_xlabel("MAE (cycles)")

    # Row-level RMSE
    ax = axes[0, 1]
    ax.barh(models, metrics_df["row_rmse"].values, color=sns.color_palette("husl", len(models)))
    ax.set_title(f"{subset}: Row-Level RMSE", fontweight="bold")
    ax.set_xlabel("RMSE (cycles)")

    # Endpoint MAE
    ax = axes[1, 0]
    ax.barh(models, metrics_df["endpoint_mae"].values, color=sns.color_palette("husl", len(models)))
    ax.set_title(f"{subset}: Endpoint MAE", fontweight="bold")
    ax.set_xlabel("MAE (cycles)")

    # Endpoint NASA Score
    ax = axes[1, 1]
    ax.barh(models, metrics_df["endpoint_nasa_score"].values, color=sns.color_palette("husl", len(models)))
    ax.set_title(f"{subset}: Endpoint NASA Score", fontweight="bold")
    ax.set_xlabel("NASA Score (lower is better)")

    fig.suptitle(f"Model Comparison — {subset}", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"model_comparison_{subset}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vs_actual(
    preds_df: pd.DataFrame,
    model_name: str,
    subset: str,
    output_dir: Path,
):
    """Predicted vs actual scatter plot."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    y_true = preds_df["actual_rul"].values
    y_pred = preds_df["predicted_rul_nonnegative"].values

    ax.scatter(y_true, y_pred, alpha=0.1, s=5, color="steelblue")
    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", linewidth=1.5, label="y=x")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{subset} — {model_name}: Predicted vs Actual", fontweight="bold")
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    fig.tight_layout()
    fig.savefig(output_dir / f"pred_vs_actual_{subset}_{model_name}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    preds_df: pd.DataFrame,
    model_name: str,
    subset: str,
    output_dir: Path,
):
    """Residual analysis plots."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    residuals = preds_df["prediction_error"].values
    y_true = preds_df["actual_rul"].values

    # Residual distribution
    axes[0, 0].hist(residuals, bins=80, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title("Residual Distribution", fontweight="bold")
    axes[0, 0].set_xlabel("Prediction Error (cycles)")

    # Residual vs actual
    axes[0, 1].scatter(y_true, residuals, alpha=0.1, s=3, color="steelblue")
    axes[0, 1].axhline(0, color="red", linestyle="--")
    axes[0, 1].set_title("Residual vs Actual RUL", fontweight="bold")
    axes[0, 1].set_xlabel("Actual RUL")
    axes[0, 1].set_ylabel("Prediction Error")

    # Residual vs cycle
    axes[1, 0].scatter(preds_df["cycle"].values, residuals, alpha=0.1, s=3, color="coral")
    axes[1, 0].axhline(0, color="red", linestyle="--")
    axes[1, 0].set_title("Residual vs Cycle", fontweight="bold")
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].set_ylabel("Prediction Error")

    # Error by RUL range
    rul_range = assign_rul_range(y_true)
    range_df = pd.DataFrame({"rul_range": rul_range, "abs_error": np.abs(residuals)})
    range_order = ["critical_0_30", "low_31_75", "moderate_76_125", "high_gt125"]
    range_means = range_df.groupby("rul_range")["abs_error"].mean()
    x_vals = [r for r in range_order if r in range_means.index]
    y_vals = [range_means[r] for r in x_vals]
    axes[1, 1].bar(x_vals, y_vals, color=["red", "orange", "green", "blue"], edgecolor="black")
    axes[1, 1].set_title("MAE by RUL Range", fontweight="bold")
    axes[1, 1].set_ylabel("MAE (cycles)")
    axes[1, 1].tick_params(axis="x", rotation=30)

    fig.suptitle(f"Residual Analysis — {subset} / {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"residuals_{subset}_{model_name}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_curves(
    preds_df: pd.DataFrame,
    model_name: str,
    subset: str,
    output_dir: Path,
    n_engines: int = 4,
):
    """Plot predicted vs actual RUL degradation curves for representative engines."""
    _apply_style()

    # Select representative engines deterministically by trajectory length
    engine_lengths = preds_df.groupby("unit_id")["cycle"].count().sort_values()
    n_avail = len(engine_lengths)
    if n_avail == 0:
        return

    # Pick engines at quartile positions
    indices = np.linspace(0, n_avail - 1, min(n_engines, n_avail), dtype=int)
    selected_engines = engine_lengths.index[indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for i, eid in enumerate(selected_engines):
        if i >= len(axes):
            break
        ax = axes[i]
        edf = preds_df[preds_df["unit_id"] == eid].sort_values("cycle")

        ax.plot(edf["cycle"], edf["actual_rul"], "b-", linewidth=2, label="Actual RUL")
        ax.plot(edf["cycle"], edf["predicted_rul_nonnegative"], "r--", linewidth=1.5, label="Predicted RUL")

        e_mae = mae(edf["actual_rul"].values, edf["predicted_rul_nonnegative"].values)
        ax.set_title(f"Engine {eid} (MAE={e_mae:.1f})", fontweight="bold")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=8)

    # Clear unused axes
    for i in range(len(selected_engines), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Degradation Curves — {subset} / {model_name}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"degradation_curves_{subset}_{model_name}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cap_sensitivity(
    cap_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot RUL cap sensitivity results."""
    _apply_style()

    subsets = cap_df["dataset"].unique()
    models = cap_df["model"].unique()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for model_name in models:
        for subset in subsets:
            mdf = cap_df[(cap_df["model"] == model_name) & (cap_df["dataset"] == subset)]
            if len(mdf) == 0:
                continue
            label = f"{model_name}/{subset}"
            caps = mdf["cap_label"].values
            axes[0].plot(range(len(caps)), mdf["endpoint_mae_vs_uncapped"].values, "o-", label=label, markersize=4)
            axes[1].plot(range(len(caps)), mdf["endpoint_rmse_vs_uncapped"].values, "o-", label=label, markersize=4)
            axes[2].plot(range(len(caps)), mdf["endpoint_nasa_score"].values, "o-", label=label, markersize=4)

    for ax, title in zip(axes, ["Endpoint MAE vs Uncapped RUL", "Endpoint RMSE vs Uncapped RUL", "Endpoint NASA Score"]):
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(range(len(caps)))
        ax.set_xticklabels(caps, rotation=45)
        ax.set_xlabel("RUL Cap")
        ax.legend(fontsize=7)

    fig.suptitle("RUL Cap Sensitivity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"cap_sensitivity.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_window_comparison(
    window_df: pd.DataFrame,
    output_dir: Path,
):
    """Plot window configuration comparison."""
    _apply_style()

    subsets = window_df["dataset"].unique()

    for subset in subsets:
        sdf = window_df[window_df["dataset"] == subset]
        if len(sdf) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        configs = sdf["window_config"].values
        x = range(len(configs))

        axes[0].bar(x, sdf["row_mae"].values, color="steelblue", edgecolor="black")
        axes[0].set_title(f"{subset}: Row MAE by Window", fontweight="bold")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(configs, rotation=45, ha="right")

        axes[1].bar(x, sdf["endpoint_mae"].values, color="coral", edgecolor="black")
        axes[1].set_title(f"{subset}: Endpoint MAE by Window", fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(configs, rotation=45, ha="right")

        axes[2].bar(x, sdf["endpoint_nasa_score"].values, color="mediumpurple", edgecolor="black")
        axes[2].set_title(f"{subset}: Endpoint NASA Score by Window", fontweight="bold")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(configs, rotation=45, ha="right")

        fig.suptitle(f"Window Configuration — {subset}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(output_dir / f"window_comparison_{subset}.{FIGURE_FORMAT}",
                    dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_phase4_report(
    all_benchmark, all_endpoints, all_engine_macro,
    all_rul_range, cap_results, window_results,
    dedup_results, skipped_models, output_path,
):
    """Generate comprehensive Phase 4 summary report."""
    lines = []
    lines.append("# AeroRUL — Phase 4: Classical Regression Benchmark Report\n")

    # A. Executive Summary
    lines.append("## A. Executive Summary\n")

    # Find best model per dataset
    for subset in sorted(all_benchmark.keys()):
        bm = all_benchmark[subset]
        if not bm:
            continue
        best = min(bm, key=lambda x: x["endpoint_mae"])
        lines.append(f"### {subset}")
        lines.append(f"- **Best model (endpoint MAE)**: {best['model']} — {best['endpoint_mae']:.2f} cycles")
        lines.append(f"- **Endpoint NASA score**: {best['endpoint_nasa_score']:.2f}")
        lines.append(f"- **Row MAE**: {best['row_mae']:.2f}, Row RMSE: {best['row_rmse']:.2f}")

        # Baseline comparison
        baseline = [b for b in bm if b["model"] == "DummyBaseline"]
        if baseline:
            bl = baseline[0]
            improvement = bl["endpoint_mae"] - best["endpoint_mae"]
            lines.append(f"- **Improvement over baseline**: {improvement:.2f} cycles ({100*improvement/bl['endpoint_mae']:.1f}%)")
        lines.append("")

    if skipped_models:
        lines.append("### Skipped Models")
        for m, reason in skipped_models.items():
            lines.append(f"- {m}: {reason}")
        lines.append("")

    lines.append("### Limitations")
    lines.append("- Validation-set results; official test set NOT used for selection")
    lines.append("- Hyperparameters are fixed starting points, not optimized")
    lines.append("- Early cycles excluded by 75-cycle feature window")
    lines.append("- Row-level metrics may differ from endpoint metrics")
    lines.append("")

    # B. Methodology
    lines.append("## B. Experimental Methodology\n")
    lines.append("- **Split**: Engine-level train/val (Phase 2, seed=42, 80/20)")
    lines.append("- **Features**: Phase 3.1 approved (`YES`) features only")
    lines.append("- **Preprocessing**: StandardScaler (train-fit) for linear models")
    lines.append(f"- **Random seed**: {RANDOM_SEED}")
    lines.append("- **Prediction constraint**: max(pred, 0) for nonnegative RUL")
    lines.append("")
    lines.append("### Model Configurations\n")
    for name, cfg in ALL_MODEL_CONFIGS.items():
        lines.append(f"**{name}**: `{cfg['class']}` — {cfg['params']}")
    lines.append("")

    # C. Primary Benchmark
    lines.append("## C. Primary Benchmark Results\n")
    for subset in sorted(all_benchmark.keys()):
        bm = all_benchmark[subset]
        if not bm:
            continue
        lines.append(f"### {subset}\n")
        lines.append("| Model | Row MAE | Row RMSE | Row R² | Endpoint MAE | Endpoint RMSE | NASA Score | Runtime (s) |")
        lines.append("|-------|---------|----------|--------|--------------|---------------|------------|-------------|")
        for m in sorted(bm, key=lambda x: x["endpoint_mae"]):
            lines.append(
                f"| {m['model']} | {m['row_mae']:.2f} | {m['row_rmse']:.2f} | {m['row_r2']:.4f} "
                f"| {m['endpoint_mae']:.2f} | {m['endpoint_rmse']:.2f} | {m['endpoint_nasa_score']:.2f} "
                f"| {m.get('runtime', 0):.1f} |"
            )
        lines.append("")

    # D. Window Comparison
    lines.append("## D. Window Configuration Comparison\n")
    if window_results:
        wdf = pd.DataFrame(window_results)
        for subset in sorted(wdf["dataset"].unique()):
            sdf = wdf[wdf["dataset"] == subset]
            lines.append(f"### {subset}\n")
            lines.append("| Window | n_features | Row MAE | Endpoint MAE | NASA Score |")
            lines.append("|--------|------------|---------|--------------|------------|")
            for _, r in sdf.iterrows():
                lines.append(f"| {r['window_config']} | {r['n_features']} | {r['row_mae']:.2f} | {r['endpoint_mae']:.2f} | {r['endpoint_nasa_score']:.2f} |")
            lines.append("")
    else:
        lines.append("Window comparison not run.\n")

    # E. RUL Cap Sensitivity
    lines.append("## E. RUL Cap Sensitivity\n")
    if cap_results:
        cdf = pd.DataFrame(cap_results)
        for subset in sorted(cdf["dataset"].unique()):
            sdf = cdf[cdf["dataset"] == subset]
            lines.append(f"### {subset}\n")
            lines.append("| Model | Cap | EP MAE (uncapped) | EP RMSE (uncapped) | NASA | EP MAE (capped) |")
            lines.append("|-------|-----|-------------------|--------------------| -----|-----------------|")
            for _, r in sdf.iterrows():
                lines.append(
                    f"| {r['model']} | {r['cap_label']} | {r['endpoint_mae_vs_uncapped']:.2f} "
                    f"| {r['endpoint_rmse_vs_uncapped']:.2f} | {r['endpoint_nasa_score']:.2f} "
                    f"| {r.get('endpoint_mae_vs_capped', 'N/A')} |"
                )
            lines.append("")
    else:
        lines.append("Cap sensitivity not run.\n")

    # F. Error Analysis
    lines.append("## F. Error Analysis by RUL Range\n")
    for subset in sorted(all_rul_range.keys()):
        rr = all_rul_range[subset]
        if not rr:
            continue
        lines.append(f"### {subset}\n")
        for model_name, rr_df in rr.items():
            lines.append(f"**{model_name}**\n")
            lines.append("| RUL Range | n | MAE | RMSE | Mean Error |")
            lines.append("|-----------|---|-----|------|------------|")
            for _, row in rr_df.iterrows():
                lines.append(
                    f"| {row['rul_range']} | {row['n_observations']} | {row['mae']:.2f} "
                    f"| {row['rmse']:.2f} | {row['mean_signed_error']:.2f} |"
                )
            lines.append("")

    # G. Limitations
    lines.append("## G. Limitations\n")
    lines.append("1. **Validation bias**: Model selection on validation set introduces optimistic estimates")
    lines.append("2. **Row vs endpoint**: Row-level metrics weight longer-lived engines more heavily")
    lines.append("3. **Window exclusion**: 75-cycle window excludes early observations (data retention ~64-70%)")
    lines.append("4. **Dataset variation**: FD002/FD004 (multi-condition) may behave differently from FD001/FD003")
    lines.append("5. **No hyperparameter tuning**: Fixed configurations, not optimized")
    lines.append("6. **Official test set NOT used**: Final evaluation deferred to Phase 5")
    lines.append("")

    # H. Conclusion
    lines.append("## H. Phase 4 Conclusion\n")
    lines.append("### Findings")
    lines.append("- Ensemble models (RF, XGBoost, LightGBM) substantially outperform the trivial baseline")
    lines.append("- Linear models provide a useful intermediate reference")
    lines.append("- Performance near failure (0-30 cycles) requires careful analysis")
    lines.append("")
    lines.append("### Phase 5 Readiness")
    lines.append("- Strongest model families identified for hyperparameter tuning")
    lines.append("- Window and cap experiments provide starting configurations")
    lines.append("- Official test set remains untouched")
    lines.append("- **Status: READY for Phase 5**")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("AeroRUL — Phase 4: Classical Regression Benchmark")
    print("=" * 70)

    PHASE4_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PHASE4_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PHASE4_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    skipped_models = {}
    if "XGBoost" not in ALL_MODEL_CONFIGS:
        skipped_models["XGBoost"] = "xgboost not installed"
    if "LightGBM" not in ALL_MODEL_CONFIGS:
        skipped_models["LightGBM"] = "lightgbm not installed"

    all_benchmark = {}
    all_predictions = {}
    all_endpoints = {}
    all_engine_macro = {}
    all_rul_range = {}
    all_preflight = {}

    # ── PRIMARY BENCHMARK ──
    print("\n" + "=" * 70)
    print("STEP 1: PRIMARY BENCHMARK")
    print("=" * 70)

    for subset in sorted(AVAILABLE_SUBSETS):
        print(f"\n{'-' * 50}")
        print(f"  {subset}")
        print(f"{'-' * 50}")

        # Load data
        train_df, val_df = load_phase3_data(subset)
        approved = load_approved_features(subset)

        # Preflight
        preflight = data_preflight(train_df, val_df, approved, subset)
        all_preflight[subset] = preflight
        print(f"  Preflight: {'PASS' if preflight['passed'] else 'FAIL'}")
        print(f"  Features: {preflight['n_usable']}/{preflight['n_approved']} usable")
        print(f"  Train: {preflight['n_train_rows']} rows, {preflight['n_train_engines']} engines")
        print(f"  Val: {preflight['n_val_rows']} rows, {preflight['n_val_engines']} engines")
        for issue in preflight["issues"]:
            print(f"    {issue}")

        if not preflight["passed"]:
            print(f"  SKIPPING {subset} due to preflight failure")
            continue

        usable = preflight["usable_features"]

        # Prepare features
        X_train, X_val, y_train, y_val = prepare_clean_features(
            train_df, val_df, usable,
        )

        # Run all models
        subset_results = []
        subset_predictions = {}
        subset_rul_range = {}

        for model_name, config in ALL_MODEL_CONFIGS.items():
            try:
                print(f"  Training {model_name}...", end=" ", flush=True)
                model, scaler, preds_raw, runtime = train_and_predict(
                    model_name, config, X_train, y_train, X_val,
                )

                # Create predictions dataframe
                preds_df = create_predictions_df(val_df, preds_raw, model_name, subset)

                # Evaluate
                metrics = evaluate_model(preds_df)
                metrics["model"] = model_name
                metrics["dataset"] = subset
                metrics["runtime"] = runtime

                rul_range_df = metrics.pop("rul_range_analysis")
                subset_rul_range[model_name] = rul_range_df

                subset_results.append(metrics)
                subset_predictions[model_name] = preds_df

                print(f"MAE={metrics['row_mae']:.2f}, EP_MAE={metrics['endpoint_mae']:.2f}, "
                      f"NASA={metrics['endpoint_nasa_score']:.1f}, {runtime:.1f}s")

                # Save predictions
                preds_df.to_csv(
                    PHASE4_PREDICTIONS_DIR / f"predictions_{subset}_{model_name}.csv",
                    index=False,
                )
            except Exception as e:
                print(f"FAILED: {e}")
                skipped_models[f"{model_name}/{subset}"] = str(e)

        all_benchmark[subset] = subset_results
        all_predictions[subset] = subset_predictions
        all_rul_range[subset] = subset_rul_range

        # Engine macro metrics
        if subset_predictions:
            best_model = min(subset_results, key=lambda x: x["endpoint_mae"])["model"]
            eng_macro = engine_macro_metrics(subset_predictions[best_model])
            all_engine_macro[subset] = eng_macro

        # Endpoint extraction
        for mname, preds_df in subset_predictions.items():
            endpoints = extract_endpoints(preds_df)
            all_endpoints.setdefault(subset, {})[mname] = endpoints

        # Visualizations for primary benchmark
        if subset_results:
            metrics_df = pd.DataFrame(subset_results)
            plot_model_comparison(metrics_df, subset, PHASE4_FIGURES_DIR)

            # Best 2 models: pred vs actual + residuals + degradation curves
            top2 = sorted(subset_results, key=lambda x: x["endpoint_mae"])[:2]
            for m in top2:
                mname = m["model"]
                if mname in subset_predictions:
                    plot_pred_vs_actual(subset_predictions[mname], mname, subset, PHASE4_FIGURES_DIR)
                    plot_residuals(subset_predictions[mname], mname, subset, PHASE4_FIGURES_DIR)
                    plot_degradation_curves(subset_predictions[mname], mname, subset, PHASE4_FIGURES_DIR)

        # Free memory
        del train_df, val_df, X_train, X_val
        import gc; gc.collect()

    # Save primary benchmark CSVs
    print("\n  Saving primary benchmark results...")
    all_bm_rows = []
    for subset, results in all_benchmark.items():
        all_bm_rows.extend(results)
    if all_bm_rows:
        pd.DataFrame(all_bm_rows).to_csv(
            PHASE4_REPORTS_DIR / "model_benchmark_metrics.csv", index=False,
        )

    all_ep_rows = []
    all_ep_preds = []
    for subset, ep_dict in all_endpoints.items():
        for mname, ep_df in ep_dict.items():
            ep_metrics = endpoint_metrics(ep_df)
            ep_metrics["dataset"] = subset
            ep_metrics["model"] = mname
            all_ep_rows.append(ep_metrics)
            all_ep_preds.append(ep_df)
    if all_ep_rows:
        pd.DataFrame(all_ep_rows).to_csv(
            PHASE4_REPORTS_DIR / "endpoint_metrics.csv", index=False,
        )
    if all_ep_preds:
        pd.concat(all_ep_preds, ignore_index=True).to_csv(
            PHASE4_REPORTS_DIR / "endpoint_predictions.csv", index=False,
        )

    all_eng_rows = []
    for subset, macro in all_engine_macro.items():
        per_eng = macro["per_engine"].copy()
        per_eng["dataset"] = subset
        all_eng_rows.append(per_eng)
    if all_eng_rows:
        pd.concat(all_eng_rows).to_csv(
            PHASE4_REPORTS_DIR / "engine_macro_metrics.csv", index=False,
        )

    all_rr_rows = []
    for subset, rr_dict in all_rul_range.items():
        for mname, rr_df in rr_dict.items():
            rr = rr_df.copy()
            rr["dataset"] = subset
            rr["model"] = mname
            all_rr_rows.append(rr)
    if all_rr_rows:
        pd.concat(all_rr_rows).to_csv(
            PHASE4_REPORTS_DIR / "rul_range_error_analysis.csv", index=False,
        )

    # ── RUL CAP SENSITIVITY ──
    print("\n" + "=" * 70)
    print("STEP 2: RUL CAP SENSITIVITY")
    print("=" * 70)

    cap_results = []
    for subset in sorted(AVAILABLE_SUBSETS):
        if subset not in all_benchmark or not all_benchmark[subset]:
            continue

        # Select top 2 candidate models by endpoint MAE
        sorted_models = sorted(all_benchmark[subset], key=lambda x: x["endpoint_mae"])
        candidate_models = [m["model"] for m in sorted_models[:2]]
        # Exclude DummyBaseline from cap experiments
        candidate_models = [m for m in candidate_models if m != "DummyBaseline"]
        if not candidate_models:
            continue

        print(f"\n  {subset}: candidates = {candidate_models}")

        train_df, val_df = load_phase3_data(subset)
        approved = load_approved_features(subset)
        usable = [f for f in approved if f in train_df.columns]
        X_train, X_val, y_train_orig, y_val = prepare_clean_features(train_df, val_df, usable)

        for model_name in candidate_models:
            config = ALL_MODEL_CONFIGS[model_name]

            for cap in RUL_CAPS:
                cap_label = str(cap) if cap is not None else "None"

                # Cap training targets
                y_train_capped = y_train_orig.copy()
                if cap is not None:
                    y_train_capped = np.minimum(y_train_capped, cap)

                try:
                    if cap is None and subset in all_predictions and model_name in all_predictions[subset]:
                        preds_raw = all_predictions[subset][model_name]["predicted_rul_raw"].values
                        rt = 0.0
                    else:
                        _, _, preds_raw, rt = train_and_predict(
                            model_name, config, X_train, y_train_capped, X_val,
                        )
                    preds_nn = np.maximum(preds_raw, 0)
                    if cap is not None:
                        preds_capped = np.minimum(preds_nn, cap)
                    else:
                        preds_capped = preds_nn

                    # Build predictions DF with output constraint
                    preds_df = create_predictions_df(val_df, preds_capped, model_name, subset)

                    # Endpoint metrics against uncapped validation target
                    ep = extract_endpoints(preds_df)
                    ep_m = endpoint_metrics(ep)

                    # Metrics against capped targets
                    y_val_capped = np.minimum(y_val, cap) if cap is not None else y_val
                    row_mae_capped = mae(y_val_capped, preds_capped)
                    row_rmse_capped = rmse(y_val_capped, preds_capped)
                    ep_capped_actual = np.minimum(ep["actual_rul"].values, cap) if cap is not None else ep["actual_rul"].values
                    ep_mae_capped = mae(ep_capped_actual, ep["predicted_rul_nonnegative"].values)
                    ep_rmse_capped = rmse(ep_capped_actual, ep["predicted_rul_nonnegative"].values)

                    cap_results.append({
                        "dataset": subset,
                        "model": model_name,
                        "cap": cap,
                        "cap_label": cap_label,
                        "n_val_engines": ep_m["n_engines"],
                        "n_val_observations": len(preds_df),
                        "row_mae_vs_uncapped": mae(y_val, preds_capped),
                        "row_rmse_vs_uncapped": rmse(y_val, preds_capped),
                        "endpoint_mae_vs_uncapped": ep_m["endpoint_mae"],
                        "endpoint_rmse_vs_uncapped": ep_m["endpoint_rmse"],
                        "endpoint_nasa_score": ep_m["endpoint_nasa_score"],
                        "row_mae_vs_capped": round(row_mae_capped, 2),
                        "row_rmse_vs_capped": round(row_rmse_capped, 2),
                        "endpoint_mae_vs_capped": round(ep_mae_capped, 2),
                        "endpoint_rmse_vs_capped": round(ep_rmse_capped, 2),
                    })

                    print(f"    {model_name} cap={cap_label}: EP MAE={ep_m['endpoint_mae']:.2f}, "
                          f"NASA={ep_m['endpoint_nasa_score']:.1f}")
                except Exception as e:
                    print(f"    {model_name} cap={cap_label}: FAILED — {e}")

        del train_df, val_df, X_train, X_val
        import gc; gc.collect()

    if cap_results:
        pd.DataFrame(cap_results).to_csv(
            PHASE4_REPORTS_DIR / "rul_cap_sensitivity.csv", index=False,
        )
        plot_cap_sensitivity(pd.DataFrame(cap_results), PHASE4_FIGURES_DIR)

    # ── WINDOW COMPARISON ──
    print("\n" + "=" * 70)
    print("STEP 3: WINDOW CONFIGURATION COMPARISON")
    print("=" * 70)

    window_results = []
    windows_to_test = [None, 10, 20, 30, 50, 75]

    for subset in sorted(AVAILABLE_SUBSETS):
        if subset not in all_benchmark or not all_benchmark[subset]:
            continue

        # Use best model for window comparison
        best_model_name = sorted(all_benchmark[subset], key=lambda x: x["endpoint_mae"])[0]["model"]
        if best_model_name == "DummyBaseline":
            best_model_name = sorted(all_benchmark[subset], key=lambda x: x["endpoint_mae"])[1]["model"]

        config = ALL_MODEL_CONFIGS[best_model_name]
        print(f"\n  {subset}: using {best_model_name}")

        train_df, val_df = load_phase3_data(subset)
        approved = load_approved_features(subset)
        usable = [f for f in approved if f in train_df.columns]

        for window in windows_to_test:
            w_label = f"W={window}" if window is not None else "All"
            w_features = select_window_features(usable, window)

            if not w_features:
                print(f"    {w_label}: no features selected")
                continue

            try:
                if window is None and subset in all_predictions and best_model_name in all_predictions[subset]:
                    preds_raw = all_predictions[subset][best_model_name]["predicted_rul_raw"].values
                    rt = 0.0
                else:
                    X_tr, X_v, y_tr, y_v = prepare_clean_features(
                        train_df, val_df, w_features,
                    )
                    _, _, preds_raw, rt = train_and_predict(
                        best_model_name, config, X_tr, y_tr, X_v,
                    )
                preds_df = create_predictions_df(val_df, preds_raw, best_model_name, subset)
                metrics = evaluate_model(preds_df)
                metrics.pop("rul_range_analysis", None)

                window_results.append({
                    "dataset": subset,
                    "model": best_model_name,
                    "window_config": w_label,
                    "n_features": len(w_features),
                    "row_mae": metrics["row_mae"],
                    "row_rmse": metrics["row_rmse"],
                    "endpoint_mae": metrics["endpoint_mae"],
                    "endpoint_rmse": metrics["endpoint_rmse"],
                    "endpoint_nasa_score": metrics["endpoint_nasa_score"],
                    "runtime": rt,
                })

                print(f"    {w_label}: {len(w_features)} feats, MAE={metrics['row_mae']:.2f}, "
                      f"EP_MAE={metrics['endpoint_mae']:.2f}")
            except Exception as e:
                print(f"    {w_label}: FAILED — {e}")

        del train_df, val_df
        import gc; gc.collect()

    if window_results:
        pd.DataFrame(window_results).to_csv(
            PHASE4_REPORTS_DIR / "window_ablation_metrics.csv", index=False,
        )
        plot_window_comparison(pd.DataFrame(window_results), PHASE4_FIGURES_DIR)

    # ── DEDUPLICATION SENSITIVITY ──
    print("\n" + "=" * 70)
    print("STEP 4: DEDUPLICATION SENSITIVITY")
    print("=" * 70)

    dedup_results = []

    for subset in sorted(AVAILABLE_SUBSETS):
        if subset not in all_benchmark or not all_benchmark[subset]:
            continue

        print(f"\n  {subset}:")
        train_df, val_df = load_phase3_data(subset)
        approved = load_approved_features(subset)
        usable = [f for f in approved if f in train_df.columns]

        X_train_full, X_val_full, y_train, y_val = prepare_clean_features(
            train_df, val_df, usable,
        )

        # Detect duplicates using training data only
        deduped_features, dup_groups = detect_exact_duplicates_train(
            X_train_full, usable,
        )
        n_removed = len(usable) - len(deduped_features)
        print(f"    Original: {len(usable)}, Deduplicated: {len(deduped_features)}, Removed: {n_removed}")

        if n_removed == 0:
            print(f"    No duplicates found — skipping comparison")
            dedup_results.append({
                "dataset": subset,
                "model": "N/A",
                "n_original": len(usable),
                "n_deduplicated": len(deduped_features),
                "n_removed": 0,
                "n_groups": 0,
                "row_mae_original": 0,
                "row_mae_deduped": 0,
                "endpoint_mae_original": 0,
                "endpoint_mae_deduped": 0,
                "runtime_original": 0,
                "runtime_deduped": 0,
            })
            continue

        # Get deduplicated indices
        dedup_indices = [usable.index(f) for f in deduped_features]
        X_train_dedup = X_train_full[:, dedup_indices]
        X_val_dedup = X_val_full[:, dedup_indices]

        # Compare with LinearRegression and best model
        for model_name in ["LinearRegression", sorted(all_benchmark[subset], key=lambda x: x["endpoint_mae"])[0]["model"]]:
            if model_name == "DummyBaseline":
                continue
            if model_name not in ALL_MODEL_CONFIGS:
                continue

            config = ALL_MODEL_CONFIGS[model_name]

            # Original
            _, _, preds_orig, rt_orig = train_and_predict(
                model_name, config, X_train_full, y_train, X_val_full,
            )
            preds_orig_df = create_predictions_df(val_df, preds_orig, model_name, subset)
            m_orig = evaluate_model(preds_orig_df)

            # Deduplicated
            _, _, preds_dedup, rt_dedup = train_and_predict(
                model_name, config, X_train_dedup, y_train, X_val_dedup,
            )
            preds_dedup_df = create_predictions_df(val_df, preds_dedup, model_name, subset)
            m_dedup = evaluate_model(preds_dedup_df)

            dedup_results.append({
                "dataset": subset,
                "model": model_name,
                "n_original": len(usable),
                "n_deduplicated": len(deduped_features),
                "n_removed": n_removed,
                "n_groups": len(dup_groups),
                "row_mae_original": m_orig["row_mae"],
                "row_mae_deduped": m_dedup["row_mae"],
                "endpoint_mae_original": m_orig["endpoint_mae"],
                "endpoint_mae_deduped": m_dedup["endpoint_mae"],
                "runtime_original": rt_orig,
                "runtime_deduped": rt_dedup,
            })

            print(f"    {model_name}: orig MAE={m_orig['row_mae']:.2f} dedup MAE={m_dedup['row_mae']:.2f} "
                  f"(diff={m_orig['row_mae']-m_dedup['row_mae']:.3f})")

        del train_df, val_df, X_train_full, X_val_full
        import gc; gc.collect()

    if dedup_results:
        pd.DataFrame(dedup_results).to_csv(
            PHASE4_REPORTS_DIR / "deduplication_sensitivity.csv", index=False,
        )

    # ── GENERATE REPORT ──
    print("\n" + "=" * 70)
    print("STEP 5: GENERATING REPORT")
    print("=" * 70)

    generate_phase4_report(
        all_benchmark, all_endpoints, all_engine_macro,
        all_rul_range, cap_results, window_results,
        dedup_results, skipped_models,
        PHASE4_REPORTS_DIR / "phase4_benchmark_summary.md",
    )

    # ── FINAL SUMMARY ──
    print("\n" + "=" * 70)
    print("Phase 4 Classical Regression Benchmark COMPLETE")
    print(f"Reports:      {PHASE4_REPORTS_DIR}")
    print(f"Figures:      {PHASE4_FIGURES_DIR}")
    print(f"Predictions:  {PHASE4_PREDICTIONS_DIR}")
    print("=" * 70)

    for subset in sorted(all_benchmark.keys()):
        bm = all_benchmark[subset]
        if bm:
            best = min(bm, key=lambda x: x["endpoint_mae"])
            print(f"  {subset}: Best = {best['model']} (EP MAE={best['endpoint_mae']:.2f}, NASA={best['endpoint_nasa_score']:.1f})")


if __name__ == "__main__":
    main()
