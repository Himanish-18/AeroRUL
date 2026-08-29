"""
AeroRUL — Phase 1 Visualizations

Publication-quality plots for C-MAPSS dataset exploration.
All plotting functions save figures to disk and optionally return the figure object.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    FIGURE_DPI,
    FIGURE_FORMAT,
    FIGURE_STYLE,
    FIGURES_DIR,
    OP_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)


def _ensure_dir(directory: Path) -> Path:
    """Create directory if it doesn't exist."""
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _apply_style():
    """Apply the project-wide matplotlib style."""
    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


# ═══════════════════════════════════════════════════════════════════════════
# 1. ENGINE TRAJECTORY LENGTH HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════

def plot_trajectory_lengths(
    all_data: dict,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot histogram of training engine trajectory lengths for each subset.

    Parameters
    ----------
    all_data : dict
        Keyed by subset name, each value has {"train": DataFrame, ...}
    output_dir : Path, optional
        Directory to save the figure.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    n_subsets = len(all_data)
    fig, axes = plt.subplots(1, n_subsets, figsize=(5 * n_subsets, 4), squeeze=False)
    axes = axes.flatten()

    for i, (subset, data) in enumerate(sorted(all_data.items())):
        train_df = data["train"]
        lengths = train_df.groupby("unit_id")["cycle"].max()

        axes[i].hist(lengths, bins=30, edgecolor="black", alpha=0.75, color=sns.color_palette("husl", n_subsets)[i])
        axes[i].set_title(f"{subset} — Training Trajectory Lengths", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Maximum Cycle (Trajectory Length)")
        axes[i].set_ylabel("Number of Engines")
        axes[i].axvline(lengths.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {lengths.mean():.0f}")
        axes[i].axvline(lengths.median(), color="orange", linestyle=":", linewidth=1.5, label=f"Median: {lengths.median():.0f}")
        axes[i].legend(fontsize=9)

    fig.suptitle("Distribution of Training Engine Lifetimes", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f"trajectory_lengths.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. SENSOR DISTRIBUTIONS (COMPACT)
# ═══════════════════════════════════════════════════════════════════════════

def plot_sensor_distributions(
    train_df: pd.DataFrame,
    subset: str = "FD001",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a compact boxplot panel showing distributions of all 21 sensors.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    sensor_cols = [c for c in SENSOR_COLUMNS if c in train_df.columns]

    # Normalize sensors for comparable visualization
    sensor_data = train_df[sensor_cols].copy()
    sensor_norm = (sensor_data - sensor_data.mean()) / (sensor_data.std() + 1e-10)

    fig, ax = plt.subplots(figsize=(16, 6))
    sensor_norm.boxplot(ax=ax, vert=True, showfliers=False)
    ax.set_title(f"{subset} — Sensor Distributions (Standardized)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Standardized Value")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(output_dir / f"sensor_distributions_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. SENSOR CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════

def plot_sensor_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    subset: str = "FD001",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot a heatmap of the sensor × sensor correlation matrix.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
        ax=ax,
    )
    ax.set_title(f"{subset} — Sensor Correlation Matrix", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / f"sensor_correlation_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. SENSOR VS CYCLE TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════

def plot_sensor_trajectories(
    train_df: pd.DataFrame,
    subset: str = "FD001",
    n_engines: int = 5,
    sensors: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot sensor readings vs cycle for representative engines.

    Selects informative sensors (high variance, correlated with cycle)
    unless specific sensors are provided.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    # Default: pick sensors with highest variance (more informative)
    if sensors is None:
        sensor_vars = train_df[SENSOR_COLUMNS].var().sort_values(ascending=False)
        # Pick top 6 variable sensors
        sensors = sensor_vars.head(6).index.tolist()

    # Pick representative engines (spread across the dataset)
    engine_ids = sorted(train_df["unit_id"].unique())
    step = max(1, len(engine_ids) // n_engines)
    selected_engines = engine_ids[::step][:n_engines]

    n_sensors = len(sensors)
    fig, axes = plt.subplots(n_sensors, 1, figsize=(14, 3 * n_sensors), sharex=False)
    if n_sensors == 1:
        axes = [axes]

    colors = sns.color_palette("husl", n_engines)

    for i, sensor in enumerate(sensors):
        for j, eid in enumerate(selected_engines):
            engine_data = train_df[train_df["unit_id"] == eid]
            axes[i].plot(
                engine_data["cycle"],
                engine_data[sensor],
                alpha=0.7,
                linewidth=0.8,
                color=colors[j],
                label=f"Engine {eid}" if i == 0 else None,
            )
        axes[i].set_ylabel(sensor, fontsize=10)
        axes[i].set_title(f"{sensor}", fontsize=11, fontweight="bold")

    axes[-1].set_xlabel("Cycle")
    # Add legend only to the first subplot
    axes[0].legend(loc="upper right", fontsize=8, ncol=n_engines)

    fig.suptitle(f"{subset} — Sensor Trajectories vs Cycle", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"sensor_trajectories_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. OPERATING CONDITION DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

def plot_operating_conditions(
    all_data: dict,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Show the distribution of the three operational settings across subsets.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    op_cols = OP_SETTING_COLUMNS
    n_subsets = len(all_data)

    fig, axes = plt.subplots(len(op_cols), n_subsets, figsize=(5 * n_subsets, 4 * len(op_cols)), squeeze=False)

    for j, (subset, data) in enumerate(sorted(all_data.items())):
        train_df = data["train"]
        for i, col in enumerate(op_cols):
            if col in train_df.columns:
                axes[i, j].hist(train_df[col], bins=50, edgecolor="black", alpha=0.75)
                axes[i, j].set_title(f"{subset} — {col}", fontsize=11, fontweight="bold")
                axes[i, j].set_xlabel(col)
                axes[i, j].set_ylabel("Frequency")

    fig.suptitle("Operating Condition Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"operating_conditions.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. ENGINE DEGRADATION EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════

def plot_degradation_examples(
    train_df: pd.DataFrame,
    subset: str = "FD001",
    n_engines: int = 4,
    sensors: Optional[list[str]] = None,
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot complete training trajectories showing degradation behavior.
    Shows 3 informative sensors for several engines.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    if sensors is None:
        # Pick 3 sensors with highest variance
        sensor_vars = train_df[SENSOR_COLUMNS].var().sort_values(ascending=False)
        sensors = sensor_vars.head(3).index.tolist()

    # Pick engines with diverse trajectory lengths
    lengths = train_df.groupby("unit_id")["cycle"].max().sort_values()
    idx = np.linspace(0, len(lengths) - 1, n_engines, dtype=int)
    selected_engines = lengths.index[idx].tolist()

    fig, axes = plt.subplots(n_engines, len(sensors), figsize=(6 * len(sensors), 3.5 * n_engines), squeeze=False)

    for i, eid in enumerate(selected_engines):
        engine_data = train_df[train_df["unit_id"] == eid]
        max_cycle = engine_data["cycle"].max()

        for j, sensor in enumerate(sensors):
            axes[i, j].plot(engine_data["cycle"], engine_data[sensor], linewidth=0.9, color="steelblue")
            axes[i, j].set_title(f"Engine {eid} (len={max_cycle}) — {sensor}", fontsize=10, fontweight="bold")
            if i == n_engines - 1:
                axes[i, j].set_xlabel("Cycle")
            if j == 0:
                axes[i, j].set_ylabel("Sensor Value")

    fig.suptitle(f"{subset} — Engine Degradation Examples", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"degradation_examples_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. SENSOR–CYCLE CORRELATION BAR CHART
# ═══════════════════════════════════════════════════════════════════════════

def plot_sensor_cycle_correlation(
    corr_df: pd.DataFrame,
    subset: str = "FD001",
    output_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of Pearson correlation between each sensor and cycle.
    """
    _apply_style()
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)

    corr_sorted = corr_df.sort_values("pearson_r")

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corr_sorted["pearson_r"]]
    ax.barh(corr_sorted["sensor"], corr_sorted["pearson_r"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pearson Correlation with Cycle", fontsize=12)
    ax.set_title(f"{subset} — Sensor–Cycle Correlation", fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_dir / f"sensor_cycle_correlation_{subset}.{FIGURE_FORMAT}", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# MASTER PLOTTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_phase1_plots(
    all_data: dict,
    validation_results: dict,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    """
    Generate all Phase 1 visualizations and return list of saved file paths.

    Parameters
    ----------
    all_data : dict
        Keyed by subset: {"FD001": {"train": df, "test": df, "rul": df}, ...}
    validation_results : dict
        Keyed by subset: {"FD001": validation_dict, ...}
    output_dir : Path, optional
        Override the default figures directory.
    """
    output_dir = _ensure_dir(output_dir or FIGURES_DIR)
    saved = []

    # 1. Trajectory lengths (all subsets)
    plot_trajectory_lengths(all_data, output_dir)
    saved.append(output_dir / f"trajectory_lengths.{FIGURE_FORMAT}")

    # 2–7. Per-subset plots
    for subset, data in sorted(all_data.items()):
        train_df = data["train"]
        vr = validation_results.get(subset, {})

        # 2. Sensor distributions
        plot_sensor_distributions(train_df, subset, output_dir)
        saved.append(output_dir / f"sensor_distributions_{subset}.{FIGURE_FORMAT}")

        # 3. Sensor correlation heatmap
        corr_matrix = vr.get("sensor_corr_matrix")
        if corr_matrix is not None:
            plot_sensor_correlation_heatmap(corr_matrix, subset, output_dir)
            saved.append(output_dir / f"sensor_correlation_{subset}.{FIGURE_FORMAT}")

        # 4. Sensor trajectories
        plot_sensor_trajectories(train_df, subset, output_dir=output_dir)
        saved.append(output_dir / f"sensor_trajectories_{subset}.{FIGURE_FORMAT}")

        # 5. Sensor–cycle correlation bar chart
        corr_df = vr.get("sensor_cycle_corr")
        if corr_df is not None:
            plot_sensor_cycle_correlation(corr_df, subset, output_dir)
            saved.append(output_dir / f"sensor_cycle_correlation_{subset}.{FIGURE_FORMAT}")

        # 6. Degradation examples
        plot_degradation_examples(train_df, subset, output_dir=output_dir)
        saved.append(output_dir / f"degradation_examples_{subset}.{FIGURE_FORMAT}")

    # 5. Operating conditions (all subsets)
    plot_operating_conditions(all_data, output_dir)
    saved.append(output_dir / f"operating_conditions.{FIGURE_FORMAT}")

    return saved
