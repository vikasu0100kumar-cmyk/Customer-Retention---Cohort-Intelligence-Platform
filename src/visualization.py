"""
Visualization module: publication-quality charts for cohort analysis.

All plots are saved to the output directory and optionally displayed.
Uses a consistent, professional theme throughout.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger("cohort_analysis")

# ---------------------------------------------------------------------------
# Theme Configuration
# ---------------------------------------------------------------------------

# Professional dark theme for all charts
THEME = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
}

# Color palettes
PALETTE_SEQUENTIAL = "rocket"
PALETTE_DIVERGING = "coolwarm"
COHORT_COLORS = [
    "#58a6ff", "#3fb950", "#f0883e", "#bc8cff",
    "#f778ba", "#79c0ff", "#56d364", "#d29922",
    "#db61a2", "#a5d6ff", "#7ee787", "#e3b341",
]


def _apply_theme():
    """Apply the project-wide chart theme."""
    plt.rcParams.update(THEME)
    sns.set_style("darkgrid", rc=THEME)


def _save_figure(fig: plt.Figure, filename: str, output_dir: str = "output"):
    """Save figure to output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    filepath = path / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    logger.info(f"Saved: {filepath}")
    plt.close(fig)


def _format_cohort_labels(labels) -> list:
    """Convert Period labels to readable strings."""
    return [str(label) for label in labels]


# ---------------------------------------------------------------------------
# Core Plots
# ---------------------------------------------------------------------------

def plot_cohort_sizes(
    cohort_sizes: pd.Series,
    output_dir: str = "output",
) -> None:
    """
    Bar chart of unique customers per cohort.

    Args:
        cohort_sizes: Series indexed by cohort_month.
        output_dir: Directory to save the plot.
    """
    _apply_theme()
    fig, ax = plt.subplots(figsize=(14, 6))

    labels = _format_cohort_labels(cohort_sizes.index)
    bars = ax.bar(
        labels,
        cohort_sizes.values,
        color=COHORT_COLORS[: len(labels)],
        edgecolor="#30363d",
        linewidth=0.8,
        zorder=3,
    )

    # Value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + cohort_sizes.max() * 0.02,
            f"{int(height):,}",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#c9d1d9",
        )

    ax.set_title("Customers per Cohort (First Purchase Month)", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Cohort (First Purchase Month)")
    ax.set_ylabel("Unique Customers")
    ax.set_ylim(0, cohort_sizes.max() * 1.15)
    plt.xticks(rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    _save_figure(fig, "01_cohort_sizes.png", output_dir)


def plot_retention_heatmap(
    matrix: pd.DataFrame,
    title: str,
    filename: str,
    fmt: str = ".1%",
    cmap: str = "YlOrRd",
    vmin: float = 0.0,
    vmax: float = None,
    output_dir: str = "output",
) -> None:
    """
    Heatmap for retention or activity rate matrices.

    Args:
        matrix: Cohort × period matrix with values.
        title: Chart title.
        filename: Output filename.
        fmt: Number format string.
        cmap: Colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        output_dir: Directory to save the plot.
    """
    _apply_theme()

    # Format index labels
    display = matrix.copy()
    display.index = _format_cohort_labels(display.index)

    fig, ax = plt.subplots(figsize=(16, max(6, len(display) * 0.8)))

    sns.heatmap(
        display,
        annot=True,
        annot_kws={"size": 9, "fontweight": "medium"},
        fmt=fmt,
        linewidths=0.5,
        linecolor="#21262d",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={
            "label": title.split(":")[0] if ":" in title else title,
            "shrink": 0.8,
        },
        ax=ax,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Months Since First Purchase", fontsize=12)
    ax.set_ylabel("Cohort (First Purchase Month)", fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    _save_figure(fig, filename, output_dir)


def plot_retention_trends(
    matrix: pd.DataFrame,
    title: str = "Retention Rate Trends by Cohort",
    filename: str = "retention_trends.png",
    output_dir: str = "output",
) -> None:
    """
    Line plot showing retention trends over time for each cohort.

    Args:
        matrix: Retention rate matrix (cohort × period).
        title: Chart title.
        filename: Output filename.
        output_dir: Directory to save the plot.
    """
    _apply_theme()
    fig, ax = plt.subplots(figsize=(14, 7))

    labels = _format_cohort_labels(matrix.index)

    for i, (cohort, row) in enumerate(matrix.iterrows()):
        values = row.dropna()
        color = COHORT_COLORS[i % len(COHORT_COLORS)]
        ax.plot(
            values.index, values.values,
            marker="o", markersize=5,
            linewidth=2, alpha=0.85,
            color=color,
            label=str(cohort),
        )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Months Since First Purchase", fontsize=12)
    ax.set_ylabel("Retention Rate", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(
        title="Cohort", bbox_to_anchor=(1.02, 1), loc="upper left",
        fontsize=9, title_fontsize=10, frameon=True,
        facecolor="#161b22", edgecolor="#30363d",
    )
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    _save_figure(fig, filename, output_dir)


def plot_clv_comparison(
    clv_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Bar chart comparing CLV metrics across cohorts.

    Args:
        clv_df: DataFrame with CLV metrics per cohort.
        output_dir: Directory to save the plot.
    """
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Customer Lifetime Value by Cohort",
        fontsize=18, fontweight="bold", y=1.02,
    )

    labels = _format_cohort_labels(clv_df.index)

    # --- Avg Revenue per Customer ---
    ax = axes[0, 0]
    colors = COHORT_COLORS[: len(labels)]
    bars = ax.bar(labels, clv_df["avg_revenue_per_customer"], color=colors, edgecolor="#30363d")
    ax.set_title("Avg Revenue per Customer", fontweight="bold")
    ax.set_ylabel("Revenue ($)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"${bar.get_height():,.0f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

    # --- Avg Orders per Customer ---
    ax = axes[0, 1]
    bars = ax.bar(labels, clv_df["avg_orders_per_customer"], color=colors, edgecolor="#30363d")
    ax.set_title("Avg Orders per Customer", fontweight="bold")
    ax.set_ylabel("Orders")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

    # --- Avg Order Value ---
    ax = axes[1, 0]
    bars = ax.bar(labels, clv_df["avg_order_value"], color=colors, edgecolor="#30363d")
    ax.set_title("Avg Order Value", fontweight="bold")
    ax.set_ylabel("Revenue ($)")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"${bar.get_height():,.0f}", ha="center", va="bottom",
                fontsize=8, color="#c9d1d9")
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

    # --- Total Revenue ---
    ax = axes[1, 1]
    bars = ax.bar(labels, clv_df["total_revenue"], color=colors, edgecolor="#30363d")
    ax.set_title("Total Revenue", fontweight="bold")
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    plt.sca(ax)
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()
    _save_figure(fig, "06_clv_comparison.png", output_dir)


def plot_revenue_heatmap(
    revenue_matrix: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Heatmap of total revenue by cohort and period.

    Args:
        revenue_matrix: Cohort × period → total revenue.
        output_dir: Directory to save the plot.
    """
    plot_retention_heatmap(
        revenue_matrix,
        title="Total Revenue by Cohort Over Time",
        filename="05_revenue_heatmap.png",
        fmt=",.0f",
        cmap="Greens",
        vmin=0,
        output_dir=output_dir,
    )


def plot_quantity_heatmap(
    quantity_matrix: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Heatmap of average order quantity by cohort and period.

    Args:
        quantity_matrix: Cohort × period → avg quantity.
        output_dir: Directory to save the plot.
    """
    plot_retention_heatmap(
        quantity_matrix,
        title="Avg Items per Order by Cohort",
        filename="07_quantity_heatmap.png",
        fmt=".1f",
        cmap="Purples",
        vmin=0,
        output_dir=output_dir,
    )


def plot_dashboard(
    activity_rates: pd.DataFrame,
    classic_ret: pd.DataFrame,
    cohort_sizes: pd.Series,
    clv_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Composite 2×2 dashboard with key metrics.

    Args:
        activity_rates: Activity rate matrix.
        classic_ret: Classic retention matrix.
        cohort_sizes: Customer count per cohort.
        clv_df: CLV metrics per cohort.
        output_dir: Directory to save the plot.
    """
    _apply_theme()
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle(
        "Cohort Analysis Dashboard",
        fontsize=20, fontweight="bold", y=1.01, color="#f0f6fc",
    )

    labels = _format_cohort_labels(cohort_sizes.index)
    colors = COHORT_COLORS[: len(labels)]

    # --- Top Left: Cohort Sizes ---
    ax = axes[0, 0]
    ax.bar(labels, cohort_sizes.values, color=colors, edgecolor="#30363d")
    ax.set_title("Customers per Cohort", fontweight="bold", fontsize=13)
    ax.set_ylabel("Unique Customers")
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontsize(8)
    ax.grid(axis="y", alpha=0.3)

    # --- Top Right: Activity Rate Heatmap ---
    ax = axes[0, 1]
    display_ar = activity_rates.copy()
    display_ar.index = _format_cohort_labels(display_ar.index)
    sns.heatmap(
        display_ar, annot=True, fmt=".0%", linewidths=0.4,
        cmap="YlOrRd", ax=ax, cbar_kws={"label": "Activity Rate", "shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_title("Activity Rate by Cohort", fontweight="bold", fontsize=13)
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("")

    # --- Bottom Left: Classic Retention Trends ---
    ax = axes[1, 0]
    for i, (cohort, row) in enumerate(classic_ret.iterrows()):
        vals = row.dropna()
        ax.plot(
            vals.index, vals.values,
            marker="o", markersize=4, linewidth=1.8,
            color=COHORT_COLORS[i % len(COHORT_COLORS)],
            label=str(cohort), alpha=0.85,
        )
    ax.set_title("Classic Retention Trends", fontweight="bold", fontsize=13)
    ax.set_xlabel("Months Since First Purchase")
    ax.set_ylabel("Retention Rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.legend(fontsize=7, loc="upper right", frameon=True,
              facecolor="#161b22", edgecolor="#30363d")
    ax.grid(alpha=0.3)

    # --- Bottom Right: CLV Comparison ---
    ax = axes[1, 1]
    ax.bar(labels, clv_df["avg_revenue_per_customer"], color=colors, edgecolor="#30363d")
    ax.set_title("Avg Revenue per Customer", fontweight="bold", fontsize=13)
    ax.set_ylabel("Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontsize(8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, "00_dashboard.png", output_dir)
    logger.info("Dashboard generated")


def generate_all_plots(
    cohort_sizes: pd.Series,
    activity_rates: pd.DataFrame,
    classic_ret: pd.DataFrame,
    rolling_ret: pd.DataFrame,
    revenue_matrix: pd.DataFrame,
    quantity_matrix: pd.DataFrame,
    clv_df: pd.DataFrame,
    output_dir: str = "output",
) -> None:
    """
    Generate all visualization outputs.

    Args:
        cohort_sizes: Customer counts per cohort.
        activity_rates: Activity rate matrix.
        classic_ret: Classic retention matrix.
        rolling_ret: Rolling retention matrix.
        revenue_matrix: Revenue by cohort × period.
        quantity_matrix: Avg quantity by cohort × period.
        clv_df: CLV metrics per cohort.
        output_dir: Directory to save all plots.
    """
    logger.info("=" * 60)
    logger.info("GENERATING ALL VISUALIZATIONS")
    logger.info("=" * 60)

    # 1. Cohort sizes bar chart
    plot_cohort_sizes(cohort_sizes, output_dir)

    # 2. Activity rate heatmap
    plot_retention_heatmap(
        activity_rates,
        title="Activity Rate: % of Cohort Active Each Month",
        filename="02_activity_rate_heatmap.png",
        cmap="YlOrRd",
        output_dir=output_dir,
    )

    # 3. Classic retention heatmap
    plot_retention_heatmap(
        classic_ret,
        title="Classic Retention: Period-over-Period",
        filename="03_classic_retention_heatmap.png",
        cmap="RdYlGn",
        output_dir=output_dir,
    )

    # 4. Rolling retention heatmap
    plot_retention_heatmap(
        rolling_ret,
        title="Rolling Retention: Active On or After Period",
        filename="04_rolling_retention_heatmap.png",
        cmap="Blues",
        output_dir=output_dir,
    )

    # 5. Revenue heatmap
    plot_revenue_heatmap(revenue_matrix, output_dir)

    # 6. CLV comparison
    plot_clv_comparison(clv_df, output_dir)

    # 7. Quantity heatmap
    plot_quantity_heatmap(quantity_matrix, output_dir)

    # 8. Retention trends (line plot)
    plot_retention_trends(
        activity_rates,
        title="Activity Rate Trends by Cohort",
        filename="08_activity_trends.png",
        output_dir=output_dir,
    )

    # 9. Classic retention trends (line plot)
    plot_retention_trends(
        classic_ret,
        title="Classic Retention Trends by Cohort",
        filename="09_classic_retention_trends.png",
        output_dir=output_dir,
    )

    # 10. Dashboard
    plot_dashboard(activity_rates, classic_ret, cohort_sizes, clv_df, output_dir)

    logger.info(f"All plots saved to {output_dir}/")
