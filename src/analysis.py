"""
Exploratory data analysis module for computing missingness, correlations,
and saving publication-quality visual plots.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless / CLI execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR

logger = logging.getLogger(__name__)


def compute_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes count and percentage of missing values per column.
    """
    total = len(df)
    missing_cnt = df.isnull().sum()
    missing_pct = (missing_cnt / total) * 100.0
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": missing_cnt.values,
        "missing_percentage": missing_pct.values
    }).sort_values(by="missing_percentage", ascending=False).reset_index(drop=True)
    return summary


def compute_correlations(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    method: str = "pearson"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Computes correlation matrix and feature correlations with the target.
    Filters out constant/zero-variance features to prevent divide-by-zero warnings.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    valid_cols = [
        c for c in numeric_df.columns
        if numeric_df[c].nunique(dropna=True) > 1 and numeric_df[c].std() > 1e-12
    ]
    numeric_df = numeric_df[valid_cols]
    corr_matrix = numeric_df.corr(method=method)
    
    target_corr = pd.Series(dtype=float)
    if target_col and target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].drop(index=[target_col]).dropna().sort_values(ascending=False)
        
    return corr_matrix, target_corr


def plot_missingness(
    df: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.60
) -> None:
    """
    Generates and saves a missingness percentage bar chart.
    """
    summary = compute_missingness_summary(df)
    # Filter to columns with at least some missingness or top 30
    to_plot = summary[summary["missing_percentage"] > 0]
    if len(to_plot) == 0:
        to_plot = summary.head(15)
    else:
        to_plot = to_plot.head(30)

    fig, ax = plt.subplots(figsize=(12, max(6, len(to_plot) * 0.35)), dpi=300)
    bars = ax.barh(to_plot["column"], to_plot["missing_percentage"], color="#3b82f6", edgecolor="#1d4ed8")
    ax.axvline(threshold * 100, color="#ef4444", linestyle="--", linewidth=1.5, label=f"Drop Threshold ({int(threshold*100)}%)")
    
    ax.set_xlabel("Missing Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Feature Missingness Analysis", fontsize=14, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    for bar in bars:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved missingness plot to {output_path}")


def plot_outlier_boxplots(
    df: pd.DataFrame,
    output_path: Path,
    max_features: int = 12
) -> None:
    """
    Generates and saves boxplots for key continuous numerical features.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if df[c].nunique() > 5 and not c.startswith("ts")
    ][:max_features]

    if not numeric_cols:
        return

    num_cols = len(numeric_cols)
    cols_per_row = 3
    num_rows = int(np.ceil(num_cols / cols_per_row))

    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 3.5 * num_rows), dpi=300)
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#60a5fa", color="#1d4ed8"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_ylabel("")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    # Hide unused subplots
    for j in range(num_cols, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Outlier Distribution of Key Numerical Features", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved outlier boxplots to {output_path}")


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    output_path: Path,
    top_n: int = 20
) -> None:
    """
    Generates and saves a high-res correlation heatmap for top features.
    """
    if corr_matrix.empty:
        return

    # Select top N most variable features or top features
    cols = corr_matrix.columns[:top_n]
    sub_corr = corr_matrix.loc[cols, cols]

    fig, ax = plt.subplots(figsize=(14, 12), dpi=300)
    mask = np.triu(np.ones_like(sub_corr, dtype=bool))
    
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        sub_corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        annot=len(cols) <= 15,
        fmt=".2f",
        annot_kws={"size": 7},
        ax=ax
    )
    ax.set_title(f"Pairwise Feature Correlation Heatmap (Top {len(cols)} Features)", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved correlation heatmap to {output_path}")


def plot_feature_target_correlation(
    target_corr: pd.Series,
    target_col: str,
    output_path: Path,
    top_n: int = 15
) -> None:
    """
    Generates and saves a horizontal bar chart of feature correlations with the target.
    """
    if target_corr.empty:
        return

    # Take top positive and top negative
    top_pos = target_corr.head(top_n // 2)
    top_neg = target_corr.tail(top_n // 2)
    combined = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values()

    fig, ax = plt.subplots(figsize=(10, max(5, len(combined) * 0.4)), dpi=300)
    colors = ["#ef4444" if val < 0 else "#10b981" for val in combined.values]
    bars = ax.barh(combined.index, combined.values, color=colors, edgecolor="#374151", linewidth=0.5)

    ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xlabel(f"Pearson Correlation with '{target_col}'", fontsize=11, fontweight="bold")
    ax.set_title(f"Top Predictive Feature Correlations with Target ({target_col})", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-1.05, 1.05)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

    for bar in bars:
        w = bar.get_width()
        offset = 0.03 if w >= 0 else -0.08
        ax.text(w + offset, bar.get_y() + bar.get_height() / 2, f"{w:.2f}", va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feature-target plot to {output_path}")


def run_full_analysis(
    df: pd.DataFrame,
    target_col: Optional[str] = "power_kw",
    plots_dir: Optional[Path] = None,
    missing_threshold: float = 0.60
) -> Dict[str, Any]:
    """
    Executes full EDA and saves all required PNG figures.
    """
    save_dir = Path(plots_dir or PLOTS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_df = compute_missingness_summary(df)
    missingness_dict = summary_df.set_index("column")["missing_percentage"].to_dict()

    corr_matrix, target_corr = compute_correlations(df, target_col=target_col)
    spearman_matrix, _ = compute_correlations(df, target_col=target_col, method="spearman")

    # Generate plots
    plot_missingness(df, save_dir / "missingness_bar_chart.png", threshold=missing_threshold)
    plot_outlier_boxplots(df, save_dir / "outlier_boxplots.png")
    plot_correlation_heatmap(corr_matrix, save_dir / "correlation_heatmap.png")
    
    if target_col and not target_corr.empty:
        plot_feature_target_correlation(target_corr, target_col, save_dir / "feature_target_bar_plot.png")

    top_correlations = []
    if not target_corr.empty:
        for feat, val in target_corr.head(10).items():
            top_correlations.append({"feature": feat, "pearson_r": round(float(val), 4)})

    return {
        "missingness_summary": {k: round(v, 2) for k, v in missingness_dict.items() if v > 0},
        "top_features_correlated_with_target": top_correlations,
        "plots_saved": [
            str(save_dir / "missingness_bar_chart.png"),
            str(save_dir / "outlier_boxplots.png"),
            str(save_dir / "correlation_heatmap.png"),
            str(save_dir / "feature_target_bar_plot.png"),
        ]
    }
