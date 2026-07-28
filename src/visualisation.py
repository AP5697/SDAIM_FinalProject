"""All figure generation for EDA and model evaluation.

Every function saves to ``reports/figures`` and returns the written path, so
callers never construct paths themselves.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless backend - required for CI and script runs
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    auc,
    precision_recall_curve,
    roc_curve,
)

from src import config  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def apply_style() -> None:
    """Apply the project-wide matplotlib style, tolerating missing styles."""
    try:
        plt.style.use(config.PLOT_STYLE)
    except OSError:
        logger.warning("Style %s unavailable - falling back to ggplot", config.PLOT_STYLE)
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": config.FIGURE_DPI,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "font.size": 9,
        }
    )


def _save(fig: plt.Figure, name: str) -> Path:
    """Persist a figure to the figures directory and close it.

    Args:
        fig: Figure to write.
        name: File stem, without extension.

    Returns:
        Path to the saved image.
    """
    config.ensure_directories()
    path = config.FIGURES_DIR / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved figure -> %s", path.name)
    return path


# --------------------------------------------------------------------------- #
# Exploratory data analysis
# --------------------------------------------------------------------------- #


def plot_target_distribution(target: pd.Series) -> Path:
    """Plot the class balance of the target variable."""
    apply_style()
    counts = target.value_counts().sort_index()
    percentages = 100 * counts / counts.sum()
    labels = [config.NEGATIVE_CLASS_LABEL, config.POSITIVE_CLASS_LABEL]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bars = axes[0].bar(labels, counts.values, color=[config.NEGATIVE_COLOUR, config.PRIMARY_COLOUR])
    axes[0].set_title("Class distribution (counts)")
    axes[0].set_ylabel("Sessions")
    for bar, count, pct in zip(bars, counts.values, percentages.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            count,
            f"{count:,}\n({pct:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.2f%%",
        colors=[config.NEGATIVE_COLOUR, config.PRIMARY_COLOUR],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title("Class balance")
    fig.suptitle("Target: session ended in a purchase", fontweight="bold")
    return _save(fig, "01_target_distribution")


def plot_numeric_distributions(frame: pd.DataFrame, columns: list[str]) -> Path:
    """Plot histograms for numeric features to expose skew and zero-inflation."""
    apply_style()
    n_cols = 4
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, column in zip(axes, columns):
        ax.hist(frame[column].dropna(), bins=50, color=config.PRIMARY_COLOUR, alpha=0.85)
        skew = float(frame[column].skew())
        zero_pct = 100 * float((frame[column] == 0).mean())
        ax.set_title(f"{column}\nskew={skew:.2f} | zeros={zero_pct:.1f}%", fontsize=9)
        ax.set_yscale("log")

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    fig.suptitle(
        "Numeric feature distributions (log-scaled counts)", fontweight="bold", y=1.0
    )
    fig.tight_layout()
    return _save(fig, "02_numeric_distributions")


def plot_outlier_boxplots(frame: pd.DataFrame, columns: list[str]) -> Path:
    """Plot standardised boxplots to compare outlier prevalence across features."""
    apply_style()
    standardised = frame[columns].apply(
        lambda s: (s - s.mean()) / (s.std() if s.std() else 1.0)
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    standardised.boxplot(ax=ax, rot=45, grid=False)
    ax.set_title("Standardised feature distributions - outlier prevalence")
    ax.set_ylabel("Standard deviations from the mean")
    ax.axhline(0, color=config.NEGATIVE_COLOUR, linewidth=1)
    fig.tight_layout()
    return _save(fig, "03_outlier_boxplots")


def plot_correlation_heatmap(frame: pd.DataFrame, columns: list[str]) -> Path:
    """Plot the Pearson correlation matrix to expose multicollinearity."""
    apply_style()
    corr = frame[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson correlation - numeric features")
    fig.tight_layout()
    return _save(fig, "04_correlation_heatmap")


def plot_conversion_by_category(
    frame: pd.DataFrame, column: str, target: str, order: list | None = None
) -> Path:
    """Plot conversion rate and session volume for one categorical feature."""
    apply_style()
    grouped = frame.groupby(column, observed=True)[target].agg(["mean", "count"])
    if order is not None:
        grouped = grouped.reindex([o for o in order if o in grouped.index])
    grouped["mean"] *= 100
    baseline = 100 * frame[target].mean()

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.bar(
        grouped.index.astype(str),
        grouped["count"],
        color=config.NEGATIVE_COLOUR,
        alpha=0.55,
        label="Sessions",
    )
    ax1.set_ylabel("Sessions", color="#475569")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(
        grouped.index.astype(str),
        grouped["mean"],
        marker="o",
        color=config.PRIMARY_COLOUR,
        linewidth=2.2,
        label="Conversion rate",
    )
    ax2.axhline(
        baseline,
        color=config.SECONDARY_COLOUR,
        linestyle="--",
        linewidth=1.5,
        label=f"Overall ({baseline:.1f}%)",
    )
    ax2.set_ylabel("Conversion rate (%)", color=config.PRIMARY_COLOUR)

    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, loc="upper left", fontsize=8)
    ax1.set_title(f"Conversion rate by {column}")
    fig.tight_layout()
    return _save(fig, f"05_conversion_by_{column.lower()}")


def plot_mutual_information(scores: pd.Series, top_n: int = 20) -> Path:
    """Plot pre-modeling feature relevance via mutual information."""
    apply_style()
    top = scores.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(top.index, top.values, color=config.PRIMARY_COLOUR)
    ax.set_title("Mutual information with the target (pre-modeling)")
    ax.set_xlabel("Mutual information (nats)")
    fig.tight_layout()
    return _save(fig, "06_mutual_information")


# --------------------------------------------------------------------------- #
# Model evaluation
# --------------------------------------------------------------------------- #


def plot_model_comparison(results: pd.DataFrame, metrics: list[str]) -> Path:
    """Plot a grouped bar chart comparing models across metrics."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(results))
    width = 0.8 / len(metrics)
    palette = sns.color_palette("Blues_r", len(metrics))

    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width
        bars = ax.bar(x + offset, results[metric], width, label=metric, color=palette[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=6.5, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels(results.index, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on the held-out test set")
    ax.legend(ncol=len(metrics), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    return _save(fig, "07_model_comparison")


def plot_roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Plot ROC curves for several models on shared axes.

    Args:
        curves: Mapping of model name to ``(y_true, y_score)``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "--", color=config.NEGATIVE_COLOUR, label="Random (0.5000)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves - test set")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "08_roc_curves")


def plot_pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], baseline: float) -> Path:
    """Plot precision-recall curves with the no-skill baseline.

    Args:
        curves: Mapping of model name to ``(y_true, y_score)``.
        baseline: Positive class prevalence, i.e. the no-skill precision.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (y_true, y_score) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP = {auc(recall, precision):.4f})")
    ax.axhline(
        baseline,
        linestyle="--",
        color=config.NEGATIVE_COLOUR,
        label=f"No skill ({baseline:.4f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves - test set")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "09_precision_recall_curves")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    """Plot raw and row-normalised confusion matrices side by side."""
    apply_style()
    labels = [config.NEGATIVE_CLASS_LABEL, config.POSITIVE_CLASS_LABEL]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", colorbar=False, ax=axes[0]
    )
    axes[0].set_title("Counts")

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        colorbar=False,
        normalize="true",
        values_format=".3f",
        ax=axes[1],
    )
    axes[1].set_title("Row-normalised (recall per class)")

    fig.suptitle(f"Confusion matrix - {model_name}", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "10_confusion_matrix")


def plot_feature_importance(importances: pd.Series, model_name: str, top_n: int = 20) -> Path:
    """Plot the top-N most important features for the selected model."""
    apply_style()
    top = importances.sort_values(ascending=True).tail(top_n)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(top))))
    ax.barh(top.index, top.values, color=config.PRIMARY_COLOUR)
    ax.set_title(f"Top {top_n} feature importances - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return _save(fig, "11_feature_importance")


def plot_threshold_analysis(threshold_table: pd.DataFrame) -> Path:
    """Plot metric trade-offs and expected business value against the threshold."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for metric, colour in [
        ("precision", config.PRIMARY_COLOUR),
        ("recall", config.SECONDARY_COLOUR),
        ("f1", "#10b981"),
    ]:
        axes[0].plot(
            threshold_table["threshold"], threshold_table[metric], label=metric, color=colour, linewidth=2
        )
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Metric trade-off vs. threshold")
    axes[0].legend(fontsize=8)

    axes[1].plot(
        threshold_table["threshold"],
        threshold_table["expected_value"],
        color=config.PRIMARY_COLOUR,
        linewidth=2,
    )
    best = threshold_table.loc[threshold_table["expected_value"].idxmax()]
    axes[1].axvline(best["threshold"], linestyle="--", color=config.SECONDARY_COLOUR)
    axes[1].annotate(
        f"optimum t={best['threshold']:.2f}\n${best['expected_value']:,.0f}",
        xy=(best["threshold"], best["expected_value"]),
        xytext=(10, -28),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Expected net value (USD)")
    axes[1].set_title("Expected campaign value vs. threshold")

    fig.suptitle("Threshold selection under the business cost model", fontweight="bold")
    fig.tight_layout()
    return _save(fig, "12_threshold_analysis")


def plot_tuning_comparison(comparison: pd.DataFrame) -> Path:
    """Plot baseline vs. tuned performance for every model."""
    apply_style()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(comparison))
    width = 0.36
    ax.bar(x - width / 2, comparison["baseline_pr_auc"], width, label="Default hyperparameters", color=config.NEGATIVE_COLOUR)
    ax.bar(x + width / 2, comparison["tuned_pr_auc"], width, label="Tuned", color=config.PRIMARY_COLOUR)
    for i, (base, tuned) in enumerate(zip(comparison["baseline_pr_auc"], comparison["tuned_pr_auc"])):
        ax.text(i - width / 2, base, f"{base:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width / 2, tuned, f"{tuned:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison.index, rotation=10, ha="right")
    ax.set_ylabel("Cross-validated PR-AUC")
    ax.set_title("Effect of hyperparameter tuning")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "13_tuning_comparison")


def plot_leakage_experiment(results: pd.DataFrame) -> Path:
    """Plot performance with and without the suspected leakage feature."""
    apply_style()
    metrics = ["roc_auc", "pr_auc", "f1", "recall"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(metrics))
    width = 0.36
    ax.bar(x - width / 2, results.loc["with_PageValues", metrics], width, label="With PageValues", color=config.SECONDARY_COLOUR)
    ax.bar(x + width / 2, results.loc["without_PageValues", metrics], width, label="Without PageValues", color=config.PRIMARY_COLOUR)
    for i, metric in enumerate(metrics):
        ax.text(i - width / 2, results.loc["with_PageValues", metric], f"{results.loc['with_PageValues', metric]:.3f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + width / 2, results.loc["without_PageValues", metric], f"{results.loc['without_PageValues', metric]:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Leakage experiment: contribution of the GA PageValues feature")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "14_leakage_experiment")
