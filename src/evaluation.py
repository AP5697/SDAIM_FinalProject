"""Model evaluation: metrics, threshold economics and error analysis.

Metric choice rationale
-----------------------
The positive class is ~15% of sessions. A model that predicts "no purchase"
for every session scores 84.5% accuracy while being commercially worthless, so
accuracy is reported for completeness only and never used for selection.

- **PR-AUC (average precision)** is the primary selection metric. It summarises
  precision across all recall levels for the minority class and, unlike ROC-AUC,
  its baseline moves with class prevalence, so it cannot be flattered by the
  84.5% negative majority.
- **ROC-AUC** is reported as a threshold-independent ranking measure and for
  comparability with published results on this dataset.
- **Recall** matters because a missed high-intent session is a lost order.
- **Precision** matters because every intervention costs real money.
- **F1** balances the two at the chosen operating point.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

DEFAULT_THRESHOLD: float = 0.50
_LATENCY_REPEATS: int = 100


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute the full classification metric suite at a given threshold.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probabilities for the positive class.
        threshold: Probability cut-off for the positive prediction.

    Returns:
        A mapping of metric name to value, rounded to four decimals.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "brier": round(float(brier_score_loss(y_true, y_proba)), 4),
        "threshold": round(float(threshold), 4),
    }


def confusion_breakdown(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float
) -> dict[str, int]:
    """Return the four confusion-matrix cells as named integers.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted positive-class probabilities.
        threshold: Probability cut-off.

    Returns:
        Mapping with ``true_negative``, ``false_positive``, ``false_negative``
        and ``true_positive`` counts.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def expected_campaign_value(true_positives: int, false_positives: int) -> float:
    """Compute expected net value of a targeting campaign.

    Cost model (stated as an assumption, not a measured business fact):
    firing one real-time incentive costs :data:`config.INTERVENTION_COST`. When
    it lands on a session that does convert, the incentive is credited with
    :data:`config.INTERVENTION_UPLIFT` incremental conversion probability on an
    order worth :data:`config.AVERAGE_ORDER_VALUE`. When it lands on a session
    that does not convert, the spend is lost.

    Args:
        true_positives: Correctly flagged converting sessions.
        false_positives: Incorrectly flagged non-converting sessions.

    Returns:
        Expected net value in USD.
    """
    gain_per_true_positive = (
        config.INTERVENTION_UPLIFT * config.AVERAGE_ORDER_VALUE - config.INTERVENTION_COST
    )
    return float(
        true_positives * gain_per_true_positive - false_positives * config.INTERVENTION_COST
    )


def threshold_analysis(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    start: float = 0.05,
    stop: float = 0.95,
    step: float = 0.01,
) -> pd.DataFrame:
    """Sweep the decision threshold and tabulate metrics and expected value.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted positive-class probabilities.
        start: Lowest threshold to evaluate.
        stop: Highest threshold to evaluate.
        step: Threshold increment.

    Returns:
        A table with one row per threshold containing precision, recall, F1,
        the confusion cells and expected campaign value.
    """
    rows = []
    for threshold in np.arange(start, stop + step / 2, step):
        y_pred = (y_proba >= threshold).astype(int)
        cells = confusion_breakdown(y_true, y_proba, threshold)
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "n_flagged": int(y_pred.sum()),
                **cells,
                "expected_value": round(
                    expected_campaign_value(cells["true_positive"], cells["false_positive"]), 2
                ),
            }
        )
    return pd.DataFrame(rows)


def calibration_curve_data(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Bin predictions and compare mean predicted probability to observed rate.

    A model is *calibrated* when sessions it scores at 0.30 convert about 30% of
    the time. This is distinct from ranking quality: a model can separate buyers
    from non-buyers superbly (high ROC-AUC) while its probabilities are
    systematically too confident. Gradient-boosted trees optimised for a ranking
    loss commonly are, so the curve is worth reporting rather than assuming.

    Uniform-width bins are used rather than quantile bins so the horizontal axis
    stays comparable to the diagonal reference line; bins containing no
    predictions are dropped.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted positive-class probabilities.
        n_bins: Number of equal-width probability bins.

    Returns:
        A table with one row per non-empty bin containing the mean predicted
        probability, the observed conversion fraction and the bin count.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # side="right" then clip keeps p == 1.0 inside the final bin rather than
    # creating a spurious (n_bins + 1)th bin containing only the maxima.
    membership = np.clip(np.digitize(y_proba, edges[1:-1], right=True), 0, n_bins - 1)

    rows = []
    for index in range(n_bins):
        mask = membership == index
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": round(float(edges[index]), 4),
                "bin_upper": round(float(edges[index + 1]), 4),
                "mean_predicted": round(float(y_proba[mask].mean()), 4),
                "observed_rate": round(float(y_true[mask].mean()), 4),
                "count": count,
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(threshold_table: pd.DataFrame, objective: str = "f1") -> float:
    """Select the threshold maximising a chosen column.

    Args:
        threshold_table: Output of :func:`threshold_analysis`.
        objective: Column to maximise, e.g. ``"f1"`` or ``"expected_value"``.

    Returns:
        The maximising threshold.

    Raises:
        KeyError: If ``objective`` is not a column of the table.
    """
    if objective not in threshold_table.columns:
        raise KeyError(
            f"'{objective}' is not a column of the threshold table. "
            f"Available: {list(threshold_table.columns)}"
        )
    return float(threshold_table.loc[threshold_table[objective].idxmax(), "threshold"])


def measure_inference_latency(model: Any, sample: pd.DataFrame) -> dict[str, float]:
    """Measure single-row prediction latency, which drives the app's UX.

    Args:
        model: Fitted pipeline exposing ``predict_proba``.
        sample: A DataFrame whose first row is used as the probe.

    Returns:
        Mapping with mean, median, p95 and max latency in milliseconds.
    """
    single_row = sample.iloc[[0]]
    model.predict_proba(single_row)  # warm caches before timing

    timings = []
    for _ in range(_LATENCY_REPEATS):
        start = time.perf_counter()
        model.predict_proba(single_row)
        timings.append((time.perf_counter() - start) * 1000)

    array = np.array(timings)
    return {
        "latency_mean_ms": round(float(array.mean()), 3),
        "latency_median_ms": round(float(np.median(array)), 3),
        "latency_p95_ms": round(float(np.percentile(array, 95)), 3),
        "latency_max_ms": round(float(array.max()), 3),
    }


def error_analysis(
    features: pd.DataFrame,
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Extract the model's most confident mistakes for qualitative review.

    Args:
        features: Test-set feature matrix, index-aligned with ``y_true``.
        y_true: Ground-truth labels.
        y_proba: Predicted positive-class probabilities.
        threshold: Decision threshold in force.
        top_n: Number of examples to return per error class.

    Returns:
        Mapping with ``false_positives`` and ``false_negatives`` tables, each
        sorted by how confidently the model got it wrong, plus ``profile``
        comparing mean feature values across the four outcome groups.
    """
    frame = features.copy()
    frame["actual"] = y_true.to_numpy()
    frame["probability"] = y_proba
    frame["predicted"] = (y_proba >= threshold).astype(int)

    false_positives = frame[(frame["actual"] == 0) & (frame["predicted"] == 1)].nlargest(
        top_n, "probability"
    )
    false_negatives = frame[(frame["actual"] == 1) & (frame["predicted"] == 0)].nsmallest(
        top_n, "probability"
    )

    frame["outcome"] = np.select(
        [
            (frame["actual"] == 1) & (frame["predicted"] == 1),
            (frame["actual"] == 0) & (frame["predicted"] == 0),
            (frame["actual"] == 0) & (frame["predicted"] == 1),
        ],
        ["true_positive", "true_negative", "false_positive"],
        default="false_negative",
    )

    profile_columns = [
        column for column in config.TRUE_NUMERIC_FEATURES if column in frame.columns
    ]
    profile = frame.groupby("outcome")[profile_columns + ["probability"]].mean().round(3)

    logger.info(
        "Error analysis: %d false positives, %d false negatives at threshold %.2f",
        int(((frame["actual"] == 0) & (frame["predicted"] == 1)).sum()),
        int(((frame["actual"] == 1) & (frame["predicted"] == 0)).sum()),
        threshold,
    )
    return {
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "profile": profile,
    }


def text_classification_report(
    y_true: np.ndarray | pd.Series, y_proba: np.ndarray, threshold: float
) -> str:
    """Render scikit-learn's per-class classification report as text.

    Args:
        y_true: Ground-truth labels.
        y_proba: Predicted positive-class probabilities.
        threshold: Decision threshold.

    Returns:
        The formatted report string.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return classification_report(
        y_true,
        y_pred,
        target_names=[config.NEGATIVE_CLASS_LABEL, config.POSITIVE_CLASS_LABEL],
        digits=4,
        zero_division=0,
    )
