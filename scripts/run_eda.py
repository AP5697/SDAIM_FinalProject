"""Run the full exploratory data analysis and persist every artifact.

Outputs:
    reports/figures/*.png   All EDA figures.
    reports/tables/*.csv    Data-quality, correlation and relevance tables.
    reports/eda_summary.json  Machine-readable summary of every finding, so the
        written report quotes measured values rather than transcribed ones.

Usage:
    python -m scripts.run_eda
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from src import config, visualisation  # noqa: E402
from src.data_loader import load_raw_data  # noqa: E402
from src.preprocessing import (  # noqa: E402
    add_engineered_features,
    remove_duplicates,
    summarise_missingness,
)
from src.utils import get_logger, save_json, save_table  # noqa: E402

logger = get_logger(__name__)

MONTH_ORDER: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]
IQR_MULTIPLIER: float = 1.5
HIGH_CORRELATION_THRESHOLD: float = 0.80


def outlier_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Quantify outliers per numeric column using the 1.5 x IQR rule.

    Args:
        frame: Source DataFrame.
        columns: Numeric columns to assess.

    Returns:
        A table of bounds, outlier counts and percentages per column.
    """
    rows = []
    for column in columns:
        series = frame[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
        mask = (series < lower) | (series > upper)
        rows.append(
            {
                "column": column,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "n_outliers": int(mask.sum()),
                "pct_outliers": round(100.0 * float(mask.mean()), 2),
                "max": round(float(series.max()), 4),
                "skew": round(float(series.skew()), 3),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_outliers", ascending=False)


def high_correlation_pairs(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Identify feature pairs exceeding the multicollinearity threshold.

    Args:
        frame: Source DataFrame.
        columns: Numeric columns to correlate.

    Returns:
        A table of correlated pairs, sorted by absolute correlation.
    """
    corr = frame[columns].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b", 0: "abs_corr"})
    )
    pairs = pairs[pairs["abs_corr"] >= HIGH_CORRELATION_THRESHOLD]
    return pairs.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def main() -> dict:
    """Execute the EDA and return a summary of every measured finding."""
    config.ensure_directories()
    logger.info("=" * 78)
    logger.info("EXPLORATORY DATA ANALYSIS - %s", config.DATASET_NAME)
    logger.info("=" * 78)

    raw = load_raw_data()
    summary: dict = {
        "dataset": config.DATASET_NAME,
        "source_url": config.DATASET_HOME,
        "licence": config.DATASET_LICENCE,
        "raw_shape": list(raw.shape),
    }

    # ---------------------------------------------------------------- quality
    quality = summarise_missingness(raw)
    save_table(quality, "eda_data_quality", index=False)
    summary["total_nulls"] = int(raw.isna().sum().sum())
    summary["n_duplicates"] = int(raw.duplicated().sum())
    summary["duplicate_pct"] = round(100.0 * float(raw.duplicated().mean()), 3)

    logger.info("Nulls across all cells: %d", summary["total_nulls"])
    logger.info(
        "Exact duplicate rows: %d (%.2f%%)",
        summary["n_duplicates"],
        summary["duplicate_pct"],
    )

    # The integer-coded categorical finding.
    integer_coded = {
        column: {
            "inferred_dtype": str(raw[column].dtype),
            "n_unique": int(raw[column].nunique()),
            "values": sorted(int(v) for v in raw[column].unique())[:30],
        }
        for column in config.INTEGER_CODED_CATEGORICAL_FEATURES
    }
    summary["integer_coded_categoricals"] = integer_coded
    for column, meta in integer_coded.items():
        logger.info(
            "%s inferred as %s but is nominal with %d distinct codes",
            column,
            meta["inferred_dtype"],
            meta["n_unique"],
        )

    # Zero-inflation: a zero duration means "never visited this page type".
    zero_inflation = {
        column: round(100.0 * float((raw[column] == 0).mean()), 2)
        for column in config.TRUE_NUMERIC_FEATURES
    }
    summary["zero_inflation_pct"] = zero_inflation

    # ----------------------------------------------------------------- target
    target = raw[config.TARGET_COLUMN].astype(int)
    summary["target_positive_rate"] = round(100.0 * float(target.mean()), 3)
    summary["target_counts"] = {
        "no_purchase": int((target == 0).sum()),
        "purchase": int((target == 1).sum()),
    }
    summary["imbalance_ratio"] = round(
        float((target == 0).sum() / max((target == 1).sum(), 1)), 2
    )
    summary["majority_class_accuracy"] = round(100.0 * float((target == 0).mean()), 3)
    logger.info(
        "Target: %.2f%% positive | imbalance ratio %.2f:1 | majority-class accuracy %.2f%%",
        summary["target_positive_rate"],
        summary["imbalance_ratio"],
        summary["majority_class_accuracy"],
    )
    visualisation.plot_target_distribution(target)

    # ----------------------------------------------------- clean + engineered
    deduplicated, n_removed = remove_duplicates(raw)
    summary["shape_after_dedup"] = list(deduplicated.shape)
    summary["duplicates_removed"] = n_removed

    enriched = add_engineered_features(deduplicated)
    numeric_all = config.TRUE_NUMERIC_FEATURES + config.ENGINEERED_FEATURES

    # --------------------------------------------------------------- outliers
    outliers = outlier_report(enriched, numeric_all)
    save_table(outliers, "eda_outliers", index=False)
    summary["outliers"] = outliers.set_index("column").to_dict(orient="index")
    visualisation.plot_numeric_distributions(enriched, numeric_all)
    visualisation.plot_outlier_boxplots(enriched, numeric_all)

    # ---------------------------------------------------------- correlations
    corr = enriched[numeric_all].corr()
    save_table(corr.round(4), "eda_correlation_matrix")
    visualisation.plot_correlation_heatmap(enriched, numeric_all)

    collinear = high_correlation_pairs(enriched, numeric_all)
    save_table(collinear, "eda_high_correlation_pairs", index=False)
    summary["high_correlation_pairs"] = collinear.round(4).to_dict(orient="records")
    logger.info(
        "Feature pairs with |r| >= %.2f: %d", HIGH_CORRELATION_THRESHOLD, len(collinear)
    )

    target_corr = (
        enriched[numeric_all]
        .corrwith(deduplicated[config.TARGET_COLUMN].astype(int))
        .sort_values(ascending=False)
    )
    save_table(target_corr.round(4).to_frame("correlation_with_target"), "eda_target_correlation")
    summary["target_correlation"] = target_corr.round(4).to_dict()

    # ------------------------------------------------------------ categorical
    categorical_findings: dict = {}
    for column in config.NATIVE_CATEGORICAL_FEATURES:
        grouped = deduplicated.groupby(column, observed=True)[config.TARGET_COLUMN].agg(
            ["mean", "count"]
        )
        grouped["mean"] = (100 * grouped["mean"]).round(2)
        categorical_findings[column] = grouped.rename(
            columns={"mean": "conversion_pct", "count": "sessions"}
        ).to_dict(orient="index")
        order = MONTH_ORDER if column == "Month" else None
        visualisation.plot_conversion_by_category(
            deduplicated, column, config.TARGET_COLUMN, order=order
        )
    summary["categorical_conversion"] = categorical_findings

    observed_months = [m for m in MONTH_ORDER if m in set(deduplicated["Month"])]
    summary["months_present"] = observed_months
    summary["months_missing"] = [m for m in MONTH_ORDER if m not in observed_months]
    logger.info("Months present: %s | absent: %s", observed_months, summary["months_missing"])

    # ------------------------------------------------- mutual information (MI)
    features = enriched[numeric_all].copy()
    for column in config.CATEGORICAL_FEATURES:
        features[column] = pd.factorize(deduplicated[column])[0]

    discrete_mask = [column in config.CATEGORICAL_FEATURES for column in features.columns]
    mi_scores = pd.Series(
        mutual_info_classif(
            features,
            deduplicated[config.TARGET_COLUMN].astype(int),
            discrete_features=discrete_mask,
            random_state=config.RANDOM_STATE,
        ),
        index=features.columns,
    ).sort_values(ascending=False)

    save_table(mi_scores.round(5).to_frame("mutual_information"), "eda_mutual_information")
    summary["mutual_information"] = mi_scores.round(5).to_dict()
    visualisation.plot_mutual_information(mi_scores)
    logger.info("Top 5 features by mutual information: %s", list(mi_scores.head(5).index))

    save_json(summary, config.REPORTS_DIR / "eda_summary.json")
    logger.info("EDA complete - %d figures written", len(list(config.FIGURES_DIR.glob("*.png"))))
    return summary


if __name__ == "__main__":
    main()
