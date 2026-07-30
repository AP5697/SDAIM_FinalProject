"""Cleaning, feature engineering and the sklearn preprocessing pipeline.

Design note - leakage prevention
--------------------------------
Every stateful transformation (imputation statistics, scaling parameters,
one-hot vocabularies) lives *inside* a :class:`sklearn.pipeline.Pipeline`.
Consequently, when the pipeline is passed to ``RandomizedSearchCV``, each
transformer is re-fitted on that fold's training partition only. No statistic
computed from a validation fold, or from the held-out test set, can reach the
model. Fitting a scaler on the full dataset before splitting - the most common
leakage mistake in student projects - is structurally impossible here.

Feature engineering is implemented as a stateless :class:`FunctionTransformer`
placed at the head of the pipeline, so the serialised artifact accepts raw
Google Analytics fields and does all derivation internally. The Streamlit app
therefore never re-implements preprocessing logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mlops.model_building import config  # noqa: E402
from mlops.model_building.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

# Guard against division by zero for sessions with no recorded page views.
_EPSILON: float = 1e-6


def remove_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop exact duplicate rows.

    Duplicate sessions are an artefact of the log-aggregation process rather
    than genuine repeat behaviour: two sessions agreeing on all 18 attributes
    including six continuous duration fields is not plausible by chance.
    Retaining them would let the same observation appear in both the training
    and test partitions, inflating measured performance.

    Args:
        frame: Raw DataFrame.

    Returns:
        A ``(deduplicated_frame, n_removed)`` tuple.
    """
    before = len(frame)
    cleaned = frame.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    logger.info("Removed %d duplicate rows (%d -> %d)", removed, before, len(cleaned))
    return cleaned, removed


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive behavioural ratio features from raw session counters.

    The raw columns record page counts and dwell times separately. The derived
    features below express *browsing style* rather than raw volume, which
    generalises better across sessions of different lengths:

    - ``TotalPages`` / ``TotalDuration``: overall session engagement.
    - ``AvgDurationPerPage``: depth of attention per page. A visitor spending
      90 seconds per page is evaluating; one spending 3 seconds is skimming.
    - ``ProductPageRatio``: proportion of the session spent on product pages,
      distinguishing purchase-oriented from informational browsing.
    - ``BounceExitDelta``: the gap between exit and bounce rate, separating
      "left immediately" from "browsed then left".

    This function is stateless and row-wise, so it is safe at any pipeline
    position and identical at train and inference time.

    Args:
        frame: DataFrame containing the raw session columns.

    Returns:
        A copy of ``frame`` with the engineered columns appended.
    """
    out = frame.copy()

    out["TotalPages"] = (
        out["Administrative"] + out["Informational"] + out["ProductRelated"]
    )
    out["TotalDuration"] = (
        out["Administrative_Duration"]
        + out["Informational_Duration"]
        + out["ProductRelated_Duration"]
    )
    out["AvgDurationPerPage"] = out["TotalDuration"] / (out["TotalPages"] + _EPSILON)
    out["ProductPageRatio"] = out["ProductRelated"] / (out["TotalPages"] + _EPSILON)
    out["BounceExitDelta"] = out["ExitRates"] - out["BounceRates"]

    return out


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Construct the column-wise preprocessing transformer.

    Numeric branch: median imputation followed by standardisation. The median
    is chosen over the mean because the duration and page-count distributions
    are heavily right-skewed.

    Categorical branch: most-frequent imputation followed by one-hot encoding
    with ``handle_unknown="ignore"``. That setting is what allows the deployed
    application to survive a category it never saw during training - for
    example a new browser identifier - by encoding it as all-zeros rather than
    raising at prediction time.

    Args:
        numeric_features: Numeric column names. Defaults to the true numeric
            features plus the engineered features.
        categorical_features: Categorical column names. Defaults to
            :data:`config.CATEGORICAL_FEATURES`, which deliberately includes the
            integer-coded identifiers.

    Returns:
        An unfitted :class:`~sklearn.compose.ColumnTransformer`.
    """
    numeric_features = numeric_features or (
        config.TRUE_NUMERIC_FEATURES + config.ENGINEERED_FEATURES
    )
    categorical_features = categorical_features or config.CATEGORICAL_FEATURES

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(classifier: object, drop_features: list[str] | None = None) -> Pipeline:
    """Assemble the full feature-engineering + preprocessing + model pipeline.

    Args:
        classifier: Any fitted-or-unfitted scikit-learn compatible estimator
            exposing ``fit``/``predict``/``predict_proba``.
        drop_features: Columns to exclude from the model, used by the leakage
            experiment to remove ``PageValues``.

    Returns:
        A three-stage :class:`~sklearn.pipeline.Pipeline`:
        ``engineer -> preprocess -> classifier``.
    """
    drop_features = drop_features or []

    numeric = [
        column
        for column in config.TRUE_NUMERIC_FEATURES + config.ENGINEERED_FEATURES
        if column not in drop_features
    ]
    categorical = [
        column for column in config.CATEGORICAL_FEATURES if column not in drop_features
    ]

    return Pipeline(
        steps=[
            (
                "engineer",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocess", build_preprocessor(numeric, categorical)),
            ("classifier", classifier),
        ]
    )


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Extract post-transformation feature names from a fitted pipeline.

    Args:
        pipeline: A fitted pipeline built by :func:`build_pipeline`.

    Returns:
        Feature names in the order the classifier received them.

    Raises:
        RuntimeError: If the pipeline has not been fitted.
    """
    try:
        return list(pipeline.named_steps["preprocess"].get_feature_names_out())
    except Exception as exc:
        raise RuntimeError(
            "Could not read feature names - the pipeline appears unfitted. "
            f"Call fit() before get_feature_names(). Underlying error: {exc}"
        ) from exc


def summarise_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a per-column data-quality summary.

    Args:
        frame: DataFrame to profile.

    Returns:
        A table of dtype, null count, null percentage, distinct-value count and
        zero-value percentage for every column, sorted by null percentage.
    """
    rows = []
    for column in frame.columns:
        series = frame[column]
        zero_pct = (
            100.0 * float((series == 0).sum()) / len(series)
            if pd.api.types.is_numeric_dtype(series)
            else np.nan
        )
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_null": int(series.isna().sum()),
                "pct_null": round(100.0 * series.isna().mean(), 3),
                "n_unique": int(series.nunique()),
                "pct_zero": round(zero_pct, 2) if not np.isnan(zero_pct) else None,
            }
        )
    return pd.DataFrame(rows).sort_values("pct_null", ascending=False)
