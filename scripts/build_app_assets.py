"""Derive the application's input schema and presets from the training data.

The Streamlit application must not ship the raw dataset, but its widget bounds,
category lists and example sessions should still be grounded in real data
rather than guessed. This script measures them once and writes
``models/app_schema.json``, which the deployed application reads.

Usage:
    python -m scripts.build_app_assets
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.data_loader import load_raw_data  # noqa: E402
from src.preprocessing import remove_duplicates  # noqa: E402
from src.utils import get_logger, save_json  # noqa: E402

logger = get_logger(__name__)

SLIDER_UPPER_PERCENTILE: float = 0.99
MONTH_ORDER: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "June",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def numeric_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Measure display ranges for every numeric input widget.

    The true maxima are extreme (one session records over 60,000 seconds on
    product pages), so sliders are capped at the 99th percentile while the hard
    maximum is retained for validation messages.

    Args:
        frame: Deduplicated dataset.

    Returns:
        Mapping of column name to its min, max, median, p99 and step size.
    """
    ranges: dict[str, dict[str, float]] = {}
    for column in config.TRUE_NUMERIC_FEATURES:
        series = frame[column]
        is_rate = column in {"BounceRates", "ExitRates", "SpecialDay"}
        is_count = column in {"Administrative", "Informational", "ProductRelated"}
        ranges[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "median": float(series.median()),
            "mean": round(float(series.mean()), 4),
            "p99": float(series.quantile(SLIDER_UPPER_PERCENTILE)),
            "step": 0.001 if is_rate else (1.0 if is_count else 1.0),
            "decimals": 3 if is_rate else (0 if is_count else 1),
        }
    return ranges


def categorical_levels(frame: pd.DataFrame) -> dict[str, list[Any]]:
    """List every category level observed during training.

    Args:
        frame: Deduplicated dataset.

    Returns:
        Mapping of column name to its sorted level list. ``Month`` is returned
        in calendar order rather than alphabetical order.
    """
    levels: dict[str, list[Any]] = {}
    for column in config.CATEGORICAL_FEATURES:
        observed = frame[column].unique().tolist()
        if column == "Month":
            observed = [m for m in MONTH_ORDER if m in set(observed)]
        elif column == "Weekend":
            observed = [False, True]
        else:
            observed = sorted(observed, key=lambda v: (isinstance(v, str), v))
        levels[column] = observed
    return levels


def build_presets(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Construct example sessions from measured medians of real subgroups.

    Each preset is the column-wise median of an actual behavioural segment in
    the dataset, so the application demonstrates the model on realistic input
    rather than invented numbers.

    Args:
        frame: Deduplicated dataset.

    Returns:
        Mapping of preset name to a complete feature payload.
    """
    target = config.TARGET_COLUMN

    segments = {
        "High-intent buyer": frame[
            (frame[target]) & (frame["PageValues"] > frame["PageValues"].median())
        ],
        "Engaged researcher": frame[
            (~frame[target])
            & (frame["ProductRelated"] > frame["ProductRelated"].quantile(0.75))
            & (frame["PageValues"] == 0)
        ],
        "Quick bouncer": frame[(~frame[target]) & (frame["BounceRates"] > 0.15)],
        "Returning window shopper": frame[
            (~frame[target])
            & (frame["VisitorType"] == "Returning_Visitor")
            & (frame["ProductRelated"].between(5, 30))
        ],
    }

    presets: dict[str, dict[str, Any]] = {}
    for name, segment in segments.items():
        if segment.empty:
            logger.warning("Segment '%s' matched no rows - skipping preset", name)
            continue

        payload: dict[str, Any] = {}
        for column in config.TRUE_NUMERIC_FEATURES:
            value = float(segment[column].median())
            payload[column] = round(value, 4) if value % 1 else int(value)
        for column in config.CATEGORICAL_FEATURES:
            mode = segment[column].mode()
            value = mode.iloc[0] if not mode.empty else frame[column].mode().iloc[0]
            payload[column] = bool(value) if column == "Weekend" else (
                int(value) if isinstance(value, (int, float)) else str(value)
            )

        presets[name] = payload
        logger.info(
            "Preset '%s' built from %d real sessions (%.1f%% of data)",
            name,
            len(segment),
            100 * len(segment) / len(frame),
        )
    return presets


def field_help() -> dict[str, str]:
    """Return the Google Analytics definition shown beside each input."""
    return {
        "Administrative": "Account//login/contact pages viewed in this session.",
        "Administrative_Duration": "Total seconds spent on administrative pages.",
        "Informational": "About/FAQ/shipping-policy pages viewed.",
        "Informational_Duration": "Total seconds spent on informational pages.",
        "ProductRelated": "Product and category pages viewed - the strongest volume signal.",
        "ProductRelated_Duration": "Total seconds spent on product pages.",
        "BounceRates": "GA bounce rate for the entry page: share of sessions that leave without interacting.",
        "ExitRates": "GA exit rate: share of pageviews on these pages that were the last of a session.",
        "PageValues": "GA Page Value in USD: revenue and goal value attributed to the pages visited.",
        "SpecialDay": "Proximity to a major shopping holiday, 0 (far) to 1 (on the day).",
        "Month": "Calendar month of the session. Two months are absent from the source data.",
        "VisitorType": "GA classification: new, returning, or other.",
        "Weekend": "Whether the session occurred on a Saturday or Sunday.",
        "OperatingSystems": "Anonymised OS identifier - a nominal code, not a magnitude.",
        "Browser": "Anonymised browser identifier - a nominal code, not a magnitude.",
        "Region": "Anonymised geographic region code.",
        "TrafficType": "Anonymised acquisition-channel code, e.g. direct, organic, referral.",
    }


def main() -> dict[str, Any]:
    """Build and persist the application schema."""
    config.ensure_directories()
    raw = load_raw_data()
    deduplicated, _ = remove_duplicates(raw)

    schema: dict[str, Any] = {
        "numeric_ranges": numeric_ranges(deduplicated),
        "categorical_levels": categorical_levels(deduplicated),
        "presets": build_presets(deduplicated),
        "field_help": field_help(),
        "feature_order": config.ALL_FEATURES,
        "numeric_features": config.TRUE_NUMERIC_FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "baseline_conversion_rate_pct": round(
            100 * float(deduplicated[config.TARGET_COLUMN].mean()), 2
        ),
        "n_training_sessions": int(len(deduplicated)),
    }

    save_json(schema, config.MODEL_DIR / "app_schema.json")
    logger.info(
        "Schema built: %d numeric ranges, %d categorical fields, %d presets",
        len(schema["numeric_ranges"]),
        len(schema["categorical_levels"]),
        len(schema["presets"]),
    )
    return schema


if __name__ == "__main__":
    main()
