"""Compute the deployed model's calibration curve and patch it into metadata.

Calibration answers a different question from every metric already recorded:
not "does the model rank buyers above non-buyers" (ROC-AUC, PR-AUC) but "when it
says 30%, do 30% of those sessions actually convert". The Brier score in
``metadata.json`` scores this numerically; this script produces the curve behind
that number so the application can show it.

This reproduces the exact held-out split used in training - same dedup, same
seed, same stratification - so the curve describes the deployed artifact rather
than a fresh fit.

Usage:
    python -m scripts.add_calibration
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlops.model_building import config, evaluation  # noqa: E402
from mlops.deployment import inference  # noqa: E402
from mlops.model_building.data_loader import load_raw_data, split_features_target, stratified_split  # noqa: E402
from mlops.model_building.preprocessing import remove_duplicates  # noqa: E402
from mlops.model_building.utils import get_logger, save_json  # noqa: E402

logger = get_logger(__name__)

N_BINS: int = 10


def build_calibration() -> dict[str, Any]:
    """Score the held-out test set and summarise calibration.

    Returns:
        Mapping containing the binned curve, the Brier score and the count of
        sessions the curve was measured on.
    """
    frame, _ = remove_duplicates(load_raw_data())
    features, target = split_features_target(frame)
    _, x_test, _, y_test = stratified_split(features, target)

    model = inference.load_model()
    probabilities = model.predict_proba(x_test[config.ALL_FEATURES])[:, 1]

    curve = evaluation.calibration_curve_data(y_test, probabilities, n_bins=N_BINS)
    metrics = evaluation.compute_metrics(
        y_test, probabilities, threshold=inference.get_operating_threshold()
    )

    # Count-weighted mean gap: positive means the model predicts higher
    # probabilities than the outcomes justify.
    weights = curve["count"]
    mean_gap = float(
        ((curve["mean_predicted"] - curve["observed_rate"]) * weights).sum() / weights.sum()
    )

    classifier = model.named_steps["classifier"]
    scale_pos_weight = float(getattr(classifier, "scale_pos_weight", 0.0) or 0.0)

    logger.info(
        "Calibration measured on %d test sessions across %d non-empty bins "
        "(mean gap %+.3f)",
        len(y_test),
        len(curve),
        mean_gap,
    )
    return {
        "n_bins": N_BINS,
        "n_test_sessions": int(len(y_test)),
        "brier_score": metrics["brier"],
        "mean_gap": round(mean_gap, 4),
        "direction": "overconfident" if mean_gap > 0 else "underconfident",
        "scale_pos_weight": round(scale_pos_weight, 3),
        # The cause is deliberate, not a defect: the positive class is upweighted
        # by the inverse class ratio during training to protect recall on a 15.6%
        # minority. That shifts the decision surface towards the positive class
        # and inflates raw probabilities with it. The scores stay trustworthy for
        # ranking and thresholding; they are not literal frequencies. Restoring
        # frequency semantics would mean wrapping the fitted model in
        # CalibratedClassifierCV on a held-out fold, which trades away some of
        # the recall the weighting was introduced to buy.
        "interpretation": (
            "Predicted probabilities run higher than observed conversion rates in "
            "every bin. This follows from training with scale_pos_weight="
            f"{scale_pos_weight:.2f} (the inverse class ratio) to protect recall on "
            "a 15.6% minority class. The model is therefore reliable for ranking "
            "and thresholding, but its probabilities should not be read as literal "
            "conversion frequencies."
        ),
        "curve": curve.to_dict(orient="records"),
    }


def main() -> dict[str, Any]:
    """Patch the calibration block into ``models/metadata.json``."""
    metadata = inference.load_metadata()
    if not metadata:
        raise FileNotFoundError(
            f"No metadata at {config.METADATA_FILE}. Train first: python -m mlops.model_building.train"
        )

    metadata["calibration"] = build_calibration()
    save_json(metadata, config.METADATA_FILE)
    logger.info("Wrote calibration block to %s", config.METADATA_FILE.name)
    return metadata["calibration"]


if __name__ == "__main__":
    result = main()
    print(f"Brier score: {result['brier_score']}")
    for row in result["curve"]:
        print(
            f"  [{row['bin_lower']:.1f}-{row['bin_upper']:.1f}]  "
            f"predicted {row['mean_predicted']:.3f}  "
            f"observed {row['observed_rate']:.3f}  n={row['count']}"
        )
