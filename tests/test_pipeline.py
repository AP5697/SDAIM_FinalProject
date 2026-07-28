"""End-to-end tests against the real trained artifact.

These are deliberately not unit tests with mocked models. Every assertion runs
against ``models/model.joblib`` as actually produced by ``python -m src.train``,
because the failure mode this project most needs to catch is a serialised
pipeline that no longer loads or predicts in the deployment environment.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, inference  # noqa: E402


@pytest.fixture(scope="module")
def model():
    """Load the trained pipeline once for the whole module."""
    return inference.load_model()


@pytest.fixture(scope="module")
def metadata():
    """Load training metadata once for the whole module."""
    return inference.load_metadata()


@pytest.fixture(scope="module")
def schema():
    """Load the application schema once for the whole module."""
    return inference.load_schema()


@pytest.fixture()
def payload(schema):
    """Return a fresh copy of a realistic input payload."""
    return copy.deepcopy(schema["presets"]["High-intent buyer"])


# --------------------------------------------------------------------------- #
# Artifact integrity
# --------------------------------------------------------------------------- #


def test_model_artifact_exists() -> None:
    """The serialised pipeline is present on disk."""
    assert config.MODEL_FILE.exists(), f"Missing artifact: {config.MODEL_FILE}"


def test_model_fits_free_tier_size_budget() -> None:
    """The artifact stays inside the Hugging Face free-tier guardrail."""
    size_mb = config.MODEL_FILE.stat().st_size / (1024 * 1024)
    assert size_mb < config.MAX_MODEL_SIZE_MB, f"Model is {size_mb:.2f} MB"


def test_pipeline_carries_its_own_preprocessing(model) -> None:
    """Preprocessing travels with the model, so the app cannot drift from it."""
    assert "engineer" in model.named_steps
    assert "preprocess" in model.named_steps
    assert "classifier" in model.named_steps


def test_health_check_passes() -> None:
    """The CI smoke check reports a loadable, scoring model."""
    health = inference.model_health()
    assert health["ok"], health.get("error")
    assert 0.0 <= health["smoke_test_probability"] <= 1.0


# --------------------------------------------------------------------------- #
# Prediction behaviour
# --------------------------------------------------------------------------- #


def test_predict_returns_valid_probability(payload, model, metadata) -> None:
    """A well-formed payload produces a probability in [0, 1]."""
    result = inference.predict(payload, model, metadata)
    assert 0.0 <= result.probability <= 1.0
    assert result.predicted_class in (0, 1)
    assert result.label in (config.POSITIVE_CLASS_LABEL, config.NEGATIVE_CLASS_LABEL)


def test_prediction_is_deterministic(payload, model, metadata) -> None:
    """Identical input yields identical output across calls."""
    first = inference.predict(payload, model, metadata)
    second = inference.predict(payload, model, metadata)
    assert first.probability == second.probability


def test_high_intent_scores_above_bouncer(schema, model, metadata) -> None:
    """The model separates the two most distinct real behavioural segments."""
    buyer = inference.predict(schema["presets"]["High-intent buyer"], model, metadata)
    bouncer = inference.predict(schema["presets"]["Quick bouncer"], model, metadata)
    assert buyer.probability > bouncer.probability


def test_page_values_moves_the_prediction(payload, model, metadata) -> None:
    """Raising the dominant feature raises the predicted probability."""
    low = copy.deepcopy(payload)
    low["PageValues"] = 0.0
    high = copy.deepcopy(payload)
    high["PageValues"] = 100.0
    assert (
        inference.predict(high, model, metadata).probability
        > inference.predict(low, model, metadata).probability
    )


def test_threshold_governs_the_class_label(payload, model, metadata) -> None:
    """The reported class is consistent with the probability and threshold."""
    result = inference.predict(payload, model, metadata)
    assert result.predicted_class == int(result.probability >= result.threshold)


# --------------------------------------------------------------------------- #
# Boundary and edge cases
# --------------------------------------------------------------------------- #


def test_all_zero_session_is_handled(schema, model, metadata) -> None:
    """A session with no recorded activity scores without dividing by zero."""
    payload = {name: 0 for name in config.TRUE_NUMERIC_FEATURES}
    payload.update(
        {
            "Month": schema["categorical_levels"]["Month"][0],
            "VisitorType": schema["categorical_levels"]["VisitorType"][0],
            "Weekend": False,
            "OperatingSystems": schema["categorical_levels"]["OperatingSystems"][0],
            "Browser": schema["categorical_levels"]["Browser"][0],
            "Region": schema["categorical_levels"]["Region"][0],
            "TrafficType": schema["categorical_levels"]["TrafficType"][0],
        }
    )
    result = inference.predict(payload, model, metadata)
    assert 0.0 <= result.probability <= 1.0


def test_extreme_values_do_not_crash(payload, model, metadata) -> None:
    """Values far beyond the observed maxima still produce a probability."""
    payload["ProductRelated"] = 100_000
    payload["ProductRelated_Duration"] = 1_000_000
    payload["PageValues"] = 50_000
    result = inference.predict(payload, model, metadata)
    assert 0.0 <= result.probability <= 1.0


def test_unseen_categorical_level_is_absorbed(payload, model, metadata) -> None:
    """An unknown category encodes to all-zeros instead of raising.

    This is the behaviour that keeps the deployed app alive when the analytics
    platform introduces a new browser or traffic-source identifier.
    """
    payload["Browser"] = 999
    payload["TrafficType"] = 888
    result = inference.predict(payload, model, metadata)
    assert 0.0 <= result.probability <= 1.0


def test_missing_field_is_rejected(payload, model, metadata) -> None:
    """An incomplete payload fails loudly rather than silently imputing."""
    del payload["PageValues"]
    with pytest.raises(ValueError, match="Missing required fields"):
        inference.predict(payload, model, metadata)


def test_negative_value_is_rejected(payload) -> None:
    """Negative counts and durations are impossible and are caught."""
    payload["ProductRelated"] = -5
    problems = inference.validate_payload(payload)
    assert any("cannot be negative" in problem for problem in problems)


def test_out_of_range_rate_is_rejected(payload) -> None:
    """A bounce rate above 1.0 is caught before it reaches the model."""
    payload["BounceRates"] = 4.2
    problems = inference.validate_payload(payload)
    assert any("between 0 and 1" in problem for problem in problems)


def test_contradictory_duration_is_rejected(payload) -> None:
    """Time on pages that were never visited is flagged as inconsistent."""
    payload["Informational"] = 0
    payload["Informational_Duration"] = 300
    problems = inference.validate_payload(payload)
    assert any("never visited" in problem for problem in problems)


def test_valid_payload_produces_no_complaints(payload) -> None:
    """A realistic payload passes validation cleanly."""
    assert inference.validate_payload(payload) == []


# --------------------------------------------------------------------------- #
# Batch scoring
# --------------------------------------------------------------------------- #


def test_batch_scoring_returns_a_row_per_input(schema, model, metadata) -> None:
    """Batch mode scores every uploaded row and appends the output columns."""
    frame = pd.DataFrame(list(schema["presets"].values()))
    scored = inference.predict_batch(frame, model, metadata)
    assert len(scored) == len(frame)
    for column in ("purchase_probability", "prediction", "recommendation"):
        assert column in scored.columns
    assert scored["purchase_probability"].between(0, 1).all()


def test_batch_rejects_missing_columns(schema, model, metadata) -> None:
    """An upload missing required columns fails with a specific message."""
    frame = pd.DataFrame(list(schema["presets"].values())).drop(columns=["PageValues"])
    with pytest.raises(ValueError, match="missing"):
        inference.predict_batch(frame, model, metadata)


def test_batch_matches_single_prediction(schema, model, metadata) -> None:
    """Batch and single-row paths agree, proving one shared code path."""
    preset = schema["presets"]["High-intent buyer"]
    single = inference.predict(preset, model, metadata).probability
    batch = inference.predict_batch(pd.DataFrame([preset]), model, metadata)
    assert abs(single - float(batch["purchase_probability"].iloc[0])) < 1e-3


# --------------------------------------------------------------------------- #
# Decision-policy override
# --------------------------------------------------------------------------- #


def test_threshold_override_changes_the_label(payload, model, metadata) -> None:
    """A stricter threshold can only ever withdraw a positive prediction."""
    probability = inference.predict(payload, model, metadata).probability
    lenient = inference.predict(payload, model, metadata, probability - 0.01)
    strict = inference.predict(payload, model, metadata, min(probability + 0.01, 1.0))

    assert lenient.predicted_class == 1
    assert strict.predicted_class == 0
    # The probability is a property of the model, not the decision rule.
    assert abs(lenient.probability - strict.probability) < 1e-9


def test_threshold_override_propagates_to_batch(schema, model, metadata) -> None:
    """Raising the threshold cannot increase the number of flagged sessions."""
    frame = pd.DataFrame(list(schema["presets"].values()))
    lenient = inference.predict_batch(frame, model, metadata, 0.10)
    strict = inference.predict_batch(frame, model, metadata, 0.90)

    flagged_lenient = (lenient["prediction"] == config.POSITIVE_CLASS_LABEL).sum()
    flagged_strict = (strict["prediction"] == config.POSITIVE_CLASS_LABEL).sum()
    assert flagged_strict <= flagged_lenient


# --------------------------------------------------------------------------- #
# Per-prediction explanation and what-if analysis
# --------------------------------------------------------------------------- #


def test_explanation_matches_the_scored_probability(payload, model) -> None:
    """Attribution reports the same probability the scorer produced."""
    explanation = inference.explain_prediction(payload, model)
    predicted = inference.predict(payload, model).probability
    assert abs(explanation.probability - predicted) < 1e-9


def test_explanation_ranks_page_values_first_for_a_buyer(payload, model) -> None:
    """The strongest driver of a high-intent session is its PageValues."""
    explanation = inference.explain_prediction(payload, model)
    if not explanation.available:
        pytest.skip("SHAP unavailable in this environment")
    assert explanation.contributions
    assert explanation.contributions[0].feature == "PageValues"
    assert explanation.contributions[0].value > 0


def test_explanation_contributions_are_sorted_by_magnitude(payload, model) -> None:
    """Contributions arrive ranked so the UI can render them directly."""
    explanation = inference.explain_prediction(payload, model)
    if not explanation.available:
        pytest.skip("SHAP unavailable in this environment")
    magnitudes = [abs(c.value) for c in explanation.contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_survives_an_all_zero_session(schema, model) -> None:
    """Attribution degrades gracefully rather than raising on edge input."""
    blank = copy.deepcopy(schema["presets"]["Quick bouncer"])
    for name in config.TRUE_NUMERIC_FEATURES:
        blank[name] = 0
    explanation = inference.explain_prediction(blank, model)
    assert 0.0 <= explanation.probability <= 1.0


def test_sensitivity_curve_spans_the_requested_grid(payload, model, schema) -> None:
    """The sweep returns one probability per grid point, all in range."""
    curve = inference.sensitivity_curve(
        payload, "PageValues", model, n_points=12, schema=schema
    )
    assert len(curve) == 12
    assert curve["probability"].between(0, 1).all()
    assert curve["value"].is_monotonic_increasing


def test_sensitivity_curve_moves_for_an_influential_feature(
    payload, model, schema
) -> None:
    """Sweeping PageValues visibly changes the prediction."""
    curve = inference.sensitivity_curve(
        payload, "PageValues", model, n_points=20, schema=schema
    )
    assert curve["probability"].max() - curve["probability"].min() > 0.05


def test_sensitivity_curve_rejects_a_categorical_feature(payload, model) -> None:
    """Only numeric fields are sweepable; asking for a category is an error."""
    with pytest.raises(ValueError, match="not a numeric feature"):
        inference.sensitivity_curve(payload, "Month", model)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


def test_calibration_block_is_present_and_consistent(metadata) -> None:
    """The persisted calibration curve is well-formed and covers the test set."""
    calibration = metadata.get("calibration")
    if not calibration:
        pytest.skip("calibration not computed; run python -m scripts.add_calibration")

    curve = calibration["curve"]
    assert curve
    assert sum(row["count"] for row in curve) == calibration["n_test_sessions"]
    for row in curve:
        assert 0.0 <= row["mean_predicted"] <= 1.0
        assert 0.0 <= row["observed_rate"] <= 1.0
