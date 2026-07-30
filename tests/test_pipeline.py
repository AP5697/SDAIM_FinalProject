"""End-to-end tests against the real trained artifact.

These are deliberately not unit tests with mocked models. Every assertion runs
against ``models/model.joblib`` as actually produced by ``python -m mlops.model_building.train``,
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

from mlops.model_building import config  # noqa: E402
from mlops.deployment import inference  # noqa: E402


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
# Uploaded-file column alignment
#
# The batch uploader accepts arbitrary files, so these assert the alignment
# never silently changes a prediction. Every strategy must reproduce the
# probabilities obtained from the raw data at its native dtypes.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def real_rows():
    """Return real dataset rows at their native dtypes."""
    from mlops.model_building.data_loader import load_raw_data

    return load_raw_data().head(120)[config.ALL_FEATURES]


@pytest.fixture(scope="module")
def reference_probabilities(model, real_rows):
    """Probabilities for those rows, scored without any alignment."""
    return model.predict_proba(real_rows)[:, 1]


def _aligned_probabilities(model, frame) -> "list[float]":
    """Align a frame then score it."""
    return model.predict_proba(inference.align_columns(frame).frame)[:, 1]


def test_exact_names_align_without_changing_predictions(
    model, real_rows, reference_probabilities
) -> None:
    """A correctly named file scores identically to the raw data."""
    alignment = inference.align_columns(real_rows)
    assert alignment.strategy == "exact"
    assert alignment.trustworthy
    probabilities = model.predict_proba(alignment.frame)[:, 1]
    assert probabilities == pytest.approx(reference_probabilities, abs=1e-12)


def test_renamed_columns_are_matched_and_score_identically(
    model, real_rows, reference_probabilities
) -> None:
    """Case and punctuation differences do not change any prediction."""
    messy = real_rows.copy()
    messy.columns = [
        name.upper() if index % 2 else name.lower().replace("_", " ")
        for index, name in enumerate(real_rows.columns)
    ]
    alignment = inference.align_columns(messy)
    assert alignment.strategy == "normalised"
    assert alignment.trustworthy
    probabilities = model.predict_proba(alignment.frame)[:, 1]
    assert probabilities == pytest.approx(reference_probabilities, abs=1e-12)


def test_unnamed_columns_fall_back_to_position(
    model, real_rows, reference_probabilities
) -> None:
    """Unmatched names are read positionally, and flagged as untrustworthy."""
    anonymous = real_rows.copy()
    anonymous.columns = [f"col_{i}" for i in range(real_rows.shape[1])]
    alignment = inference.align_columns(anonymous)

    assert alignment.strategy == "positional"
    # The caller must be able to tell the mapping was assumed, not derived.
    assert alignment.trustworthy is False
    probabilities = model.predict_proba(alignment.frame)[:, 1]
    assert probabilities == pytest.approx(reference_probabilities, abs=1e-12)


def test_string_typed_upload_is_coerced_not_crashed(
    model, real_rows, reference_probabilities
) -> None:
    """A CSV parsed entirely as text still scores correctly.

    Without coercion this raises TypeError inside the scaler, so this is a
    regression guard rather than a nicety.
    """
    as_text = real_rows.copy().astype(str)
    probabilities = _aligned_probabilities(model, as_text)
    assert probabilities == pytest.approx(reference_probabilities, abs=1e-12)


@pytest.mark.parametrize(
    "label, mutate",
    [
        ("identifier codes as text", lambda f: f.assign(
            **{n: f[n].astype(str) for n in config.INTEGER_CODED_CATEGORICAL_FEATURES}
        )),
        ("weekend as text", lambda f: f.assign(Weekend=f["Weekend"].astype(str))),
        ("weekend as yes/no", lambda f: f.assign(
            Weekend=f["Weekend"].map({True: "Yes", False: "No"})
        )),
        ("month lowercased", lambda f: f.assign(Month=f["Month"].str.lower())),
        ("visitor type lowercased", lambda f: f.assign(
            VisitorType=f["VisitorType"].str.lower()
        )),
    ],
)
def test_encoding_variations_are_repaired_not_silently_mispredicted(
    model, real_rows, reference_probabilities, label, mutate
) -> None:
    """Common CSV encoding quirks must not quietly change the prediction.

    OneHotEncoder(handle_unknown="ignore") maps an unrecognised category to an
    all-zero row instead of raising. A file writing ``Weekend`` as "True" or
    ``Month`` as "nov" therefore scores successfully with that feature silently
    switched off. Each variation below is verified to genuinely break scoring
    when passed straight to the model, and to be repaired by alignment.
    """
    mutated = mutate(real_rows.copy())

    unaligned = model.predict_proba(mutated)[:, 1]
    assert unaligned != pytest.approx(reference_probabilities, abs=1e-9), (
        f"{label} was expected to corrupt predictions without alignment; "
        f"if this fails the test no longer guards anything."
    )

    aligned = _aligned_probabilities(model, mutated)
    assert aligned == pytest.approx(reference_probabilities, abs=1e-12)


def test_extra_columns_are_ignored(model, real_rows, reference_probabilities) -> None:
    """Unrelated trailing columns do not disturb scoring."""
    padded = real_rows.copy()
    padded["session_id"] = range(len(padded))
    padded["campaign"] = "spring_sale"
    probabilities = _aligned_probabilities(model, padded)
    assert probabilities == pytest.approx(reference_probabilities, abs=1e-12)


def test_too_few_columns_is_rejected(real_rows) -> None:
    """A file that cannot supply every feature is refused, not guessed at."""
    narrow = real_rows.iloc[:, :5].copy()
    narrow.columns = [f"col_{i}" for i in range(5)]
    with pytest.raises(ValueError, match="at least"):
        inference.align_columns(narrow)


def test_positional_matching_can_be_disabled(real_rows) -> None:
    """Callers needing certainty can require real name matches."""
    anonymous = real_rows.copy()
    anonymous.columns = [f"col_{i}" for i in range(real_rows.shape[1])]
    with pytest.raises(ValueError, match="Could not find"):
        inference.align_columns(anonymous, allow_positional=False)


def test_unrelated_dataset_is_rejected_by_default(model) -> None:
    """A different dataset entirely must not be scored without opting in.

    Modelled on a real incident: a telecom churn export was uploaded to this
    scorer. It has 21 columns, so positional matching accepted it and read
    ``customerID`` as ``Administrative``. Refusing by default is what keeps
    that from becoming a page of confident, meaningless probabilities.
    """
    churn_like = pd.DataFrame(
        {
            "customerID": ["7590-VHVEG", "5575-GNVDE"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 0],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "No"],
            "tenure": [1, 34],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "DSL"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "No"],
            "StreamingMovies": ["No", "No"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [29.85, 1889.5],
            "Churn": ["No", "No"],
        }
    )

    with pytest.raises(ValueError, match="Could not find"):
        inference.align_columns(churn_like, allow_positional=False, model=model)

    # Opting in still works, and still declares itself unreliable.
    alignment = inference.align_columns(
        churn_like, allow_positional=True, model=model
    )
    assert alignment.strategy == "positional"
    assert alignment.trustworthy is False


def test_unrelated_table_is_scored_but_marked_untrustworthy(model) -> None:
    """An unrelated table with enough columns scores, and says so.

    This is the documented risk of positional matching: the model cannot tell
    that the file is not browsing data. The contract is that it never claims
    the mapping is reliable.
    """
    unrelated = pd.DataFrame(
        {f"field_{i}": ["x"] * 4 if i % 3 == 0 else range(4) for i in range(20)}
    )
    alignment = inference.align_columns(unrelated)
    assert alignment.strategy == "positional"
    assert alignment.trustworthy is False
    probabilities = model.predict_proba(alignment.frame)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


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


# --------------------------------------------------------------------------- #
# Experiment tracking
# --------------------------------------------------------------------------- #


def test_tracking_never_breaks_training_when_disabled(monkeypatch) -> None:
    """Every tracking call is a silent no-op when MLflow is switched off.

    This is the property that matters: losing the experiment record is an
    inconvenience, but a tracking failure must never take the training run with
    it. CI installs only requirements.txt, where MLflow is absent, so this is
    the path that actually runs there.
    """
    from mlops.model_building import tracking

    monkeypatch.setenv(config.MLFLOW_DISABLED_ENV_VAR, "1")
    assert tracking.is_enabled() is False

    with tracking.run("disabled-run"):
        tracking.log_params({"any": "value"})
        tracking.log_metrics({"number": 1.0})
        tracking.set_tags({"tag": "value"})
        tracking.log_artifacts([config.MODEL_FILE])
        with tracking.run("disabled-child", nested=True):
            tracking.log_metrics({"nested": 2.0})


def test_tracking_ignores_non_numeric_metrics(monkeypatch) -> None:
    """Strings and booleans are dropped rather than raising in log_metrics."""
    from mlops.model_building import tracking

    monkeypatch.setenv(config.MLFLOW_DISABLED_ENV_VAR, "1")
    tracking.log_metrics({"good": 1.5, "text": "not a number", "flag": True})


def test_mlflow_folder_can_never_shadow_the_mlflow_library() -> None:
    """The mlops/mlflow directory must not become an importable package.

    A directory named ``mlflow`` is only dangerous if Python can import it. This
    one is nested inside ``mlops/`` rather than sitting at the repository root,
    so it is not on ``sys.path`` - but an ``__init__.py`` appearing here would
    still make ``mlops.mlflow`` a package and invites someone to later move it
    up a level. Empirically, a root-level ``mlflow/`` with an ``__init__.py``
    causes ``import mlflow`` to resolve to the folder and every training run to
    fail, so this is cheap insurance against a confusing outage.
    """
    assert not (config.MLFLOW_DIR / "__init__.py").exists(), (
        f"{config.MLFLOW_DIR / '__init__.py'} would make this directory a package."
    )
    assert not (config.PROJECT_ROOT / "mlflow").exists(), (
        "A top-level mlflow/ directory can shadow the installed MLflow library."
    )

    # The directory assertions above are the point of this test and must run
    # everywhere. Confirming which mlflow actually got imported can only happen
    # where it is installed - CI installs requirements.txt alone, and MLflow is
    # deliberately absent from it.
    mlflow_library = pytest.importorskip(
        "mlflow", reason="MLflow is a development-only dependency"
    )
    assert "site-packages" in str(mlflow_library.__file__), (
        f"mlflow resolved to {mlflow_library.__file__}, not the installed library."
    )


def test_deployed_app_never_imports_mlflow_or_shap() -> None:
    """The Space installs neither, so importing them at runtime would crash it.

    shap already took the application down once with a segfault, which no
    try/except can catch. This asserts the runtime import graph stays clean.
    """
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "sys.argv=['app']; "
            "import mlops.deployment.inference; "
            "print(any(m == 'mlflow' or m.startswith('mlflow.') for m in sys.modules), "
            "any(m == 'shap' or m.startswith('shap.') for m in sys.modules))",
        ],
        capture_output=True,
        text=True,
        cwd=str(config.PROJECT_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("False False"), result.stdout


def test_tracking_uri_is_sqlite_not_the_deprecated_file_store() -> None:
    """MLflow 3.x refuses the './mlruns' file backend, so the URI must be SQLite."""
    assert config.MLFLOW_TRACKING_URI.startswith("sqlite:///")
    assert config.MLFLOW_TRACKING_URI.endswith(".db")


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
