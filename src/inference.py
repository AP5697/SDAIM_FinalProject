"""Thin prediction wrapper - the only module the Streamlit application imports.

Keeping the application's contact with the model surface this narrow means the
UI never re-implements preprocessing. It hands over raw Google Analytics field
values; the serialised pipeline performs feature engineering, imputation,
scaling and encoding exactly as it did during training.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

_model_cache: dict[str, Pipeline] = {}

CONFIDENCE_BANDS: tuple[tuple[float, str], ...] = (
    (0.80, "Very high"),
    (0.60, "High"),
    (0.40, "Moderate"),
    (0.20, "Low"),
    (0.00, "Very low"),
)


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of scoring a single browsing session.

    Attributes:
        probability: Model probability that the session ends in a purchase.
        predicted_class: 1 if the probability meets the operating threshold.
        label: Human-readable class label.
        threshold: Operating threshold applied.
        confidence_band: Qualitative description of the probability.
        recommendation: Suggested commercial action.
        drivers: Global feature importances relevant to this model.
    """

    probability: float
    predicted_class: int
    label: str
    threshold: float
    confidence_band: str
    recommendation: str
    drivers: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Contribution:
    """One feature's additive contribution to a single prediction.

    Attributes:
        feature: Transformed feature name, e.g. ``PageValues`` or ``Month_Nov``.
        value: SHAP value in log-odds space. Positive pushes towards purchase.
        display_value: The raw input value this feature was derived from.
    """

    feature: str
    value: float
    display_value: str


@dataclass(frozen=True)
class Explanation:
    """Per-session SHAP attribution for one prediction.

    Attributes:
        base_probability: Model output for an average session (the SHAP base
            value converted from log-odds to a probability).
        probability: Probability actually predicted for this session.
        contributions: Per-feature contributions, largest absolute value first.
        available: False when SHAP could not be computed, in which case the
            application falls back to global importances.
        message: Explanation of why attribution is unavailable, if applicable.
    """

    base_probability: float
    probability: float
    contributions: list[Contribution] = field(default_factory=list)
    available: bool = True
    message: str = ""


def load_model(path: Path | None = None) -> Pipeline:
    """Load the serialised pipeline, caching it per process.

    Args:
        path: Model path. Defaults to :data:`config.MODEL_FILE`.

    Returns:
        The fitted pipeline.

    Raises:
        FileNotFoundError: If the artifact is missing, with instructions for
            regenerating it.
    """
    path = path or config.MODEL_FILE
    key = str(path)
    if key in _model_cache:
        return _model_cache[key]

    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train it first with:\n"
            f"    python -m src.train"
        )

    model = joblib.load(path)
    _model_cache[key] = model
    logger.info("Loaded model from %s", path.name)
    return model


def load_metadata(path: Path | None = None) -> dict[str, Any]:
    """Load training metadata written alongside the model.

    Args:
        path: Metadata path. Defaults to :data:`config.METADATA_FILE`.

    Returns:
        The metadata mapping, or an empty dict if the file is absent.
    """
    path = path or config.METADATA_FILE
    if not path.exists():
        logger.warning("Metadata not found at %s", path)
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_schema(path: Path | None = None) -> dict[str, Any]:
    """Load the data-derived input schema used to configure the UI widgets.

    Args:
        path: Schema path. Defaults to ``models/app_schema.json``.

    Returns:
        The schema mapping.

    Raises:
        FileNotFoundError: If the schema is missing, with regeneration steps.
    """
    path = path or (config.MODEL_DIR / "app_schema.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Application schema not found at {path}. Generate it with:\n"
            f"    python -m scripts.build_app_assets"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def get_operating_threshold(metadata: dict[str, Any] | None = None) -> float:
    """Return the decision threshold chosen during training.

    Args:
        metadata: Pre-loaded metadata. Loaded from disk if omitted.

    Returns:
        The operating threshold, defaulting to 0.5 if unavailable.
    """
    metadata = metadata if metadata is not None else load_metadata()
    return float(metadata.get("operating_threshold", 0.50))


def _confidence_band(probability: float) -> str:
    """Map a probability to a qualitative confidence band."""
    for cutoff, label in CONFIDENCE_BANDS:
        if probability >= cutoff:
            return label
    return CONFIDENCE_BANDS[-1][1]


def _recommendation(probability: float, threshold: float) -> str:
    """Translate a probability into a commercial action.

    The bands exist so the application communicates a decision rather than a
    bare number: a session just below the threshold is the most valuable
    intervention target, because it is winnable but not yet won.
    """
    if probability >= threshold + 0.25:
        return "High intent - let the session run; avoid discounting a sale you already have."
    if probability >= threshold:
        return "Likely to convert - light-touch nudge such as free-shipping messaging."
    if probability >= threshold - 0.15:
        return "On the margin - highest-value intervention target; fire the incentive."
    if probability >= 0.05:
        return "Low intent - retarget later rather than spending on this live session."
    return "Very low intent - no intervention; the spend will not be recovered."


def validate_payload(payload: dict[str, Any]) -> list[str]:
    """Check a raw input payload for missing or malformed fields.

    Args:
        payload: Mapping of feature name to value.

    Returns:
        A list of human-readable problems. Empty means the payload is valid.
    """
    problems: list[str] = []

    missing = [name for name in config.ALL_FEATURES if name not in payload]
    if missing:
        problems.append(f"Missing required fields: {', '.join(sorted(missing))}")

    for name in config.TRUE_NUMERIC_FEATURES:
        value = payload.get(name)
        if value is None:
            continue
        if not isinstance(value, (int, float, bool)):
            problems.append(f"'{name}' must be numeric, got {type(value).__name__}")
        elif float(value) < 0:
            problems.append(f"'{name}' cannot be negative (got {value})")

    for name in ("BounceRates", "ExitRates"):
        value = payload.get(name)
        if isinstance(value, (int, float)) and not 0 <= float(value) <= 1:
            problems.append(f"'{name}' is a rate and must lie between 0 and 1 (got {value})")

    pairs = [
        ("Administrative", "Administrative_Duration"),
        ("Informational", "Informational_Duration"),
        ("ProductRelated", "ProductRelated_Duration"),
    ]
    for count_field, duration_field in pairs:
        count = payload.get(count_field)
        duration = payload.get(duration_field)
        if isinstance(count, (int, float)) and isinstance(duration, (int, float)):
            if count == 0 and duration > 0:
                problems.append(
                    f"'{duration_field}' is {duration} but '{count_field}' is 0 - "
                    f"a session cannot accumulate time on pages it never visited."
                )
    return problems


def build_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a payload dictionary into a single-row model input frame.

    Args:
        payload: Mapping of raw feature name to value.

    Returns:
        A one-row DataFrame with columns in the order the pipeline expects.

    Raises:
        ValueError: If required fields are absent.
    """
    missing = [name for name in config.ALL_FEATURES if name not in payload]
    if missing:
        raise ValueError(
            f"Cannot build a model input row - missing fields: {sorted(missing)}"
        )
    return pd.DataFrame([{name: payload[name] for name in config.ALL_FEATURES}])


def predict(
    payload: dict[str, Any] | pd.DataFrame,
    model: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> PredictionResult:
    """Score a single browsing session.

    Args:
        payload: Raw feature mapping, or a one-row DataFrame.
        model: Pre-loaded pipeline. Loaded from disk if omitted.
        metadata: Pre-loaded metadata. Loaded from disk if omitted.
        threshold: Decision threshold override. Defaults to the operating
            threshold chosen during training. The application passes this so a
            reader can compare decision policies without refitting anything -
            the probability is unchanged, only the cut-off applied to it.

    Returns:
        The prediction, its probability and the recommended action.

    Raises:
        ValueError: If the payload is invalid or has more than one row.
    """
    model = model or load_model()
    metadata = metadata if metadata is not None else load_metadata()
    threshold = (
        get_operating_threshold(metadata) if threshold is None else float(threshold)
    )

    if isinstance(payload, pd.DataFrame):
        if len(payload) != 1:
            raise ValueError(f"predict() scores one session; received {len(payload)} rows")
        frame = payload
    else:
        problems = validate_payload(payload)
        if problems:
            raise ValueError("Invalid input:\n- " + "\n- ".join(problems))
        frame = build_frame(payload)

    probability = float(model.predict_proba(frame)[0, 1])
    predicted = int(probability >= threshold)

    return PredictionResult(
        probability=probability,
        predicted_class=predicted,
        label=config.POSITIVE_CLASS_LABEL if predicted else config.NEGATIVE_CLASS_LABEL,
        threshold=threshold,
        confidence_band=_confidence_band(probability),
        recommendation=_recommendation(probability, threshold),
        drivers=metadata.get("top_features", {}),
    )


@dataclass(frozen=True)
class ColumnAlignment:
    """Result of mapping an arbitrary uploaded file onto the model's schema.

    Attributes:
        frame: The uploaded data reduced to the model's features, renamed and
            cast to the dtypes the pipeline was fitted on.
        strategy: How the columns were matched - ``exact``, ``normalised`` or
            ``positional``.
        mapping: Required feature name -> the source column it was taken from.
        notes: Human-readable remarks about coercions and assumptions applied.
        trustworthy: False when columns were matched by position, meaning the
            mapping is an assumption the uploader must confirm rather than
            something the data itself supports.
    """

    frame: pd.DataFrame
    strategy: str
    mapping: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    trustworthy: bool = True


def _normalise_name(name: Any) -> str:
    """Reduce a column name to a comparable form.

    ``Page Values``, ``page_values`` and ``PageValues`` all describe the same
    field when exported by different analytics tools, so casing, spacing and
    punctuation are stripped before matching.

    Args:
        name: Raw column label.

    Returns:
        Lowercase alphanumeric form of the name.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def fitted_categories(model: Pipeline) -> dict[str, list[Any]]:
    """Read the category values the encoder was actually fitted on.

    The encoder is the authority on what counts as a known level, so canonical
    spellings are taken from it rather than duplicated in a constant that could
    drift away from the artifact after a retrain.

    Args:
        model: The fitted pipeline.

    Returns:
        Mapping of categorical feature name to its fitted categories. Empty if
        the encoder cannot be located.
    """
    try:
        column_transformer = model.named_steps["preprocess"]
        for _, transformer, columns in column_transformer.transformers_:
            # Located by capability rather than by branch or step name, so
            # renaming a pipeline step cannot silently empty this mapping and
            # disable canonicalisation without any test noticing.
            encoder = None
            if hasattr(transformer, "categories_"):
                encoder = transformer
            elif hasattr(transformer, "named_steps"):
                encoder = next(
                    (
                        step
                        for step in transformer.named_steps.values()
                        if hasattr(step, "categories_")
                    ),
                    None,
                )
            if encoder is None:
                continue
            return {
                str(column): list(values)
                for column, values in zip(columns, encoder.categories_)
            }
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read fitted categories: %s", exc)
    return {}


def _coerce_to_training_types(
    frame: pd.DataFrame,
    categories: dict[str, list[Any]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Cast columns to the exact dtypes the fitted encoder expects.

    This is not cosmetic. ``OneHotEncoder`` was fitted with
    ``handle_unknown="ignore"``, so a category it does not recognise silently
    becomes an all-zero row rather than an error - and an integer-coded field
    arriving as ``2.0`` instead of ``2`` does not match the fitted category.
    The prediction still succeeds and is quietly wrong. Casting here keeps the
    uploaded data on the same footing as the training data.

    Args:
        frame: Data already reduced to the model's feature columns.

    Returns:
        A ``(coerced_frame, notes)`` tuple describing any values that could not
        be interpreted and were left for the pipeline's imputer to fill.
    """
    out = frame.copy()
    notes: list[str] = []

    for name in config.TRUE_NUMERIC_FEATURES:
        before = out[name].notna().sum()
        out[name] = pd.to_numeric(out[name], errors="coerce").astype(float)
        lost = int(before - out[name].notna().sum())
        if lost:
            notes.append(
                f"{name}: {lost} value(s) were not numeric and will be imputed."
            )

    for name in config.INTEGER_CODED_CATEGORICAL_FEATURES:
        numeric = pd.to_numeric(out[name], errors="coerce")
        unparsed = int(numeric.isna().sum() - out[name].isna().sum())
        # -1 is deliberately outside every fitted category, so the encoder
        # treats it as unknown instead of the imputer inventing a plausible
        # browser or region that the row never actually had.
        out[name] = numeric.fillna(-1).round().astype(int)
        if unparsed > 0:
            notes.append(
                f"{name}: {unparsed} value(s) were not integer codes and are "
                f"treated as unknown."
            )

    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}

    def _to_bool(value: Any) -> bool:
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return bool(value) if pd.notna(value) else False

    if "Weekend" in out.columns:
        out["Weekend"] = out["Weekend"].map(_to_bool).astype(bool)

    # Text categories are matched to the fitted spelling case-insensitively.
    # "nov" and "Nov" are the same month, but the encoder only recognises the
    # spelling it saw in training - anything else becomes an all-zero row and
    # quietly removes the month signal from the prediction.
    categories = categories or {}
    for name in ("Month", "VisitorType"):
        out[name] = out[name].astype(str).str.strip()
        known = categories.get(name)
        if not known:
            continue
        canonical = {str(level).strip().casefold(): level for level in known}
        recased = 0

        def _canonicalise(value: Any) -> Any:
            nonlocal recased
            match = canonical.get(str(value).strip().casefold())
            if match is not None and match != value:
                recased += 1
                return match
            return value

        out[name] = out[name].map(_canonicalise)
        if recased:
            notes.append(
                f"{name}: {recased} value(s) matched a known category after "
                f"ignoring case."
            )

    return out, notes


def align_columns(
    frame: pd.DataFrame,
    allow_positional: bool = True,
    model: Pipeline | None = None,
) -> ColumnAlignment:
    """Map an uploaded file onto the model's feature schema.

    Three strategies are tried in descending order of confidence:

    1. **exact** - every required column is present by name.
    2. **normalised** - names match once casing and punctuation are ignored,
       which covers the same fields exported by a different tool.
    3. **positional** - names carry no usable information, so the first columns
       are assumed to be the features in their trained order.

    Positional matching is an assumption about the file, not a fact derived
    from it: an unrelated table with enough columns will be scored happily and
    the resulting probabilities will be meaningless. The strategy is reported
    back so the caller can surface that rather than hide it.

    Args:
        frame: The uploaded data, exactly as parsed.
        allow_positional: Whether to fall back to positional matching.
        model: Pre-loaded pipeline, used to read the encoder's fitted category
            spellings. Loaded from disk if omitted.

    Returns:
        The alignment, including the strategy used and any coercion notes.

    Raises:
        ValueError: If the file has too few columns, or names do not match and
            positional matching is disallowed.
    """
    required = list(config.ALL_FEATURES)
    notes: list[str] = []
    categories = fitted_categories(model or load_model())

    present = [name for name in required if name in frame.columns]
    if len(present) == len(required):
        selected = frame[required].copy()
        mapping = {name: name for name in required}
        coerced, coercion_notes = _coerce_to_training_types(selected, categories)
        return ColumnAlignment(
            frame=coerced,
            strategy="exact",
            mapping=mapping,
            notes=notes + coercion_notes,
        )

    lookup: dict[str, str] = {}
    for column in frame.columns:
        lookup.setdefault(_normalise_name(column), str(column))
    resolved = {name: lookup.get(_normalise_name(name)) for name in required}

    if all(resolved.values()):
        mapping = {name: str(source) for name, source in resolved.items()}
        selected = frame[[mapping[name] for name in required]].copy()
        selected.columns = required
        renamed = [name for name in required if mapping[name] != name]
        if renamed:
            notes.append(
                f"Matched {len(renamed)} column(s) by name ignoring case and "
                f"punctuation, e.g. '{mapping[renamed[0]]}' -> '{renamed[0]}'."
            )
        coerced, coercion_notes = _coerce_to_training_types(selected, categories)
        return ColumnAlignment(
            frame=coerced,
            strategy="normalised",
            mapping=mapping,
            notes=notes + coercion_notes,
        )

    unmatched = [name for name, source in resolved.items() if source is None]

    if not allow_positional:
        raise ValueError(
            f"Could not find {len(unmatched)} required column(s): {sorted(unmatched)}"
        )

    if frame.shape[1] < len(required):
        raise ValueError(
            f"Positional matching needs at least {len(required)} columns; this "
            f"file has {frame.shape[1]}."
        )

    sources = [str(column) for column in frame.columns[: len(required)]]
    mapping = dict(zip(required, sources))
    selected = frame.iloc[:, : len(required)].copy()
    selected.columns = required

    notes.append(
        f"{len(unmatched)} column name(s) did not match the schema, so columns "
        f"were mapped by position instead."
    )
    if frame.shape[1] > len(required):
        notes.append(
            f"Only the first {len(required)} of {frame.shape[1]} columns were used."
        )

    coerced, coercion_notes = _coerce_to_training_types(selected, categories)
    return ColumnAlignment(
        frame=coerced,
        strategy="positional",
        mapping=mapping,
        notes=notes + coercion_notes,
        trustworthy=False,
    )


def predict_batch(
    frame: pd.DataFrame,
    model: Pipeline | None = None,
    metadata: dict[str, Any] | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Score many sessions at once, for the application's batch upload mode.

    Args:
        frame: DataFrame containing at least the required feature columns.
        model: Pre-loaded pipeline. Loaded from disk if omitted.
        metadata: Pre-loaded metadata. Loaded from disk if omitted.
        threshold: Decision threshold override. Defaults to the trained
            operating threshold.

    Returns:
        A copy of ``frame`` with ``purchase_probability``, ``prediction`` and
        ``recommendation`` columns appended.

    Raises:
        ValueError: If required columns are absent.
    """
    model = model or load_model()
    threshold = (
        get_operating_threshold(metadata) if threshold is None else float(threshold)
    )

    missing = [name for name in config.ALL_FEATURES if name not in frame.columns]
    if missing:
        raise ValueError(
            f"Uploaded file is missing {len(missing)} required column(s): {sorted(missing)}"
        )

    probabilities = model.predict_proba(frame[config.ALL_FEATURES])[:, 1]
    out = frame.copy()
    out["purchase_probability"] = probabilities.round(4)
    out["prediction"] = [
        config.POSITIVE_CLASS_LABEL if p >= threshold else config.NEGATIVE_CLASS_LABEL
        for p in probabilities
    ]
    out["recommendation"] = [_recommendation(p, threshold) for p in probabilities]
    return out


def _transformed_feature_names(model: Pipeline) -> list[str]:
    """Return output feature names of the preprocessing stage.

    Args:
        model: The fitted pipeline.

    Returns:
        Feature names as produced by the ColumnTransformer, with the
        ``num__``/``cat__`` prefixes stripped for display.
    """
    names = model.named_steps["preprocess"].get_feature_names_out()
    return [str(name).split("__", 1)[-1] for name in names]


def _raw_value_for(
    feature: str,
    payload: dict[str, Any],
    engineered: pd.DataFrame | None = None,
) -> str:
    """Map a transformed feature name back to the raw input it came from.

    Three cases need distinguishing for the reader:

    * a plain field (``PageValues``) shows its own value;
    * an engineered field (``TotalPages``) is absent from the payload, so its
      computed value is read back from the engineering step's output;
    * a one-hot column (``Month_May``) is a yes/no indicator. Showing the raw
      ``Month`` value alone would be actively misleading - "Month = Nov" next to
      a ``Month_May`` contribution reads as a contradiction - so the inactive
      case is spelled out explicitly.

    Args:
        feature: Transformed feature name.
        payload: The raw input mapping.
        engineered: Output of the pipeline's engineering step, when available.

    Returns:
        A display string, empty when the origin cannot be resolved.
    """

    def _fmt(value: Any) -> str:
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    if feature in payload:
        return _fmt(payload[feature])

    if engineered is not None and feature in engineered.columns:
        return _fmt(float(engineered.iloc[0][feature]))

    for source in config.CATEGORICAL_FEATURES:
        prefix = f"{source}_"
        if feature.startswith(prefix):
            level = feature[len(prefix) :]
            actual = payload.get(source, "")
            if str(actual) == level:
                return f"{source} = {actual}"
            return f"{source} is {actual}, not {level}"
    return ""


def explain_prediction(
    payload: dict[str, Any],
    model: Pipeline | None = None,
    top_n: int = 8,
) -> Explanation:
    """Attribute one prediction to its individual features using SHAP.

    Unlike the global importances shown on the insights page, this answers
    "why did *this* session score what it scored". SHAP values are additive in
    log-odds space: the base value plus every contribution reconstructs the
    model's output for this row exactly.

    Failure is tolerated rather than raised - the application degrades to global
    importances - because attribution is an enhancement, not a requirement.

    Args:
        payload: Raw feature mapping for one session.
        model: Pre-loaded pipeline. Loaded from disk if omitted.
        top_n: Number of contributions to return, ranked by absolute value.

    Returns:
        The attribution, with ``available`` False if SHAP was unavailable.
    """
    model = model or load_model()
    probability = float(model.predict_proba(build_frame(payload))[0, 1])

    try:
        import numpy as np
        import shap

        frame = build_frame(payload)
        engineered = model.named_steps["engineer"].transform(frame)
        transformed = model.named_steps["preprocess"].transform(engineered)
        explainer = shap.TreeExplainer(model.named_steps["classifier"])
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        row = np.asarray(values).reshape(-1)

        base = explainer.expected_value
        if isinstance(base, (list, tuple, np.ndarray)):
            base = np.asarray(base).reshape(-1)[-1]
        base_probability = float(1.0 / (1.0 + np.exp(-float(base))))

        names = _transformed_feature_names(model)
        if len(names) != row.size:
            raise ValueError(
                f"SHAP produced {row.size} values for {len(names)} feature names"
            )

        order = np.argsort(np.abs(row))[::-1][:top_n]
        contributions = [
            Contribution(
                feature=names[i],
                value=float(row[i]),
                display_value=_raw_value_for(names[i], payload, engineered),
            )
            for i in order
            if abs(float(row[i])) > 1e-9
        ]
        return Explanation(
            base_probability=base_probability,
            probability=probability,
            contributions=contributions,
        )
    except Exception as exc:  # pragma: no cover - optional enhancement
        logger.warning("Per-prediction SHAP unavailable: %s", exc)
        return Explanation(
            base_probability=float("nan"),
            probability=probability,
            available=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def sensitivity_curve(
    payload: dict[str, Any],
    feature: str,
    model: Pipeline | None = None,
    n_points: int = 40,
    schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Sweep one feature across its observed range, holding the rest fixed.

    This is a one-dimensional what-if analysis: it shows how the model would
    have scored this same session had a single field differed, which is the
    most direct demonstration that the model is not a black box.

    Args:
        payload: Raw feature mapping for the session being varied.
        feature: Name of the numeric feature to sweep.
        model: Pre-loaded pipeline. Loaded from disk if omitted.
        n_points: Number of points to evaluate across the range.
        schema: Pre-loaded application schema. Loaded from disk if omitted.

    Returns:
        A DataFrame with ``value`` and ``probability`` columns.

    Raises:
        ValueError: If the feature is not a sweepable numeric field.
    """
    if feature not in config.TRUE_NUMERIC_FEATURES:
        raise ValueError(
            f"'{feature}' is not a numeric feature; sweepable fields are "
            f"{config.TRUE_NUMERIC_FEATURES}"
        )

    model = model or load_model()
    schema = schema if schema is not None else load_schema()
    spec = schema["numeric_ranges"][feature]

    # Sweep to the 99th percentile rather than the maximum: the extreme tail is
    # sparse enough that the curve there reflects a handful of training rows.
    upper = float(spec["p99"]) or float(spec["max"]) or 1.0
    step = upper / max(n_points - 1, 1)
    grid = [round(i * step, 6) for i in range(n_points)]

    frame = pd.concat([build_frame(payload)] * len(grid), ignore_index=True)
    frame[feature] = grid
    probabilities = model.predict_proba(frame)[:, 1]

    return pd.DataFrame({"value": grid, "probability": probabilities})


def model_health() -> dict[str, Any]:
    """Report whether the model artifact loads and can score a row.

    Used by the smoke test in CI so a broken artifact fails the build rather
    than the live application.

    Returns:
        Mapping with ``ok`` plus diagnostic details or the error message.
    """
    try:
        model = load_model()
        metadata = load_metadata()
        schema = load_schema()
        result = predict(schema["presets"]["High-intent buyer"], model, metadata)
        return {
            "ok": True,
            "model_name": metadata.get("model_name", "unknown"),
            "threshold": result.threshold,
            "smoke_test_probability": round(result.probability, 4),
            "trained_at_utc": metadata.get("trained_at_utc"),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
