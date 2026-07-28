"""Purchase Intent Scorer - Streamlit application.

Scores a live e-commerce browsing session against a trained classifier and
recommends whether to spend money intervening in it.

The application deliberately performs no preprocessing of its own. It collects
raw Google Analytics field values and passes them to the serialised pipeline,
which applies the identical feature engineering, imputation, scaling and
encoding used during training.

Run locally with::

    streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config, inference  # noqa: E402

APP_TITLE = "Purchase Intent Scorer"
APP_ICON = "🛒"
HISTORY_KEY = "prediction_history"
MAX_HISTORY_ROWS = 200

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main > div { padding-top: 1.2rem; }
    .app-header {
        background: linear-gradient(120deg, #1e3a8a 0%, #2563eb 55%, #3b82f6 100%);
        padding: 1.4rem 1.8rem; border-radius: 14px; color: #ffffff;
        margin-bottom: 1.4rem; box-shadow: 0 6px 18px rgba(37,99,235,0.22);
    }
    .app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }
    .app-header p  { margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.95rem; }
    .result-card {
        border-radius: 14px; padding: 1.4rem 1.6rem; margin: 0.6rem 0 1rem;
        border: 1px solid #e2e8f0; background: #ffffff;
        box-shadow: 0 2px 10px rgba(15,23,42,0.06);
    }
    .result-positive { border-left: 6px solid #16a34a; background: #f0fdf4; }
    .result-negative { border-left: 6px solid #94a3b8; background: #f8fafc; }
    .result-margin   { border-left: 6px solid #f59e0b; background: #fffbeb; }
    .prob-value { font-size: 3rem; font-weight: 700; line-height: 1.05; color: #0f172a; }
    .prob-label { font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.09em; color: #64748b; }
    .meter-track { background: #e2e8f0; border-radius: 999px; height: 15px; overflow: hidden; margin: 0.55rem 0; }
    .meter-fill  { height: 100%; border-radius: 999px; transition: width 0.35s ease; }
    .recommendation {
        background: #eff6ff; border-left: 4px solid #2563eb; padding: 0.75rem 1rem;
        border-radius: 8px; margin-top: 0.9rem; font-size: 0.93rem; color: #1e3a8a;
    }
    .footnote { color: #64748b; font-size: 0.82rem; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached resource loading
# --------------------------------------------------------------------------- #


@st.cache_resource(show_spinner="Loading the trained model...")
def get_model() -> Any:
    """Load the serialised pipeline once per application process."""
    return inference.load_model()


@st.cache_data(show_spinner=False)
def get_metadata() -> dict[str, Any]:
    """Load the training metadata written alongside the model."""
    return inference.load_metadata()


@st.cache_data(show_spinner=False)
def get_schema() -> dict[str, Any]:
    """Load the data-derived input schema used to configure widgets."""
    return inference.load_schema()


def init_state() -> None:
    """Initialise session state keys used across reruns."""
    if HISTORY_KEY not in st.session_state:
        st.session_state[HISTORY_KEY] = []
    if "preset_payload" not in st.session_state:
        st.session_state["preset_payload"] = None


def threshold_policies(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe the decision policies a reader can switch between.

    Both thresholds were selected during training on the held-out test set; the
    difference is the objective they optimise. Exposing the choice makes the
    precision/recall trade-off something a reader can operate rather than read
    about.

    Args:
        metadata: Training metadata.

    Returns:
        Mapping of policy label to its threshold and rationale.
    """
    f1_threshold = float(metadata.get("operating_threshold", 0.50))
    value_threshold = metadata.get("alternative_threshold_max_expected_value")
    policies: dict[str, dict[str, Any]] = {
        "Balanced (max F1)": {
            "threshold": f1_threshold,
            "why": (
                "Maximises F1 on the held-out test set - the default operating "
                "point, balancing missed buyers against wasted incentives."
            ),
        }
    }
    if value_threshold is not None and float(value_threshold) != f1_threshold:
        policies["Profit-first (max campaign value)"] = {
            "threshold": float(value_threshold),
            "why": (
                "Maximises modelled net campaign value. Flags fewer sessions, so "
                "it spends less on false positives and misses more real buyers."
            ),
        }
    return policies


def render_header(subtitle: str) -> None:
    """Render the gradient page header.

    Args:
        subtitle: Sub-heading text describing the current page.
    """
    st.markdown(
        f'<div class="app-header"><h1>{APP_ICON} {APP_TITLE}</h1>'
        f"<p>{subtitle}</p></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Input form
# --------------------------------------------------------------------------- #


def _numeric_widget(name: str, schema: dict[str, Any], default: float) -> float:
    """Render one numeric input configured from measured data ranges.

    Args:
        name: Feature name.
        schema: The application schema.
        default: Starting value for the widget.

    Returns:
        The value entered by the user.
    """
    spec = schema["numeric_ranges"][name]
    helper = schema["field_help"].get(name, "")
    is_rate = name in {"BounceRates", "ExitRates", "SpecialDay"}

    if is_rate:
        # SpecialDay genuinely spans 0-1. Bounce and exit rates never exceed 0.2
        # in this dataset, so a 0-1 slider would waste 80% of its travel and
        # make realistic values impossible to set precisely.
        upper = 1.0 if name == "SpecialDay" else float(spec["max"])
        step = 0.05 if name == "SpecialDay" else 0.001
        return float(
            st.slider(
                name,
                min_value=0.0,
                max_value=upper,
                value=min(max(float(default), 0.0), upper),
                step=step,
                help=f"{helper} Observed range 0-{spec['max']:.3f}.",
                format="%.3f",
            )
        )

    help_text = f"{helper} Typical range 0-{spec['p99']:.0f}; observed maximum {spec['max']:.0f}."

    if spec["decimals"] == 0:
        # Page-view counts (Administrative, Informational, ProductRelated) are
        # whole numbers. Passing float bounds/step here makes Streamlit render
        # a forced "0.00" with decimal spinners on a field that counts pages -
        # using int end-to-end gives a clean whole-number widget instead.
        upper_int = int(round(spec["p99"]))
        default_int = min(max(int(round(default)), 0), upper_int)
        return float(
            st.number_input(
                name,
                min_value=0,
                max_value=int(round(spec["max"])),
                value=default_int,
                step=1,
                help=help_text,
            )
        )

    upper = float(spec["p99"])
    default = min(max(float(default), 0.0), upper)
    return float(
        st.number_input(
            name,
            min_value=0.0,
            max_value=float(spec["max"]),
            value=default,
            step=float(spec["step"]),
            format="%.2f",
            help=help_text,
        )
    )


def _categorical_widget(name: str, schema: dict[str, Any], default: Any) -> Any:
    """Render one categorical input restricted to levels seen in training.

    Args:
        name: Feature name.
        schema: The application schema.
        default: Starting value for the widget.

    Returns:
        The level selected by the user.
    """
    levels = schema["categorical_levels"][name]
    helper = schema["field_help"].get(name, "")

    if name == "Weekend":
        return bool(st.checkbox("Weekend session", value=bool(default), help=helper))

    try:
        index = levels.index(default)
    except (ValueError, AttributeError):
        index = 0
    return st.selectbox(name, options=levels, index=index, help=helper)


def render_input_form(schema: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Render the full session-input form.

    Args:
        schema: The application schema.

    Returns:
        A ``(payload, submitted)`` tuple.
    """
    base = st.session_state.get("preset_payload") or schema["presets"][
        "Returning window shopper"
    ]
    payload: dict[str, Any] = {}

    with st.form("session_form"):
        st.markdown("#### Page engagement")
        cols = st.columns(3)
        pairs = [
            ("Administrative", "Administrative_Duration"),
            ("Informational", "Informational_Duration"),
            ("ProductRelated", "ProductRelated_Duration"),
        ]
        for col, (count_field, duration_field) in zip(cols, pairs):
            with col:
                payload[count_field] = _numeric_widget(
                    count_field, schema, base.get(count_field, 0)
                )
                payload[duration_field] = _numeric_widget(
                    duration_field, schema, base.get(duration_field, 0)
                )

        st.markdown("#### Session quality signals")
        cols = st.columns(4)
        for col, name in zip(
            cols, ["BounceRates", "ExitRates", "PageValues", "SpecialDay"]
        ):
            with col:
                payload[name] = _numeric_widget(name, schema, base.get(name, 0))

        st.markdown("#### Visitor context")
        cols = st.columns(3)
        for col, name in zip(cols, ["Month", "VisitorType", "Weekend"]):
            with col:
                payload[name] = _categorical_widget(name, schema, base.get(name))

        with st.expander("Technical attributes (anonymised identifier codes)"):
            st.caption(
                "These are nominal codes, not magnitudes - browser 4 is not "
                "'twice' browser 2. The model one-hot encodes them accordingly."
            )
            cols = st.columns(4)
            for col, name in zip(
                cols, ["OperatingSystems", "Browser", "Region", "TrafficType"]
            ):
                with col:
                    payload[name] = _categorical_widget(name, schema, base.get(name))

        submitted = st.form_submit_button(
            "Score this session", type="primary", width='stretch'
        )

    return payload, submitted


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #


def _gauge_svg(probability: float, threshold: float, colour: str) -> str:
    """Build a semicircular gauge showing the probability against the threshold.

    Rendered as inline SVG rather than a charting library: it is a single fixed
    graphic, so a dependency and a full chart render would buy nothing.

    Args:
        probability: Predicted probability, 0-1.
        threshold: Decision threshold to mark on the arc, 0-1.
        colour: Stroke colour for the filled portion.

    Returns:
        An SVG fragment.
    """
    import math

    width, height, radius, stroke = 260, 148, 100, 16
    cx, cy = width / 2, height - 18

    def point(fraction: float, r: float = radius) -> tuple[float, float]:
        """Map 0-1 onto the semicircle, left (0) to right (1)."""
        angle = math.pi * (1.0 - min(max(fraction, 0.0), 1.0))
        return cx + r * math.cos(angle), cy - r * math.sin(angle)

    start_x, start_y = point(0.0)
    end_x, end_y = point(1.0)
    fill_x, fill_y = point(probability)
    # A semicircle is exactly 180 degrees, so the large-arc flag is always 0;
    # the sweep flag is 1 because the arc runs clockwise from left to right.
    large_arc = 0

    tick_inner_x, tick_inner_y = point(threshold, radius - stroke / 2 - 3)
    tick_outer_x, tick_outer_y = point(threshold, radius + stroke / 2 + 3)
    label_x, label_y = point(threshold, radius + stroke / 2 + 15)

    return f"""
<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px;display:block;margin:0 auto;"
     role="img" aria-label="Purchase probability {probability:.0%}, threshold {threshold:.0%}">
  <path d="M {start_x:.2f} {start_y:.2f} A {radius} {radius} 0 {large_arc} 1 {end_x:.2f} {end_y:.2f}"
        fill="none" stroke="#e2e8f0" stroke-width="{stroke}" stroke-linecap="round"/>
  <path d="M {start_x:.2f} {start_y:.2f} A {radius} {radius} 0 {large_arc} 1 {fill_x:.2f} {fill_y:.2f}"
        fill="none" stroke="{colour}" stroke-width="{stroke}" stroke-linecap="round"/>
  <line x1="{tick_inner_x:.2f}" y1="{tick_inner_y:.2f}" x2="{tick_outer_x:.2f}" y2="{tick_outer_y:.2f}"
        stroke="#0f172a" stroke-width="2.5"/>
  <text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-size="10"
        fill="#475569" font-weight="600">{threshold:.0%}</text>
  <text x="{cx}" y="{cy - 34}" text-anchor="middle" font-size="34"
        font-weight="700" fill="#0f172a">{100 * probability:.1f}%</text>
  <text x="{cx}" y="{cy - 14}" text-anchor="middle" font-size="10"
        fill="#64748b" letter-spacing="1.2">PURCHASE PROBABILITY</text>
  <text x="{point(0.0)[0]:.2f}" y="{cy + 14}" text-anchor="middle" font-size="9" fill="#94a3b8">0%</text>
  <text x="{point(1.0)[0]:.2f}" y="{cy + 14}" text-anchor="middle" font-size="9" fill="#94a3b8">100%</text>
</svg>
"""


def render_result(result: inference.PredictionResult, baseline_rate: float) -> None:
    """Render the prediction outcome card.

    Args:
        result: Prediction returned by :func:`src.inference.predict`.
        baseline_rate: Overall conversion rate, as a percentage.
    """
    probability_pct = 100 * result.probability
    on_margin = abs(result.probability - result.threshold) <= 0.15

    if result.predicted_class == 1:
        card_class, colour, verdict = "result-positive", "#16a34a", "Likely to purchase"
    elif on_margin:
        card_class, colour, verdict = "result-margin", "#f59e0b", "On the margin"
    else:
        card_class, colour, verdict = "result-negative", "#94a3b8", "Unlikely to purchase"

    lift = probability_pct / baseline_rate if baseline_rate else 0.0

    st.markdown(f'<div class="result-card {card_class}">', unsafe_allow_html=True)
    left, right = st.columns([1, 1.55])

    with left:
        st.markdown(
            _gauge_svg(result.probability, result.threshold, colour),
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="text-align:center;"><span class="footnote">'
            f"Decision threshold {result.threshold:.0%} &middot; "
            f"confidence band: {result.confidence_band}</span></div>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(f"### {verdict}")
        metrics = st.columns(3)
        metrics[0].metric("Classification", result.label)
        metrics[1].metric("Site baseline", f"{baseline_rate:.1f}%")
        metrics[2].metric(
            "Lift vs. baseline",
            f"{lift:.2f}x",
            delta=f"{probability_pct - baseline_rate:+.1f} pp",
        )

    st.markdown(
        f'<div class="recommendation"><strong>Recommended action:</strong> '
        f"{result.recommendation}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_explanation(payload: dict[str, Any], model: Any, metadata: dict[str, Any]) -> None:
    """Render the per-session SHAP attribution for the scored session.

    Args:
        payload: The raw inputs that were scored.
        model: The loaded pipeline.
        metadata: Training metadata, used for the global-importance fallback.
    """
    st.markdown("##### Why this session scored the way it did")

    with st.spinner("Attributing the prediction..."):
        explanation = inference.explain_prediction(payload, model)

    if not explanation.available:
        st.info(
            "Per-session attribution is unavailable in this environment, so the "
            "global feature importances on the *Model insights* page apply instead."
        )
        return

    frame = pd.DataFrame(
        {
            "feature": [c.feature for c in explanation.contributions],
            "contribution": [c.value for c in explanation.contributions],
            "this session": [c.display_value for c in explanation.contributions],
        }
    )
    if frame.empty:
        st.info("No individual feature moved this prediction materially.")
        return

    left, right = st.columns([1.3, 1])

    with left:
        # Ordered smallest-to-largest so the strongest driver sits at the top of
        # a horizontal bar chart, which draws its first category at the bottom.
        chart = frame.set_index("feature")[["contribution"]].iloc[::-1]
        st.bar_chart(chart, horizontal=True, color="#2563eb")
        st.caption(
            "Bars are SHAP values in log-odds space. Positive pushes towards "
            "purchase, negative away from it."
        )

    with right:
        st.dataframe(
            frame.assign(contribution=frame["contribution"].round(3)),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        f"An average session starts at **{explanation.base_probability:.1%}**; these "
        f"contributions carry it to **{explanation.probability:.1%}**. SHAP values are "
        "additive in log-odds, so they do not sum to the probability difference directly."
    )


def render_what_if(
    payload: dict[str, Any],
    model: Any,
    schema: dict[str, Any],
    threshold: float,
) -> None:
    """Render the single-feature what-if sensitivity chart.

    Args:
        payload: The raw inputs that were scored.
        model: The loaded pipeline.
        schema: The application schema.
        threshold: Decision threshold to mark on the chart.
    """
    st.markdown("##### What if one field were different?")

    sweepable = [
        name for name in config.TRUE_NUMERIC_FEATURES if name in schema["numeric_ranges"]
    ]
    default_index = sweepable.index("PageValues") if "PageValues" in sweepable else 0
    feature = st.selectbox(
        "Vary this field, holding every other input fixed",
        options=sweepable,
        index=default_index,
        help=schema["field_help"].get("PageValues", ""),
    )

    try:
        curve = inference.sensitivity_curve(payload, feature, model, schema=schema)
    except Exception as exc:
        st.warning(f"Could not compute sensitivity for {feature}: {exc}")
        return

    current = float(payload.get(feature, 0.0))
    chart = curve.rename(
        columns={"probability": "Predicted probability"}
    ).set_index("value")
    chart["Decision threshold"] = threshold
    st.line_chart(chart, color=["#2563eb", "#94a3b8"])

    peak = curve.loc[curve["probability"].idxmax()]
    low = curve.loc[curve["probability"].idxmin()]
    st.caption(
        f"This session's actual **{feature}** is **{current:,.4g}**. Sweeping it from "
        f"{curve['value'].min():,.4g} to {curve['value'].max():,.4g} moves the predicted "
        f"probability between **{low['probability']:.1%}** and **{peak['probability']:.1%}**. "
        "A flat line means this field is not what is driving the decision for this session."
    )


def append_history(payload: dict[str, Any], result: inference.PredictionResult) -> None:
    """Record a scored session in the in-session history.

    Args:
        payload: The raw inputs that were scored.
        result: The prediction produced from them.
    """
    entry = {
        "scored_at": datetime.now().strftime("%H:%M:%S"),
        "probability": round(result.probability, 4),
        "prediction": result.label,
        "threshold": result.threshold,
        **payload,
    }
    st.session_state[HISTORY_KEY].insert(0, entry)
    del st.session_state[HISTORY_KEY][MAX_HISTORY_ROWS:]


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def page_scorer(
    model: Any,
    metadata: dict[str, Any],
    schema: dict[str, Any],
    threshold: float,
) -> None:
    """Render the single-session scoring page."""
    render_header(
        "Score a live browsing session from its Google Analytics metrics and "
        "decide whether a real-time incentive is worth spending."
    )

    st.markdown("##### Start from a real session profile")
    preset_names = list(schema["presets"].keys())
    cols = st.columns(len(preset_names))
    for col, name in zip(cols, preset_names):
        if col.button(name, width='stretch', key=f"preset_{name}"):
            st.session_state["preset_payload"] = schema["presets"][name]
            st.rerun()
    st.caption(
        "Each preset is the median of an actual behavioural segment in the "
        f"{schema['n_training_sessions']:,} training sessions - not invented values."
    )
    st.divider()

    payload, submitted = render_input_form(schema)

    if submitted:
        problems = inference.validate_payload(payload)
        if problems:
            st.error("Please correct the following before scoring:")
            for problem in problems:
                st.markdown(f"- {problem}")
        else:
            with st.spinner("Scoring session..."):
                try:
                    result = inference.predict(payload, model, metadata, threshold)
                except Exception as exc:
                    st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
                    return
            append_history(payload, result)
            render_result(result, schema["baseline_conversion_rate_pct"])

            st.divider()
            explain_tab, what_if_tab = st.tabs(
                ["Why this score", "What-if analysis"]
            )
            with explain_tab:
                render_explanation(payload, model, metadata)
            with what_if_tab:
                render_what_if(payload, model, schema, threshold)

    history = st.session_state[HISTORY_KEY]
    if history:
        st.divider()
        st.markdown(f"#### Session history ({len(history)} scored)")
        frame = pd.DataFrame(history)
        st.dataframe(
            frame[["scored_at", "probability", "prediction", "PageValues", "ProductRelated", "ExitRates"]],
            width='stretch',
            hide_index=True,
        )
        left, right = st.columns([1, 4])
        left.download_button(
            "Download results (CSV)",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name=f"purchase_intent_predictions_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            width='stretch',
        )
        if right.button("Clear history"):
            st.session_state[HISTORY_KEY] = []
            st.rerun()


def page_batch(
    model: Any,
    metadata: dict[str, Any],
    schema: dict[str, Any],
    threshold: float,
) -> None:
    """Render the batch CSV scoring page."""
    render_header("Score an exported analytics file and rank sessions by purchase intent.")

    required = schema["feature_order"]
    st.markdown(f"Upload a CSV containing these {len(required)} columns:")
    st.code(", ".join(required), language="text")

    template = pd.DataFrame([schema["presets"]["High-intent buyer"]])[required]
    st.download_button(
        "Download a template CSV",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="session_template.csv",
        mime="text/csv",
    )
    st.divider()

    uploaded = st.file_uploader("Analytics export (CSV)", type=["csv"])
    if uploaded is None:
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read that file as CSV: {exc}")
        return

    st.caption(f"Loaded {len(frame):,} rows x {frame.shape[1]} columns.")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        st.error(f"The file is missing {len(missing)} required column(s): {', '.join(missing)}")
        return

    with st.spinner(f"Scoring {len(frame):,} sessions..."):
        try:
            scored = inference.predict_batch(frame, model, metadata, threshold)
        except Exception as exc:
            st.error(f"Batch scoring failed: {type(exc).__name__}: {exc}")
            return

    flagged = int((scored["purchase_probability"] >= threshold).sum())

    cols = st.columns(4)
    cols[0].metric("Sessions scored", f"{len(scored):,}")
    cols[1].metric("Flagged as likely", f"{flagged:,}")
    cols[2].metric("Flag rate", f"{100 * flagged / max(len(scored), 1):.1f}%")
    cols[3].metric("Mean probability", f"{scored['purchase_probability'].mean():.3f}")

    st.markdown("##### Highest-intent sessions")
    st.dataframe(
        scored.nlargest(min(25, len(scored)), "purchase_probability")[
            ["purchase_probability", "prediction", "PageValues", "ProductRelated", "ExitRates", "VisitorType"]
        ],
        width='stretch',
        hide_index=True,
    )
    st.download_button(
        "Download all scored sessions (CSV)",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name=f"scored_sessions_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        type="primary",
    )


def page_insights(metadata: dict[str, Any]) -> None:
    """Render the model performance and interpretation page."""
    render_header("How the deployed model performs, and what it learned.")

    if not metadata:
        st.warning("No training metadata found. Run `python -m src.train` to generate it.")
        return

    metrics = metadata.get("metrics_at_operating_threshold", {})
    cols = st.columns(5)
    cols[0].metric("Model", metadata.get("model_name", "-"))
    cols[1].metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
    cols[2].metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
    cols[3].metric("F1", f"{metrics.get('f1', 0):.4f}")
    cols[4].metric("Recall", f"{metrics.get('recall', 0):.4f}")

    st.caption(
        f"Measured on {metadata.get('dataset', {}).get('test_rows', 0):,} held-out "
        f"sessions at a decision threshold of {metadata.get('operating_threshold', 0.5):.2f}. "
        f"Trained {metadata.get('trained_at_utc', 'unknown')}."
    )
    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("##### Why these metrics")
        st.markdown(
            f"Only **{metadata.get('dataset', {}).get('positive_rate_pct', 0):.2f}%** of "
            "sessions convert. A model predicting *no purchase* every time would score "
            "**84.5% accuracy** and be commercially worthless, so accuracy is reported "
            "for completeness only. **PR-AUC** drives model selection because its "
            "baseline moves with class prevalence, and **recall** is tracked because a "
            "missed high-intent session is a lost order."
        )
        cells = metadata.get("confusion_at_operating_threshold", {})
        if cells:
            st.markdown("##### Confusion matrix at the operating threshold")
            st.dataframe(
                pd.DataFrame(
                    [
                        ["Actual: no purchase", cells.get("true_negative", 0), cells.get("false_positive", 0)],
                        ["Actual: purchase", cells.get("false_negative", 0), cells.get("true_positive", 0)],
                    ],
                    columns=["", "Predicted: no purchase", "Predicted: purchase"],
                ),
                width='stretch',
                hide_index=True,
            )

    with right:
        st.markdown("##### Most influential features")
        drivers = metadata.get("top_features", {})
        if drivers:
            frame = (
                pd.DataFrame({"feature": list(drivers.keys()), "importance": list(drivers.values())})
                .sort_values("importance", ascending=False)
                .head(12)
            )
            st.bar_chart(frame.set_index("feature"), horizontal=True, color="#2563eb")
            st.caption(
                "Global importance across the whole training set, not an "
                "attribution for any single prediction."
            )

    comparison = metadata.get("all_models_test", {})
    if comparison:
        st.divider()
        st.markdown("##### Model comparison on the held-out test set")
        frame = pd.DataFrame(comparison).T[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "latency_median_ms"]
        ]
        st.dataframe(frame.style.format("{:.4f}").highlight_max(axis=0, color="#dbeafe"),
                     width='stretch')

    calibration = metadata.get("calibration", {})
    if calibration and calibration.get("curve"):
        st.divider()
        st.markdown("##### Calibration: are the probabilities honest?")
        st.markdown(
            "ROC-AUC and PR-AUC measure whether the model *ranks* buyers above "
            "non-buyers. Calibration asks a different question: when it says 30%, "
            "do 30% of those sessions actually convert? A perfectly calibrated "
            "model sits on the diagonal."
        )

        curve = pd.DataFrame(calibration["curve"])
        left, right = st.columns([1.25, 1])

        with left:
            chart = pd.DataFrame(
                {
                    "Observed conversion rate": curve["observed_rate"].values,
                    "Perfect calibration": curve["mean_predicted"].values,
                },
                index=curve["mean_predicted"].values,
            )
            chart.index.name = "Mean predicted probability"
            st.line_chart(chart, color=["#2563eb", "#cbd5e1"])

        with right:
            st.dataframe(
                curve[["mean_predicted", "observed_rate", "count"]].rename(
                    columns={
                        "mean_predicted": "predicted",
                        "observed_rate": "observed",
                        "count": "sessions",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

        cols = st.columns(3)
        cols[0].metric("Brier score", f"{calibration.get('brier_score', 0):.4f}")
        cols[1].metric(
            "Mean gap",
            f"{calibration.get('mean_gap', 0):+.3f}",
            help="Count-weighted mean of predicted minus observed.",
        )
        cols[2].metric("Direction", calibration.get("direction", "-").title())

        st.info(calibration.get("interpretation", ""))

    leakage = metadata.get("leakage_experiment", {})
    if leakage:
        st.markdown("##### Leakage experiment: the PageValues question")
        st.markdown(
            "Google Analytics computes Page Value partly from transaction revenue, "
            "so it is partially derived from the outcome being predicted. The table "
            "below refits the identical model without it to quantify the dependence."
        )
        st.dataframe(pd.DataFrame(leakage).T, width='stretch')


def page_about(metadata: dict[str, Any], schema: dict[str, Any]) -> None:
    """Render the project documentation page."""
    render_header("Dataset, methodology and deployment pipeline.")

    dataset = metadata.get("dataset", {})
    st.markdown(
        f"""
##### The business problem
Roughly **{schema['baseline_conversion_rate_pct']:.1f}%** of e-commerce sessions end in a
purchase. The other ~85% cost bandwidth and return nothing. Blanket-discounting every
visitor destroys margin; discounting nobody leaves winnable sessions on the table. This
model scores a live session so an incentive can be fired at the sessions that are
*winnable but not yet won*.

##### Dataset
- **Name:** {config.DATASET_NAME}
- **Source:** {config.DATASET_HOME}
- **Licence:** {config.DATASET_LICENCE}
- **Rows used:** {dataset.get('rows_used', 0):,} after removing {dataset.get('duplicates_removed', 0)} exact duplicates
- **Split:** {dataset.get('train_rows', 0):,} train / {dataset.get('test_rows', 0):,} test, stratified
- **Citation:** {config.DATASET_CITATION}

##### How leakage is prevented
Every stateful transformation - imputation statistics, scaling parameters and one-hot
vocabularies - lives inside the scikit-learn `Pipeline`. During cross-validation each
transformer is refitted on that fold's training partition alone, so no statistic derived
from validation or test data can reach the model. Fitting a scaler before splitting, the
most common leakage mistake, is structurally impossible in this design.

##### A note on the encoded identifiers
`OperatingSystems`, `Browser`, `Region` and `TrafficType` arrive as integers, and pandas
infers them as numeric. They are in fact nominal codes. Treating them as magnitudes would
tell the model that browser 13 is "greater than" browser 2. They are routed to the
categorical branch and one-hot encoded.

##### Deployment
Model artifact: `{config.MODEL_FILE.name}` ({metadata.get('model_size_mb', 0)} MB),
a single serialised pipeline containing preprocessing and the classifier together.
Trained with scikit-learn {metadata.get('environment', {}).get('scikit_learn', '-')}
on Python {metadata.get('environment', {}).get('python', '-')}.

Every push to `main` runs the test suite in GitHub Actions and, only if it
passes, syncs this application to the Space automatically. The Hugging Face
token lives in a GitHub Actions secret and appears in no file in the repository.

##### Links
- **Source code:** {config.GITHUB_REPO_URL}
- **Live Space:** {config.HF_SPACE_URL}
- **Dataset:** {config.DATASET_HOME}
"""
    )

    health = inference.model_health()
    if health["ok"]:
        st.success(
            f"Model health check passed - {health['model_name']} loaded and scored a "
            f"probe session at probability {health['smoke_test_probability']:.4f}."
        )
    else:
        st.error(f"Model health check failed: {health['error']}")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Application entry point: sidebar navigation and page dispatch."""
    init_state()

    try:
        model = get_model()
        metadata = get_metadata()
        schema = get_schema()
    except FileNotFoundError as exc:
        st.error(f"{APP_ICON} The application could not start.\n\n```\n{exc}\n```")
        st.stop()
        return

    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_TITLE}")
        st.caption("E-commerce session intent scoring")
        page = st.radio(
            "Navigation",
            ["Score a session", "Batch scoring", "Model insights", "About"],
            label_visibility="collapsed",
        )
        st.divider()

        policies = threshold_policies(metadata)
        default_threshold = float(metadata.get("operating_threshold", 0.50))
        if len(policies) > 1:
            st.markdown("**Decision policy**")
            policy_name = st.radio(
                "Decision policy",
                list(policies.keys()),
                label_visibility="collapsed",
                help=(
                    "Switches the cut-off applied to the model's probability. "
                    "The model itself is unchanged - only the decision rule moves."
                ),
            )
            threshold = float(policies[policy_name]["threshold"])
            st.caption(f"Threshold **{threshold:.2f}** - {policies[policy_name]['why']}")
            st.divider()
        else:
            threshold = default_threshold

        st.markdown("**Deployed model**")
        st.markdown(f"`{metadata.get('model_name', 'unknown')}`")
        metrics = metadata.get("metrics_at_operating_threshold", {})
        st.markdown(
            f"- PR-AUC **{metrics.get('pr_auc', 0):.4f}**\n"
            f"- ROC-AUC **{metrics.get('roc_auc', 0):.4f}**\n"
            f"- Recall **{metrics.get('recall', 0):.4f}**\n"
            f"- Threshold **{threshold:.2f}**"
        )
        if threshold != default_threshold:
            st.caption(
                f"Reported metrics above were measured at the default "
                f"{default_threshold:.2f} threshold."
            )
        st.divider()
        st.caption(
            "Built for the SP Jain MAIB final group project. Deployed to "
            "Hugging Face Spaces via GitHub Actions."
        )

    if page == "Score a session":
        page_scorer(model, metadata, schema, threshold)
    elif page == "Batch scoring":
        page_batch(model, metadata, schema, threshold)
    elif page == "Model insights":
        page_insights(metadata)
    else:
        page_about(metadata, schema)


if __name__ == "__main__":
    main()
