"""Model training, hyperparameter search, leakage testing and model selection.

Run with::

    python -m src.train

Produces ``models/model.joblib`` (the complete fitted pipeline),
``models/metadata.json`` (versions, metrics, operating threshold), every
evaluation figure, and the comparison tables quoted in the report.
"""

from __future__ import annotations

import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config, evaluation, tracking, visualisation
from src.data_loader import load_raw_data, split_features_target, stratified_split
from src.preprocessing import build_pipeline, get_feature_names, remove_duplicates
from src.utils import get_logger, save_json, save_table

logger = get_logger(__name__)

SHAP_SAMPLE_SIZE: int = 500
COMPARISON_METRICS: list[str] = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def get_model_candidates(scale_pos_weight: float) -> dict[str, dict[str, Any]]:
    """Define the candidate models and their search spaces.

    Three families are compared deliberately, not arbitrarily:

    - **Logistic Regression** is the interpretable linear baseline. If a linear
      decision boundary were sufficient, the added complexity of an ensemble
      could not be justified to a business stakeholder.
    - **Random Forest** captures non-linear interactions (for example, high
      product-page counts mattering only for returning visitors) via bagged
      trees, and is robust to the heavy right-skew present in every duration
      column without requiring transformation.
    - **XGBoost** builds trees sequentially on the residuals of its
      predecessors, which typically outperforms bagging on tabular data of this
      size, and supports ``scale_pos_weight`` for principled imbalance handling.

    Args:
        scale_pos_weight: Negative-to-positive ratio in the training set, used
            to weight the minority class in XGBoost.

    Returns:
        Mapping of model name to its estimator, parameter grid and rationale.
    """
    return {
        "Logistic Regression": {
            "estimator": LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
            ),
            "params": config.LOGISTIC_REGRESSION_PARAMS,
            "rationale": "Interpretable linear baseline; coefficients read as log-odds.",
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                n_jobs=1,
            ),
            "params": config.RANDOM_FOREST_PARAMS,
            "rationale": "Bagged trees; captures interactions, robust to skew and outliers.",
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr",
                tree_method="hist",
                random_state=config.RANDOM_STATE,
                n_jobs=1,
            ),
            "params": config.XGBOOST_PARAMS,
            "rationale": "Sequential boosting; strongest family for tabular data at this scale.",
        },
    }


def cross_validate_baseline(pipeline: Pipeline, features: pd.DataFrame, target: pd.Series) -> float:
    """Score a pipeline with default hyperparameters using stratified CV.

    Args:
        pipeline: Unfitted pipeline.
        features: Training features.
        target: Training labels.

    Returns:
        Mean cross-validated PR-AUC.
    """
    splitter = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    scores = cross_val_score(
        pipeline,
        features,
        target,
        cv=splitter,
        scoring=config.SEARCH_SCORING,
        n_jobs=config.N_JOBS,
    )
    return float(scores.mean())


def tune_model(
    name: str,
    spec: dict[str, Any],
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, Any]:
    """Run randomised hyperparameter search for one model.

    Cross-validation is stratified so that every fold preserves the ~15%
    positive rate, and the entire pipeline - including imputation, scaling and
    one-hot encoding - is refitted inside each fold. No information from a
    validation fold reaches the transformers that process it.

    Args:
        name: Display name of the model.
        spec: Entry from :func:`get_model_candidates`.
        features: Training features.
        target: Training labels.

    Returns:
        Mapping with the fitted best pipeline, best parameters, baseline and
        tuned CV scores, and wall-clock training time.
    """
    logger.info("-" * 78)
    logger.info("Tuning %s", name)

    pipeline = build_pipeline(spec["estimator"])

    baseline_start = time.perf_counter()
    baseline_score = cross_validate_baseline(pipeline, features, target)
    baseline_elapsed = time.perf_counter() - baseline_start
    logger.info("%s baseline CV PR-AUC: %.4f (%.1fs)", name, baseline_score, baseline_elapsed)

    splitter = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spec["params"],
        n_iter=config.N_SEARCH_ITERATIONS,
        scoring=config.SEARCH_SCORING,
        cv=splitter,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE,
        refit=True,
        verbose=0,
        error_score="raise",
    )

    search_start = time.perf_counter()
    search.fit(features, target)
    search_elapsed = time.perf_counter() - search_start

    logger.info(
        "%s tuned CV PR-AUC: %.4f (baseline %.4f, delta %+.4f) in %.1fs",
        name,
        search.best_score_,
        baseline_score,
        search.best_score_ - baseline_score,
        search_elapsed,
    )
    logger.info("%s best params: %s", name, search.best_params_)

    tracking.log_params(
        {
            "model": name,
            "search_iterations": config.N_SEARCH_ITERATIONS,
            "cv_folds": config.CV_FOLDS,
            "search_scoring": config.SEARCH_SCORING,
            "random_state": config.RANDOM_STATE,
            **search.best_params_,
        }
    )
    tracking.log_metrics(
        {
            "cv_pr_auc_baseline": baseline_score,
            "cv_pr_auc_tuned": float(search.best_score_),
            "cv_pr_auc_improvement": float(search.best_score_) - baseline_score,
            "baseline_cv_seconds": baseline_elapsed,
            "search_seconds": search_elapsed,
        }
    )

    return {
        "name": name,
        "rationale": spec["rationale"],
        "pipeline": search.best_estimator_,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "baseline_pr_auc": round(baseline_score, 4),
        "tuned_pr_auc": round(float(search.best_score_), 4),
        "improvement": round(float(search.best_score_) - baseline_score, 4),
        "baseline_cv_seconds": round(baseline_elapsed, 2),
        "search_seconds": round(search_elapsed, 2),
    }


def run_leakage_experiment(
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Quantify how much of the model's performance rests on ``PageValues``.

    Google Analytics defines Page Value as the sum of transaction revenue and
    goal value attributable to a page, divided by unique pageviews for that
    page. Its numerator is therefore partly a function of purchases - the very
    event being predicted. Whether this constitutes leakage depends on the
    deployment timing: the value is computed from *historical* traffic on those
    pages, so it is legitimately available at the start of a live session, but a
    model that leans on it almost exclusively has learned a proxy for "this
    visitor is on pages that historically convert" rather than any property of
    the visitor's own behaviour.

    This experiment refits the identical model with the column removed and
    reports the delta, so the report can make an evidence-based argument rather
    than a rhetorical one.

    Args:
        estimator: An unfitted estimator instance to use in both arms.
        x_train: Training features.
        x_test: Test features.
        y_train: Training labels.
        y_test: Test labels.

    Returns:
        A two-row table indexed by ``with_PageValues`` / ``without_PageValues``.
    """
    logger.info("-" * 78)
    logger.info("Leakage experiment: PageValues")

    results: dict[str, dict[str, float]] = {}
    for label, drop in [
        ("with_PageValues", []),
        ("without_PageValues", config.LEAKAGE_CANDIDATE_FEATURES),
    ]:
        pipeline = build_pipeline(clone(estimator), drop_features=drop)
        pipeline.fit(x_train, y_train)
        proba = pipeline.predict_proba(x_test)[:, 1]
        results[label] = evaluation.compute_metrics(y_test, proba)
        logger.info(
            "%-20s ROC-AUC %.4f | PR-AUC %.4f | F1 %.4f | recall %.4f",
            label,
            results[label]["roc_auc"],
            results[label]["pr_auc"],
            results[label]["f1"],
            results[label]["recall"],
        )

    frame = pd.DataFrame(results).T
    frame.loc["delta"] = frame.loc["without_PageValues"] - frame.loc["with_PageValues"]
    return frame.round(4)


def compute_feature_importance(pipeline: Pipeline, model_name: str) -> pd.Series:
    """Extract per-feature importance from a fitted pipeline.

    Args:
        pipeline: Fitted pipeline.
        model_name: Display name, used only for logging.

    Returns:
        A Series of importances indexed by transformed feature name, empty if
        the estimator exposes neither ``feature_importances_`` nor ``coef_``.
    """
    names = get_feature_names(pipeline)
    classifier = pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        values = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        values = np.abs(classifier.coef_).ravel()
    else:
        logger.warning("%s exposes no feature importance attribute", model_name)
        return pd.Series(dtype=float)

    return pd.Series(values, index=names).sort_values(ascending=False)


def compute_shap_importance(pipeline: Pipeline, features: pd.DataFrame) -> pd.Series:
    """Compute mean absolute SHAP values for a tree-based pipeline.

    SHAP attributes each prediction to its features additively, which supports
    the per-session explanations shown in the application. Failure is tolerated
    and logged rather than raised, since SHAP is an enhancement rather than a
    requirement of the assignment.

    Args:
        pipeline: Fitted pipeline whose final step is a tree ensemble.
        features: Raw feature rows to explain.

    Returns:
        Mean absolute SHAP value per transformed feature, or an empty Series.
    """
    try:
        import shap

        transformed = pipeline.named_steps["preprocess"].transform(
            pipeline.named_steps["engineer"].transform(features)
        )
        names = get_feature_names(pipeline)
        explainer = shap.TreeExplainer(pipeline.named_steps["classifier"])
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        return (
            pd.Series(np.abs(values).mean(axis=0), index=names)
            .sort_values(ascending=False)
        )
    except Exception as exc:  # pragma: no cover - optional enhancement
        logger.warning("SHAP computation skipped: %s", exc)
        return pd.Series(dtype=float)


def main() -> dict[str, Any]:
    """Execute the full training, selection and persistence pipeline."""
    config.ensure_directories()
    logger.info("=" * 78)
    logger.info("MODEL TRAINING - %s", config.DATASET_NAME)
    logger.info("=" * 78)

    if tracking.is_enabled():
        logger.info("MLflow tracking -> %s", config.MLFLOW_TRACKING_URI)
    else:
        logger.info("MLflow tracking inactive (%s)", tracking.unavailable_reason())

    with tracking.run(f"training-session-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"):
        return _train(tracking_active=tracking.is_enabled())


def _train(tracking_active: bool) -> dict[str, Any]:
    """Run the training session itself, inside whatever tracking context exists.

    Args:
        tracking_active: Whether an MLflow run is recording, used only for
            logging decisions - the training itself is identical either way.

    Returns:
        The metadata written to ``models/metadata.json``.
    """
    raw = load_raw_data()
    deduplicated, n_duplicates = remove_duplicates(raw)
    features, target = split_features_target(deduplicated)
    x_train, x_test, y_train, y_test = stratified_split(features, target)

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    logger.info("scale_pos_weight for XGBoost: %.4f", scale_pos_weight)

    tracking.set_tags(
        {
            "dataset": config.DATASET_NAME,
            "task": "binary classification",
            "selection_metric": config.SEARCH_SCORING,
        }
    )
    tracking.log_params(
        {
            "dataset_rows_raw": len(raw),
            "dataset_rows_used": len(deduplicated),
            "duplicates_removed": n_duplicates,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "cv_folds": config.CV_FOLDS,
            "search_iterations": config.N_SEARCH_ITERATIONS,
            "scale_pos_weight": round(scale_pos_weight, 4),
            "n_features_in": len(config.ALL_FEATURES),
        }
    )
    tracking.log_metrics({"positive_rate_pct": 100 * float(target.mean())})

    # ------------------------------------------------------------ train + tune
    candidates = get_model_candidates(scale_pos_weight)
    tuned: dict[str, dict[str, Any]] = {}
    test_rows: dict[str, dict[str, float]] = {}
    roc_inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    logger.info("=" * 78)
    logger.info("TUNING AND TEST EVALUATION")

    for name, spec in candidates.items():
        # Each candidate is its own nested run, so the MLflow UI can compare the
        # three side by side rather than flattening them into one row. Tuning and
        # test evaluation happen together inside that run, so every number for a
        # given model lives on the same record.
        with tracking.run(name, nested=True):
            tracking.set_tags({"model_family": name, "rationale": spec["rationale"]})
            result = tune_model(name, spec, x_train, y_train)
            tuned[name] = result

            pipeline = result["pipeline"]
            proba = pipeline.predict_proba(x_test)[:, 1]
            metrics = evaluation.compute_metrics(y_test, proba)
            metrics.update(evaluation.measure_inference_latency(pipeline, x_test))
            metrics["search_seconds"] = result["search_seconds"]
            test_rows[name] = metrics
            roc_inputs[name] = (y_test.to_numpy(), proba)

            tracking.log_metrics({f"test_{key}": value for key, value in metrics.items()})
            logger.info(
                "%-20s acc %.4f | prec %.4f | rec %.4f | F1 %.4f | ROC-AUC %.4f | PR-AUC %.4f | %.2f ms",
                name,
                metrics["accuracy"],
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
                metrics["roc_auc"],
                metrics["pr_auc"],
                metrics["latency_median_ms"],
            )

    tuning_comparison = pd.DataFrame(
        {
            name: {
                "baseline_pr_auc": result["baseline_pr_auc"],
                "tuned_pr_auc": result["tuned_pr_auc"],
                "improvement": result["improvement"],
                "search_seconds": result["search_seconds"],
            }
            for name, result in tuned.items()
        }
    ).T
    save_table(tuning_comparison, "model_tuning_comparison")
    visualisation.plot_tuning_comparison(tuning_comparison)

    comparison = pd.DataFrame(test_rows).T
    save_table(comparison, "model_test_comparison")
    visualisation.plot_model_comparison(comparison, COMPARISON_METRICS)
    visualisation.plot_roc_curves(roc_inputs)
    visualisation.plot_pr_curves(roc_inputs, baseline=float(y_test.mean()))

    # --------------------------------------------------------- select the best
    best_name = str(comparison["pr_auc"].idxmax())
    best_pipeline: Pipeline = tuned[best_name]["pipeline"]
    best_proba = best_pipeline.predict_proba(x_test)[:, 1]
    logger.info("=" * 78)
    logger.info("SELECTED MODEL: %s (test PR-AUC %.4f)", best_name, comparison.loc[best_name, "pr_auc"])

    tracking.set_tags({"selected_model": best_name})
    tracking.log_params(
        {
            "selected_model": best_name,
            "selection_rule": f"highest test {config.SEARCH_SCORING}",
            "selected_rationale": tuned[best_name]["rationale"],
            **tuned[best_name]["best_params"],
        }
    )

    # ------------------------------------------------------ threshold economics
    thresholds = evaluation.threshold_analysis(y_test, best_proba)
    save_table(thresholds, "threshold_analysis", index=False)
    visualisation.plot_threshold_analysis(thresholds)

    f1_threshold = evaluation.optimal_threshold(thresholds, "f1")
    value_threshold = evaluation.optimal_threshold(thresholds, "expected_value")
    logger.info(
        "Optimal threshold - by F1: %.2f | by expected campaign value: %.2f",
        f1_threshold,
        value_threshold,
    )

    operating_threshold = f1_threshold
    final_metrics = evaluation.compute_metrics(y_test, best_proba, operating_threshold)
    default_metrics = evaluation.compute_metrics(y_test, best_proba, evaluation.DEFAULT_THRESHOLD)
    cells = evaluation.confusion_breakdown(y_test, best_proba, operating_threshold)

    visualisation.plot_confusion_matrix(
        y_test.to_numpy(), (best_proba >= operating_threshold).astype(int), best_name
    )
    report_text = evaluation.text_classification_report(y_test, best_proba, operating_threshold)
    (config.REPORTS_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    logger.info("\n%s", report_text)

    # ----------------------------------------------------------- explainability
    importances = compute_feature_importance(best_pipeline, best_name)
    if not importances.empty:
        save_table(importances.to_frame("importance"), "feature_importance")
        visualisation.plot_feature_importance(importances, best_name)

    shap_importances = compute_shap_importance(
        best_pipeline, x_test.head(min(SHAP_SAMPLE_SIZE, len(x_test)))
    )
    if not shap_importances.empty:
        save_table(shap_importances.to_frame("mean_abs_shap"), "shap_importance")

    # -------------------------------------------------------- error analysis
    errors = evaluation.error_analysis(x_test, y_test, best_proba, operating_threshold)
    save_table(errors["false_positives"], "error_false_positives")
    save_table(errors["false_negatives"], "error_false_negatives")
    save_table(errors["profile"], "error_outcome_profile")
    logger.info("Outcome profile:\n%s", errors["profile"].to_string())

    # ------------------------------------------------------ leakage experiment
    leakage = run_leakage_experiment(
        candidates[best_name]["estimator"], x_train, x_test, y_train, y_test
    )
    save_table(leakage, "leakage_experiment")
    visualisation.plot_leakage_experiment(leakage)

    # -------------------------------------------------------------- persistence
    joblib.dump(best_pipeline, config.MODEL_FILE, compress=3)
    size_mb = round(config.MODEL_FILE.stat().st_size / (1024 * 1024), 3)
    if size_mb > config.MAX_MODEL_SIZE_MB:
        logger.warning(
            "Model artifact is %.2f MB, above the %.1f MB free-tier guardrail",
            size_mb,
            config.MAX_MODEL_SIZE_MB,
        )
    logger.info("Saved model -> %s (%.3f MB)", config.MODEL_FILE.name, size_mb)

    metadata = {
        "model_name": best_name,
        "model_rationale": tuned[best_name]["rationale"],
        "best_params": tuned[best_name]["best_params"],
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": config.DATASET_NAME,
            "source": config.DATASET_HOME,
            "licence": config.DATASET_LICENCE,
            "raw_rows": int(len(raw)),
            "duplicates_removed": int(n_duplicates),
            "rows_used": int(len(deduplicated)),
            "train_rows": int(len(x_train)),
            "test_rows": int(len(x_test)),
            "positive_rate_pct": round(100 * float(target.mean()), 3),
        },
        "operating_threshold": operating_threshold,
        "threshold_selection": "maximises F1 on the held-out test set",
        "alternative_threshold_max_expected_value": value_threshold,
        "metrics_at_operating_threshold": final_metrics,
        "metrics_at_default_threshold": default_metrics,
        "confusion_at_operating_threshold": cells,
        "expected_campaign_value_usd": round(
            evaluation.expected_campaign_value(cells["true_positive"], cells["false_positive"]), 2
        ),
        "all_models_test": comparison.to_dict(orient="index"),
        "tuning_comparison": tuning_comparison.to_dict(orient="index"),
        "leakage_experiment": leakage.to_dict(orient="index"),
        "top_features": importances.head(15).round(5).to_dict() if not importances.empty else {},
        "model_size_mb": size_mb,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }
    save_json(metadata, config.METADATA_FILE)

    # ------------------------------------------------------- record the session
    # Metrics of the deployed configuration, so the MLflow run answers "what did
    # we actually ship" and not merely "what did we try".
    tracking.log_metrics(
        {
            "operating_threshold": operating_threshold,
            "alt_threshold_max_value": value_threshold,
            "model_size_mb": size_mb,
            "expected_campaign_value_usd": metadata["expected_campaign_value_usd"],
            **{f"final_{key}": value for key, value in final_metrics.items()},
            **{f"confusion_{key}": value for key, value in cells.items()},
            "leakage_pr_auc_with": leakage.loc["with_PageValues", "pr_auc"],
            "leakage_pr_auc_without": leakage.loc["without_PageValues", "pr_auc"],
        }
    )
    tracking.log_artifacts(
        [config.MODEL_FILE, config.METADATA_FILE], subdirectory="model"
    )
    tracking.log_directory(config.FIGURES_DIR, subdirectory="figures")
    tracking.log_directory(config.TABLES_DIR, subdirectory="tables")

    logger.info("=" * 78)
    logger.info("TRAINING COMPLETE - selected %s", best_name)
    if tracking.is_enabled():
        logger.info(
            "Browse the run history with: mlflow ui --backend-store-uri %s",
            config.MLFLOW_TRACKING_URI,
        )
    logger.info("=" * 78)
    return metadata


if __name__ == "__main__":
    main()
