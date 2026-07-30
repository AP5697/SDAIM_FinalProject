"""MLflow experiment tracking, wrapped so it can never break training.

Every function here degrades to a no-op when MLflow is unavailable, disabled by
environment variable, or raises. That is deliberate: tracking records *how* a
model was arrived at, and losing that record is an inconvenience, whereas losing
the training run itself is a real cost. The same reasoning applies as for the
deployment artifact - `models/model.joblib` is produced identically whether or
not anything is being tracked.

What gets recorded per training session:

* one parent run for the session, holding the dataset shape, the split, the
  selection decision and the artifacts quoted in the report;
* one nested child run per candidate model, holding its hyperparameter search
  result and its held-out test metrics, so the three are directly comparable in
  the MLflow UI.

Browse the history with::

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

_UNAVAILABLE_REASON: str | None = None


def _mlflow() -> Any | None:
    """Return the mlflow module, or None if tracking is unavailable.

    Returns:
        The imported module, or None when MLflow is not installed or has been
        disabled via the environment.
    """
    global _UNAVAILABLE_REASON

    if os.environ.get(config.MLFLOW_DISABLED_ENV_VAR, "").strip() not in ("", "0", "false"):
        _UNAVAILABLE_REASON = f"{config.MLFLOW_DISABLED_ENV_VAR} is set"
        return None

    try:
        import mlflow

        return mlflow
    except Exception as exc:  # pragma: no cover - environment dependent
        _UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        return None


def is_enabled() -> bool:
    """Report whether experiment tracking will actually record anything."""
    return _mlflow() is not None


def unavailable_reason() -> str:
    """Explain why tracking is inactive, for logging."""
    return _UNAVAILABLE_REASON or "unknown"


def _configure(mlflow: Any) -> None:
    """Point MLflow at the project's local SQLite store and artifact directory."""
    config.MLFLOW_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # set_experiment() cannot specify where artifacts land, so the experiment is
    # created explicitly the first time to pin them inside the project rather
    # than in a default directory relative to the current working directory.
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT_NAME,
            artifact_location=config.MLFLOW_ARTIFACT_DIR.as_uri(),
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)


@contextmanager
def run(name: str, nested: bool = False) -> Iterator[None]:
    """Open an MLflow run, or do nothing at all if tracking is unavailable.

    Args:
        name: Run name shown in the MLflow UI.
        nested: True for a per-model child of the session run.

    Yields:
        Nothing. The context is purely for scoping.
    """
    mlflow = _mlflow()
    if mlflow is None:
        yield
        return

    try:
        _configure(mlflow)
        with mlflow.start_run(run_name=name, nested=nested):
            yield
    except Exception as exc:  # pragma: no cover - tracking must never be fatal
        logger.warning("MLflow run %r could not be recorded: %s", name, exc)
        yield


def log_params(params: dict[str, Any]) -> None:
    """Record hyperparameters or configuration values on the active run."""
    mlflow = _mlflow()
    if mlflow is None or not params:
        return
    try:
        # MLflow stores parameters as strings and rejects oversized values, so
        # everything is stringified and truncated to the documented 500-char cap.
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log_params skipped: %s", exc)


def log_metrics(metrics: dict[str, Any]) -> None:
    """Record numeric metrics on the active run, ignoring non-numeric entries."""
    mlflow = _mlflow()
    if mlflow is None or not metrics:
        return
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not numeric:
        return
    try:
        mlflow.log_metrics(numeric)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log_metrics skipped: %s", exc)


def log_artifacts(paths: list[Path], subdirectory: str | None = None) -> None:
    """Attach files to the active run.

    Args:
        paths: Files to upload. Missing paths are skipped silently, so callers
            do not have to guard every optional figure.
        subdirectory: Folder to group them under inside the run.
    """
    mlflow = _mlflow()
    if mlflow is None:
        return
    for path in paths:
        if not path or not Path(path).exists():
            continue
        try:
            mlflow.log_artifact(str(path), artifact_path=subdirectory)
        except Exception as exc:  # pragma: no cover
            logger.warning("MLflow log_artifact skipped for %s: %s", path, exc)


def log_directory(directory: Path, subdirectory: str | None = None) -> None:
    """Attach an entire directory to the active run, if it exists."""
    mlflow = _mlflow()
    if mlflow is None or not directory.exists():
        return
    try:
        mlflow.log_artifacts(str(directory), artifact_path=subdirectory)
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow log_artifacts skipped for %s: %s", directory, exc)


def set_tags(tags: dict[str, Any]) -> None:
    """Tag the active run, for filtering in the UI."""
    mlflow = _mlflow()
    if mlflow is None or not tags:
        return
    try:
        mlflow.set_tags({k: str(v)[:500] for k, v in tags.items()})
    except Exception as exc:  # pragma: no cover
        logger.warning("MLflow set_tags skipped: %s", exc)
