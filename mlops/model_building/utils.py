"""Shared utilities: logging setup, timing, and artifact persistence helpers.

All modules obtain their logger through :func:`get_logger` so that formatting
and log level are configured in exactly one place.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from mlops.model_building import config  # noqa: E402

_CONFIGURED: bool = False


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once for the whole process.

    Args:
        level: Logging level name. Defaults to :data:`config.LOG_LEVEL`.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    config.ensure_directories()
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(config.LOG_DIR / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level or config.LOG_LEVEL)
    root.handlers.clear()
    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger.

    Args:
        name: Logger name, conventionally ``__name__`` of the calling module.

    Returns:
        A logger writing to both stdout and ``logs/pipeline.log``.
    """
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def timed(label: str, logger: logging.Logger | None = None) -> Iterator[dict[str, float]]:
    """Time a block of work and log the elapsed seconds.

    Args:
        label: Human-readable description of the work being timed.
        logger: Logger to emit through. Falls back to this module's logger.

    Yields:
        A dict that is populated with an ``elapsed`` key when the block exits,
        allowing the caller to record the duration.
    """
    log = logger or get_logger(__name__)
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed"] = elapsed
        log.info("%s completed in %.2fs", label, elapsed)


def save_table(frame: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Persist a DataFrame to ``reports/tables`` as CSV.

    Args:
        frame: Table to write.
        name: File stem, without extension.
        index: Whether to write the DataFrame index.

    Returns:
        The path the table was written to.
    """
    config.ensure_directories()
    path = config.TABLES_DIR / f"{name}.csv"
    frame.to_csv(path, index=index)
    get_logger(__name__).info("Saved table -> %s", path.name)
    return path


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write a dictionary to disk as indented UTF-8 JSON.

    Args:
        payload: JSON-serialisable mapping.
        path: Destination file path.

    Returns:
        The path written to.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    get_logger(__name__).info("Saved JSON -> %s", path.name)
    return path


def file_size_mb(path: Path) -> float:
    """Return a file's size in megabytes.

    Args:
        path: File to measure.

    Returns:
        Size in MB, rounded to three decimal places.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Cannot size a file that does not exist: {path}")
    return round(path.stat().st_size / (1024 * 1024), 3)
