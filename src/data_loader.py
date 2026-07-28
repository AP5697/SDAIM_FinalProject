"""Dataset acquisition, loading, validation and splitting.

The raw CSV is committed to the repository so that the analysis is reproducible
from a clean clone, but :func:`download_dataset` will fetch it from the UCI
Machine Learning Repository if it is missing.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.utils import get_logger

logger = get_logger(__name__)

_DOWNLOAD_TIMEOUT_SECONDS: int = 120
_CSV_MEMBER_SUFFIX: str = "online_shoppers_intention.csv"


def download_dataset(destination: Path | None = None, force: bool = False) -> Path:
    """Download and extract the dataset from the UCI repository.

    Args:
        destination: Directory to extract into. Defaults to ``data/raw``.
        force: Re-download even if the CSV already exists.

    Returns:
        Path to the extracted CSV file.

    Raises:
        RuntimeError: If the download fails or the archive does not contain the
            expected CSV member.
    """
    destination = destination or config.RAW_DATA_DIR
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / _CSV_MEMBER_SUFFIX

    if target.exists() and not force:
        logger.info("Dataset already present at %s - skipping download", target.name)
        return target

    logger.info("Downloading dataset from %s", config.DATASET_URL)
    try:
        with urllib.request.urlopen(  # noqa: S310 - fixed, trusted UCI URL
            config.DATASET_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            payload = response.read()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"Failed to download the dataset from {config.DATASET_URL}. "
            f"Check your network connection or download it manually to "
            f"{target}. Underlying error: {exc}"
        ) from exc

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = [n for n in archive.namelist() if n.endswith(_CSV_MEMBER_SUFFIX)]
        if not members:
            raise RuntimeError(
                f"Archive downloaded from {config.DATASET_URL} did not contain "
                f"a member ending in '{_CSV_MEMBER_SUFFIX}'. "
                f"Found: {archive.namelist()}"
            )
        target.write_bytes(archive.read(members[0]))

    logger.info("Dataset extracted to %s", target)
    return target


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw dataset and validate its schema.

    Args:
        path: CSV path. Defaults to :data:`config.RAW_DATA_FILE`.

    Returns:
        The raw DataFrame, unmodified.

    Raises:
        FileNotFoundError: If the CSV is missing and cannot be downloaded.
        ValueError: If expected columns are absent.
    """
    path = path or config.RAW_DATA_FILE
    if not path.exists():
        logger.warning("Raw data not found at %s - attempting download", path)
        path = download_dataset()

    frame = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows x %d columns", *frame.shape)

    expected = set(config.ALL_FEATURES) | {config.TARGET_COLUMN}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing expected columns: {sorted(missing)}. "
            f"Columns present: {sorted(frame.columns)}"
        )
    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target vector.

    Args:
        frame: DataFrame containing both features and the target column.

    Returns:
        A ``(X, y)`` tuple where ``y`` is cast to integer 0/1.
    """
    features = frame[config.ALL_FEATURES].copy()
    target = frame[config.TARGET_COLUMN].astype(int)
    return features, target


def stratified_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train and test sets, preserving the class ratio.

    Stratification matters here because the positive class is only ~15% of the
    data; an unstratified split can materially shift the test-set base rate and
    make metric comparisons unreliable.

    Args:
        features: Feature matrix.
        target: Binary target vector.
        test_size: Test proportion. Defaults to :data:`config.TEST_SIZE`.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size or config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=target,
    )
    logger.info(
        "Split -> train %d rows (%.2f%% positive) | test %d rows (%.2f%% positive)",
        len(x_train),
        100 * y_train.mean(),
        len(x_test),
        100 * y_test.mean(),
    )
    return x_train, x_test, y_train, y_test
