"""Central configuration for the Online Shoppers Purchasing Intention project.

Every path, constant, feature grouping and hyperparameter search space used
anywhere in this project is defined here. No other module may hardcode a path,
a magic number, or a column name.

Dataset:
    Online Shoppers Purchasing Intention Dataset (UCI ML Repository, ID 468).
    Sakar, C. O., Polat, S. O., Katircioglu, M. and Kastro, Y. (2018).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

# This module lives at mlops/model_building/config.py, so the repository root is
# two levels up. README.md and requirements.txt must stay at that root: Hugging
# Face reads the Space configuration from the root README's front-matter.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MLOPS_DIR: Final[Path] = PROJECT_ROOT / "mlops"

DATA_DIR: Final[Path] = MLOPS_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

MODEL_DIR: Final[Path] = PROJECT_ROOT / "models"
REPORTS_DIR: Final[Path] = PROJECT_ROOT / "reports"
FIGURES_DIR: Final[Path] = REPORTS_DIR / "figures"
TABLES_DIR: Final[Path] = REPORTS_DIR / "tables"
LOG_DIR: Final[Path] = PROJECT_ROOT / "logs"

RAW_DATA_FILE: Final[Path] = RAW_DATA_DIR / "online_shoppers_intention.csv"
MODEL_FILE: Final[Path] = MODEL_DIR / "model.joblib"
METADATA_FILE: Final[Path] = MODEL_DIR / "metadata.json"

ALL_DIRS: Final[tuple[Path, ...]] = (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_DIR,
    FIGURES_DIR,
    TABLES_DIR,
    LOG_DIR,
)

# --------------------------------------------------------------------------- #
# Dataset provenance
# --------------------------------------------------------------------------- #

DATASET_NAME: Final[str] = "Online Shoppers Purchasing Intention Dataset"
DATASET_URL: Final[str] = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
DATASET_HOME: Final[str] = (
    "https://archive.ics.uci.edu/dataset/468/"
    "online+shoppers+purchasing+intention+dataset"
)
DATASET_CITATION: Final[str] = (
    "Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y. (2019). "
    "Real-time prediction of online shoppers' purchasing intention using "
    "multilayer perceptron and LSTM recurrent neural networks. "
    "Neural Computing and Applications, 31, 6893-6908."
)
DATASET_LICENCE: Final[str] = "Creative Commons Attribution 4.0 International (CC BY 4.0)"

# --------------------------------------------------------------------------- #
# Deployment targets
#
# Kept here rather than inline in the application so the repository and Space
# are named in exactly one place. The GitHub Actions workflow holds the same
# Space coordinates as env vars; they must be changed together if the Space
# is ever renamed.
# --------------------------------------------------------------------------- #

GITHUB_REPO_URL: Final[str] = "https://github.com/AP5697/SDAIM_FinalProject"
HF_SPACE_URL: Final[str] = "https://huggingface.co/spaces/Aishawarya/SDAIM_Final"

# --------------------------------------------------------------------------- #
# MLflow experiment tracking
#
# Tracking is a development-time concern: it records how the deployed model was
# arrived at, not how it is served. The Hugging Face Space never imports MLflow,
# and models/model.joblib remains the single deployment artifact.
#
# A local SQLite store is used rather than a tracking server so the experiment
# history is reproducible from a clone with no infrastructure to stand up.
#
# SQLite specifically, not the older './mlruns' file store: MLflow 3.x puts the
# filesystem backend in maintenance mode and raises rather than writing to it.
# SQLite is the supported local backend and is what `mlflow ui` expects here.
# --------------------------------------------------------------------------- #

# Everything MLflow lives under mlops/mlflow/. The store is regenerable and is
# gitignored; the exports beside it are committed, so the run history is visible
# on GitHub and to the deployed app without anyone installing MLflow.
#
# Note this directory is NOT importable as `mlflow` - it sits inside the mlops
# package, not on sys.path - so it cannot shadow the installed library. A folder
# named `mlflow/` at the repository root would, the moment it gained an
# __init__.py, and there is a test guarding against exactly that.
MLFLOW_DIR: Final[Path] = MLOPS_DIR / "mlflow"
MLFLOW_STORE_DIR: Final[Path] = MLFLOW_DIR / "store"
MLFLOW_EXPORT_DIR: Final[Path] = MLFLOW_DIR / "exports"

MLFLOW_DB_FILE: Final[Path] = MLFLOW_STORE_DIR / "mlflow.db"
# SQLAlchemy URL form; as_posix() keeps the Windows drive letter valid in a URI.
MLFLOW_TRACKING_URI: Final[str] = f"sqlite:///{MLFLOW_DB_FILE.as_posix()}"
MLFLOW_ARTIFACT_DIR: Final[Path] = MLFLOW_STORE_DIR / "artifacts"
MLFLOW_EXPERIMENT_NAME: Final[str] = "purchase-intent-scorer"

# Exported run history, committed so it is readable without MLflow installed.
MLFLOW_RUNS_CSV: Final[Path] = MLFLOW_EXPORT_DIR / "runs_summary.csv"
MLFLOW_RUNS_MARKDOWN: Final[Path] = MLFLOW_EXPORT_DIR / "model_comparison.md"
MLFLOW_SESSION_JSON: Final[Path] = MLFLOW_EXPORT_DIR / "session_summary.json"

# Set MLFLOW_DISABLED=1 to skip tracking entirely, for example in CI where the
# run history would be discarded with the runner anyway.
MLFLOW_DISABLED_ENV_VAR: Final[str] = "MLFLOW_DISABLED"

# --------------------------------------------------------------------------- #
# Target
# --------------------------------------------------------------------------- #

TARGET_COLUMN: Final[str] = "Revenue"
POSITIVE_CLASS_LABEL: Final[str] = "Purchase"
NEGATIVE_CLASS_LABEL: Final[str] = "No purchase"

# --------------------------------------------------------------------------- #
# Feature groupings
#
# NOTE (key EDA finding): pandas infers OperatingSystems, Browser, Region and
# TrafficType as int64, but they are *integer-coded nominal categories* - the
# integers are arbitrary identifiers, not magnitudes. Browser 4 is not "twice"
# Browser 2. Scaling them as numerics would inject a false ordering into every
# distance- and gradient-based model. They are therefore routed to the
# categorical branch of the ColumnTransformer.
# --------------------------------------------------------------------------- #

TRUE_NUMERIC_FEATURES: Final[list[str]] = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

NATIVE_CATEGORICAL_FEATURES: Final[list[str]] = [
    "Month",
    "VisitorType",
    "Weekend",
]

INTEGER_CODED_CATEGORICAL_FEATURES: Final[list[str]] = [
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]

CATEGORICAL_FEATURES: Final[list[str]] = (
    NATIVE_CATEGORICAL_FEATURES + INTEGER_CODED_CATEGORICAL_FEATURES
)

ALL_FEATURES: Final[list[str]] = TRUE_NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Features whose value is only knowable after (or partly because of) a
# conversion. See mlops/model_building/train.py::run_leakage_experiment for the empirical test.
#
# PageValues is computed by Google Analytics as the transaction revenue plus
# goal value attributable to a page, divided by unique pageviews. Its numerator
# is therefore partly derived from the very purchase event we are predicting.
LEAKAGE_CANDIDATE_FEATURES: Final[list[str]] = ["PageValues"]

# Engineered features created in mlops/model_building/preprocessing.py.
ENGINEERED_FEATURES: Final[list[str]] = [
    "TotalPages",
    "TotalDuration",
    "AvgDurationPerPage",
    "ProductPageRatio",
    "BounceExitDelta",
]

# --------------------------------------------------------------------------- #
# Splitting / cross-validation
# --------------------------------------------------------------------------- #

RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.20
CV_FOLDS: Final[int] = 5
N_SEARCH_ITERATIONS: Final[int] = 30
N_JOBS: Final[int] = -1

# Average precision (area under the precision-recall curve) is the tuning
# objective. With a ~15% positive rate, PR-AUC responds to the minority class
# in a way accuracy and ROC-AUC do not. See docs for the full justification.
SEARCH_SCORING: Final[str] = "average_precision"

# --------------------------------------------------------------------------- #
# Business cost model
#
# Used to convert model probabilities into a decision threshold. A "retention
# intervention" here means a real-time incentive (discount code, free shipping,
# proactive live-chat) fired at a browsing session.
# --------------------------------------------------------------------------- #

INTERVENTION_COST: Final[float] = 5.00  # cost of firing one incentive (USD)
AVERAGE_ORDER_VALUE: Final[float] = 120.00  # mean basket value (USD)
INTERVENTION_UPLIFT: Final[float] = 0.10  # incremental conversion probability

# --------------------------------------------------------------------------- #
# Hyperparameter search spaces
# --------------------------------------------------------------------------- #

LOGISTIC_REGRESSION_PARAMS: Final[dict[str, Any]] = {
    "classifier__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear", "saga"],
}

RANDOM_FOREST_PARAMS: Final[dict[str, Any]] = {
    "classifier__n_estimators": [200, 300, 400, 500],
    "classifier__max_depth": [8, 12, 16, 20, None],
    "classifier__min_samples_split": [2, 5, 10, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 8],
    "classifier__max_features": ["sqrt", "log2", 0.5],
}

XGBOOST_PARAMS: Final[dict[str, Any]] = {
    "classifier__n_estimators": [200, 300, 400, 600],
    "classifier__max_depth": [3, 4, 5, 6, 8],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.6, 0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5],
}

# --------------------------------------------------------------------------- #
# Model size guardrail for the free-tier Hugging Face Space
# --------------------------------------------------------------------------- #

MAX_MODEL_SIZE_MB: Final[float] = 25.0

# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

FIGURE_DPI: Final[int] = 130
FIGURE_FORMAT: Final[str] = "png"
PLOT_STYLE: Final[str] = "seaborn-v0_8-whitegrid"
PRIMARY_COLOUR: Final[str] = "#2563eb"
SECONDARY_COLOUR: Final[str] = "#f59e0b"
NEGATIVE_COLOUR: Final[str] = "#94a3b8"

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)-22s | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: Final[str] = "INFO"


def ensure_directories() -> None:
    """Create every project directory that later stages write into.

    Idempotent - safe to call on every entry point.
    """
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)
