# MLflow experiment tracking

Everything MLflow for this project lives here.

```
mlops/mlflow/
├── exports/        committed - the run history as CSV, markdown and JSON
└── store/          gitignored - the SQLite database and run artifacts
```

## Browsing the runs

```bash
pip install -r requirements-dev.txt
python -m mlops.model_building.train          # records a session
mlflow ui --backend-store-uri sqlite:///mlops/mlflow/store/mlflow.db
```

## What gets recorded

Each training session produces:

- **one parent run** — dataset shape, the train/test split, the scale_pos_weight
  used, the selection decision, the deployed threshold, the confusion matrix and
  the leakage-experiment result, plus the figures and tables as artifacts;
- **one nested child run per candidate model** — Logistic Regression, Random
  Forest and XGBoost — each holding its hyperparameter search result
  (`cv_pr_auc_baseline` vs `cv_pr_auc_tuned`) and its held-out test metrics.

Nesting them this way means the three models line up as sibling rows in the UI
and can be compared directly, rather than being flattened into one record.

## Why the store is not committed

`store/mlflow.db` is a SQLite binary. GitHub push protection rejects it as a
**"Lob Test API Key"**: in the raw database pages a metric name sits flush
against its 32-character hexadecimal run id, so `test_f1` followed by
`e70688…f42` reads as `test_<32 hex>` — exactly Lob's key format. It is a false
positive, but the database is regenerable derived data, and allowlisting a
secret-scanner hit in order to commit a binary is a bad habit to establish.

`exports/` exists so that nothing is lost by ignoring it. Those files are
committed, so the run history is readable on GitHub, quotable in the report, and
renderable by the deployed Streamlit app — which reads the CSV and therefore
never imports MLflow.

## Why this folder is nested inside `mlops/`

A directory named `mlflow/` at the repository root would sit on `sys.path`. As a
bare directory it is harmless, but the moment it gained an `__init__.py`,
`import mlflow` would resolve to it instead of the installed library and every
training run would fail. Nested here it is not importable as `mlflow` at all.
A test asserts no `__init__.py` ever appears in this directory.

## Regenerating the exports

They refresh automatically at the end of every training run. To rebuild them
from an existing store without retraining:

```bash
python -m scripts.export_mlflow_runs
```
