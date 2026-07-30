---
title: Purchase Intent Scorer
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.59.1
python_version: "3.12"
app_file: mlops/deployment/app.py
pinned: false
license: mit
short_description: Predict purchase intent from a live browsing session
---

# 🛒 Purchase Intent Scorer

Predicts whether a live e-commerce browsing session will end in a purchase, so a
retailer can fire a real-time incentive at the sessions that are *winnable but not
yet won* — instead of discounting everyone or no one.

[![MLOps pipeline](https://github.com/AP5697/SDAIM_FinalProject/actions/workflows/pipeline.yml/badge.svg)](https://github.com/AP5697/SDAIM_FinalProject/actions/workflows/pipeline.yml)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.59.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Live demo:** <https://huggingface.co/spaces/Aishawarya/SDAIM_Final>
> **Repository:** <https://github.com/AP5697/SDAIM_FinalProject>

---

## The problem

Only **15.47%** of sessions in the source data end in a purchase. The other ~85%
consume bandwidth and return nothing. Two naive policies both lose money:

| Policy | Outcome |
|---|---|
| Discount every visitor | Margin destroyed on the 15% who would have bought anyway |
| Discount nobody | Winnable sessions walk away |

This model scores a session in ~10 ms and ranks it, so incentive spend goes to
the sessions where it changes the outcome.

## Results

Held-out test set of 2,441 sessions, XGBoost at the selected operating threshold of 0.70:

| Metric | Value |
|---|---|
| ROC-AUC | **0.9367** |
| PR-AUC (average precision) | **0.7546** |
| Precision (purchase) | 0.6768 |
| Recall (purchase) | 0.7016 |
| F1 (purchase) | 0.6889 |
| Accuracy | 0.9009 |
| Median inference latency | 10.4 ms |
| Model artifact size | 0.474 MB |

**Accuracy is reported for completeness only.** Predicting "no purchase" for every
session scores 84.5% accuracy and is worthless, so model selection was driven by
PR-AUC, whose baseline moves with class prevalence.

### Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Latency |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.8697 | 0.5615 | 0.7644 | 0.6475 | 0.9157 | 0.6701 | 8.2 ms |
| Random Forest | 0.8685 | 0.5546 | 0.8115 | 0.6589 | 0.9328 | 0.7428 | 48.9 ms |
| **XGBoost** | **0.8710** | 0.5585 | **0.8377** | **0.6702** | **0.9367** | **0.7546** | 10.4 ms |

XGBoost wins on every ranking metric *and* runs 4.7× faster than the Random
Forest at inference — a decision that matters for a real-time scoring UI.

## Features

- **Session scorer** — score one session from its Google Analytics metrics, with
  four presets built from the medians of real behavioural segments in the data
- **Per-prediction SHAP attribution** — why *this* session scored what it did,
  not just which features matter globally
- **What-if analysis** — sweep any one field across its observed range, holding
  everything else fixed, to see how the prediction responds
- **Switchable decision policy** — compare the max-F1 threshold (0.70) against
  the max-campaign-value threshold (0.75) live
- **Batch scoring** — upload an analytics export and rank every session
- **Model insights** — live metrics, feature importance, the calibration curve
  and the leakage experiment
- **Input validation** — rejects negative counts, out-of-range rates, and
  physically impossible sessions (time spent on pages never visited)
- **Prediction history** with CSV download
- **Graceful degradation** — unseen category levels encode to all-zeros, and
  SHAP falling over degrades to global importances rather than breaking the app

## Quick start

```bash
git clone <your-repo-url>
cd Final_project
pip install -r requirements.txt
streamlit run mlops/deployment/app.py
```

To reproduce the model from scratch:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m scripts.run_eda
python -m mlops.model_building.train
python -m scripts.add_calibration
python -m scripts.build_app_assets
python -m pytest tests/ -v
```

## Experiment tracking (MLflow)

`python -m mlops.model_building.train` records every training session to a local MLflow store.
Browse it with:

```bash
mlflow ui --backend-store-uri sqlite:///mlops/mlflow/store/mlflow.db
```

Each session is one **parent run** holding the dataset shape, the split, the
selection decision and the deployed configuration, with one **nested child run
per candidate model** carrying its hyperparameter search result and held-out
test metrics — so the three models are directly comparable in the UI.

The store is generated locally and is not committed — run `python -m mlops.model_building.train`
once and `mlops/mlflow/store/` appears, then open the UI above.

Three decisions worth noting:

- **SQLite, not the `./mlruns` file store.** MLflow 3.x puts the filesystem
  backend in maintenance mode and raises rather than writing to it, so the older
  `mlflow ui --backend-store-uri ./mlruns` form fails on this version.
- **The store is gitignored.** GitHub push protection flags `mlflow.db` as a
  "Lob Test API Key": in the raw SQLite page bytes a metric name abuts its
  32-character hex run id, so `test_f1` + `e70688…f42` matches Lob's
  `test_<32 hex>` key format exactly. It is a false positive, but the database
  is regenerable derived data, so ignoring it beats teaching the scanner to be
  ignored.
- **Tracking cannot break training.** Every call in `mlops/model_building/tracking.py` degrades
  to a no-op if MLflow is missing, disabled via `MLFLOW_DISABLED=1`, or raises.
  Losing the experiment record is an inconvenience; losing the training run is
  not. This is also why MLflow is in `requirements-dev.txt` and not
  `requirements.txt` — the Space never imports it, and `models/model.joblib`
  remains the single deployment artifact.

## Project structure

```
Final_project/
├── README.md                   # This file; HF Space config in the front-matter
├── requirements.txt            # Exactly pinned runtime dependencies (the Space)
├── requirements-dev.txt        # Training, MLflow and test dependencies only
│
├── mlops/                      # The pipeline, organised by stage
│   ├── data/raw/               # Source CSV (committed for reproducibility)
│   │
│   ├── model_building/         # STAGE 1 - everything that produces the model
│   │   ├── config.py           # All paths, constants, feature groups, search spaces
│   │   ├── data_loader.py      # Acquisition, validation, stratified splitting
│   │   ├── preprocessing.py    # Cleaning, feature engineering, ColumnTransformer
│   │   ├── train.py            # Training, tuning, selection, leakage experiment
│   │   ├── evaluation.py       # Metrics, threshold economics, error analysis
│   │   ├── visualisation.py    # All figure generation
│   │   ├── tracking.py         # MLflow logging, no-op if MLflow is absent
│   │   └── utils.py            # Logging, timing, artifact persistence
│   │
│   ├── deployment/             # STAGE 2 - everything that serves the model
│   │   ├── app.py              # Streamlit application (HF Space entry point)
│   │   └── inference.py        # Thin wrapper the app imports
│   │
│   └── mlflow/                 # STAGE 3 - experiment tracking
│       ├── README.md           # What is recorded and how to browse it
│       ├── exports/            # Committed run history (CSV, markdown, JSON)
│       └── store/              # SQLite database + artifacts (gitignored)
│
├── models/                     # The single deployment artifact
│   ├── model.joblib            # Complete fitted pipeline (preprocessing + model)
│   ├── metadata.json           # Versions, metrics, threshold, provenance
│   └── app_schema.json         # Data-derived widget bounds and presets
│
├── scripts/
│   ├── run_eda.py              # Full EDA run
│   ├── build_app_assets.py     # Derives UI schema from the training data
│   ├── add_calibration.py      # Measures the calibration curve on the test split
│   └── export_mlflow_runs.py   # Writes the committed MLflow run history
│
├── tests/test_pipeline.py      # 47 tests against the real artifact
├── reports/figures/            # 16 generated figures
├── reports/tables/             # 15 generated result tables
├── docs/                       # Presentation script and workflow one-pager
├── .streamlit/config.toml      # Theme and upload limits
└── .github/workflows/pipeline.yml   # 4-stage MLOps pipeline
```

`README.md` and `requirements.txt` stay at the repository root because Hugging
Face reads the Space configuration from the root README's front-matter and
installs dependencies from the root requirements file. The front-matter's
`app_file` therefore points at `mlops/deployment/app.py`.

## Dataset

**Online Shoppers Purchasing Intention Dataset** — UCI Machine Learning Repository (ID 468).
12,330 sessions × 18 attributes from a 12-month period of a real online retailer.

- Source: <https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset>
- Licence: CC BY 4.0
- Citation: Sakar, C.O., Polat, S.O., Katircioglu, M., Kastro, Y. (2019).
  *Real-time prediction of online shoppers' purchasing intention using multilayer
  perceptron and LSTM recurrent neural networks.* Neural Computing and
  Applications, 31, 6893–6908.

## Design decisions worth knowing

**Preprocessing travels with the model.** `model.joblib` is a single
`sklearn.Pipeline` containing feature engineering, imputation, scaling, encoding
*and* the classifier. The app never re-implements preprocessing, so train/serve
skew is structurally impossible.

**Four "numeric" columns are actually categorical.** pandas infers
`OperatingSystems`, `Browser`, `Region` and `TrafficType` as `int64`, but they are
nominal identifier codes. Scaling them would tell the model browser 13 is
"greater than" browser 2. They are one-hot encoded instead.

**`PageValues` was interrogated, not just used.** Google Analytics derives Page
Value partly from transaction revenue, so it is partly a function of the outcome
being predicted. Refitting without it drops PR-AUC from 0.7068 to 0.3437 — a 51%
collapse. The feature is retained (it is legitimately available at session start,
computed from historical traffic) but the dependence is documented rather than
hidden.

**The probabilities are deliberately not calibrated.** Every bin of the
reliability curve sits above the diagonal: the model predicts higher
probabilities than the observed conversion rate justifies, by a count-weighted
mean of +0.126 (Brier 0.0952). This is a consequence of training with
`scale_pos_weight = 5.4` — the inverse class ratio — to protect recall on a
15.6% minority. The scores are therefore sound for *ranking* and *thresholding*,
which is all the application uses them for, but should not be read as literal
conversion frequencies. See **Model insights → Calibration** in the app.

## The pipeline

Pushing to `main` triggers `.github/workflows/pipeline.yml`, which runs four
stages mirroring the project's own structure. Each gates the next, so nothing
reaches the live Space unless every stage before it is green:

```
register-dataset ──▶ data-prep ──▶ model-training ──▶ deploy-hosting
```

| Stage | What it asserts |
|---|---|
| **register-dataset** | The CSV is present and unchanged: checksum, row count, expected columns, and the 15.47% target rate every committed metric assumes |
| **data-prep** | The preprocessing pipeline fits and transforms the real data — row count preserved, feature names aligned, integer-coded identifiers still one-hot encoded rather than scaled, split still stratified |
| **model-training** | The committed artifact loads and scores; retraining from scratch **reproduces the committed baseline** within 0.01 on PR-AUC, ROC-AUC, F1, recall and precision; all 49 tests pass |
| **deploy-hosting** | Space metadata is valid, `app_file` exists, then the application files sync to Hugging Face using the `HF_TOKEN` secret |

**On the training stage.** It retrains but does *not* deploy what it trains.
The retrained model is a reproducibility check — it answers "is this still the
pipeline that produced the model we serve?" — after which the committed,
tested artifact is restored and deployed. That keeps the metrics quoted in this
README the ones actually shipped, while still failing loudly if the pipeline
ever stops reproducing them.

The gate was verified in both directions before being relied on: an identical
baseline passes, and a deliberately injected 0.25 drift in PR-AUC fails the run
naming the offending metric. A gate only ever observed passing is
indistinguishable from no gate.

Note that `python_version` in this file's front-matter and `PYTHON_VERSION` in
the workflow must stay in step — `xgboost`, `scipy` and `shap` all require
Python ≥ 3.12, and a mismatch would fail the Space build after CI had already
passed. The workflow asserts this rather than trusting it.

## Future work

- Wrap the classifier in `CalibratedClassifierCV` if calibrated probabilities are
  ever needed; the trade-off is giving back some of the recall the class
  weighting was introduced to buy
- Monitor for drift: the source data is missing January and April entirely, so
  seasonal coverage is incomplete
- A/B test the intervention itself; the 10% uplift assumption in the threshold
  economics is stated, not measured

## Licence

MIT
