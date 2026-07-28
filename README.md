---
title: Purchase Intent Scorer
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.59.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Predict whether an e-commerce browsing session will end in a purchase
---

# 🛒 Purchase Intent Scorer

Predicts whether a live e-commerce browsing session will end in a purchase, so a
retailer can fire a real-time incentive at the sessions that are *winnable but not
yet won* — instead of discounting everyone or no one.

[![Deploy](https://img.shields.io/badge/deploy-GitHub%20Actions-2088FF?logo=githubactions&logoColor=white)](../../actions)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.59.1-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **Live demo:** _add your Hugging Face Space URL here_
> **Repository:** _add your GitHub repository URL here_

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
- **Batch scoring** — upload an analytics export and rank every session
- **Model insights** — live metrics, feature importance, and the leakage experiment
- **Input validation** — rejects negative counts, out-of-range rates, and
  physically impossible sessions (time spent on pages never visited)
- **Prediction history** with CSV download
- **Graceful degradation** — unseen category levels encode to all-zeros rather
  than crashing the deployed app

## Quick start

```bash
git clone <your-repo-url>
cd Final_project
pip install -r requirements.txt
streamlit run app.py
```

To reproduce the model from scratch:

```bash
python -m scripts.run_eda
python -m src.train
python -m scripts.build_app_assets
python -m pytest tests/ -v
```

## Project structure

```
Final_project/
├── app.py                      # Streamlit application (HF Space entry point)
├── requirements.txt            # Exactly pinned runtime dependencies
├── README.md                   # This file; HF Space config in the front-matter
├── models/
│   ├── model.joblib            # Complete fitted pipeline (preprocessing + model)
│   ├── metadata.json           # Versions, metrics, threshold, provenance
│   └── app_schema.json         # Data-derived widget bounds and presets
├── src/
│   ├── config.py               # All paths, constants, feature groups, search spaces
│   ├── data_loader.py          # Acquisition, validation, stratified splitting
│   ├── preprocessing.py        # Cleaning, feature engineering, ColumnTransformer
│   ├── train.py                # Training, tuning, selection, leakage experiment
│   ├── evaluation.py           # Metrics, threshold economics, error analysis
│   ├── visualisation.py        # All figure generation
│   ├── inference.py            # Thin wrapper the app imports
│   └── utils.py                # Logging, timing, artifact persistence
├── scripts/
│   ├── run_eda.py              # Full EDA run
│   └── build_app_assets.py     # Derives UI schema from the training data
├── tests/test_pipeline.py      # 20 tests against the real artifact
├── data/raw/                   # Source CSV (committed for reproducibility)
├── reports/figures/            # 14 generated figures
├── reports/tables/             # Generated result tables
└── .github/workflows/deploy.yml
```

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

## Deployment

Pushing to `main` triggers `.github/workflows/deploy.yml`, which runs the test
suite and then syncs the application files to the Hugging Face Space using an
`HF_TOKEN` stored as a GitHub Actions secret. The Space rebuilds automatically.
See the workflow file for setup instructions.

## Future work

- Calibrate probabilities (isotonic or Platt) — the Brier score of 0.0952 suggests headroom
- Replace global feature importance in the app with per-prediction SHAP attributions
- Monitor for drift: the source data is missing January and April entirely, so
  seasonal coverage is incomplete
- A/B test the intervention itself; the 10% uplift assumption in the threshold
  economics is stated, not measured

## Licence

MIT
