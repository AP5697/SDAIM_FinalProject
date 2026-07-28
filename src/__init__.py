"""Online Shoppers Purchasing Intention - modular ML pipeline.

Package layout:
    config         Paths, constants, feature groupings, search spaces.
    utils          Logging, timing, artifact persistence.
    data_loader    Dataset acquisition, loading and validation.
    preprocessing  Cleaning, feature engineering and the sklearn ColumnTransformer.
    train          Model training, hyperparameter search, model selection.
    evaluation     Metrics, threshold analysis, error analysis.
    visualisation  All figure generation.
    inference      Thin prediction wrapper imported by the Streamlit app.
"""

__version__ = "1.0.0"
