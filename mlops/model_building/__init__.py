"""Model-building stage: data acquisition through to a serialised pipeline.

Everything that produces `models/model.joblib` lives here - loading, cleaning,
feature engineering, training, tuning, evaluation and experiment tracking. The
deployment stage imports none of it directly; the two communicate only through
the saved artifact.
"""
