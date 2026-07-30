"""MLOps pipeline for the Purchase Intent Scorer.

Organised by pipeline stage rather than by module type:

    data/            raw dataset as downloaded from UCI
    model_building/  loading, preprocessing, training, evaluation, tracking
    deployment/      the Streamlit application and its inference wrapper
    mlflow/          experiment tracking store and exported run history
"""
