"""Deployment stage: everything that serves the trained model.

Contains the Streamlit application and the thin inference wrapper it imports.
This stage deliberately performs no preprocessing of its own - it hands raw
field values to the serialised pipeline, which applies the identical feature
engineering, imputation, scaling and encoding used during training.
"""
