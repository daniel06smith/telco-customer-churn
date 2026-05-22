"""
Shared utilities for the Streamlit app.

Loaded by both app.py (Churn Predictor) and pages/Dashboard.py.
Streamlit's cache is session-level and persists across page navigations,
so compute_pdp() is only run once per session regardless of which page
triggers it first.
"""
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st

from telco_churn.config import MODEL_PATH, FEATURE_COLS_PATH
from telco_churn.data.preprocess import preprocess

# ── Typical customer used as the fixed baseline when computing PDP curves ──────
# All other features are held at these values while one feature is varied.
# Values match the training-data median / mode for each column.
BASELINE = {
    "gender": "Male", "SeniorCitizen": 0,
    "Partner": "No",  "Dependents": "No",
    "tenure": 29,
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 65.0, "TotalCharges": "1885.0",
    "Churn": "No",
}

# Features to show PDP curves for, with their grid of values to sweep
PDP_FEATURES = {
    "tenure": {
        "type": "numeric",
        "grid": list(range(0, 73)),
        "label": "Tenure (months)",
    },
    "MonthlyCharges": {
        "type": "numeric",
        "grid": [round(i * 2.0, 1) for i in range(61)],   # 0 – 120 in steps of 2
        "label": "Monthly Charges ($)",
    },
    "Contract": {
        "type": "categorical",
        "grid": ["Month-to-month", "One year", "Two year"],
        "label": "Contract Type",
    },
    "InternetService": {
        "type": "categorical",
        "grid": ["DSL", "Fiber optic", "No"],
        "label": "Internet Service",
    },
}


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None
    model = joblib.load(MODEL_PATH)
    feature_cols = json.load(open(FEATURE_COLS_PATH))
    return model, feature_cols


def predict_single(model, feature_cols: list, raw_values: dict) -> float:
    """Run model.predict_proba on one raw-input row."""
    df = pd.DataFrame([raw_values])
    proc = preprocess(df)
    X = proc.reindex(columns=feature_cols, fill_value=0)
    return model.predict_proba(X)[0, 1]


@st.cache_data
def compute_pdp() -> dict:
    """
    Compute PDP curves for all features in PDP_FEATURES.

    For each feature, sweeps its grid while holding everything else at BASELINE.
    This is technically an ICE curve at the baseline customer rather than a true
    PDP (which would average over the training distribution), but for the range
    of features we show the difference is negligible and no training data is
    needed at runtime.

    Cached with no arguments so it runs once per session regardless of
    which page calls it first.
    """
    model, feature_cols = load_model()
    if model is None:
        return {}

    result = {}
    for feat, cfg in PDP_FEATURES.items():
        probs = []
        for val in cfg["grid"]:
            row = {**BASELINE, feat: val}
            # Keep TotalCharges proportional when sweeping tenure
            if feat == "tenure":
                row["TotalCharges"] = str(int(val) * int(BASELINE["MonthlyCharges"]))
            probs.append(predict_single(model, feature_cols, row))
        result[feat] = {**cfg, "proba": probs}

    return result
