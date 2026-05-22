import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
from telco_churn.config import PREDICTION_THRESHOLD


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = PREDICTION_THRESHOLD) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "recall_churn": recall_score(y_test, y_pred),
        "precision_churn": precision_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
    }

    print(classification_report(y_test, y_pred, digits=3))
    return metrics
