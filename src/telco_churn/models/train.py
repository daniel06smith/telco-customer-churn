import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from telco_churn.config import RANDOM_STATE, BEST_LGBM_PARAMS


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,  # suppress LightGBM info logs
    )
    model.fit(X_train, y_train)
    return model


def train_best_lightgbm(X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    """LightGBM with Optuna-tuned hyperparameters. This is the model saved for the app."""
    model = LGBMClassifier(
        **BEST_LGBM_PARAMS,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
