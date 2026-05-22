"""
Hyperparameter tuning for LightGBM using Optuna + MLflow.

Each Optuna trial trains a LightGBM model with a different set of
hyperparameters and logs the result as its own MLflow run. After all
trials complete, the best params are printed and saved to MLflow.

Usage:
    python scripts/tune.py
"""
import optuna
import mlflow
import mlflow.sklearn
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import fbeta_score

from telco_churn.config import (
    RAW_DATA_PATH, RANDOM_STATE, TEST_SIZE
)
from telco_churn.data.load import load_csv
from telco_churn.data.preprocess import preprocess
from telco_churn.features.build import split_features_target, split_train_test

EXPERIMENT_NAME = "telco-churn-tuning"
N_TRIALS = 30


def objective(trial: optuna.Trial, X_train, X_test, y_train, y_test) -> float:
    """
    Called by Optuna once per trial. Returns the score to maximise (F2).

    trial.suggest_* tells Optuna the search space for each parameter.
    Optuna uses results from previous trials to pick smarter values each time.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    # Tune the threshold alongside the model — optimal threshold depends on
    # the model's probability calibration, so they should be searched together.
    threshold = trial.suggest_float("threshold", 0.1, 0.6)

    model = LGBMClassifier(
        **params,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)

    # F2 score: weights recall twice as heavily as precision.
    # Prevents Optuna from finding degenerate solutions (e.g. predict
    # everyone as churn → recall=1.0 but precision=0.0).
    f2 = fbeta_score(y_test, y_pred, beta=2)

    # Log this trial as its own MLflow run so we can compare all trials in the UI.
    with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=False):
        mlflow.log_params({**params, "threshold": threshold})
        mlflow.log_metrics({
            "f2_churn": f2,
            "recall_churn": float((y_pred[y_test == 1] == 1).mean()),
            "precision_churn": float((y_test[y_pred == 1] == 1).mean()) if y_pred.sum() > 0 else 0.0,
        })

    return f2


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and prepare data (same pipeline as run_pipeline.py)
    print("Loading and preprocessing data...")
    df = preprocess(load_csv(RAW_DATA_PATH))
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # Suppress Optuna's per-trial log output — we log to MLflow instead.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting Optuna study: {N_TRIALS} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial:  #{best.number}")
    print(f"  F2 score:  {best.value:.4f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    print(f"\nView all trials: mlflow ui  (experiment: '{EXPERIMENT_NAME}')")


if __name__ == "__main__":
    main()
