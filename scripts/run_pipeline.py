"""
End-to-end training pipeline with MLflow tracking.

Usage:
    python scripts/run_pipeline.py
"""
import json
import joblib
import mlflow
import mlflow.sklearn

from telco_churn.config import (
    RAW_DATA_PATH, PROCESSED_DATA_PATH, PREDICTION_THRESHOLD, TEST_SIZE, RANDOM_STATE,
    MODELS_DIR, MODEL_PATH, FEATURE_COLS_PATH, BEST_THRESHOLD,
)
from telco_churn.data.load import load_csv
from telco_churn.data.preprocess import preprocess
from telco_churn.features.build import split_features_target, split_train_test
from telco_churn.models.train import train_random_forest, train_lightgbm, train_xgboost, train_best_lightgbm
from telco_churn.models.evaluate import evaluate

EXPERIMENT_NAME = "telco-churn"

# Each entry: display name -> (train function, hyperparams to log)
MODELS = {
    "RandomForest": (
        train_random_forest,
        {"n_estimators": 300, "class_weight": "balanced"},
    ),
    "LightGBM": (
        train_lightgbm,
        {"n_estimators": 300, "learning_rate": 0.05, "class_weight": "balanced"},
    ),
    "XGBoost": (
        train_xgboost,
        {"n_estimators": 300, "learning_rate": 0.05, "class_weight": "scale_pos_weight"},
    ),
}


def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    # 1. Load
    print("Loading data...")
    df_raw = load_csv(RAW_DATA_PATH)
    print(f"  {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

    # 2. Preprocess
    print("Preprocessing...")
    df = preprocess(df_raw)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"  Saved processed data -> {PROCESSED_DATA_PATH}")

    # 3. Split
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # 4. Train & evaluate — one MLflow run per model
    for name, (train_fn, model_params) in MODELS.items():
        print(f"\n--- {name} ---")
        with mlflow.start_run(run_name=name):
            # Log all inputs so you can reproduce this run later
            mlflow.log_params({
                **model_params,
                "threshold": PREDICTION_THRESHOLD,
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
            })

            model = train_fn(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            # Log all output metrics
            mlflow.log_metrics(metrics)

            # Save the model file as an artifact
            mlflow.sklearn.log_model(model, name="model")

            print(f"  recall_churn={metrics['recall_churn']:.3f}  "
                  f"precision_churn={metrics['precision_churn']:.3f}  "
                  f"f1_churn={metrics['f1_churn']:.3f}")

    # 5. Train and save the best model for the Streamlit app
    print("\n--- Saving best model (tuned LightGBM) ---")
    MODELS_DIR.mkdir(exist_ok=True)
    best_model = train_best_lightgbm(X_train, y_train)
    metrics = evaluate(best_model, X_test, y_test, threshold=BEST_THRESHOLD)
    print(f"  recall_churn={metrics['recall_churn']:.3f}  "
          f"precision_churn={metrics['precision_churn']:.3f}  "
          f"f1_churn={metrics['f1_churn']:.3f}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"  Model saved -> {MODEL_PATH}")

    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(list(X_train.columns), f)
    print(f"  Feature columns saved -> {FEATURE_COLS_PATH}")

    print(f"\nDone. Run: mlflow ui  to view results.")


if __name__ == "__main__":
    main()
