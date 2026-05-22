from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parents[2]  # telco-customer-churn/
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "telco_customer_churn_data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "telco_customer_churn_data_processed.csv"
MODELS_DIR = ROOT_DIR / "models"

# --- Column definitions ---
TARGET_COL = "Churn"
DROP_COLS = ["customerID", "CustomerID", "customer_id"]

BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"
]
BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

MULTI_CLASS_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

# All known category values for each multi-class column, in alphabetical order.
# Must match training-time ordering so drop_first=True always drops the same
# reference category regardless of how many unique values appear in the batch.
KNOWN_CATEGORIES = {
    "MultipleLines":    ["No", "No phone service", "Yes"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "No internet service", "Yes"],
    "OnlineBackup":     ["No", "No internet service", "Yes"],
    "DeviceProtection": ["No", "No internet service", "Yes"],
    "TechSupport":      ["No", "No internet service", "Yes"],
    "StreamingTV":      ["No", "No internet service", "Yes"],
    "StreamingMovies":  ["No", "No internet service", "Yes"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaymentMethod":    ["Bank transfer (automatic)", "Credit card (automatic)",
                         "Electronic check", "Mailed check"],
}

# --- Modelling ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
PREDICTION_THRESHOLD = 0.3  # tuned for recall/precision balance

# --- Best model (from Optuna trial #24) ---
BEST_THRESHOLD = 0.331
BEST_LGBM_PARAMS = {
    "n_estimators": 186,
    "learning_rate": 0.01356,
    "num_leaves": 133,
    "max_depth": 6,
    "min_child_samples": 47,
    "subsample": 0.7857,
    "colsample_bytree": 0.7544,
}

# --- Saved model artefacts ---
MODEL_PATH = MODELS_DIR / "best_model.pkl"
FEATURE_COLS_PATH = MODELS_DIR / "feature_columns.json"
