import pandas as pd
from telco_churn.config import (
    TARGET_COL, DROP_COLS, BINARY_COLS, BINARY_MAP, MULTI_CLASS_COLS, KNOWN_CATEGORIES
)


def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in BINARY_COLS if c in df.columns]
    df[cols] = df[cols].apply(lambda s: s.map(BINARY_MAP))
    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return df


def encode_multiclass_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = [c for c in MULTI_CLASS_COLS if c in df.columns]
    # Wrap each column in pd.Categorical with the full known category list so
    # pd.get_dummies always emits the same dummy columns regardless of how many
    # unique values are present in this batch (critical for single-row inference).
    for col in cols:
        df[col] = pd.Categorical(df[col], categories=KNOWN_CATEGORIES[col])
    df = pd.get_dummies(df, columns=cols, drop_first=True)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    if "TotalCharges" not in df.columns:
        return df
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)
    return df


def consolidate_no_service_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the redundant 'No internet/phone service' dummy columns.

    When a customer has no internet, every internet-service feature becomes
    'No internet service' — these are perfectly collinear. We replace them
    with a single No_internet_service flag (and similarly for phone).
    """
    df = df.copy()
    no_internet_cols = [c for c in df.columns if "No internet service" in c]
    if no_internet_cols:
        df["No_internet_service"] = (df[no_internet_cols].sum(axis=1) > 0).astype(int)
        df = df.drop(columns=no_internet_cols)

    no_phone_col = "MultipleLines_No phone service"
    if no_phone_col in df.columns:
        df = df.rename(columns={no_phone_col: "No_phone_service"})

    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = drop_id_columns(df)
    df = encode_binary_columns(df)
    df = fix_total_charges(df)
    df = encode_multiclass_columns(df)
    df = consolidate_no_service_columns(df)
    return df
