import pandas as pd
from sklearn.model_selection import train_test_split
from telco_churn.config import TARGET_COL, TEST_SIZE, RANDOM_STATE


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserves class ratio in both splits
    )
