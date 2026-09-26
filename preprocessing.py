"""
Data loading, cleaning, scaling, and train/test splitting.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")


def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Either download the Kaggle "
            "creditcard.csv into data/, or run "
            "`python data/generate_synthetic_data.py` to create a synthetic one."
        )
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.dropna()
    df = df.drop_duplicates()
    return df


def scale_features(df, columns_to_scale=("Time", "Amount")):
    """Scales Time and Amount columns (the raw, non-anonymized ones)."""
    scaler = StandardScaler()
    df = df.copy()
    existing_cols = [c for c in columns_to_scale if c in df.columns]
    df[existing_cols] = scaler.fit_transform(df[existing_cols])
    return df, scaler


def split_data(df, target_col="Class", test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def get_prepared_data():
    """Convenience function: load -> clean -> scale -> split."""
    df = load_data()
    df = clean_data(df)
    df, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(df)
    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = get_prepared_data()
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Fraud rate in train: {y_train.mean()*100:.3f}%")
    print(f"Fraud rate in test: {y_test.mean()*100:.3f}%")
