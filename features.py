"""
Feature engineering: derives extra signals from the raw transaction data
that tend to help fraud models (transaction hour, log-amount, rolling
frequency proxy).
"""

import numpy as np
import pandas as pd


def add_time_features(df, time_col="Time"):
    """Converts raw seconds-since-start into an hour-of-day feature."""
    df = df.copy()
    if time_col in df.columns:
        seconds_in_day = 24 * 60 * 60
        df["hour_of_day"] = (df[time_col] % seconds_in_day) // 3600
    return df


def add_amount_features(df, amount_col="Amount"):
    df = df.copy()
    if amount_col in df.columns:
        df["log_amount"] = np.log1p(df[amount_col].clip(lower=0))
    return df


def add_frequency_proxy(df, window=50):
    """
    Simulates a 'recent transaction frequency' feature by counting how many
    of the previous `window` rows had a similar amount bucket. This is a
    stand-in for a real per-account rolling count, which needs an account ID
    the anonymized dataset doesn't provide.
    """
    df = df.copy()
    if "Amount" in df.columns:
        amount_bucket = pd.qcut(df["Amount"], q=10, labels=False, duplicates="drop")
        df["amount_bucket"] = amount_bucket
        df["recent_freq"] = (
            amount_bucket.rolling(window=window, min_periods=1)
            .apply(lambda x: (x == x.iloc[-1]).sum(), raw=False)
        )
    return df


def engineer_features(df):
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_frequency_proxy(df)
    return df


if __name__ == "__main__":
    from preprocessing import load_data, clean_data

    df = clean_data(load_data())
    df = engineer_features(df)
    print(df.head())
    print(f"Columns after feature engineering: {list(df.columns)}")
