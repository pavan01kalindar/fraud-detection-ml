"""
Generates a synthetic credit-card-transaction dataset that mirrors the
structure of the well-known Kaggle "Credit Card Fraud Detection" dataset
(Time, V1-V10 anonymized features, Amount, Class), so you can run the whole
pipeline without needing to download the real data first.

Usage:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 20000
FRAUD_RATIO = 0.015  # ~1.5% fraud, deliberately imbalanced like real data

def generate_dataset(n_samples=N_SAMPLES, fraud_ratio=FRAUD_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions: features drawn from a "normal" distribution
    legit_features = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 10))
    legit_amount = np.abs(rng.normal(loc=60, scale=40, size=n_legit))
    legit_time = rng.uniform(0, 172800, size=n_legit)  # 2 days, in seconds
    legit_labels = np.zeros(n_legit)

    # Fraudulent transactions: shifted distribution + different amount pattern
    fraud_features = rng.normal(loc=2.5, scale=1.8, size=(n_fraud, 10))
    fraud_amount = np.abs(rng.normal(loc=250, scale=180, size=n_fraud))
    fraud_time = rng.uniform(0, 172800, size=n_fraud)
    fraud_labels = np.ones(n_fraud)

    features = np.vstack([legit_features, fraud_features])
    amounts = np.concatenate([legit_amount, fraud_amount])
    times = np.concatenate([legit_time, fraud_time])
    labels = np.concatenate([legit_labels, fraud_labels])

    columns = [f"V{i}" for i in range(1, 11)]
    df = pd.DataFrame(features, columns=columns)
    df.insert(0, "Time", times)
    df["Amount"] = amounts
    df["Class"] = labels.astype(int)

    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = os.path.join(os.path.dirname(__file__), "creditcard.csv")
    df.to_csv(out_path, index=False)
    print(f"Synthetic dataset written to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
