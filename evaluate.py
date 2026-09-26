"""
Evaluates each trained model on the held-out test set and writes a
comparison table to reports/metrics.csv (and prints it).
"""

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from train import train_all

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_evaluation():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    models, X_test, y_test = train_all()

    rows = []
    for name, model in models.items():
        if model is None:
            continue
        metrics = evaluate_model(model, X_test, y_test)
        row = {"model": name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}}
        rows.append(row)
        print(f"\n{name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall   : {metrics['recall']:.4f}")
        print(f"  F1 Score : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  Confusion Matrix: {metrics['confusion_matrix']}")

    results_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    out_path = os.path.join(REPORTS_DIR, "metrics.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table -> {out_path}")
    print(f"\nBest model by ROC-AUC: {results_df.iloc[0]['model']}")
    return results_df


if __name__ == "__main__":
    run_evaluation()
