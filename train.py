"""
Trains Logistic Regression, Random Forest, and XGBoost models on the
(optionally SMOTE-balanced) training data, and saves each model to models/.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

from preprocessing import get_prepared_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def balance_data(X_train, y_train):
    if HAS_SMOTE:
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        return X_res, y_res
    print("imbalanced-learn not installed — training on raw imbalanced data "
          "with class_weight='balanced' instead.")
    return X_train, y_train


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    if not HAS_XGB:
        return None
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, y_train, y_test, scaler = get_prepared_data()
    X_train_bal, y_train_bal = balance_data(X_train, y_train)

    models = {}

    print("Training Logistic Regression...")
    models["logistic_regression"] = train_logistic_regression(X_train_bal, y_train_bal)

    print("Training Random Forest...")
    models["random_forest"] = train_random_forest(X_train_bal, y_train_bal)

    if HAS_XGB:
        print("Training XGBoost...")
        models["xgboost"] = train_xgboost(X_train_bal, y_train_bal)
    else:
        print("xgboost not installed — skipping (pip install xgboost to enable).")

    for name, model in models.items():
        if model is not None:
            path = os.path.join(MODELS_DIR, f"{name}.pkl")
            joblib.dump(model, path)
            print(f"Saved {name} -> {path}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    return models, X_test, y_test


if __name__ == "__main__":
    train_all()
