"""
Generates SHAP explanations for the best tree-based model, saving a summary
plot to reports/shap_summary.png. This is the "XAI" piece referenced in the
portfolio — it shows which features push a prediction toward fraud vs
legitimate, and by how much.
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from preprocessing import get_prepared_data

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_best_tree_model():
    """Prefers XGBoost, falls back to Random Forest — both work with SHAP's TreeExplainer."""
    for name in ("xgboost", "random_forest"):
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            return name, joblib.load(path)
    raise FileNotFoundError(
        "No trained tree-based model found. Run `python src/train.py` first."
    )


def explain_model():
    if not HAS_SHAP:
        print("shap is not installed. Run `pip install shap` to enable this step.")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)
    name, model = load_best_tree_model()
    _, X_test, _, y_test, _ = get_prepared_data()

    # Use a sample for speed on larger datasets
    sample = X_test.sample(min(1000, len(X_test)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    out_path = os.path.join(REPORTS_DIR, "shap_summary.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"SHAP summary plot for '{name}' saved -> {out_path}")


if __name__ == "__main__":
    explain_model()
