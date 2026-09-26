# AI-Based Financial Fraud Detection

Machine learning pipeline that detects fraudulent credit card transactions using
classic ML models (Logistic Regression, Random Forest, XGBoost) with SHAP-based
explainability (XAI), built to accompany the portfolio site.

## Project Structure

```
fraud-detection-ml/
├── data/
│   └── generate_synthetic_data.py   # creates a synthetic dataset if you don't have the real one
├── src/
│   ├── preprocessing.py             # cleaning, scaling, train/test split
│   ├── features.py                  # feature engineering
│   ├── train.py                     # trains and saves models
│   ├── evaluate.py                  # accuracy/precision/recall/F1/ROC-AUC
│   └── explain.py                   # SHAP explainability
├── notebooks/
│   └── fraud_detection_demo.ipynb   # end-to-end walkthrough notebook
├── models/                          # trained models get saved here (.pkl)
├── reports/                         # metrics + SHAP plots get saved here
├── requirements.txt
└── README.md
```

## 1. Get the data

You have two options:

**Option A — use the real dataset (recommended for your portfolio):**
Download the popular Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`)
from https://www.kaggle.com/mlg-ulb/creditcardfraud and place it at `data/creditcard.csv`.

**Option B — generate a synthetic dataset (no download needed):**
```bash
python data/generate_synthetic_data.py
```
This creates `data/creditcard.csv` with the same column layout (Time, V1–V10, Amount, Class),
so all the other scripts work unchanged either way.

## 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the pipeline

```bash
python src/train.py       # trains Logistic Regression, Random Forest, XGBoost
python src/evaluate.py    # prints/saves comparison metrics to reports/
python src/explain.py     # generates SHAP summary plot to reports/
```

Or open the full walkthrough:
```bash
jupyter notebook notebooks/fraud_detection_demo.ipynb
```

## 4. Push this to GitHub

From inside the `fraud-detection-ml` folder:

```bash
git init
git add .
git commit -m "Initial commit: fraud detection ML pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/fraud-detection-ml.git
git push -u origin main
```

(Create an empty repo named `fraud-detection-ml` on github.com first — no README/license,
so it doesn't conflict with what you're pushing.)

Once pushed, the repo URL is what you paste into your portfolio's
"GitHub Repository" link:
```
https://github.com/<your-username>/fraud-detection-ml
```

## Results (fill in after running)

| Model               | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---------------------|----------|-----------|--------|----|---------|
| Logistic Regression |          |           |        |    |         |
| Random Forest       |          |           |        |    |         |
| XGBoost             |          |           |        |    |         |

## Notes

- The dataset is highly imbalanced (fraud is rare); the pipeline uses class
  weighting / SMOTE to handle this rather than plain accuracy as the metric of truth.
- SHAP values in `reports/shap_summary.png` show which features push a prediction
  toward "fraud" vs "legitimate" — this is what feeds the "XAI" section of the portfolio.
