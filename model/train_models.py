"""
train_models.py
----------------
Trains 6 classification models on the Breast Cancer Wisconsin (Diagnostic)
dataset, evaluates them with 6 metrics, and saves:
  - trained model pipelines (model/*.pkl)
  - a scaler-free test CSV for the Streamlit app (test_data.csv)
  - a metrics comparison table (model/metrics_summary.csv)

Dataset source: UCI Machine Learning Repository /
                 sklearn.datasets.load_breast_cancer
                 (569 instances, 30 numeric features, binary target)
"""

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

# ---------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
X = data.data.copy()
y = data.target.copy()  # 0 = malignant, 1 = benign

feature_names = list(X.columns)
target_names = list(data.target_names)  # ['malignant', 'benign']

print(f"Dataset shape: {X.shape}, classes: {target_names}")
assert X.shape[1] >= 12, "Need at least 12 features"
assert X.shape[0] >= 500, "Need at least 500 instances"

# ---------------------------------------------------------------------
# 2. Train / test split (stratified)
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Save the held-out test data (features + true label) for the Streamlit app
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv("/home/claude/project/test_data.csv", index=False)
print("Saved test_data.csv:", test_df.shape)

# ---------------------------------------------------------------------
# 3. Define models (each wrapped in a Pipeline with scaling, since
#    KNN / LogReg / SVM are scale-sensitive; scaling doesn't hurt trees)
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
    ]),
    "Decision Tree": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
    ]),
    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=7)),
    ]),
    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ]),
    "Random Forest (Ensemble)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        )),
    ]),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ]),
}

# ---------------------------------------------------------------------
# 4. Train, evaluate, save
# ---------------------------------------------------------------------
results = []

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    metrics = {
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }
    results.append(metrics)
    print(metrics)

    fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    with open(f"/home/claude/project/model/{fname}.pkl", "wb") as f:
        pickle.dump(pipe, f)

# ---------------------------------------------------------------------
# 5. Save comparison table + metadata for the Streamlit app
# ---------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv("/home/claude/project/model/metrics_summary.csv", index=False)
print("\nComparison table:\n", results_df.to_string(index=False))

meta = {"feature_names": feature_names, "target_names": target_names}
with open("/home/claude/project/model/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nAll models trained and saved.")
