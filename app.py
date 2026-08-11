"""
Streamlit App - Breast Cancer Classification Demo
--------------------------------------------------
Loads 6 pre-trained classification models (trained in model/train_models.py)
and lets the user:
  1. Upload a CSV of test data (features + 'target' column)
  2. Select which model to evaluate
  3. View evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  4. View the confusion matrix and full classification report
"""

import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

st.set_page_config(page_title="ML Classifier Comparison App", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "kNN": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest (Ensemble)": "model/random_forest_ensemble.pkl",
    "SVM (RBF)": "model/svm_rbf.pkl",
}


@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta():
    with open("model/meta.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics_summary():
    return pd.read_csv("model/metrics_summary.csv")


meta = load_meta()
FEATURE_NAMES = meta["feature_names"]
TARGET_NAMES = meta["target_names"]  # ['malignant', 'benign']

st.title("🩺 Breast Cancer Classification — Model Comparison App")
st.markdown(
    """
This app demonstrates **6 classification models** trained on the
**Breast Cancer Wisconsin (Diagnostic) dataset** (569 instances, 30 numeric
features, binary target — sourced from the UCI ML Repository).

Upload a test CSV (same 30 features + a `target` column), pick a model,
and see how it performs.
"""
)

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test data (CSV with features + 'target' column)", type=["csv"]
)

selected_model_name = st.sidebar.selectbox(
    "Select a model to evaluate", list(MODEL_FILES.keys())
)

use_sample = st.sidebar.checkbox(
    "No file? Use bundled test_data.csv instead", value=True
)

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded uploaded file: {df.shape[0]} rows")
elif use_sample:
    try:
        df = pd.read_csv("test_data.csv")
        st.sidebar.info(f"Using bundled test_data.csv: {df.shape[0]} rows")
    except FileNotFoundError:
        df = None
        st.sidebar.error("test_data.csv not found in repo.")
else:
    df = None

# ---------------------------------------------------------------------
# Overall comparison table (always visible)
# ---------------------------------------------------------------------
with st.expander("📊 View comparison table for ALL 6 models (from training run)"):
    st.dataframe(load_metrics_summary(), use_container_width=True)

# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
if df is None:
    st.warning("Upload a CSV file (or tick 'use bundled test_data.csv') to continue.")
    st.stop()

missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required feature columns: {missing_cols}")
    st.stop()

has_target = "target" in df.columns

X = df[FEATURE_NAMES]
model = load_model(MODEL_FILES[selected_model_name])

y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]

st.subheader(f"Results — {selected_model_name}")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("**Predictions preview**")
    preview = df.copy()
    preview["prediction"] = y_pred
    preview["prediction_label"] = [TARGET_NAMES[p] for p in y_pred]
    st.dataframe(preview.head(20), use_container_width=True)

with col2:
    if has_target:
        y_true = df["target"]
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "AUC": roc_auc_score(y_true, y_proba),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1 Score": f1_score(y_true, y_pred),
            "MCC": matthews_corrcoef(y_true, y_pred),
        }
        st.markdown("**Evaluation metrics**")
        metrics_df = pd.DataFrame(
            {"Metric": metrics.keys(), "Value": [round(v, 4) for v in metrics.values()]}
        )
        st.table(metrics_df)
    else:
        st.info(
            "Uploaded CSV has no 'target' column — showing predictions only "
            "(metrics require ground-truth labels)."
        )

if has_target:
    st.markdown("---")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    with c2:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=TARGET_NAMES, output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.markdown("---")
st.caption(
    "Built for BITS Pilani WILP M.Tech (AIML/DSE) — Machine Learning Assignment 2. "
    "Dataset: Breast Cancer Wisconsin (Diagnostic), UCI ML Repository."
)
