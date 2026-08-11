# ML Assignment 2 — Classification Models & Streamlit Deployment

**Course:** M.Tech (AIML/DSE) — Machine Learning, BITS Pilani WILP
**Student:** [Your Name] | [BITS ID]

---

## a. Problem Statement

Breast cancer diagnosis is a critical, time-sensitive task where cell-nuclei
measurements from a digitized image of a fine needle aspirate (FNA) of a
breast mass can be used to distinguish **malignant** from **benign** tumors.
This project builds and compares six supervised classification models that
predict tumor diagnosis from these measurements, and deploys the best
workflow as an interactive Streamlit web app so predictions and model
performance can be explored live.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (also available via
  `sklearn.datasets.load_breast_cancer`, which mirrors the original UCI data)
- **Instances:** 569 (≥ 500 required ✅)
- **Features:** 30 numeric features (≥ 12 required ✅) — mean, standard
  error, and "worst" values of 10 real-valued cell-nucleus properties
  (radius, texture, perimeter, area, smoothness, compactness, concavity,
  concave points, symmetry, fractal dimension)
- **Target:** Binary — `0 = malignant`, `1 = benign`
- **Class balance:** 212 malignant / 357 benign (reasonably balanced)
- **Split used:** 80% train (455 rows) / 20% test (114 rows), stratified by
  class, `random_state = 42`

The held-out **test split** (114 rows, features + true `target` label) is
saved as `test_data.csv` and is what the Streamlit app uses/accepts for
evaluation.

## c. GitHub Repository Link

👉 `https://github.com/<your-username>/<your-repo-name>`
*(Replace with your actual repo link once pushed — see Deployment Steps below.)*

## d. Models Used

All 6 models below were trained on the **same** dataset and the **same**
train/test split, each wrapped in a `Pipeline(StandardScaler → Classifier)`
for consistent preprocessing.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes (Gaussian) | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9940 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| SVM (RBF)* | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

*\*The assignment brief lists 5 named models but states "all 6 ML models" —
SVM (RBF) is included as the 6th model to satisfy the stated count. If only
the 5 explicitly named models are required, ignore the SVM row.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performer alongside SVM. The dataset is largely linearly separable after scaling, so a simple linear decision boundary generalizes very well (Acc 0.983, MCC 0.962). |
| Decision Tree | Weakest model here. A single tree (even depth-limited to 5) overfits training splits and is sensitive to small perturbations, giving the lowest AUC (0.916) and MCC (0.834) of all models. |
| kNN | Very strong recall (1.00 — caught every malignant case in the test set) but slightly lower precision than Logistic Regression/SVM, meaning it produces a few more false positives. Performs well because scaled feature space has clear local neighborhoods for each class. |
| Naive Bayes | Decent but not top-tier (Acc 0.930). The conditional-independence assumption is violated here since many cell-nucleus features (e.g., radius, perimeter, area) are highly correlated, which hurts Naive Bayes more than the other models. |
| Random Forest (Ensemble) | Solid, well-balanced performance (Acc 0.947, AUC 0.994 — second-highest AUC). Averaging many trees fixes most of the single Decision Tree's overfitting, though it still trails the linear/SVM models on this particular dataset. |
| **Overall Winner for your dataset?** | **Logistic Regression** (tied with SVM) — highest Accuracy (0.983), Precision/Recall/F1 (0.986), and MCC (0.962). Given its simplicity and interpretability versus SVM at equal performance, **Logistic Regression** is the recommended model for this dataset. |

---

## Repository Structure

```
project-folder/
│-- app.py                     # Streamlit application
│-- requirements.txt           # Python dependencies
│-- README.md                  # This file
│-- test_data.csv              # Held-out test split (features + target)
│-- model/
│   │-- train_models.py        # Trains all 6 models, computes metrics, saves .pkl files
│   │-- metrics_summary.csv    # Auto-generated comparison table
│   │-- meta.json              # Feature names / target class names
│   │-- logistic_regression.pkl
│   │-- decision_tree.pkl
│   │-- knn.pkl
│   │-- naive_bayes.pkl
│   │-- random_forest_ensemble.pkl
│   │-- svm_rbf.pkl
```

## How to Reproduce

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain all models from scratch
python model/train_models.py

# 4. Run the app locally
streamlit run app.py
```

## Streamlit App Features

- 📤 **CSV upload** — upload your own test CSV (30 feature columns + optional
  `target` column), or use the bundled `test_data.csv`
- 🔽 **Model selection dropdown** — switch between all 6 trained models
- 📈 **Evaluation metrics display** — Accuracy, AUC, Precision, Recall, F1, MCC
  computed live on the uploaded data (when `target` is present)
- 🔲 **Confusion matrix** + full **classification report**
- 📊 An expandable panel showing the full 6-model comparison table from
  the original training run

## Deployment Steps (Streamlit Community Cloud)

1. Push this project folder to a **public GitHub repository**.
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign
   in with GitHub.
3. Click **"New app"** → select your repository and branch (`main`).
4. Set the main file path to `app.py`.
5. Click **Deploy**. Wait for the build to finish (watch the logs for any
   missing-dependency errors — everything needed is already pinned in
   `requirements.txt`).
6. Copy the live app URL (`https://<app-name>.streamlit.app`) for submission.

## Live Links (fill in before submission)

- **GitHub Repository:** `<paste link here>`
- **Live Streamlit App:** `<paste link here>`
