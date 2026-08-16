# ML Assignment 2 — Classification Models & Streamlit Deployment

**Course:** M.Tech (AIML/DSE) — Machine Learning, BITS Pilani WILP  
**Student:** Biresh Kumar  
**Student ID:** 2025ac05265

---

## 1. Problem Statement

Breast cancer diagnosis is an important classification problem in which measurements computed from digitized images of fine needle aspirate (FNA) samples can be used to distinguish **malignant** and **benign** tumors.

The objective of this assignment is to:

1. Use a dataset satisfying the assignment requirements.
2. Perform supervised binary classification.
3. Train and compare multiple machine-learning classification models.
4. Evaluate the models using Accuracy, AUC, Precision, Recall, F1-Score and Matthews Correlation Coefficient (MCC).
5. Identify the strongest model based on the overall evaluation.
6. Deploy the trained models through an interactive Streamlit application.

---

## 2. Dataset Description

### Breast Cancer Wisconsin (Diagnostic) Dataset

- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository
- **Implementation source:** `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Predictor features:** 30 numeric features
- **Target:** Binary classification
- **Classes:** Malignant and Benign
- **Malignant:** 212 samples
- **Benign:** 357 samples
- **Train/Test split:** 80% / 20%
- **Random state:** 42
- **Stratification:** Yes

The 30 predictor variables represent measurements of cell-nucleus characteristics such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

Each characteristic is represented using mean, standard-error and worst-value measurements.

The held-out test set contains **114 samples** and is stored in `test_data.csv`. The file is used by the Streamlit application for evaluation.

---

## 3. Machine Learning Workflow

The overall workflow is:

```text
Dataset
   ↓
Load Breast Cancer Wisconsin Dataset
   ↓
Separate Features (X) and Target (y)
   ↓
Stratified 80:20 Train/Test Split
   ↓
StandardScaler + Classification Model
   ↓
Train on Training Data
   ↓
Predict on Held-out Test Data
   ↓
Calculate Evaluation Metrics
   ↓
Compare Six Models
   ↓
Save Trained Pipelines
   ↓
Streamlit Deployment
```

The models are implemented using scikit-learn Pipelines. `StandardScaler` is included in each pipeline so that the preprocessing used during training is automatically applied when the saved model is used for prediction.

---

## 4. Models Implemented

Six classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest
6. Support Vector Machine with RBF kernel

### Model configuration

| Model | Main Configuration |
|---|---|
| Logistic Regression | `max_iter=5000` |
| Decision Tree | `max_depth=5` |
| kNN | `n_neighbors=7` |
| Gaussian Naive Bayes | Default GaussianNB |
| Random Forest | `n_estimators=300`, `max_depth=8` |
| SVM | RBF kernel, `probability=True` |

`random_state=42` is used where applicable to improve reproducibility.

---

## 5. Evaluation Metrics

Six evaluation metrics are used.

### Accuracy

Measures the proportion of correctly classified samples.

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

### AUC

The Area Under the ROC Curve measures the ability of the classifier to distinguish between the two classes across classification thresholds.

### Precision

Measures how many predicted positive samples are actually positive.

\[
Precision = \frac{TP}{TP + FP}
\]

### Recall

Measures how many actual positive samples are correctly identified.

\[
Recall = \frac{TP}{TP + FN}
\]

### F1-Score

The harmonic mean of Precision and Recall.

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Matthews Correlation Coefficient (MCC)

MCC provides a balanced measure of binary classification quality using all four confusion-matrix components and is particularly useful when class distributions are not perfectly equal.

---

## 6. Model Performance Comparison

The following results were obtained on the held-out test set of 114 samples.

| ML Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | **0.9825** | **0.9954** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| kNN | 0.9737 | 0.9884 | 0.9600 | **1.0000** | 0.9796 | 0.9442 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest | 0.9474 | 0.9940 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |
| **SVM (RBF)** | **0.9825** | **0.9950** | **0.9861** | **0.9861** | **0.9861** | **0.9623** |

The comparison table is also stored in:

```text
model/metrics_summary.csv
```

---

## 7. Results and Observations

### 7.1 Logistic Regression

Logistic Regression is the strongest overall model in this experiment.

It achieved:

- Accuracy: **98.25%**
- AUC: **0.9954**
- Precision: **0.9861**
- Recall: **0.9861**
- F1-Score: **0.9861**
- MCC: **0.9623**

It provides the highest Accuracy, AUC, Precision, Recall, F1-Score and MCC among the evaluated models, with SVM achieving the same Accuracy, Precision, Recall, F1 and MCC and a slightly lower AUC.

### 7.2 SVM (RBF)

SVM is effectively tied with Logistic Regression on the major classification metrics.

Its AUC of **0.9950** is only slightly below Logistic Regression's **0.9954**.

Therefore, SVM is also a very strong model for this dataset.

### 7.3 kNN

kNN achieved:

- Accuracy: **97.37%**
- Recall: **100%**
- F1-Score: **0.9796**

The 100% recall means that all positive-class samples in the held-out test set were identified by the model under the selected class encoding.

Its precision is slightly lower than Logistic Regression and SVM, indicating a somewhat higher false-positive rate.

### 7.4 Random Forest

Random Forest achieved:

- Accuracy: **94.74%**
- AUC: **0.9940**
- F1-Score: **0.9583**

Its AUC is very high, showing strong ranking/discrimination ability, although its classification accuracy and F1-Score are lower than Logistic Regression, SVM and kNN on this particular test split.

### 7.5 Naive Bayes

Naive Bayes achieved:

- Accuracy: **92.98%**
- AUC: **0.9868**
- F1-Score: **0.9444**

The model performs reasonably well, but its overall classification metrics are lower than the leading models.

The dataset contains several related measurements of cell characteristics, so the conditional-independence assumption of Gaussian Naive Bayes may not represent the relationships between features particularly well.

### 7.6 Decision Tree

Decision Tree produced the lowest overall performance among the six models:

- Accuracy: **92.11%**
- AUC: **0.9163**
- F1-Score: **0.9362**
- MCC: **0.8341**

Although the tree depth was limited to 5, its performance on this held-out split was below the other evaluated approaches.

---

## 8. Overall Best Model

### Selected Model: Logistic Regression

Logistic Regression is selected as the **overall recommended model** for this experiment.

The main reasons are:

1. Highest Accuracy: **98.25%**
2. Highest AUC: **0.9954**
3. Highest Precision: **0.9861**, tied with SVM
4. Highest Recall: **0.9861**, tied with SVM
5. Highest F1-Score: **0.9861**, tied with SVM
6. Highest MCC: **0.9623**, tied with SVM
7. Simple and comparatively interpretable model
8. Excellent performance after feature standardization

SVM produces almost identical classification results, but Logistic Regression is preferred for this assignment because it provides comparable predictive performance with a simpler and more interpretable linear decision function.

> **Important:** The selection is based on the reported held-out test split. It should not be interpreted as proof that Logistic Regression will always outperform the other models on every possible dataset split.

---

## 9. Streamlit Application

A Streamlit application has been developed to provide an interactive interface for the trained models.

### Application capabilities

- Upload a CSV test dataset.
- Use the bundled `test_data.csv`.
- Select any of the six trained models.
- Generate predictions.
- Display prediction labels.
- Display Accuracy, AUC, Precision, Recall, F1-Score and MCC when ground-truth `target` values are available.
- Display the confusion matrix.
- Display the complete classification report.
- Display the six-model comparison table.

### Input format

The uploaded CSV should contain the same 30 feature columns used during model training.

A `target` column is optional:

- If `target` is present → predictions and evaluation metrics are displayed.
- If `target` is absent → predictions are displayed without ground-truth evaluation metrics.

---

## 10. Streamlit Application Link

**Live Application:**  
https://biresh2025ac05265.streamlit.app/

The application is intended to demonstrate the trained classification pipelines and their evaluation interactively.

---

## 11. GitHub Repository

**Repository:**  
https://github.com/2025ac05265/ML_Assignment_2

---

## 12. Repository Structure

```text
ML_Assignment_2/
│
├── app.py
├── README.md
├── requirements.txt
├── test_data.csv
│
└── model/
    ├── train_models.py
    ├── metrics_summary.csv
    ├── meta.json
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    └── svm_rbf.pkl
```

### File description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `test_data.csv` | Held-out test data used by the application |
| `model/train_models.py` | Training, evaluation and model serialization script |
| `model/metrics_summary.csv` | Model performance comparison |
| `model/meta.json` | Feature names and target class names |
| `model/*.pkl` | Serialized trained ML pipelines |
| `README.md` | Project and assignment documentation |

---

## 13. Installation and Execution

### Step 1 — Clone the repository

```bash
git clone https://github.com/2025ac05265/ML_Assignment_2.git
cd ML_Assignment_2
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Run the Streamlit application

```bash
streamlit run app.py
```

The application can then be opened in the browser using the local Streamlit URL displayed in the terminal.

---

## 14. Reproducing the Model Training

The training script is:

```text
model/train_models.py
```

Run:

```bash
python model/train_models.py
```

The script:

1. Loads the Breast Cancer Wisconsin dataset.
2. Performs an 80:20 stratified train/test split.
3. Creates six classification pipelines.
4. Trains each model.
5. Generates predictions and probability estimates.
6. Calculates six evaluation metrics.
7. Saves the trained pipelines.
8. Generates `metrics_summary.csv`.
9. Generates `meta.json`.
10. Generates the held-out test data used by the Streamlit application.

---

## 15. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

---

## 16. Conclusion

The experiment demonstrates that multiple supervised classification algorithms can achieve strong performance on the Breast Cancer Wisconsin (Diagnostic) dataset.

Among the evaluated models, **Logistic Regression and SVM (RBF)** provide the best overall results. Logistic Regression is selected as the recommended model because it achieves the highest AUC and matches SVM on Accuracy, Precision, Recall, F1-Score and MCC while offering a simpler and more interpretable model.

The Streamlit application extends the assignment from offline model evaluation to an interactive prediction and evaluation workflow, allowing users to upload test data, select a trained classifier and inspect the resulting predictions and performance metrics.

---

## 17. Assignment Deliverables Checklist

- [x] Dataset with at least 500 instances
- [x] At least 12 predictor features
- [x] Binary classification problem
- [x] Six classification models implemented
- [x] Train/test split with stratification
- [x] Feature scaling through pipelines
- [x] Accuracy evaluation
- [x] AUC evaluation
- [x] Precision evaluation
- [x] Recall evaluation
- [x] F1-Score evaluation
- [x] MCC evaluation
- [x] Model performance comparison
- [x] Model observations and conclusion
- [x] Trained model artifacts
- [x] Streamlit application
- [x] Live deployment
- [x] README documentation

---

## 18. Author

**Biresh Kumar**  
**BITS Pilani WILP — M.Tech (AIML/DSE)**  
**Student ID:** 2025ac05265
