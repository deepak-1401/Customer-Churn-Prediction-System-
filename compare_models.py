"""
compare_models.py - Compare 4 churn models on the same train/test split.

Models: Logistic Regression, Random Forest, XGBoost, Gradient Boosting
Method: SMOTE inside stratified 5-fold CV (imblearn Pipeline), F1 as tuning metric.

Outputs (saved in ./results/):
    model_comparison.csv      -> metrics table (use it in your README / report)
    confusion_matrices.png    -> 2x2 grid, one matrix per model
    roc_curves.png            -> ROC curves for all models
    pr_curves.png             -> precision-recall curves for all models
    best_model.pkl            -> best model by test F1 (does NOT overwrite model.pkl)

Run:  python compare_models.py
Needs: WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder.
Install: pip install pandas scikit-learn imbalanced-learn xgboost matplotlib joblib
"""

import os
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")                     # save figures without opening windows
import matplotlib.pyplot as plt
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline    # applies SMOTE only to training folds
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
os.makedirs("results", exist_ok=True)
SEED = 42

# ---------------------------------------------------------
# 1. DATA PREPARATION (identical to train_model.py)
# ---------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])
df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Churn", axis=1).astype(float)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# Scale numeric columns (fit on training data only)
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()
X_train_s, X_test_s = X_train.copy(), X_test.copy()
X_train_s[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_s[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# 2. MODELS AND SMALL TUNING GRIDS
# ---------------------------------------------------------
models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, random_state=SEED),
        {"clf__C": [0.001, 0.005, 0.01, 0.1, 1, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [200, 300],
         "clf__max_depth": [10, 15],
         "clf__min_samples_leaf": [2, 4]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1),
        {"clf__n_estimators": [200],
         "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [3, 5]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=SEED),
        {"clf__n_estimators": [100, 200],
         "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [3, 5]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ---------------------------------------------------------
# 3. TRAIN, TUNE AND EVALUATE EACH MODEL
# ---------------------------------------------------------
rows, fitted, probas = [], {}, {}

for name, (estimator, grid) in models.items():
    print(f"\n>>> Tuning {name} ...")
    pipe = Pipeline([("smote", SMOTE(random_state=SEED)), ("clf", estimator)])
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1 if name == "Logistic Regression" else 1)
    search.fit(X_train_s, y_train)

    best = search.best_estimator_.named_steps["clf"]    # the fitted classifier itself
    y_pred = best.predict(X_test_s)
    y_prob = best.predict_proba(X_test_s)[:, 1]

    fitted[name], probas[name] = best, (y_pred, y_prob)
    rows.append({
        "Model": name,
        "CV F1": round(search.best_score_, 3),
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1": round(f1_score(y_test, y_pred), 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3),
        "Best Params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    })
    print(f"    test F1 = {rows[-1]['F1']}  recall = {rows[-1]['Recall']}  AUC = {rows[-1]['ROC-AUC']}")

results = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
results.to_csv("results/model_comparison.csv", index=False)
print("\n================ MODEL COMPARISON (test set) ================")
print(results.drop(columns="Best Params").to_string(index=False))

# ---------------------------------------------------------
# 4. PLOTS
# ---------------------------------------------------------
# 4a. Confusion matrices (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
for ax, (name, (y_pred, _)) in zip(axes.ravel(), probas.items()):
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Stay", "Churn"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(name)
fig.suptitle("Confusion Matrices (test set, threshold = 0.5)", fontsize=14)
fig.tight_layout()
fig.savefig("results/confusion_matrices.png", dpi=150)
plt.close(fig)

# 4b. ROC curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, (_, y_prob) in probas.items():
    RocCurveDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig("results/roc_curves.png", dpi=150)
plt.close(fig)

# 4c. Precision-recall curves
fig, ax = plt.subplots(figsize=(7, 6))
for name, (_, y_prob) in probas.items():
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, name=name, ax=ax)
ax.set_title("Precision-Recall Curves")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig("results/pr_curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------
# 5. SAVE THE BEST MODEL (by test F1)
# ---------------------------------------------------------
best_name = results.loc[0, "Model"]
joblib.dump(fitted[best_name], "results/best_model.pkl")
print(f"\nBest model by F1: {best_name}  -> saved to results/best_model.pkl")
print("Saved: model_comparison.csv, confusion_matrices.png, roc_curves.png, pr_curves.png")
if best_name != "Logistic Regression":
    print("Note: app.py's 'driver' section uses coef_ and only works with Logistic Regression.")
