"""
train_model.py - Train the Logistic Regression churn model and export
the three files that app.py needs: model.pkl, scaler.pkl, columns.pkl

Run:  python train_model.py
Needs WA_Fn-UseC_-Telco-Customer-Churn.csv in the same folder.
"""

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline            # imblearn Pipeline applies SMOTE inside CV folds only
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ---------------------------------------------------------
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# TotalCharges has blank strings (customers with tenure 0) -> convert and fill
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])   # (no inplace: deprecated)

df = df.drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# One-hot encode all text columns (SeniorCitizen is already 0/1 so it stays numeric)
categorical_cols = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Churn", axis=1).astype(float)   # force plain floats (dummies may be bool)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 2. SCALING (fit on training data only -> no leakage)
# ---------------------------------------------------------
numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ---------------------------------------------------------
# 3. LOGISTIC REGRESSION + SMOTE, TUNED WITH STRATIFIED 5-FOLD CV
# ---------------------------------------------------------
# SMOTE sits inside the pipeline, so it only resamples the training part of each fold.
pipe = Pipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42)),
])

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10],
}

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
)
print("Tuning Logistic Regression with SMOTE... please wait.")
grid.fit(X_train_scaled, y_train)
print("Best parameters:", grid.best_params_)

# Pull out the fitted Logistic Regression (the app needs coef_ and predict_proba)
best_lr = grid.best_estimator_.named_steps["clf"]

# ---------------------------------------------------------
# 4. EVALUATION ON THE UNTOUCHED TEST SET
# ---------------------------------------------------------
y_pred = best_lr.predict(X_test_scaled)
print(f"\nTest F1-score (churn class): {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, target_names=["Stay", "Churn"]))

# ---------------------------------------------------------
# 5. EXPORT FILES FOR app.py
# ---------------------------------------------------------
joblib.dump(best_lr, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_train.columns.tolist(), "columns.pkl")   # exact training column order

print("Saved: model.pkl, scaler.pkl, columns.pkl")
