import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. DATA LOADING AND PREPROCESSING
# ---------------------------------------------------------
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Handle TotalCharges Data Type Issue
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['MonthlyCharges'], inplace=True)

# Customer ID Removal
df = df.drop('customerID', axis=1)

# Target Variable Encoding
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Categorical Encoding (One-Hot Encoding)
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define Features (X) and Target (y)
X = df.drop('Churn', axis=1)
y = df['Churn']

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature Scaling (Fit on training data only to prevent data leakage)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ---------------------------------------------------------
# 2. CLASS IMBALANCE TREATMENT (SMOTE)
# ---------------------------------------------------------
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# ---------------------------------------------------------
# 3 & 4. MODEL TRAINING AND HYPERPARAMETER TUNING
# ---------------------------------------------------------
print("Tuning Hyperparameters for Random Forest... Please wait.")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)
best_rf = grid_search.best_estimator_

print(f"Best Random Forest parameters: {grid_search.best_params_}")

# ---------------------------------------------------------
# 5. EXPORT THE FINAL MODEL
# ---------------------------------------------------------
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_rf, file)

print("Model saved successfully as best_model.pkl")