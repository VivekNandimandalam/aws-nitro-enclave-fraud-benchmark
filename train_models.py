import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# 1. Load the copied CSV
print("Loading creditcard.csv...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

# 2. Scale Time & Amount; leave V1-V28 (already PCA-transformed)
print("Preprocessing features...")
scaler = RobustScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

X = df.drop('Class', axis=1)
y = df['Class']

# 3. Stratified Split (80% Train, 20% Test) to preserve fraud ratio without leakage
print("Splitting dataset (80/20 stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Export the test payload for the load generator harness later
test_payload = pd.concat([X_test, y_test], axis=1)
test_payload.to_csv('models/test_payload.csv', index=False)
print("Saved models/test_payload.csv (for benchmarking load harness).")

# 4. Train Primary Model: XGBoost
print("Training Primary Model: XGBoost Classifier...")
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_model.save_model('models/xgb_fraud.json')
print("Saved models/xgb_fraud.json")

# 5. Train Baseline Control Model: Logistic Regression
print("Training Baseline Control: Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr_model.fit(X_train, y_train)
joblib.dump(lr_model, 'models/lr_fraud.joblib')
print("Saved models/lr_fraud.joblib")

print("\nAll model artifacts successfully created in D:\\Research_Project_Implementation\\models\\")