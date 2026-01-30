"""
Retrain Traditional ML Models (XGBoost and Random Forest)
with current 60-feature set for compatibility
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib

# Add parent to path
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.feature_extraction import extract_features
from src.utils.config import Config

print("=" * 70)
print("Retraining Traditional ML Models with Current Feature Set")
print("=" * 70)

# Load dataset
print("\nLoading dataset...")
data = pd.read_csv(Config.DATASET_DATNOU)
print(f"Loaded {len(data)} URLs")

# Extract features
print("\nExtracting features (this may take a while)...")
feature_data = []
skipped = 0

for idx, row in data.iterrows():
    if idx % 10000 == 0:
        print(f"Processed {idx}/{len(data)} URLs... (Skipped: {skipped})")
    
    try:
        features = extract_features(row['url'])
        
        if features is None:
            skipped += 1
            continue
        
        features['label'] = 1 if row['type'] == 'phishing' else 0
        feature_data.append(features)
    
    except Exception:
        skipped += 1
        continue

print(f"\nSuccessfully extracted features from {len(feature_data)} URLs")
print(f"Skipped {skipped} invalid URLs")

# Create DataFrame
features_df = pd.DataFrame(feature_data)

# Handle missing/infinite values
features_df.replace([np.inf, -np.inf], np.nan, inplace=True)
features_df.fillna(0, inplace=True)

# Split features and labels
X = features_df.drop('label', axis=1)
y = features_df['label']

print(f"\nFeature shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=Config.RANDOM_STATE, stratify=y
)

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
scaler_path = Config.MODEL_PRODUCTION_DIR / "scaler.pkl"
joblib.dump(scaler, scaler_path)
print(f"Scaler saved to {scaler_path}")

# Train XGBoost
print("\n" + "=" * 70)
print("Training XGBoost Model")
print("=" * 70)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=Config.RANDOM_STATE,
    n_jobs=-1
)

print("Training XGBoost...")
xgb_model.fit(X_train_scaled, y_train)

# Evaluate XGBoost
y_pred_xgb = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)

print(f"\nXGBoost Results:")
print(f"Accuracy:  {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall:    {xgb_recall:.4f}")
print(f"F1-Score:  {xgb_f1:.4f}")

# Save XGBoost
xgb_path = Config.MODEL_PRODUCTION_DIR / "best_xgboost.joblib"
joblib.dump(xgb_model, xgb_path)
print(f"XGBoost model saved to {xgb_path}")

# Train Random Forest
print("\n" + "=" * 70)
print("Training Random Forest Model")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=Config.RANDOM_STATE,
    n_jobs=-1
)

print("Training Random Forest...")
rf_model.fit(X_train_scaled, y_train)

# Evaluate Random Forest
y_pred_rf = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)

print(f"\nRandom Forest Results:")
print(f"Accuracy:  {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall:    {rf_recall:.4f}")
print(f"F1-Score:  {rf_f1:.4f}")

# Save Random Forest
rf_path = Config.MODEL_PRODUCTION_DIR / "best_random_forest.joblib"
joblib.dump(rf_model, rf_path)
print(f"Random Forest model saved to {rf_path}")

# Final summary
print("\n" + "=" * 70)
print("Training Complete!")
print("=" * 70)
print(f"\nModels saved to: {Config.MODEL_PRODUCTION_DIR}")
print(f"\nXGBoost:        Accuracy {xgb_accuracy:.2%}")
print(f"Random Forest:  Accuracy {rf_accuracy:.2%}")
print(f"\nAll models are now compatible with the current 60-feature set!")
