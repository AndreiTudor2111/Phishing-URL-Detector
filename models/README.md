# Trained Models

This directory contains the trained machine learning models for phishing URL detection.

## Production Models

Located in `models/production/`:

### 1. best_xgboost.joblib
- **Algorithm**: XGBoost Classifier
- **Accuracy**: 93.3%
- **Precision**: 93.5%
- **Recall**: 93.1%
- **F1-Score**: 93.3%
- **Training Data**: DataNou.csv (~430k URLs)
- **Hyperparameters**: Fine-tuned via GridSearchCV
  - Learning rate: 0.1
  - Max depth: 6
  - N estimators: 200
  - Subsample: 0.8
  - Colsample_bytree: 0.8
- **Size**: ~4.8 MB
- **Recommended Use**: General-purpose phishing detection

### 2. best_random_forest.joblib
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 93.5%
- **Precision**: 93.8%
- **Recall**: 93.2%
- **F1-Score**: 93.5%
- **Training Data**: DataNou.csv (~430k URLs)
- **Hyperparameters**: Fine-tuned via GridSearchCV
  - N estimators: 200
  - Max depth: 20
  - Min samples split: 5
  - Min samples leaf: 2
- **Size**: ~223 MB
- **Recommended Use**: High-stakes detection where interpretability matters

### 3. scaler.pkl
- **Type**: StandardScaler
- **Purpose**: Feature normalization
- **Fitted on**: Training data features
- **Required**: Must be used with both models
- **Size**: ~1.3 KB

## Experimental Models

Located in `models/experimental/`:

Contains various experimental models and previous iterations:
- Decision Tree models
- Logistic Regression models
- Untuned baseline models
- Different hyperparameter configurations

These models are kept for reference and comparison but are not recommended for production use.

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|---------|
| **Web Application** | XGBoost | Faster inference, smaller size |
| **Batch Processing** | Random Forest | Better with varied data |
| **High Accuracy** | Random Forest | Slightly better F1-score |
| **Low Memory** | XGBoost | Much smaller file size |
| **Interpretability** | Random Forest | Feature importance clearer |

## Loading Models

### Python

```python
from src.models.predict import PhishingDetector

# Load XGBoost model
detector = PhishingDetector(model_type='xgboost')

# Load Random Forest model
detector = PhishingDetector(model_type='random_forest')

# Make predictions
prediction, probability = detector.predict("http://example.com")
```

### Direct Loading

```python
import joblib
from src.utils.config import Config

# Load model directly
model = joblib.load(Config.MODEL_XGBOOST)
scaler = joblib.load(Config.MODEL_SCALER)

# Use for predictions (after feature extraction)
# prediction = model.predict(scaled_features)
```

## Model Versioning

- **Version**: 1.0
- **Trained**: January 2025
- **Framework**: scikit-learn 1.x, XGBoost 2.x
- **Python**: 3.8+

## Retraining

To retrain models with new data:
1. Update datasets in `data/raw/`
2. Run training scripts in `scripts/`
3. Models will be saved to `models/production/`
4. Backup previous models to `models/experimental/`

## Performance Metrics

Detailed performance metrics and confusion matrices are available in:
- [docs/model_performance.md](../docs/model_performance.md)

## License

Models are provided under the MIT License. See project LICENSE file.
