# Supervised Router Model Report (Phase 6)

## Best Model

- Best model: `Gradient Boosting`
- Estimator: `GradientBoostingClassifier`
- Validation accuracy: 1.0000
- Validation precision: 1.0000
- Validation recall: 1.0000
- Validation F1: 1.0000
- Validation ROC AUC: 1.0000
- Optimized threshold: `0.01`

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9971 | 0.9976 | 0.9968 | 0.9972 | 0.9999 | 0.9977 | 0.34 |
| Random Forest | 0.9992 | 1.0000 | 0.9984 | 0.9992 | 1.0000 | 0.9991 | 0.63 |
| Linear SVM | 0.9979 | 0.9992 | 0.9968 | 0.9980 | 0.9999 | 0.9979 | 0.36 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.01 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.01 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.01 |
| CatBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.01 |
