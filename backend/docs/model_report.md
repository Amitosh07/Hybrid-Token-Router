# Supervised Router Model Report (Phase 6)

## Best Model

- Best model: `XGBoost`
- Estimator: `XGBClassifier`
- Validation accuracy: 0.6620
- Validation precision: 0.1361
- Validation recall: 1.0000
- Validation F1: 0.2396
- Validation ROC AUC: 0.8653
- Optimized threshold: `0.02`

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean | Threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6848 | 0.1360 | 0.9184 | 0.2368 | 0.8375 | 0.0000 | 0.04 |
| Random Forest | 0.7141 | 0.1433 | 0.8776 | 0.2464 | 0.8608 | 0.0453 | 0.03 |
| Linear SVM | 0.7272 | 0.1493 | 0.8776 | 0.2552 | 0.8290 | 0.0000 | 0.05 |
| Gradient Boosting | 0.6728 | 0.1337 | 0.9388 | 0.2341 | 0.8613 | 0.0098 | 0.03 |
| XGBoost | 0.6620 | 0.1361 | 1.0000 | 0.2396 | 0.8653 | 0.0184 | 0.02 |
| LightGBM | 0.6761 | 0.1349 | 0.9388 | 0.2359 | 0.8480 | 0.0381 | 0.03 |
| CatBoost | 0.7674 | 0.1660 | 0.8367 | 0.2770 | 0.8710 | 0.0277 | 0.05 |
