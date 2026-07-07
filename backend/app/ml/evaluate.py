"""Evaluation helpers for supervised router models."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from app.ml.model_utils import numeric_to_provider, provider_to_numeric
except ModuleNotFoundError:  # pragma: no cover
    from model_utils import numeric_to_provider, provider_to_numeric


SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a fitted estimator on a holdout set."""
    y_true = y_test.map(provider_to_numeric)
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000.0 / max(len(X_test), 1)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        raw = model.decision_function(X_test)
        y_score = 1.0 / (1.0 + np.exp(-raw))
    else:
        y_score = y_pred

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6) if len(set(y_true)) > 1 else 0.0,
        "confusion_matrix": cm.tolist(),
        "prediction_latency_ms_per_sample": round(float(latency_ms), 6),
    }


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> dict[str, Any]:
    """Run stratified cross validation and summarize each metric."""
    y_num = y.map(provider_to_numeric)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_validate(model, X, y_num, cv=cv, scoring=SCORING, n_jobs=None, error_score="raise")
    summary: dict[str, Any] = {}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        summary[metric] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values)), 6),
            "folds": [round(float(v), 6) for v in values],
        }
    return summary


def prediction_frame(model: Any, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
    """Return predictions, probabilities, confidence, and optional actual labels."""
    pred_num = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    remote_probability = probabilities[:, 1] if probabilities is not None else pred_num.astype(float)
    local_probability = 1.0 - remote_probability
    predicted_labels = [numeric_to_provider(value) for value in pred_num]
    frame = pd.DataFrame({
        "predicted_label": predicted_labels,
        "probability_local": local_probability,
        "probability_remote": remote_probability,
        "confidence": np.maximum(local_probability, remote_probability),
    }, index=X.index)
    if y is not None:
        frame["actual_label"] = y.values
        frame["is_correct"] = frame["actual_label"] == frame["predicted_label"]
    return frame
