"""Production-oriented prediction module for the supervised ML router."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from app.ml.model_utils import (
        FEATURE_COLUMNS_PATH,
        METADATA_PATH,
        MODEL_PATH,
        REMOTE_LABEL,
        load_artifact,
        load_json,
        numeric_to_provider,
    )
    from app.ml.preprocess import align_features
    from app.services.feature_extractor import extract_features
except ModuleNotFoundError:  # pragma: no cover
    from model_utils import FEATURE_COLUMNS_PATH, METADATA_PATH, MODEL_PATH, REMOTE_LABEL, load_artifact, load_json, numeric_to_provider
    from preprocess import align_features
    from app.services.feature_extractor import extract_features


def load_router_model(model_path: Path | str = MODEL_PATH) -> Any:
    """Load the trained router model pipeline."""
    return load_artifact(Path(model_path))


def predict_prompt(prompt: str, model: Any | None = None) -> dict[str, Any]:
    """Extract prompt features and predict LOCAL or REMOTE."""
    features = extract_features(prompt)
    return predict_from_features(features, model=model)


def predict_from_features(features: dict[str, Any], model: Any | None = None) -> dict[str, Any]:
    """Predict routing provider from an already extracted feature dictionary."""
    estimator = model or load_router_model()
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    X = align_features(features, feature_columns)

    pred_numeric = int(estimator.predict(X)[0])
    provider = numeric_to_provider(pred_numeric)

    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(X)[0]
        remote_probability = float(probabilities[1])
    else:
        remote_probability = float(pred_numeric)

    provider_probability = remote_probability if provider == REMOTE_LABEL else 1.0 - remote_probability
    return {
        "predicted_provider": provider,
        "probability": round(provider_probability, 6),
        "probability_local": round(1.0 - remote_probability, 6),
        "probability_remote": round(remote_probability, 6),
        "confidence": round(max(remote_probability, 1.0 - remote_probability), 6),
        "feature_contributions": feature_contributions(estimator, X),
    }


def feature_contributions(model: Any, X: Any, top_n: int = 15) -> list[dict[str, Any]]:
    """Return approximate feature contributions when supported by the model."""
    try:
        classifier = model.named_steps.get("classifier")
        preprocessor = model.named_steps.get("preprocessor")
    except AttributeError:
        return []

    try:
        transformed = preprocessor.transform(X)
        names = preprocessor.get_feature_names_out()
    except Exception:
        return []

    if hasattr(classifier, "coef_"):
        weights = classifier.coef_[0]
        contrib = np.asarray(transformed)[0] * weights
        ranked = sorted(zip(names, contrib), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        return [{"feature": str(name), "contribution": round(float(value), 6)} for name, value in ranked]

    if hasattr(classifier, "feature_importances_"):
        ranked = sorted(zip(names, classifier.feature_importances_), key=lambda item: item[1], reverse=True)[:top_n]
        return [{"feature": str(name), "importance": round(float(value), 6)} for name, value in ranked]

    return []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict LOCAL or REMOTE for a prompt.")
    parser.add_argument("prompt", help="Prompt to route")
    args = parser.parse_args()
    print(predict_prompt(args.prompt))
