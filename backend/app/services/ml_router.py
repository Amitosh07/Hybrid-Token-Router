"""Live ML router integration with heuristic fallback."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.ml.model_utils import (
    FEATURE_COLUMNS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    REMOTE_LABEL,
    load_artifact,
    load_json,
)
from app.ml.preprocess import align_features
from app.ml.predict import feature_contributions
from app.services.router import route as heuristic_route
from app.services.routing_confidence import probability_confidence

logger = logging.getLogger(__name__)

ROUTING_METHOD_ML = "ML"
ROUTING_METHOD_FALLBACK = "Heuristic Fallback"


@lru_cache(maxsize=1)
def _load_bundle() -> dict[str, Any]:
    """Load trained model artifacts once per process."""
    model = load_artifact(MODEL_PATH)
    feature_columns = load_json(FEATURE_COLUMNS_PATH)
    metadata = load_json(METADATA_PATH) if METADATA_PATH.exists() else {}
    return {
        "model": model,
        "feature_columns": feature_columns,
        "model_version": metadata.get("pipeline_version", "phase_3_supervised_router_v1"),
    }


def clear_model_cache() -> None:
    """Clear cached model artifacts, useful for tests and fallback verification."""
    _load_bundle.cache_clear()


def route(features: dict[str, Any]) -> dict[str, Any]:
    """Route with the trained ML model, falling back to the heuristic router."""
    try:
        bundle = _load_bundle()
        model = bundle["model"]
        X = align_features(features, bundle["feature_columns"])
        probabilities = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        remote_probability = float(probabilities[1])
        
        # Load optimized threshold
        threshold = getattr(model, "threshold", 0.5)
        prediction = 1 if remote_probability >= threshold else 0
        
        provider = "remote" if prediction == 1 else "local"
        selected_probability = remote_probability if provider == "remote" else 1.0 - remote_probability
        confidence = max(remote_probability, 1.0 - remote_probability)
        confidence_label = probability_confidence(remote_probability, threshold)
        logger.info(
            "ML routing scores local=%.6f remote=%.6f threshold=%.6f confidence=%s",
            1.0 - remote_probability, remote_probability, threshold, confidence_label,
        )

        return {
            "provider": provider,
            "selected_provider": provider,
            "prediction_probability": round(selected_probability, 6),
            "prediction_confidence": round(confidence, 6),
            "confidence": round(confidence, 6),
            "routing_confidence": confidence_label,
            "local_score": round(1.0 - remote_probability, 6),
            "remote_score": round(remote_probability, 6),
            "model_version": bundle["model_version"],
            "routing_method": ROUTING_METHOD_ML,
            "reason": [
                f"ML router selected {provider.upper()} with probability {selected_probability:.4f}.",
            ],
            "routing_score": 0,
            "feature_contributions": feature_contributions(model, X),
            "fallback_error": "",
        }
    except Exception as exc:  # noqa: BLE001 - no user request should fail due to router artifact issues.
        logger.warning("ML router unavailable; using heuristic fallback: %s", exc)
        heuristic = heuristic_route(features)
        provider = heuristic["provider"]
        confidence = float(heuristic.get("confidence", 0.0))
        return {
            **heuristic,
            "selected_provider": provider,
            "prediction_probability": confidence,
            "prediction_confidence": confidence,
            "model_version": "heuristic-router",
            "routing_method": ROUTING_METHOD_FALLBACK,
            "feature_contributions": [],
            "fallback_error": str(exc),
        }
