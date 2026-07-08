"""Prediction module for embedding-based and hybrid-based routing models.

Integrates embedding inference and tabular feature pre-processing for real-time classification.
"""

from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import Any
import numpy as np
import pandas as pd

from app.ml.preprocess import align_features
from app.ml.model_utils import load_artifact, load_json, numeric_to_provider
from app.services.feature_extractor import extract_features
from app.services.router import route as heuristic_route
from app.embedding_router.embedding_utils import (
    EMBEDDING_MODEL_PATH,
    EMBEDDING_METADATA_PATH,
    HYBRID_MODEL_PATH,
    HYBRID_METADATA_PATH,
)
from app.embedding_router.embedding_extractor import EmbeddingExtractor

logger = logging.getLogger(__name__)

ROUTING_METHOD_EMBEDDING = "Embedding"
ROUTING_METHOD_HYBRID = "Hybrid"
ROUTING_METHOD_FALLBACK = "Heuristic Fallback"


@lru_cache(maxsize=1)
def _load_embedding_bundle() -> dict[str, Any]:
    """Load pure embedding router model artifacts once."""
    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(f"Embedding model artifact not found at {EMBEDDING_MODEL_PATH}. Run training first.")
    bundle = load_artifact(EMBEDDING_MODEL_PATH)
    metadata = load_json(EMBEDDING_METADATA_PATH) if EMBEDDING_METADATA_PATH.exists() else {}
    return {
        "model": bundle["model"],
        "embedding_model_name": bundle["embedding_model_name"],
        "embedding_model_version": bundle["embedding_model_version"],
        "classifier_name": metadata.get("best_classifier", "Classifier"),
    }


@lru_cache(maxsize=1)
def _load_hybrid_bundle() -> dict[str, Any]:
    """Load hybrid router model artifacts once."""
    if not HYBRID_MODEL_PATH.exists():
        raise FileNotFoundError(f"Hybrid model artifact not found at {HYBRID_MODEL_PATH}. Run training first.")
    bundle = load_artifact(HYBRID_MODEL_PATH)
    metadata = load_json(HYBRID_METADATA_PATH) if HYBRID_METADATA_PATH.exists() else {}
    return {
        "model": bundle["model"],
        "embedding_model_name": bundle["embedding_model_name"],
        "embedding_model_version": bundle["embedding_model_version"],
        "preprocessor": bundle["preprocessor"],
        "feature_columns": bundle["feature_columns"],
        "classifier_name": metadata.get("best_classifier", "Classifier"),
    }


def clear_predict_cache() -> None:
    """Clear cached model bundles."""
    _load_embedding_bundle.cache_clear()
    _load_hybrid_bundle.cache_clear()


def route_embedding(prompt: str, features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route a prompt using pure semantic embeddings."""
    try:
        bundle = _load_embedding_bundle()
        model = bundle["model"]
        
        # Extract embedding for the prompt (will use cache if already computed)
        extractor = EmbeddingExtractor(model_name=bundle["embedding_model_name"])
        embedding, _ = extractor.extract([prompt])
        
        # Predict
        probabilities = model.predict_proba(embedding)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        remote_probability = float(probabilities[1])
        
        # Load optimized threshold
        threshold = getattr(model, "threshold", 0.5)
        prediction = 1 if remote_probability >= threshold else 0
        
        provider = "remote" if prediction == 1 else "local"
        selected_probability = remote_probability if provider == "remote" else 1.0 - remote_probability
        confidence = max(remote_probability, 1.0 - remote_probability)
        
        version_str = f"emb-{bundle['classifier_name']}-{bundle['embedding_model_name']}"
        
        return {
            "provider": provider,
            "selected_provider": provider,
            "prediction_probability": round(selected_probability, 6),
            "prediction_confidence": round(confidence, 6),
            "confidence": round(confidence, 6),
            "model_version": version_str,
            "routing_method": ROUTING_METHOD_EMBEDDING,
            "reason": [
                f"Embedding router ({bundle['classifier_name']}) selected {provider.upper()} with probability {selected_probability:.4f}.",
            ],
            "routing_score": 0,
            "feature_contributions": [],
            "fallback_error": "",
        }
        
    except Exception as exc:
        logger.warning("Embedding router unavailable; using heuristic fallback: %s", exc)
        return _fallback_to_heuristic(prompt, features, exc)


def route_hybrid(prompt: str, features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Route a prompt using combined semantic embeddings and handcrafted features."""
    try:
        bundle = _load_hybrid_bundle()
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        feature_columns = bundle["feature_columns"]
        
        # Extract embedding
        extractor = EmbeddingExtractor(model_name=bundle["embedding_model_name"])
        embedding, _ = extractor.extract([prompt])
        
        # Get handcrafted features
        feat_dict = features or extract_features(prompt)
        
        # Align and scale handcrafted features
        X_ml = align_features(feat_dict, feature_columns)
        X_ml_scaled = preprocessor.transform(X_ml)
        if hasattr(X_ml_scaled, "toarray"):
            X_ml_scaled = X_ml_scaled.toarray()
        elif not isinstance(X_ml_scaled, np.ndarray):
            X_ml_scaled = np.array(X_ml_scaled)
            
        # Combine
        X_hybrid = np.hstack([embedding, X_ml_scaled])
        
        # Predict
        probabilities = model.predict_proba(X_hybrid)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        remote_probability = float(probabilities[1])
        
        # Load optimized threshold
        threshold = getattr(model, "threshold", 0.5)
        prediction = 1 if remote_probability >= threshold else 0
        
        provider = "remote" if prediction == 1 else "local"
        selected_probability = remote_probability if provider == "remote" else 1.0 - remote_probability
        confidence = max(remote_probability, 1.0 - remote_probability)
        
        version_str = f"hybrid-{bundle['classifier_name']}-{bundle['embedding_model_name']}"
        
        return {
            "provider": provider,
            "selected_provider": provider,
            "prediction_probability": round(selected_probability, 6),
            "prediction_confidence": round(confidence, 6),
            "confidence": round(confidence, 6),
            "model_version": version_str,
            "routing_method": ROUTING_METHOD_HYBRID,
            "reason": [
                f"Hybrid router ({bundle['classifier_name']}) selected {provider.upper()} with probability {selected_probability:.4f}.",
            ],
            "routing_score": 0,
            "feature_contributions": [],
            "fallback_error": "",
        }
        
    except Exception as exc:
        logger.warning("Hybrid router unavailable; using heuristic fallback: %s", exc)
        return _fallback_to_heuristic(prompt, features, exc)


def _fallback_to_heuristic(prompt: str, features: dict[str, Any] | None, exc: Exception) -> dict[str, Any]:
    """Helper to route using the heuristic router as fallback."""
    feat_dict = features or extract_features(prompt)
    heuristic = heuristic_route(feat_dict)
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
