"""Centralized router dispatcher that resolves ROUTER_MODE from settings.

Supports:
- ROUTER_MODE=heuristic
- ROUTER_MODE=ml
- ROUTER_MODE=embedding
- ROUTER_MODE=hybrid
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.services.router import route as heuristic_route
from app.services.ml_router import route as ml_route
from app.embedding_router.embedding_predict import route_embedding, route_hybrid

logger = logging.getLogger(__name__)


def wrap_heuristic_result(heuristic: dict[str, Any]) -> dict[str, Any]:
    """Wraps the pure heuristic result with keys required by the chat response schema."""
    provider = heuristic["provider"]
    confidence = float(heuristic.get("confidence", 0.0))
    return {
        **heuristic,
        "selected_provider": provider,
        "prediction_probability": confidence,
        "prediction_confidence": confidence,
        "routing_confidence": heuristic["routing_confidence"],
        "local_score": heuristic.get("local_score"),
        "remote_score": heuristic.get("remote_score"),
        "model_version": "heuristic-router",
        "routing_method": "Heuristic",
        "feature_contributions": [],
        "fallback_error": "",
    }


def route(features: dict[str, Any], prompt: str | None = None) -> dict[str, Any]:
    """Dispatches the routing request based on the configured ROUTER_MODE setting."""
    try:
        settings = get_settings()
        mode = getattr(settings, "ROUTER_MODE", "ml").lower().strip()
    except Exception as e:
        logger.warning("Could not read ROUTER_MODE from settings: %s. Defaulting to 'ml'.", e)
        mode = "ml"

    logger.debug("Routing prompt using mode: %s", mode)

    # 1. Heuristic Router Mode
    if mode == "heuristic":
        res = heuristic_route(features)
        return wrap_heuristic_result(res)

    # 2. Traditional ML Router Mode
    elif mode == "ml":
        return ml_route(features)

    # 3. Embedding Router Mode
    elif mode == "embedding":
        if not prompt:
            raise ValueError("Prompt text is required for ROUTER_MODE=embedding.")
        return route_embedding(prompt, features)

    # 4. Hybrid Router Mode
    elif mode == "hybrid":
        if not prompt:
            raise ValueError("Prompt text is required for ROUTER_MODE=hybrid.")
        return route_hybrid(prompt, features)

    # Fallback to Traditional ML
    else:
        logger.warning("Unsupported ROUTER_MODE: '%s'. Falling back to Traditional ML.", mode)
        return ml_route(features)
