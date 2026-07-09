"""Prediction-only endpoint for validating the live ML router."""

from fastapi import APIRouter

from app.schemas import ChatRequest, PredictResponse
from app.services.feature_extractor import extract_features
from app.services.ml_router import route

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(request: ChatRequest) -> PredictResponse:
    """Return routing decision metadata without calling an LLM provider."""
    features = extract_features(request.prompt)
    result = route(features)
    return PredictResponse(
        selected_provider=result["selected_provider"],
        prediction_probability=result["prediction_probability"],
        prediction_confidence=result["prediction_confidence"],
        routing_confidence=result["routing_confidence"],
        local_score=result.get("local_score"),
        remote_score=result.get("remote_score"),
        model_version=result["model_version"],
        routing_method=result["routing_method"],
        feature_contributions=result.get("feature_contributions", []),
    )
