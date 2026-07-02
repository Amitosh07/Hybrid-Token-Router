import logging
import time
import uuid

from fastapi import APIRouter
from fastapi import HTTPException

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse
from app.services import ollama
from app.services import remote_llm
from app.services.feature_extractor import extract_features
from app.services.router import route

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    settings = get_settings()
    start_time = time.perf_counter()
    prompt_id = str(uuid.uuid4())

    features = extract_features(request.prompt)
    route_result = route(features)
    provider = route_result["provider"]

    try:
        if provider == "local":
            try:
                answer = await ollama.generate(request.prompt)
            except Exception as e:
                logger.warning(f"Prompt ID: {prompt_id} | Local model failed ({e}), falling back to remote.")
                answer = await remote_llm.generate(request.prompt)
                provider = "remote"
        else:
            answer = await remote_llm.generate(request.prompt)
            
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info(f"Prompt ID: {prompt_id} | Provider chosen: {provider} | Confidence: {route_result['confidence']} | Latency: {latency}ms | Success")

        return ChatResponse(
            answer=answer,
            provider=provider,
            routing_score=route_result["routing_score"],
            confidence=route_result["confidence"],
            reason=route_result["reason"],
            latency=latency,
            features=features,
        )
    except ValueError as exc:
        logger.error(f"Prompt ID: {prompt_id} | Provider chosen: {provider} | Failure: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConnectionError as exc:
        logger.error(f"Prompt ID: {prompt_id} | Provider chosen: {provider} | Failure: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        logger.error(f"Prompt ID: {prompt_id} | Provider chosen: {provider} | Failure: {exc}")
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"Prompt ID: {prompt_id} | Provider chosen: {provider} | Failure: {exc}")
        status_code = 429 if "rate limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
