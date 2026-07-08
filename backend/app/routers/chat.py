"""Chat endpoint with full pipeline latency instrumentation.

Stage timestamps (all via time.perf_counter()):

  T0  Request received by FastAPI handler
  T1  Feature extraction completed          → feature_extraction_ms
  T2  Router decision completed             → routing_decision_ms
  T3  Provider call initiated
  T4  Provider response received            → provider_ms  (T4 - T3)
  T5  (same as T4 for non-streaming; kept for future streaming support)
  T6  Response serialised and returned      → total_ms     (T6 - T0)

All stage timings are written to the DEBUG log so they are available
in development without polluting INFO output in production.
"""

import logging
import time
import uuid

from fastapi import APIRouter, HTTPException

from app.schemas import ChatRequest, ChatResponse
from app.services import ollama
from app.services import remote_llm
from app.services.feature_extractor import extract_features
from app.services.router_dispatcher import route
from app.services.stats_tracker import stats as stats_tracker

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # ── T0: request received ───────────────────────────────────────────
    t0 = time.perf_counter()
    prompt_id = str(uuid.uuid4())
    fallback_used = False

    # ── T1: feature extraction ─────────────────────────────────────────
    features = extract_features(request.prompt)
    t1 = time.perf_counter()

    # ── T2: routing decision ───────────────────────────────────────────
    route_result = route(features, prompt=request.prompt)
    provider = route_result["provider"]
    t2 = time.perf_counter()

    feature_extraction_ms = round((t1 - t0) * 1000, 3)
    routing_decision_ms   = round((t2 - t1) * 1000, 3)

    logger.debug(
        f"[{prompt_id}] Pipeline stage timings | "
        f"feature_extraction={feature_extraction_ms}ms | "
        f"routing_decision={routing_decision_ms}ms | "
        f"provider_selected={provider}"
    )

    # ── T3 / T4: provider call ─────────────────────────────────────────
    try:
        t3 = time.perf_counter()

        if provider == "local":
            try:
                answer = await ollama.generate(request.prompt)
            except TimeoutError:
                # Single timeout → fall back immediately. A retry would add
                # another 20 s for a model that has already demonstrated it
                # is not responding within budget.
                logger.warning(
                    f"[{prompt_id}] Local model timed out — falling back to remote."
                )
                answer = await remote_llm.generate(request.prompt)
                provider = "remote"
                fallback_used = True
            except Exception as e:
                logger.warning(
                    f"[{prompt_id}] Local model failed ({e}) — falling back to remote."
                )
                answer = await remote_llm.generate(request.prompt)
                provider = "remote"
                fallback_used = True
        else:
            answer = await remote_llm.generate(request.prompt)

        t4 = time.perf_counter()

        # ── T6: response ready (T5 == T4 for non-streaming) ───────────
        provider_ms = round((t4 - t3) * 1000, 2)
        total_ms    = round((t4 - t0) * 1000, 2)

        logger.debug(
            f"[{prompt_id}] Pipeline stage timings | "
            f"provider_wait={provider_ms}ms | "
            f"total_request={total_ms}ms | "
            f"actual_provider={provider} | fallback={fallback_used}"
        )

        logger.info(
            f"[{prompt_id}] Provider={provider} | "
            f"Confidence={route_result['prediction_confidence']} | "
            f"RoutingMethod={route_result['routing_method']} | "
            f"ProviderMs={provider_ms}ms | TotalMs={total_ms}ms | "
            f"Fallback={fallback_used}"
        )

        estimated_input_tokens = features.get("estimated_input_tokens", 0)

        # Record to in-memory stats — uses total_ms as the authoritative latency
        stats_tracker.record(
            provider=provider,
            latency_ms=total_ms,
            confidence=route_result["prediction_confidence"],
            estimated_input_tokens=estimated_input_tokens,
            fallback_used=fallback_used,
            routing_method=route_result["routing_method"],
        )

        return ChatResponse(
            answer=answer,
            provider=provider,
            selected_provider=provider,
            routing_score=route_result["routing_score"],
            confidence=route_result["prediction_confidence"],
            reason=route_result["reason"],
            latency=total_ms,
            features=features,
            task_type=features.get("task_type", ""),
            complexity=features.get("complexity", ""),
            estimated_input_tokens=estimated_input_tokens,
            fallback_used=fallback_used,
            prompt_id=prompt_id,
            prediction_probability=route_result["prediction_probability"],
            prediction_confidence=route_result["prediction_confidence"],
            model_version=route_result["model_version"],
            routing_method=route_result["routing_method"],
        )

    except ValueError as exc:
        logger.error(f"[{prompt_id}] Provider={provider} | ValueError: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConnectionError as exc:
        logger.error(f"[{prompt_id}] Provider={provider} | ConnectionError: {exc}")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        logger.error(f"[{prompt_id}] Provider={provider} | TimeoutError: {exc}")
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error(f"[{prompt_id}] Provider={provider} | RuntimeError: {exc}")
        status_code = 429 if "rate limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
