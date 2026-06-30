import time
from typing import Literal

from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Query

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse
from app.services import ollama
from app.services import remote_llm

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    provider: Literal["local", "remote"] = Query(...),
) -> ChatResponse:
    settings = get_settings()
    start_time = time.perf_counter()

    try:
        if provider == "local":
            answer = await ollama.generate(request.prompt)
            model = settings.OLLAMA_MODEL
        else:
            answer = await remote_llm.generate(request.prompt)
            model = settings.REMOTE_MODEL
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ConnectionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except RuntimeError as exc:
        status_code = 429 if "rate limit" in str(exc).lower() else 502
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

    return ChatResponse(
        answer=answer,
        model=model or "none",
        latency=round(time.perf_counter() - start_time, 4),
        tokens=0,
        confidence=0,
        cost=0,
    )
