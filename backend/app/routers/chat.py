from fastapi import APIRouter

from app.schemas import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(
        answer="Backend is working",
        model="none",
        latency=0,
        tokens=0,
        confidence=0,
        cost=0,
    )
