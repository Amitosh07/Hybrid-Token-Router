from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    model: str
    latency: float = 0
    tokens: int = 0
    confidence: float = 0
    cost: float = 0


class HealthResponse(BaseModel):
    status: str
    service: str
