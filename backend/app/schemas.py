from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    model: str
    latency: float
    tokens: int
    confidence: float
    cost: float


class HealthResponse(BaseModel):
    status: str
    service: str
