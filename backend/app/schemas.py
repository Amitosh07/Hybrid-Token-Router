from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    routing_score: int = 0
    confidence: float = 0
    reason: list[str] = Field(default_factory=list)
    latency: float = 0
    features: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    service: str
