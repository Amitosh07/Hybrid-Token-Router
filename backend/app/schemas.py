from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    routing_score: int = 0
    confidence: float = 0
    reason: list[str] = Field(default_factory=list)
    latency: float = 0          # milliseconds — measured end-to-end on the backend
    features: dict = Field(default_factory=dict)
    task_type: str = ""
    complexity: str = ""
    estimated_input_tokens: int = 0
    # estimated_cost_saved removed: output token count is not available from
    # either provider in the current non-streaming implementation, making any
    # cost calculation materially incomplete. Removed to avoid displaying an
    # incorrect metric.
    fallback_used: bool = False
    prompt_id: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str


class StatsResponse(BaseModel):
    current_provider: str
    total_requests: int
    local_requests: int
    remote_requests: int
    fallback_count: int
    average_latency_ms: float
    average_confidence: float
    # estimated_cost_saved removed: see ChatResponse note above.
    router_version: str
    uptime_seconds: float
    timestamp: str
