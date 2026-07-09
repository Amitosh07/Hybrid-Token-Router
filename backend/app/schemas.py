from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    provider: str
    selected_provider: str = ""
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
    prediction_probability: float = 0
    prediction_confidence: float = 0
    routing_confidence: str = "Low"
    local_score: float | None = None
    remote_score: float | None = None
    model_version: str = ""
    routing_method: str = ""


class HealthResponse(BaseModel):
    status: str
    service: str


class PredictResponse(BaseModel):
    selected_provider: str
    prediction_probability: float
    prediction_confidence: float
    routing_confidence: str
    local_score: float | None = None
    remote_score: float | None = None
    model_version: str
    routing_method: str
    feature_contributions: list[dict] = Field(default_factory=list)


class StatsResponse(BaseModel):
    current_provider: str
    current_router: str
    total_requests: int
    local_requests: int
    remote_requests: int
    fallback_count: int
    ml_predictions: int
    heuristic_fallbacks: int
    average_latency_ms: float
    average_confidence: float
    average_prediction_confidence: float
    current_routing_confidence: str = "Low"
    routing_distribution: dict[str, int] = Field(default_factory=dict)
    # estimated_cost_saved removed: see ChatResponse note above.
    router_version: str
    uptime_seconds: float
    timestamp: str
