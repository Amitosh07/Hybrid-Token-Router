"""GET /stats — live runtime statistics for the Hybrid Token Router."""

import time

from fastapi import APIRouter

from app.config import get_settings
from app.schemas import StatsResponse
from app.services.stats_tracker import stats

router = APIRouter(tags=["Stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return live router statistics collected from real request outcomes."""
    settings = get_settings()
    snapshot = stats.snapshot()

    return StatsResponse(
        current_provider=snapshot["current_provider"],
        total_requests=snapshot["total_requests"],
        local_requests=snapshot["local_requests"],
        remote_requests=snapshot["remote_requests"],
        fallback_count=snapshot["fallback_count"],
        average_latency_ms=snapshot["average_latency_ms"],
        average_confidence=snapshot["average_confidence"],
        router_version=settings.APP_VERSION,
        uptime_seconds=snapshot["uptime_seconds"],
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
