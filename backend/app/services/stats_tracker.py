"""Lightweight in-memory runtime statistics tracker.

Thread-safe singleton updated after every routing decision in chat.py.
All values are derived from real request outcomes — nothing is hardcoded.

Cost estimation has been removed from this module. Without output token
counts from providers (which require streaming or a separate usage-metadata
call), any cost calculation is materially incomplete. It is better to
show no cost metric than an incorrect one.

Usage::

    from app.services.stats_tracker import stats

    # After a request completes:
    stats.record(
        provider="local",
        latency_ms=312.4,
        confidence=0.91,
        fallback_used=False,
    )

    # Retrieve snapshot for /stats endpoint:
    snapshot = stats.snapshot()
"""

from __future__ import annotations

import threading
import time
from typing import Any


class _StatsTracker:
    """Thread-safe in-memory accumulator for router runtime statistics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time: float = time.time()

        # Counters
        self._total_requests: int = 0
        self._local_requests: int = 0
        self._remote_requests: int = 0
        self._fallback_count: int = 0
        self._ml_predictions: int = 0
        self._heuristic_fallbacks: int = 0

        # Running sums (divided by total_requests for averages)
        self._sum_latency_ms: float = 0.0
        self._sum_confidence: float = 0.0

        # Current state
        self._last_provider: str = "none"
        self._last_router: str = "ML Router"
        self._last_routing_confidence: str = "Low"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        provider: str,
        latency_ms: float,
        confidence: float,
        estimated_input_tokens: int = 0,
        fallback_used: bool = False,
        routing_method: str = "ML",
        routing_confidence: str = "Low",
    ) -> None:
        """Record one completed request.

        Args:
            provider:               ``"local"`` or ``"remote"``.
            latency_ms:             End-to-end latency in milliseconds.
            confidence:             Router confidence in [0, 1].
            estimated_input_tokens: Token estimate (stored but not used for cost).
            fallback_used:          True when local timed out and remote was used.
            routing_method:         "ML" or "Heuristic Fallback".
        """
        with self._lock:
            self._total_requests += 1
            self._sum_latency_ms += latency_ms
            self._sum_confidence += confidence
            self._last_provider = provider
            self._last_router = "ML Router" if routing_method == "ML" else "Heuristic Fallback"
            self._last_routing_confidence = routing_confidence

            if provider == "local":
                self._local_requests += 1
            else:
                self._remote_requests += 1

            if fallback_used:
                self._fallback_count += 1

            if routing_method == "ML":
                self._ml_predictions += 1
            else:
                self._heuristic_fallbacks += 1

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time snapshot of all stats."""
        with self._lock:
            total = self._total_requests
            avg_latency = round(self._sum_latency_ms / total, 2) if total else 0.0
            avg_confidence = round(self._sum_confidence / total, 4) if total else 0.0
            uptime = round(time.time() - self._start_time, 1)

            return {
                "current_provider": self._last_provider,
                "current_router": self._last_router,
                "total_requests": total,
                "local_requests": self._local_requests,
                "remote_requests": self._remote_requests,
                "fallback_count": self._fallback_count,
                "ml_predictions": self._ml_predictions,
                "heuristic_fallbacks": self._heuristic_fallbacks,
                "average_latency_ms": avg_latency,
                "average_confidence": avg_confidence,
                "average_prediction_confidence": avg_confidence,
                "current_routing_confidence": self._last_routing_confidence,
                "routing_distribution": {
                    "local": self._local_requests,
                    "remote": self._remote_requests,
                },
                "uptime_seconds": uptime,
            }


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

stats = _StatsTracker()
