"""Decision Engine Configuration for Phase 2.

Defines configurable weights, pricing, and thresholds for routing decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class DecisionConfig:
    """Configurable weights and thresholds for the Decision Engine."""

    # Weights for the utility formula (should sum to 1.0)
    quality_weight: float = 0.60
    latency_weight: float = 0.20
    cost_weight: float = 0.20

    # Local preference bias (bonus added to local score to favor local provider)
    local_preference_bias: float = 0.05

    # Pricing per million tokens (in USD)
    remote_input_price_per_m: float = 2.50
    remote_output_price_per_m: float = 10.00
    local_input_price_per_m: float = 0.00
    local_output_price_per_m: float = 0.00

    # Safety and trade-off overrides
    quality_threshold_high: float = 0.85     # If local quality >= this, select LOCAL
    quality_delta_threshold: float = 0.08     # If remote_quality - local_quality <= this, select LOCAL
    max_remote_latency_ms: float = 8000.0     # If remote is slower than this, select LOCAL unless local fails
    min_remote_quality: float = 0.35         # Remote quality must be at least this to justify remote routing

    def to_dict(self) -> dict[str, Any]:
        """Convert config properties to a dictionary."""
        return {
            "quality_weight": self.quality_weight,
            "latency_weight": self.latency_weight,
            "cost_weight": self.cost_weight,
            "local_preference_bias": self.local_preference_bias,
            "remote_input_price_per_m": self.remote_input_price_per_m,
            "remote_output_price_per_m": self.remote_output_price_per_m,
            "local_input_price_per_m": self.local_input_price_per_m,
            "local_output_price_per_m": self.local_output_price_per_m,
            "quality_threshold_high": self.quality_threshold_high,
            "quality_delta_threshold": self.quality_delta_threshold,
            "max_remote_latency_ms": self.max_remote_latency_ms,
            "min_remote_quality": self.min_remote_quality,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecisionConfig:
        """Create a config instance from a dictionary."""
        return cls(
            quality_weight=data.get("quality_weight", 0.60),
            latency_weight=data.get("latency_weight", 0.20),
            cost_weight=data.get("cost_weight", 0.20),
            local_preference_bias=data.get("local_preference_bias", 0.05),
            remote_input_price_per_m=data.get("remote_input_price_per_m", 2.50),
            remote_output_price_per_m=data.get("remote_output_price_per_m", 10.00),
            local_input_price_per_m=data.get("local_input_price_per_m", 0.00),
            local_output_price_per_m=data.get("local_output_price_per_m", 0.00),
            quality_threshold_high=data.get("quality_threshold_high", 0.85),
            quality_delta_threshold=data.get("quality_delta_threshold", 0.08),
            max_remote_latency_ms=data.get("max_remote_latency_ms", 8000.0),
            min_remote_quality=data.get("min_remote_quality", 0.35),
        )

    def save(self, path: Path) -> None:
        """Save configuration as JSON to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path: Path) -> DecisionConfig:
        """Load configuration from a JSON file, fallback to defaults."""
        if not path.exists():
            return cls()
        try:
            with path.open("r", encoding="utf-8") as file:
                return cls.from_dict(json.load(file))
        except Exception:
            return cls()
