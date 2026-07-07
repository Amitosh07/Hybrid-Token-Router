"""Decision Engine for the Hybrid Token Router — Phase 2.

Determines the optimal provider (LOCAL vs REMOTE) for a given prompt run by
comparing response quality, latency, cost, and structural correctness.
"""

from __future__ import annotations

from typing import Any

from app.ml.config import DecisionConfig


class DecisionEngine:
    """Evaluates Local vs Remote performance to produce optimal routing labels."""

    def __init__(self, config: DecisionConfig | None = None) -> None:
        self.config = config or DecisionConfig()

    def calculate_cost(self, input_tokens: int, output_tokens: int, provider: str) -> float:
        """Calculate the estimated execution cost in USD."""
        if provider == "local":
            in_price = self.config.local_input_price_per_m
            out_price = self.config.local_output_price_per_m
        else:
            in_price = self.config.remote_input_price_per_m
            out_price = self.config.remote_output_price_per_m

        in_cost = (input_tokens / 1_000_000.0) * in_price
        out_cost = (output_tokens / 1_000_000.0) * out_price
        return in_cost + out_cost

    def decide(self, record: dict[str, Any]) -> dict[str, Any]:
        """Determine which model should have been selected for this prompt in production.

        Args:
            record: A benchmark run record containing 'local', 'remote',
                    and 'evaluation' dicts.

        Returns:
            A dictionary containing:
              - "label": "LOCAL" or "REMOTE"
              - "local_cost": float (USD)
              - "remote_cost": float (USD)
              - "local_quality": float [0.0, 1.0]
              - "remote_quality": float [0.0, 1.0]
              - "local_utility": float
              - "remote_utility": float
              - "reason": str explanation
        """
        local_data = record.get("local") or {}
        remote_data = record.get("remote") or {}
        evaluation = record.get("evaluation") or {}
        metrics = evaluation.get("metrics") or {}

        # 1. Capture errors and structural validity (Safety Override)
        local_err = local_data.get("error") or metrics.get("structural_validity", {}).get("local_error")
        remote_err = remote_data.get("error") or metrics.get("structural_validity", {}).get("remote_error")

        local_valid = metrics.get("structural_validity", {}).get("local_valid", True) and not local_err
        remote_valid = metrics.get("structural_validity", {}).get("remote_valid", True) and not remote_err

        # Get quality scores
        local_quality = float(metrics.get("estimated_quality", {}).get("local", {}).get("score", 0.0))
        remote_quality = float(metrics.get("estimated_quality", {}).get("remote", {}).get("score", 0.0))

        # Re-verify error overrides
        if not local_valid and remote_valid:
            remote_in = remote_data.get("estimated_input_tokens", 0)
            remote_out = remote_data.get("estimated_output_tokens", 0)
            return self._decision_payload(
                label="REMOTE",
                local_cost=0.0,
                remote_cost=self.calculate_cost(remote_in, remote_out, "remote"),
                local_quality=0.0,
                remote_quality=remote_quality,
                local_util=0.0,
                remote_util=1.0,
                reason="Local model encountered an error or generated empty text; remote succeeded."
            )

        if not remote_valid and local_valid:
            local_in = local_data.get("estimated_input_tokens", 0)
            local_out = local_data.get("estimated_output_tokens", 0)
            return self._decision_payload(
                label="LOCAL",
                local_cost=self.calculate_cost(local_in, local_out, "local"),
                remote_cost=0.0,
                local_quality=local_quality,
                remote_quality=0.0,
                local_util=1.0,
                remote_util=0.0,
                reason="Remote model encountered an error or generated empty text; local succeeded."
            )

        if not local_valid and not remote_valid:
            return self._decision_payload(
                label="LOCAL",
                local_cost=0.0,
                remote_cost=0.0,
                local_quality=0.0,
                remote_quality=0.0,
                local_util=0.0,
                remote_util=0.0,
                reason="Both providers failed or returned invalid responses; defaulting to LOCAL."
            )

        # 2. Extract operational parameters
        local_latency = float(local_data.get("latency_ms", 0))
        remote_latency = float(remote_data.get("latency_ms", 0))

        local_in = local_data.get("estimated_input_tokens", 0)
        local_out = local_data.get("estimated_output_tokens", 0)
        remote_in = remote_data.get("estimated_input_tokens", 0)
        remote_out = remote_data.get("estimated_output_tokens", 0)

        local_cost = self.calculate_cost(local_in, local_out, "local")
        remote_cost = self.calculate_cost(remote_in, remote_out, "remote")

        # 3. Apply expert heuristics (Overriding thresholds)
        quality_delta = remote_quality - local_quality

        # Heuristic A: Local is already exceptionally high quality
        if local_quality >= self.config.quality_threshold_high:
            return self._decision_payload(
                label="LOCAL",
                local_cost=local_cost,
                remote_cost=remote_cost,
                local_quality=local_quality,
                remote_quality=remote_quality,
                local_util=1.0,
                remote_util=0.0,
                reason=f"Local model quality is exceptionally high ({local_quality:.3f} >= {self.config.quality_threshold_high:.2f})."
            )

        # Heuristic B: Quality improvement of Remote is marginal
        if quality_delta <= self.config.quality_delta_threshold:
            return self._decision_payload(
                label="LOCAL",
                local_cost=local_cost,
                remote_cost=remote_cost,
                local_quality=local_quality,
                remote_quality=remote_quality,
                local_util=1.0,
                remote_util=0.0,
                reason=f"Remote quality improvement is marginal ({quality_delta:.3f} <= {self.config.quality_delta_threshold:.2f})."
            )

        # Heuristic C: Remote latency is excessive
        if remote_latency >= self.config.max_remote_latency_ms:
            return self._decision_payload(
                label="LOCAL",
                local_cost=local_cost,
                remote_cost=remote_cost,
                local_quality=local_quality,
                remote_quality=remote_quality,
                local_util=1.0,
                remote_util=0.0,
                reason=f"Remote latency is excessive ({remote_latency}ms >= {self.config.max_remote_latency_ms:.0f}ms)."
            )

        # Heuristic D: Remote quality is too poor to justify remote routing
        if remote_quality < self.config.min_remote_quality:
            return self._decision_payload(
                label="LOCAL",
                local_cost=local_cost,
                remote_cost=remote_cost,
                local_quality=local_quality,
                remote_quality=remote_quality,
                local_util=1.0,
                remote_util=0.0,
                reason=f"Remote quality is too poor ({remote_quality:.3f} < {self.config.min_remote_quality:.2f})."
            )

        # 4. Calculate Utility scores
        # Normalize speed in [0.0, 1.0]: 1 / (1 + latency_seconds)
        local_speed = 1.0 / (1.0 + local_latency / 1000.0)
        remote_speed = 1.0 / (1.0 + remote_latency / 1000.0)

        # Normalize cost in [0.0, 1.0] by dividing by $0.01 max cost threshold
        local_cost_norm = min(1.0, local_cost / 0.01)
        remote_cost_norm = min(1.0, remote_cost / 0.01)

        # Compute weighted utility
        local_utility = (
            self.config.quality_weight * local_quality
            + self.config.latency_weight * local_speed
            - self.config.cost_weight * local_cost_norm
            + self.config.local_preference_bias
        )

        remote_utility = (
            self.config.quality_weight * remote_quality
            + self.config.latency_weight * remote_speed
            - self.config.cost_weight * remote_cost_norm
        )

        if remote_utility > local_utility:
            label = "REMOTE"
            reason = (
                f"Remote utility ({remote_utility:.3f}) exceeds Local utility ({local_utility:.3f}). "
                f"Quality delta: {quality_delta:.3f}."
            )
        else:
            label = "LOCAL"
            reason = (
                f"Local utility ({local_utility:.3f}) is higher due to cost/latency savings. "
                f"Quality delta: {quality_delta:.3f}."
            )

        return self._decision_payload(
            label=label,
            local_cost=local_cost,
            remote_cost=remote_cost,
            local_quality=local_quality,
            remote_quality=remote_quality,
            local_util=local_utility,
            remote_util=remote_utility,
            reason=reason
        )

    def _decision_payload(
        self,
        label: str,
        local_cost: float,
        remote_cost: float,
        local_quality: float,
        remote_quality: float,
        local_util: float,
        remote_util: float,
        reason: str
    ) -> dict[str, Any]:
        """Format the output dictionary structure."""
        return {
            "label": label,
            "local_cost": round(local_cost, 6),
            "remote_cost": round(remote_cost, 6),
            "local_quality": round(local_quality, 4),
            "remote_quality": round(remote_quality, 4),
            "local_utility": round(local_util, 4),
            "remote_utility": round(remote_util, 4),
            "reason": reason
        }
