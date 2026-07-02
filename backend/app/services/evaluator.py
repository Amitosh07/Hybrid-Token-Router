"""Deterministic evaluation metrics for local and remote model outputs.

The evaluator measures provider outputs but does not choose a final winner.
Final label selection belongs in a separate decision engine so evaluation can
evolve independently from routing or training logic.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any


ProviderName = str


@dataclass(frozen=True)
class ResponseLengthMetrics:
    """Response length comparison for local and remote outputs."""

    local_chars: int
    remote_chars: int
    local_words: int
    remote_words: int
    char_delta: int
    word_delta: int


@dataclass(frozen=True)
class LatencyMetrics:
    """Latency comparison without making a winner decision."""

    local_ms: float | None
    remote_ms: float | None
    delta_ms: float | None
    local_speed_score: float
    remote_speed_score: float


@dataclass(frozen=True)
class StructuralValidityMetrics:
    """Basic checks that a provider returned usable text."""

    local_valid: bool
    remote_valid: bool
    local_error: str | None
    remote_error: str | None


@dataclass(frozen=True)
class ProviderQualityMetrics:
    """Deterministic heuristic quality signals for one provider response."""

    score: float
    response_chars: int
    response_words: int
    sentence_count: int
    prompt_term_coverage: float
    has_error: bool
    is_empty: bool


def _get_response(provider: dict[str, Any]) -> str:
    """Return provider response text or an empty string when unavailable."""

    response = provider.get("response", "")
    return response if isinstance(response, str) else ""


def _get_latency(provider: dict[str, Any]) -> float | None:
    """Return provider latency in milliseconds when it is numeric."""

    latency = provider.get("latency_ms")
    if isinstance(latency, bool) or not isinstance(latency, int | float):
        return None
    return float(latency)


def _words(text: str) -> list[str]:
    """Extract normalized words for deterministic text comparisons."""

    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """Clamp a floating-point score into a bounded range."""

    return max(minimum, min(maximum, value))


def compare_response_lengths(
    local: dict[str, Any],
    remote: dict[str, Any],
) -> dict[str, int]:
    """Compare local and remote response lengths without selecting a winner."""

    local_response = _get_response(local)
    remote_response = _get_response(remote)
    local_words = len(_words(local_response))
    remote_words = len(_words(remote_response))
    metrics = ResponseLengthMetrics(
        local_chars=len(local_response),
        remote_chars=len(remote_response),
        local_words=local_words,
        remote_words=remote_words,
        char_delta=len(local_response) - len(remote_response),
        word_delta=local_words - remote_words,
    )
    return asdict(metrics)


def compare_latency(local: dict[str, Any], remote: dict[str, Any]) -> dict[str, Any]:
    """Compare provider latencies and expose normalized speed scores."""

    local_ms = _get_latency(local)
    remote_ms = _get_latency(remote)
    delta_ms = None if local_ms is None or remote_ms is None else local_ms - remote_ms
    local_score = _speed_score(local_ms)
    remote_score = _speed_score(remote_ms)
    metrics = LatencyMetrics(
        local_ms=local_ms,
        remote_ms=remote_ms,
        delta_ms=delta_ms,
        local_speed_score=local_score,
        remote_speed_score=remote_score,
    )
    return asdict(metrics)


def structural_validity(
    local: dict[str, Any],
    remote: dict[str, Any],
) -> dict[str, Any]:
    """Check whether each provider produced non-empty text and no error."""

    local_error = _error_text(local)
    remote_error = _error_text(remote)
    metrics = StructuralValidityMetrics(
        local_valid=bool(_get_response(local).strip()) and local_error is None,
        remote_valid=bool(_get_response(remote).strip()) and remote_error is None,
        local_error=local_error,
        remote_error=remote_error,
    )
    return asdict(metrics)


def estimate_quality(prompt: str, provider: dict[str, Any]) -> dict[str, Any]:
    """Estimate response quality using deterministic, replaceable heuristics."""

    response = _get_response(provider)
    response_words = _words(response)
    prompt_terms = set(_words(prompt))
    response_terms = set(response_words)
    coverage = _term_coverage(prompt_terms, response_terms)
    sentence_count = _sentence_count(response)
    has_error = _error_text(provider) is not None
    is_empty = not bool(response.strip())
    score = _quality_score(
        response_words=len(response_words),
        sentence_count=sentence_count,
        prompt_term_coverage=coverage,
        has_error=has_error,
        is_empty=is_empty,
    )
    metrics = ProviderQualityMetrics(
        score=score,
        response_chars=len(response),
        response_words=len(response_words),
        sentence_count=sentence_count,
        prompt_term_coverage=coverage,
        has_error=has_error,
        is_empty=is_empty,
    )
    return asdict(metrics)


def evaluate_run(run: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one dataset entry and return metrics for a decision engine."""

    prompt = str(run.get("prompt", ""))
    local = _provider_block(run, "local")
    remote = _provider_block(run, "remote")
    return {
        "prompt": prompt,
        "metrics": {
            "response_length": compare_response_lengths(local, remote),
            "latency": compare_latency(local, remote),
            "structural_validity": structural_validity(local, remote),
            "estimated_quality": {
                "local": estimate_quality(prompt, local),
                "remote": estimate_quality(prompt, remote),
            },
        },
        "evaluation_method": "rule",
    }


def evaluate_dataset(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evaluate every raw dataset entry and return metrics-only results."""

    return [evaluate_run(run) for run in dataset]


def choose_winner(metrics: dict[str, Any]) -> dict[str, Any]:
    """Reserved handoff for a future decision engine.

    The evaluator intentionally does not combine metrics into a final label.
    Implement that policy in a separate decision engine to avoid coupling
    measurement with routing or training decisions.
    """

    raise NotImplementedError(
        "Winner selection belongs in a separate decision_engine module."
    )


def _provider_block(run: dict[str, Any], name: ProviderName) -> dict[str, Any]:
    """Return a provider block from a raw run, defaulting to an error block."""

    provider = run.get(name)
    if isinstance(provider, dict):
        return provider
    return {"error": f"Missing {name} provider result."}


def _error_text(provider: dict[str, Any]) -> str | None:
    """Normalize provider error values to optional strings."""

    error = provider.get("error")
    if error is None:
        return None
    return str(error)


def _speed_score(latency_ms: float | None) -> float:
    """Convert latency to a bounded score where lower latency scores higher."""

    if latency_ms is None or latency_ms < 0:
        return 0.0
    return round(1 / (1 + latency_ms / 1000), 4)


def _term_coverage(prompt_terms: set[str], response_terms: set[str]) -> float:
    """Measure the share of prompt terms that appear in the response."""

    if not prompt_terms:
        return 0.0
    return round(len(prompt_terms & response_terms) / len(prompt_terms), 4)


def _sentence_count(text: str) -> int:
    """Estimate sentence count from terminal punctuation."""

    sentences = re.findall(r"[^.!?]+[.!?]", text.strip())
    return len(sentences) if sentences else int(bool(text.strip()))


def _quality_score(
    *,
    response_words: int,
    sentence_count: int,
    prompt_term_coverage: float,
    has_error: bool,
    is_empty: bool,
) -> float:
    """Combine simple quality heuristics into a deterministic provider score."""

    if has_error or is_empty:
        return 0.0

    length_score = _clamp(response_words / 80)
    structure_score = _clamp(math.log1p(sentence_count) / math.log(6))
    coverage_score = _clamp(prompt_term_coverage)
    score = (length_score * 0.4) + (structure_score * 0.3) + (coverage_score * 0.3)
    return round(score, 4)
