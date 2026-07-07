"""Reasoning depth library defining cognitive complexity levels for prompt routing."""

from __future__ import annotations

from dataclasses import dataclass

REASONING_DEPTHS: tuple[str, ...] = ("low", "medium", "high")


@dataclass(frozen=True)
class ReasoningProfile:
    """Detailed profile for a reasoning depth level."""

    depth: str
    description: str
    cognitive_load: float
    indicators: tuple[str, ...]


REASONING_PROFILES: dict[str, ReasoningProfile] = {
    "low": ReasoningProfile(
        depth="low",
        description="Direct retrieval, pattern matching, simple translation, or summarization without conflicting constraints.",
        cognitive_load=0.1,
        indicators=("retrieve", "list", "summarize", "convert", "translate"),
    ),
    "medium": ReasoningProfile(
        depth="medium",
        description="Multi-step logical deduction, basic planning, moderate comparison, or structured formatting.",
        cognitive_load=0.45,
        indicators=("compare", "explain", "analyze", "resolve", "refactor"),
    ),
    "high": ReasoningProfile(
        depth="high",
        description="Complex synthesis, edge-case mitigation, mathematical proof, deep technical audits, or planning under uncertainty.",
        cognitive_load=0.85,
        indicators=("prove", "audit", "optimize", "evaluate", "synthesize"),
    ),
}


def get_reasoning_profile(depth: str) -> ReasoningProfile:
    """Retrieve reasoning depth profile by name."""
    if depth not in REASONING_PROFILES:
        raise ValueError(f"Unsupported reasoning depth: {depth}")
    return REASONING_PROFILES[depth]
