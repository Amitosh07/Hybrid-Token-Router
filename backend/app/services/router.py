"""Deterministic baseline Routing Engine for the Hybrid Token Router.

Receives the feature dictionary produced by ``extract_features()`` and decides
whether to forward the prompt to the LOCAL or REMOTE model.

Design principles
-----------------
* **No ML, no LLM, no network calls.**  All decisions are rule-based.
* **Weights in one place.**  ``RouterConfig`` is the single source of truth.
  Swap or train new weights there without touching the scoring logic.
* **reasoning_score dominates task_type.**  A trivial coding task stays local;
  a deeply constrained planning task goes remote.  task_type is just one signal.
* **Explainable output.**  Every point contributed to the routing score is
  narrated in the ``reason`` list so operators can audit decisions.

Public API
----------
    route(features: dict) -> dict
    route_prompt(prompt: str) -> dict          (convenience wrapper)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

from app.services.feature_extractor import (
    COMPLEXITY_EASY,
    COMPLEXITY_HARD,
    COMPLEXITY_MEDIUM,
    TASK_CODING,
    TASK_CREATIVE_WRITING,
    TASK_GENERAL,
    TASK_MATHEMATICS,
    TASK_PLANNING,
    TASK_QUESTION_ANSWERING,
    TASK_REASONING,
    TASK_SUMMARIZATION,
    TASK_TRANSLATION,
    extract_features,
)


# ---------------------------------------------------------------------------
# Provider labels
# ---------------------------------------------------------------------------

PROVIDER_LOCAL: Final = "local"
PROVIDER_REMOTE: Final = "remote"


# ---------------------------------------------------------------------------
# Configuration — all tuneable knobs live here.
# ---------------------------------------------------------------------------

@dataclass
class RouterConfig:
    """All routing weights and thresholds in one place.

    A future ML training step can serialise a ``RouterConfig`` to/from JSON
    and inject it into ``route()`` without touching any scoring logic.

    Attributes
    ----------
    threshold:
        routing_score at or above which the REMOTE model is chosen.
        Below this the LOCAL model is chosen.
        Default: 25 (out of a theoretical max of ~100).

    max_confidence_distance:
        routing_score distance from the threshold that maps to confidence 1.0.
        Scores farther than this are clamped to 1.0.
        Default: 20.

    # ── Reasoning score (weight per raw point, 0-10 scale) ──────────────
    reasoning_score_weight:
        Points added per unit of reasoning_score.  This is the heaviest
        weight by design so that prompt complexity drives routing more than
        task_type alone.
        Default: 5.  (max contribution: 50)

    # ── Token / length pressure ──────────────────────────────────────────
    token_weight_per_256:
        Points added per 256 estimated input tokens.
        Default: 3.  (max practical contribution: ~15 for 1 000-token prompt)

    # ── Boolean feature flags ─────────────────────────────────────────────
    code_weight:
        Flat bonus when contains_code is True.
        Default: 5.
    math_weight:
        Flat bonus when contains_math is True.
        Default: 4.
    json_weight:
        Flat bonus when contains_json is True.
        Default: 3.

    # ── Complexity label ──────────────────────────────────────────────────
    complexity_weights:
        Mapping from complexity label to flat bonus.
        Default: easy→0, medium→5, hard→10.

    # ── Task-type nudge (intentionally small) ────────────────────────────
    task_type_weights:
        Per-task flat nudge.  Kept small so that a short task of any type
        does not unconditionally go remote.
        Default: see below.
    """

    threshold: int = 25

    max_confidence_distance: int = 20

    # Reasoning score (dominant signal)
    reasoning_score_weight: int = 5

    # Token pressure
    token_weight_per_256: int = 3

    # Boolean flags
    code_weight: int = 5
    math_weight: int = 4
    json_weight: int = 3

    # Complexity
    complexity_weights: dict[str, int] = field(
        default_factory=lambda: {
            COMPLEXITY_EASY: 0,
            COMPLEXITY_MEDIUM: 5,
            COMPLEXITY_HARD: 10,
        }
    )

    # Task-type nudge (small — reasoning_score is the real driver)
    task_type_weights: dict[str, int] = field(
        default_factory=lambda: {
            TASK_CODING: 4,
            TASK_MATHEMATICS: 4,
            TASK_REASONING: 5,
            TASK_SUMMARIZATION: 3,
            TASK_TRANSLATION: 2,
            TASK_CREATIVE_WRITING: 3,
            TASK_PLANNING: 4,
            TASK_QUESTION_ANSWERING: 1,
            TASK_GENERAL: 0,
        }
    )


# Shared default config instance.  Pass a custom RouterConfig to route() to
# override any weight without mutating this singleton.
_DEFAULT_CONFIG: Final = RouterConfig()


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _score_reasoning(features: dict, cfg: RouterConfig) -> tuple[int, list[str]]:
    """Score from the reasoning_score feature (dominant signal).

    Returns:
        (points, reasons)
    """
    raw: int = features.get("reasoning_score", 0)
    points = raw * cfg.reasoning_score_weight
    reasons: list[str] = []

    if raw >= 8:
        reasons.append(f"Very high reasoning score ({raw}/10)")
    elif raw >= 5:
        reasons.append(f"Elevated reasoning score ({raw}/10)")
    elif raw >= 2:
        reasons.append(f"Moderate reasoning score ({raw}/10)")
    # Low score contributes 0 and warrants no reason entry.

    return points, reasons


def _score_tokens(features: dict, cfg: RouterConfig) -> tuple[int, list[str]]:
    """Score from estimated input token count.

    Returns:
        (points, reasons)
    """
    tokens: int = features.get("estimated_input_tokens", 0)
    blocks = tokens // 256
    points = blocks * cfg.token_weight_per_256
    reasons: list[str] = []

    if tokens > 512:
        reasons.append(f"Long prompt (~{tokens} estimated tokens)")
    elif tokens > 256:
        reasons.append(f"Medium-length prompt (~{tokens} estimated tokens)")

    return points, reasons


def _score_complexity(features: dict, cfg: RouterConfig) -> tuple[int, list[str]]:
    """Score from the complexity label.

    Returns:
        (points, reasons)
    """
    complexity: str = features.get("complexity", COMPLEXITY_EASY)
    points = cfg.complexity_weights.get(complexity, 0)
    reasons: list[str] = []

    if complexity == COMPLEXITY_HARD:
        reasons.append("Prompt is high complexity")
    elif complexity == COMPLEXITY_MEDIUM:
        reasons.append("Prompt is medium complexity")

    return points, reasons


def _score_task_type(features: dict, cfg: RouterConfig) -> tuple[int, list[str]]:
    """Small nudge from task_type (intentionally subordinate to reasoning_score).

    Returns:
        (points, reasons)
    """
    task: str = features.get("task_type", TASK_GENERAL)
    points = cfg.task_type_weights.get(task, 0)
    reasons: list[str] = []

    if points > 0:
        reasons.append(f"Task type is {task.replace('_', ' ')}")

    return points, reasons


def _score_boolean_flags(
    features: dict, cfg: RouterConfig
) -> tuple[int, list[str]]:
    """Score from boolean content flags (code, math, JSON).

    Returns:
        (points, reasons)
    """
    points = 0
    reasons: list[str] = []

    if features.get("contains_code"):
        points += cfg.code_weight
        reasons.append("Contains code")

    if features.get("contains_math"):
        points += cfg.math_weight
        reasons.append("Contains mathematical content")

    if features.get("contains_json"):
        points += cfg.json_weight
        reasons.append("Contains JSON structure")

    return points, reasons


# ---------------------------------------------------------------------------
# Confidence calculation
# ---------------------------------------------------------------------------

def _compute_confidence(
    routing_score: int, threshold: int, max_distance: int
) -> float:
    """Convert distance from threshold into a [0, 1] confidence score.

    The farther the score is from the threshold, the higher the confidence.
    Uses a sigmoid-like curve so confidence grows quickly near the threshold
    and then tapers toward 1.0.

    Args:
        routing_score: The computed routing score.
        threshold:     Decision boundary.
        max_distance:  Distance that maps to confidence ~0.98.

    Returns:
        Float in [0.0, 1.0].
    """
    distance = abs(routing_score - threshold)
    if max_distance <= 0:
        return 1.0
    # Sigmoid over normalised distance; k=5 gives a smooth curve.
    k = 5.0
    x = distance / max_distance
    confidence = 1 / (1 + math.exp(-k * (x - 0.5)))
    return round(min(confidence, 1.0), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route(features: dict, config: RouterConfig | None = None) -> dict:
    """Decide LOCAL vs REMOTE from a pre-computed feature dictionary.

    This function is the sole public entry point of this module.  It is
    intentionally pure: given the same ``features`` and ``config`` it always
    returns the same result.

    The interface is stable.  A future ML-based weight set can be injected via
    ``config`` without changing callers.

    Args:
        features:
            Dictionary returned by ``extract_features()``.  Must contain the
            keys documented in ``feature_extractor.py``.
        config:
            Optional ``RouterConfig``.  Defaults to ``_DEFAULT_CONFIG``.

    Returns:
        A dictionary with the following keys:

        ``provider`` (str):
            ``"local"`` or ``"remote"``.

        ``routing_score`` (int):
            Raw integer score before thresholding.  Higher → more remote.

        ``confidence`` (float):
            Value in [0.0, 1.0].  Distance from the threshold normalised by
            ``config.max_confidence_distance``.

        ``reason`` (list[str]):
            Human-readable list explaining every score contribution.

    Example::

        >>> from app.services.feature_extractor import extract_features
        >>> features = extract_features("Write hello world in Python")
        >>> result = route(features)
        >>> result["provider"]
        'local'
    """
    cfg = config if config is not None else _DEFAULT_CONFIG

    # --- Accumulate sub-scores and reasons ---------------------------------
    all_reasons: list[str] = []
    total_score = 0

    for scorer in (
        _score_reasoning,
        _score_tokens,
        _score_complexity,
        _score_task_type,
        _score_boolean_flags,
    ):
        pts, rsns = scorer(features, cfg)
        total_score += pts
        all_reasons.extend(rsns)

    # --- Decision -----------------------------------------------------------
    provider = PROVIDER_REMOTE if total_score >= cfg.threshold else PROVIDER_LOCAL

    # --- Confidence ---------------------------------------------------------
    confidence = _compute_confidence(total_score, cfg.threshold, cfg.max_confidence_distance)

    # --- Add decision summary to reasons ------------------------------------
    if provider == PROVIDER_REMOTE:
        all_reasons.append(
            f"Routing score {total_score} meets or exceeds threshold {cfg.threshold} → remote"
        )
    else:
        if not all_reasons:
            all_reasons.append("Simple or short prompt")
        all_reasons.append(
            f"Routing score {total_score} is below threshold {cfg.threshold} → local"
        )

    return {
        "provider": provider,
        "routing_score": total_score,
        "confidence": confidence,
        "reason": all_reasons,
    }


def route_prompt(prompt: str, config: RouterConfig | None = None) -> dict:
    """Convenience wrapper: extract features then route in one call.

    Calls ``extract_features(prompt)`` and passes the result to ``route()``.
    Useful for quick one-shot usage; prefer calling both separately when the
    feature dict is needed for other purposes (e.g. logging, evaluation).

    Args:
        prompt: Raw user prompt string.
        config: Optional ``RouterConfig``.  Defaults to ``_DEFAULT_CONFIG``.

    Returns:
        Same shape as ``route()``.
    """
    return route(extract_features(prompt), config)
