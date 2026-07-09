"""Operator-facing confidence labels for threshold-based routing decisions."""

from __future__ import annotations


def probability_confidence(remote_probability: float, threshold: float) -> str:
    """Classify distance from the calibrated decision boundary without showing a percent."""
    margin = abs(float(remote_probability) - float(threshold))
    if margin >= 0.25:
        return "High"
    if margin >= 0.10:
        return "Medium"
    return "Low"


def score_confidence(routing_score: int, threshold: int) -> str:
    """Classify a heuristic score's distance from its decision boundary."""
    margin = abs(int(routing_score) - int(threshold))
    if margin >= 12:
        return "High"
    if margin >= 5:
        return "Medium"
    return "Low"
