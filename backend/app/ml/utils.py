"""Utility functions for the ML module."""

from __future__ import annotations

import math
from pathlib import Path


def get_ml_dir() -> Path:
    """Return the absolute path to the backend/app/ml directory."""
    return Path(__file__).resolve().parent


def get_data_dir() -> Path:
    """Return the absolute path to the backend/app/data directory."""
    return get_ml_dir().parent / "data"


def clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a floating point value to a bounded range."""
    return max(min_val, min(max_val, val))
