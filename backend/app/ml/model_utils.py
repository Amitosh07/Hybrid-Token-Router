"""Shared utilities for the Phase 3 supervised ML pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


ML_DIR = Path(__file__).resolve().parent
APP_DIR = ML_DIR.parent
BACKEND_DIR = APP_DIR.parent
DATASET_PATH = APP_DIR / "data" / "training" / "training_dataset_large.csv"
LOCKED_EVAL_PATH = APP_DIR / "data" / "training" / "locked_evaluation_dataset.csv"
MODELS_DIR = BACKEND_DIR / "models"
DOCS_DIR = BACKEND_DIR / "docs"
PLOTS_DIR = ML_DIR / "plots"

MODEL_PATH = MODELS_DIR / "router_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
METADATA_PATH = MODELS_DIR / "router_model_metadata.json"

TARGET_COLUMN = "label"
LOCAL_LABEL = "LOCAL"
REMOTE_LABEL = "REMOTE"

POST_INFERENCE_COLUMNS = {
    "local_latency_ms",
    "remote_latency_ms",
    "local_cost",
    "remote_cost",
    "local_quality_score",
    "remote_quality_score",
    "local_llm_quality",
    "remote_llm_quality",
    "local_heuristic_quality",
    "remote_heuristic_quality",
    "local_tokens",
    "remote_tokens",
}

NON_PRODUCTION_COLUMNS = {
    "prompt_id",
    "prompt",
    "category",
    "difficulty",
    "expected_reasoning",
}


def ensure_output_dirs() -> None:
    """Create output directories used by training, reports, and plots."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    """Persist JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_artifact(path: Path, artifact: Any) -> None:
    """Persist a Python artifact with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: Path) -> Any:
    """Load a Python artifact persisted with joblib."""
    return joblib.load(path)


def normalize_provider_label(value: Any) -> str:
    """Normalize provider labels to the canonical uppercase target labels."""
    text = str(value).strip().upper()
    if text not in {LOCAL_LABEL, REMOTE_LABEL}:
        raise ValueError(f"Unsupported provider label: {value!r}")
    return text


def provider_to_numeric(value: Any) -> int:
    """Map provider labels to binary classes for sklearn metrics."""
    return 1 if normalize_provider_label(value) == REMOTE_LABEL else 0


def numeric_to_provider(value: int) -> str:
    """Map binary classes back to provider labels."""
    return REMOTE_LABEL if int(value) == 1 else LOCAL_LABEL
