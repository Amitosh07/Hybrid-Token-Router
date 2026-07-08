"""Experiment Tracker for Phase 6.

Logs model, hyperparameter, and performance metadata to json files for reproducibility.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[2] / "app" / "data" / "experiments"
LOG_FILE_PATH = EXPERIMENTS_DIR / "experiment_log.json"


class ExperimentTracker:
    """Logs and indexes ML routing experiments."""

    def __init__(self) -> None:
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    def log_experiment(
        self,
        experiment_name: str,
        dataset_version: str,
        feature_version: str,
        embedding_model: str,
        classifier: str,
        hyperparameters: dict[str, Any],
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """Save a new training experiment run record."""
        run_id = f"exp_{int(datetime.now(timezone.utc).timestamp())}"
        
        record = {
            "run_id": run_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_version": dataset_version,
            "feature_version": feature_version,
            "embedding_model": embedding_model,
            "classifier": classifier,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
        }
        
        history = []
        if LOG_FILE_PATH.exists():
            try:
                with LOG_FILE_PATH.open("r", encoding="utf-8") as f:
                    history = json.load(f)
                    if not isinstance(history, list):
                        history = []
            except Exception as exc:
                logger.error("Failed to load experiment logs: %s", exc)
                
        history.append(record)
        
        try:
            with LOG_FILE_PATH.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, sort_keys=True)
            logger.info("Successfully recorded experiment run %s", run_id)
        except Exception as exc:
            logger.error("Failed to write experiment run %s: %s", run_id, exc)
            
        return record
