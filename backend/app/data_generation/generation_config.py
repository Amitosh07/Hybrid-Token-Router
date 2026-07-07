"""Configuration manager for the prompt generation and benchmark pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Default paths relative to backend directory
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = BACKEND_DIR / "generation_config.json"


class GenerationConfig:
    """Settings governing dataset size, limits, paths, and parallel execution."""

    def __init__(
        self,
        size: int = 1000,
        batch_size: int = 10,
        concurrency_limit: int = 5,
        simulate_llm: bool = False,
        prompt_catalog_path: str = "app/data/generated/generated_prompts.jsonl",
        checkpoint_path: str = "app/data/generated/large_benchmark_checkpoint.jsonl",
        results_path: str = "app/data/generated/large_benchmark_results.json",
        output_csv_path: str = "app/data/training/training_dataset_large.csv",
        min_words: int = 8,
        dedup_threshold: float = 0.82,
    ) -> None:
        self.size = size
        self.batch_size = batch_size
        self.concurrency_limit = concurrency_limit
        self.simulate_llm = simulate_llm
        self.prompt_catalog_path = prompt_catalog_path
        self.checkpoint_path = checkpoint_path
        self.results_path = results_path
        self.output_csv_path = output_csv_path
        self.min_words = min_words
        self.dedup_threshold = dedup_threshold

    @classmethod
    def load_from_json(cls, path: Path | str | None = None) -> GenerationConfig:
        """Load configuration from a JSON file, creating defaults if missing."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not config_path.exists():
            cfg = cls()
            cfg.save_to_json(config_path)
            return cfg

        try:
            with config_path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return cls(
                size=data.get("size", 1000),
                batch_size=data.get("batch_size", 10),
                concurrency_limit=data.get("concurrency_limit", 5),
                simulate_llm=data.get("simulate_llm", False),
                prompt_catalog_path=data.get("prompt_catalog_path", "app/data/generated/generated_prompts.jsonl"),
                checkpoint_path=data.get("checkpoint_path", "app/data/generated/large_benchmark_checkpoint.jsonl"),
                results_path=data.get("results_path", "app/data/generated/large_benchmark_results.json"),
                output_csv_path=data.get("output_csv_path", "app/data/training/training_dataset_large.csv"),
                min_words=data.get("min_words", 8),
                dedup_threshold=data.get("dedup_threshold", 0.82),
            )
        except Exception as exc:
            print(f"Error loading generation config: {exc}. Falling back to defaults.")
            return cls()

    def save_to_json(self, path: Path | str | None = None) -> None:
        """Save settings to a JSON file on disk."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        config_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "size": self.size,
            "batch_size": self.batch_size,
            "concurrency_limit": self.concurrency_limit,
            "simulate_llm": self.simulate_llm,
            "prompt_catalog_path": self.prompt_catalog_path,
            "checkpoint_path": self.checkpoint_path,
            "results_path": self.results_path,
            "output_csv_path": self.output_csv_path,
            "min_words": self.min_words,
            "dedup_threshold": self.dedup_threshold,
        }
        with config_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
            file.write("\n")

    def get_absolute_path(self, relative_path_str: str) -> Path:
        """Resolve a configured relative path to an absolute path inside backend/."""
        return BACKEND_DIR / relative_path_str
