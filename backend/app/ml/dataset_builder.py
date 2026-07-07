"""Dataset Builder for Phase 2.

Reads completed benchmark JSON records, applies the Decision Engine, flattens
the structure into row-wise training features, and writes training_dataset.csv.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.ml.config import DecisionConfig
from app.ml.decision_engine import DecisionEngine

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_BENCHMARK_RESULTS_DIR = _PROJECT_ROOT / "app" / "data" / "benchmarks" / "results"
_DEFAULT_CSV_OUTPUT_PATH = _PROJECT_ROOT / "app" / "data" / "training" / "training_dataset.csv"


def find_latest_benchmark_file(results_dir: Path) -> Path | None:
    """Scan results_dir and return the path to the newest JSON benchmark file."""
    if not results_dir.exists():
        return None
    files = list(results_dir.glob("benchmark_*.json"))
    if not files:
        return None
    # Sort files by name (which has timestamp) or modification time
    files.sort(key=lambda x: x.name, reverse=True)
    return files[0]


def build_training_dataset(
    benchmark_json_path: Path | None = None,
    csv_output_path: Path | None = None,
    config: DecisionConfig | None = None,
) -> Path:
    """Build the final CSV training dataset from a JSON benchmark file.

    Args:
        benchmark_json_path: Path to the benchmark results JSON.
                             If None, looks for the latest file.
        csv_output_path:     Path to write the CSV dataset.
                             If None, writes to the default training path.
        config:              Configuration settings for the Decision Engine.

    Returns:
        Absolute Path to the generated CSV file.
    """
    json_path = benchmark_json_path or find_latest_benchmark_file(_DEFAULT_BENCHMARK_RESULTS_DIR)
    if json_path is None or not json_path.exists():
        raise FileNotFoundError(
            f"No benchmark results file found. Check directory: {_DEFAULT_BENCHMARK_RESULTS_DIR}"
        )

    output_csv = csv_output_path or _DEFAULT_CSV_OUTPUT_PATH
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading benchmark results from: {json_path}")
    with json_path.open("r", encoding="utf-8") as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError("Benchmark JSON root must be a list of records.")

    engine = DecisionEngine(config)

    # Header columns definition
    headers = [
        "prompt_id",
        "prompt",
        "category",
        "difficulty",
        "expected_reasoning",
        # --- Lexical Features ---
        "prompt_length",
        "word_count",
        "estimated_input_tokens",
        # --- Structural Features (binary features mapped to 0/1) ---
        "contains_code",
        "contains_math",
        "contains_json",
        "contains_markdown",
        "contains_numbers",
        "contains_question",
        # --- Feature Extractor Scores ---
        "reasoning_score",
        "technical_complexity",
        "reasoning_depth",
        "task_complexity",
        "constraint_complexity",
        "context_complexity",
        "complexity_score",
        # --- Operational Stats ---
        "local_latency_ms",
        "remote_latency_ms",
        "local_cost",
        "remote_cost",
        "local_quality_score",
        "remote_quality_score",
        # --- Decision Engine Target ---
        "label",
    ]

    rows = []
    print(f"Generating training records for {len(records)} samples...")

    for idx, rec in enumerate(records, start=1):
        # Calculate target label from Decision Engine
        decision = engine.decide(rec)

        features = rec.get("features", {})
        local_block = rec.get("local", {})
        remote_block = rec.get("remote", {})
        metrics = rec.get("evaluation", {}).get("metrics", {})

        # Build tabular row
        row = {
            "prompt_id":              rec.get("id", f"prompt_{idx:03d}"),
            "prompt":                 rec.get("prompt", ""),
            "category":               rec.get("category", "general"),
            "difficulty":             rec.get("difficulty", "medium"),
            "expected_reasoning":     rec.get("expected_reasoning", "medium"),
            # Lexical
            "prompt_length":          int(features.get("prompt_length", len(rec.get("prompt", "")))),
            "word_count":             int(features.get("word_count", len(rec.get("prompt", "").split()))),
            "estimated_input_tokens": int(features.get("estimated_input_tokens", 0)),
            # Structural
            "contains_code":          1 if features.get("contains_code", False) else 0,
            "contains_math":          1 if features.get("contains_math", False) else 0,
            "contains_json":          1 if features.get("contains_json", False) else 0,
            "contains_markdown":      1 if features.get("contains_markdown", False) else 0,
            "contains_numbers":       1 if features.get("contains_numbers", False) else 0,
            "contains_question":      1 if features.get("contains_question", False) else 0,
            # Feature extractor scores
            "reasoning_score":        int(features.get("reasoning_score", 0)),
            "technical_complexity":   float(features.get("technical_complexity", {}).get("score", 0.0)),
            "reasoning_depth":        float(features.get("reasoning_depth", {}).get("score", 0.0)),
            "task_complexity":        float(features.get("task_complexity", {}).get("score", 0.0)),
            "constraint_complexity":  float(features.get("constraint_complexity", {}).get("score", 0.0)),
            "context_complexity":     float(features.get("context_complexity", {}).get("score", 0.0)),
            "complexity_score":       float(features.get("complexity_score", 0.0)),
            # Operational stats
            "local_latency_ms":       int(local_block.get("latency_ms", 0)),
            "remote_latency_ms":      int(remote_block.get("latency_ms", 0)),
            "local_cost":             float(decision["local_cost"]),
            "remote_cost":            float(decision["remote_cost"]),
            "local_quality_score":    float(decision["local_quality"]),
            "remote_quality_score":   float(decision["remote_quality"]),
            # Target
            "label":                  decision["label"],
        }
        rows.append(row)

    # Write CSV file
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully generated dataset with {len(rows)} samples.")
    print(f"Dataset path: {output_csv.resolve()}")
    return output_csv


if __name__ == "__main__":
    build_training_dataset()
