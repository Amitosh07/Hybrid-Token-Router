"""Scalable research-quality routing dataset generator.

This module supports two workflows:

1. Generate a high-diversity prompt catalog and quality/coverage reports.
2. Execute a resumable real-provider benchmark pipeline that produces
   `training_dataset_large.csv`.

It intentionally does not modify the existing benchmark framework, Decision
Engine, Feature Extraction Pipeline, or ML training code.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.data_generation.constraint_library import (
    OUTPUT_FORMATS,
    PROMPT_LENGTHS,
    REASONING_DEPTHS,
    constraints_for,
)
from app.data_generation.dataset_statistics import (
    write_coverage_report,
    write_generation_statistics,
    write_quality_report,
)
from app.data_generation.deduplicator import Deduplicator
from app.data_generation.domain_library import CATEGORIES, DIFFICULTIES, DOMAINS, domain_profile
from app.data_generation.template_engine import TemplateEngine
from app.data_generation.validator import validate_prompts
from app.ml.decision_engine import DecisionEngine
from app.services import evaluator, ollama, remote_llm
from app.services.dataset_generator import estimate_tokens
from app.services.feature_extractor import extract_features


APP_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = APP_DIR.parent
GENERATED_DIR = APP_DIR / "data" / "generated"
TRAINING_DIR = APP_DIR / "data" / "training"
DOCS_DIR = BACKEND_DIR / "docs"

PROMPT_CATALOG_PATH = GENERATED_DIR / "generated_prompts.jsonl"
BENCHMARK_CHECKPOINT_PATH = GENERATED_DIR / "large_benchmark_checkpoint.jsonl"
BENCHMARK_RESULTS_PATH = GENERATED_DIR / "large_benchmark_results.json"
TRAINING_DATASET_LARGE_PATH = TRAINING_DIR / "training_dataset_large.csv"

SUPPORTED_SIZES = {100, 1000, 2500, 5000, 10000, 50000}


@dataclass(frozen=True)
class ProviderRun:
    """Serializable provider run result."""

    model: str
    response: str | None
    latency_ms: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    error: str | None = None


class PromptGenerator:
    """Generate balanced, metadata-rich prompt catalogs."""

    def __init__(self) -> None:
        self.engine = TemplateEngine()

    def generate(self, size: int) -> list[dict[str, object]]:
        """Generate a unique balanced prompt catalog."""
        if size <= 0:
            raise ValueError("size must be positive")

        oversample = max(size * 2, size + 500)
        rows: list[dict[str, object]] = []
        variant = 0

        while len(rows) < oversample:
            category = CATEGORIES[variant % len(CATEGORIES)]
            difficulty = DIFFICULTIES[(variant // len(CATEGORIES)) % len(DIFFICULTIES)]
            domain = DOMAINS[(variant // (len(CATEGORIES) * len(DIFFICULTIES))) % len(DOMAINS)]
            output_format = OUTPUT_FORMATS[(variant // 3) % len(OUTPUT_FORMATS)]
            expected_reasoning = self._expected_reasoning(category, difficulty, variant)
            prompt_length = PROMPT_LENGTHS[(variant // 11) % len(PROMPT_LENGTHS)]
            constraint_count = self._constraint_count(difficulty, prompt_length, variant)
            constraints = constraints_for(category, variant, constraint_count)
            prompt_id = f"gen_{variant + 1:06d}"
            spec = self.engine.render(
                prompt_id=prompt_id,
                category=category,
                difficulty=difficulty,
                expected_reasoning=expected_reasoning,
                domain=domain_profile(domain),
                output_format=output_format,
                constraints=constraints,
                prompt_length=prompt_length,
                variant=variant,
            )
            row_dict = spec.to_dict()
            prompt_str = str(row_dict["prompt"])
            row_dict["id"] = row_dict["prompt_id"]
            row_dict["output_type"] = row_dict["output_format"]
            row_dict["estimated_input_tokens"] = estimate_tokens(prompt_str)
            row_dict["timestamp"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            rows.append(row_dict)
            variant += 1

        deduped = Deduplicator().deduplicate(rows).kept
        validation = validate_prompts(deduped)
        return validation.valid_rows[:size]

    def _expected_reasoning(self, category: str, difficulty: str, variant: int) -> str:
        if difficulty == "hard":
            return "high"
        if difficulty == "easy" and category in {"translation", "summarization", "creative_writing"}:
            return "low"
        return REASONING_DEPTHS[(variant // 5) % len(REASONING_DEPTHS)]

    def _constraint_count(self, difficulty: str, prompt_length: str, variant: int) -> int:
        base = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
        if prompt_length == "long":
            base += 1
        return min(5, base + (variant % 2))


def generate_prompt_catalog(size: int, output_path: Path = PROMPT_CATALOG_PATH) -> list[dict[str, object]]:
    """Generate prompts, dedupe, validate, and write reports."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate 50% synthetic base prompts
    base_size = max(100, size // 2)
    base_rows = PromptGenerator().generate(base_size)

    # Ingest, deduplicate, balance, and merge public datasets
    from app.data_generation.dataset_importer import DatasetImporter
    importer = DatasetImporter()
    merged_rows = importer.load_and_merge(base_rows, target_size=size)

    # Deduplicate and validate
    dedupe_result = Deduplicator().deduplicate(merged_rows)
    validation = validate_prompts(dedupe_result.kept)
    final_rows = validation.valid_rows[:size]

    _write_jsonl(output_path, final_rows)
    write_quality_report(
        path=DOCS_DIR / "dataset_quality_report.md",
        validation=validation,
        duplicate_count=dedupe_result.duplicate_count,
        near_duplicate_count=dedupe_result.near_duplicate_count,
        requested_size=size,
        final_size=len(final_rows),
    )
    write_generation_statistics(final_rows, DOCS_DIR / "generation_statistics.md")
    write_coverage_report(final_rows, DOCS_DIR / "coverage_report.md")
    return final_rows


async def run_real_provider_pipeline(
    *,
    prompt_path: Path = PROMPT_CATALOG_PATH,
    checkpoint_path: Path = BENCHMARK_CHECKPOINT_PATH,
    results_path: Path = BENCHMARK_RESULTS_PATH,
    output_csv_path: Path = TRAINING_DATASET_LARGE_PATH,
    batch_size: int = 10,
    concurrency_limit: int = 5,
    max_records: int | None = None,
    simulate: bool = False,
) -> Path:
    """Run Prompt -> Features -> Providers -> Evaluator -> Decision Engine -> CSV.

    The pipeline uses real configured providers by default, but allows simulation
    mode for development and testing.
    """
    _assert_real_provider_mode(simulate=simulate)
    prompts = _read_jsonl(prompt_path)
    if max_records is not None:
        prompts = prompts[:max_records]
    completed = _load_checkpoint(checkpoint_path)
    completed_ids = set(completed)

    records: list[dict[str, Any]] = [completed[key] for key in sorted(completed)]
    pending = [row for row in prompts if str(row["prompt_id"]) not in completed_ids]

    sem = asyncio.Semaphore(concurrency_limit)

    async def sem_run_one(row: dict[str, object]) -> dict[str, Any]:
        async with sem:
            return await _run_one(row, simulate=simulate)

    # Process pending prompts in batches
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        batch_records = await asyncio.gather(*[sem_run_one(row) for row in batch])
        for record in batch_records:
            _append_jsonl(checkpoint_path, record)
            records.append(record)
        print(f"  Processed and checkpointed batch of {len(batch)} prompts (total completed: {len(records)})")

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_training_dataset_large(records, output_csv_path)
    return output_csv_path


async def _run_one(row: dict[str, object], simulate: bool = False) -> dict[str, Any]:
    prompt = str(row["prompt"])
    features = extract_features(prompt)

    # Heuristic routing decision for benchmarking
    from app.services.router import route as heuristic_route
    routing = heuristic_route(features)

    if simulate:
        from app.services.simulator import generate_simulated
        async def local_sim(p: str) -> str:
            return await generate_simulated(p, "local")
        async def remote_sim(p: str) -> str:
            return await generate_simulated(p, "remote")

        local_task = _run_provider(prompt=prompt, provider="local", model="simulated-local", generate=local_sim)
        remote_task = _run_provider(prompt=prompt, provider="remote", model="simulated-remote", generate=remote_sim)
    else:
        local_task = _run_provider(prompt=prompt, provider="local", model=get_settings().OLLAMA_MODEL, generate=ollama.generate)
        remote_task = _run_provider(prompt=prompt, provider="remote", model=get_settings().REMOTE_MODEL, generate=remote_llm.generate)

    local_result, remote_result = await asyncio.gather(local_task, remote_task)
    
    # Run LLM Evaluator
    from app.services.llm_evaluator import LLMEvaluator
    evaluator_instance = LLMEvaluator()
    llm_eval = await evaluator_instance.evaluate(
        prompt=prompt,
        category=str(row["category"]),
        difficulty=str(row["difficulty"]),
        local_response=local_result.get("response"),
        remote_response=remote_result.get("response"),
        local_error=local_result.get("error"),
        remote_error=remote_result.get("error"),
    )

    record = {
        "id": row["prompt_id"],
        "prompt": prompt,
        "category": row["category"],
        "difficulty": row["difficulty"],
        "expected_reasoning": row["expected_reasoning"],
        "domain": row["domain"],
        "constraint_count": row["constraint_count"],
        "estimated_complexity": row["estimated_complexity"],
        "output_format": row.get("output_format", ""),
        "features": features,
        "router": routing,
        "local": local_result,
        "remote": remote_result,
        "llm_evaluator": llm_eval,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    record["evaluation"] = evaluator.evaluate_run(record)
    record["decision"] = DecisionEngine().decide(record)
    return record


async def _run_provider(*, prompt: str, provider: str, model: str, generate: Any) -> dict[str, Any]:
    start = time.perf_counter()
    input_tokens = estimate_tokens(prompt)
    try:
        response = await generate(prompt)
        output_tokens = estimate_tokens(response)
        error = None
    except Exception as exc:  # noqa: BLE001 - checkpoint failures for later inspection.
        response = None
        output_tokens = 0
        error = str(exc)
    latency_ms = round((time.perf_counter() - start) * 1000)
    result = ProviderRun(
        model=model or f"configured-{provider}",
        response=response,
        latency_ms=latency_ms,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        error=error,
    )
    data = asdict(result)
    if error is None:
        data.pop("error")
    if response is None:
        data.pop("response")
    return data


def write_training_dataset_large(records: list[dict[str, Any]], output_csv_path: Path) -> None:
    """Flatten benchmark records into the large training CSV."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    engine = DecisionEngine()
    headers = [
        "prompt_id",
        "prompt",
        "category",
        "difficulty",
        "expected_reasoning",
        "domain",
        "constraint_count",
        "estimated_complexity",
        "output_format",
        # --- Lexical & Structural ---
        "prompt_length",
        "word_count",
        "estimated_input_tokens",
        "contains_code",
        "contains_math",
        "contains_json",
        "contains_markdown",
        "contains_numbers",
        "contains_question",
        # --- Aggregation Metrics ---
        "reasoning_score",
        "technical_complexity",
        "reasoning_depth",
        "task_complexity",
        "constraint_complexity",
        "context_complexity",
        "complexity_score",
        # --- Phase 6 Features ---
        "constraint_density",
        "requested_format",
        "presence_of_tables",
        "sql_indicators",
        "api_keywords",
        "system_design_keywords",
        "algorithmic_complexity",
        "dependency_between_subtasks",
        "multi_turn_context",
        "code_indicators",
        "math_indicators",
        # --- Operational Stats & Quality Details ---
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
        "label",
    ]
    with output_csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for record in records:
            features = record.get("features", {})
            local = record.get("local", {})
            remote = record.get("remote", {})
            decision = record.get("decision") or engine.decide(record)
            writer.writerow({
                "prompt_id": record.get("id", ""),
                "prompt": record.get("prompt", ""),
                "category": record.get("category", "general"),
                "difficulty": record.get("difficulty", "medium"),
                "expected_reasoning": record.get("expected_reasoning", "medium"),
                "domain": record.get("domain", ""),
                "constraint_count": record.get("constraint_count", 0),
                "estimated_complexity": record.get("estimated_complexity", 0.0),
                "output_format": record.get("output_format", ""),
                "prompt_length": int(features.get("prompt_length", len(str(record.get("prompt", ""))))),
                "word_count": int(features.get("word_count", len(str(record.get("prompt", "")).split()))),
                "estimated_input_tokens": int(features.get("estimated_input_tokens", 0)),
                "contains_code": 1 if features.get("contains_code", False) else 0,
                "contains_math": 1 if features.get("contains_math", False) else 0,
                "contains_json": 1 if features.get("contains_json", False) else 0,
                "contains_markdown": 1 if features.get("contains_markdown", False) else 0,
                "contains_numbers": 1 if features.get("contains_numbers", False) else 0,
                "contains_question": 1 if features.get("contains_question", False) else 0,
                "reasoning_score": int(features.get("reasoning_score", 0)),
                "technical_complexity": _score(features.get("technical_complexity")),
                "reasoning_depth": _score(features.get("reasoning_depth")),
                "task_complexity": _score(features.get("task_complexity")),
                "constraint_complexity": _score(features.get("constraint_complexity")),
                "context_complexity": _score(features.get("context_complexity")),
                "complexity_score": float(features.get("complexity_score", 0.0)),
                # Phase 6 Features
                "constraint_density": float(features.get("constraint_density", 0.0)),
                "requested_format": str(features.get("requested_format", "text")),
                "presence_of_tables": int(features.get("presence_of_tables", 0)),
                "sql_indicators": int(features.get("sql_indicators", 0)),
                "api_keywords": int(features.get("api_keywords", 0)),
                "system_design_keywords": int(features.get("system_design_keywords", 0)),
                "algorithmic_complexity": int(features.get("algorithmic_complexity", 0)),
                "dependency_between_subtasks": int(features.get("dependency_between_subtasks", 0)),
                "multi_turn_context": int(features.get("multi_turn_context", 0)),
                "code_indicators": int(features.get("code_indicators", 0)),
                "math_indicators": int(features.get("math_indicators", 0)),
                # Operational Stats & Decision details
                "local_latency_ms": int(local.get("latency_ms", 0)),
                "remote_latency_ms": int(remote.get("latency_ms", 0)),
                "local_cost": float(decision["local_cost"]),
                "remote_cost": float(decision["remote_cost"]),
                "local_quality_score": float(decision["local_quality"]),
                "remote_quality_score": float(decision["remote_quality"]),
                "local_llm_quality": float(decision.get("local_llm_quality", 0.0)),
                "remote_llm_quality": float(decision.get("remote_llm_quality", 0.0)),
                "local_heuristic_quality": float(decision.get("local_heuristic_quality", 0.0)),
                "remote_heuristic_quality": float(decision.get("remote_heuristic_quality", 0.0)),
                "local_tokens": int(decision.get("local_tokens", 0)),
                "remote_tokens": int(decision.get("remote_tokens", 0)),
                "label": decision["label"],
            })


def _assert_real_provider_mode(simulate: bool = False) -> None:
    if simulate:
        return
    if os.getenv("SIMULATE_LLM", "false").lower() == "true":
        raise RuntimeError("Large dataset generation requires real providers. Set SIMULATE_LLM=false.")
    settings = get_settings()
    missing = []
    if not settings.OLLAMA_BASE_URL:
        missing.append("OLLAMA_BASE_URL")
    if not settings.OLLAMA_MODEL:
        missing.append("OLLAMA_MODEL")
    if not settings.REMOTE_BASE_URL:
        missing.append("REMOTE_BASE_URL")
    if not settings.REMOTE_API_KEY:
        missing.append("REMOTE_API_KEY")
    if not settings.REMOTE_MODEL:
        missing.append("REMOTE_MODEL")
    if missing:
        raise RuntimeError(f"Real provider configuration is incomplete: {', '.join(missing)}")


def _score(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("score", 0.0))
    return float(value or 0.0)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt catalog not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = _read_jsonl(path)
    return {str(row["id"]): row for row in rows}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and benchmark large routing datasets.")
    parser.add_argument("--size", type=int, default=1000, help="Prompt catalog size.")
    parser.add_argument("--generate-only", action="store_true", help="Only generate prompt catalog and reports.")
    parser.add_argument("--run-benchmark", action="store_true", help="Run real-provider benchmark and write training_dataset_large.csv.")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of prompts to benchmark per checkpoint batch.")
    parser.add_argument("--concurrency-limit", type=int, default=5, help="Number of concurrent model requests.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for smoke tests or staged runs.")
    parser.add_argument("--simulate", action="store_true", help="Use simulated models rather than real providers.")
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    if args.size not in SUPPORTED_SIZES:
        print(f"Warning: --size {args.size} is not in standard set {SUPPORTED_SIZES}. Proceeding anyway.")
    rows = generate_prompt_catalog(args.size)
    print(f"Generated {len(rows)} valid unique prompts at {PROMPT_CATALOG_PATH}")
    if args.run_benchmark and not args.generate_only:
        output = await run_real_provider_pipeline(
            batch_size=args.batch_size,
            concurrency_limit=args.concurrency_limit,
            max_records=args.max_records,
            simulate=args.simulate,
        )
        print(f"Wrote large training dataset to {output}")


if __name__ == "__main__":
    asyncio.run(_main())

