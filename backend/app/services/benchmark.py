"""Benchmark Runner for the Hybrid Token Router.

Measures how well the current heuristic router performs by running every
prompt in ``backend/app/data/prompts/`` through the full pipeline:

    Prompt → Feature Extractor → Router → Both Models → Evaluator → Record

Design principles
-----------------
* Uses only existing service modules — no new logic is introduced.
* Both models are **always** executed regardless of the router decision.
  This produces ground-truth latency and quality data for every prompt.
* Provider failures are caught per-prompt; the benchmark continues.
* ``router_correct`` is left ``null`` to be filled by a future ML pipeline.

Public API
----------
    run_benchmark(prompts_dir, results_dir, summaries_dir) -> BenchmarkSummary
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services import evaluator, ollama, remote_llm
from app.services.dataset_generator import estimate_tokens, _run_provider
from app.services.feature_extractor import extract_features
from app.services.router import PROVIDER_LOCAL, PROVIDER_REMOTE, route
from app.config import get_settings


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROUTER_VERSION: str = "v1.0"

_DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "data" / "prompts"
_DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "results"
_DEFAULT_SUMMARIES_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks" / "summaries"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PromptEntry:
    """One entry loaded from a category prompt JSON file."""

    id: str
    category: str
    difficulty: str
    expected_reasoning: str
    prompt: str


@dataclass
class BenchmarkRecord:
    """Full benchmark result for one prompt."""

    id: str
    category: str
    difficulty: str
    expected_reasoning: str
    prompt: str
    features: dict[str, Any]
    router: dict[str, Any]
    local: dict[str, Any]
    remote: dict[str, Any]
    evaluation: dict[str, Any]
    router_correct: None  # filled by a future ML/decision pipeline
    router_version: str
    timestamp: str


@dataclass
class BenchmarkSummary:
    """Aggregated statistics for one complete benchmark run."""

    total_prompts: int
    categories_processed: list[str]
    difficulty_distribution: dict[str, int]
    router_version: str
    local_selected: int
    remote_selected: int
    average_routing_score: float
    average_confidence: float
    average_local_latency_ms: float
    average_remote_latency_ms: float
    average_local_tokens: float
    average_remote_tokens: float
    benchmark_duration_seconds: float
    timestamp: str
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompt_files(prompts_dir: Path) -> list[PromptEntry]:
    """Load every structured prompt JSON file from *prompts_dir*.

    Skips ``example_prompts.json`` which uses a plain string-array format
    incompatible with the structured schema.

    Args:
        prompts_dir: Directory containing category prompt JSON files.

    Returns:
        List of ``PromptEntry`` objects, one per prompt across all files.

    Raises:
        FileNotFoundError: If *prompts_dir* does not exist.
    """
    if not prompts_dir.exists():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    entries: list[PromptEntry] = []
    for path in sorted(prompts_dir.glob("*.json")):
        if path.name == "example_prompts.json":
            continue
        entries.extend(_load_single_prompt_file(path))
    return entries


def _load_single_prompt_file(path: Path) -> list[PromptEntry]:
    """Parse one structured prompt JSON file into ``PromptEntry`` objects.

    Args:
        path: Absolute path to a JSON file containing a list of prompt objects.

    Returns:
        List of ``PromptEntry`` objects.  Malformed entries are skipped.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] Could not load {path.name}: {exc}")
        return []

    if not isinstance(data, list):
        print(f"  [WARN] {path.name}: root is not a list, skipping.")
        return []

    entries: list[PromptEntry] = []
    for item in data:
        entry = _parse_prompt_entry(item, path.name)
        if entry is not None:
            entries.append(entry)
    return entries


def _parse_prompt_entry(item: Any, filename: str) -> PromptEntry | None:
    """Validate and convert a raw dict into a ``PromptEntry``.

    Args:
        item:     Raw dictionary from the JSON file.
        filename: Source filename (used in warning messages only).

    Returns:
        A ``PromptEntry``, or ``None`` if required fields are missing.
    """
    if not isinstance(item, dict):
        return None
    required = ("id", "category", "difficulty", "expected_reasoning", "prompt")
    for key in required:
        if not item.get(key):
            print(f"  [WARN] {filename}: entry missing {key!r}, skipping.")
            return None
    return PromptEntry(
        id=item["id"],
        category=item["category"],
        difficulty=item["difficulty"],
        expected_reasoning=item["expected_reasoning"],
        prompt=item["prompt"],
    )


# ---------------------------------------------------------------------------
# Single-prompt benchmark
# ---------------------------------------------------------------------------

async def benchmark_one(entry: PromptEntry) -> BenchmarkRecord | str:
    """Run the full benchmark pipeline for one prompt.

    Both providers are executed regardless of the router decision.

    Args:
        entry: A ``PromptEntry`` loaded from a prompt JSON file.

    Returns:
        A ``BenchmarkRecord`` on success, or an error string on failure.
    """
    settings = get_settings()
    timestamp = _utc_now()

    # 1. Feature extraction
    features = extract_features(entry.prompt)

    # 2. Routing decision
    routing = route(features)

    # 3. Execute BOTH providers (benchmark mode — always run both)
    local_task = _run_provider(
        prompt=entry.prompt,
        model=settings.OLLAMA_MODEL or "local-model",
        generate=ollama.generate,
    )
    remote_task = _run_provider(
        prompt=entry.prompt,
        model=settings.REMOTE_MODEL or "remote-model",
        generate=remote_llm.generate,
    )
    local_result, remote_result = await asyncio.gather(local_task, remote_task)

    # 4. Evaluation
    evaluation = evaluator.evaluate_run(
        {
            "prompt": entry.prompt,
            "local": local_result,
            "remote": remote_result,
        }
    )
    # Keep only the metrics block — the top-level prompt is already in the record.
    eval_block = {
        "metrics": evaluation.get("metrics", {}),
        "evaluation_method": evaluation.get("evaluation_method", "rule"),
    }

    return BenchmarkRecord(
        id=entry.id,
        category=entry.category,
        difficulty=entry.difficulty,
        expected_reasoning=entry.expected_reasoning,
        prompt=entry.prompt,
        features=features,
        router=routing,
        local=local_result,
        remote=remote_result,
        evaluation=eval_block,
        router_correct=None,
        router_version=ROUTER_VERSION,
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _results_filename(ts: datetime) -> str:
    """Build a timestamped results filename from a datetime object."""
    return "benchmark_" + ts.strftime("%Y-%m-%dT%H-%M-%S") + ".json"


def _summaries_filename(ts: datetime) -> str:
    """Build a timestamped summary filename from a datetime object."""
    return "summary_" + ts.strftime("%Y-%m-%dT%H-%M-%S") + ".json"


def save_results(
    records: list[BenchmarkRecord],
    results_dir: Path,
    ts: datetime,
) -> Path:
    """Serialise benchmark records to a timestamped JSON file.

    Args:
        records:     List of completed ``BenchmarkRecord`` objects.
        results_dir: Destination directory.
        ts:          Timestamp used in the filename.

    Returns:
        Absolute path to the written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / _results_filename(ts)
    payload = [_record_to_dict(r) for r in records]
    _write_json(dest, payload)
    return dest


def save_summary(summary: BenchmarkSummary, summaries_dir: Path, ts: datetime) -> Path:
    """Serialise a ``BenchmarkSummary`` to a timestamped JSON file.

    Args:
        summary:      The completed summary object.
        summaries_dir: Destination directory.
        ts:           Timestamp used in the filename.

    Returns:
        Absolute path to the written file.
    """
    summaries_dir.mkdir(parents=True, exist_ok=True)
    dest = summaries_dir / _summaries_filename(ts)
    _write_json(dest, asdict(summary))
    return dest


def _write_json(path: Path, payload: Any) -> None:
    """Write *payload* to *path* as pretty-printed UTF-8 JSON."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")


def _record_to_dict(record: BenchmarkRecord) -> dict[str, Any]:
    """Convert a ``BenchmarkRecord`` to a plain JSON-serialisable dict."""
    return asdict(record)


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _compute_summary(
    records: list[BenchmarkRecord],
    errors: list[str],
    duration_seconds: float,
    ts: datetime,
) -> BenchmarkSummary:
    """Derive a ``BenchmarkSummary`` from the completed records.

    Args:
        records:          Successfully completed benchmark records.
        errors:           Error messages collected during the run.
        duration_seconds: Wall-clock runtime of the benchmark.
        ts:               Run start timestamp.

    Returns:
        A ``BenchmarkSummary`` populated from the records.
    """
    total = len(records)

    categories: list[str] = sorted({r.category for r in records})
    difficulty_dist: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    local_selected = remote_selected = 0
    routing_scores: list[float] = []
    confidences: list[float] = []
    local_latencies: list[float] = []
    remote_latencies: list[float] = []
    local_tokens: list[float] = []
    remote_tokens: list[float] = []

    for r in records:
        difficulty_dist[r.difficulty] = difficulty_dist.get(r.difficulty, 0) + 1
        if r.router.get("provider") == PROVIDER_LOCAL:
            local_selected += 1
        else:
            remote_selected += 1

        routing_scores.append(float(r.router.get("routing_score", 0)))
        confidences.append(float(r.router.get("confidence", 0.0)))

        ll = r.local.get("latency_ms")
        if isinstance(ll, (int, float)):
            local_latencies.append(float(ll))

        rl = r.remote.get("latency_ms")
        if isinstance(rl, (int, float)):
            remote_latencies.append(float(rl))

        lt = r.local.get("estimated_input_tokens", 0) + r.local.get("estimated_output_tokens", 0)
        local_tokens.append(float(lt))

        rt = r.remote.get("estimated_input_tokens", 0) + r.remote.get("estimated_output_tokens", 0)
        remote_tokens.append(float(rt))

    def _avg(values: list[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    return BenchmarkSummary(
        total_prompts=total,
        categories_processed=categories,
        difficulty_distribution=difficulty_dist,
        router_version=ROUTER_VERSION,
        local_selected=local_selected,
        remote_selected=remote_selected,
        average_routing_score=_avg(routing_scores),
        average_confidence=_avg(confidences),
        average_local_latency_ms=_avg(local_latencies),
        average_remote_latency_ms=_avg(remote_latencies),
        average_local_tokens=_avg(local_tokens),
        average_remote_tokens=_avg(remote_tokens),
        benchmark_duration_seconds=round(duration_seconds, 2),
        timestamp=ts.isoformat(),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_DIVIDER = "-" * 45


def _print_summary(summary: BenchmarkSummary, results_path: Path, summary_path: Path) -> None:
    """Print a clean benchmark summary to stdout."""
    print(_DIVIDER)
    print("Hybrid Token Router Benchmark")
    print(_DIVIDER)
    print(f"  Total prompts        : {summary.total_prompts}")
    print(f"  Categories processed : {', '.join(summary.categories_processed)}")
    print(f"  Local selected       : {summary.local_selected}")
    print(f"  Remote selected      : {summary.remote_selected}")
    print(f"  Average conf.        : {summary.average_confidence:.2f}")
    print(f"  Avg routing score    : {summary.average_routing_score:.1f}")
    print(f"  Avg local latency    : {summary.average_local_latency_ms:.0f} ms")
    print(f"  Avg remote latency   : {summary.average_remote_latency_ms:.0f} ms")
    print(f"  Total runtime        : {summary.benchmark_duration_seconds:.1f} seconds")
    if summary.errors:
        print(f"  Errors               : {len(summary.errors)}")
    print(_DIVIDER)
    print(f"  Results  -> {results_path}")
    print(f"  Summary  -> {summary_path}")
    print(_DIVIDER)
    if summary.errors:
        print("  Completed with errors (see summary.errors for details).")
    else:
        print("  Benchmark completed successfully.")
    print(_DIVIDER)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_benchmark(
    prompts_dir: Path | None = None,
    results_dir: Path | None = None,
    summaries_dir: Path | None = None,
) -> BenchmarkSummary:
    """Load all prompts, run the full benchmark pipeline, and save results.

    This is the primary public entry point.  It:

    1. Loads every structured prompt JSON file from *prompts_dir*.
    2. For each prompt: extracts features, routes, executes both providers,
       evaluates, and records a ``BenchmarkRecord``.
    3. Saves timestamped records to *results_dir*.
    4. Computes and saves a ``BenchmarkSummary`` to *summaries_dir*.
    5. Prints a formatted summary to stdout.

    Provider failures for individual prompts are caught and logged; the
    benchmark continues to the next prompt so partial results are still saved.

    Args:
        prompts_dir:   Directory of structured prompt JSON files.
                       Defaults to ``backend/app/data/prompts/``.
        results_dir:   Directory to write per-prompt result JSON.
                       Defaults to ``backend/app/data/benchmarks/results/``.
        summaries_dir: Directory to write the summary JSON.
                       Defaults to ``backend/app/data/benchmarks/summaries/``.

    Returns:
        The completed ``BenchmarkSummary``.
    """
    pd = prompts_dir or _DEFAULT_PROMPTS_DIR
    rd = results_dir or _DEFAULT_RESULTS_DIR
    sd = summaries_dir or _DEFAULT_SUMMARIES_DIR

    ts = datetime.now(timezone.utc).replace(microsecond=0)
    wall_start = time.perf_counter()

    # --- Load prompts -------------------------------------------------------
    print(f"\nLoading prompts from: {pd}")
    entries = load_prompt_files(pd)
    print(f"  Loaded {len(entries)} prompts from {pd}")

    # --- Run each prompt ----------------------------------------------------
    records: list[BenchmarkRecord] = []
    errors: list[str] = []

    for idx, entry in enumerate(entries, start=1):
        print(f"  [{idx:>3}/{len(entries)}] {entry.id}", end=" ... ", flush=True)
        try:
            record = await benchmark_one(entry)
            if isinstance(record, str):
                # benchmark_one returned an error string (shouldn't normally happen)
                errors.append(f"{entry.id}: {record}")
                print("ERROR")
            else:
                records.append(record)
                print(f"{record.router['provider'].upper()} (score={record.router['routing_score']})")
        except Exception as exc:  # noqa: BLE001
            msg = f"{entry.id}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print(f"ERROR — {exc}")

    # --- Persist results ----------------------------------------------------
    duration = time.perf_counter() - wall_start
    results_path = save_results(records, rd, ts)
    summary = _compute_summary(records, errors, duration, ts)
    summary_path = save_summary(summary, sd, ts)

    # --- Console output -----------------------------------------------------
    _print_summary(summary, results_path, summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_benchmark())
