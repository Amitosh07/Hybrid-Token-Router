"""Orchestrator pipeline for generating, benchmarking, and validating large routing datasets.

This script executes the complete flow:
1. Compositional prompt generation, deduplication, and initial validation.
2. Parallel benchmarking against local/remote providers (real or simulated).
3. Checkpoint/Resume handling to allow resilient execution.
4. Outputting training_dataset_large.csv.
5. ML dataset validation and quality/coverage reports.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.data_generation.generation_config import GenerationConfig
from app.data_generation.prompt_generator import (
    generate_prompt_catalog,
    run_real_provider_pipeline,
)
from app.ml.validation import validate_dataset


def parse_arguments() -> argparse.Namespace:
    """Parse command line overrides for the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Hybrid Token Router - Phase 4 Orchestrator Pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom generation_config.json. Defaults to backend/generation_config.json.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help="Override target prompt catalog size (e.g. 100, 1000, 2500, 5000, 10000, 50000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for checkpointing.",
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=None,
        help="Override concurrency limit for parallel provider calls.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=None,
        help="Run model generation in simulated mode for testing.",
    )
    parser.add_argument(
        "--no-simulate",
        action="store_false",
        dest="simulate",
        help="Force real model generation (disable simulation).",
    )
    return parser.parse_args()


async def main() -> None:
    """Execute the end-to-end dataset orchestration pipeline."""
    args = parse_arguments()
    start_time = time.perf_counter()

    # 1. Load config from JSON
    config_path = Path(args.config) if args.config else None
    print(f"[*] Loading generation config from: {config_path or 'default JSON path'}")
    config = GenerationConfig.load_from_json(config_path)

    # Apply overrides from CLI
    if args.size is not None:
        config.size = args.size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.concurrency_limit is not None:
        config.concurrency_limit = args.concurrency_limit
    if args.simulate is not None:
        config.simulate_llm = args.simulate

    # Save resolved config for visibility
    config.save_to_json(config_path)

    # Resolve absolute paths
    prompt_catalog_path = config.get_absolute_path(config.prompt_catalog_path)
    checkpoint_path = config.get_absolute_path(config.checkpoint_path)
    results_path = config.get_absolute_path(config.results_path)
    output_csv_path = config.get_absolute_path(config.output_csv_path)

    print("========================================================================")
    print("                     HYBRID TOKEN ROUTER ORCHESTRATOR")
    print("========================================================================")
    print(f"  Target size        : {config.size}")
    print(f"  Batch size         : {config.batch_size}")
    print(f"  Concurrency limit  : {config.concurrency_limit}")
    print(f"  Simulation Mode    : {config.simulate_llm}")
    print(f"  Prompt Catalog     : {prompt_catalog_path}")
    print(f"  Checkpoint File    : {checkpoint_path}")
    print(f"  Final CSV Path     : {output_csv_path}")
    print("========================================================================")

    # --- Phase 1: Prompt Generation ---
    print("\n[Phase 1] Launching Compositional Prompt Generation...")
    gen_start = time.perf_counter()
    rows = generate_prompt_catalog(config.size, prompt_catalog_path)
    gen_duration = time.perf_counter() - gen_start
    print(f"[+] Successfully generated and validated {len(rows)} prompts in {gen_duration:.2f}s.")

    # --- Phase 2 & 3: Run Benchmark/Inference and Dataset Builder ---
    print("\n[Phase 2 & 3] Benchmarking against Local & Remote providers...")
    bench_start = time.perf_counter()
    try:
        await run_real_provider_pipeline(
            prompt_path=prompt_catalog_path,
            checkpoint_path=checkpoint_path,
            results_path=results_path,
            output_csv_path=output_csv_path,
            batch_size=config.batch_size,
            concurrency_limit=config.concurrency_limit,
            simulate=config.simulate_llm,
        )
        bench_duration = time.perf_counter() - bench_start
        print(f"[+] Benchmark and dataset flattening completed in {bench_duration:.2f}s.")
    except Exception as exc:
        print(f"\n[!] Execution interrupted: {exc}")
        print("[!] Progress saved. Run orchestrator again with same parameters to resume.")
        sys.exit(1)

    # --- Phase 4: Validation ---
    print("\n[Phase 4] Validating the training dataset structure...")
    validation_start = time.perf_counter()
    val_report = validate_dataset(output_csv_path)
    validation_duration = time.perf_counter() - validation_start
    print(f"[+] Dataset validation finished in {validation_duration:.2f}s.")

    if not val_report["success"]:
        print("[!] Validation found issues. Please check the dataset logs above.")
        sys.exit(1)

    # Summary report
    total_duration = time.perf_counter() - start_time
    print("========================================================================")
    print("                     ORCHESTRATION PIPELINE PASSED")
    print("========================================================================")
    print(f"  Total pipeline duration : {total_duration:.2f} seconds")
    print(f"  Prompts processed       : {val_report['total_samples']}")
    print(f"  Class balance LOCAL     : {val_report['local_count']} ({val_report['local_percentage']:.1f}%)")
    print(f"  Class balance REMOTE    : {val_report['remote_count']} ({val_report['remote_percentage']:.1f}%)")
    print("========================================================================")


if __name__ == "__main__":
    asyncio.run(main())
