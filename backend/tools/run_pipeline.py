#!/usr/bin/env python3
"""Pipeline execution script for Phase 2.

Runs the complete data generation pipeline sequentially:
  1. Benchmark Execution (calls both models for all expanded prompts)
  2. Dataset Builder (runs the Decision Engine and flattens to CSV)
  3. Validation Module (runs checks on the generated CSV dataset)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

# Path setup: import app
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.services.benchmark import run_benchmark
from app.ml.dataset_builder import build_training_dataset
from app.ml.validation import validate_dataset


async def main() -> None:
    start_time = time.perf_counter()
    print("========================================================================")
    print("                STARTING HYBRID ROUTER DATA GENERATION PIPELINE")
    print("========================================================================")
    
    # 1. Run Benchmark
    print("\n[STEP 1] Executing prompt benchmarks (calling local & remote models)...")
    benchmark_start = time.perf_counter()
    # run_benchmark loads all JSON files under app/data/prompts/
    summary = await run_benchmark()
    benchmark_duration = time.perf_counter() - benchmark_start
    print(f"\n[STEP 1 COMPLETE] Benchmark execution finished in {benchmark_duration:.2f} seconds.")
    print(f"  Total prompts processed: {summary.total_prompts}")
    print(f"  Errors encountered     : {len(summary.errors)}")
    
    # 2. Run Dataset Builder (generates CSV with Decision Engine labels)
    print("\n[STEP 2] Building CSV training dataset using Decision Engine...")
    builder_start = time.perf_counter()
    csv_path = build_training_dataset()
    builder_duration = time.perf_counter() - builder_start
    print(f"[STEP 2 COMPLETE] Dataset CSV built in {builder_duration:.2f} seconds.")
    print(f"  CSV Path: {csv_path}")

    # 3. Run Validation Module
    print("\n[STEP 3] Running validation checks on generated CSV dataset...")
    validation_start = time.perf_counter()
    val_metrics = validate_dataset(csv_path)
    validation_duration = time.perf_counter() - validation_start
    print(f"[STEP 3 COMPLETE] Validation completed in {validation_duration:.2f} seconds.")
    print(f"  Status: {val_metrics['status']}")
    
    total_duration = time.perf_counter() - start_time
    print("\n========================================================================")
    print(f"            PIPELINE EXECUTION COMPLETE IN {total_duration:.2f} SECONDS")
    print("========================================================================")


if __name__ == "__main__":
    asyncio.run(main())
