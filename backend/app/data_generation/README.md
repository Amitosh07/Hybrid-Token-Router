# Data Generation Module - Phase 4

This module provides a research-grade compositional dataset generation pipeline for the Hybrid Token Router. It is designed to synthesize high-quality, high-diversity, and realistic production-like LLM routing prompts and benchmark them to produce training datasets.

## Directory Structure

```text
backend/app/data_generation/
├── README.md                 # This documentation file
├── constraint_library.py     # Base constraints with parameter variation
├── dataset_statistics.py     # Markdown report compilation
├── deduplicator.py           # Exact (hash) and near-duplicate (shingle) detection
├── domain_library.py         # 15 domains and profiles
├── generation_config.py      # Configuration model loader
├── orchestrator.py           # End-to-end pipeline driver
├── output_format_library.py  # Output specifications (JSON, markdown table, etc.)
├── prompt_generator.py       # Compositional prompt generation core
├── reasoning_library.py      # Reasoning levels and indicators
├── template_engine.py        # Scenario task synthesis engine
└── validator.py              # Prompt formatting and metadata validator
```

## Features

- **Compositional Diversity**: Prompts are created by combining 8 categories, 3 difficulties, 15 domains, 10 output formats, 3 reasoning depths, 3 lengths, and 21 constraints.
- **Checkpointing & Resumability**: If the benchmark process halts (due to network drops, rate limits, or interruption), running the orchestrator again with the same parameters resumes seamlessly.
- **Rate-Limit Safe Concurrency**: Parallel requests are limited by a configurable `concurrency_limit` Semaphore to prevent hitting provider rate limits.
- **Incremental Disk Writes**: Benchmark outputs are appended to the checkpoint file row-by-row, keeping memory overhead to a minimum.
- **Simulated & Real Execution**: Supports a `--simulate` flag for rapid offline validation. The production dataset can be run with this flag omitted to capture ground-truth provider latency and quality.

## Usage

### 1. Verification Run (100 Prompts, Simulation Mode)
To verify the entire pipeline (generation, benchmarking, decision engine, dataset building, validation, and report writing):
```bash
$env:PYTHONPATH="."; python backend/app/data_generation/orchestrator.py --size 100 --simulate
```

### 2. Large Scale Execution (Real Providers)
To generate the final dataset using real configured models (Ollama local and remote Llama-3.1 on Groq/Fireworks):
```bash
# Ensure local Ollama is running and REMOTE API keys are active in backend/.env
$env:PYTHONPATH="."; python backend/app/data_generation/orchestrator.py --size 1000 --batch-size 20 --concurrency-limit 5
```

## Configuration

Settings are stored in `backend/generation_config.json`:
- `size`: Target dataset size.
- `batch_size`: Batch size before flushing checkpoint logs.
- `concurrency_limit`: Concurrent requests limit.
- `simulate_llm`: Set `true` to use simulator or `false` for real models.
- `dedup_threshold`: Jaccard similarity threshold for shingles.
