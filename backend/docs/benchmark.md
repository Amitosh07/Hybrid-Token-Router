# Benchmark Runner

The Benchmark Runner measures how well the current heuristic router performs by executing the full pipeline against every curated prompt in the dataset. It is a **read-only observer** — it does not modify the router, the evaluator, or any existing service.

---

## Benchmark Workflow

```
backend/app/data/prompts/*.json
        │
        ▼  load_prompt_files()
 List[PromptEntry]
        │
        ▼  for each entry:
┌─────────────────────────────────────────────────────────────┐
│  1. extract_features(prompt)         → features dict        │
│  2. route(features)                  → routing decision     │
│  3. ollama.generate(prompt)          → local response       │  ← always
│  4. remote_llm.generate(prompt)      → remote response      │  ← always
│  5. evaluator.evaluate_run(run)      → evaluation metrics   │
│  6. assemble BenchmarkRecord                                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼  save_results()
backend/app/data/benchmarks/results/benchmark_TIMESTAMP.json

        │
        ▼  _compute_summary()  →  save_summary()
backend/app/data/benchmarks/summaries/summary_TIMESTAMP.json

        │
        ▼  _print_summary()
Console output
```

---

## Benchmark Mode — Why Both Models Are Always Executed

In production the router selects one provider and only that provider is called. In **benchmark mode**, both the local and remote models are always executed regardless of the router's decision.

### Reasons

1. **Ground-truth data.** To eventually train or validate a router you need the real outputs from both providers for every prompt. Skipping the non-selected provider produces incomplete training data.

2. **Latency measurement.** Comparing local vs remote response times requires actual timings from both.

3. **Quality measurement.** The evaluator compares both responses. Without both responses, the quality delta (a future ML signal) cannot be computed.

4. **Router accuracy analysis.** `router_correct` will be populated by a future ML/decision pipeline that compares the actual quality of both providers against what the router selected. This requires both provider outputs.

Benchmark runs are expected to be slower than production runs for this reason.

---

## Folder Structure

```
backend/
└── app/
    ├── services/
    │   └── benchmark.py              ← Benchmark Runner module
    ├── tests/
    │   └── test_benchmark.py         ← Unit tests (all providers mocked)
    └── data/
        └── benchmarks/
            ├── results/              ← Per-run benchmark records
            └── summaries/            ← Per-run aggregated summaries
```

---

## Running the Benchmark

```bash
# From the backend/ directory
python -m app.services.benchmark
```

Or invoke programmatically:

```python
import asyncio
from app.services.benchmark import run_benchmark

summary = asyncio.run(run_benchmark())
```

Custom directories:

```python
from pathlib import Path
from app.services.benchmark import run_benchmark

summary = asyncio.run(run_benchmark(
    prompts_dir=Path("app/data/prompts"),
    results_dir=Path("app/data/benchmarks/results"),
    summaries_dir=Path("app/data/benchmarks/summaries"),
))
```

---

## Console Output

```
-----------------------------------------
Hybrid Token Router Benchmark
-----------------------------------------
  Total prompts        : 160
  Categories processed : coding, creative_writing, ...
  Local selected       : 93
  Remote selected      : 67
  Average conf.        : 0.84
  Avg routing score    : 18.3
  Avg local latency    : 812 ms
  Avg remote latency   : 1517 ms
  Total runtime        : 214.0 seconds
-----------------------------------------
  Results  → .../benchmarks/results/benchmark_2026-07-03T00-00-00.json
  Summary  → .../benchmarks/summaries/summary_2026-07-03T00-00-00.json
-----------------------------------------
  Benchmark completed successfully.
-----------------------------------------
```

---

## Output JSON Schema

### BenchmarkRecord

Stored in `results/benchmark_TIMESTAMP.json` as a JSON array. Each element:

```json
{
  "id": "coding_001",
  "category": "coding",
  "difficulty": "easy",
  "expected_reasoning": "low",
  "prompt": "Write a Python Hello World program.",

  "features": {
    "reasoning_score": 0,
    "estimated_input_tokens": 7,
    "complexity": "easy",
    "task_type": "coding",
    "contains_code": false,
    "contains_math": false,
    "contains_json": false
  },

  "router": {
    "provider": "local",
    "routing_score": 9,
    "confidence": 0.97,
    "reason": [
      "Task type is coding",
      "Routing score 9 is below threshold 25 → local"
    ]
  },

  "local": {
    "model": "llama3",
    "response": "print('Hello, World!')",
    "latency_ms": 412,
    "estimated_input_tokens": 7,
    "estimated_output_tokens": 6
  },

  "remote": {
    "model": "gpt-4o",
    "response": "print('Hello, World!')",
    "latency_ms": 987,
    "estimated_input_tokens": 7,
    "estimated_output_tokens": 6
  },

  "evaluation": {
    "metrics": {
      "response_length": { ... },
      "latency": { ... },
      "structural_validity": { ... },
      "estimated_quality": {
        "local": { ... },
        "remote": { ... }
      }
    },
    "evaluation_method": "rule"
  },

  "router_correct": null,
  "router_version": "v1.0",
  "timestamp": "2026-07-03T00:00:00+00:00"
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `id` | `string` | Unique prompt ID from the prompt dataset |
| `category` | `string` | Task category (coding, mathematics, …) |
| `difficulty` | `string` | easy / medium / hard |
| `expected_reasoning` | `string` | low / medium / high |
| `prompt` | `string` | The raw prompt text |
| `features` | `object` | Feature dict from `extract_features()` |
| `router.provider` | `string` | `"local"` or `"remote"` — router's decision |
| `router.routing_score` | `int` | Raw weighted score |
| `router.confidence` | `float` | [0.0, 1.0] distance from threshold |
| `router.reason` | `array[string]` | Human-readable score explanation |
| `local` | `object` | Local (Ollama) provider result |
| `remote` | `object` | Remote LLM provider result |
| `evaluation` | `object` | Evaluator metrics block |
| `router_correct` | `null` | Reserved — filled by ML pipeline later |
| `router_version` | `string` | `"v1.0"` — router config identifier |
| `timestamp` | `string` | ISO-8601 UTC timestamp |

#### Provider block shape

```json
{
  "model": "llama3",
  "response": "...",
  "latency_ms": 412,
  "estimated_input_tokens": 7,
  "estimated_output_tokens": 6
}
```

On error, `response` is omitted and `error` is added:

```json
{
  "model": "llama3",
  "latency_ms": 150,
  "estimated_input_tokens": 7,
  "estimated_output_tokens": 0,
  "error": "Could not connect to Ollama."
}
```

---

## Summary Schema

Stored in `summaries/summary_TIMESTAMP.json` as a single JSON object:

```json
{
  "total_prompts": 160,
  "categories_processed": ["coding", "creative_writing", "general", ...],
  "difficulty_distribution": { "easy": 56, "medium": 56, "hard": 48 },
  "router_version": "v1.0",
  "local_selected": 93,
  "remote_selected": 67,
  "average_routing_score": 18.3,
  "average_confidence": 0.84,
  "average_local_latency_ms": 812.0,
  "average_remote_latency_ms": 1517.0,
  "average_local_tokens": 42.5,
  "average_remote_tokens": 38.2,
  "benchmark_duration_seconds": 214.0,
  "timestamp": "2026-07-03T00:00:00+00:00",
  "errors": []
}
```

### Summary Field Reference

| Field | Type | Description |
|---|---|---|
| `total_prompts` | `int` | Number of prompts that produced records |
| `categories_processed` | `array[string]` | Sorted list of unique categories run |
| `difficulty_distribution` | `object` | Count of easy / medium / hard prompts |
| `router_version` | `string` | Router version tag (`"v1.0"`) |
| `local_selected` | `int` | Count of prompts routed to local |
| `remote_selected` | `int` | Count of prompts routed to remote |
| `average_routing_score` | `float` | Mean routing score across all prompts |
| `average_confidence` | `float` | Mean router confidence |
| `average_local_latency_ms` | `float` | Mean local provider latency in milliseconds |
| `average_remote_latency_ms` | `float` | Mean remote provider latency in milliseconds |
| `average_local_tokens` | `float` | Mean total tokens (input + output) from local |
| `average_remote_tokens` | `float` | Mean total tokens (input + output) from remote |
| `benchmark_duration_seconds` | `float` | Wall-clock runtime of the full benchmark run |
| `timestamp` | `string` | ISO-8601 UTC start time of the benchmark |
| `errors` | `array[string]` | Error messages from failed individual prompts |

---

## Error Handling

- If a **single prompt** fails (e.g. provider timeout, network error), the error is logged to `summary.errors` and the benchmark continues.
- If a **provider block** fails for an individual prompt, the record is still saved with an `error` field in the failed provider block and `estimated_output_tokens: 0`.
- If the **prompts directory** is missing, `FileNotFoundError` is raised immediately.
- If a **prompt JSON file** is malformed or has an unexpected root type, it is skipped with a warning message.

---

## `router_correct` Field

The `router_correct` field in every `BenchmarkRecord` is intentionally set to `null` during the benchmark run.

It is reserved for a future **ML decision pipeline** that will:

1. Compare the `estimated_quality.score` of the `local` and `remote` responses.
2. Determine which provider produced the better response for each prompt.
3. Check whether `router.provider` matches the better provider.
4. Write `true` or `false` to `router_correct`.

This labelling step is deliberately separated from the benchmark runner so that the routing accuracy metric can evolve independently (e.g. by using human preference labels instead of quality heuristics).

---

## Future ML Integration

The `BenchmarkRecord` schema is designed as a training-ready format.

### Proposed ML workflow

1. **Collect benchmark runs.** Run `run_benchmark()` against the full 160-prompt dataset.
2. **Label `router_correct`.** Use a decision engine or human preference labels to fill `router_correct` for each record.
3. **Extract training features.** The `features` block in each record maps directly to the `RouterConfig` weight inputs.
4. **Train a weight vector.** Fit a logistic regression on `features` → `router_correct` labels. The output is a `RouterConfig`-compatible weight set.
5. **Inject weights.** Serialise the trained `RouterConfig` to JSON and load it at router startup — no code changes needed.

### What does NOT change

- The `route(features, config)` interface.
- The `RouterConfig` field names.
- The `BenchmarkRecord` schema.
- The evaluator logic.
- Any existing tests.

---

## Module Structure

```
benchmark.py
│
├── PromptEntry                   dataclass: one prompt from a JSON file
├── BenchmarkRecord               dataclass: full result for one prompt
├── BenchmarkSummary              dataclass: aggregated run statistics
│
├── load_prompt_files()           load all structured prompt JSON files
├── _load_single_prompt_file()    parse one file into PromptEntry list
├── _parse_prompt_entry()         validate one dict into PromptEntry
│
├── benchmark_one()               run full pipeline for one prompt (async)
│
├── save_results()                write records to timestamped JSON file
├── save_summary()                write summary to timestamped JSON file
├── _compute_summary()            aggregate records into BenchmarkSummary
│
├── _print_summary()              formatted console output
│
└── run_benchmark()               ← primary public entry point (async)
```
