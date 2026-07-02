# Evaluator

The evaluator converts raw local and remote model outputs into deterministic metrics for future training-label generation. It does not route prompts, train a classifier, call external APIs, or return a final winner.

## Architecture

`backend/app/services/evaluator.py` is intentionally measurement-only.

It accepts a raw run shaped like:

```json
{
  "prompt": "...",
  "local": {
    "response": "...",
    "latency_ms": 800
  },
  "remote": {
    "response": "...",
    "latency_ms": 1400
  }
}
```

It returns:

```json
{
  "prompt": "...",
  "metrics": {
    "response_length": {},
    "latency": {},
    "structural_validity": {},
    "estimated_quality": {
      "local": {},
      "remote": {}
    }
  },
  "evaluation_method": "rule"
}
```

No `winner` field is returned. A future `decision_engine` should combine these metrics into labels such as `local`, `remote`, or `tie`.

## Metric Helpers

- `compare_response_lengths(local, remote)` measures characters, words, and deltas.
- `compare_latency(local, remote)` exposes raw latencies and normalized speed scores.
- `structural_validity(local, remote)` checks for non-empty responses and provider errors.
- `estimate_quality(prompt, provider)` uses deterministic heuristics for a quality score.
- `evaluate_run(run)` combines all evaluator metrics for one raw run.
- `evaluate_dataset(dataset)` evaluates a list of raw runs.
- `choose_winner(metrics)` is a reserved handoff point and raises `NotImplementedError`.

## Quality Heuristic

The current quality estimate is rule-based and deterministic. It considers:

- response length
- estimated sentence count
- prompt term coverage
- provider errors
- empty responses

The score is bounded from `0.0` to `1.0`.

## Future Extension Points

A future LLM judge can be plugged in behind `estimate_quality()` or added as a new metric group without changing the evaluator output contract.

A future `decision_engine` can consume evaluator metrics and decide the final training label. That module should own weighting, tie thresholds, confidence, and final label policy.

Potential future metrics:

- factuality checks
- instruction-following checks
- toxicity or safety checks
- task-specific rubric scores
- cost estimates
- provider retry metadata
