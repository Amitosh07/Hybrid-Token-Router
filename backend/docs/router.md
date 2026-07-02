# Router

The Routing Engine is a deterministic, ML-free decision module that receives the feature dictionary produced by the Feature Extractor and decides whether to forward a prompt to the **local** or **remote** language model.

It makes no network calls, invokes no LLM, and applies no machine learning at runtime.

---

## Architecture

```
extract_features(prompt)
        │
        ▼  feature dict
┌─────────────────────────────────┐
│          router.py              │
│                                 │
│  _score_reasoning()             │  dominant signal
│  _score_tokens()                │  token pressure
│  _score_complexity()            │  hard / medium / easy
│  _score_task_type()             │  small nudge
│  _score_boolean_flags()         │  code, math, JSON
│                                 │
│  routing_score = Σ sub-scores   │
│  provider = threshold decision  │
│  confidence = sigmoid(distance) │
│  reason = narrated contributions│
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │             │
      local         remote
```

The full pipeline for a single prompt is:

```python
from app.services.feature_extractor import extract_features
from app.services.router import route

features = extract_features(prompt)
decision = route(features)
```

Or in one call:

```python
from app.services.router import route_prompt
decision = route_prompt(prompt)
```

---

## Routing Score

The `routing_score` is a non-negative integer computed by summing five independent sub-scores. **Higher score → stronger case for remote.**

### Sub-score Breakdown

| Component | Formula | Max contribution | Notes |
|---|---|---|---|
| Reasoning score | `reasoning_score × reasoning_score_weight` | ~50 | **Dominant signal** |
| Token pressure | `(tokens // 256) × token_weight_per_256` | ~15 (1 000-token prompt) | Proxy for prompt size |
| Complexity | Lookup: easy→0, medium→5, hard→10 | 10 | Derived from reasoning_score |
| Task-type nudge | Lookup per task type | 5 | **Intentionally small** |
| Boolean flags | code(+5), math(+4), JSON(+3) | 12 | Additive |

### Why reasoning_score Dominates

Task type alone is an unreliable router. A trivial `coding` prompt such as *"Write hello world in Python"* should stay local, while a deeply constrained `planning` prompt about globally distributed architectures should go remote.

By weighting `reasoning_score` most heavily (default weight `5` per point, max contribution `50`), the router naturally handles this: complexity of thought drives routing, not superficial category labels.

---

## Thresholds

### Decision Threshold

```
routing_score < threshold  →  local
routing_score ≥ threshold  →  remote
```

Default threshold: **25**.

The threshold is stored in `RouterConfig.threshold` and can be changed without touching any scoring logic.

### Configuring the Threshold

```python
from app.services.router import RouterConfig, route

cfg = RouterConfig(threshold=40)   # more aggressive remote routing
result = route(features, config=cfg)
```

---

## Confidence Calculation

Confidence measures how far the `routing_score` is from the threshold, normalised by `RouterConfig.max_confidence_distance`.

### Formula

```
distance = |routing_score − threshold|
x        = distance / max_confidence_distance   (clamped to [0, 1])
confidence = sigmoid(k × (x − 0.5))  where k = 5
```

This is a logistic (sigmoid) function:

- Score **exactly at** the threshold → confidence ≈ 0.07 (very uncertain)
- Score **half a max_distance away** → confidence ≈ 0.5
- Score **at or beyond** max_distance → confidence ≈ 0.99

Default `max_confidence_distance`: **20**.

### Interpretation

| routing_score (default threshold=25) | confidence |
|---|---|
| 25 (exactly at threshold) | ~0.07 |
| 5 or 45 (±20 away) | ~0.99 |
| 15 (−10 away, local side) | ~0.27 |
| 35 (+10 away, remote side) | ~0.73 |

---

## Scoring Weights

All weights live in `RouterConfig` — a plain Python dataclass. This is the **single source of truth** for all tunable numbers.

```python
@dataclass
class RouterConfig:
    threshold:               int = 25
    max_confidence_distance: int = 20
    reasoning_score_weight:  int = 5      # per point (0-10 scale)
    token_weight_per_256:    int = 3      # per 256-token block
    code_weight:             int = 5      # flat bonus
    math_weight:             int = 4      # flat bonus
    json_weight:             int = 3      # flat bonus
    complexity_weights:     dict = {easy: 0, medium: 5, hard: 10}
    task_type_weights:      dict = {coding: 4, mathematics: 4, …}
```

### Default Task-Type Weights

| Task type | Nudge | Rationale |
|---|---|---|
| `reasoning` | 5 | Explicitly demands multi-step thinking |
| `coding` | 4 | Often needs accurate syntax and context |
| `mathematics` | 4 | Often needs rigorous step-by-step work |
| `planning` | 4 | Often multi-constraint |
| `summarization` | 3 | Moderate complexity |
| `creative_writing` | 3 | Moderate complexity |
| `translation` | 2 | Usually single-step |
| `question_answering` | 1 | Often factual lookups |
| `general` | 0 | No signal |

---

## Output Shape

```json
{
  "provider": "local",
  "routing_score": 14,
  "confidence": 0.9231,
  "reason": [
    "Moderate reasoning score (3/10)",
    "Contains code",
    "Task type is coding",
    "Routing score 14 is below threshold 50 → local"
  ]
}
```

### Fields

| Field | Type | Description |
|---|---|---|
| `provider` | `str` | `"local"` or `"remote"` |
| `routing_score` | `int` | Raw score (higher = stronger case for remote) |
| `confidence` | `float` | [0.0, 1.0] — distance from threshold, normalised |
| `reason` | `list[str]` | Human-readable explanation of every contribution |

---

## Future ML Replacement

The `RouterConfig` dataclass is the designed injection point for ML-trained weights.

### Proposed Migration Path

1. **Collect data.** Use the Dataset Generator to gather local/remote comparison runs. Use the Evaluator to label which provider produced the better response.

2. **Train a weight vector.** Fit a logistic regression (or gradient-boosted tree) on the labelled features. The output is a set of weights that mirrors the `RouterConfig` fields.

3. **Serialise `RouterConfig`.** Add `RouterConfig.from_json()` / `to_json()` methods.

4. **Inject at startup.** Load the trained config from disk and pass it to `route()` — no other code changes needed.

```python
# Future usage — existing callers unchanged
cfg = RouterConfig.from_json("trained_weights.json")
result = route(features, config=cfg)
```

5. **A/B shadow mode.** Run both the rule-based default config and the ML config in parallel, comparing decisions, before promoting the ML config to production.

### What Does Not Change

- The `route(features, config)` interface.
- The `RouterConfig` field names (ML training targets the same fields).
- The Feature Extractor output schema.
- All existing tests.

---

## Module Structure

```
backend/app/services/router.py
│
├── PROVIDER_LOCAL / PROVIDER_REMOTE      string constants
│
├── RouterConfig                          all weights + threshold (dataclass)
│
├── _score_reasoning()                    dominant signal
├── _score_tokens()                       token/length pressure
├── _score_complexity()                   complexity label
├── _score_task_type()                    small task-type nudge
├── _score_boolean_flags()                code, math, JSON flags
│
├── _compute_confidence()                 sigmoid distance → [0,1]
│
├── route(features, config) → dict        ← primary public API
└── route_prompt(prompt, config) → dict   ← convenience wrapper
```

---

## Future Improvements

- **Per-task complexity multiplier.** Multiply task-type weight by a complexity factor so a `hard` coding task outscores a `medium` one more aggressively.
- **Cost awareness.** Add a `remote_cost_bias` weight to prefer local when both models are roughly equivalent.
- **Latency feedback.** Feed observed provider latency back into the config after each request.
- **Ensemble voting.** Run multiple lightweight RouterConfigs and take a majority vote for border-zone prompts (low confidence).
- **Version metadata.** Embed `config_version` in the output for traceability across deployments.
