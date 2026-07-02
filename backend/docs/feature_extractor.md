# Feature Extractor

The Feature Extractor is a stateless, deterministic module that converts a raw user prompt into a structured feature dictionary. It has one responsibility: **measure properties of the prompt**. It does not call any language model, external service, or database. Routing decisions belong entirely in the Routing Engine, which consumes the features this module produces.

---

## Architecture

```
User Prompt (str)
        │
        ▼
┌─────────────────────────────┐
│      feature_extractor.py   │
│                             │
│  Primitive detectors        │  Boolean flags (code, math, JSON, …)
│  Token estimator            │  Lightweight char / 4 approximation
│  Task classifier            │  Ordered keyword pattern matching
│  Reasoning scorer           │  Additive sub-score system
│  Complexity mapper          │  Threshold band mapping
│                             │
└─────────────┬───────────────┘
              │
              ▼
       Feature Dict (dict)
              │
              ▼
      Routing Engine  ← (future module, not implemented here)
```

All helper functions are private (prefixed with `_`). The sole public entry point is `extract_features(prompt: str) -> dict`.

---

## Extracted Features

### `prompt_length` · `int`

Number of characters (`len(prompt)`) in the raw input string. Useful as a cheap proxy for verbosity before token estimation.

### `word_count` · `int`

Number of whitespace-separated tokens (`len(prompt.split())`). Differs from `estimated_input_tokens` because it counts words, not subword units.

### `estimated_input_tokens` · `int`

Lightweight approximation: `ceil(len(prompt) / 4)`. The divisor of 4 reflects the average English byte-pair-encoding token length observed in major tokenizers.

The implementation is isolated inside `_estimate_tokens()` so a real tokenizer (e.g., `tiktoken` or `transformers.AutoTokenizer`) can be swapped in without changing the public interface.

### `contains_code` · `bool`

`True` when any of the following are detected:

- Fenced code blocks (` ```…``` `)
- Inline backtick code (`` `…` ``)
- Programming keywords: `def`, `class`, `import`, `function`, `return`, `var`, `let`, `const`, `#include`, `public static`, `void`, `int`, `float`, `bool`, `lambda`, `async def`, `await`

### `contains_math` · `bool`

`True` when any of the following are detected:

- Display LaTeX (`$$…$$`)
- Inline LaTeX (`$…$`)
- Math vocabulary: `integral`, `derivative`, `matrix`, `equation`, `solve`, `factorial`, `logarithm`, `calculus`, `algebra`, `geometry`, `trigonometry`, `eigenvalue`, `polynomial`
- Math symbols and operators: `= < > ≤ ≥ ≠ ± ∑ ∏ √ ∫ ∂ ∇`
- Bare arithmetic expressions: `12 * 8`, `3 + 5`, etc.

### `contains_json` · `bool`

`True` when the prompt contains a JSON-like object (`{…}` with ≥ 2 inner characters) or array (`[…]` with ≥ 2 inner characters).

### `contains_markdown` · `bool`

`True` when any of the following Markdown elements are found:

- ATX headings (`# … ######`)
- Horizontal rules (`---`, `***`)
- Bold (`**…**`, `__…__`)
- Italic (`*…*`, `_…_`)
- Unordered list items (`- `, `* `, `+ `)
- Ordered list items (`1. `, `2. `, …)
- Markdown links (`[text](url)`)

### `contains_numbers` · `bool`

`True` when at least one standalone integer or decimal number appears (e.g., `42`, `3.14`). Numbers that are part of a larger word are excluded.

### `contains_question` · `bool`

`True` when the prompt ends with a `?`, contains a `?` anywhere, or begins with / contains interrogative words: `what`, `why`, `how`, `when`, `where`, `which`, `who`, `whom`, `whose`, `can you`, `could you`, `would you`, `is it`, `are there`, `do you`, `does it`, `explain`, `tell me`.

### `task_type` · `str`

The dominant task category detected by ordered keyword pattern matching. Patterns are tested from highest to lowest specificity; the first match wins.

| Value | Example triggers |
|---|---|
| `coding` | `code`, `function`, `debug`, `algorithm`, `api`, `unit test`, `import` |
| `mathematics` | `calculate`, `solve`, `integral`, `factorial`, `probability`, `proof` |
| `reasoning` | `why`, `analyze`, `cause`, `effect`, `infer`, `compare`, `justify` |
| `summarization` | `summarize`, `tldr`, `brief`, `overview`, `highlights` |
| `translation` | `translate`, `in French`, `in German`, `traduction` |
| `creative_writing` | `write a story`, `write a poem`, `fiction`, `narrative` |
| `planning` | `plan`, `roadmap`, `schedule`, `steps to`, `checklist`, `milestones` |
| `question_answering` | `what is`, `who is`, `define`, `how does`, `tell me about` |
| `general` | Fallback when no other pattern matches |

### `reasoning_score` · `int` (0–10)

An explainable integer measuring how much multi-step reasoning the prompt demands. See the [Reasoning Score Calculation](#reasoning-score-calculation) section below.

### `complexity` · `str`

Human-readable complexity label derived from `reasoning_score`. See the [Complexity Mapping](#complexity-mapping) section below.

---

## Reasoning Score Calculation

The score is computed by summing five independent sub-scores, capped at 10.

```
reasoning_score = min(
    length_score          # 0–3
  + keyword_score         # 0–3
  + constraint_score      # 0–2
  + coding_score          # 0–1
  + math_score            # 0–1
, 10)
```

### Sub-score Details

#### `length_score` (0–3)

Measures prompt verbosity. Longer prompts typically encode more requirements.

| Prompt length (chars) | Sub-score |
|---|---|
| 0–200 | 0 |
| 201–500 | 1 |
| 501–900 | 2 |
| 901+ | 3 |

#### `keyword_score` (0–3)

Counts reasoning-indicator keywords: `because`, `therefore`, `hence`, `thus`, `since`, `given that`, `it follows`, `consequently`, `analyze`, `evaluate`, `critique`, `compare`, `contrast`, `justify`, `deduce`, `infer`, `conclude`, `explain why`, `explain how`, `step by step`, `chain of thought`, `reason`, `argument`, `logic`, `proof`, `implication`, `cause`, `effect`.

| Match count | Sub-score |
|---|---|
| 0 | 0 |
| 1–2 | 1 |
| 3–5 | 2 |
| 6+ | 3 |

#### `constraint_score` (0–2)

Counts explicit constraint phrases: `must`, `should not`, `do not`, `must not`, `only if`, `at least`, `at most`, `exactly`, `no more than`, `no less than`, `without`, `except`, `unless`, `provided that`, `assuming`, `given that`, `such that`, `subject to`, `constraint`.

| Match count | Sub-score |
|---|---|
| 0 | 0 |
| 1–2 | 1 |
| 3+ | 2 |

#### `coding_score` (0–1)

Adds 1 point when `contains_code` is `True`, reflecting that code tasks typically demand structured reasoning about logic and syntax.

#### `math_score` (0–1)

Adds 1 point when `contains_math` is `True`, reflecting that mathematical prompts require formal reasoning steps.

---

## Complexity Mapping

```
reasoning_score 0–3   →  easy
reasoning_score 4–7   →  medium
reasoning_score 8–10  →  hard
```

These thresholds are intentionally coarse. A downstream routing engine can use the raw `reasoning_score` for finer-grained decisions.

---

## Module Structure

```
backend/app/services/feature_extractor.py
│
├── Public constants          TASK_*, COMPLEXITY_*
│
├── _estimate_tokens()        Token approximation (isolated for future swap)
│
├── Primitive detectors
│   ├── _contains_code()
│   ├── _contains_math()
│   ├── _contains_json()
│   ├── _contains_markdown()
│   ├── _contains_numbers()
│   └── _contains_question()
│
├── Task classifier
│   ├── _TASK_PATTERNS        Ordered list of (task_type, pattern)
│   └── _detect_task_type()
│
├── Reasoning scorer
│   ├── _length_score()
│   ├── _keyword_score()
│   ├── _constraint_score()
│   ├── _coding_complexity_score()
│   ├── _math_indicator_score()
│   └── _compute_reasoning_score()
│
├── Complexity mapper
│   └── _map_complexity()
│
└── extract_features()        ← sole public entry point
```

---

## Future Improvements

### Replace the token estimator

Swap `_estimate_tokens()` with a real tokenizer (e.g., `tiktoken` for OpenAI models or `transformers.AutoTokenizer`) without touching `extract_features()`:

```python
# Future drop-in replacement for _estimate_tokens
import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")

def _estimate_tokens(text: str) -> int:
    return len(_enc.encode(text))
```

### Extend task-type detection

Add more patterns or integrate a lightweight text-classification model (e.g., a zero-shot `transformers` pipeline) behind `_detect_task_type()`. The Routing Engine interface remains unchanged.

### Improve reasoning score

- Add sentence complexity measurement (average words per sentence).
- Detect multi-step question structures ("First … then … finally …").
- Detect rhetorical questions versus genuine reasoning requests.
- Weight keywords by proximity to the start of the prompt.

### Add language detection

Detect the prompt language (using `langdetect` or `fasttext`) to improve translation task detection and support language-specific routing.

### Caching

Wrap `extract_features()` in a memoization layer for repeated identical prompts in high-traffic deployments.

### Expose version metadata

Return a `feature_extractor_version` key in the dictionary so that changes to heuristics can be traced in datasets and logs.
