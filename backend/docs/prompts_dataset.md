# Prompt Dataset

This document describes the structure, schema, and conventions of the prompt dataset used for benchmarking, evaluation, and future ML training of the Hybrid Token Router.

---

## Folder Structure

```
backend/
└── app/
    └── data/
        ├── prompts/          ← Curated prompt JSON files (one per category)
        ├── raw_runs/         ← Raw local + remote model outputs from dataset_generator.py
        ├── evaluations/      ← Deterministic evaluation metrics from evaluator.py
        ├── benchmarks/       ← Benchmark comparison reports
        ├── processed/        ← Feature-extracted, normalised ML-ready datasets
        └── training/         ← Training datasets, labels, and model artefacts
```

### Directory Roles

| Directory | Contents | Produced By |
|---|---|---|
| `prompts/` | Curated prompt JSON files | Humans / this dataset |
| `raw_runs/` | Raw provider outputs per prompt | `dataset_generator.py` |
| `evaluations/` | Deterministic quality metrics | `evaluator.py` |
| `benchmarks/` | Aggregated benchmark comparison results | Benchmarking scripts |
| `processed/` | Feature vectors and normalised inputs | Preprocessing pipeline |
| `training/` | Label files, split datasets, trained model weights | ML training pipeline |

---

## JSON Schema

Every prompt file contains a JSON **array** of objects. Each object must conform to the following schema:

```json
{
  "id": "coding_001",
  "category": "coding",
  "difficulty": "easy",
  "expected_reasoning": "low",
  "prompt": "Write a Python Hello World program."
}
```

### Field Definitions

#### `id` · `string` · **required**

A unique identifier for the prompt across the entire dataset.

**Format:** `{category}_{three-digit-number}` — e.g., `coding_001`, `mathematics_015`.

IDs must be unique globally. No two prompts across all files may share an ID.

#### `category` · `string` · **required**

The task category the prompt belongs to. Must match the filename it is stored in.

Valid values: `coding`, `mathematics`, `reasoning`, `planning`, `translation`, `summarization`, `creative_writing`, `general`.

#### `difficulty` · `string` · **required**

The estimated difficulty of the prompt for a language model.

Valid values: `easy`, `medium`, `hard`.

#### `expected_reasoning` · `string` · **required**

The expected level of multi-step reasoning required to answer the prompt well.

Valid values: `low`, `medium`, `high`.

#### `prompt` · `string` · **required**

The natural language prompt sent to the language model. Must be self-contained: no references to prior messages, external files, or real-time internet access.

---

## Category Definitions

| Category | Description | Example Topics |
|---|---|---|
| `coding` | Prompts requiring code, debugging, or system design | Python, SQL, algorithms, APIs, data structures |
| `mathematics` | Prompts requiring numeric computation or formal proof | Algebra, calculus, probability, statistics, geometry |
| `reasoning` | Prompts requiring logic, inference, or decision making | Puzzles, ethical dilemmas, multi-step deduction, analogies |
| `planning` | Prompts asking for structured plans, schedules, or roadmaps | Study plans, travel itineraries, project plans, career plans |
| `translation` | Prompts requiring language translation | Sentences, paragraphs, technical text, legal text |
| `summarization` | Prompts requiring condensation or distillation of content | Emails, meeting notes, articles, research papers |
| `creative_writing` | Prompts requiring original creative text generation | Stories, poems, scripts, dialogues, satire |
| `general` | Broad knowledge prompts not fitting other categories | Science, history, technology, health, everyday facts |

---

## Difficulty Definitions

| Value | Description |
|---|---|
| `easy` | Short, single-step task. Requires minimal reasoning. A capable small model can answer correctly. |
| `medium` | Multi-step task with moderate depth. Requires several interconnected reasoning steps or domain knowledge. |
| `hard` | Complex, multi-constraint task. Requires sustained reasoning, deep domain knowledge, or synthesis across sources. |

**Distribution per category file:** 7 easy, 7 medium, 6 hard (total 20 per file).

---

## Expected Reasoning Definitions

`expected_reasoning` is a label consumed by the Routing Engine and future ML classifiers to indicate how much reasoning load a prompt places on a model.

| Value | Description | Typical Routing Implication |
|---|---|---|
| `low` | Single-step factual or mechanical response. Model retrieves or computes directly. | Likely routes to local model. |
| `medium` | Several reasoning steps or moderate synthesis. Model must connect multiple pieces of information. | May route to either model depending on other features. |
| `high` | Deep, multi-constraint reasoning. Model must plan, infer, synthesise, and self-check. | Likely routes to remote model. |

`expected_reasoning` is independent of `difficulty`. A short but logically tricky puzzle may be `easy` difficulty with `high` expected_reasoning, while a long but mechanical translation may be `medium` difficulty with `low` expected_reasoning.

---

## Prompt Quality Rules

All prompts in this dataset must follow these rules:

1. **Self-contained.** The prompt must make complete sense without any prior conversation, external files, or context not present in the prompt itself.
2. **Unique.** No two prompts may be semantically identical or near-identical across the entire dataset.
3. **Realistic.** Prompts must reflect genuine tasks that users might submit to a language model in practice.
4. **Safe.** Prompts must not request illegal, harmful, or unethical content.
5. **Offline.** Prompts must not require real-time internet access, live data, or current news.
6. **No follow-ups.** Prompts must not begin with "as I mentioned" or reference a previous exchange.

---

## Current Dataset Statistics

| File | Category | Easy | Medium | Hard | Total |
|---|---|---|---|---|---|
| `coding.json` | `coding` | 7 | 7 | 6 | 20 |
| `mathematics.json` | `mathematics` | 7 | 7 | 6 | 20 |
| `reasoning.json` | `reasoning` | 7 | 7 | 6 | 20 |
| `planning.json` | `planning` | 7 | 7 | 6 | 20 |
| `translation.json` | `translation` | 7 | 7 | 6 | 20 |
| `summarization.json` | `summarization` | 7 | 7 | 6 | 20 |
| `creative_writing.json` | `creative_writing` | 7 | 7 | 6 | 20 |
| `general.json` | `general` | 7 | 7 | 6 | 20 |
| **Total** | | **56** | **56** | **48** | **160** |

---

## How to Add New Prompts

Follow these steps when adding new prompts to an existing category file:

### Step 1 — Choose the correct file

Open the file matching the prompt's category, e.g., `backend/app/data/prompts/coding.json`.

### Step 2 — Determine the next ID

Find the highest existing ID number in the file and increment it. For example, if the last entry is `coding_020`, the next ID is `coding_021`.

### Step 3 — Create the JSON object

```json
{
  "id": "coding_021",
  "category": "coding",
  "difficulty": "medium",
  "expected_reasoning": "medium",
  "prompt": "Your new prompt text here."
}
```

### Step 4 — Validate before committing

Run the following validation checks manually or via a script:

- [ ] The JSON file parses without errors (`python -m json.tool coding.json`).
- [ ] The `id` is unique across all 8 prompt files.
- [ ] The `category` field matches the filename.
- [ ] The `difficulty` is one of `easy`, `medium`, `hard`.
- [ ] The `expected_reasoning` is one of `low`, `medium`, `high`.
- [ ] The `prompt` is self-contained and does not violate any quality rules.

### Step 5 — Add to a new category (if needed)

To add a new category:

1. Create a new file named `{category}.json` in `backend/app/data/prompts/`.
2. Follow the ID format `{category}_{number}`.
3. Ensure the `category` field in every object matches the filename.
4. Add the category to the `category` field validation list in this document and in any validation scripts.

---

## Validation Script (Recommended)

A future validation script should check:

```python
# Pseudocode for prompt dataset validation
all_ids = set()
for each json_file in prompts/:
    data = load_json(json_file)
    assert isinstance(data, list)
    category = filename_without_extension
    for prompt in data:
        assert prompt["id"] not in all_ids          # globally unique
        all_ids.add(prompt["id"])
        assert prompt["category"] == category        # matches filename
        assert prompt["difficulty"] in {"easy", "medium", "hard"}
        assert prompt["expected_reasoning"] in {"low", "medium", "high"}
        assert len(prompt["prompt"].strip()) > 0
```

---

## Future Improvements

- Add a `tags` field to each prompt for finer-grained topic filtering (e.g., `["python", "recursion", "algorithms"]`).
- Add a `source` field to track whether a prompt was human-authored or machine-generated.
- Add a `language` field to the translation category for explicit source and target language metadata.
- Add a `word_count` field pre-computed at creation time to avoid re-counting at load time.
- Introduce a validation CI step that runs the validation script on every pull request.
- Add evaluation result references linking each prompt ID to its entry in `evaluations/`.
