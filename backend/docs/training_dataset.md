# Training Dataset Documentation — Phase 2

This document details the machine learning data generation pipeline, the Decision Engine logic, the schema and features of the compiled dataset, and the validation rules.

---

## 1. Pipeline Architecture

The dataset generation pipeline flows sequentially as follows:

```
            Expanded Prompt Dataset (640 curated items)
                              │
                              ▼
            Benchmark Runner (sequential execution)
               ├── Feature Extractor V3 (complexity features)
               ├── Router V1.0 (heuristic decisions)
               ├── Local Model (Ollama: qwen2.5:3b)
               └── Remote Model (Groq: llama-3.1-8b-instant)
                              │
                              ▼
           Benchmark Records (JSON files in /benchmarks/)
                              │
                              ▼
             Decision Engine (app/ml/decision_engine.py)
                              │ (optimal label generation)
                              ▼
             Dataset Builder (app/ml/dataset_builder.py)
                              │ (CSV compilation)
                              ▼
            Validation Module (app/ml/validation.py)
                              │ (sanity checks)
                              ▼
                      training_dataset.csv
```

---

## 2. Decision Engine & Label Generation Logic

The Decision Engine replaces the simple heuristic router's choice with a robust, multi-criteria optimal decision. It answers:
> *"If this prompt arrived in production today, which model should have answered it?"*

### Configuration (`backend/app/ml/config.py`)
All parameters are externalized in `DecisionConfig` to allow tuning without altering the codebase:
- `quality_weight` (default: `0.60`)
- `latency_weight` (default: `0.20`)
- `cost_weight` (default: `0.20`)
- `local_preference_bias` (default: `0.05`)
- `quality_threshold_high` (default: `0.85`)
- `quality_delta_threshold` (default: `0.08`)
- `max_remote_latency_ms` (default: `8000.0`)
- `min_remote_quality` (default: `0.35`)

### Decision Steps:
1. **Safety Override (Error Checks):**
   - If the Local run encountered an error or generated empty text, but Remote succeeded, select **REMOTE**.
   - If the Remote run encountered an error or rate limit, but Local succeeded, select **LOCAL**.
   - If both failed, select **LOCAL** (fallback).
2. **Quality Threshold Heuristics:**
   - **Exceptional Local Quality:** If Local quality score $\ge 0.85$, select **LOCAL** (redundant to pay for Remote).
   - **Marginal Remote Improvement:** If the quality gap ($Q_{\text{remote}} - Q_{\text{local}}) \le 0.08$, select **LOCAL** (cost and latency savings exceed quality gain).
   - **Excessive Latency:** If Remote latency $\ge 8000$ ms, select **LOCAL**.
   - **Insufficient Remote Quality:** If Remote quality $< 0.35$, select **LOCAL**.
3. **Weighted Utility Formula:**
   If no threshold overrides apply, compute utility scores:
   $$U_{\text{local}} = W_{\text{quality}} \times Q_{\text{local}} + W_{\text{latency}} \times S_{\text{local\_speed}} - W_{\text{cost}} \times C_{\text{local\_cost\_norm}} + \text{local\_bias}$$
   $$U_{\text{remote}} = W_{\text{quality}} \times Q_{\text{remote}} + W_{\text{latency}} \times S_{\text{remote\_speed}} - W_{\text{cost}} \times C_{\text{remote\_cost\_norm}}$$
   Where:
   - Speed score $S = \frac{1}{1 + \frac{\text{latency\_ms}}{1000}}$
   - Normalized cost $C_{\text{norm}} = \min(1.0, \frac{\text{cost}}{0.01})$ (Remote costs are normalized against a max cost baseline of $1\phi$ per transaction).
   - Select **REMOTE** if $U_{\text{remote}} > U_{\text{local}}$, else **LOCAL**.

---

## 3. Dataset Schema & Feature Descriptions

The generated `training_dataset.csv` contains one row per prompt.

### Metadata Columns
- `prompt_id`: Unique identifier (e.g. `coding_025`).
- `prompt`: Raw user input text.
- `category`: Task category (`coding`, `mathematics`, `reasoning`, `planning`, `translation`, `summarization`, `creative_writing`, `general`).
- `difficulty`: Ground truth difficulty label (`easy`, `medium`, `hard`).
- `expected_reasoning`: Ground truth reasoning expectation (`low`, `medium`, `high`).

### Lexical Features
- `prompt_length`: Character length of the prompt (integer).
- `word_count`: Number of whitespace-separated words (integer).
- `estimated_input_tokens`: Estimated input token count (length / 4, integer).

### Structural Features (Binary, mapped to 0/1)
- `contains_code`: Whether code syntax, programming keywords, or fences were found.
- `contains_math`: Whether LaTeX, equations, or math symbols were found.
- `contains_json`: Whether curly braces or json brackets were found.
- `contains_markdown`: Whether headers, lists, or markdown symbols were found.
- `contains_numbers`: Whether digit characters were found.
- `contains_question`: Whether a question mark or interrogative words were found.

### Complexity Scores (Feature Extractor V3 output)
- `reasoning_score`: Integer complexity scale [0, 10].
- `technical_complexity`: Technical domain complexity score [0.0, 1.0].
- `reasoning_depth`: Causal, logical, or proof complexity score [0.0, 1.0].
- `task_complexity`: Task verb and cognitive modifier score [0.0, 1.0].
- `constraint_complexity`: Rules and formatting restrictions score [0.0, 1.0].
- `context_complexity`: Vocabulary richness and information density score [0.0, 1.0].
- `complexity_score`: Aggregated, interaction-boosted complexity score [0.0, 1.0].

### Operational Features (Operational runtime statistics)
- `local_latency_ms`: Local model response time in milliseconds.
- `remote_latency_ms`: Remote model response time in milliseconds.
- `local_cost`: Estimated local API cost (always $0.0$).
- `remote_cost`: Estimated remote API cost based on input/output tokens (in USD).
- `local_quality_score`: Heuristic quality score of local response [0.0, 1.0].
- `remote_quality_score`: Heuristic quality score of remote response [0.0, 1.0].

### Machine Learning Target Label
- `label`: **`LOCAL`** or **`REMOTE`** (the target label for supervised classification).

---

## 4. Validation Rules

The validation module (`app/ml/validation.py`) enforces quality checks:
1. **Schema Check:** Verifies that all 28 required columns exist in the CSV headers.
2. **Missing Values:** Scans for null, NaN, or blank cells in any row.
3. **Duplicates:** Checks for duplicated prompt IDs or duplicate prompt strings.
4. **Invalid Labels:** Ensures target labels are strictly `"LOCAL"` or `"REMOTE"`.
5. **Numeric Constraints:**
   - Binary fields must be strictly `0` or `1`.
   - Feature scores and quality scores must lie in $[0.0, 1.0]$.
   - `reasoning_score` must lie in $[0, 10]$.
   - Latencies must be non-negative.
6. **Class Balance:** Reports the ratio of LOCAL vs REMOTE routing labels to warn of potential degenerate distributions.

---

## 5. Future ML Roadmap (Phase 3)

The generated `training_dataset.csv` is fully preprocessed, numericalized, and optimized for supervised training:
1. **Model Selection:** Train a light gradient boosting classifier (e.g. `XGBoost` or `LightGBM`) or a shallow neural network on the tabular features.
2. **Feature Pruning:** Exclude non-generalizable metadata (`prompt_id`, `prompt`, `category`, `difficulty`, `expected_reasoning`) and operational runtimes (`local_latency_ms`, `remote_latency_ms`, `local_cost`, `remote_cost`, `local_quality_score`, `remote_quality_score`) to avoid target leakage.
3. **Inference Integration:** Save the trained classifier as a serialization file (e.g. joblib/onnx) and load it in the FastAPI backend to replace the current heuristic router.
