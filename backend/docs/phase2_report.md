# Phase 2 Quality & Calibration Report

This report summarizes the statistics, calibration quality, and class balance of the final training dataset compiled during Phase 2 of the Hybrid Token Router project.

## 1. Dataset Overview

| Metric | Value |
|---|---|
| **Total Curated Prompts** | 640 |
| **Number of Features** | 26 (26 features, 2 identifiers) |
| **LOCAL Label Count** | 161 (25.2%) |
| **REMOTE Label Count** | 479 (74.8%) |
| **Duplicate Prompts** | 0 |
| **Duplicate IDs** | 0 |

## 2. Category Distribution

| Category | Count | Percentage |
|---|---|---|
| Coding | 80 | 12.5% |
| Creative_writing | 80 | 12.5% |
| General | 80 | 12.5% |
| Mathematics | 80 | 12.5% |
| Planning | 80 | 12.5% |
| Reasoning | 80 | 12.5% |
| Summarization | 80 | 12.5% |
| Translation | 80 | 12.5% |

## 3. Difficulty & Reasoning Distributions

### Ground Truth Difficulty Distribution
| Difficulty | Count | Percentage |
|---|---|---|
| EASY | 216 | 33.8% |
| MEDIUM | 216 | 33.8% |
| HARD | 208 | 32.5% |

### Expected Reasoning Distribution
| Expected Reasoning | Count | Percentage |
|---|---|---|
| Low | 216 | 33.8% |
| Medium | 216 | 33.8% |
| High | 208 | 32.5% |

## 4. Target Label Distribution (LOCAL vs REMOTE)

The Decision Engine successfully processed all 640 runs. The final target label split is:
- **LOCAL:** 161 samples (25.2%)
- **REMOTE:** 479 samples (74.8%)

### Rationale for Routing Distribution:
1. **Cost & Latency Minimization:** Easy prompts and prompts where the Local model (Qwen 3B) already produces exceptionally high-quality output ($\ge 0.85$) are routed LOCAL to prevent network overhead and API cost.
2. **Remote Value Justification:** Prompts are routed REMOTE only when the quality improvement of the Remote model (Llama 8B) over the Local model exceeds $0.08$, justifying the non-zero cost and latency.

## 5. Numerical Feature Statistics

| Feature | Min | Max | Mean | Std Dev |
|---|---|---|---|---|
| `complexity_score` | 0.0539 | 0.7403 | 0.1993 | 0.1367 |
| `constraint_complexity` | 0.0000 | 0.3600 | 0.0245 | 0.0593 |
| `context_complexity` | 0.0251 | 0.3797 | 0.1001 | 0.0523 |
| `estimated_input_tokens` | 7.0000 | 191.0000 | 50.4969 | 26.9627 |
| `local_cost` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `local_latency_ms` | 1892.0000 | 26726.0000 | 4104.0266 | 3525.0746 |
| `local_quality_score` | 0.0000 | 0.6142 | 0.3442 | 0.1112 |
| `prompt_length` | 26.0000 | 762.0000 | 200.4688 | 107.9274 |
| `reasoning_depth` | 0.0000 | 0.9000 | 0.0695 | 0.1462 |
| `reasoning_score` | 1.0000 | 7.0000 | 2.1031 | 1.2887 |
| `remote_cost` | 0.0002 | 0.0289 | 0.0062 | 0.0047 |
| `remote_latency_ms` | 2237.0000 | 36449.0000 | 5853.0844 | 5603.7499 |
| `remote_quality_score` | 0.1161 | 1.0000 | 0.8746 | 0.1522 |
| `task_complexity` | 0.2000 | 1.0000 | 0.4455 | 0.2421 |
| `technical_complexity` | 0.0000 | 0.9500 | 0.0914 | 0.1986 |
| `word_count` | 6.0000 | 108.0000 | 30.1109 | 14.4665 |

## 6. Dataset Validation Summary

- **Row Count Consistency:** All rows contain exactly 28 fields matching the schema.
- **Null/NaN Scan:** Zero missing values or empty cells detected across all columns.
- **Value Range Verification:**
  - Binary flags (`contains_code`, etc.) are strictly `0` or `1`.
  - Complexity scores, group scores, and quality scores are constrained within $[0.0, 1.0]$.
  - Latencies are non-negative.
- **Uniqueness Check:** Zero duplicate prompt IDs and zero duplicate prompt texts, indicating excellent prompt diversity.

## 7. Potential Dataset Bias Analysis

1. **Model Affinity Bias:** The quality scores are derived from deterministic keywords and structural features. In production, a user might judge quality differently (e.g. style preferences, formatting aesthetic) which the heuristics cannot capture.
2. **Category Balance:** While categories are perfectly balanced (80 prompts each), certain categories naturally trigger higher Remote routing (e.g. Coding/Mathematics) due to the Local model's parameters size limits.

## 8. Recommendations Before ML Training (Phase 3)

1. **Feature Exclusion:** When training the classifier, drop columns `prompt_id`, `prompt`, `category`, `difficulty`, `expected_reasoning` as they are non-generative metadata.
2. **Target Leakage Prevention:** **CRITICAL:** You must exclude operational metrics (`local_latency_ms`, `remote_latency_ms`, `local_cost`, `remote_cost`, `local_quality_score`, `remote_quality_score`) from the training features. These values are computed *after* execution and will cause severe target leakage if included during model training.
3. **Input Scaling:** Scale/normalize features like `prompt_length`, `word_count`, and `estimated_input_tokens` before inputting into algorithms that are sensitive to scale (e.g., Support Vector Machines or Logistic Regression). Tree-based models (XGBoost/RandomForest) do not require scaling.