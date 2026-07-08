# Exploratory Data Analysis Report

## Dataset Overview

- Rows: 4600
- Columns: 49
- Trainable pre-routing features: 31
- Post-inference columns reserved for offline evaluation: local_cost, local_heuristic_quality, local_latency_ms, local_llm_quality, local_quality_score, local_tokens, remote_cost, remote_heuristic_quality, remote_latency_ms, remote_llm_quality, remote_quality_score, remote_tokens

## Class Distribution

- `LOCAL`: 4354
- `REMOTE`: 246

## Feature Statistics

| stat | constraint_count | estimated_complexity | prompt_length | word_count | estimated_input_tokens | contains_code | contains_math | contains_json | contains_markdown | contains_numbers | contains_question | reasoning_score | technical_complexity | reasoning_depth | task_complexity | constraint_complexity | context_complexity | complexity_score | constraint_density | presence_of_tables | sql_indicators | api_keywords | system_design_keywords | algorithmic_complexity | dependency_between_subtasks | multi_turn_context | code_indicators | math_indicators |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 | 4600.0 |
| mean | 2.8863 | 0.775 | 389.1554 | 54.0333 | 97.6657 | 0.2291 | 0.1011 | 0.0 | 0.0 | 0.2976 | 0.3339 | 3.7083 | 0.1453 | 0.2681 | 0.6567 | 0.1619 | 0.1454 | 0.3718 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| std | 1.0655 | 0.2301 | 93.0737 | 12.0647 | 23.2696 | 0.4203 | 0.3015 | 0.0 | 0.0 | 0.4573 | 0.4717 | 1.6475 | 0.1944 | 0.2094 | 0.2728 | 0.087 | 0.0312 | 0.162 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| min | 1.0 | 0.345 | 175.0 | 25.0 | 44.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.2 | 0.08 | 0.085 | 0.0791 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 25% | 2.0 | 0.535 | 317.0 | 45.0 | 80.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.35 | 0.08 | 0.1242 | 0.2426 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 50% | 3.0 | 0.84 | 399.0 | 55.0 | 100.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.0 | 0.0 | 0.2929 | 0.75 | 0.18 | 0.1437 | 0.3999 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 75% | 4.0 | 1.0 | 460.0 | 63.0 | 115.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 5.0 | 0.1875 | 0.3309 | 0.9 | 0.24 | 0.15 | 0.4761 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| max | 5.0 | 1.0 | 598.0 | 80.0 | 150.0 | 1.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 | 9.0 | 0.95 | 0.7 | 1.0 | 0.38 | 0.3204 | 0.9011 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Missing Value Analysis

- No missing values detected.

## Outlier Analysis

- `constraint_count`: 0 IQR outliers
- `estimated_complexity`: 0 IQR outliers
- `prompt_length`: 0 IQR outliers
- `word_count`: 0 IQR outliers
- `estimated_input_tokens`: 0 IQR outliers
- `contains_code`: 0 IQR outliers
- `contains_math`: 0 IQR outliers
- `contains_json`: 0 IQR outliers
- `contains_markdown`: 0 IQR outliers
- `contains_numbers`: 0 IQR outliers
- `contains_question`: 0 IQR outliers
- `reasoning_score`: 0 IQR outliers
- `technical_complexity`: 203 IQR outliers
- `reasoning_depth`: 0 IQR outliers
- `task_complexity`: 0 IQR outliers
- `constraint_complexity`: 0 IQR outliers
- `context_complexity`: 250 IQR outliers
- `complexity_score`: 3 IQR outliers
- `constraint_density`: 0 IQR outliers
- `presence_of_tables`: 0 IQR outliers
- `sql_indicators`: 0 IQR outliers
- `api_keywords`: 0 IQR outliers
- `system_design_keywords`: 0 IQR outliers
- `algorithmic_complexity`: 0 IQR outliers
- `dependency_between_subtasks`: 0 IQR outliers
- `multi_turn_context`: 0 IQR outliers
- `code_indicators`: 0 IQR outliers
- `math_indicators`: 0 IQR outliers

## Correlation Matrix

- See `backend/app/ml/plots/correlation_heatmap.png`.
- `prompt_length` vs `word_count`: 0.9788
- `prompt_length` vs `estimated_input_tokens`: 0.9999
- `word_count` vs `estimated_input_tokens`: 0.9787
- `reasoning_score` vs `complexity_score`: 0.9857

## Feature Importance

- `complexity_score`: 0.230544
- `reasoning_score`: 0.145795
- `prompt_length`: 0.091744
- `reasoning_depth`: 0.090066
- `task_complexity`: 0.087710
- `context_complexity`: 0.069082
- `word_count`: 0.066984
- `estimated_input_tokens`: 0.065221
- `estimated_complexity`: 0.046373
- `technical_complexity`: 0.028267
- `constraint_complexity`: 0.023262
- `constraint_count`: 0.019154
- `contains_math`: 0.011143
- `contains_numbers`: 0.010258
- `contains_question`: 0.008735

## Feature Distributions and Pairwise Relationships

- `class_distribution`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/class_distribution.png`
- `histograms`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_histograms.png`
- `correlation_heatmap`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/correlation_heatmap.png`
- `boxplots`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_boxplots.png`
- `feature_importance`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_importance.png`
- `pairwise_relationships`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/pairwise_relationships.png`
