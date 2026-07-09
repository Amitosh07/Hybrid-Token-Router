# Exploratory Data Analysis Report

## Dataset Overview

- Rows: 11973
- Columns: 53
- Trainable pre-routing features: 35
- Post-inference columns reserved for offline evaluation: local_cost, local_heuristic_quality, local_latency_ms, local_llm_quality, local_quality_score, local_tokens, remote_cost, remote_heuristic_quality, remote_latency_ms, remote_llm_quality, remote_quality_score, remote_tokens

## Class Distribution

- `REMOTE`: 6257
- `LOCAL`: 5716

## Feature Statistics

| stat | constraint_count | estimated_complexity | prompt_length | word_count | estimated_input_tokens | contains_code | contains_math | contains_json | contains_markdown | contains_numbers | contains_question | reasoning_score | technical_complexity | reasoning_depth | task_complexity | constraint_complexity | context_complexity | complexity_score | constraint_density | presence_of_tables | sql_indicators | api_keywords | system_design_keywords | algorithmic_complexity | dependency_between_subtasks | multi_turn_context | code_indicators | math_indicators | routing_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 | 11973.0 |
| mean | 0.0 | 0.334 | 1415.9652 | 191.202 | 354.3631 | 0.1063 | 0.0412 | 0.0 | 0.0002 | 0.0949 | 0.4863 | 3.4237 | 0.2418 | 0.2076 | 0.576 | 0.0915 | 0.1127 | 0.334 | 0.0102 | 0.0 | 0.2816 | 0.4493 | 0.0592 | 0.0294 | 0.1149 | 0.0043 | 0.1329 | 0.0845 | 31.7779 |
| std | 0.0 | 0.226 | 2872.0204 | 387.6918 | 718.0065 | 0.3083 | 0.1987 | 0.0 | 0.0129 | 0.2931 | 0.4998 | 2.1915 | 0.2487 | 0.2549 | 0.288 | 0.1296 | 0.0474 | 0.226 | 0.0171 | 0.0 | 0.4498 | 0.4974 | 0.236 | 0.1689 | 0.3189 | 0.0651 | 0.3395 | 0.2782 | 24.3233 |
| min | 0.0 | 0.054 | 50.0 | 9.0 | 13.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0366 | 0.054 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.0 |
| 25% | 0.0 | 0.1048 | 110.0 | 16.0 | 28.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.25 | 0.0 | 0.0611 | 0.1048 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 10.0 |
| 50% | 0.0 | 0.3333 | 278.0 | 34.0 | 70.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 3.0 | 0.2647 | 0.0 | 0.5 | 0.0 | 0.1125 | 0.3333 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 26.0 |
| 75% | 0.0 | 0.5246 | 530.0 | 72.0 | 133.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 5.0 | 0.3965 | 0.3418 | 0.85 | 0.18 | 0.1453 | 0.5246 | 0.0143 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 48.0 |
| max | 0.0 | 1.0 | 15870.0 | 1988.0 | 3968.0 | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 10.0 | 1.0 | 0.9087 | 1.0 | 0.74 | 0.3538 | 1.0 | 0.1818 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 133.0 |

## Missing Value Analysis

- No missing values detected.

## Outlier Analysis

- `constraint_count`: 0 IQR outliers
- `estimated_complexity`: 0 IQR outliers
- `prompt_length`: 2595 IQR outliers
- `word_count`: 2615 IQR outliers
- `estimated_input_tokens`: 2595 IQR outliers
- `contains_code`: 0 IQR outliers
- `contains_math`: 0 IQR outliers
- `contains_json`: 0 IQR outliers
- `contains_markdown`: 0 IQR outliers
- `contains_numbers`: 0 IQR outliers
- `contains_question`: 0 IQR outliers
- `reasoning_score`: 0 IQR outliers
- `technical_complexity`: 14 IQR outliers
- `reasoning_depth`: 100 IQR outliers
- `task_complexity`: 0 IQR outliers
- `constraint_complexity`: 205 IQR outliers
- `context_complexity`: 41 IQR outliers
- `complexity_score`: 0 IQR outliers
- `constraint_density`: 1220 IQR outliers
- `presence_of_tables`: 0 IQR outliers
- `sql_indicators`: 0 IQR outliers
- `api_keywords`: 0 IQR outliers
- `system_design_keywords`: 0 IQR outliers
- `algorithmic_complexity`: 0 IQR outliers
- `dependency_between_subtasks`: 0 IQR outliers
- `multi_turn_context`: 0 IQR outliers
- `code_indicators`: 0 IQR outliers
- `math_indicators`: 0 IQR outliers
- `routing_score`: 41 IQR outliers

## Correlation Matrix

- See `backend/app/ml/plots/correlation_heatmap.png`.
- `estimated_complexity` vs `reasoning_score`: 0.9922
- `estimated_complexity` vs `complexity_score`: 1.0
- `prompt_length` vs `word_count`: 0.9989
- `prompt_length` vs `estimated_input_tokens`: 1.0
- `word_count` vs `estimated_input_tokens`: 0.9989
- `reasoning_score` vs `complexity_score`: 0.9922

## Feature Importance

- `routing_score`: 0.243664
- `estimated_complexity`: 0.144517
- `reasoning_score`: 0.135949
- `complexity_score`: 0.131024
- `context_complexity`: 0.058614
- `task_complexity`: 0.056815
- `reasoning_depth`: 0.053743
- `word_count`: 0.039752
- `technical_complexity`: 0.035005
- `prompt_length`: 0.024559
- `api_keywords`: 0.020927
- `estimated_input_tokens`: 0.020827
- `constraint_density`: 0.014214
- `constraint_complexity`: 0.008814
- `sql_indicators`: 0.003228

## Feature Distributions and Pairwise Relationships

- `class_distribution`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/class_distribution.png`
- `histograms`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_histograms.png`
- `correlation_heatmap`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/correlation_heatmap.png`
- `boxplots`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_boxplots.png`
- `feature_importance`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_importance.png`
- `pairwise_relationships`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/pairwise_relationships.png`
