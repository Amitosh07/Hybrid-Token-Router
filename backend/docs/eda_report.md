# Exploratory Data Analysis Report

## Dataset Overview

- Rows: 640
- Columns: 28
- Trainable pre-routing features: 16
- Post-inference columns reserved for offline evaluation: local_cost, local_latency_ms, local_quality_score, remote_cost, remote_latency_ms, remote_quality_score

## Class Distribution

- `REMOTE`: 479
- `LOCAL`: 161

## Feature Statistics

| stat | prompt_length | word_count | estimated_input_tokens | contains_code | contains_math | contains_json | contains_markdown | contains_numbers | contains_question | reasoning_score | technical_complexity | reasoning_depth | task_complexity | constraint_complexity | context_complexity | complexity_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| count | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 | 640.0 |
| mean | 200.4688 | 30.1109 | 50.4969 | 0.1812 | 0.1 | 0.0125 | 0.0094 | 0.3641 | 0.2844 | 2.1031 | 0.0914 | 0.0695 | 0.4455 | 0.0245 | 0.1001 | 0.1993 |
| std | 107.9274 | 14.4665 | 26.9627 | 0.3855 | 0.3002 | 0.1112 | 0.0964 | 0.4815 | 0.4515 | 1.2887 | 0.1986 | 0.1462 | 0.2421 | 0.0593 | 0.0523 | 0.1367 |
| min | 26.0 | 6.0 | 7.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.2 | 0.0 | 0.0251 | 0.0539 |
| 25% | 122.0 | 20.0 | 31.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.0 | 0.25 | 0.0 | 0.0655 | 0.0786 |
| 50% | 165.0 | 26.0 | 42.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 2.0 | 0.0 | 0.0 | 0.35 | 0.0 | 0.0813 | 0.1696 |
| 75% | 257.5 | 36.0 | 65.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 3.0 | 0.0 | 0.0 | 0.65 | 0.0 | 0.1172 | 0.3046 |
| max | 762.0 | 108.0 | 191.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 7.0 | 0.95 | 0.9 | 1.0 | 0.36 | 0.3797 | 0.7403 |

## Missing Value Analysis

- No missing values detected.

## Outlier Analysis

- `prompt_length`: 19 IQR outliers
- `word_count`: 32 IQR outliers
- `estimated_input_tokens`: 19 IQR outliers
- `contains_code`: 0 IQR outliers
- `contains_math`: 0 IQR outliers
- `contains_json`: 0 IQR outliers
- `contains_markdown`: 0 IQR outliers
- `contains_numbers`: 0 IQR outliers
- `contains_question`: 0 IQR outliers
- `reasoning_score`: 3 IQR outliers
- `technical_complexity`: 0 IQR outliers
- `reasoning_depth`: 0 IQR outliers
- `task_complexity`: 0 IQR outliers
- `constraint_complexity`: 0 IQR outliers
- `context_complexity`: 35 IQR outliers
- `complexity_score`: 3 IQR outliers

## Correlation Matrix

- See `backend/app/ml/plots/correlation_heatmap.png`.
- `prompt_length` vs `word_count`: 0.9601
- `prompt_length` vs `estimated_input_tokens`: 0.9999
- `word_count` vs `estimated_input_tokens`: 0.9601
- `reasoning_score` vs `complexity_score`: 0.9835

## Feature Importance

- `prompt_length`: 0.163564
- `context_complexity`: 0.151328
- `word_count`: 0.147974
- `complexity_score`: 0.137385
- `estimated_input_tokens`: 0.120093
- `task_complexity`: 0.078554
- `technical_complexity`: 0.033203
- `reasoning_depth`: 0.033176
- `constraint_complexity`: 0.032714
- `reasoning_score`: 0.026426
- `contains_code`: 0.021356
- `contains_numbers`: 0.020708
- `contains_question`: 0.020056
- `contains_math`: 0.009573
- `contains_json`: 0.002455

## Feature Distributions and Pairwise Relationships

- `class_distribution`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/class_distribution.png`
- `histograms`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_histograms.png`
- `correlation_heatmap`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/correlation_heatmap.png`
- `boxplots`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_boxplots.png`
- `feature_importance`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/feature_importance.png`
- `pairwise_relationships`: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/pairwise_relationships.png`
