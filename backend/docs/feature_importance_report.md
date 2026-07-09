# Feature Importance and Selection Report

## Leakage Policy

Training excludes all post-inference latency, cost, and quality metrics. These fields are reserved for offline evaluation only.

## Retained Features

- `domain`
- `estimated_complexity`
- `output_format`
- `source`
- `augmentation_band`
- `prompt_length`
- `contains_code`
- `contains_math`
- `contains_markdown`
- `contains_numbers`
- `contains_question`
- `technical_complexity`
- `reasoning_depth`
- `task_complexity`
- `constraint_complexity`
- `context_complexity`
- `constraint_density`
- `requested_format`
- `sql_indicators`
- `api_keywords`
- `system_design_keywords`
- `algorithmic_complexity`
- `dependency_between_subtasks`
- `multi_turn_context`
- `code_indicators`
- `math_indicators`
- `routing_score`
- `routing_confidence`

## Removed Features

- `complexity_score`: removed because it is redundant with a highly correlated feature.
- `constraint_count`: removed because it is constant.
- `contains_json`: removed because it is constant.
- `estimated_input_tokens`: removed because it is redundant with a highly correlated feature.
- `presence_of_tables`: removed because it is constant.
- `reasoning_score`: removed because it is redundant with a highly correlated feature.
- `word_count`: removed because it is redundant with a highly correlated feature.

## Low Variance Features

- `contains_markdown`
- `constraint_density`

## Highly Correlated Pairs

- `estimated_complexity` and `reasoning_score`: correlation 0.9922
- `estimated_complexity` and `complexity_score`: correlation 1.0
- `prompt_length` and `word_count`: correlation 0.9989
- `prompt_length` and `estimated_input_tokens`: correlation 1.0
- `word_count` and `estimated_input_tokens`: correlation 0.9989
- `reasoning_score` and `complexity_score`: correlation 0.9922

## Top Feature Importance

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

## Target Correlation

- `reasoning_score`: 0.848111
- `estimated_complexity`: 0.847433
- `complexity_score`: 0.847433
- `routing_score`: 0.815339
- `task_complexity`: 0.747375
- `context_complexity`: 0.732914
- `api_keywords`: 0.710450
- `reasoning_depth`: 0.703883
- `technical_complexity`: 0.691646
- `constraint_complexity`: 0.562190
- `estimated_input_tokens`: 0.419940
- `prompt_length`: 0.419935
- `word_count`: 0.419932
- `dependency_between_subtasks`: 0.331307
- `sql_indicators`: 0.317633
