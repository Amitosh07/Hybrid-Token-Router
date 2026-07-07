# Feature Importance and Selection Report

## Leakage Policy

Training excludes all post-inference latency, cost, and quality metrics. These fields are reserved for offline evaluation only.

## Retained Features

- `prompt_length`
- `contains_code`
- `contains_math`
- `contains_json`
- `contains_markdown`
- `contains_numbers`
- `contains_question`
- `reasoning_score`
- `technical_complexity`
- `reasoning_depth`
- `task_complexity`
- `constraint_complexity`
- `context_complexity`

## Removed Features

- `complexity_score`: removed because it is redundant with a highly correlated feature.
- `estimated_input_tokens`: removed because it is redundant with a highly correlated feature.
- `word_count`: removed because it is redundant with a highly correlated feature.

## Low Variance Features

- None detected above the constant-feature threshold.

## Highly Correlated Pairs

- `prompt_length` and `word_count`: correlation 0.9601
- `prompt_length` and `estimated_input_tokens`: correlation 0.9999
- `word_count` and `estimated_input_tokens`: correlation 0.9601
- `reasoning_score` and `complexity_score`: correlation 0.9835

## Top Feature Importance

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

## Target Correlation

- `context_complexity`: -0.153398
- `prompt_length`: -0.104264
- `estimated_input_tokens`: -0.104120
- `technical_complexity`: -0.093294
- `reasoning_score`: -0.082208
- `word_count`: -0.079751
- `complexity_score`: -0.073429
- `task_complexity`: -0.062505
- `constraint_complexity`: 0.047962
- `contains_numbers`: 0.042015
- `contains_question`: 0.030208
- `contains_code`: 0.029737
- `contains_markdown`: -0.018333
- `contains_math`: -0.010803
- `reasoning_depth`: -0.007061
