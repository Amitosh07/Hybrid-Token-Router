# Feature Importance and Selection Report

## Leakage Policy

Training excludes all post-inference latency, cost, and quality metrics. These fields are reserved for offline evaluation only.

## Retained Features

- `domain`
- `constraint_count`
- `estimated_complexity`
- `output_format`
- `prompt_length`
- `contains_code`
- `contains_math`
- `contains_numbers`
- `contains_question`
- `reasoning_score`
- `technical_complexity`
- `reasoning_depth`
- `task_complexity`
- `constraint_complexity`
- `context_complexity`
- `requested_format`

## Removed Features

- `algorithmic_complexity`: removed because it is constant.
- `api_keywords`: removed because it is constant.
- `code_indicators`: removed because it is constant.
- `complexity_score`: removed because it is redundant with a highly correlated feature.
- `constraint_density`: removed because it is constant.
- `contains_json`: removed because it is constant.
- `contains_markdown`: removed because it is constant.
- `dependency_between_subtasks`: removed because it is constant.
- `estimated_input_tokens`: removed because it is redundant with a highly correlated feature.
- `math_indicators`: removed because it is constant.
- `multi_turn_context`: removed because it is constant.
- `presence_of_tables`: removed because it is constant.
- `sql_indicators`: removed because it is constant.
- `system_design_keywords`: removed because it is constant.
- `word_count`: removed because it is redundant with a highly correlated feature.

## Low Variance Features

- `context_complexity`

## Highly Correlated Pairs

- `prompt_length` and `word_count`: correlation 0.9788
- `prompt_length` and `estimated_input_tokens`: correlation 0.9999
- `word_count` and `estimated_input_tokens`: correlation 0.9787
- `reasoning_score` and `complexity_score`: correlation 0.9857

## Top Feature Importance

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

## Target Correlation

- `complexity_score`: -0.244117
- `reasoning_score`: -0.242388
- `task_complexity`: -0.220969
- `reasoning_depth`: -0.193343
- `constraint_count`: -0.105230
- `estimated_complexity`: -0.075593
- `estimated_input_tokens`: -0.069716
- `prompt_length`: -0.069701
- `constraint_complexity`: -0.054193
- `word_count`: -0.054000
- `context_complexity`: -0.053919
- `contains_math`: -0.038040
- `contains_numbers`: 0.035480
- `technical_complexity`: -0.028085
- `contains_code`: -0.026132
