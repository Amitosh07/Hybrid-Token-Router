# Supervised Router Model Report

## Best Model

- Best model: `Random Forest`
- Estimator: `RandomForestClassifier`
- Training accuracy: 0.9453
- Validation accuracy: 0.8790
- Validation precision: 0.1927
- Validation recall: 0.3889
- Validation F1: 0.2577
- Validation ROC AUC: 0.8391
- Prediction latency: 0.084955 ms/sample

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7190 | 0.1279 | 0.7222 | 0.2173 | 0.8031 | 0.2364 |
| Random Forest | 0.8790 | 0.1927 | 0.3889 | 0.2577 | 0.8391 | 0.2284 |
| Gradient Boosting | 0.9450 | 0.3333 | 0.0185 | 0.0351 | 0.8614 | 0.0136 |

## Cross Validation

- `accuracy`: mean 0.8702, std 0.0094, folds [0.869, 0.878, 0.855, 0.882, 0.867]
- `precision`: mean 0.1685, std 0.0157, folds [0.146789, 0.179245, 0.152672, 0.186275, 0.177419]
- `recall`: mean 0.3571, std 0.0382, folds [0.296296, 0.351852, 0.37037, 0.351852, 0.415094]
- `f1`: mean 0.2284, std 0.0195, folds [0.196319, 0.2375, 0.216216, 0.24359, 0.248588]
- `roc_auc`: mean 0.8370, std 0.0189, folds [0.817536, 0.865261, 0.818906, 0.830573, 0.852484]

## Confusion Matrix

Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.

`[[858, 88], [33, 21]]`

## Top 15 Most Influential Features

- `reasoning_score`: 0.260692
- `reasoning_depth`: 0.154570
- `task_complexity`: 0.130491
- `prompt_length`: 0.078122
- `context_complexity`: 0.067420
- `estimated_complexity`: 0.042776
- `technical_complexity`: 0.032223
- `constraint_count`: 0.025285
- `constraint_complexity`: 0.023688
- `contains_numbers`: 0.013483
- `domain_education`: 0.012716
- `output_format_markdown table`: 0.010676
- `contains_question`: 0.010547
- `output_format_plain paragraphs`: 0.009059
- `output_format_annotated code block`: 0.008010

## Misclassified Examples

- `gen_001933`: actual `LOCAL`, predicted `REMOTE`, confidence 0.8563. Prompt: Summarize a detailed assessment rubric for a curriculum designer, highlighting attendance, accessibility gap, and impact on student cohort. Include a realistic constraint and one exception case. Format the response as JSON object. Provide e
- `gen_000049`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6953. Prompt: Write production-ready code for a store operations lead that processes a customer review feed involving loyalty segment and handles stockout. Format the response as checklist. Provide enough context for a precise answer without requiring ex
- `gen_001753`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6007. Prompt: Write production-ready code for a policy analyst that processes a benefits workflow involving district and handles equity gap. Format the response as JSON object. Discuss edge cases and failure modes.
- `gen_000296`: actual `REMOTE`, predicted `LOCAL`, confidence 0.5865. Prompt: Compute the break-even threshold for a pricing model when budget gap adds a variable cost to each feature request. Format the response as annotated code block. Use enough detail to support a robust answer, including context, assumptions, an
- `gen_000130`: actual `REMOTE`, predicted `LOCAL`, confidence 0.9889. Prompt: Design a phased rollout for course outline adoption across teams that depend on online module. Include a realistic constraint and one exception case. Format the response as markdown table. Use enough detail to support a robust answer, inclu
- `gen_003758`: actual `REMOTE`, predicted `LOCAL`, confidence 0.8964. Prompt: Compose a reflective story in which embedding model becomes a symbol for changing inference latency. Include a realistic constraint and one exception case. Format the response as bullet list. Use enough detail to support a robust answer, in
- `gen_001205`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9001. Prompt: Summarize a detailed student survey for a curriculum designer, highlighting completion rate, grading inconsistency, and impact on student cohort. Format the response as numbered steps. Provide enough context for a precise answer without req
- `gen_001469`: actual `LOCAL`, predicted `REMOTE`, confidence 0.5436. Prompt: Summarize a detailed fraud alert for a risk analyst, highlighting default rate, liquidity shortfall, and impact on loan applicant. Format the response as comparison matrix. Provide enough context for a precise answer without requiring exter
- `gen_002384`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6347. Prompt: Compute the break-even threshold for a motion planner when actuator failure adds a variable cost to each mobile robot. Format the response as JSON object. Avoid unexplained jargon. Make assumptions explicit for auditability.
- `gen_005692`: actual `LOCAL`, predicted `REMOTE`, confidence 0.5343. Prompt: Translate this operational notice about scope creep into French for stakeholders using quarterly plan. Format the response as short executive brief. Provide enough context for a precise answer without requiring external documents. Mark any 
- `gen_000516`: actual `LOCAL`, predicted `REMOTE`, confidence 0.7207. Prompt: Translate this operational notice about serving skew into French for stakeholders using model card. Include a realistic constraint and one exception case. Format the response as numbered steps. Provide enough context for a precise answer wi
- `gen_003181`: actual `LOCAL`, predicted `REMOTE`, confidence 0.5985. Prompt: Summarize a detailed quarterly plan for a operations director, highlighting ARR, churn increase, and impact on feature request. Include a realistic constraint and one exception case. Format the response as plain paragraphs. Provide enough c
- `gen_000457`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9249. Prompt: Write production-ready code for a SOC analyst that processes a access log involving identity provider and handles privilege escalation. Format the response as bullet list. Use enough detail to support a robust answer, including context, ass
- `gen_002187`: actual `REMOTE`, predicted `LOCAL`, confidence 0.6382. Prompt: Analyze whether a portfolio manager should prioritize cash-flow forecast improvements or direct mitigation of model drift. Format the response as annotated code block. Discuss edge cases and failure modes.
- `gen_002192`: actual `REMOTE`, predicted `LOCAL`, confidence 0.8008. Prompt: Compute the break-even threshold for a ledger export when regulatory breach adds a variable cost to each loan applicant. Format the response as plain paragraphs. Provide enough context for a precise answer without requiring external documen

## Interpretation

Logistic Regression underperforms Random Forest, which suggests the routing boundary is not purely linear. Interactions among complexity, structural flags, and token pressure appear important. Random Forest was selected because it produced the strongest holdout ranking while remaining robust on cross validation.

## Limitations

- The target is generated by the Phase 2 Decision Engine, so the model learns that policy rather than independent human preferences.
- The model intentionally does not use post-inference latency, cost, or quality metrics at prediction time.
- Dataset metadata columns are excluded because live prompts only provide extracted pre-routing features.

## Recommendations

- Keep this model behind a feature flag until it is shadow-tested against live router traffic.
- Retrain whenever benchmark composition, local model quality, or remote pricing changes materially.
- Add more LOCAL-positive examples if live traffic shows over-routing to REMOTE.
