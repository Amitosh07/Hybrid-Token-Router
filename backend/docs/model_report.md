# Supervised Router Model Report

## Best Model

- Best model: `Gradient Boosting`
- Estimator: `GradientBoostingClassifier`
- Training accuracy: 0.8887
- Validation accuracy: 0.7031
- Validation precision: 0.7544
- Validation recall: 0.8958
- Validation F1: 0.8190
- Validation ROC AUC: 0.6107
- Prediction latency: 0.017312 ms/sample

## Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | CV F1 Mean |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5547 | 0.7746 | 0.5729 | 0.6587 | 0.5908 | 0.7045 |
| Random Forest | 0.6094 | 0.7614 | 0.6979 | 0.7283 | 0.6214 | 0.7491 |
| Gradient Boosting | 0.7031 | 0.7544 | 0.8958 | 0.8190 | 0.6107 | 0.8222 |

## Cross Validation

- `accuracy`: mean 0.7125, std 0.0322, folds [0.742188, 0.757812, 0.695312, 0.671875, 0.695312]
- `precision`: mean 0.7651, std 0.0198, folds [0.788991, 0.787611, 0.747826, 0.759615, 0.741379]
- `recall`: mean 0.8894, std 0.0351, folds [0.895833, 0.927083, 0.895833, 0.822917, 0.905263]
- `f1`: mean 0.8222, std 0.0214, folds [0.839024, 0.851675, 0.815166, 0.79, 0.815166]
- `roc_auc`: mean 0.6569, std 0.0351, folds [0.659831, 0.70459, 0.630371, 0.606934, 0.682775]

## Confusion Matrix

Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.

`[[4, 28], [10, 86]]`

## Top 15 Most Influential Features

- `prompt_length`: 0.457567
- `context_complexity`: 0.162005
- `task_complexity`: 0.111805
- `reasoning_depth`: 0.086401
- `constraint_complexity`: 0.065073
- `technical_complexity`: 0.048673
- `contains_numbers`: 0.023172
- `reasoning_score`: 0.014217
- `contains_code`: 0.010330
- `contains_markdown`: 0.008052
- `contains_question`: 0.007052
- `contains_math`: 0.004385
- `contains_json`: 0.001268

## Misclassified Examples

- `general_070`: actual `LOCAL`, predicted `REMOTE`, confidence 0.8542. Prompt: Provide a comprehensive, technical explanation of the aerodynamic and fluid dynamics principles governing the design of modern offshore wind turbine rotors. Detail the lift and drag forces acting on the blades, the blade element momentum (B
- `translation_062`: actual `REMOTE`, predicted `LOCAL`, confidence 0.6882. Prompt: Translate this highly technical scientific paper extract about semiconductor fabrication from English to Japanese. The translation must utilize the exact academic and industry terminology for physical vapor deposition, photolithography, sub
- `planning_022`: actual `LOCAL`, predicted `REMOTE`, confidence 0.8557. Prompt: Provide a simple packing checklist for a 3-day summer weekend trip to a beach resort, listing essentials, clothing, and toiletries.
- `mathematics_001`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9827. Prompt: Solve for x: 5x - 12 = 28.
- `reasoning_074`: actual `REMOTE`, predicted `LOCAL`, confidence 0.5649. Prompt: Evaluate the implications of the halting problem in computer science on the feasibility of writing a perfect static analysis tool that guarantees memory safety and deadlock absence in arbitrary multi-threaded programs.
- `general_072`: actual `LOCAL`, predicted `REMOTE`, confidence 0.5035. Prompt: Provide a detailed scientific explanation of the geological and geophysical processes driving the formation and migration of hydrocarbons in sedimentary basins. Trace the organic matter deposition, thermal maturation (diagenesis, catagenesi
- `general_062`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9195. Prompt: Provide a detailed, rigorous explanation of the physical and mechanical principles of operation of a multi-stage industrial gas turbine. Detail the thermodynamic transformations (Brayton cycle), the aerodynamic design of the compressor and 
- `general_055`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6450. Prompt: Explain the mechanics of how a GPS receiver determines its precise location on Earth using trilateration, clock synchronization, and signals from at least four satellites.
- `creative_writing_050`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6155. Prompt: Draft a creative exchange of letters between two rival astronomers in the 18th century, each claiming to have discovered the same comet first. Maintain historical syntax.
- `general_057`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6708. Prompt: Explain the physiological effects of chronic stress on the human endocrine system, tracing the activation of the hypothalamic-pituitary-adrenal (HPA) axis and cortisol release.
- `translation_035`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9035. Prompt: Translate this status update from English to Italian: 'Your order has been shipped and will arrive by Thursday.'
- `reasoning_066`: actual `LOCAL`, predicted `REMOTE`, confidence 0.5452. Prompt: Analyze the ethical, legal, and systemic safety implications of deploying autonomous weapons systems equipped with computer vision for target selection. Address accountability gaps and human-in-the-loop overrides.
- `coding_079`: actual `LOCAL`, predicted `REMOTE`, confidence 0.9108. Prompt: Write a C++ class implementing a lock-free Compare-and-Swap (CAS) based queue. Define structure for node pointers, atomic operations, enqueue/dequeue methods, and handle memory allocation safety under high concurrency.
- `translation_056`: actual `LOCAL`, predicted `REMOTE`, confidence 0.6322. Prompt: Translate this clinical trial protocol summary from English to Traditional Chinese, focusing on eligibility criteria and primary outcome measures: 'Eligible participants must be aged 18 to 65 with a documented history of chronic hypertensio
- `planning_056`: actual `LOCAL`, predicted `REMOTE`, confidence 0.7850. Prompt: Create a comprehensive plan for a software company to run a beta test program for 200 external users, including feedback collection channels and bug tracking.

## Interpretation

Logistic Regression underperforms Random Forest, which suggests the routing boundary is not purely linear. Interactions among complexity, structural flags, and token pressure appear important. Gradient Boosting was selected as the XGBoost fallback and performed best among the available baselines.

## Limitations

- The target is generated by the Phase 2 Decision Engine, so the model learns that policy rather than independent human preferences.
- The model intentionally does not use post-inference latency, cost, or quality metrics at prediction time.
- Dataset metadata columns are excluded because live prompts only provide extracted pre-routing features.

## Recommendations

- Keep this model behind a feature flag until it is shadow-tested against live router traffic.
- Retrain whenever benchmark composition, local model quality, or remote pricing changes materially.
- Add more LOCAL-positive examples if live traffic shows over-routing to REMOTE.
