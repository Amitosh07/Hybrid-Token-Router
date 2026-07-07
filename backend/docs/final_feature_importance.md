# Final Feature Importance Report

- Importance method: native feature importance
- Bar chart: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/final_feature_importance_bar.png`
- Sorted chart: `C:/Users/Amitosh Nigam/Desktop/Hybrid-Token-Router/backend/app/ml/plots/final_feature_importance_sorted.png`

## Top 20 Features

| Rank | Feature | Importance | Normalized Share | Interpretation |
|---:|---|---:|---:|---|
| 1 | `prompt_length` | 0.457567 | 0.4576 | Longer prompts often imply more context, constraints, or scope, shifting routing utility. |
| 2 | `context_complexity` | 0.162005 | 0.1620 | Dense terminology and context load change whether remote quality is worth the trade-off. |
| 3 | `task_complexity` | 0.111805 | 0.1118 | Implementation, synthesis, planning, and multi-deliverable wording influences provider choice. |
| 4 | `reasoning_depth` | 0.086401 | 0.0864 | Deeper analytical or multi-step reasoning can favor stronger remote responses. |
| 5 | `constraint_complexity` | 0.065073 | 0.0651 | Explicit formatting, safety, or consistency constraints affect routing confidence. |
| 6 | `technical_complexity` | 0.048673 | 0.0487 | Specialized technical domains alter expected response quality requirements. |
| 7 | `contains_numbers` | 0.023172 | 0.0232 | Numerical or mathematical content can indicate precision-sensitive work. |
| 8 | `reasoning_score` | 0.014217 | 0.0142 | Deeper analytical or multi-step reasoning can favor stronger remote responses. |
| 9 | `contains_code` | 0.010330 | 0.0103 | Code-like prompts follow different routing patterns than general questions. |
| 10 | `contains_markdown` | 0.008052 | 0.0081 | This pre-routing signal contributes to the learned Decision Engine approximation. |
| 11 | `contains_question` | 0.007052 | 0.0071 | This pre-routing signal contributes to the learned Decision Engine approximation. |
| 12 | `contains_math` | 0.004385 | 0.0044 | Numerical or mathematical content can indicate precision-sensitive work. |
| 13 | `contains_json` | 0.001268 | 0.0013 | This pre-routing signal contributes to the learned Decision Engine approximation. |

## Examples

- `prompt_length`: A detailed cloud migration request carries more operational context than a one-line definition.
- `context_complexity`: A healthcare consent translation with clinical terminology has more context pressure than a casual reminder.
- `task_complexity`: A rollout plan with milestones, risks, and owners is more demanding than a simple checklist.
- `reasoning_depth`: Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE.
- `constraint_complexity`: Requests with exact output formats or compliance constraints raise the cost of a weak answer.
- `technical_complexity`: Distributed consensus, cryptography, and Kubernetes prompts trigger specialized technical signals.
- `contains_numbers`: Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE.
- `reasoning_score`: Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE.
- `contains_code`: Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE.
- `contains_markdown`: Prompts where this feature is active change the learned probability assigned to LOCAL or REMOTE.
