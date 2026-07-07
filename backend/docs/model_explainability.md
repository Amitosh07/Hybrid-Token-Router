# Model Explainability

## How Routing Decisions Are Made

The live ML router extracts pre-routing prompt features, aligns them to the saved training feature list, applies the saved preprocessing pipeline inside `router_model.pkl`, and returns the provider with the highest predicted probability.

The model never receives post-inference latency, cost, or quality columns at prediction time.

## Probability and Confidence

- `prediction_probability` is the probability assigned to the selected provider.
- `prediction_confidence` is the larger of LOCAL and REMOTE probabilities.
- High confidence means the model is far from its learned decision boundary; it does not guarantee the provider response will be better for every individual prompt.

## Example Predictions

- Prompt: Write a Go service that validates signed payment webhooks and retries failed database writes. Selected `REMOTE` with confidence 0.8564. Top signal: prompt_length
- Prompt: Translate this short delivery update into Spanish using a friendly tone. Selected `REMOTE` with confidence 0.9298. Top signal: prompt_length
- Prompt: Analyze the legal and ethical risks of deploying facial recognition in public schools. Selected `REMOTE` with confidence 0.8926. Top signal: prompt_length

## Feature Ranking Summary

- 1. `prompt_length` (0.457567): Longer prompts often imply more context, constraints, or scope, shifting routing utility.
- 2. `context_complexity` (0.162005): Dense terminology and context load change whether remote quality is worth the trade-off.
- 3. `task_complexity` (0.111805): Implementation, synthesis, planning, and multi-deliverable wording influences provider choice.
- 4. `reasoning_depth` (0.086401): Deeper analytical or multi-step reasoning can favor stronger remote responses.
- 5. `constraint_complexity` (0.065073): Explicit formatting, safety, or consistency constraints affect routing confidence.
- 6. `technical_complexity` (0.048673): Specialized technical domains alter expected response quality requirements.
- 7. `contains_numbers` (0.023172): Numerical or mathematical content can indicate precision-sensitive work.
- 8. `reasoning_score` (0.014217): Deeper analytical or multi-step reasoning can favor stronger remote responses.
- 9. `contains_code` (0.010330): Code-like prompts follow different routing patterns than general questions.
- 10. `contains_markdown` (0.008052): This pre-routing signal contributes to the learned Decision Engine approximation.

## Limitations

- The model learns from Decision Engine labels, so it approximates that policy rather than independent human preference.
- It cannot observe live provider failures until after routing; provider-level fallback still protects local runtime failures.
- Feature importance explains the fitted model globally, while individual predictions can depend on feature interactions.
