# ML vs Heuristic Router Comparison

The Decision Engine labels in `training_dataset.csv` are treated as the offline ground truth for this comparison.

## Summary Metrics

- Heuristic Router accuracy vs Decision Engine: 0.3212
- ML Router accuracy vs Decision Engine: 0.9320
- Heuristic and ML agreement: 0.2668
- Heuristic average prediction latency: 1.959997 ms
- ML average prediction latency: 195.197704 ms
- Heuristic average confidence: 0.4856
- ML average confidence: 0.8865

## Strengths

- Heuristic Router: transparent, deterministic, and independent of training data drift.
- Decision Engine: uses offline benchmark outcomes to create utility-aware labels.
- ML Router: learns the Decision Engine policy from pre-routing features and can capture nonlinear feature interactions.

## Weaknesses

- Heuristic Router: hand-tuned thresholds may not match the validated Decision Engine labels.
- Decision Engine: depends on post-inference benchmark signals and cannot run before routing in production.
- ML Router: only generalizes as well as the Phase 2 dataset covers real traffic.

## Heuristic Misclassifications

- `gen_000001`: actual `REMOTE`, predicted `LOCAL`. Prompt: Write production-ready code for a clinical operations team that processes a triage protocol involving patient cohort and handles privacy exposure. Format the response as plain paragraphs. Keep the ans
- `gen_000006`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compose a reflective story in which patient cohort becomes a symbol for changing wait time. Format the response as numbered steps. Compare risks, trade-offs, and mitigations. Return the core result us
- `gen_000010`: actual `LOCAL`, predicted `REMOTE`. Prompt: Design a phased rollout for appointment workflow adoption across teams that depend on nurse queue. Include a realistic constraint and one exception case. Format the response as markdown table. Address
- `gen_000011`: actual `LOCAL`, predicted `REMOTE`. Prompt: Analyze whether a patient safety committee should prioritize triage protocol improvements or direct mitigation of audit failure. Include a realistic constraint and one exception case. Format the respo
- `gen_000018`: actual `LOCAL`, predicted `REMOTE`. Prompt: Design a phased rollout for consent form adoption across teams that depend on specialist referral. Include interacting constraints, uncertainty, and second-order effects. Format the response as YAML b
- `gen_000019`: actual `LOCAL`, predicted `REMOTE`. Prompt: Analyze whether a patient safety committee should prioritize appointment workflow improvements or direct mitigation of care delay. Include interacting constraints, uncertainty, and second-order effect
- `gen_000020`: actual `LOCAL`, predicted `REMOTE`. Prompt: Translate this operational notice about care delay into French for stakeholders using appointment workflow. Include interacting constraints, uncertainty, and second-order effects. Format the response 
- `gen_000022`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compose a reflective story in which EHR record becomes a symbol for changing readmission rate. Include interacting constraints, uncertainty, and second-order effects. Format the response as short exec
- `gen_000023`: actual `LOCAL`, predicted `REMOTE`. Prompt: Explain how a claims feed affects readmission rate, audit failure, and day-to-day decisions for a patient safety committee. Include interacting constraints, uncertainty, and second-order effects. Form
- `gen_000024`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compute the break-even threshold for a claims feed when audit failure adds a variable cost to each EHR record. Include interacting constraints, uncertainty, and second-order effects. Format the respon

## ML Misclassifications

- `gen_000009`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write production-ready code for a clinical operations team that processes a appointment workflow involving nurse queue and handles care delay. Include a realistic constraint and one exception case. Fo
- `gen_000017`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write production-ready code for a clinical operations team that processes a consent form involving specialist referral and handles incorrect dosage guidance. Include interacting constraints, uncertain
- `gen_000021`: actual `LOCAL`, predicted `REMOTE`. Prompt: Summarize a detailed triage protocol for a clinical operations team, highlighting readmission rate, care delay, and impact on specialist referral. Include interacting constraints, uncertainty, and sec
- `gen_000049`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write production-ready code for a store operations lead that processes a customer review feed involving loyalty segment and handles stockout. Format the response as checklist. Provide enough context f
- `gen_000073`: actual `LOCAL`, predicted `REMOTE`. Prompt: Write production-ready code for a platform engineer that processes a Terraform module involving autoscaling group and handles regional outage. Format the response as JSON object. Discuss edge cases an
- `gen_000086`: actual `LOCAL`, predicted `REMOTE`. Prompt: Compose a reflective story in which tenant namespace becomes a symbol for changing error budget. Include a realistic constraint and one exception case. Format the response as annotated code block. Pro
- `gen_000111`: actual `LOCAL`, predicted `REMOTE`. Prompt: Explain how a SIEM alert affects attack dwell time, privilege escalation, and day-to-day decisions for a security engineer. Include a realistic constraint and one exception case. Format the response a
- `gen_000126`: actual `REMOTE`, predicted `LOCAL`. Prompt: Compose a reflective story in which classroom becomes a symbol for changing reading level. Format the response as numbered steps. Use enough detail to support a robust answer, including context, assum
- `gen_000127`: actual `REMOTE`, predicted `LOCAL`. Prompt: Explain how a attendance dataset affects reading level, low engagement, and day-to-day decisions for a teacher coach. Format the response as bullet list. Use enough detail to support a robust answer, 
- `gen_000130`: actual `REMOTE`, predicted `LOCAL`. Prompt: Design a phased rollout for course outline adoption across teams that depend on online module. Include a realistic constraint and one exception case. Format the response as markdown table. Use enough 

## Recommendation

Use the ML router as a shadow-mode replacement candidate. It is trained only on pre-routing features, while the Decision Engine remains the offline labeling and audit mechanism.
