# Unseen Prompt Evaluation

- Unseen prompts evaluated: 104
- Ground truth: Phase 2 Decision Engine applied to benchmark-compatible simulated local/remote runs.
- Prompt overlap check: no exact prompts from `training_dataset.csv` were reused.

## Overall Metrics

| System | Accuracy | Precision | Recall | F1 | Agreement with Decision Engine |
|---|---:|---:|---:|---:|---:|
| ML Router | 0.6731 | 0.0000 | 0.0000 | 0.0000 | 0.6731 |
| Heuristic Router | 0.7596 | 0.9000 | 0.2727 | 0.4186 | 0.7596 |

## Confusion Matrix

Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.

- ML Router: `[[70, 1], [33, 0]]`
- Heuristic Router: `[[70, 1], [24, 9]]`

## Per-Category Accuracy

| Category | ML Accuracy | Heuristic Accuracy | Count |
|---|---:|---:|---:|
| coding | 0.6923 | 0.8462 | 13 |
| creative_writing | 0.6923 | 0.6923 | 13 |
| general | 0.6923 | 0.6923 | 13 |
| mathematics | 0.5385 | 0.7692 | 13 |
| planning | 0.6923 | 0.7692 | 13 |
| reasoning | 0.5385 | 0.7692 | 13 |
| summarization | 0.7692 | 0.7692 | 13 |
| translation | 0.7692 | 0.7692 | 13 |

## Per-Difficulty Accuracy

| Difficulty | ML Accuracy | Heuristic Accuracy | Count |
|---|---:|---:|---:|
| easy | 0.9677 | 1.0000 | 31 |
| hard | 0.0606 | 0.2727 | 33 |
| medium | 0.9500 | 0.9750 | 40 |

## Top 20 Misclassified Prompts

- `unseen_mathematics_012` (mathematics, medium): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9984. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Optimize the allocation of a limited advertising budget across two channels with linear returns.
- `unseen_reasoning_004` (reasoning, medium): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9984. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Evaluate the trade-offs of using synthetic data for financial credit-risk model development.
- `unseen_mathematics_006` (mathematics, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9940. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Prove that the sum of the first n odd numbers equals n squared using induction.
- `unseen_mathematics_008` (mathematics, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9909. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Analyze the convergence of a simple gradient descent update for a one-dimensional quadratic loss.
- `unseen_planning_006` (planning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9732. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Build a phased disaster recovery exercise for a fintech platform with regional failover requirements.
- `unseen_coding_011` (coding, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9650. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Implement a Java circuit breaker with half-open recovery, metrics hooks, and thread-safe state transitions.
- `unseen_reasoning_005` (reasoning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9591. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Reason through the second-order effects of moving a monolith to event-driven microservices in a regulated bank.
- `unseen_coding_005` (coding, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9523. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Design a Rust async rate limiter for multiple API tenants using token buckets and lock-free counters.
- `unseen_reasoning_003` (reasoning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.9506. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Analyze whether a hospital should prioritize a scarce diagnostic device for emergency cases or scheduled oncology care.
- `unseen_reasoning_007` (reasoning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.8755. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Assess the legal and ethical implications of automated hiring filters trained on historical promotion data.
- `unseen_general_006` (general, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7741. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Explain how consensus protocols help distributed databases tolerate node failures.
- `unseen_general_009` (general, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7735. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Explain the major privacy risks in linking education records with workforce outcome data.
- `unseen_general_003` (general, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7735. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Explain the relationship between central bank interest-rate policy and startup financing conditions.
- `unseen_reasoning_009` (reasoning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7679. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Analyze whether a research lab should publish a dual-use cybersecurity exploit with mitigations.
- `unseen_reasoning_012` (reasoning, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7679. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Evaluate the systemic risk of relying on a single identity provider across all internal enterprise tools.
- `unseen_translation_003` (translation, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7669. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Translate a healthcare consent paragraph from English to Hindi with formal medical terminology.
- `unseen_general_012` (general, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7665. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Explain why legal discovery systems need careful access controls and audit logging.
- `unseen_mathematics_011` (mathematics, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7622. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Use Bayes theorem to estimate fraud probability after two independent risk signals fire.
- `unseen_translation_009` (translation, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7618. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Translate a distributed systems research abstract from English to Mandarin Chinese with academic style.
- `unseen_coding_006` (coding, hard): Decision Engine `REMOTE`, ML `LOCAL` at confidence 0.7474. Why: pre-routing features looked simple or local-friendly, but Decision Engine favored remote after simulated response quality gains outweighed latency and cost. Prompt: Implement a Kubernetes operator reconciliation loop that rolls back failed deployments after health-check drift.
