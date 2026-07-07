# Unseen Prompt Evaluation

- Unseen prompts evaluated: 104
- Ground truth: Phase 2 Decision Engine applied to benchmark-compatible simulated local/remote runs.
- Prompt overlap check: no exact prompts from `training_dataset.csv` were reused.

## Overall Metrics

| System | Accuracy | Precision | Recall | F1 | Agreement with Decision Engine |
|---|---:|---:|---:|---:|---:|
| ML Router | 0.3365 | 0.3200 | 0.9697 | 0.4812 | 0.3365 |
| Heuristic Router | 0.7596 | 0.9000 | 0.2727 | 0.4186 | 0.7596 |

## Confusion Matrix

Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.

- ML Router: `[[3, 68], [1, 32]]`
- Heuristic Router: `[[70, 1], [24, 9]]`

## Per-Category Accuracy

| Category | ML Accuracy | Heuristic Accuracy | Count |
|---|---:|---:|---:|
| coding | 0.3077 | 0.8462 | 13 |
| creative_writing | 0.3846 | 0.6923 | 13 |
| general | 0.3077 | 0.6923 | 13 |
| mathematics | 0.4615 | 0.7692 | 13 |
| planning | 0.3077 | 0.7692 | 13 |
| reasoning | 0.3846 | 0.7692 | 13 |
| summarization | 0.3077 | 0.7692 | 13 |
| translation | 0.2308 | 0.7692 | 13 |

## Per-Difficulty Accuracy

| Difficulty | ML Accuracy | Heuristic Accuracy | Count |
|---|---:|---:|---:|
| easy | 0.0968 | 1.0000 | 31 |
| hard | 0.9091 | 0.2727 | 33 |
| medium | 0.0500 | 0.9750 | 40 |

## Top 20 Misclassified Prompts

- `unseen_mathematics_005` (mathematics, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9838. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Find the percentage change when revenue increases from 80,000 to 92,000 dollars.
- `unseen_mathematics_001` (mathematics, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9785. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Calculate the monthly payment for a 12-month loan of 1200 dollars at zero interest.
- `unseen_mathematics_010` (mathematics, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9785. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Calculate a 95 percent confidence interval for a small customer satisfaction sample.
- `unseen_planning_012` (planning, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9753. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Build a customer education webinar plan for an accounting automation feature.
- `unseen_coding_001` (coding, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9706. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Build a Ruby method that normalizes phone numbers into E.164 format and returns nil for invalid inputs.
- `unseen_reasoning_001` (reasoning, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9686. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: A school changes class start times by 20 minutes. Explain the likely effects on bus scheduling and attendance.
- `unseen_general_005` (general, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9684. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Explain the difference between a debit card and a credit card.
- `unseen_general_004` (general, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9684. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Describe how hospitals use triage systems during emergency department crowding.
- `unseen_general_010` (general, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9684. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Describe how cyber insurance underwriting evaluates company security posture.
- `unseen_mathematics_013` (mathematics, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9678. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Compute the mean, median, and range for a list of weekly ticket counts.
- `unseen_general_001` (general, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9665. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Explain what a password manager does for a non-technical audience.
- `unseen_general_007` (general, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9644. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Describe common causes of machine-learning data drift in retail demand forecasting.
- `unseen_summarization_005` (summarization, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9644. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Summarize a parent-teacher meeting note in plain language.
- `unseen_mathematics_002` (mathematics, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9639. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Given a confusion matrix, compute precision, recall, and F1 for the positive class.
- `unseen_mathematics_004` (mathematics, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9614. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Solve a linear system representing inventory balances across three warehouses.
- `unseen_coding_004` (coding, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9562. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Implement a Python function that batches database writes with exponential backoff and preserves input order.
- `unseen_planning_007` (planning, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9474. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Create a hiring plan for a small machine-learning platform team over two quarters.
- `unseen_creative_writing_002` (creative_writing, medium): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9474. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Draft a short story scene about a chef discovering a recipe that predicts memories.
- `unseen_reasoning_011` (reasoning, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9405. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Explain why adding more checkout lanes does not always reduce wait times in a grocery store.
- `unseen_reasoning_006` (reasoning, easy): Decision Engine `LOCAL`, ML `REMOTE` at confidence 0.9405. Why: pre-routing features such as length, technical content, or constraints looked remote-worthy, but Decision Engine utility favored local after simulated quality/latency trade-offs. Prompt: Explain why a small online shop might prefer simpler inventory software over a full ERP system.
