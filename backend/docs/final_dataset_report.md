# Final Dataset Report

## Overall statistics

- Total prompts: 14,967
- LOCAL: 7,148 (47.8%)
- REMOTE: 7,819 (52.2%)
- Average prompt length: 191.1 words
- Median prompt length: 34 words
- Prompt length range: 9–1988 words
- Exact duplicates removed: 32
- Near duplicates removed: 1
- Invalid prompts removed: 0

## Length distribution

| Length | Count | Percentage |
|---|---:|---:|
| under 100 words | 11,375 | 76.0% |
| 100-249 words | 898 | 6.0% |
| 250-499 words | 898 | 6.0% |
| 500-999 words | 898 | 6.0% |
| 1000-2000 words | 898 | 6.0% |
| over 2000 words | 0 | 0.0% |

## Category distribution and routing ratios

| Category | Count | LOCAL | REMOTE | REMOTE ratio |
|---|---:|---:|---:|---:|
| coding | 2,045 | 712 | 1,333 | 65.2% |
| creative_writing | 2,058 | 1,472 | 586 | 28.5% |
| general | 1,659 | 908 | 751 | 45.3% |
| planning | 2,193 | 744 | 1,449 | 66.1% |
| reasoning | 2,883 | 870 | 2,013 | 69.8% |
| summarization | 2,185 | 1,250 | 935 | 42.8% |
| translation | 1,944 | 1,192 | 752 | 38.7% |

## Split validation

- Train / validation / test: 11,973 / 1,497 / 1,497
- Exact duplicates remaining: 0
- Conflicting labels: 0
- Train/validation, train/test, validation/test overlaps: [0, 0, 0]
- Invalid labels: 0
- Empty prompts: 0
- Metadata length mismatches: 0
- Validation result: **PASS**

## Augmented prompt examples

### 500-1000 words — reasoning / REMOTE

> Analyze whether a patient safety committee should prioritize claims feed improvements or direct mitigation of privacy exposure. Format the response as plain paragraphs. Discuss edge cases and failure modes. Context: I am preparing material about healthcare for a real project. The audience includes people with different levels of experience, so the response needs to be precise without assuming unstated background. Address analyze through the lens of underlying mechanism and assumptions. Connect it to whether, distinguish established facts from assumptions, give a concrete example, and explain what evidence or constraints would lead to a different recommendation. Address whether through the le…

### 100-250 words — planning / REMOTE

> Design a phased rollout for appointment workflow adoption across teams that depend on nurse queue. Include a realistic constraint and one exception case. Format the response as markdown table. Address security and privacy implications. Optimize for low-latency operational use. Maintain a professional, neutral tone. Context: I am preparing material about healthcare for a real project. The audience includes people with different levels of experience, so the response needs to be precise without assuming unstated background. Address design through the lens of underlying mechanism and assumptions. Connect it to phased, distinguish established facts from assumptions, give a concrete example, and e…

### 250-500 words — general / REMOTE

> Explain how a clinical note affects patient satisfaction, privacy exposure, and day-to-day decisions for a patient safety committee. Include a realistic constraint and one exception case. Format the response as JSON object. Provide enough context for a precise answer without requiring external documents. Discuss edge cases and failure modes. Avoid unexplained jargon. Context: I am preparing material about healthcare for a real project. The audience includes people with different levels of experience, so the response needs to be precise without assuming unstated background. Address clinical through the lens of underlying mechanism and assumptions. Connect it to affects, distinguish establishe…

### 1000-2000 words — translation / REMOTE

> Translate this operational notice about care delay into French for stakeholders using appointment workflow. Include interacting constraints, uncertainty, and second-order effects. Format the response as checklist. Provide enough context for a precise answer without requiring external documents. Compare risks, trade-offs, and mitigations. Return the core result using the requested schema. Mark any facts that would need external verification. Use accessible language for a mixed-expertise audience. Context: I am preparing material about healthcare for a real project. The audience includes people with different levels of experience, so the response needs to be precise without assuming unstated b…

## Recommendation

The dataset passes structural validation and now includes production-scale prompt lengths. Use the fixed train, validation, and locked test files for retraining. Keep the test split untouched during model selection and threshold calibration.
