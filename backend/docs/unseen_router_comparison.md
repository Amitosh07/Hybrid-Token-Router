# Unseen Prompt Routing Evaluation and Policy Agreement

This evaluation executes all four routers on a completely new, hand-crafted set of 104 unseen prompts, using the offline Decision Engine as the policy ground truth.

## Policy Accuracy vs Decision Engine Ground Truth

| Router | Accuracy | Precision | Recall | F1 Score | Agreement with Decision Engine |
|---|---:|---:|---:|---:|---:|
| Heuristic | 0.8269 | 0.9091 | 0.3704 | 0.5263 | 0.8269 |
| Traditional ML | 0.4423 | 0.2281 | 0.4815 | 0.3095 | 0.4423 |
| Embedding Router | 0.6923 | 0.1429 | 0.0370 | 0.0588 | 0.6923 |
| Hybrid Router | 0.7404 | 0.0000 | 0.0000 | 0.0000 | 0.7404 |

## Router Policy Agreement Matrix

Describes how frequently routers make identical routing decisions:

- Heuristic vs Traditional ML: 34.6154%
- Heuristic vs Embedding Router: 82.6923%
- Heuristic vs Hybrid Router: 89.4231%
- Traditional ML vs Embedding Router: 44.2308%
- Traditional ML vs Hybrid Router: 45.1923%
- Embedding Router vs Hybrid Router: 93.2692%

## Performance per Category

| Category | Heuristic Acc | ML Acc | Embedding Acc | Hybrid Acc | Count |
|---|---:|---:|---:|---:|---:|
| coding | 0.8462 | 0.2308 | 0.6154 | 0.6154 | 13 |
| creative_writing | 1.0000 | 0.6923 | 0.6154 | 1.0000 | 13 |
| general | 0.8462 | 0.7692 | 0.7692 | 0.8462 | 13 |
| mathematics | 0.9231 | 0.4615 | 0.6923 | 0.6923 | 13 |
| planning | 0.7692 | 0.3077 | 0.6923 | 0.6923 | 13 |
| reasoning | 0.6923 | 0.5385 | 0.5385 | 0.5385 | 13 |
| summarization | 0.7692 | 0.2308 | 0.8462 | 0.7692 | 13 |
| translation | 0.7692 | 0.3077 | 0.7692 | 0.7692 | 13 |

## Performance per Difficulty

| Difficulty | Heuristic Acc | ML Acc | Embedding Acc | Hybrid Acc | Count |
|---|---:|---:|---:|---:|---:|
| easy | 1.0000 | 0.3548 | 0.9355 | 1.0000 | 31 |
| hard | 0.5152 | 0.5152 | 0.1818 | 0.2424 | 33 |
| medium | 0.9500 | 0.4500 | 0.9250 | 0.9500 | 40 |

## Confidence Analysis

Average confidence score on correct vs incorrect predictions:

- **Traditional ML**: Correct Pred Confidence = 0.9795, Incorrect Pred Confidence = 0.9784
- **Embedding Router**: Correct Pred Confidence = 0.9801, Incorrect Pred Confidence = 0.9759
- **Hybrid Router**: Correct Pred Confidence = 0.9826, Incorrect Pred Confidence = 0.9858

## Representative Misclassified Examples

Shows a few prompts where the Embedding Router mismatched the Decision Engine policy:

- **unseen_coding_005** (coding, hard): Actual=REMOTE, Pred=LOCAL (conf=0.990). Prompt: "Design a lock-free thread-safe ring buffer in C++ using atomic memory operations and compare-and-swap."
- **unseen_coding_006** (coding, hard): Actual=REMOTE, Pred=LOCAL (conf=0.990). Prompt: "Implement a custom memory allocator in Rust that manages a pre-allocated byte arena and prevents fragmentation."
- **unseen_coding_009** (coding, hard): Actual=REMOTE, Pred=LOCAL (conf=0.967). Prompt: "Build a compiler frontend in Go that tokenizes and parses a tiny arithmetic language into an AST."
- **unseen_coding_010** (coding, medium): Actual=REMOTE, Pred=LOCAL (conf=0.985). Prompt: "Create a Python script that uses asyncio to scrape multiple web pages concurrently with a concurrency limit of 5."
- **unseen_coding_011** (coding, hard): Actual=REMOTE, Pred=LOCAL (conf=0.994). Prompt: "Implement a distributed lock manager in Java using a custom consensus protocol over sockets."
- **unseen_reasoning_002** (reasoning, medium): Actual=REMOTE, Pred=LOCAL (conf=0.991). Prompt: "Evaluate the ethical implications of a self-driving car algorithm prioritizing passengers over pedestrians."
- **unseen_reasoning_003** (reasoning, hard): Actual=REMOTE, Pred=LOCAL (conf=0.988). Prompt: "Analyze how a central bank's decision to implement negative interest rates affects commercial lending and consumer savings."
- **unseen_reasoning_005** (reasoning, hard): Actual=REMOTE, Pred=LOCAL (conf=0.993). Prompt: "Deconstruct the systemic risk of high-frequency trading algorithms on public equity market stability during liquidity crises."
- **unseen_reasoning_007** (reasoning, hard): Actual=REMOTE, Pred=LOCAL (conf=0.991). Prompt: "Evaluate the legal and economic trade-offs of treating social media platforms as common carriers rather than publishers."
- **unseen_reasoning_009** (reasoning, hard): Actual=REMOTE, Pred=LOCAL (conf=0.994). Prompt: "Reason about the potential impact of general-purpose quantum computing on current cryptographic standards."