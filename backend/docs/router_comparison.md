# Router Performance and Benchmarking Comparison

This report evaluates and compares the four prompt routing approaches: Heuristic Router, Traditional ML Router, Semantic Embedding Router, and Hybrid Router.

## Performance Metrics Comparison

| Router | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Heuristic | 0.2950 | 0.0000 | 0.0000 | 0.0000 | 0.4964 |
| Traditional ML | 0.8375 | 0.1818 | 0.5217 | 0.2697 | 0.8697 |
| Embedding Router | 0.9425 | 0.0000 | 0.0000 | 0.0000 | 0.8103 |
| Hybrid Router | 0.9450 | 1.0000 | 0.0435 | 0.0833 | 0.8833 |

## Runtime and Resource Benchmarks

| Router | Feature/Emb Extraction (ms) | Classifier Inference (ms) | End-to-End Latency (ms) | Model Loading (s) | Memory Footprint (MB) | File Size on Disk (MB) |
|---|---:|---:|---:|---:|---:|---:|
| Heuristic | 2.043 | 0.024 | 0.025 | 0.0000 | 0.1 | 0.00 |
| Traditional ML | 2.043 | 17.230 | 17.232 | 0.1310 | 5.0 | 0.11 |
| Embedding Router | 253.510 | 207.492 | 461.005 | 0.0423 | 0.0 | 120.00 |
| Hybrid Router | 202.152 | 210.580 | 410.692 | 0.1030 | 5.0 | 120.19 |

## Confusion Matrices

### Heuristic

- `[Actual LOCAL, Actual REMOTE]`
- Predicted LOCAL: `[118, 23]`
- Predicted REMOTE: `[259, 0]`

### Traditional ML

- `[Actual LOCAL, Actual REMOTE]`
- Predicted LOCAL: `[323, 11]`
- Predicted REMOTE: `[54, 12]`

### Embedding Router

- `[Actual LOCAL, Actual REMOTE]`
- Predicted LOCAL: `[377, 23]`
- Predicted REMOTE: `[0, 0]`

### Hybrid Router

- `[Actual LOCAL, Actual REMOTE]`
- Predicted LOCAL: `[377, 22]`
- Predicted REMOTE: `[0, 1]`
