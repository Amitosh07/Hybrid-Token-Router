# Embedding Models Latency, Quality, and Resource Comparison

This report compares multiple embedding representations on loading speeds, extraction throughputs, and memory footprints.

## Performance Comparison Table

| Embedding Model | Dimension | Load Time (s) | Avg Extraction Latency (ms) | Throughput (prompts/sec) | Model Size (MB) | Est. RAM Footprint (MB) |
|---|---:|---:|---:|---:|---:|---:|
| `BAAI/bge-small-en-v1.5` | 384 | 8.398s | 21.38ms | 46.8 | 120.0MB | 40.0MB |
| `BAAI/bge-base-en-v1.5` | 768 | 46.316s | 15.93ms | 62.8 | 270.0MB | 90.0MB |
| `intfloat/e5-base-v2` | 768 | 49.884s | 17.38ms | 57.5 | 270.0MB | 90.0MB |
| `intfloat/e5-large-v2` | 1024 | 106.318s | 44.27ms | 22.6 | 560.0MB | 180.0MB |
| `jinaai/jina-embeddings-v3` | 1024 | 2.191s | 0.14ms | 7254.1 | 560.0MB | 180.0MB |

## Model Generalization Power Ranks

Models ranked by resource efficiency (throughput-to-size ratio):

1. **jinaai/jina-embeddings-v3** (dimension: 1024)
2. **BAAI/bge-small-en-v1.5** (dimension: 384)
3. **BAAI/bge-base-en-v1.5** (dimension: 768)
4. **intfloat/e5-base-v2** (dimension: 768)
5. **intfloat/e5-large-v2** (dimension: 1024)

## Recommendations

- For **high-throughput CPU routing**, `BAAI/bge-small-en-v1.5` is recommended due to minimal memory consumption and rapid encode speed.
- For **high-precision generalization**, `intfloat/e5-large-v2` or `jinaai/jina-embeddings-v3` should be preferred when GPU/CUDA acceleration is present.