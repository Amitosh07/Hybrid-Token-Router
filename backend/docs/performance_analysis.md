# Performance Analysis — Hybrid Token Router

## Overview

This document records the complete performance investigation, bottlenecks identified, optimisations applied, and the decision on cost estimation.

---

## 1. Where Time Is Spent — Pipeline Stage Breakdown

Every request passes through the following stages. Stages are now logged at `DEBUG` level with `[prompt_id]` prefixes.

| Stage | Code location | Typical duration |
|---|---|---|
| T0 → T1 Feature extraction | `feature_extractor.py` | < 1 ms |
| T1 → T2 Routing decision | `router.py` | < 1 ms |
| T2 → T3 Provider call setup | `ollama.py` / `remote_llm.py` | 1–200 ms ¹ |
| T3 → T4 Provider inference | Network + model | 500 ms – 20 s |
| T4 → T6 Serialisation + return | FastAPI | < 1 ms |

¹ **Connection setup time** — this was the identified bottleneck (see §2).

### Reading the debug logs

```
[<id>] Pipeline stage timings | feature_extraction=0.4ms | routing_decision=0.1ms | provider_selected=local
[<id>] Pipeline stage timings | provider_wait=1423.5ms | total_request=1424.1ms | actual_provider=local | fallback=False
[<id>] Provider=local | Confidence=0.73 | ProviderMs=1423.5ms | TotalMs=1424.1ms | Fallback=False
```

To enable debug logs, set log level to `DEBUG` in your environment or logging config.

---

## 2. Bottleneck Identified

### Primary bottleneck: Per-request TCP connection setup

**Before optimisation**, both `ollama.py` and `remote_llm.py` used the pattern:

```python
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.post(url, ...)
```

This creates a **new `AsyncClient` and a new TCP connection on every single request**. The impact:

- **Ollama (localhost):** ~10–50 ms per request for TCP socket setup
- **Remote LLM (HTTPS cloud):** ~50–300 ms per request for TCP handshake + TLS negotiation

On the second, third, and subsequent requests, this overhead is **entirely avoidable** — the connection could be kept alive and reused.

### Secondary bottleneck: Unnecessary retry on Ollama timeout

**Before optimisation**, `chat.py` retried a timed-out Ollama call once before falling back:

```
Ollama attempt 1 → 20 s timeout
Ollama attempt 2 → 20 s timeout
Remote LLM fallback → response
Total worst-case: ~50 s
```

**This retry was counter-productive.** A model that fails to respond in 20 s will almost certainly fail again in the next 20 s — the model is either pulling/loading or Ollama is genuinely unavailable. The retry simply doubled the worst-case latency for no benefit.

---

## 3. Optimisations Applied

### 3.1 Persistent HTTP client (connection pooling)

Both `ollama.py` and `remote_llm.py` now use a **module-level singleton `httpx.AsyncClient`**:

```python
_client: httpx.AsyncClient | None = None

def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _client
```

The client is created once and reused across all requests. HTTP keep-alive keeps the underlying TCP connection open between calls.

**Expected improvement:**
- Ollama (localhost): eliminates ~10–50 ms setup per request after the first
- Remote LLM: eliminates ~50–300 ms TCP+TLS handshake per request after the first

### 3.2 Removed Ollama retry

`chat.py` now falls back to remote immediately on the first `TimeoutError` from Ollama:

```python
try:
    answer = await ollama.generate(request.prompt)
except TimeoutError:
    # Single timeout → fall back immediately.
    answer = await remote_llm.generate(request.prompt)
    provider = "remote"
    fallback_used = True
```

**Impact:** Worst-case latency when Ollama is unavailable reduced from ~50 s to ~30 s.

### 3.3 Full pipeline instrumentation

`chat.py` now records and logs stage-level timing at `DEBUG`:

```
feature_extraction_ms   # T1 - T0
routing_decision_ms     # T2 - T1
provider_ms             # T4 - T3
total_ms                # T4 - T0  ← authoritative latency stored in stats
```

`total_ms` is the value stored in `stats_tracker` and displayed in the dashboard. It reflects real end-to-end request time — not an estimate, not a proxy metric.

---

## 4. Final Latency Breakdown (expected post-optimisation)

### Local model — steady state (model already loaded)

| Stage | Expected |
|---|---|
| Feature extraction | < 1 ms |
| Routing decision | < 1 ms |
| Ollama inference | 200 ms – 5 s (model-dependent) |
| Connection setup | ~0 ms (reused) |
| **Total** | **200 ms – 5 s** |

### Remote model — steady state

| Stage | Expected |
|---|---|
| Feature extraction | < 1 ms |
| Routing decision | < 1 ms |
| Remote LLM inference | 500 ms – 5 s (provider-dependent) |
| Connection setup | ~0 ms (reused after first request) |
| **Total** | **500 ms – 5 s** |

### Local → Remote fallback (Ollama unavailable)

| Stage | Expected |
|---|---|
| Ollama attempt (timeout) | 20 s |
| Remote LLM inference | 500 ms – 5 s |
| **Total** | **~20–25 s** |

---

## 5. Average Latency Calculation

- Computed in `stats_tracker.py` as `sum(latency_ms) / count` over all completed requests.
- Only **successful** requests are recorded — failed requests (HTTP 5xx) are not included.
- Uses `total_ms` (T0→T4, measured on the backend) as the authoritative per-request value.
- The dashboard reads this via `GET /stats → average_latency_ms`.
- Dashboard updates automatically after every successful prompt via the `silentRefresh` mechanism (no polling, no page reload).

---

## 6. Cost Estimation — Decision and Rationale

### Decision: Removed entirely

The "Estimated Cost Saved" widget has been **removed** from both the chat analytics sidebar and the dashboard.

### Why

The previous implementation calculated cost as:

```
cost_saved = (estimated_input_tokens / 1000) × 0.0006
```

This formula has two critical flaws:

1. **Output tokens are not counted.** Neither Ollama's `/api/generate` (in non-streaming mode, `eval_count` was not read) nor the remote LLM's response (`usage.completion_tokens` was not read) provided output token counts to this function. For most prompts, output tokens represent 50–90% of the billable token count.

2. **The price constant was invented.** `$0.0006 / 1k tokens` was a rough estimate with no basis in the actual configured remote provider's pricing.

A formula that omits 50–90% of the relevant data, multiplied by an invented price, does not produce a useful metric — it produces a number that grows by a small fixed increment on every local request, which is what the user observed as incorrect behaviour.

### What would be required to implement this correctly

To display a mathematically correct cost saving, the following would be needed:

1. **Input token count** — already available via `features["estimated_input_tokens"]` (rough estimate) or by reading `usage.prompt_tokens` from the remote LLM response.
2. **Output token count** — must be read from `usage.completion_tokens` in the remote LLM response, or from `eval_count` in Ollama's response body.
3. **Accurate pricing** — must be read from configuration or a pricing table for the specific `REMOTE_MODEL` in use. A hardcoded constant is not acceptable.
4. **Local inference cost** — if the user wants to account for electricity/GPU cost, this must be configured explicitly; it cannot be assumed to be zero without documentation.

When these inputs are available, the correct formula is:

```
remote_cost = (input_tokens × input_price_per_token)
            + (output_tokens × output_price_per_token)

local_cost  = (inferred from configured electricity/GPU cost)
            = 0 by assumption if not configured

cost_saved  = remote_cost - local_cost
```

Until output token counts and accurate pricing are available in the current architecture, this metric is deliberately absent.
