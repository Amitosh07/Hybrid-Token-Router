"""Ollama local LLM client with pipeline instrumentation.

Client strategy:
  - A single persistent AsyncClient is created at module level and reused
    across all requests. This avoids the per-request TCP connection setup
    (~20-200 ms overhead) that exists when using `async with AsyncClient()`.
  - Timeout is 20 s per attempt. One attempt only — if Ollama is not
    responding within 20 s it will not recover on a retry; the caller
    should fall back to remote immediately.
  - The caller (chat.py) handles the fallback; this module only raises
    TimeoutError when the model is genuinely unavailable.
"""

from __future__ import annotations

import httpx

from app.config import get_settings

# ---------------------------------------------------------------------------
# Persistent client — created once, reused for all requests.
# Connection pooling eliminates per-request TCP setup (~20-200 ms saved).
# ---------------------------------------------------------------------------
_TIMEOUT_SECONDS: float = 20.0

# Lazy singleton — initialised on first use so settings are available.
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the module-level persistent AsyncClient, creating it if needed."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS,
            # Keep connections alive between requests.
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _client


async def generate(prompt: str) -> str:
    import os
    if os.getenv("SIMULATE_LLM", "false").lower() == "true":
        from app.services.simulator import generate_simulated
        return await generate_simulated(prompt, "local")

    settings = get_settings()

    if not settings.OLLAMA_BASE_URL:
        raise ValueError("OLLAMA_BASE_URL is not configured.")

    if not settings.OLLAMA_MODEL:
        raise ValueError("OLLAMA_MODEL is not configured.")

    url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 2048,
            "temperature": 0.0,
            "num_ctx": 4096
        }
    }

    client = _get_client()
    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
    except httpx.ConnectError as exc:
        raise ConnectionError("Could not connect to Ollama.") from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError("Ollama request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.response.status_code}.") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("Ollama request failed.") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    answer = data.get("response")
    if not isinstance(answer, str):
        raise RuntimeError("Ollama response did not include generated text.")

    return answer.strip()
