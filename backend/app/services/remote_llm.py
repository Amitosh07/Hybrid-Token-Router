"""Remote LLM client with persistent connection pooling.

Client strategy:
  - A single persistent AsyncClient is created at module level and reused
    across all requests. This avoids the per-request TCP + TLS handshake
    overhead (~50-300 ms) that the previous `async with AsyncClient()` pattern
    incurred on every call.
  - Headers (including Authorization) are set at client creation time so
    they don't need to be reconstructed per request.
  - Timeout: 30 s — sufficient for any cloud LLM under normal conditions.
"""

from __future__ import annotations

import asyncio
import random
import httpx

from app.config import get_settings

# ---------------------------------------------------------------------------
# Persistent client — created on first use, reused for all requests.
# ---------------------------------------------------------------------------
_TIMEOUT_SECONDS: float = 30.0

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the persistent AsyncClient, creating it when first needed."""
    global _client
    if _client is None or _client.is_closed:
        settings = get_settings()
        _client = httpx.AsyncClient(
            base_url=settings.REMOTE_BASE_URL.rstrip("/") if settings.REMOTE_BASE_URL else "",
            headers={
                "Authorization": f"Bearer {settings.REMOTE_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT_SECONDS,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )
    return _client


async def generate(prompt: str) -> str:
    import os
    if os.getenv("SIMULATE_LLM", "false").lower() == "true":
        from app.services.simulator import generate_simulated
        return await generate_simulated(prompt, "remote")

    # Proactive rate limiting: sleep 2.1s to respect Groq's 30 RPM limit without hitting 429s
    await asyncio.sleep(2.1)

    settings = get_settings()

    if not settings.REMOTE_API_KEY:
        raise ValueError("REMOTE_API_KEY is not configured.")

    if not settings.REMOTE_BASE_URL:
        raise ValueError("REMOTE_BASE_URL is not configured.")

    if not settings.REMOTE_MODEL:
        raise ValueError("REMOTE_MODEL is not configured.")

    payload = {
        "model": settings.REMOTE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    client = _get_client()
    
    max_retries = 5
    backoff = 2.0
    
    for attempt in range(max_retries):
        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            
            # Request succeeded, parse answer
            try:
                data = response.json()
                answer = data["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("Remote LLM returned an invalid response.") from exc

            if not isinstance(answer, str):
                raise RuntimeError("Remote LLM response did not include generated text.")

            return answer.strip()

        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 429 and attempt < max_retries - 1:
                # 429 Rate Limit - Wait and retry
                wait_time = backoff * (2 ** attempt) + random.uniform(0.1, 0.5)
                print(f"\n  [!] Groq 429 Rate Limit. Retrying in {wait_time:.2f}s (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(wait_time)
                continue
            
            if status_code == 429:
                raise RuntimeError("Remote LLM rate limit exceeded.") from exc
            raise RuntimeError(f"Remote LLM returned HTTP {status_code}.") from exc
            
        except httpx.ConnectError as exc:
            raise ConnectionError("Could not connect to Remote LLM provider.") from exc
        except httpx.TimeoutException as exc:
            raise TimeoutError("Remote LLM request timed out.") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError("Remote LLM request failed.") from exc

    raise RuntimeError("Remote LLM request failed after maximum retries.")
