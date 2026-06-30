import httpx

from app.config import get_settings


async def generate(prompt: str) -> str:
    settings = get_settings()

    if not settings.REMOTE_API_KEY:
        raise ValueError("REMOTE_API_KEY is not configured.")

    if not settings.REMOTE_BASE_URL:
        raise ValueError("REMOTE_BASE_URL is not configured.")

    if not settings.REMOTE_MODEL:
        raise ValueError("REMOTE_MODEL is not configured.")

    url = f"{settings.REMOTE_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.REMOTE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.REMOTE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise ConnectionError("Could not connect to Remote LLM provider.") from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError("Remote LLM request timed out.") from exc
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if status_code == 429:
            raise RuntimeError("Remote LLM rate limit exceeded.") from exc
        raise RuntimeError(f"Remote LLM returned HTTP {status_code}.") from exc
    except httpx.HTTPError as exc:
        raise RuntimeError("Remote LLM request failed.") from exc

    try:
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Remote LLM returned an invalid response.") from exc

    if not isinstance(answer, str):
        raise RuntimeError("Remote LLM response did not include generated text.")

    return answer.strip()
