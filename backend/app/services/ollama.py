import httpx

from app.config import get_settings


async def generate(prompt: str) -> str:
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
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
