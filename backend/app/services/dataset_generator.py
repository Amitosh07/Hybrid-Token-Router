"""Utilities for building raw routing datasets from local and remote LLM runs.

This module does not make routing decisions. For each prompt, it always calls
both configured providers, records latency and approximate token counts, and
returns a JSON-serializable dataset entry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.services import ollama, remote_llm


ProviderGenerate = Callable[[str], Awaitable[str]]


@dataclass(frozen=True)
class PromptMetadata:
    """Descriptive metadata stored with each prompt result."""

    prompt_length: int
    word_count: int
    category: str | None = None
    difficulty: str | None = None


@dataclass(frozen=True)
class ProviderRun:
    """Serializable result for one provider attempt."""

    model: str
    response: str | None
    latency_ms: int
    estimated_input_tokens: int
    estimated_output_tokens: int
    error: str | None = None


def estimate_tokens(text: str) -> int:
    """Estimate token count with a replaceable character-length heuristic."""

    return math.ceil(len(text) / 4)


def load_prompts(path: str) -> list[str]:
    """Load a JSON array of prompt strings from disk."""

    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError("Prompt file must contain a JSON array of strings.")

    return data


def _metadata_for_prompt(prompt: str) -> dict[str, Any]:
    """Build the metadata block for a dataset entry."""

    return asdict(
        PromptMetadata(
            prompt_length=len(prompt),
            word_count=len(prompt.split()),
        )
    )


def _provider_result_dict(result: ProviderRun) -> dict[str, Any]:
    """Convert provider result dataclass into the stored JSON shape."""

    data = asdict(result)
    if result.error is None:
        data.pop("error")
    if result.response is None:
        data.pop("response")
    return data


async def _run_provider(
    *,
    prompt: str,
    model: str,
    generate: ProviderGenerate,
) -> dict[str, Any]:
    """Call one provider and capture success or failure without raising."""

    start_time = time.perf_counter()
    input_tokens = estimate_tokens(prompt)

    try:
        response = await generate(prompt)
        output_tokens = estimate_tokens(response)
        error = None
    except Exception as exc:  # noqa: BLE001 - dataset generation must continue.
        response = None
        output_tokens = 0
        error = str(exc)

    latency_ms = round((time.perf_counter() - start_time) * 1000)
    result = ProviderRun(
        model=model or "unknown",
        response=response,
        latency_ms=latency_ms,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        error=error,
    )
    return _provider_result_dict(result)


async def run_prompt(prompt: str) -> dict[str, Any]:
    """Run one prompt against both providers and return a raw dataset entry."""

    settings = get_settings()
    local_task = _run_provider(
        prompt=prompt,
        model=settings.OLLAMA_MODEL,
        generate=ollama.generate,
    )
    remote_task = _run_provider(
        prompt=prompt,
        model=settings.REMOTE_MODEL,
        generate=remote_llm.generate,
    )
    local_result, remote_result = await asyncio.gather(local_task, remote_task)

    return {
        "prompt": prompt,
        "metadata": _metadata_for_prompt(prompt),
        "local": local_result,
        "remote": remote_result,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


async def generate_dataset(prompts: Sequence[str]) -> list[dict[str, Any]]:
    """Run all prompts and return raw local/remote generations for each one."""

    dataset: list[dict[str, Any]] = []
    for prompt in prompts:
        dataset.append(await run_prompt(prompt))
    return dataset


def save_dataset(dataset: Sequence[dict[str, Any]], output_path: str) -> None:
    """Write a dataset to disk as pretty-printed JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as file:
        json.dump(list(dataset), file, indent=2, ensure_ascii=False)
        file.write("\n")


async def _main() -> None:
    """CLI entry point for generating a dataset from a prompt file."""

    parser = argparse.ArgumentParser(description="Generate a raw routing dataset.")
    parser.add_argument("prompts_path", help="Path to a JSON array of prompt strings.")
    parser.add_argument("output_path", help="Where to write the generated JSON dataset.")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_path)
    dataset = await generate_dataset(prompts)
    save_dataset(dataset, args.output_path)


if __name__ == "__main__":
    asyncio.run(_main())
