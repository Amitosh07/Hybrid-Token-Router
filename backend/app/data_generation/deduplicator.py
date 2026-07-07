"""Duplicate and near-duplicate detection for generated prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha1


@dataclass(frozen=True)
class DuplicateResult:
    """Deduplication summary."""

    kept: list[dict[str, object]]
    duplicate_count: int
    near_duplicate_count: int
    removed_ids: list[str]


def normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for exact duplicate detection."""
    text = prompt.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(prompt: str) -> str:
    """Stable hash for normalized prompt text."""
    return sha1(normalize_prompt(prompt).encode("utf-8")).hexdigest()


def shingles(prompt: str, size: int = 5) -> set[tuple[str, ...]]:
    """Return word shingles for near-duplicate detection."""
    words = normalize_prompt(prompt).split()
    if len(words) < size:
        return {tuple(words)}
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


def jaccard(left: set[tuple[str, ...]], right: set[tuple[str, ...]]) -> float:
    """Jaccard similarity for shingle sets."""
    if not left and not right:
        return 1.0
    return len(left & right) / max(len(left | right), 1)


class Deduplicator:
    """Exact and shingle-based near-duplicate detector."""

    def __init__(self, threshold: float = 0.82) -> None:
        self.threshold = threshold

    def deduplicate(self, rows: list[dict[str, object]]) -> DuplicateResult:
        """Remove exact duplicates and near duplicates."""
        seen_hashes: set[str] = set()
        seen_shingles: list[tuple[str, set[tuple[str, ...]]]] = []
        kept: list[dict[str, object]] = []
        duplicate_count = 0
        near_duplicate_count = 0
        removed_ids: list[str] = []

        for row in rows:
            prompt = str(row["prompt"])
            row_hash = prompt_hash(prompt)
            prompt_id = str(row.get("prompt_id", "unknown"))
            if row_hash in seen_hashes:
                duplicate_count += 1
                removed_ids.append(prompt_id)
                continue

            row_shingles = shingles(prompt)
            near_duplicate = False
            for _, previous in seen_shingles[-1500:]:
                if jaccard(row_shingles, previous) >= self.threshold:
                    near_duplicate = True
                    break
            if near_duplicate:
                near_duplicate_count += 1
                removed_ids.append(prompt_id)
                continue

            seen_hashes.add(row_hash)
            seen_shingles.append((prompt_id, row_shingles))
            kept.append(row)

        return DuplicateResult(
            kept=kept,
            duplicate_count=duplicate_count,
            near_duplicate_count=near_duplicate_count,
            removed_ids=removed_ids,
        )

