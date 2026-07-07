"""Dataset statistics and Markdown report generation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


def summarize(rows: list[dict[str, object]]) -> dict[str, Any]:
    """Summarize generated prompt metadata."""
    if not rows:
        return {
            "total": 0,
            "category_counts": {},
            "difficulty_counts": {},
            "domain_counts": {},
            "average_words": 0.0,
            "average_constraints": 0.0,
            "average_estimated_complexity": 0.0,
        }
    word_counts = [len(str(row["prompt"]).split()) for row in rows]
    return {
        "total": len(rows),
        "category_counts": dict(Counter(str(row["category"]) for row in rows)),
        "difficulty_counts": dict(Counter(str(row["difficulty"]) for row in rows)),
        "domain_counts": dict(Counter(str(row["domain"]) for row in rows)),
        "expected_reasoning_counts": dict(Counter(str(row["expected_reasoning"]) for row in rows)),
        "output_format_counts": dict(Counter(str(row.get("output_format", "")) for row in rows)),
        "average_words": round(mean(word_counts), 2),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
        "average_constraints": round(mean(int(row["constraint_count"]) for row in rows), 2),
        "average_estimated_complexity": round(mean(float(row["estimated_complexity"]) for row in rows), 4),
    }


def write_generation_statistics(rows: list[dict[str, object]], path: Path) -> None:
    """Write generation statistics report."""
    stats = summarize(rows)
    lines = [
        "# Generation Statistics",
        "",
        f"- Total prompts: {stats['total']}",
        f"- Average prompt words: {stats['average_words']}",
        f"- Prompt word range: {stats['min_words']} to {stats['max_words']}" if stats["total"] else "- Prompt word range: n/a",
        f"- Average constraints: {stats['average_constraints']}",
        f"- Average estimated complexity: {stats['average_estimated_complexity']}",
        "",
        "## Category Distribution",
        "",
        _count_table(stats["category_counts"]),
        "",
        "## Difficulty Distribution",
        "",
        _count_table(stats["difficulty_counts"]),
        "",
        "## Expected Reasoning Distribution",
        "",
        _count_table(stats.get("expected_reasoning_counts", {})),
    ]
    _write(path, lines)


def write_coverage_report(rows: list[dict[str, object]], path: Path) -> None:
    """Write coverage report across domains, formats, and constraints."""
    stats = summarize(rows)
    constraint_counts = Counter()
    category_domain = Counter()
    for row in rows:
        for name in row.get("constraint_names", ()):
            constraint_counts[str(name)] += 1
        category_domain[(str(row["category"]), str(row["domain"]))] += 1

    lines = [
        "# Coverage Report",
        "",
        "## Domain Distribution",
        "",
        _count_table(stats["domain_counts"]),
        "",
        "## Output Format Distribution",
        "",
        _count_table(stats.get("output_format_counts", {})),
        "",
        "## Constraint Coverage",
        "",
        _count_table(dict(constraint_counts)),
        "",
        "## Category x Domain Coverage",
        "",
        "| Category | Domain | Count |",
        "|---|---|---:|",
    ]
    for (category, domain), count in sorted(category_domain.items()):
        lines.append(f"| {category} | {domain} | {count} |")
    _write(path, lines)


def write_quality_report(
    *,
    path: Path,
    validation: Any,
    duplicate_count: int,
    near_duplicate_count: int,
    requested_size: int,
    final_size: int,
) -> None:
    """Write quality-control report."""
    lines = [
        "# Dataset Quality Report",
        "",
        f"- Requested size: {requested_size}",
        f"- Final valid unique prompts: {final_size}",
        f"- Invalid prompts removed: {len(validation.invalid_rows)}",
        f"- Exact duplicates removed: {duplicate_count}",
        f"- Near duplicates removed: {near_duplicate_count}",
        "",
        "## Quality Checks",
        "",
        "- Duplicate detection: exact normalized text hash",
        "- Near-duplicate detection: 5-word shingle Jaccard similarity",
        "- Short prompt detection: minimum word threshold",
        "- Metadata validation: category, difficulty, domain, reasoning, complexity",
        "- Balance checks: category, difficulty, and domain distribution",
        "",
        "## Issues",
        "",
    ]
    if validation.issues:
        lines.extend(f"- {issue}" for issue in validation.issues[:100])
        if len(validation.issues) > 100:
            lines.append(f"- ... {len(validation.issues) - 100} additional issues omitted")
    else:
        lines.append("- No blocking quality issues detected.")
    _write(path, lines)


def _count_table(counts: dict[str, int]) -> str:
    if not counts:
        return "- No data."
    lines = ["| Bucket | Count |", "|---|---:|"]
    for key, value in sorted(counts.items()):
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

