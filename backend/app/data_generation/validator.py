"""Quality validation for generated routing prompts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from app.data_generation.domain_library import CATEGORIES, DIFFICULTIES, DOMAINS


@dataclass(frozen=True)
class ValidationResult:
    """Validation summary and row-level issues."""

    valid_rows: list[dict[str, object]]
    invalid_rows: list[dict[str, object]]
    issues: list[str]
    category_balance: dict[str, int]
    difficulty_balance: dict[str, int]
    domain_balance: dict[str, int]


REQUIRED_FIELDS = {
    "prompt_id",
    "prompt",
    "category",
    "difficulty",
    "expected_reasoning",
    "domain",
    "constraint_count",
    "estimated_complexity",
}


def validate_prompts(rows: list[dict[str, object]], min_words: int = 8) -> ValidationResult:
    """Validate prompt metadata and basic quality."""
    valid: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    issues: list[str] = []

    for row in rows:
        row_issues = _row_issues(row, min_words)
        if row_issues:
            item = dict(row)
            item["issues"] = row_issues
            invalid.append(item)
            issues.extend(f"{row.get('prompt_id', 'unknown')}: {issue}" for issue in row_issues)
        else:
            valid.append(row)

    category_balance = dict(Counter(str(row["category"]) for row in valid))
    difficulty_balance = dict(Counter(str(row["difficulty"]) for row in valid))
    domain_balance = dict(Counter(str(row["domain"]) for row in valid))
    issues.extend(_balance_issues("category", category_balance, CATEGORIES))
    issues.extend(_balance_issues("difficulty", difficulty_balance, DIFFICULTIES))
    issues.extend(_balance_issues("domain", domain_balance, DOMAINS))

    return ValidationResult(
        valid_rows=valid,
        invalid_rows=invalid,
        issues=issues,
        category_balance=category_balance,
        difficulty_balance=difficulty_balance,
        domain_balance=domain_balance,
    )


def _row_issues(row: dict[str, Any], min_words: int) -> list[str]:
    issues: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(row))
    if missing:
        issues.append(f"missing required fields: {', '.join(missing)}")
    prompt = str(row.get("prompt", "")).strip()
    if len(prompt.split()) < min_words:
        issues.append("prompt is too short")
    if not prompt.endswith((".", "?", "!", ":", ")")):
        issues.append("prompt appears incomplete")
    if row.get("category") not in CATEGORIES:
        issues.append("unsupported category")
    if row.get("difficulty") not in DIFFICULTIES:
        issues.append("unsupported difficulty")
    if row.get("domain") not in DOMAINS:
        issues.append("unsupported domain")
    try:
        complexity = float(row.get("estimated_complexity", -1))
        if not 0.0 <= complexity <= 1.0:
            issues.append("estimated_complexity is outside [0, 1]")
    except (TypeError, ValueError):
        issues.append("estimated_complexity is not numeric")
    try:
        if int(row.get("constraint_count", -1)) < 0:
            issues.append("constraint_count is negative")
    except (TypeError, ValueError):
        issues.append("constraint_count is not numeric")
    return issues


def _balance_issues(name: str, counts: dict[str, int], expected: tuple[str, ...]) -> list[str]:
    if not counts:
        return [f"no valid rows for {name} balance"]
    values = [counts.get(item, 0) for item in expected]
    minimum = min(values)
    maximum = max(values)
    if minimum == 0:
        return [f"{name} coverage is missing at least one bucket"]
    if maximum / minimum > 1.35:
        return [f"{name} distribution is imbalanced: min={minimum}, max={maximum}"]
    return []

