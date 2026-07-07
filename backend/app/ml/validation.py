"""Dataset Validation Module for Phase 2.

Inbound quality checks for training_dataset.csv. Verifies row/col consistency,
duplicates, nulls, out-of-range feature values, class balance, and prints report.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CSV_PATH = _PROJECT_ROOT / "app" / "data" / "training" / "training_dataset.csv"


def validate_dataset(csv_path: Path | None = None) -> dict[str, Any]:
    """Inspect and validate the training dataset CSV.

    Args:
        csv_path: Path to the generated CSV. If None, checks the default path.

    Returns:
        Dictionary containing validation success flag and stats.
    """
    path = csv_path or _DEFAULT_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"Training dataset CSV not found at: {path.resolve()}")

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        for r in reader:
            rows.append(r)

    total_samples = len(rows)
    issues: list[str] = []

    # 1. Check schema / columns
    expected_cols = [
        "prompt_id", "prompt", "category", "difficulty", "expected_reasoning",
        "prompt_length", "word_count", "estimated_input_tokens",
        "contains_code", "contains_math", "contains_json", "contains_markdown",
        "contains_numbers", "contains_question", "reasoning_score",
        "technical_complexity", "reasoning_depth", "task_complexity",
        "constraint_complexity", "context_complexity", "complexity_score",
        "local_latency_ms", "remote_latency_ms", "local_cost", "remote_cost",
        "local_quality_score", "remote_quality_score", "label"
    ]

    missing_cols = [col for col in expected_cols if col not in headers]
    if missing_cols:
        issues.append(f"Schema mismatch: missing columns {missing_cols}")

    # 2. Check for missing values, duplicates, and ranges
    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    dup_ids = 0
    dup_prompts = 0
    null_count = 0
    invalid_labels = 0
    out_of_range_features = 0

    class_counts = {"LOCAL": 0, "REMOTE": 0}

    for idx, row in enumerate(rows, start=1):
        # A. Missing values check
        for col in expected_cols:
            if col in row and (row[col] is None or row[col].strip() == ""):
                null_count += 1
                issues.append(f"Row {idx}: missing value in column {col}")

        # B. Duplicate IDs / Prompts
        pid = row.get("prompt_id", "")
        if pid:
            if pid in seen_ids:
                dup_ids += 1
            seen_ids.add(pid)

        prompt_txt = row.get("prompt", "").strip()
        if prompt_txt:
            if prompt_txt in seen_prompts:
                dup_prompts += 1
            seen_prompts.add(prompt_txt)

        # C. Invalid labels
        label = row.get("label", "").upper()
        if label not in {"LOCAL", "REMOTE"}:
            invalid_labels += 1
            issues.append(f"Row {idx} ({pid}): invalid routing label '{label}'")
        else:
            class_counts[label] += 1

        # D. Numeric ranges validation
        try:
            # Binary indicators
            for col in ["contains_code", "contains_math", "contains_json", "contains_markdown", "contains_numbers", "contains_question"]:
                if col in row:
                    val = int(row[col])
                    if val not in {0, 1}:
                        out_of_range_features += 1
                        issues.append(f"Row {idx}: {col} has invalid binary value {val}")

            # reasoning score range [0, 10]
            if "reasoning_score" in row:
                rs = int(row["reasoning_score"])
                if rs < 0 or rs > 10:
                    out_of_range_features += 1
                    issues.append(f"Row {idx}: reasoning_score {rs} is outside [0, 10]")

            # complexity scores range [0.0, 1.0]
            for col in ["technical_complexity", "reasoning_depth", "task_complexity", "constraint_complexity", "context_complexity", "complexity_score", "local_quality_score", "remote_quality_score"]:
                if col in row:
                    val = float(row[col])
                    if val < 0.0 or val > 1.0:
                        out_of_range_features += 1
                        issues.append(f"Row {idx}: {col} score {val:.4f} is outside [0.0, 1.0]")

            # Latency ranges
            for col in ["local_latency_ms", "remote_latency_ms"]:
                if col in row:
                    val = int(row[col])
                    if val < 0:
                        out_of_range_features += 1
                        issues.append(f"Row {idx}: {col} {val} is negative")

        except ValueError as exc:
            out_of_range_features += 1
            issues.append(f"Row {idx}: numeric formatting error: {exc}")

    # Report results summary
    has_issues = len(issues) > 0
    validation_status = "FAILED" if has_issues else "PASSED"

    if dup_ids > 0:
        issues.append(f"Detected {dup_ids} duplicate prompt IDs.")
    if dup_prompts > 0:
        issues.append(f"Detected {dup_prompts} duplicate prompt texts.")

    # Class balance details
    local_cnt = class_counts["LOCAL"]
    remote_cnt = class_counts["REMOTE"]
    local_pct = (local_cnt / max(total_samples, 1)) * 100
    remote_pct = (remote_cnt / max(total_samples, 1)) * 100

    report = [
        "========================================================================",
        "                     TRAINING DATASET VALIDATION REPORT",
        "========================================================================",
        f"  Dataset Path        : {path.resolve()}",
        f"  Total Rows          : {total_samples}",
        f"  Total Columns       : {len(headers)}",
        f"  Validation Status   : {validation_status}",
        "------------------------------------------------------------------------",
        f"  Duplicate IDs       : {dup_ids}",
        f"  Duplicate Prompts   : {dup_prompts}",
        f"  Missing Cells (Null): {null_count}",
        f"  Invalid Labels      : {invalid_labels}",
        f"  Out of Range Feats  : {out_of_range_features}",
        "------------------------------------------------------------------------",
        "  Class Balance (Target Routing Labels):",
        f"    LOCAL  : {local_cnt:<5} ({local_pct:.1f}%)",
        f"    REMOTE : {remote_cnt:<5} ({remote_pct:.1f}%)",
        "========================================================================"
    ]

    if has_issues:
        report.append("\nIssues Found:")
        for iss in issues[:20]:  # Cap console logs at 20 lines to keep clean
            report.append(f"  [!] {iss}")
        if len(issues) > 20:
            report.append(f"  ... and {len(issues) - 20} more issues.")
        report.append("========================================================================")

    print("\n".join(report))

    return {
        "success": not has_issues,
        "status": validation_status,
        "total_samples": total_samples,
        "duplicate_ids": dup_ids,
        "duplicate_prompts": dup_prompts,
        "missing_values": null_count,
        "out_of_range_features": out_of_range_features,
        "invalid_labels": invalid_labels,
        "local_count": local_cnt,
        "remote_count": remote_cnt,
        "local_percentage": local_pct,
        "remote_percentage": remote_pct,
        "issues": issues,
    }


if __name__ == "__main__":
    validate_dataset()
