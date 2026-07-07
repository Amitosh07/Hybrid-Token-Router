"""Evaluate heuristic and ML routers on unseen prompts.

This module creates new prompts, builds benchmark-style local/remote records,
uses the Decision Engine as offline ground truth, and writes a final evaluation
report. It does not retrain models or modify existing benchmark code.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app.ml.decision_engine import DecisionEngine
from app.ml.model_utils import DATASET_PATH, DOCS_DIR, LOCAL_LABEL, REMOTE_LABEL
from app.services import evaluator
from app.services.feature_extractor import extract_features
from app.services.ml_router import route as ml_route
from app.services.router import route as heuristic_route


REPORT_PATH = DOCS_DIR / "unseen_evaluation.md"


def generate_unseen_prompts() -> list[dict[str, str]]:
    """Create diverse prompts that are not copied from the training dataset."""
    seeds = {
        "coding": [
            ("easy", "Build a Ruby method that normalizes phone numbers into E.164 format and returns nil for invalid inputs."),
            ("easy", "Write a Go helper that reads environment variables into a typed configuration struct with defaults."),
            ("medium", "Create a Node.js middleware that validates signed webhooks, rejects replayed timestamps, and logs audit metadata."),
            ("medium", "Implement a Python function that batches database writes with exponential backoff and preserves input order."),
            ("hard", "Design a Rust async rate limiter for multiple API tenants using token buckets and lock-free counters."),
            ("hard", "Implement a Kubernetes operator reconciliation loop that rolls back failed deployments after health-check drift."),
            ("medium", "Write a SQL migration that backfills missing invoice currency codes without blocking reads."),
            ("easy", "Create a TypeScript utility that deeply freezes a settings object while preserving literal types."),
            ("hard", "Build a streaming parser for HL7 healthcare messages that validates segment order and emits structured errors."),
            ("medium", "Write a Python test harness for a fraud detection feature store with deterministic fixture generation."),
            ("hard", "Implement a Java circuit breaker with half-open recovery, metrics hooks, and thread-safe state transitions."),
            ("medium", "Create a Terraform module for a private S3 bucket with encryption, lifecycle rules, and least-privilege IAM."),
            ("easy", "Write a Bash script that archives old log files by date and prints a dry-run summary first."),
        ],
        "reasoning": [
            ("easy", "A school changes class start times by 20 minutes. Explain the likely effects on bus scheduling and attendance."),
            ("medium", "Compare two cybersecurity incident response options: immediate shutdown versus monitored containment."),
            ("hard", "Analyze whether a hospital should prioritize a scarce diagnostic device for emergency cases or scheduled oncology care."),
            ("medium", "Evaluate the trade-offs of using synthetic data for financial credit-risk model development."),
            ("hard", "Reason through the second-order effects of moving a monolith to event-driven microservices in a regulated bank."),
            ("easy", "Explain why a small online shop might prefer simpler inventory software over a full ERP system."),
            ("hard", "Assess the legal and ethical implications of automated hiring filters trained on historical promotion data."),
            ("medium", "A cloud team sees rising egress bills after a CDN migration. Infer plausible causes and diagnostic steps."),
            ("hard", "Analyze whether a research lab should publish a dual-use cybersecurity exploit with mitigations."),
            ("medium", "Compare remote-first and hybrid work policies for a customer support team with strict coverage targets."),
            ("easy", "Explain why adding more checkout lanes does not always reduce wait times in a grocery store."),
            ("hard", "Evaluate the systemic risk of relying on a single identity provider across all internal enterprise tools."),
            ("medium", "Reason about why a machine-learning classifier can improve accuracy while worsening fairness metrics."),
        ],
        "planning": [
            ("easy", "Create a one-week study plan for a high school biology exam covering genetics and ecology."),
            ("medium", "Plan a cloud cost optimization sprint for a startup running workloads on AWS and Kubernetes."),
            ("hard", "Design a 90-day rollout plan for zero-trust access controls across a 2,000-person healthcare organization."),
            ("medium", "Create a launch checklist for a B2B analytics product entering the education market."),
            ("easy", "Plan a half-day onboarding workshop for new customer success representatives."),
            ("hard", "Build a phased disaster recovery exercise for a fintech platform with regional failover requirements."),
            ("medium", "Create a hiring plan for a small machine-learning platform team over two quarters."),
            ("hard", "Develop a research operations plan for a multi-site clinical survey with privacy constraints."),
            ("easy", "Make a checklist for preparing a neighborhood community meeting about road safety."),
            ("medium", "Plan a DevOps migration from manual releases to trunk-based development and automated deployment."),
            ("hard", "Create an incident communications plan for a ransomware event affecting legal document systems."),
            ("medium", "Build a customer education webinar plan for an accounting automation feature."),
            ("easy", "Plan a three-day agenda for a small nonprofit board retreat."),
        ],
        "mathematics": [
            ("easy", "Calculate the monthly payment for a 12-month loan of 1200 dollars at zero interest."),
            ("medium", "Given a confusion matrix, compute precision, recall, and F1 for the positive class."),
            ("hard", "Derive the expected number of retries before success for an API call with exponential backoff and failure probability p."),
            ("medium", "Solve a linear system representing inventory balances across three warehouses."),
            ("easy", "Find the percentage change when revenue increases from 80,000 to 92,000 dollars."),
            ("hard", "Prove that the sum of the first n odd numbers equals n squared using induction."),
            ("medium", "Compute the break-even point for a SaaS product with fixed costs and per-customer support costs."),
            ("hard", "Analyze the convergence of a simple gradient descent update for a one-dimensional quadratic loss."),
            ("easy", "Convert 3.5 hours into minutes and seconds."),
            ("medium", "Calculate a 95 percent confidence interval for a small customer satisfaction sample."),
            ("hard", "Use Bayes theorem to estimate fraud probability after two independent risk signals fire."),
            ("medium", "Optimize the allocation of a limited advertising budget across two channels with linear returns."),
            ("easy", "Compute the mean, median, and range for a list of weekly ticket counts."),
        ],
        "translation": [
            ("easy", "Translate a short appointment reminder from English to Spanish in a polite tone."),
            ("medium", "Translate a cloud outage status update from English to German while preserving technical terms."),
            ("hard", "Translate a healthcare consent paragraph from English to Hindi with formal medical terminology."),
            ("medium", "Translate a cybersecurity training notice from English to Japanese using concise workplace language."),
            ("easy", "Translate a restaurant allergy note from English to French."),
            ("hard", "Translate a legal data-processing clause from English to Portuguese without changing defined terms."),
            ("medium", "Translate a machine-learning model card summary from English to Korean."),
            ("easy", "Translate a delivery delay notification from English to Italian."),
            ("hard", "Translate a distributed systems research abstract from English to Mandarin Chinese with academic style."),
            ("medium", "Translate a finance dashboard tooltip set from English to Arabic."),
            ("easy", "Translate a classroom announcement from English to Dutch."),
            ("hard", "Translate an incident response procedure from English to Polish with exact severity labels."),
            ("medium", "Translate a product onboarding email from English to Swedish."),
        ],
        "summarization": [
            ("easy", "Summarize a short team standup note into three bullet points."),
            ("medium", "Summarize a cloud architecture review into risks, decisions, and follow-up actions."),
            ("hard", "Summarize a clinical research protocol into eligibility criteria, endpoints, and safety monitoring."),
            ("medium", "Summarize a quarterly finance memo for executives and highlight budget variances."),
            ("easy", "Summarize a parent-teacher meeting note in plain language."),
            ("hard", "Summarize a legal brief about data retention obligations into arguments and open questions."),
            ("medium", "Summarize a cybersecurity tabletop exercise report into lessons learned."),
            ("easy", "Summarize a short product feedback thread into themes."),
            ("hard", "Summarize a distributed database design document with emphasis on consistency and failover."),
            ("medium", "Summarize a machine-learning experiment log into metrics, regressions, and next steps."),
            ("easy", "Summarize a travel itinerary into morning, afternoon, and evening plans."),
            ("hard", "Summarize an education policy research paper into methodology, findings, and limitations."),
            ("medium", "Summarize a DevOps postmortem into root cause, impact, and corrective actions."),
        ],
        "creative_writing": [
            ("easy", "Write a warm birthday message for a mentor who loves astronomy."),
            ("medium", "Draft a short story scene about a chef discovering a recipe that predicts memories."),
            ("hard", "Write a courtroom monologue from an AI archivist defending the preservation of disputed historical records."),
            ("medium", "Compose a poem about a city data center during a thunderstorm."),
            ("easy", "Write a friendly thank-you note to volunteers after a school fundraiser."),
            ("hard", "Create a speculative fiction opening where cloud regions are treated as independent city-states."),
            ("medium", "Write dialogue between a cybersecurity analyst and a nervous CEO before a press conference."),
            ("easy", "Write a playful slogan for a community library reading challenge."),
            ("hard", "Draft a literary scene about a doctor using an imperfect diagnostic oracle in a rural clinic."),
            ("medium", "Write a product launch narrative for a small finance app without sounding like an advertisement."),
            ("easy", "Create a four-line poem about learning algebra."),
            ("hard", "Write a research expedition journal entry from a climate scientist stranded at a remote sensor station."),
            ("medium", "Write a reflective essay opening about leaving a family business to study machine learning."),
        ],
        "general": [
            ("easy", "Explain what a password manager does for a non-technical audience."),
            ("medium", "Explain how container orchestration helps a growing software team manage deployments."),
            ("hard", "Explain the relationship between central bank interest-rate policy and startup financing conditions."),
            ("medium", "Describe how hospitals use triage systems during emergency department crowding."),
            ("easy", "Explain the difference between a debit card and a credit card."),
            ("hard", "Explain how consensus protocols help distributed databases tolerate node failures."),
            ("medium", "Describe common causes of machine-learning data drift in retail demand forecasting."),
            ("easy", "Explain why regular backups matter for a small business."),
            ("hard", "Explain the major privacy risks in linking education records with workforce outcome data."),
            ("medium", "Describe how cyber insurance underwriting evaluates company security posture."),
            ("easy", "Explain what a mutual fund is in simple terms."),
            ("hard", "Explain why legal discovery systems need careful access controls and audit logging."),
            ("medium", "Describe how peer review works in academic research."),
        ],
    }
    prompts: list[dict[str, str]] = []
    for category, items in seeds.items():
        for idx, (difficulty, prompt) in enumerate(items, start=1):
            prompts.append({
                "id": f"unseen_{category}_{idx:03d}",
                "category": category,
                "difficulty": difficulty,
                "prompt": prompt,
            })
    return prompts


def run_unseen_evaluation() -> dict[str, Any]:
    """Run evaluation and write the report."""
    prompts = _dedupe_training_prompts(generate_unseen_prompts())
    engine = DecisionEngine()
    rows: list[dict[str, Any]] = []

    for item in prompts:
        features = extract_features(item["prompt"])
        heuristic = heuristic_route(features)
        ml = ml_route(features)
        record = _decision_record(item, features)
        decision = engine.decide(record)
        rows.append({
            **item,
            "decision_engine": decision["label"],
            "heuristic": heuristic["provider"].upper(),
            "heuristic_confidence": heuristic.get("confidence", 0.0),
            "ml": ml["selected_provider"].upper(),
            "ml_confidence": ml["prediction_confidence"],
            "ml_probability": ml["prediction_probability"],
            "routing_method": ml["routing_method"],
            "complexity": features.get("complexity", ""),
            "reasoning_score": features.get("reasoning_score", 0),
            "token_estimate": features.get("estimated_input_tokens", 0),
            "decision_reason": decision["reason"],
        })

    frame = pd.DataFrame(rows)
    report = _build_report(frame)
    REPORT_PATH.write_text(report, encoding="utf-8")
    return {
        "rows": len(frame),
        "ml_accuracy": _metrics(frame, "ml")["accuracy"],
        "heuristic_accuracy": _metrics(frame, "heuristic")["accuracy"],
        "report_path": str(REPORT_PATH),
    }


def _dedupe_training_prompts(prompts: list[dict[str, str]]) -> list[dict[str, str]]:
    if not DATASET_PATH.exists():
        return prompts
    training = set(pd.read_csv(DATASET_PATH)["prompt"].astype(str).str.strip().str.lower())
    unseen = [item for item in prompts if item["prompt"].strip().lower() not in training]
    if len(unseen) != len(prompts):
        raise ValueError("Generated unseen prompts unexpectedly overlap the training dataset.")
    return unseen


def _decision_record(item: dict[str, str], features: dict[str, Any]) -> dict[str, Any]:
    local_quality, remote_quality, local_latency, remote_latency = _provider_profile(item, features)
    local_words = max(20, round(local_quality * 120))
    remote_words = max(25, round(remote_quality * 140))
    local_response = _response_text(item["prompt"], local_words, "local")
    remote_response = _response_text(item["prompt"], remote_words, "remote")
    input_tokens = int(features.get("estimated_input_tokens", math.ceil(len(item["prompt"]) / 4)))
    record = {
        "id": item["id"],
        "prompt": item["prompt"],
        "category": item["category"],
        "difficulty": item["difficulty"],
        "features": features,
        "local": {
            "model": "simulated-local",
            "response": local_response,
            "latency_ms": local_latency,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": math.ceil(len(local_response) / 4),
        },
        "remote": {
            "model": "simulated-remote",
            "response": remote_response,
            "latency_ms": remote_latency,
            "estimated_input_tokens": input_tokens,
            "estimated_output_tokens": math.ceil(len(remote_response) / 4),
        },
    }
    record["evaluation"] = evaluator.evaluate_run(record)
    metrics = record["evaluation"]["metrics"]["estimated_quality"]
    metrics["local"]["score"] = local_quality
    metrics["remote"]["score"] = remote_quality
    return record


def _provider_profile(item: dict[str, str], features: dict[str, Any]) -> tuple[float, float, int, int]:
    difficulty = item["difficulty"]
    complexity = float(features.get("complexity_score", 0.0))
    constraint = _score(features.get("constraint_complexity"))
    technical = _score(features.get("technical_complexity"))
    tokens = int(features.get("estimated_input_tokens", 0))

    difficulty_boost = {"easy": 0.0, "medium": 0.08, "hard": 0.16}[difficulty]
    remote_quality = min(0.98, 0.62 + complexity * 0.20 + difficulty_boost + technical * 0.08)
    local_quality = min(0.92, 0.70 - difficulty_boost * 0.8 - complexity * 0.18 + (0.08 if difficulty == "easy" else 0.0))

    if item["category"] in {"translation", "summarization", "creative_writing"} and difficulty != "hard":
        local_quality += 0.07
    if item["category"] in {"coding", "mathematics", "reasoning"} and difficulty == "hard":
        remote_quality += 0.05
        local_quality -= 0.06
    if constraint > 0.2:
        remote_quality += 0.04
        local_quality -= 0.02

    local_quality = round(max(0.25, min(local_quality, 0.95)), 4)
    remote_quality = round(max(0.35, min(remote_quality, 0.99)), 4)
    local_latency = int(300 + tokens * 12 + complexity * 900)
    remote_latency = int(600 + tokens * 18 + complexity * 1200 + (250 if difficulty == "hard" else 0))
    return local_quality, remote_quality, local_latency, remote_latency


def _response_text(prompt: str, words: int, provider: str) -> str:
    terms = " ".join([word.strip(".,:;!?") for word in prompt.split()[:14]])
    base = f"{provider} response covering {terms}. "
    filler = "It addresses requirements, constraints, examples, risks, and practical trade-offs. "
    text = base
    while len(text.split()) < words:
        text += filler
    return text


def _score(value: Any) -> float:
    if isinstance(value, dict):
        return float(value.get("score", 0.0))
    return float(value or 0.0)


def _metrics(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    y_true = (frame["decision_engine"] == REMOTE_LABEL).astype(int)
    y_pred = (frame[column] == REMOTE_LABEL).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _build_report(frame: pd.DataFrame) -> str:
    ml_metrics = _metrics(frame, "ml")
    heuristic_metrics = _metrics(frame, "heuristic")
    lines = [
        "# Unseen Prompt Evaluation",
        "",
        f"- Unseen prompts evaluated: {len(frame)}",
        "- Ground truth: Phase 2 Decision Engine applied to benchmark-compatible simulated local/remote runs.",
        "- Prompt overlap check: no exact prompts from `training_dataset.csv` were reused.",
        "",
        "## Overall Metrics",
        "",
        "| System | Accuracy | Precision | Recall | F1 | Agreement with Decision Engine |",
        "|---|---:|---:|---:|---:|---:|",
        f"| ML Router | {ml_metrics['accuracy']:.4f} | {ml_metrics['precision']:.4f} | {ml_metrics['recall']:.4f} | {ml_metrics['f1']:.4f} | {ml_metrics['accuracy']:.4f} |",
        f"| Heuristic Router | {heuristic_metrics['accuracy']:.4f} | {heuristic_metrics['precision']:.4f} | {heuristic_metrics['recall']:.4f} | {heuristic_metrics['f1']:.4f} | {heuristic_metrics['accuracy']:.4f} |",
        "",
        "## Confusion Matrix",
        "",
        "Rows are actual `[LOCAL, REMOTE]`; columns are predicted `[LOCAL, REMOTE]`.",
        "",
        f"- ML Router: `{ml_metrics['confusion_matrix']}`",
        f"- Heuristic Router: `{heuristic_metrics['confusion_matrix']}`",
        "",
        "## Per-Category Accuracy",
        "",
        "| Category | ML Accuracy | Heuristic Accuracy | Count |",
        "|---|---:|---:|---:|",
    ]
    for category, group in frame.groupby("category"):
        lines.append(
            f"| {category} | {(group['ml'] == group['decision_engine']).mean():.4f} | "
            f"{(group['heuristic'] == group['decision_engine']).mean():.4f} | {len(group)} |"
        )
    lines.extend(["", "## Per-Difficulty Accuracy", "", "| Difficulty | ML Accuracy | Heuristic Accuracy | Count |", "|---|---:|---:|---:|"])
    for difficulty, group in frame.groupby("difficulty"):
        lines.append(
            f"| {difficulty} | {(group['ml'] == group['decision_engine']).mean():.4f} | "
            f"{(group['heuristic'] == group['decision_engine']).mean():.4f} | {len(group)} |"
        )
    lines.extend(["", "## Top 20 Misclassified Prompts", ""])
    misses = frame[frame["ml"] != frame["decision_engine"]].copy()
    if misses.empty:
        lines.append("- No ML misclassifications on the unseen prompt set.")
    else:
        misses["confidence_gap"] = (misses["ml_confidence"] - 0.5).abs()
        for _, row in misses.sort_values("confidence_gap", ascending=False).head(20).iterrows():
            lines.append(
                f"- `{row['id']}` ({row['category']}, {row['difficulty']}): Decision Engine `{row['decision_engine']}`, "
                f"ML `{row['ml']}` at confidence {row['ml_confidence']:.4f}. "
                f"Why: {_misclassification_reason(row)} Prompt: {row['prompt']}"
            )
    return "\n".join(lines) + "\n"


def _misclassification_reason(row: pd.Series) -> str:
    if row["ml"] == REMOTE_LABEL and row["decision_engine"] == LOCAL_LABEL:
        return (
            "pre-routing features such as length, technical content, or constraints looked remote-worthy, "
            "but Decision Engine utility favored local after simulated quality/latency trade-offs."
        )
    return (
        "pre-routing features looked simple or local-friendly, but Decision Engine favored remote after "
        "simulated response quality gains outweighed latency and cost."
    )


if __name__ == "__main__":
    result = run_unseen_evaluation()
    print(result)
