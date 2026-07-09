"""Prepare, validate, and split the routing dataset without training models."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from hashlib import sha1
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.data_generation.deduplicator import Deduplicator, normalize_prompt
from app.ml.decision_engine import DecisionEngine
from app.services.feature_extractor import extract_features

DATA_DIR = BACKEND_DIR / "app" / "data"
SOURCE_PATH = DATA_DIR / "datasets" / "10_000_chatbot_prompts.json"
EXISTING_PATH = DATA_DIR / "training" / "training_dataset_large.csv"
OUTPUT_DIR = DATA_DIR / "training"
FINAL_PATH = OUTPUT_DIR / "training_dataset_merged.csv"
TRAIN_PATH = OUTPUT_DIR / "training_dataset_large_train.csv"
VALIDATION_PATH = OUTPUT_DIR / "validation_dataset.csv"
TEST_PATH = OUTPUT_DIR / "locked_evaluation_dataset.csv"
REPORT_PATH = BACKEND_DIR / "docs" / "final_dataset_report.md"
RANDOM_STATE = 42
LONG_FRACTION = 0.24

CATEGORIES = ("planning", "translation", "reasoning", "summarization", "coding", "creative_writing", "general")
REMOTE_TARGET = {
    "planning": 0.50, "translation": 0.38, "reasoning": 0.62, "summarization": 0.48,
    "coding": 0.58, "creative_writing": 0.40, "general": 0.42,
}
FEATURE_COLUMNS = [
    "prompt_length", "word_count", "estimated_input_tokens", "contains_code", "contains_math",
    "contains_json", "contains_markdown", "contains_numbers", "contains_question",
    "reasoning_score", "technical_complexity", "reasoning_depth", "task_complexity",
    "constraint_complexity", "context_complexity", "complexity_score", "constraint_density",
    "requested_format", "presence_of_tables", "sql_indicators", "api_keywords",
    "system_design_keywords", "algorithmic_complexity", "dependency_between_subtasks",
    "multi_turn_context", "code_indicators", "math_indicators",
]

LOCAL_TEMPLATES = {
    "planning": (
        "Can you make a short, practical checklist for learning {topic}? Focus on {terms}.",
        "Help me plan a beginner-friendly afternoon project about {topic}. Keep the steps simple.",
    ),
    "translation": (
        "Translate this short sentence into plain English: “A brief introduction to {topic} and {term}.”",
        "How would I say “I am learning about {topic}” in Spanish? Give only the translation.",
    ),
    "reasoning": (
        "In simple terms, what is the difference between {term} and {term2} in {topic}?",
        "Why is {term} useful when studying {topic}? Give one everyday example.",
    ),
    "summarization": (
        "Summarize this in two sentences: {topic} covers {terms}.",
        "Give me a brief overview of {topic}, especially {term}.",
    ),
    "coding": (
        "Show a small Python example that represents {term}. Keep it beginner-friendly.",
        "What does {term} mean in software development? Explain it without designing a full system.",
    ),
    "creative_writing": (
        "Write a short, lighthearted paragraph inspired by {topic} and {term}.",
        "Give me three story ideas that use {topic} as a backdrop.",
    ),
    "general": (
        "What is {topic}? Explain {term} for a curious beginner.",
        "Give me five useful facts about {topic}, including {term}.",
    ),
}
REMOTE_TEMPLATES = {
    "planning": (
        "Create a phased implementation plan for an organization adopting {topic}. Compare options involving {terms}, identify dependencies, risks, owners, measurable milestones, and rollback criteria.",
        "Develop a twelve-month strategy for {topic}. Analyze trade-offs among {terms}, budget constraints, operational risks, and how success should be measured.",
    ),
    "translation": (
        "Translate a formal technical policy about {topic} into Spanish while preserving terminology for {terms}. Explain ambiguous phrases, maintain headings, and flag wording that needs legal or domain review.",
        "Prepare a publication-quality French translation of documentation about {topic}, keeping {terms} consistent. Include a terminology table and note context-sensitive choices.",
    ),
    "reasoning": (
        "Analyze {topic} by comparing {terms}. State assumptions, derive the important relationships step by step, test edge cases, and explain when each approach fails.",
        "Evaluate competing approaches to {topic}, focusing on {terms}. Build a reasoned recommendation with counterarguments, failure modes, and evidence needed to change the conclusion.",
    ),
    "summarization": (
        "Produce an executive summary of a detailed report on {topic}. Synthesize findings about {terms}, separate evidence from interpretation, preserve caveats, and list unresolved questions.",
        "Summarize research notes about {topic} for both specialists and executives. Reconcile claims involving {terms}, identify contradictions, and retain methodological limitations.",
    ),
    "coding": (
        "Design and implement a production-ready component for {topic} using {terms}. Explain the architecture, data structures, error handling, tests, performance trade-offs, security concerns, and deployment strategy.",
        "Review a proposed {topic} service that depends on {terms}. Find correctness and scalability risks, propose concrete code-level fixes, and provide a test and observability plan.",
    ),
    "creative_writing": (
        "Develop a layered short-story outline centered on {topic} and {terms}. Track character motivations, continuity, pacing, thematic callbacks, and revise the ending to resolve each major arc.",
        "Write a polished narrative treatment inspired by {topic}. Weave {terms} into the setting naturally, use distinct character voices, and maintain continuity across multiple scenes.",
    ),
    "general": (
        "Prepare an in-depth briefing on {topic}, connecting {terms}. Compare credible perspectives, explain practical consequences, identify uncertainty, and recommend questions for further investigation.",
        "Teach {topic} to an advanced learner. Integrate {terms}, explain underlying mechanisms, compare alternatives, and include realistic applications, limitations, and misconceptions.",
    ),
}


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return re.sub(
        r"\b(include additional (detail|context|explanation|analysis|guideline|outline) if necessary|"
        r"remember to|always aim to|avoid discussing topics outside[^.]*\.)\b",
        "", text, flags=re.IGNORECASE,
    ).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "general"


def _stable_number(value: str) -> int:
    return int(sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def _category(item: dict[str, Any], index: int) -> str:
    """Use realistic task intent as category while preserving subject as domain."""
    text = f"{item.get('parent_category', '')} {item.get('subcategory', '')}".lower()
    if any(word in text for word in ("program", "software", "computer", "api", "devops", "cloud", "cyber", "data structure")):
        choices = ("coding", "reasoning", "planning", "summarization", "general")
    elif any(word in text for word in ("writing", "literature", "story", "film", "arts", "music", "theater", "comics")):
        choices = ("creative_writing", "summarization", "reasoning", "general", "planning")
    else:
        choices = CATEGORIES
    return choices[index % len(choices)]


def _natural_prompt(item: dict[str, Any], index: int) -> tuple[str, str]:
    topic = _clean(item.get("subcategory")) or _clean(item.get("parent_category"))
    keywords = list(dict.fromkeys(_clean(word) for word in item.get("keywords", []) if _clean(word)))
    terms = keywords[:5] or [topic]
    category = _category(item, index)
    target_remote = (_stable_number(str(item.get("id", index))) % 10_000) / 10_000 < REMOTE_TARGET[category]
    templates = REMOTE_TEMPLATES[category] if target_remote else LOCAL_TEMPLATES[category]
    template = templates[_stable_number(topic + str(index)) % len(templates)]
    prompt = template.format(
        topic=topic, term=terms[0], term2=terms[1] if len(terms) > 1 else topic,
        terms=", ".join(terms),
    )
    return _clean(prompt), category


def import_json(path: Path) -> tuple[list[dict[str, Any]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Imported JSON root must be a list.")
    rows, invalid = [], 0
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            invalid += 1
            continue
        prompt, category = _natural_prompt(item, index)
        if len(prompt.split()) < 4:
            invalid += 1
            continue
        rows.append({
            "prompt_id": f"imported_{_clean(item.get('id')) or index}",
            "prompt": prompt,
            "category": category,
            "domain": _slug(_clean(item.get("parent_category"))),
            "topic": _clean(item.get("subcategory")),
            "keywords": [_clean(word) for word in item.get("keywords", []) if _clean(word)],
            "source_context": _clean(item.get("system_message")),
            "source": "10k_chatbot_prompts",
        })
    return rows, invalid


def _source_sentences(row: dict[str, Any]) -> list[str]:
    text = _clean(row.get("source_context"))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    rejected = ("you are ", "your expertise", "you can ", "when faced", "always ", "remember ", "avoid ")
    artifacts = ("if necessary", "include additional", "friendly and professional demeanor",
                 "feel comfortable asking", "outside the realm")
    return [
        sentence for sentence in sentences
        if 8 <= len(sentence.split()) <= 45
        and not sentence.lower().startswith(rejected)
        and not any(artifact in sentence.lower() for artifact in artifacts)
    ]


def _expand_prompt(row: dict[str, Any], target_words: int) -> str:
    """Expand one prompt into a coherent production-style request."""
    base = _clean(row["prompt"])
    base = re.sub(
        r"\b(keep (the answer|the request|your response)[^.]*|answer in (under|no more than) [^.]*|"
        r"limit (the answer|your response)[^.]*)\.?",
        "", base, flags=re.IGNORECASE,
    ).strip()
    topic = _clean(row.get("topic")) or _clean(row.get("domain")).replace("_", " ") or "the subject"
    terms = list(dict.fromkeys(row.get("keywords") or []))[:12] or [topic]
    category = str(row.get("category", "general"))
    rng = random.Random(_stable_number(str(row.get("prompt_id"))))
    source = _source_sentences(row)
    paragraphs = [
        base,
        f"Context: I am preparing material about {topic} for a real project. The audience includes people with "
        f"different levels of experience, so the response needs to be precise without assuming unstated background.",
    ]
    if source:
        paragraphs.append("Background notes: " + " ".join(source[:4]))

    lenses = [
        "underlying mechanism and assumptions", "realistic example and edge cases",
        "alternatives and decision criteria", "risks, limitations, and failure modes",
        "implementation sequence and dependencies", "testing or evidence needed to validate the result",
        "operational impact and maintenance concerns", "terminology that must remain consistent",
        "trade-offs involving cost, quality, time, and reliability", "questions a reviewer should ask",
    ]
    index = 0
    while len(" ".join(paragraphs).split()) < target_words:
        term = terms[index % len(terms)]
        other = terms[(index + 1) % len(terms)]
        lens = lenses[index % len(lenses)]
        if category == "coding":
            paragraph = (
                f"For {term}, describe the {lens}. Relate it to {other}, show representative interfaces or "
                f"pseudocode where useful, and discuss validation, error handling, security, performance, and "
                f"observability. Call out assumptions that would change the implementation."
            )
        elif category == "translation":
            paragraph = (
                f"The source section about {term} uses {other} in a domain-specific sense. Preserve headings and "
                f"references, keep terminology consistent, and annotate genuine ambiguity rather than silently "
                f"guessing. The translation should remain natural for a professional reader."
            )
        elif category == "summarization":
            paragraph = (
                f"The notes discuss {term} in relation to {other}, including the {lens}. Capture the main claim, "
                f"supporting evidence, caveats, and any disagreement. Do not turn tentative observations into facts."
            )
        elif category == "creative_writing":
            paragraph = (
                f"In the section involving {term}, connect it naturally to {other}. Develop sensory detail, character "
                f"motivation, and consequences without exposition dumps. Maintain established voice, chronology, "
                f"and continuity while advancing the central conflict."
            )
        else:
            paragraph = (
                f"Address {term} through the lens of {lens}. Connect it to {other}, distinguish established facts "
                f"from assumptions, give a concrete example, and explain what evidence or constraints would lead "
                f"to a different recommendation."
            )
        paragraphs.append(paragraph)
        index += 1

    closing_options = [
        "Finish with a concise decision summary and a checklist I can use during review.",
        "Organize the response with descriptive headings and make unresolved questions explicit.",
        "Prioritize actionable conclusions, but preserve important caveats and competing interpretations.",
    ]
    paragraphs.append(rng.choice(closing_options))
    words = "\n\n".join(paragraphs).split()
    if len(words) > target_words:
        words = words[:target_words]
        if words[-1][-1] not in ".!?":
            words[-1] += "."
    return " ".join(words)


def _augment_long_prompts(rows: list[dict[str, Any]], engine: DecisionEngine) -> Counter:
    """Replace 24% of prompts with deterministic long variants across four bands."""
    target_count = round(len(rows) * LONG_FRACTION)
    ranked: list[tuple[int, int, int]] = []
    for index, row in enumerate(rows):
        decision = engine.decide_from_features(extract_features(_clean(row["prompt"])))
        # Prefer prompts that are already REMOTE so augmentation does not distort
        # category-level routing behavior merely by adding context length.
        priority = 0 if decision["label"] == "REMOTE" else 1
        ranked.append((priority, _stable_number(str(row.get("prompt_id")) + "long"), index))
    candidates = [item[2] for item in sorted(ranked)[:target_count]]
    bands = ((100, 250), (250, 500), (500, 1000), (1000, 2000))
    counts: Counter = Counter()
    for position, row_index in enumerate(candidates):
        low, high = bands[position % len(bands)]
        span = high - low
        target = low + 10 + (_stable_number(str(rows[row_index].get("prompt_id"))) % max(span - 20, 1))
        rows[row_index]["prompt"] = _expand_prompt(rows[row_index], target)
        rows[row_index]["augmentation_band"] = f"{low}-{high}"
        counts[f"{low}-{high}"] += 1
    return counts


def _score(value: Any) -> float:
    return float(value.get("score", 0.0)) if isinstance(value, dict) else float(value or 0.0)


def enrich(row: dict[str, Any], engine: DecisionEngine) -> dict[str, Any]:
    prompt = _clean(row["prompt"])
    features = extract_features(prompt)
    decision = engine.decide_from_features(features)
    result = {
        "prompt_id": str(row.get("prompt_id") or sha1(prompt.encode()).hexdigest()[:16]),
        "prompt": prompt, "category": str(row.get("category") or features["task_type"]),
        "difficulty": str(features["complexity"]),
        "expected_reasoning": "high" if features["reasoning_score"] >= 7 else (
            "medium" if features["reasoning_score"] >= 3 else "low"),
        "domain": str(row.get("domain") or "general"),
        "constraint_count": len(features["constraint_complexity"].get("matched", [])),
        "estimated_complexity": float(features["complexity_score"]),
        "output_format": str(features["requested_format"]),
        "source": str(row.get("source") or "existing_project"),
        "augmentation_band": str(row.get("augmentation_band") or "under_100"),
    }
    for column in FEATURE_COLUMNS:
        value = features.get(column, 0)
        result[column] = _score(value) if isinstance(value, dict) else int(value) if isinstance(value, bool) else value
    result.update({
        "routing_score": decision["routing_score"], "routing_confidence": decision["routing_confidence"],
        "local_latency_ms": 0, "remote_latency_ms": 0, "local_cost": 0.0, "remote_cost": 0.0,
        "local_quality_score": 0.0, "remote_quality_score": 0.0,
        "local_llm_quality": 0.0, "remote_llm_quality": 0.0,
        "local_heuristic_quality": 0.0, "remote_heuristic_quality": 0.0,
        "local_tokens": 0, "remote_tokens": 0, "label": decision["label"],
    })
    return result


def _stratify_key(frame: pd.DataFrame) -> pd.Series:
    combined = frame["label"].astype(str) + "|" + frame["category"].astype(str) + "|" + frame["augmentation_band"]
    counts = combined.value_counts()
    fallback = frame["label"].astype(str) + "|sparse"
    return combined.where(combined.map(counts) >= 20, fallback)


def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(frame, test_size=.20, random_state=RANDOM_STATE, stratify=_stratify_key(frame))
    validation, test = train_test_split(
        remainder, test_size=.50, random_state=RANDOM_STATE, stratify=_stratify_key(remainder))
    return train, validation, test


def _validate(frame: pd.DataFrame, splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> dict[str, Any]:
    normalized = frame["prompt"].map(normalize_prompt)
    exact_duplicates = int(normalized.duplicated().sum())
    conflicts = int(frame.assign(_norm=normalized).groupby("_norm")["label"].nunique().gt(1).sum())
    train, validation, test = splits
    split_sets = [set(part["prompt"].map(normalize_prompt)) for part in splits]
    overlaps = [len(split_sets[0] & split_sets[1]), len(split_sets[0] & split_sets[2]), len(split_sets[1] & split_sets[2])]
    required = {"prompt", "category", "difficulty", "label", *FEATURE_COLUMNS}
    missing_columns = sorted(required - set(frame.columns))
    invalid_labels = int((~frame["label"].isin(["LOCAL", "REMOTE"])).sum())
    empty = int(frame["prompt"].isna().sum() + frame["prompt"].astype(str).str.strip().eq("").sum())
    consistent_lengths = int((frame["word_count"] != frame["prompt"].str.split().str.len()).sum())
    return {
        "passed": not any((exact_duplicates, conflicts, *overlaps, invalid_labels, empty, consistent_lengths)) and not missing_columns,
        "exact_duplicates": exact_duplicates, "conflicting_labels": conflicts, "split_overlaps": overlaps,
        "invalid_labels": invalid_labels, "empty_prompts": empty, "metadata_length_mismatches": consistent_lengths,
        "missing_columns": missing_columns, "split_sizes": [len(train), len(validation), len(test)],
    }


def _length_band(words: int) -> str:
    if words < 100: return "under 100"
    if words < 250: return "100-249"
    if words < 500: return "250-499"
    if words < 1000: return "500-999"
    if words <= 2000: return "1000-2000"
    return "over 2000"


def _write_report(frame: pd.DataFrame, splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
                  removal: dict[str, int], validation: dict[str, Any]) -> None:
    labels = frame["label"].value_counts()
    lengths = frame["word_count"].map(_length_band).value_counts()
    train, val, test = splits
    lines = [
        "# Final Dataset Report", "", "## Overall statistics", "",
        f"- Total prompts: {len(frame):,}",
        f"- LOCAL: {labels.get('LOCAL', 0):,} ({labels.get('LOCAL', 0) / len(frame):.1%})",
        f"- REMOTE: {labels.get('REMOTE', 0):,} ({labels.get('REMOTE', 0) / len(frame):.1%})",
        f"- Average prompt length: {frame.word_count.mean():.1f} words",
        f"- Median prompt length: {frame.word_count.median():.0f} words",
        f"- Prompt length range: {frame.word_count.min()}–{frame.word_count.max()} words",
        f"- Exact duplicates removed: {removal['exact']:,}",
        f"- Near duplicates removed: {removal['near']:,}",
        f"- Invalid prompts removed: {removal['invalid']:,}",
        "", "## Length distribution", "",
        "| Length | Count | Percentage |", "|---|---:|---:|",
    ]
    for band in ("under 100", "100-249", "250-499", "500-999", "1000-2000", "over 2000"):
        count = int(lengths.get(band, 0))
        lines.append(f"| {band} words | {count:,} | {count / len(frame):.1%} |")
    lines += ["", "## Category distribution and routing ratios", "",
              "| Category | Count | LOCAL | REMOTE | REMOTE ratio |", "|---|---:|---:|---:|---:|"]
    for category, group in frame.groupby("category"):
        local = int((group.label == "LOCAL").sum()); remote = int((group.label == "REMOTE").sum())
        lines.append(f"| {category} | {len(group):,} | {local:,} | {remote:,} | {remote / len(group):.1%} |")
    lines += [
        "", "## Split validation", "",
        f"- Train / validation / test: {len(train):,} / {len(val):,} / {len(test):,}",
        f"- Exact duplicates remaining: {validation['exact_duplicates']}",
        f"- Conflicting labels: {validation['conflicting_labels']}",
        f"- Train/validation, train/test, validation/test overlaps: {validation['split_overlaps']}",
        f"- Invalid labels: {validation['invalid_labels']}",
        f"- Empty prompts: {validation['empty_prompts']}",
        f"- Metadata length mismatches: {validation['metadata_length_mismatches']}",
        f"- Validation result: **{'PASS' if validation['passed'] else 'FAIL'}**",
        "", "## Augmented prompt examples", "",
    ]
    for _, row in frame[frame.augmentation_band != "under_100"].groupby("augmentation_band").head(1).iterrows():
        excerpt = row.prompt[:700].replace("\n", " ")
        lines += [f"### {row.augmentation_band} words — {row.category} / {row.label}", "", f"> {excerpt}…", ""]
    lines += [
        "## Recommendation", "",
        "The dataset passes structural validation and now includes production-scale prompt lengths. "
        "Use the fixed train, validation, and locked test files for retraining. Keep the test split untouched "
        "during model selection and threshold calibration.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare(source: Path = SOURCE_PATH, existing: Path = EXISTING_PATH) -> None:
    old = pd.read_csv(existing).to_dict("records")
    for row in old:
        if row.get("category") == "mathematics":
            row["category"] = "reasoning"
        prompt_terms = re.findall(r"\b[a-zA-Z][a-zA-Z-]{5,}\b", str(row.get("prompt", "")))
        stop = {"include", "provide", "explain", "format", "response", "request", "using", "should", "would"}
        keywords = list(dict.fromkeys(term.lower() for term in prompt_terms if term.lower() not in stop))[:12]
        row.update({"source": "existing_project", "topic": row.get("domain", ""), "keywords": keywords})
        row["prompt"] = _clean(row.get("prompt"))
    imported, invalid = import_json(source)
    candidates = old + imported
    valid = [row for row in candidates if len(_clean(row.get("prompt")) .split()) >= 3]
    invalid += len(candidates) - len(valid)
    dedup = Deduplicator(threshold=.82).deduplicate(valid)
    rows = list(dedup.kept)
    engine = DecisionEngine()
    long_counts = _augment_long_prompts(rows, engine)
    frame = pd.DataFrame(enrich(row, engine) for row in rows)
    splits = _split(frame)
    validation = _validate(frame, splits)
    if not validation["passed"]:
        raise RuntimeError(f"Dataset validation failed: {validation}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(FINAL_PATH, index=False)
    for part, path in zip(splits, (TRAIN_PATH, VALIDATION_PATH, TEST_PATH)):
        part.to_csv(path, index=False)
    _write_report(frame, splits, {
        "exact": dedup.duplicate_count, "near": dedup.near_duplicate_count, "invalid": invalid,
    }, validation)
    print(json.dumps({
        "total": len(frame), "labels": dict(Counter(frame.label)),
        "long_bands": dict(long_counts), "splits": validation["split_sizes"],
        "validation": validation, "report": str(REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--existing", type=Path, default=EXISTING_PATH)
    args = parser.parse_args()
    prepare(args.source, args.existing)
