#!/usr/bin/env python3
"""Feature Extractor Validation & Calibration Tool.

Offline validation — no LLM calls required.

Runs the Feature Extraction Pipeline (v2) against every structured benchmark
prompt and produces:

  1. V1 vs V2 comparison table  (simulated V1 score is reconstructed)
  2. Difficulty separation report
  3. Score distribution by difficulty band
  4. Detailed routing trace for 20 randomly selected prompts
  5. ML compatibility notes (feature correlation warnings)

Usage
-----
    cd backend
    python tools/validate_feature_extractor.py

    # Enable verbose debug traces for 20 random prompts:
    python tools/validate_feature_extractor.py --verbose

    # Save report to file:
    python tools/validate_feature_extractor.py --output report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup: make sure `app` is importable when running from backend/
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from app.services.feature_extractor import extract_features  # noqa: E402
from app.services.router import route, PROVIDER_LOCAL, PROVIDER_REMOTE  # noqa: E402

# ---------------------------------------------------------------------------
# V1 reasoning_score simulation (for comparison only)
# ---------------------------------------------------------------------------

_V1_REASONING_KW = re.compile(
    r"\b(because|therefore|hence|thus|since|given that|it follows|consequently"
    r"|analyze|analyse|evaluate|critique|compare|contrast|justify|deduce"
    r"|infer|conclude|explain why|explain how|step by step|chain of thought"
    r"|reason|argument|logic|proof|implication|cause|effect)\b",
    re.IGNORECASE,
)
_V1_CONSTRAINT = re.compile(
    r"\b(must|should not|do not|must not|only if|at least|at most|exactly"
    r"|no more than|no less than|without|except|unless|provided that"
    r"|assuming|given that|such that|subject to|constraint)\b",
    re.IGNORECASE,
)
_V1_CODE = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`"
    r"|\b(def |class |import |function |return |var |let |const |#include"
    r"|public static|void |int |float |bool |lambda |async def |await )\b",
    re.IGNORECASE,
)
_V1_MATH = re.compile(
    r"\$\$[\s\S]*?\$\$|\$[^\$\n]+\$"
    r"|\b(integral|derivative|matrix|equation|solve|factorial|logarithm"
    r"|calculus|algebra|geometry|trigonometry|eigenvalue|polynomial)\b"
    r"|[=<>≤≥≠±∑∏√∫∂∇]|\b\d+\s*[+\-*/^]\s*\d+",
    re.IGNORECASE,
)


def _v1_reasoning_score(prompt: str) -> int:
    """Reconstruct the V1 reasoning_score for comparison purposes."""
    length = len(prompt)
    if length > 900:
        ls = 3
    elif length > 500:
        ls = 2
    elif length > 200:
        ls = 1
    else:
        ls = 0

    kw_count = len(_V1_REASONING_KW.findall(prompt))
    if kw_count >= 6:
        ks = 3
    elif kw_count >= 3:
        ks = 2
    elif kw_count >= 1:
        ks = 1
    else:
        ks = 0

    cs_count = len(_V1_CONSTRAINT.findall(prompt))
    if cs_count >= 3:
        cs = 2
    elif cs_count >= 1:
        cs = 1
    else:
        cs = 0

    code = 1 if _V1_CODE.search(prompt) else 0
    math = 1 if _V1_MATH.search(prompt) else 0
    return min(10, ls + ks + cs + code + math)


def _v1_routing_score(prompt: str) -> int:
    """Reconstruct the full V1 routing_score for comparison."""
    rs = _v1_reasoning_score(prompt)
    tokens = math.ceil(len(prompt) / 4)

    comp_map = {0: "easy", 1: "easy", 2: "easy", 3: "easy",
                4: "medium", 5: "medium", 6: "medium", 7: "medium",
                8: "hard", 9: "hard", 10: "hard"}
    complexity = comp_map[rs]
    comp_w = {"easy": 0, "medium": 5, "hard": 10}[complexity]

    # Task type detection (V1)
    v1_task_patterns = [
        ("coding", re.compile(
            r"\b(code|program|script|function|class|debug|implement|algorithm"
            r"|api|library|refactor|compile|syntax|error|bug|unit test"
            r"|import|module|package|repository|github|git)\b", re.I)),
        ("mathematics", re.compile(
            r"\b(calculate|compute|solve|equation|integral|derivative|matrix"
            r"|factorial|prime|fibonacci|probability|statistics|percentage"
            r"|algebra|geometry|calculus|proof|theorem|formula|math)\b", re.I)),
        ("reasoning", re.compile(
            r"\b(why|reason|analyze|analyse|explain|cause|effect|impact|infer"
            r"|deduce|conclude|argument|logic|because|therefore|hence|compare"
            r"|contrast|evaluate|assess|critique|justify|implication)\b", re.I)),
        ("planning", re.compile(
            r"\b(plan|roadmap|schedule|strategy|steps to|how to|checklist"
            r"|outline|agenda|milestone|goal|objective|task list|project"
            r"|timeline|action items|organize|prioritize)\b", re.I)),
        ("summarization", re.compile(
            r"\b(summarize|summarise|summary|tldr|brief|overview|condense"
            r"|shorten|abstract|synopsis|recap|key points|highlights)\b", re.I)),
        ("translation", re.compile(
            r"\b(translate|translation)\b", re.I)),
        ("creative_writing", re.compile(
            r"\b(write a (story|poem|song|essay|novel|script|blog|letter"
            r"|article)|creative|fiction|narrative|plot|character|dialogue"
            r"|rhyme|stanza|verse|haiku|limerick|short story)\b", re.I)),
        ("question_answering", re.compile(
            r"\b(what is|what are|who is|who are|when did|where is|where are"
            r"|how does|how do|how many|how much|define|definition of"
            r"|meaning of|tell me about)\b", re.I)),
    ]
    task_weights = {
        "coding": 4, "mathematics": 4, "reasoning": 5,
        "summarization": 3, "translation": 2, "creative_writing": 3,
        "planning": 4, "question_answering": 1, "general": 0,
    }
    task_type = "general"
    for tt, pat in v1_task_patterns:
        if pat.search(prompt):
            task_type = tt
            break

    task_w = task_weights.get(task_type, 0)
    code_w = 5 if _V1_CODE.search(prompt) else 0
    math_w = 4 if _V1_MATH.search(prompt) else 0

    return rs * 5 + (tokens // 256) * 3 + comp_w + task_w + code_w + math_w


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def _load_all_prompts(prompts_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted(prompts_dir.glob("*.json")):
        if path.name == "example_prompts.json":
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("prompt"):
                        entries.append(item)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARN] Could not load {path.name}: {exc}")
    return entries


# ---------------------------------------------------------------------------
# Difficulty-to-score separation metric
# ---------------------------------------------------------------------------

def _separation_score(easy_scores: list[float], hard_scores: list[float]) -> float:
    """Cohen's d between easy and hard complexity_score distributions."""
    if not easy_scores or not hard_scores:
        return 0.0
    ne, nh = len(easy_scores), len(hard_scores)
    me = sum(easy_scores) / ne
    mh = sum(hard_scores) / nh
    ve = sum((x - me) ** 2 for x in easy_scores) / max(ne - 1, 1)
    vh = sum((x - mh) ** 2 for x in hard_scores) / max(nh - 1, 1)
    pooled_sd = math.sqrt((ve + vh) / 2)
    return round((mh - me) / pooled_sd, 3) if pooled_sd > 0 else float("inf")


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _pct(part: int, total: int) -> str:
    return f"{100 * part // max(total, 1)}%"


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

_W = 72
_SEP = "─" * _W
_DBL = "═" * _W


def _header(title: str) -> str:
    pad = (_W - len(title) - 2) // 2
    return f"\n{'═' * _W}\n{'═' * pad} {title} {'═' * (pad + (_W - len(title) - 2) % 2)}\n{'═' * _W}"


def _section(title: str) -> str:
    return f"\n{_SEP}\n  {title}\n{_SEP}"


def _bar(value: float, width: int = 30) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def run_validation(verbose: bool = False, seed: int = 42) -> str:
    random.seed(seed)

    _PROMPTS_DIR = _BACKEND_DIR / "app" / "data" / "prompts"
    entries = _load_all_prompts(_PROMPTS_DIR)
    if not entries:
        return "[ERROR] No prompt files found."

    lines: list[str] = []

    # -----------------------------------------------------------------------
    # Run both V1 and V2 on every prompt
    # -----------------------------------------------------------------------
    results: list[dict] = []
    for entry in entries:
        prompt = entry["prompt"]
        difficulty = entry.get("difficulty", "unknown")
        category = entry.get("category", "unknown")
        pid = entry.get("id", "?")

        v2_features = extract_features(prompt, debug=True)
        v2_route = route(v2_features)
        v1_rs = _v1_routing_score(prompt)
        v1_provider = PROVIDER_REMOTE if v1_rs >= 25 else PROVIDER_LOCAL

        results.append({
            "id": pid,
            "category": category,
            "difficulty": difficulty,
            "prompt": prompt,
            "v1_routing_score": v1_rs,
            "v1_provider": v1_provider,
            "v2_complexity_score": v2_features["complexity_score"],
            "v2_reasoning_score": v2_features["reasoning_score"],
            "v2_routing_score": v2_route["routing_score"],
            "v2_provider": v2_route["provider"],
            "v2_complexity_label": v2_features["complexity"],
            "features": v2_features,
        })

    # -----------------------------------------------------------------------
    # Aggregate statistics
    # -----------------------------------------------------------------------
    total = len(results)

    # V1 stats
    v1_remote = sum(1 for r in results if r["v1_provider"] == PROVIDER_REMOTE)
    v1_local = total - v1_remote
    v1_scores = [r["v1_routing_score"] for r in results]

    # V2 stats
    v2_remote = sum(1 for r in results if r["v2_provider"] == PROVIDER_REMOTE)
    v2_local = total - v2_remote
    v2_cs = [r["v2_complexity_score"] for r in results]
    v2_rs = [r["v2_routing_score"] for r in results]

    # Per-difficulty aggregation
    diffs = ["easy", "medium", "hard"]
    by_diff: dict[str, list[dict]] = {d: [] for d in diffs}
    for r in results:
        d = r["difficulty"]
        if d in by_diff:
            by_diff[d].append(r)

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    lines.append(_header("HYBRID TOKEN ROUTER — FEATURE EXTRACTOR CALIBRATION REPORT"))
    lines.append(f"\n  Total prompts : {total}")
    lines.append(f"  Categories    : {', '.join(sorted({r['category'] for r in results}))}")
    lines.append(f"  Seed (sample) : {seed}")

    # -----------------------------------------------------------------------
    # Section 1: V1 vs V2 comparison
    # -----------------------------------------------------------------------
    lines.append(_section("1. V1 vs V2 — Overall Routing Distribution"))
    lines.append(
        f"\n  {'Metric':<35} {'V1':>10} {'V2':>10}"
    )
    lines.append(f"  {'─'*35} {'─'*10} {'─'*10}")
    lines.append(f"  {'Routed → LOCAL':<35} {v1_local:>10} {v2_local:>10}")
    lines.append(f"  {'Routed → REMOTE':<35} {v1_remote:>10} {v2_remote:>10}")
    lines.append(f"  {'Remote %':<35} {_pct(v1_remote, total):>10} {_pct(v2_remote, total):>10}")
    lines.append(f"  {'Average routing score':<35} {_avg(v1_scores):>10.1f} {_avg(v2_rs):>10.1f}")
    lines.append(f"  {'Average complexity score (V2)':<35} {'—':>10} {_avg(v2_cs):>10.4f}")

    # -----------------------------------------------------------------------
    # Section 2: Score distribution by difficulty
    # -----------------------------------------------------------------------
    lines.append(_section("2. Complexity Score Distribution by Difficulty (V2)"))
    lines.append(f"\n  {'Difficulty':<10} {'N':>4} {'Min':>6} {'Avg':>6} {'Max':>6} "
                 f"{'REMOTE':>7}  {'Distribution (0→1)'}")
    lines.append(f"  {'─'*10} {'─'*4} {'─'*6} {'─'*6} {'─'*6} {'─'*7}  {'─'*32}")

    for diff in diffs:
        group = by_diff[diff]
        if not group:
            continue
        cs = [r["v2_complexity_score"] for r in group]
        remote_n = sum(1 for r in group if r["v2_provider"] == PROVIDER_REMOTE)
        mn, av, mx = min(cs), _avg(cs), max(cs)
        bar = _bar(av)
        lines.append(
            f"  {diff:<10} {len(group):>4} {mn:>6.3f} {av:>6.3f} {mx:>6.3f} "
            f"{remote_n:>4}/{len(group):<2}  {bar}"
        )

    # -----------------------------------------------------------------------
    # Section 3: Difficulty separation
    # -----------------------------------------------------------------------
    easy_cs = [r["v2_complexity_score"] for r in by_diff["easy"]]
    medium_cs = [r["v2_complexity_score"] for r in by_diff["medium"]]
    hard_cs = [r["v2_complexity_score"] for r in by_diff["hard"]]

    sep_eh = _separation_score(easy_cs, hard_cs)
    sep_em = _separation_score(easy_cs, medium_cs)
    sep_mh = _separation_score(medium_cs, hard_cs)

    lines.append(_section("3. Difficulty Separation (Cohen's d)"))
    lines.append(
        "\n  Cohen's d measures the separation between difficulty groups."
        "\n  |d| > 0.8 = large effect (good separation)."
        "\n  |d| > 0.5 = medium effect."
        "\n  |d| < 0.2 = poor separation."
    )
    lines.append(f"\n  Easy   vs Hard   : d = {sep_eh:.3f}  "
                 f"{'[LARGE ✓]' if sep_eh > 0.8 else '[MEDIUM]' if sep_eh > 0.5 else '[POOR ✗]'}")
    lines.append(f"  Easy   vs Medium : d = {sep_em:.3f}  "
                 f"{'[LARGE ✓]' if sep_em > 0.8 else '[MEDIUM]' if sep_em > 0.5 else '[POOR ✗]'}")
    lines.append(f"  Medium vs Hard   : d = {sep_mh:.3f}  "
                 f"{'[LARGE ✓]' if sep_mh > 0.8 else '[MEDIUM]' if sep_mh > 0.5 else '[POOR ✗]'}")

    # -----------------------------------------------------------------------
    # Section 4: Per-category summary
    # -----------------------------------------------------------------------
    lines.append(_section("4. Routing Summary by Category"))
    lines.append(f"\n  {'Category':<20} {'N':>3} {'Avg Cmplx':>10} {'REMOTE':>7}  Group weights contributing most")
    lines.append(f"  {'─'*20} {'─'*3} {'─'*10} {'─'*7}  {'─'*30}")

    categories = sorted({r["category"] for r in results})
    for cat in categories:
        group = [r for r in results if r["category"] == cat]
        avg_c = _avg([r["v2_complexity_score"] for r in group])
        remote_n = sum(1 for r in group if r["v2_provider"] == PROVIDER_REMOTE)
        # Top contributing group
        group_sums: dict[str, float] = {}
        for r in group:
            cs = r["features"].get("category_scores", {})
            for k, v in cs.items():
                group_sums[k] = group_sums.get(k, 0.0) + v
        top_group = max(group_sums, key=group_sums.get) if group_sums else "—"
        lines.append(
            f"  {cat:<20} {len(group):>3} {avg_c:>10.4f} {remote_n:>4}/{len(group):<2}  {top_group}"
        )

    # -----------------------------------------------------------------------
    # Section 5: Misrouted prompts (hardest local / easiest remote)
    # -----------------------------------------------------------------------
    lines.append(_section("5. Potential Misrouting — Hard→LOCAL / Easy→REMOTE"))

    hard_local = [r for r in results if r["difficulty"] == "hard" and r["v2_provider"] == PROVIDER_LOCAL]
    easy_remote = [r for r in results if r["difficulty"] == "easy" and r["v2_provider"] == PROVIDER_REMOTE]

    if hard_local:
        lines.append(f"\n  Hard prompts routed LOCAL ({len(hard_local)}):")
        for r in sorted(hard_local, key=lambda x: x["v2_complexity_score"])[:10]:
            lines.append(
                f"    [{r['id']:20s}] cs={r['v2_complexity_score']:.3f}  "
                f"rs={r['v2_routing_score']:>3}  "
                f"{r['prompt'][:60]}…"
            )
    else:
        lines.append("\n  ✓ No hard prompts routed LOCAL.")

    if easy_remote:
        lines.append(f"\n  Easy prompts routed REMOTE ({len(easy_remote)}):")
        for r in sorted(easy_remote, key=lambda x: -x["v2_complexity_score"])[:10]:
            lines.append(
                f"    [{r['id']:20s}] cs={r['v2_complexity_score']:.3f}  "
                f"rs={r['v2_routing_score']:>3}  "
                f"{r['prompt'][:60]}…"
            )
    else:
        lines.append("\n  ✓ No easy prompts routed REMOTE.")

    # -----------------------------------------------------------------------
    # Section 6: Feature group importance
    # -----------------------------------------------------------------------
    lines.append(_section("6. Feature Group Importance (average scores across all prompts)"))
    group_names = ["technical", "reasoning", "task", "constraint", "context"]
    group_avgs = {}
    for gn in group_names:
        vals = [r["features"].get("category_scores", {}).get(gn, 0.0) for r in results]
        group_avgs[gn] = _avg(vals)

    lines.append(f"\n  {'Group':<15} {'Weight':>8} {'AvgScore':>9} {'WtdContrib':>11}  Bar")
    lines.append(f"  {'─'*15} {'─'*8} {'─'*9} {'─'*11}  {'─'*30}")
    from app.services.feature_extractor import _COMPLEXITY_WEIGHTS
    for gn in group_names:
        w = _COMPLEXITY_WEIGHTS[gn]
        av = group_avgs[gn]
        contrib = round(av * w, 4)
        bar = _bar(av)
        lines.append(f"  {gn:<15} {w:>8.2f} {av:>9.4f} {contrib:>11.4f}  {bar}")

    # -----------------------------------------------------------------------
    # Section 7: Sample traces (20 random prompts)
    # -----------------------------------------------------------------------
    sample_size = min(20, len(results))
    sample = random.sample(results, sample_size)
    # Sort by difficulty then complexity score for readability
    diff_order = {"easy": 0, "medium": 1, "hard": 2}
    sample.sort(key=lambda r: (diff_order.get(r["difficulty"], 3), r["v2_complexity_score"]))

    lines.append(_section(f"7. Detailed Routing Traces — {sample_size} Sampled Prompts"))

    for i, r in enumerate(sample, 1):
        f = r["features"]
        dbg = f.get("_debug", {})
        lines.append(f"\n  ── [{i:02d}/{sample_size}] {r['id']}  (category={r['category']}, difficulty={r['difficulty']})")
        lines.append(f"  Prompt: {r['prompt'][:100]}{'…' if len(r['prompt']) > 100 else ''}")
        lines.append(f"\n  {'Group':<17} {'Score':>6}   Bar")
        lines.append(f"  {'─'*17} {'─'*6}   {'─'*32}")
        cs_map = f.get("category_scores", {})
        for gn in group_names:
            score = cs_map.get(gn, 0.0)
            w = _COMPLEXITY_WEIGHTS[gn]
            lines.append(f"  {gn:<17} {score:>6.3f}   {_bar(score, 20)}  (wt={w:.2f})")

        lines.append(f"\n  Overall complexity_score : {f['complexity_score']:.4f}")
        lines.append(f"  Complexity label         : {f['complexity']}")
        lines.append(f"  reasoning_score (bridge) : {f['reasoning_score']}/10")
        lines.append(f"  V2 routing_score         : {r['v2_routing_score']}")
        lines.append(f"  V1 routing_score         : {r['v1_routing_score']}")
        lines.append(f"  Decision                 : {r['v2_provider'].upper()}  (threshold=25)")
        lines.append(f"  V1 decision              : {r['v1_provider'].upper()}")

        if verbose and dbg:
            # Technical domain hits
            td = dbg.get("stage_4a_technical", {}).get("domain_contributions", {})
            if td:
                lines.append(f"  Technical domains hit    : {', '.join(f'{k}={v:.2f}' for k,v in td.items())}")
            # Reasoning groups
            rg = dbg.get("stage_4b_reasoning", {}).get("group_contributions", {})
            if rg:
                lines.append(f"  Reasoning groups hit     : {', '.join(f'{k}={v:.2f}' for k,v in rg.items())}")
            # Task
            tsk = dbg.get("stage_4c_task", {})
            lines.append(f"  Task group               : {tsk.get('matched_group')} (base={tsk.get('base_score')}, mods={tsk.get('modifier_bonus')})")
            mods = tsk.get("fired_modifiers", [])
            if mods:
                lines.append(f"  Task modifiers           : {', '.join(mods)}")
            # Constraints
            cst = dbg.get("stage_4d_constraint", {}).get("fired_patterns", [])
            if cst:
                lines.append(f"  Constraint patterns      : {', '.join(cst[:6])}")
            # Context
            ctx = dbg.get("stage_4e_context", {})
            lines.append(f"  Context                  : tokens={f['estimated_input_tokens']} concept_hits={ctx.get('concept_hits')} density={ctx.get('concept_density'):.3f}")

        # Explain misrouting
        if r["difficulty"] == "hard" and r["v2_provider"] == PROVIDER_LOCAL:
            lines.append(f"\n  ⚠ MISROUTING ANALYSIS: Hard prompt routed LOCAL.")
            lines.append(f"    complexity_score={f['complexity_score']:.3f} is below the effective threshold (~0.50).")
            lines.append(f"    Dominant group: {max(cs_map, key=cs_map.get)} = {max(cs_map.values()):.3f}")
            lines.append(f"    This prompt may lack explicit technical keywords despite being hard.")
            lines.append(f"    Recommendation: verify the prompt relies on knowledge rather than reasoning.")
        elif r["difficulty"] == "easy" and r["v2_provider"] == PROVIDER_REMOTE:
            lines.append(f"\n  ⚠ MISROUTING ANALYSIS: Easy prompt routed REMOTE.")
            lines.append(f"    complexity_score={f['complexity_score']:.3f} exceeds the effective threshold (~0.50).")
            lines.append(f"    Dominant group: {max(cs_map, key=cs_map.get)} = {max(cs_map.values()):.3f}")
            lines.append(f"    Review whether the prompt is genuinely easy or mislabelled in the dataset.")

    # -----------------------------------------------------------------------
    # Section 8: ML compatibility notes
    # -----------------------------------------------------------------------
    lines.append(_section("8. ML Compatibility & Recommendations"))
    lines.append("""
  Feature design for supervised ML training:

  RECOMMENDED INPUT FEATURES (low collinearity)
  ─────────────────────────────────────────────
  technical_complexity   — domain-specific engineering depth [0, 1]
  reasoning_complexity   — analytical / logical demands      [0, 1]
  task_complexity        — verb-driven task ambition         [0, 1]
  constraint_complexity  — explicit requirements density     [0, 1]
  context_complexity     — concept density + token load      [0, 1]
  estimated_input_tokens — raw token count                   [int]
  contains_code          — structural code presence          [bool]
  contains_math          — math notation presence            [bool]

  AVOID AS DIRECT FEATURES (derived / redundant)
  ──────────────────────────────────────────────
  reasoning_score        — is round(complexity_score × 10); purely derived
  complexity             — is a bucketing of complexity_score; use score directly
  complexity_score       — is a weighted avg of the five group scores; use groups
  routing_score          — is computed by the router; this IS the label

  RECOMMENDED TARGET VARIABLE
  ───────────────────────────
  provider               — "local" | "remote"  (binary classification)
  OR
  complexity_score       — [0, 1]  (regression → threshold to classify)

  TRAINING DATA NOTES
  ───────────────────
  • The five group scores are designed to be orthogonal.  Run a correlation
    matrix before training and drop any pair with |r| > 0.85.
  • task_complexity and technical_complexity will correlate for coding tasks;
    monitor this during feature selection.
  • context_complexity has intentionally low variance (weight 0.08); it may
    not add predictive power beyond token count alone.
  • Difficulty labels in the current dataset serve as WEAK ground truth;
    prefer expert-annotated routing labels if available.
""")

    # -----------------------------------------------------------------------
    # Footer summary
    # -----------------------------------------------------------------------
    lines.append(_DBL)
    lines.append(f"  V1 routing: {v1_local} LOCAL / {v1_remote} REMOTE  (remote={_pct(v1_remote, total)})")
    lines.append(f"  V2 routing: {v2_local} LOCAL / {v2_remote} REMOTE  (remote={_pct(v2_remote, total)})")
    lines.append(f"  Hard prompts correctly routed REMOTE: "
                 f"{sum(1 for r in by_diff['hard'] if r['v2_provider']==PROVIDER_REMOTE)}"
                 f"/{len(by_diff['hard'])}")
    lines.append(f"  Easy prompts correctly routed LOCAL : "
                 f"{sum(1 for r in by_diff['easy'] if r['v2_provider']==PROVIDER_LOCAL)}"
                 f"/{len(by_diff['easy'])}")
    lines.append(f"  Easy→Hard separation (Cohen's d)    : {sep_eh:.3f}")
    lines.append(_DBL + "\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and calibrate the Feature Extraction Pipeline."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print full debug traces for each sampled prompt."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Write the report to this file path instead of stdout."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for prompt sampling (default: 42)."
    )
    args = parser.parse_args()

    report = run_validation(verbose=args.verbose, seed=args.seed)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to {out_path.resolve()}")
    else:
        print(report)


if __name__ == "__main__":
    main()
