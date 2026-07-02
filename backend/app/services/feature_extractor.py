"""Feature extraction module for the Hybrid Token Router.

Receives a raw user prompt and returns a structured feature dictionary that
the Routing Engine will later consume to decide which model to invoke.

This module is intentionally decision-free: it only measures and labels
properties of the prompt. Routing policy belongs in a separate engine.
"""

from __future__ import annotations

import math
import re
from typing import Final


# ---------------------------------------------------------------------------
# Public constants – task type literals kept in one place to avoid typos.
# ---------------------------------------------------------------------------

TASK_CODING: Final = "coding"
TASK_MATHEMATICS: Final = "mathematics"
TASK_REASONING: Final = "reasoning"
TASK_SUMMARIZATION: Final = "summarization"
TASK_TRANSLATION: Final = "translation"
TASK_CREATIVE_WRITING: Final = "creative_writing"
TASK_PLANNING: Final = "planning"
TASK_QUESTION_ANSWERING: Final = "question_answering"
TASK_GENERAL: Final = "general"

COMPLEXITY_EASY: Final = "easy"
COMPLEXITY_MEDIUM: Final = "medium"
COMPLEXITY_HARD: Final = "hard"


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Estimate the number of input tokens using a character-length heuristic.

    The approximation ``ceil(len(text) / 4)`` matches the average English
    token size observed in byte-pair-encoding tokenizers and is cheap to
    compute.  The isolation of this function means a real tokenizer (e.g.
    ``tiktoken`` or ``transformers.AutoTokenizer``) can be swapped in later
    without touching the public interface.

    Args:
        text: The raw prompt string.

    Returns:
        An integer token count estimate.
    """
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Primitive boolean detectors
# ---------------------------------------------------------------------------

# Code: fenced blocks, inline backticks, or common programming keywords.
_CODE_PATTERN: Final = re.compile(
    r"```[\s\S]*?```"                           # fenced code block
    r"|`[^`\n]+`"                               # inline code
    r"|\b(def |class |import |function |return |var |let |const |#include"
    r"|public static|void |int |float |bool |lambda |async def |await )\b",
    re.IGNORECASE,
)

# Mathematics: LaTeX delimiters, operators, or math keywords.
_MATH_PATTERN: Final = re.compile(
    r"\$\$[\s\S]*?\$\$"                         # display LaTeX
    r"|\$[^\$\n]+\$"                            # inline LaTeX
    r"|\b(integral|derivative|matrix|equation|solve|factorial|logarithm"
    r"|calculus|algebra|geometry|trigonometry|eigenvalue|polynomial)\b"
    r"|[=<>≤≥≠±∑∏√∫∂∇]"                        # math symbols & operators
    r"|\b\d+\s*[\+\-\*\/\^]\s*\d+",            # bare arithmetic expression
    re.IGNORECASE,
)

# JSON: object or array literals (minimum two characters inside braces/brackets).
_JSON_PATTERN: Final = re.compile(
    r"\{[^{}]{2,}\}"                            # JSON-like object
    r"|\[[^\[\]]{2,}\]",                        # JSON-like array
)

# Markdown: headers, bold/italic, list markers, horizontal rules.
_MARKDOWN_PATTERN: Final = re.compile(
    r"^#{1,6}\s"                                # ATX heading
    r"|^\*{3,}$|^-{3,}$"                        # horizontal rule
    r"|\*\*[^*]+\*\*"                           # bold
    r"|__[^_]+__"                               # bold (underscores)
    r"|\*[^*]+\*"                               # italic
    r"|_[^_]+_"                                 # italic (underscores)
    r"|^\s*[-*+]\s"                             # unordered list item
    r"|^\s*\d+\.\s"                             # ordered list item
    r"|\[.+?\]\(.+?\)",                         # Markdown link
    re.MULTILINE,
)

# Numbers: standalone integers or decimals, not part of a larger word.
_NUMBERS_PATTERN: Final = re.compile(r"\b\d+(?:\.\d+)?\b")

# Questions: ends with "?" or starts/contains interrogative words.
_QUESTION_PATTERN: Final = re.compile(
    r"\?$"
    r"|\?"
    r"|\b(what|why|how|when|where|which|who|whom|whose|can you|could you"
    r"|would you|is it|are there|do you|does it|explain|tell me)\b",
    re.IGNORECASE,
)


def _contains_code(prompt: str) -> bool:
    """Return True when the prompt appears to contain source code."""
    return bool(_CODE_PATTERN.search(prompt))


def _contains_math(prompt: str) -> bool:
    """Return True when the prompt appears to contain mathematical content."""
    return bool(_MATH_PATTERN.search(prompt))


def _contains_json(prompt: str) -> bool:
    """Return True when the prompt appears to contain JSON-like structures."""
    return bool(_JSON_PATTERN.search(prompt))


def _contains_markdown(prompt: str) -> bool:
    """Return True when the prompt appears to use Markdown formatting."""
    return bool(_MARKDOWN_PATTERN.search(prompt))


def _contains_numbers(prompt: str) -> bool:
    """Return True when the prompt contains at least one standalone number."""
    return bool(_NUMBERS_PATTERN.search(prompt))


def _contains_question(prompt: str) -> bool:
    """Return True when the prompt reads as a question."""
    return bool(_QUESTION_PATTERN.search(prompt))


# ---------------------------------------------------------------------------
# Task-type classification – deterministic keyword heuristics
# ---------------------------------------------------------------------------

# Each entry is (task_type, compiled_pattern). Checked in priority order.
_TASK_PATTERNS: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        TASK_CODING,
        re.compile(
            r"\b(code|program|script|function|class|debug|implement|algorithm"
            r"|api|library|refactor|compile|syntax|error|bug|unit test"
            r"|import|module|package|repository|github|git)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_MATHEMATICS,
        re.compile(
            r"\b(calculate|compute|solve|equation|integral|derivative|matrix"
            r"|factorial|prime|fibonacci|probability|statistics|percentage"
            r"|algebra|geometry|calculus|proof|theorem|formula|math)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_REASONING,
        re.compile(
            r"\b(why|reason|analyze|analyse|explain|cause|effect|impact|infer"
            r"|deduce|conclude|argument|logic|because|therefore|hence|compare"
            r"|contrast|evaluate|assess|critique|justify|implication)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_SUMMARIZATION,
        re.compile(
            r"\b(summarize|summarise|summary|tldr|brief|overview|condense"
            r"|shorten|abstract|synopsis|recap|key points|highlights)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_TRANSLATION,
        re.compile(
            r"\b(translate|translation|traduction|übersetzen|traduzir|翻译"
            r"|翻訳|traducir|переводить|in (spanish|french|german|japanese"
            r"|chinese|portuguese|italian|russian|arabic|hindi|korean))\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_CREATIVE_WRITING,
        re.compile(
            r"\b(write a (story|poem|song|essay|novel|script|blog|letter"
            r"|article)|creative|fiction|narrative|plot|character|dialogue"
            r"|rhyme|stanza|verse|haiku|limerick|short story)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_PLANNING,
        re.compile(
            r"\b(plan|roadmap|schedule|strategy|steps to|how to|checklist"
            r"|outline|agenda|milestone|goal|objective|task list|project"
            r"|timeline|action items|organize|prioritize)\b",
            re.IGNORECASE,
        ),
    ),
    (
        TASK_QUESTION_ANSWERING,
        re.compile(
            r"\b(what is|what are|who is|who are|when did|where is|where are"
            r"|how does|how do|how many|how much|define|definition of"
            r"|meaning of|tell me about)\b",
            re.IGNORECASE,
        ),
    ),
]


def _detect_task_type(prompt: str) -> str:
    """Identify the dominant task type using ordered keyword heuristics.

    Patterns are checked from highest to lowest specificity.  The first match
    wins.  Falls back to ``general`` when no pattern matches.

    Args:
        prompt: The raw user prompt.

    Returns:
        A task-type string constant defined at module level.
    """
    for task_type, pattern in _TASK_PATTERNS:
        if pattern.search(prompt):
            return task_type
    return TASK_GENERAL


# ---------------------------------------------------------------------------
# Reasoning score
# ---------------------------------------------------------------------------

# Keywords that signal explicit reasoning demands.
_REASONING_KEYWORDS: Final = re.compile(
    r"\b(because|therefore|hence|thus|since|given that|it follows|consequently"
    r"|analyze|analyse|evaluate|critique|compare|contrast|justify|deduce"
    r"|infer|conclude|explain why|explain how|step by step|chain of thought"
    r"|reason|argument|logic|proof|implication|cause|effect)\b",
    re.IGNORECASE,
)

# Phrases that add explicit constraints ("must", "should not", "only if", …).
_CONSTRAINT_PATTERN: Final = re.compile(
    r"\b(must|should not|do not|must not|only if|at least|at most|exactly"
    r"|no more than|no less than|without|except|unless|provided that"
    r"|assuming|given that|such that|subject to|constraint)\b",
    re.IGNORECASE,
)


def _count_reasoning_keywords(prompt: str) -> int:
    """Count distinct reasoning-keyword matches in the prompt."""
    return len(_REASONING_KEYWORDS.findall(prompt))


def _count_constraints(prompt: str) -> int:
    """Count constraint phrases that indicate multi-step demands."""
    return len(_CONSTRAINT_PATTERN.findall(prompt))


def _length_score(prompt: str) -> int:
    """Map prompt character length to a 0–3 sub-score.

    Bands:
        0–200   → 0
        201–500 → 1
        501–900 → 2
        900+    → 3
    """
    length = len(prompt)
    if length > 900:
        return 3
    if length > 500:
        return 2
    if length > 200:
        return 1
    return 0


def _keyword_score(prompt: str) -> int:
    """Map reasoning keyword count to a 0–3 sub-score.

    Bands:
        0      → 0
        1–2    → 1
        3–5    → 2
        6+     → 3
    """
    count = _count_reasoning_keywords(prompt)
    if count >= 6:
        return 3
    if count >= 3:
        return 2
    if count >= 1:
        return 1
    return 0


def _constraint_score(prompt: str) -> int:
    """Map constraint count to a 0–2 sub-score.

    Bands:
        0    → 0
        1–2  → 1
        3+   → 2
    """
    count = _count_constraints(prompt)
    if count >= 3:
        return 2
    if count >= 1:
        return 1
    return 0


def _coding_complexity_score(prompt: str) -> int:
    """Add 1 point when the prompt contains code, 0 otherwise."""
    return 1 if _contains_code(prompt) else 0


def _math_indicator_score(prompt: str) -> int:
    """Add 1 point when the prompt contains mathematical content, 0 otherwise."""
    return 1 if _contains_math(prompt) else 0


def _compute_reasoning_score(prompt: str) -> int:
    """Combine sub-scores into a final reasoning score in [0, 10].

    Sub-score breakdown (maximum contributions):
        - Prompt length          : 0–3
        - Reasoning keywords     : 0–3
        - Explicit constraints   : 0–2
        - Code presence          : 0–1
        - Math presence          : 0–1
        Total possible           : 10

    Each component is explainable and independently adjustable.

    Args:
        prompt: The raw user prompt.

    Returns:
        An integer in the inclusive range [0, 10].
    """
    raw = (
        _length_score(prompt)
        + _keyword_score(prompt)
        + _constraint_score(prompt)
        + _coding_complexity_score(prompt)
        + _math_indicator_score(prompt)
    )
    return min(raw, 10)


# ---------------------------------------------------------------------------
# Complexity mapping
# ---------------------------------------------------------------------------

def _map_complexity(reasoning_score: int) -> str:
    """Convert a reasoning score to a human-readable complexity label.

    Mapping:
        0–3  → ``easy``
        4–7  → ``medium``
        8–10 → ``hard``

    Args:
        reasoning_score: Integer in [0, 10] from ``_compute_reasoning_score``.

    Returns:
        One of the string constants ``easy``, ``medium``, or ``hard``.
    """
    if reasoning_score >= 8:
        return COMPLEXITY_HARD
    if reasoning_score >= 4:
        return COMPLEXITY_MEDIUM
    return COMPLEXITY_EASY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(prompt: str) -> dict:
    """Extract structured features from a raw user prompt.

    This function is the sole public entry point of this module.  It derives
    every feature deterministically from the prompt text without calling any
    external service or language model.

    The returned dictionary is designed to be directly consumed by the
    Routing Engine.  The engine is responsible for any model-selection
    decision; this function must remain decision-free.

    Args:
        prompt: The raw user prompt string.

    Returns:
        A flat dictionary with the following keys:

        ``prompt_length`` (int):
            Number of characters in the prompt.

        ``word_count`` (int):
            Number of whitespace-separated tokens.

        ``estimated_input_tokens`` (int):
            Lightweight token count approximation (``ceil(len / 4)``).

        ``contains_code`` (bool):
            True when fenced code blocks, inline code, or programming
            keywords are detected.

        ``contains_math`` (bool):
            True when LaTeX delimiters, math symbols, or math vocabulary
            are detected.

        ``contains_json`` (bool):
            True when JSON-like object or array literals are detected.

        ``contains_markdown`` (bool):
            True when Markdown formatting elements are detected.

        ``contains_numbers`` (bool):
            True when at least one standalone number appears.

        ``contains_question`` (bool):
            True when the prompt contains a question mark or interrogative
            phrasing.

        ``task_type`` (str):
            The dominant task category detected by keyword heuristics.
            One of: ``coding``, ``mathematics``, ``reasoning``,
            ``summarization``, ``translation``, ``creative_writing``,
            ``planning``, ``question_answering``, ``general``.

        ``reasoning_score`` (int):
            Explainable integer score in [0, 10] measuring how much
            multi-step reasoning the prompt demands.

        ``complexity`` (str):
            Human-readable label derived from ``reasoning_score``.
            One of: ``easy`` (0–3), ``medium`` (4–7), ``hard`` (8–10).

    Example::

        >>> features = extract_features("What is 2 + 2?")
        >>> features["task_type"]
        'question_answering'
        >>> features["complexity"]
        'easy'
    """
    reasoning_score = _compute_reasoning_score(prompt)

    return {
        "prompt_length": len(prompt),
        "word_count": len(prompt.split()),
        "estimated_input_tokens": _estimate_tokens(prompt),
        "contains_code": _contains_code(prompt),
        "contains_math": _contains_math(prompt),
        "contains_json": _contains_json(prompt),
        "contains_markdown": _contains_markdown(prompt),
        "contains_numbers": _contains_numbers(prompt),
        "contains_question": _contains_question(prompt),
        "task_type": _detect_task_type(prompt),
        "reasoning_score": reasoning_score,
        "complexity": _map_complexity(reasoning_score),
    }
