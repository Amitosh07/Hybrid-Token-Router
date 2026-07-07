"""Feature Extraction Pipeline for the Hybrid Token Router — Version 2.

This module is a ground-up redesign.  The public function ``extract_features``
preserves its original signature and all output keys so the router, benchmark,
and chat endpoints continue working without modification.

Architecture
------------
The prompt flows through five sequential stages.  Each stage has one
responsibility and passes its output forward as a plain dictionary.

    Stage 1  Normalization
        Produces lowercase, sentence-split, and word-tokenised views of the
        prompt.  All later stages operate on these canonical forms.

    Stage 2  Lexical Features
        Surface statistics: character count, word count, sentence count,
        vocabulary richness, average word length, token estimate.

    Stage 3  Structural Features
        Presence of fenced code, inline code, LaTeX math, JSON, Markdown lists,
        numbered lists, and whether the prompt is imperative or interrogative.

    Stage 4  Semantic Feature Groups  (five independent groups, 0→1 each)
        4a  Technical Complexity   – domain-specific engineering depth.
        4b  Reasoning Complexity   – analytical, logical, and causal demands.
        4c  Task Complexity        – verb-driven task scope and ambition.
        4d  Constraint Complexity  – explicit requirements, restrictions,
                                     structured outputs, and dependencies.
        4e  Context Complexity     – concept density and token load.
            This group is intentionally a WEAK prior.  A short technically
            hard prompt must score higher than a long but trivial one.

    Stage 5  Complexity Aggregation
        A single weighted combination of the five group scores produces
        ``complexity_score`` in [0, 1], which then drives ``reasoning_score``
        and the complexity label.

Group Weights (single source of truth — adjust here only)
----------------------------------------------------------
    Technical   0.30  – strongest discriminator: hard engineering tasks
    Reasoning   0.25  – strong discriminator: analytical / logical tasks
    Task        0.25  – captures the ambition and scope of the request
    Constraint  0.12  – explicit multi-requirement density signals real effort
    Context     0.08  – weak length / density prior

Router bridge
-------------
The existing router reads ``reasoning_score`` (0–10) as its dominant signal
(weight ×5) against a threshold of 25.  The new pipeline sets

    reasoning_score = round(complexity_score × 10)

so that complexity_score ≥ 0.50 → reasoning_score ≥ 5 → routing_score ≥ 25
→ REMOTE.  The router itself is unchanged.

Backward-compatible keys (preserved from v1)
--------------------------------------------
    prompt_length, word_count, estimated_input_tokens,
    contains_code, contains_math, contains_json, contains_markdown,
    contains_numbers, contains_question, task_type, reasoning_score,
    complexity

New keys added by v2
--------------------
    technical_complexity, reasoning_complexity, task_complexity,
    constraint_complexity, context_complexity, complexity_score,
    category_scores, task_labels

Optional debug key (only when ``debug=True``)
---------------------------------------------
    _debug  –  full stage-by-stage trace for every routing decision

ML compatibility notes
----------------------
All five group scores are orthogonal by design to minimise multi-collinearity.
``reasoning_score`` is intentionally redundant (it is a linear function of
``complexity_score``); it exists only to bridge the unchanged router API.
Future training pipelines should use the five group scores directly as features
and treat ``reasoning_score`` as a derived target variable.
"""

from __future__ import annotations

import math
import re
from typing import Final

# ===========================================================================
# Public constants – preserved from v1 (task type and complexity labels)
# ===========================================================================

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

# ===========================================================================
# Stage 5 – Complexity Aggregation Weights  (single source of truth)
# ===========================================================================
# Increasing a weight amplifies that group's influence on the final score.
# All weights must sum to 1.0.

_COMPLEXITY_WEIGHTS: Final[dict[str, float]] = {
    "technical":  0.30,
    "reasoning":  0.25,
    "task":       0.25,
    "constraint": 0.12,
    "context":    0.08,
}

assert abs(sum(_COMPLEXITY_WEIGHTS.values()) - 1.0) < 1e-9, (
    "Complexity weights must sum to 1.0"
)

# ===========================================================================
# Stage 1 – Normalization
# ===========================================================================

_RE_WHITESPACE = re.compile(r"\s+")
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_RE_WORD_TOKEN = re.compile(r"\b[a-z][a-z0-9\-]*\b")


def _normalize(prompt: str) -> dict:
    """Produce canonical text forms for all downstream stages.

    Returns
    -------
    raw:       original prompt string (unchanged)
    lower:     lowercased string for case-insensitive matching
    stripped:  whitespace-collapsed single-line version
    sentences: list of sentence strings (split on terminal punctuation)
    words:     list of lowercase word tokens (alpha-numeric only)
    """
    raw = prompt
    lower = prompt.lower()
    stripped = _RE_WHITESPACE.sub(" ", prompt).strip()
    sentences = [s.strip() for s in _RE_SENTENCE_SPLIT.split(stripped) if s.strip()]
    words = _RE_WORD_TOKEN.findall(lower)
    return {
        "raw": raw,
        "lower": lower,
        "stripped": stripped,
        "sentences": sentences,
        "words": words,
    }


# ===========================================================================
# Stage 2 – Lexical Features
# ===========================================================================

def _lexical_features(norm: dict) -> dict:
    """Extract surface-level statistics from the normalized prompt.

    Returns
    -------
    prompt_length:          character count of raw prompt
    word_count:             whitespace-separated token count
    sentence_count:         number of detected sentences
    unique_word_count:      number of distinct lower-cased word tokens
    unique_word_ratio:      unique_word_count / max(word_count, 1)
    avg_word_length:        mean characters per word token
    estimated_input_tokens: ceil(len / 4) — cheap BPE approximation
    """
    raw = norm["raw"]
    words = norm["words"]
    wc = len(raw.split())
    uniq = len(set(words))
    avg_len = (sum(len(w) for w in words) / max(len(words), 1)) if words else 0.0
    return {
        "prompt_length": len(raw),
        "word_count": wc,
        "sentence_count": len(norm["sentences"]),
        "unique_word_count": uniq,
        "unique_word_ratio": round(uniq / max(len(words), 1), 4),
        "avg_word_length": round(avg_len, 2),
        "estimated_input_tokens": math.ceil(len(raw) / 4),
    }


# ===========================================================================
# Stage 3 – Structural Features
# ===========================================================================

# Fenced code blocks (``` or ~~~)
_RE_CODE_FENCE = re.compile(r"```[\s\S]*?```|~~~[\s\S]*?~~~", re.MULTILINE)
# Inline code: `something`
_RE_INLINE_CODE = re.compile(r"`[^`\n]+`")
# Programming keywords appearing as tokens in the prompt text
# (kept minimal — we detect code presence, not coding difficulty)
_RE_CODE_KEYWORD = re.compile(
    r"\b(def |class |import |function\(|#include|public static|void |"
    r"return |lambda |async def |await |var |let |const )\b",
    re.IGNORECASE,
)
# LaTeX display math
_RE_MATH_DISPLAY = re.compile(r"\$\$[\s\S]*?\$\$")
# LaTeX inline math
_RE_MATH_INLINE = re.compile(r"\$[^\$\n]{1,80}\$")
# Math vocabulary (not in LaTeX delimiters)
_RE_MATH_VOCAB = re.compile(
    r"\b(integral|derivative|matrix|equation|eigenvalue|polynomial|"
    r"factorial|logarithm|calculus|algebra|trigonometry|probability|"
    r"statistics|differential|theorem|proof|determinant)\b",
    re.IGNORECASE,
)
# Inline math operators
_RE_MATH_OPERATORS = re.compile(r"[=<>≤≥≠±∑∏√∫∂∇]")
# Bare arithmetic: "3 + 5", "x² - 4"
_RE_BARE_ARITHMETIC = re.compile(r"\b\d+\s*[+\-*/^]\s*\d+")
# JSON-like object / array (≥3 chars inside braces/brackets)
_RE_JSON = re.compile(r"\{[^{}]{3,}\}|\[[^\[\]]{3,}\]")
# Markdown headers
_RE_MD_HEADER = re.compile(r"^#{1,6}\s", re.MULTILINE)
# Markdown bold / italic
_RE_MD_EMPHASIS = re.compile(r"\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_")
# Numbered list items (1. / 1) style)
_RE_NUMBERED_LIST = re.compile(r"^\s*\d+[.)]\s", re.MULTILINE)
# Bullet list items
_RE_BULLET_LIST = re.compile(r"^\s*[-*+•]\s", re.MULTILINE)
# Standalone numbers
_RE_NUMBERS = re.compile(r"\b\d+(?:\.\d+)?\b")
# Question marks anywhere in the prompt
_RE_QUESTION_MARK = re.compile(r"\?")


def _structural_features(norm: dict) -> dict:
    """Detect structural elements in the prompt.

    Returns boolean flags and counts.  These feed both the router's boolean
    weights (contains_code, contains_math, contains_json) and the constraint
    and context complexity detectors.
    """
    raw = norm["raw"]

    code_fences = _RE_CODE_FENCE.findall(raw)
    inline_code = _RE_INLINE_CODE.findall(raw)
    code_keywords = _RE_CODE_KEYWORD.findall(raw)

    has_math = bool(
        _RE_MATH_DISPLAY.search(raw)
        or _RE_MATH_INLINE.search(raw)
        or _RE_MATH_VOCAB.search(raw)
        or _RE_MATH_OPERATORS.search(raw)
        or _RE_BARE_ARITHMETIC.search(raw)
    )

    numbered_items = len(_RE_NUMBERED_LIST.findall(raw))
    bullet_items = len(_RE_BULLET_LIST.findall(raw))

    return {
        "contains_code": bool(code_fences or inline_code or code_keywords),
        "code_fence_count": len(code_fences),
        "contains_math": has_math,
        "contains_json": bool(_RE_JSON.search(raw)),
        "contains_markdown": bool(
            _RE_MD_HEADER.search(raw) or _RE_MD_EMPHASIS.search(raw)
        ),
        "contains_numbers": bool(_RE_NUMBERS.search(raw)),
        "contains_question": bool(_RE_QUESTION_MARK.search(raw)),
        "question_count": len(_RE_QUESTION_MARK.findall(raw)),
        "numbered_list_items": numbered_items,
        "bullet_list_items": bullet_items,
        "list_item_count": numbered_items + bullet_items,
    }


# ===========================================================================
# Stage 4a – Technical Complexity
# ===========================================================================
# Each domain entry: (base_weight, tier_a_phrases, tier_b_phrases)
#
# Tier A phrases are specific technical concepts that strongly indicate a hard
# problem (each hit is worth 1.0 equivalent units).
# Tier B phrases are general technical terms that indicate the domain is present
# but do not by themselves indicate depth (each hit is worth 0.4 units).
#
# Within a domain: raw_score = min(1.0, tier_a_hits + tier_b_hits × 0.4)
# Domain contribution: raw_score × base_weight
# Final score: weighted top-3 average × breadth multiplier (see function).

_TECH_DOMAINS: Final[dict[str, tuple[float, list[str], list[str]]]] = {
    "algorithms": (
        0.90,
        # Tier A — specific algorithms and complexity theory
        [
            "dijkstra", "bellman-ford", "floyd-warshall", "a-star",
            "dynamic programming", "memoization", "tabulation",
            "topological sort", "topological sorting",
            "minimum spanning tree", "kruskal", "prim",
            "knapsack", "longest common subsequence", "edit distance",
            "divide and conquer", "backtracking", "branch and bound",
            "ford-fulkerson", "max flow", "min cut", "network flow",
            "np-hard", "np-complete", "polynomial time", "p vs np",
            "amortized", "recurrence relation",
        ],
        # Tier B — general algorithm vocabulary
        [
            "algorithm", "time complexity", "space complexity",
            "big-o", "big o", "o(n)", "o(log n)", "o(n log n)",
            "breadth-first search", "depth-first search", "bfs", "dfs",
            "binary search", "merge sort", "quicksort", "heap sort",
            "greedy", "heuristic", "shortest path",
        ],
    ),
    "data_structures": (
        0.75,
        # Tier A
        [
            "avl tree", "red-black tree", "b-tree", "b+ tree",
            "trie", "patricia trie", "segment tree", "fenwick tree",
            "bloom filter", "skip list", "van emde boas",
            "disjoint set", "union-find", "suffix tree", "suffix array",
            "fibonacci heap",
        ],
        # Tier B
        [
            "binary tree", "binary search tree", "heap", "priority queue",
            "linked list", "doubly linked list", "hash table", "hash map",
            "adjacency list", "adjacency matrix", "graph",
            "stack", "queue", "deque", "monotonic stack",
            "data structure",
        ],
    ),
    "concurrency": (
        0.95,
        # Tier A — concurrency correctness concepts
        [
            "thread-safe", "thread safe",
            "deadlock", "race condition", "livelock", "starvation",
            "memory barrier", "compare-and-swap", "cas",
            "lock-free", "wait-free", "linearizable", "linearizability",
            "producer-consumer", "producer consumer",
            "readers-writers", "dining philosophers",
            "actor model", "communicating sequential processes",
            "transactional memory", "software transactional",
        ],
        # Tier B
        [
            "mutex", "semaphore", "lock", "synchronized", "spinlock",
            "thread", "threading", "concurrent", "parallelism",
            "async", "await", "coroutine", "event loop",
            "atomic", "volatile", "synchronization",
        ],
    ),
    "system_design": (
        0.90,
        # Tier A
        [
            "microservices", "service-oriented architecture",
            "distributed system", "distributed systems",
            "consensus algorithm", "raft", "paxos", "two-phase commit",
            "saga pattern", "cap theorem", "eventual consistency",
            "strong consistency", "linearizable",
            "event sourcing", "cqrs",
            "service mesh", "api gateway", "circuit breaker",
            "consistent hashing", "gossip protocol",
            "vector clock", "logical clock",
        ],
        # Tier B
        [
            "load balancer", "load balancing", "kafka", "rabbitmq",
            "distributed", "scalable", "high availability",
            "fault tolerant", "replication", "sharding", "partitioning",
            "caching", "cdn", "rate limiting", "message queue",
            "event bus", "pub-sub", "fanout",
        ],
    ),
    "database": (
        0.75,
        # Tier A
        [
            "normalization", "denormalization", "acid properties",
            "isolation level", "mvcc", "write-ahead log",
            "row-level security", "multi-tenant", "database sharding",
            "query optimization", "execution plan", "index strategy",
            "referential integrity", "foreign key constraint",
            "serializability", "snapshot isolation",
        ],
        # Tier B
        [
            "database schema", "sql", "nosql", "indexing",
            "transaction", "join", "query", "stored procedure",
            "orm", "connection pool", "relational database",
            "ddl", "primary key", "foreign key",
        ],
    ),
    "security": (
        0.85,
        # Tier A
        [
            "oauth 2.0", "openid connect", "saml", "kerberos",
            "sql injection", "cross-site scripting", "xss",
            "csrf", "ssrf", "xxe", "buffer overflow",
            "privilege escalation", "threat model",
            "zero trust", "penetration testing",
            "diffie-hellman", "elliptic curve", "key exchange",
            "certificate authority", "public key infrastructure",
            "authentication bypass", "injection attack",
        ],
        # Tier B
        [
            "authentication", "authorization", "encryption", "hashing",
            "bcrypt", "tls", "ssl", "firewall", "jwt", "token",
            "rbac", "acl", "permissions", "secure", "vulnerability",
        ],
    ),
    "machine_learning": (
        0.85,
        # Tier A
        [
            "backpropagation", "gradient descent", "stochastic gradient",
            "attention mechanism", "self-attention", "multi-head attention",
            "transformer architecture", "bert", "gpt",
            "reinforcement learning", "policy gradient", "q-learning",
            "variational autoencoder", "generative adversarial network",
            "embedding space", "fine-tuning", "rlhf",
            "cross-entropy loss", "kl divergence", "regularization",
            "convolutional neural network", "recurrent neural network",
            "lstm", "gru",
        ],
        # Tier B
        [
            "neural network", "deep learning", "machine learning",
            "model training", "overfitting", "underfitting",
            "training data", "validation set", "test set",
            "classification", "regression", "clustering",
            "feature engineering", "hyperparameter", "embedding",
        ],
    ),
    "cloud_devops": (
        0.70,
        # Tier A
        [
            "kubernetes", "container orchestration", "helm chart",
            "terraform", "infrastructure as code", "pulumi",
            "blue-green deployment", "canary release",
            "service discovery", "sidecar pattern",
            "gitops", "argo cd", "continuous delivery",
        ],
        # Tier B
        [
            "docker", "container", "serverless", "lambda",
            "aws", "azure", "gcp", "cloud",
            "ci/cd", "pipeline", "devops", "deployment",
            "autoscaling", "health check",
        ],
    ),
    "api_networking": (
        0.65,
        # Tier A
        [
            "grpc", "graphql", "websocket", "server-sent events",
            "openapi specification", "idempotent", "api versioning",
            "oauth flow", "webhook", "long polling",
            "protocol buffer", "protobuf", "thrift",
            "dependency injection", "inversion of control",
        ],
        # Tier B
        [
            "rest", "restful", "api", "http", "endpoint",
            "request", "response", "middleware",
            "pagination", "rate limit", "http status",
        ],
    ),
}


def _hits_to_unit(tier_a_hits: int, tier_b_hits: int) -> float:
    """Convert tier hit counts to a [0, 1] unit score.

    Tier A hits are worth 1.0 units each.
    Tier B hits are worth 0.4 units each.
    The combined score is clamped to 1.0.

    Design: saturation is intentional — a prompt mentioning five Tier A
    keywords should not score 5× higher than one mentioning two.
    """
    return min(1.0, tier_a_hits * 1.0 + tier_b_hits * 0.4)


def _technical_complexity(norm: dict) -> tuple[float, dict[str, float]]:
    """Compute Technical Complexity score in [0, 1].

    Detects domain-specific engineering depth across nine domains.
    A short prompt containing 'Dijkstra' scores higher than a long prompt
    containing only 'function'.

    Returns
    -------
    score:        float in [0, 1]
    domain_hits:  dict mapping domain name → domain contribution score
    """
    lower = norm["lower"]
    domain_contributions: dict[str, float] = {}

    for domain, (base_w, tier_a, tier_b) in _TECH_DOMAINS.items():
        a_hits = sum(1 for kw in tier_a if kw in lower)
        b_hits = sum(1 for kw in tier_b if kw in lower)
        if a_hits == 0 and b_hits == 0:
            continue
        unit = _hits_to_unit(a_hits, b_hits)
        domain_contributions[domain] = round(unit * base_w, 4)

    if not domain_contributions:
        return 0.0, {}

    # Use the top-3 domain scores to avoid dilution from many weak signals.
    top_scores = sorted(domain_contributions.values(), reverse=True)[:3]
    n = len(top_scores)
    base_score = sum(top_scores) / n

    # Breadth multiplier: touching 2+ domains indicates a harder problem.
    breadth = len(domain_contributions)
    breadth_multiplier = min(1.25, 1.0 + (breadth - 1) * 0.08)

    score = round(min(1.0, base_score * breadth_multiplier), 4)
    return score, domain_contributions


# ===========================================================================
# Stage 4b – Reasoning Complexity
# ===========================================================================
# Each group entry: (base_weight, [phrases])
#
# Scoring within a group:
#   1 hit  → 0.35 × base_weight
#   2 hits → 0.65 × base_weight
#   3+ hits → 1.0  × base_weight

_REASONING_GROUPS: Final[dict[str, tuple[float, list[str]]]] = {
    "analytical": (
        0.70,
        [
            "analyze", "analyse", "evaluate", "assess", "examine",
            "investigate", "critique", "compare", "contrast",
            "pros and cons", "advantages and disadvantages",
            "trade-off", "trade-offs", "weighing", "weigh the options",
        ],
    ),
    "logical_deduction": (
        0.90,
        [
            "deduce", "infer", "conclude", "prove", "disprove",
            "verify", "demonstrate that", "show that",
            "it follows that", "necessarily", "if and only if",
            "necessary and sufficient",
            "logical fallacy", "counter-example",
            "inconsistent", "consistent with",
            "who is telling", "who is lying", "who committed",
            "exactly one", "exactly two", "at least one of",
            "if all", "if none", "if some",
            "determine who", "determine which", "determine whether",
        ],
    ),
    "optimization": (
        0.85,
        [
            "optimize", "minimise", "maximize", "minimise",
            "best strategy", "most efficient", "optimal solution",
            "minimum cost", "maximum throughput", "minimize latency",
            "best approach given", "under the constraint",
            "subject to", "pareto", "trade-off between",
            "opportunity cost",
        ],
    ),
    "mathematical_reasoning": (
        0.90,
        [
            "proof by induction", "proof by contradiction",
            "mathematical induction",
            "derive the formula", "derive an expression",
            "calculate the probability", "expected value",
            "bayes theorem", "bayes",
            "generating function", "fourier transform", "taylor series",
            "eigenvalue", "determinant", "integration by parts",
            "differential equation", "convergence",
            "show all steps", "show all working",
        ],
    ),
    "multi_step_reasoning": (
        0.75,
        [
            "step by step", "step-by-step",
            "walk me through", "explain the reasoning behind",
            "chain of thought", "reason through",
            "trace through", "explain each step",
            "describe each phase", "in what order",
            "sequence of steps",
        ],
    ),
    "causal_systemic": (
        0.70,
        [
            "why does", "why would", "what would happen if",
            "consequences of", "second-order effect", "second order",
            "root cause", "cause and effect",
            "impact of", "implications of",
            "if we change", "counterfactual",
            "feedback loop", "cascade",
        ],
    ),
    "ethical_frameworks": (
        0.75,
        [
            "ethical", "utilitarian", "deontological",
            "rights-based", "moral framework",
            "fairness", "justice", "equity",
            "stakeholder", "bias",
            "unintended consequence", "responsible",
            "ethical concern", "bioethics",
        ],
    ),
    "comparative_assessment": (
        0.70,
        [
            "compare and contrast",
            "similarities and differences",
            "dimensions", "across four", "across three", "across five",
            "each system", "each approach",
            "historical track record", "balanced assessment",
            "most effectively", "least effectively",
        ],
    ),
}


def _hits_to_group_score(hits: int) -> float:
    """Map hit count to a [0, 1] base score for a reasoning group.

    1 hit  → 0.35  (present but not dense)
    2 hits → 0.65  (clearly present)
    3+ hits → 1.0  (dominant signal)
    """
    if hits <= 0:
        return 0.0
    if hits == 1:
        return 0.35
    if hits == 2:
        return 0.65
    return 1.0


def _reasoning_complexity(norm: dict) -> tuple[float, dict[str, float]]:
    """Compute Reasoning Complexity score in [0, 1].

    Pure factual look-ups score near 0.  Multi-hop logical deduction or
    ethical framework analysis scores near 1.

    Returns
    -------
    score:        float in [0, 1]
    group_scores: dict mapping group name → group contribution score
    """
    lower = norm["lower"]
    group_contributions: dict[str, float] = {}

    for group, (base_w, phrases) in _REASONING_GROUPS.items():
        hits = sum(1 for p in phrases if p in lower)
        if hits == 0:
            continue
        g_score = _hits_to_group_score(hits) * base_w
        group_contributions[group] = round(g_score, 4)

    if not group_contributions:
        return 0.0, {}

    n = len(group_contributions)
    avg = sum(group_contributions.values()) / n

    # Breadth bonus: multiple reasoning modes signal a harder request.
    breadth_multiplier = min(1.30, 1.0 + (n - 1) * 0.10)

    score = round(min(1.0, avg * breadth_multiplier), 4)
    return score, group_contributions


# ===========================================================================
# Stage 4c – Task Complexity
# ===========================================================================
# Verb groups represent the cognitive ambition of the primary request.
# Only the HIGHEST-weight matched group contributes the base score
# (prevents double-counting when multiple verbs appear).
#
# Modifier patterns are then added as small bonuses for additional scope.

_TASK_VERB_GROUPS: Final[dict[str, tuple[float, list[str]]]] = {
    # Highest-complexity tasks: system-level design and architecture
    "architect": (
        0.95,
        [
            "architect", "design a system", "design the architecture",
            "design a distributed", "design a scalable",
            "design a database schema", "design an api",
            "design a pipeline", "design a framework",
            "design a comprehensive", "design a phased",
            "design a go-to-market",
        ],
    ),
    # Complex technical implementation and engineering
    "implement_complex": (
        0.85,
        [
            "implement a", "implement an", "implement the",
            "build a", "build an",
            "write a class", "write a module",
            "create a class", "develop a",
        ],
    ),
    # Refactoring, optimization, and migration
    "refactor_optimize": (
        0.80,
        [
            "refactor", "optimize", "improve the performance of",
            "rewrite", "redesign", "restructure",
            "migrate", "migration plan",
        ],
    ),
    # Solving, determining, and verifying
    "determine_solve": (
        0.78,
        [
            "determine", "figure out", "find out",
            "solve", "calculate", "compute",
            "prove", "show that", "verify",
            "find all", "find the", "identify",
        ],
    ),
    # Debugging and diagnosis
    "debug_diagnose": (
        0.72,
        [
            "debug", "fix the bug", "diagnose",
            "troubleshoot", "find the bug",
            "identify the issue", "what is wrong with",
            "why does this fail", "why does this not",
        ],
    ),
    # Analysis and evaluation (without implementation)
    "analyze_evaluate": (
        0.65,
        [
            "analyze", "analyse", "review", "evaluate",
            "assess", "critique", "compare", "contrast",
            "examine", "investigate",
        ],
    ),
    # Explanation, teaching, and description
    "explain_teach": (
        0.40,
        [
            "explain", "describe", "what is", "what are",
            "how does", "how do", "define", "tell me about",
            "overview of", "introduction to",
            "in plain language", "in simple terms",
        ],
    ),
    # Simple generation (functions, scripts, lists)
    "generate_simple": (
        0.35,
        [
            "write a function that", "write a function to",
            "write a script that", "write a script to",
            "write a simple", "create a simple",
            "generate a list", "make a list",
            "give me an example of",
        ],
    ),
    # Translation, formatting, and transformation
    "transform_format": (
        0.20,
        [
            "translate", "convert to", "format as",
            "rewrite in", "in spanish", "in french", "in german",
            "in japanese", "in chinese",
            "summarize", "summarise", "condense", "shorten",
        ],
    ),
}

# Small bonuses added on top of the verb group base score.
# These capture scope amplifiers that appear regardless of verb type.
_TASK_MODIFIERS: Final[list[tuple[str, float, str]]] = [
    # Multiple deliverables (conjunctions of distinct requirements)
    (r"\b(and|also|additionally|furthermore|as well as)\b.{5,100}\b(and|also)\b", 0.10, "multiple-deliverables"),
    # Specific performance constraint (time/space complexity)
    (r"\bin\s+o\s*\([^)]+\)", 0.12, "complexity-constraint"),
    # Specific implementation language that adds overhead
    (r"\bin\s+(c\+\+|rust|go\b|java\b|kotlin|swift|scala|assembly)\b", 0.05, "low-level-language"),
    # Requesting explanation alongside implementation
    (r"\b(explain|justify|document|comment)\s+(why|how|the|your|each)\b", 0.08, "explain-plus-implement"),
    # Without using a simpler built-in approach
    (r"\bwithout\s+(using\s+)?(?:the\s+built-in|built-in|eval|stdlib|standard library)\b", 0.08, "without-builtin"),
    # Multi-part numbered deliverables inside the prompt
    (r"\b(?:part\s+[a-z0-9]+|step\s+[0-9]+)\b", 0.08, "multi-part"),
    # Comprehensive / detailed / thorough scope signals
    (r"\b(comprehensive|thorough|detailed|complete|end-to-end|full)\b", 0.06, "scope-amplifier"),
]


def _task_complexity(norm: dict) -> tuple[float, dict[str, str | float]]:
    """Compute Task Complexity score in [0, 1].

    Selects the highest-weight matching verb group (base score), then adds
    small bonuses for scope amplifiers.

    Returns
    -------
    score:   float in [0, 1]
    details: dict with matched_group, base_score, modifier_bonus
    """
    lower = norm["lower"]

    # Find the highest-weight verb group that fires.
    best_weight = 0.0
    best_group = "none"
    for group, (weight, verbs) in _TASK_VERB_GROUPS.items():
        for verb in verbs:
            if verb in lower:
                if weight > best_weight:
                    best_weight = weight
                    best_group = group
                break  # one match per group is enough to claim it

    # Default for prompts with no recognisable verb (rare edge case)
    if best_weight == 0.0:
        best_weight = 0.25
        best_group = "default"

    modifier_bonus = 0.0
    fired_modifiers: list[str] = []
    for pattern, bonus, name in _TASK_MODIFIERS:
        if re.search(pattern, lower):
            modifier_bonus += bonus
            fired_modifiers.append(name)

    score = round(min(1.0, best_weight + modifier_bonus), 4)
    details = {
        "matched_group": best_group,
        "base_score": best_weight,
        "modifier_bonus": round(modifier_bonus, 4),
        "fired_modifiers": fired_modifiers,
    }
    return score, details


# ===========================================================================
# Stage 4d – Constraint Complexity
# ===========================================================================
# Explicit requirements, restrictions, and structured-output demands inflate
# real difficulty because they narrow the solution space and increase the
# number of things the response must satisfy simultaneously.

# Each entry: (compiled pattern, weight, label)
_CONSTRAINT_PATTERNS: Final[list[tuple[re.Pattern, float, str]]] = [
    # Hard requirements
    (re.compile(r"\bmust\b(?!\s+not)", re.I), 0.12, "must-req"),
    (re.compile(r"\bmust\s+not\b|\bmustn't\b", re.I), 0.12, "must-not-req"),
    (re.compile(r"\bshould\s+not\b|\bshouldn't\b", re.I), 0.08, "should-not"),
    (re.compile(r"\bdo\s+not\b|\bdon't\b", re.I), 0.06, "negation"),
    (re.compile(r"\bwithout\s+(using|relying on)\b", re.I), 0.10, "without-constraint"),
    # Complexity requirements
    (re.compile(r"\bin\s+o\s*\([^)]+\)\s+time", re.I), 0.18, "time-complexity-req"),
    (re.compile(r"o\s*\([^)]+\)\s+space", re.I), 0.15, "space-complexity-req"),
    (re.compile(r"\bthread.safe\b|\bconcurrency.safe\b|\batomic\b", re.I), 0.18, "thread-safety"),
    (re.compile(r"\bbackward.compat", re.I), 0.12, "backward-compat"),
    (re.compile(r"\bno\s+deadlock\b|\bdeadlock.free\b", re.I), 0.15, "deadlock-free"),
    # Output format / structured output
    (re.compile(r"\breturn.{0,40}\b(json|xml|yaml|csv|object|dict|schema)\b", re.I), 0.10, "output-format"),
    (re.compile(r"\bformat.{0,30}\b(as|like)\b", re.I), 0.08, "format-as"),
    (re.compile(r"\binclude\s+(a\s+)?(the\s+)?(base case|inductive|example|docstring|type hint|unit test|test case|http status)", re.I), 0.10, "include-specific"),
    (re.compile(r"\bunit\s+test|test\s+case", re.I), 0.10, "testing-req"),
    (re.compile(r"\bdocstring|type\s+hint|annotation", re.I), 0.08, "code-doc-req"),
    # Multi-deliverable counting
    (re.compile(r"\b(also|additionally|furthermore|moreover|as well)\b", re.I), 0.07, "additional-deliverable"),
    (re.compile(r"\bexplain\s+(why|how|the|your)\b", re.I), 0.08, "explain-deliverable"),
    # Specific cardinality constraints
    (re.compile(r"\bexactly\s+(\d+|one|two|three|four|five)\b", re.I), 0.10, "exact-count"),
    (re.compile(r"\bat\s+least\s+\d+\b|\bno\s+more\s+than\s+\d+\b", re.I), 0.10, "bounded-count"),
    (re.compile(r"\bgiven\s+that\b|\bassuming\s+that\b|\bsubject\s+to\b", re.I), 0.08, "given-constraint"),
    # Structured planning / phased output
    (re.compile(r"\b(phase|milestone|deliverable|checkpoint|cadence|roadmap)\b", re.I), 0.08, "structured-plan"),
    (re.compile(r"\b(rto|rpo|sla|slo|kpi|okr)\b", re.I), 0.10, "formal-metric"),
    (re.compile(r"\bstate\s+(the|each|your)\b|\blist\s+(the|each)\b", re.I), 0.06, "explicit-enumeration"),
]

# Structured output patterns (each adds a fixed bonus)
_STRUCTURED_OUTPUT_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"\bjson\s+(schema|format|object|array)\b", re.I),
    re.compile(r"\brest(ful)?\s+api\b", re.I),
    re.compile(r"\bsql\s+(ddl|schema|query|statement)\b", re.I),
    re.compile(r"\bclass\s+diagram\b", re.I),
    re.compile(r"\buml\b", re.I),
    re.compile(r"\bopenapi\b|\bswagger\b", re.I),
]


def _constraint_complexity(norm: dict, structural: dict) -> tuple[float, list[str]]:
    """Compute Constraint Complexity score in [0, 1].

    Measures the density of explicit requirements, restrictions, and
    structured-output demands.

    Returns
    -------
    score:           float in [0, 1]
    fired_patterns:  list of pattern labels that contributed
    """
    lower = norm["lower"]
    total = 0.0
    fired: list[str] = []

    for pattern, weight, label in _CONSTRAINT_PATTERNS:
        if pattern.search(lower):
            total += weight
            fired.append(label)

    # Structured output bonus
    for pat in _STRUCTURED_OUTPUT_PATTERNS:
        if pat.search(lower):
            total += 0.10
            fired.append(f"structured-output:{pat.pattern[:20]}")

    # List items imply multiple parallel requirements
    list_items = structural["list_item_count"]
    if list_items >= 2:
        bonus = min(0.20, list_items * 0.04)
        total += bonus
        fired.append(f"list-items:{list_items}")

    return round(min(1.0, total), 4), fired


# ===========================================================================
# Stage 4e – Context Complexity  (intentionally a WEAK prior)
# ===========================================================================
# The contribution of this group is capped by its weight (0.08).
# A long but trivial prompt must not outscore a short but technically hard one.

# Hard-concept words that indicate domain depth regardless of prompt length.
# These are DIFFERENT from the technical domain keywords above — they are
# abstract, domain-agnostic markers of conceptual density.
_HARD_CONCEPT_WORDS: Final[frozenset[str]] = frozenset({
    # CS theory and algorithms
    "dijkstra", "backtracking", "memoization", "heuristic", "linearizable",
    "complexity", "amortized", "topological",
    # Concurrency
    "deadlock", "mutex", "semaphore", "concurrent", "synchronization",
    "coroutine", "atomic",
    # System design
    "distributed", "consensus", "sharding", "replication", "microservices",
    "orchestration", "partitioning",
    # Security
    "authentication", "authorization", "encryption", "injection", "vulnerability",
    # ML
    "backpropagation", "gradient", "transformer", "embedding", "regularization",
    # Database
    "normalization", "transaction", "indexing", "mvcc", "serializability",
    # Cloud/DevOps
    "kubernetes", "containerization", "terraform",
    # Analysis / reasoning
    "optimization", "differential", "eigenvalue", "determinant", "induction",
    # General technical breadth signals
    "architecture", "infrastructure", "framework", "protocol", "interface",
    "middleware", "abstraction", "encapsulation", "polymorphism",
})


def _context_complexity(norm: dict, lexical: dict) -> tuple[float, dict]:
    """Compute Context Complexity score in [0, 1] — weak prior only.

    Components:
    - token_load:         logarithmic scaling of token count (max contribution 0.40)
    - concept_density:    proportion of unique words that are hard concepts (max 0.50)
    - vocab_richness:     unique-word ratio as a mild depth signal (max 0.10)

    Design rationale: the logarithmic token load ensures a 1 000-token simple
    essay does not outrank a 60-token Dijkstra implementation request.
    Concept density is the dominant sub-signal within this group.

    Returns
    -------
    score:    float in [0, 1]
    details:  dict with sub-component values
    """
    tokens = lexical["estimated_input_tokens"]
    word_set = set(norm["words"])
    concept_hits = len(word_set & _HARD_CONCEPT_WORDS)

    # Logarithmic token load: fully saturates around 200 tokens (≈800 chars)
    # log1p(tokens/50) / log1p(20) ≈ 0 at 0 tokens, ≈1 at 1000 tokens
    token_load = min(0.40, math.log1p(tokens / 50) / math.log1p(20))

    # Concept density: % of unique word tokens that are hard concepts
    concept_density = concept_hits / max(len(word_set), 1)
    concept_score = min(0.50, concept_density * 4.0)

    vocab_richness = min(0.10, lexical["unique_word_ratio"] * 0.12)

    total = token_load * 0.35 + concept_score * 0.55 + vocab_richness * 0.10
    score = round(min(1.0, total), 4)
    details = {
        "token_load": round(token_load, 4),
        "concept_hits": concept_hits,
        "concept_density": round(concept_density, 4),
        "concept_score": round(concept_score, 4),
        "vocab_richness": round(vocab_richness, 4),
    }
    return score, details


# ===========================================================================
# Stage 5 – Complexity Aggregation
# ===========================================================================

def _aggregate_complexity(
    technical: float,
    reasoning: float,
    task: float,
    constraint: float,
    context: float,
) -> float:
    """Combine the five group scores into a single complexity_score in [0, 1].

    Uses _COMPLEXITY_WEIGHTS as the single source of truth for weights.
    The result is NOT clamped to encourage calibration inspection.
    In practice the maximum is bounded by the group scores (all ≤ 1) so
    the result is always in [0, 1].
    """
    score = (
        technical  * _COMPLEXITY_WEIGHTS["technical"]
        + reasoning  * _COMPLEXITY_WEIGHTS["reasoning"]
        + task       * _COMPLEXITY_WEIGHTS["task"]
        + constraint * _COMPLEXITY_WEIGHTS["constraint"]
        + context    * _COMPLEXITY_WEIGHTS["context"]
    )
    return round(min(1.0, max(0.0, score)), 4)


def _map_complexity(complexity_score: float) -> str:
    """Map a [0, 1] complexity score to a human-readable label.

    Thresholds
    ----------
    ≥ 0.60 → hard
    ≥ 0.30 → medium
      else → easy
    """
    if complexity_score >= 0.60:
        return COMPLEXITY_HARD
    if complexity_score >= 0.30:
        return COMPLEXITY_MEDIUM
    return COMPLEXITY_EASY


# ===========================================================================
# Backward-compat: improved task_type detection
# ===========================================================================
# Uses multi-signal detection: technical domains + task verbs + keywords.
# Returns a single dominant label to preserve the router's task_type_weights.

_TASK_TYPE_RULES: Final[list[tuple[str, re.Pattern]]] = [
    (TASK_CODING, re.compile(
        r"\b(code|program|script|function|class|debug|implement|algorithm|"
        r"api|library|refactor|compile|syntax|bug|unit test|"
        r"import|module|package|repository|github|git|"
        r"binary search|linked list|stack|queue|tree|graph|heap|"
        r"thread|mutex|deadlock|concurrency|async|coroutine|"
        r"database schema|sql\b|nosql|orm|indexing|"
        r"microservices|kubernetes|docker|ci/cd|terraform|"
        r"neural network|machine learning|deep learning)\b",
        re.IGNORECASE,
    )),
    (TASK_MATHEMATICS, re.compile(
        r"\b(calculate|compute|solve|equation|integral|derivative|matrix|"
        r"factorial|prime|fibonacci|probability|statistics|percentage|"
        r"algebra|geometry|calculus|proof|theorem|formula|"
        r"eigenvalue|determinant|induction|divergence)\b",
        re.IGNORECASE,
    )),
    (TASK_REASONING, re.compile(
        r"\b(why|reason|analyze|analyse|explain|cause|effect|impact|infer|"
        r"deduce|conclude|argument|logic|because|therefore|hence|compare|"
        r"contrast|evaluate|assess|critique|justify|implication|"
        r"ethical|fallacy|trade-off)\b",
        re.IGNORECASE,
    )),
    (TASK_PLANNING, re.compile(
        r"\b(plan|roadmap|schedule|strategy|steps to|checklist|"
        r"outline|agenda|milestone|goal|objective|task list|project|"
        r"timeline|action items|organize|prioritize|okr|kpi|"
        r"go-to-market|onboarding|hiring plan)\b",
        re.IGNORECASE,
    )),
    (TASK_SUMMARIZATION, re.compile(
        r"\b(summarize|summarise|summary|tldr|brief|overview|condense|"
        r"shorten|abstract|synopsis|recap|key points|highlights)\b",
        re.IGNORECASE,
    )),
    (TASK_TRANSLATION, re.compile(
        r"\b(translate|translation|übersetzen|traduzir|翻译|翻訳|traducir|"
        r"in (spanish|french|german|japanese|chinese|portuguese|"
        r"italian|russian|arabic|hindi|korean))\b",
        re.IGNORECASE,
    )),
    (TASK_CREATIVE_WRITING, re.compile(
        r"\b(write a (story|poem|song|essay|novel|script|blog|letter|article)|"
        r"creative|fiction|narrative|plot|character|dialogue|"
        r"rhyme|stanza|verse|haiku|limerick|short story)\b",
        re.IGNORECASE,
    )),
    (TASK_QUESTION_ANSWERING, re.compile(
        r"\b(what is|what are|who is|who are|when did|where is|where are|"
        r"how does|how do|how many|how much|define|definition of|"
        r"meaning of|tell me about)\b",
        re.IGNORECASE,
    )),
]


def _detect_task_type(norm: dict) -> tuple[str, list[str]]:
    """Classify the dominant task type using improved multi-signal heuristics.

    Evaluation order matches the priority of the router's task_type_weights.
    Returns the first (highest-priority) match, plus all matched labels for
    the ``task_labels`` multi-label output key.

    Returns
    -------
    primary:    single task-type string (backward-compat key)
    all_labels: all matched task types (new key ``task_labels``)
    """
    lower = norm["lower"]
    matched: list[str] = []
    for task_type, pattern in _TASK_TYPE_RULES:
        if pattern.search(lower):
            matched.append(task_type)

    if not matched:
        return TASK_GENERAL, [TASK_GENERAL]

    return matched[0], matched


# ===========================================================================
# Public API
# ===========================================================================

def extract_features(prompt: str, debug: bool = False) -> dict:
    """Extract structured features from a raw user prompt.

    This is the sole public entry point of this module.  It runs the full
    five-stage pipeline and returns a flat dictionary that the Routing Engine
    consumes directly.

    Backward compatibility
    ----------------------
    All keys present in Version 1 are preserved with the same semantics.
    The router, benchmark, and chat endpoints work without modification.

    Parameters
    ----------
    prompt:
        The raw user prompt string.
    debug:
        When True, an additional ``_debug`` key is included in the returned
        dictionary containing a full stage-by-stage breakdown.  This key is
        never present when ``debug=False`` so it does not affect serialisation
        in production.

    Returns
    -------
    A dictionary with the following keys:

    Backward-compatible keys (unchanged from v1)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``prompt_length`` (int):
        Character count of the raw prompt.
    ``word_count`` (int):
        Whitespace-separated token count.
    ``estimated_input_tokens`` (int):
        Lightweight token count approximation (ceil(len / 4)).
    ``contains_code`` (bool):
        True when fenced code, inline code, or programming keywords appear.
    ``contains_math`` (bool):
        True when LaTeX, math operators, or math vocabulary are detected.
    ``contains_json`` (bool):
        True when JSON-like object or array literals are detected.
    ``contains_markdown`` (bool):
        True when Markdown formatting is detected.
    ``contains_numbers`` (bool):
        True when at least one standalone number appears.
    ``contains_question`` (bool):
        True when a question mark or interrogative form is present.
    ``task_type`` (str):
        Dominant task category.  One of: coding, mathematics, reasoning,
        summarization, translation, creative_writing, planning,
        question_answering, general.
    ``reasoning_score`` (int):
        Integer in [0, 10] — now derived from ``complexity_score`` via
        ``round(complexity_score × 10)``.  Preserves the router's dominant
        signal without any router-side changes.
    ``complexity`` (str):
        Human-readable label: easy (score<0.30), medium (0.30–0.60),
        hard (≥0.60).

    New keys added by v2
    ~~~~~~~~~~~~~~~~~~~~
    ``technical_complexity`` (float):
        Stage 4a score — engineering domain depth, [0, 1].
    ``reasoning_complexity`` (float):
        Stage 4b score — analytical / logical demands, [0, 1].
    ``task_complexity`` (float):
        Stage 4c score — task verb ambition and scope, [0, 1].
    ``constraint_complexity`` (float):
        Stage 4d score — explicit requirements and restrictions, [0, 1].
    ``context_complexity`` (float):
        Stage 4e score — concept density and token load (weak prior), [0, 1].
    ``complexity_score`` (float):
        Stage 5 weighted aggregate, [0, 1].
    ``category_scores`` (dict):
        Mapping of group name → score for all five groups.
    ``task_labels`` (list[str]):
        All matched task types (multi-label; ``task_type`` is the primary).

    Optional debug key
    ~~~~~~~~~~~~~~~~~~
    ``_debug`` (dict):
        Present only when ``debug=True``.  Contains the full stage-by-stage
        trace: normalization stats, lexical features, structural features,
        per-domain and per-group breakdowns, modifier details, and the
        complete routing decision chain.
    """
    # ------------------------------------------------------------------
    # Stage 1 – Normalization
    # ------------------------------------------------------------------
    norm = _normalize(prompt)

    # ------------------------------------------------------------------
    # Stage 2 – Lexical features
    # ------------------------------------------------------------------
    lexical = _lexical_features(norm)

    # ------------------------------------------------------------------
    # Stage 3 – Structural features
    # ------------------------------------------------------------------
    structural = _structural_features(norm)

    # ------------------------------------------------------------------
    # Stage 4 – Semantic feature groups
    # ------------------------------------------------------------------
    tech_score, tech_domain_hits = _technical_complexity(norm)
    reason_score, reason_group_scores = _reasoning_complexity(norm)
    task_score, task_details = _task_complexity(norm)
    constraint_score, constraint_fired = _constraint_complexity(norm, structural)
    context_score, context_details = _context_complexity(norm, lexical)

    # ------------------------------------------------------------------
    # Stage 5 – Complexity aggregation
    # ------------------------------------------------------------------
    complexity_score = _aggregate_complexity(
        tech_score, reason_score, task_score, constraint_score, context_score
    )
    complexity_label = _map_complexity(complexity_score)

    # Bridge to the existing router: reasoning_score must be an int in [0, 10]
    reasoning_score = round(complexity_score * 10)

    # ------------------------------------------------------------------
    # Task type detection (improved; backward-compat primary label)
    # ------------------------------------------------------------------
    task_type, task_labels = _detect_task_type(norm)

    # ------------------------------------------------------------------
    # Assemble output dictionary
    # ------------------------------------------------------------------
    category_scores = {
        "technical":  tech_score,
        "reasoning":  reason_score,
        "task":       task_score,
        "constraint": constraint_score,
        "context":    context_score,
    }

    output: dict = {
        # ── backward-compatible keys ─────────────────────────────────
        "prompt_length":          lexical["prompt_length"],
        "word_count":             lexical["word_count"],
        "estimated_input_tokens": lexical["estimated_input_tokens"],
        "contains_code":          structural["contains_code"],
        "contains_math":          structural["contains_math"],
        "contains_json":          structural["contains_json"],
        "contains_markdown":      structural["contains_markdown"],
        "contains_numbers":       structural["contains_numbers"],
        "contains_question":      structural["contains_question"],
        "task_type":              task_type,
        "reasoning_score":        reasoning_score,
        "complexity":             complexity_label,
        # ── new v2 keys ──────────────────────────────────────────────
        "technical_complexity":   tech_score,
        "reasoning_complexity":   reason_score,
        "task_complexity":        task_score,
        "constraint_complexity":  constraint_score,
        "context_complexity":     context_score,
        "complexity_score":       complexity_score,
        "category_scores":        category_scores,
        "task_labels":            task_labels,
    }

    # ------------------------------------------------------------------
    # Optional debug trace
    # ------------------------------------------------------------------
    if debug:
        output["_debug"] = {
            "pipeline_version": "v2",
            "stage_1_normalization": {
                "sentence_count": len(norm["sentences"]),
                "word_token_count": len(norm["words"]),
            },
            "stage_2_lexical": lexical,
            "stage_3_structural": structural,
            "stage_4a_technical": {
                "score": tech_score,
                "domain_contributions": tech_domain_hits,
            },
            "stage_4b_reasoning": {
                "score": reason_score,
                "group_contributions": reason_group_scores,
            },
            "stage_4c_task": {
                "score": task_score,
                **task_details,
            },
            "stage_4d_constraint": {
                "score": constraint_score,
                "fired_patterns": constraint_fired,
            },
            "stage_4e_context": {
                "score": context_score,
                **context_details,
            },
            "stage_5_aggregation": {
                "weights": _COMPLEXITY_WEIGHTS,
                "complexity_score": complexity_score,
                "complexity_label": complexity_label,
                "reasoning_score_bridge": reasoning_score,
            },
            "task_detection": {
                "primary": task_type,
                "all_labels": task_labels,
            },
        }

    return output
