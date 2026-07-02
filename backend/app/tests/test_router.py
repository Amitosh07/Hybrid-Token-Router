"""Tests for the Routing Engine.

All tests are deterministic: no network calls, no LLMs, no external services.
Each test class targets a distinct prompt archetype and verifies provider
selection, output schema, confidence bounds, and reason list population.
"""

from __future__ import annotations

import unittest

from app.services.feature_extractor import extract_features
from app.services.router import (
    PROVIDER_LOCAL,
    PROVIDER_REMOTE,
    RouterConfig,
    route,
    route_prompt,
)

# ---------------------------------------------------------------------------
# Required output keys – validated in every test.
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset({"provider", "routing_score", "confidence", "reason"})


def _assert_schema(tc: unittest.TestCase, result: dict) -> None:
    """Assert that all required keys are present and types are correct."""
    tc.assertEqual(
        _REQUIRED_KEYS,
        _REQUIRED_KEYS & result.keys(),
        msg="Missing keys: " + str(_REQUIRED_KEYS - result.keys()),
    )
    tc.assertIn(result["provider"], {PROVIDER_LOCAL, PROVIDER_REMOTE})
    tc.assertIsInstance(result["routing_score"], int)
    tc.assertGreaterEqual(result["confidence"], 0.0)
    tc.assertLessEqual(result["confidence"], 1.0)
    tc.assertIsInstance(result["reason"], list)
    tc.assertTrue(len(result["reason"]) >= 1, "reason list must be non-empty")


# ---------------------------------------------------------------------------
# Schema and invariant tests
# ---------------------------------------------------------------------------

class SchemaTests(unittest.TestCase):
    """Output dictionary must always conform to the required schema."""

    def test_empty_prompt_schema(self) -> None:
        result = route_prompt("")
        _assert_schema(self, result)

    def test_greeting_schema(self) -> None:
        result = route_prompt("Hello!")
        _assert_schema(self, result)

    def test_confidence_always_in_range(self) -> None:
        for prompt in [
            "Hi",
            "Write a Python function",
            "Solve the integral of x^2",
            "Translate this to German",
            "Design a distributed event-driven architecture with failover",
        ]:
            with self.subTest(prompt=prompt):
                result = route_prompt(prompt)
                self.assertGreaterEqual(result["confidence"], 0.0)
                self.assertLessEqual(result["confidence"], 1.0)

    def test_routing_score_is_non_negative(self) -> None:
        result = route_prompt("Hello")
        self.assertGreaterEqual(result["routing_score"], 0)

    def test_route_is_deterministic(self) -> None:
        prompt = "Explain the difference between TCP and UDP with examples."
        first = route_prompt(prompt)
        second = route_prompt(prompt)
        self.assertEqual(first, second)


# ---------------------------------------------------------------------------
# Simple greeting — should go LOCAL
# ---------------------------------------------------------------------------

class GreetingRoutingTests(unittest.TestCase):
    """A plain greeting has near-zero complexity and must stay local."""

    def setUp(self) -> None:
        self.result = route_prompt("Hello!")

    def test_schema(self) -> None:
        _assert_schema(self, self.result)

    def test_routes_local(self) -> None:
        self.assertEqual(self.result["provider"], PROVIDER_LOCAL)

    def test_routing_score_is_low(self) -> None:
        # A greeting has no reasoning keywords, no code, no math, short length
        self.assertLess(self.result["routing_score"], 25)

    def test_hi_variant_local(self) -> None:
        result = route_prompt("Hi there, how are you?")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)

    def test_good_morning_local(self) -> None:
        result = route_prompt("Good morning!")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)


# ---------------------------------------------------------------------------
# Simple coding task — should go LOCAL
# ---------------------------------------------------------------------------

class SimpleCodingRoutingTests(unittest.TestCase):
    """Short, trivial coding requests should stay local.

    'Write hello world in Python' is low-complexity regardless of task_type.
    """

    # A slightly longer prompt so the code keyword fires the code detector.
    _PROMPT = "Write a Python function that prints hello world."

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)
        self.result = route(self.features)

    def test_schema(self) -> None:
        _assert_schema(self, self.result)

    def test_routes_local(self) -> None:
        self.assertEqual(self.result["provider"], PROVIDER_LOCAL)

    def test_routing_score_below_threshold(self) -> None:
        # Simple task: code bonus + task nudge only, no reasoning
        self.assertLess(self.result["routing_score"], 25)

    def test_contains_code_flag_fires(self) -> None:
        # "function" keyword triggers contains_code
        self.assertTrue(self.features["contains_code"])

    def test_task_type_is_coding(self) -> None:
        self.assertEqual(self.features["task_type"], "coding")

    def test_simple_vs_complex_coding_score_ordering(self) -> None:
        """Complex coding must score strictly higher than simple coding."""
        complex_result = route_prompt(
            "Design and implement a Python class that acts as an async "
            "connection pool for PostgreSQL. It must support configurable "
            "min/max pool size, connection health checks, automatic "
            "reconnection on failure, and graceful shutdown. Include type "
            "hints, docstrings, and unit tests. The implementation must not "
            "use any third-party libraries except asyncpg."
        )
        self.assertGreater(
            complex_result["routing_score"],
            self.result["routing_score"],
        )


# ---------------------------------------------------------------------------
# Complex coding task — should go REMOTE
# ---------------------------------------------------------------------------

class ComplexCodingRoutingTests(unittest.TestCase):
    """A deeply constrained, multi-requirement coding prompt should go remote.

    reasoning_score carries more weight than task_type alone.
    """

    _PROMPT = (
        "Design and implement a Python class that acts as an async connection "
        "pool for PostgreSQL. It must support configurable min/max pool size, "
        "connection health checks, automatic reconnection on failure, and "
        "graceful shutdown. Include type hints, docstrings, and unit tests. "
        "The implementation must not use any third-party libraries except asyncpg."
    )

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)
        self.result = route(self.features)

    def test_schema(self) -> None:
        _assert_schema(self, self.result)

    def test_routes_remote(self) -> None:
        self.assertEqual(self.result["provider"], PROVIDER_REMOTE)

    def test_routing_score_above_threshold(self) -> None:
        self.assertGreaterEqual(self.result["routing_score"], 25)

    def test_confidence_reasonable(self) -> None:
        self.assertGreater(self.result["confidence"], 0.3)


# ---------------------------------------------------------------------------
# Mathematics — simple calculation LOCAL, complex proof REMOTE
# ---------------------------------------------------------------------------

class MathRoutingTests(unittest.TestCase):
    """Math routing should depend on complexity, not task_type alone."""

    def test_simple_arithmetic_is_local(self) -> None:
        result = route_prompt("What is 12 * 8?")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)
        _assert_schema(self, result)

    def test_complex_math_is_remote(self) -> None:
        # Prompt contains explicit reasoning keywords: Analyze, evaluate, deduce,
        # explain, compare — these push reasoning_score up, crossing the threshold.
        prompt = (
            "Analyze and evaluate the implications of Goldbach's conjecture. "
            "Deduce which mathematical tools are most likely to yield a "
            "breakthrough, and explain why each approach either succeeds or fails. "
            "You must compare and contrast at least three proof strategies, "
            "justify each elimination step, and conclude with a formal argument. "
            "Do not omit edge cases."
        )
        result = route_prompt(prompt)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)
        _assert_schema(self, result)

    def test_complex_math_score_exceeds_threshold(self) -> None:
        prompt = (
            "Analyze and evaluate the implications of Goldbach's conjecture. "
            "Deduce which mathematical tools are most likely to yield a "
            "breakthrough, and explain why each approach either succeeds or fails. "
            "You must compare and contrast at least three proof strategies, "
            "justify each elimination step, and conclude with a formal argument. "
            "Do not omit edge cases."
        )
        result = route_prompt(prompt)
        self.assertGreaterEqual(result["routing_score"], 25)

    def test_math_reason_present(self) -> None:
        result = route_prompt("Solve the integral of x^2 from 0 to 5.")
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("math", reasons_text)


# ---------------------------------------------------------------------------
# Translation — simple LOCAL, multi-language complex REMOTE
# ---------------------------------------------------------------------------

class TranslationRoutingTests(unittest.TestCase):
    """Short translation requests stay local; elaborate ones may go remote."""

    def test_short_translation_is_local(self) -> None:
        result = route_prompt("Translate 'Good morning' into French.")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)
        _assert_schema(self, result)

    def test_schema_for_translation(self) -> None:
        result = route_prompt("Translate the following paragraph into Spanish.")
        _assert_schema(self, result)

    def test_routing_score_lower_than_complex_coding(self) -> None:
        simple_tx = route_prompt("Translate 'hello' into German.")
        complex_code = route(extract_features(ComplexCodingRoutingTests._PROMPT))
        self.assertLess(
            simple_tx["routing_score"],
            complex_code["routing_score"],
        )


# ---------------------------------------------------------------------------
# Planning — simple LOCAL, complex architecture REMOTE
# ---------------------------------------------------------------------------

class PlanningRoutingTests(unittest.TestCase):
    """Planning routing hinges on complexity, not task category."""

    def test_simple_plan_is_local(self) -> None:
        result = route_prompt("Give me a checklist for packing a suitcase.")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)
        _assert_schema(self, result)

    def test_complex_architecture_plan_is_remote(self) -> None:
        # Prompt is deliberately rich in reasoning keywords (analyze, evaluate,
        # compare, contrast, justify, deduce) and constraints (must, must not)
        # so the reasoning_score drives it above the threshold.
        prompt = (
            "Analyze and evaluate the trade-offs between Kafka, Pulsar, and "
            "AWS EventBridge for a globally distributed event-driven architecture. "
            "You must compare and contrast their consistency guarantees, deduce "
            "which suits CQRS with eventual consistency, and justify the choice "
            "with concrete reasoning. Also outline a phased migration strategy "
            "from a monolith. You must not ignore geo-replication constraints "
            "or automatic failover requirements."
        )
        result = route_prompt(prompt)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)
        _assert_schema(self, result)

    def test_planning_reason_present(self) -> None:
        result = route_prompt("Create a project roadmap for a mobile app launch.")
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("plan", reasons_text)


# ---------------------------------------------------------------------------
# Creative writing — short LOCAL, elaborate multi-constraint REMOTE
# ---------------------------------------------------------------------------

class CreativeWritingRoutingTests(unittest.TestCase):
    """Creative writing routing is driven by constraints, not task type alone."""

    def test_simple_creative_is_local(self) -> None:
        result = route_prompt("Write a haiku about autumn.")
        self.assertEqual(result["provider"], PROVIDER_LOCAL)
        _assert_schema(self, result)

    def test_complex_creative_is_remote(self) -> None:
        # Rich in reasoning keywords and constraints to cross the threshold.
        prompt = (
            "Write a short story that explores and analyzes the contrast between "
            "utilitarianism and existentialism. The narrative must evaluate the "
            "moral implications of each philosophy, deduce which worldview the "
            "protagonist ultimately justifies, and conclude with an argument "
            "that the reader can critique. You must include at least two "
            "unreliable narrators and must not use any cliches."
        )
        result = route_prompt(prompt)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)
        _assert_schema(self, result)

    def test_creative_score_increases_with_constraints(self) -> None:
        """Adding constraints must increase the routing score."""
        simple = route_prompt("Write a poem about the sea.")
        constrained = route_prompt(
            "Write a poem about the sea. You must analyze three metaphors, "
            "evaluate their emotional impact, and justify why each was chosen. "
            "Do not use rhyme. The poem must not exceed 14 lines."
        )
        self.assertGreater(
            constrained["routing_score"],
            simple["routing_score"],
        )


# ---------------------------------------------------------------------------
# Threshold configurability
# ---------------------------------------------------------------------------

class ThresholdConfigTests(unittest.TestCase):
    """RouterConfig.threshold must be the sole routing decision boundary."""

    def test_zero_threshold_always_remote(self) -> None:
        cfg = RouterConfig(threshold=0)
        result = route(extract_features("Hi"), config=cfg)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)

    def test_very_high_threshold_always_local(self) -> None:
        cfg = RouterConfig(threshold=9999)
        result = route(
            extract_features(
                "Analyze and evaluate every known algorithm for NP-complete "
                "problems. Deduce which heuristic best approximates the optimal "
                "solution given arbitrary constraints. You must compare all "
                "major approaches and justify each conclusion."
            ),
            config=cfg,
        )
        self.assertEqual(result["provider"], PROVIDER_LOCAL)

    def test_custom_weights_change_score(self) -> None:
        """Increasing reasoning_score_weight must increase the routing score."""
        prompt = "Explain why the sky is blue."
        features = extract_features(prompt)

        low_cfg = RouterConfig(reasoning_score_weight=1)
        high_cfg = RouterConfig(reasoning_score_weight=10)

        low_score = route(features, config=low_cfg)["routing_score"]
        high_score = route(features, config=high_cfg)["routing_score"]

        self.assertGreater(high_score, low_score)

    def test_threshold_boundary_exact(self) -> None:
        """A score exactly equal to threshold must route remote."""
        # Set threshold to 0 and use a prompt that scores > 0.
        cfg = RouterConfig(threshold=0)
        result = route(extract_features("Hello"), config=cfg)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)


# ---------------------------------------------------------------------------
# Confidence distance tests
# ---------------------------------------------------------------------------

class ConfidenceTests(unittest.TestCase):
    """Confidence must increase as routing_score moves away from threshold."""

    def test_score_far_below_threshold_has_high_confidence(self) -> None:
        """A score of 0 far below threshold=25 should have high confidence."""
        cfg = RouterConfig(threshold=25, max_confidence_distance=20)
        # Inject a minimal feature dict that scores 0.
        features = {
            "reasoning_score": 0,
            "estimated_input_tokens": 0,
            "complexity": "easy",
            "task_type": "general",
            "contains_code": False,
            "contains_math": False,
            "contains_json": False,
        }
        result = route(features, config=cfg)
        # score=0, threshold=25, distance=25 → high confidence LOCAL
        self.assertGreater(result["confidence"], 0.6)
        self.assertEqual(result["provider"], PROVIDER_LOCAL)

    def test_score_at_boundary_has_lower_confidence(self) -> None:
        """A score exactly equal to threshold should have lower confidence."""
        cfg = RouterConfig(threshold=25, max_confidence_distance=20)
        # Build a features dict that produces routing_score == 25 exactly.
        # reasoning_score=5 → 25 pts, all else False/0.
        features = {
            "reasoning_score": 5,
            "estimated_input_tokens": 0,
            "complexity": "easy",
            "task_type": "general",
            "contains_code": False,
            "contains_math": False,
            "contains_json": False,
        }
        result = route(features, config=cfg)
        # score=25, threshold=25, distance=0 → very low confidence
        self.assertLess(result["confidence"], 0.3)
        self.assertEqual(result["provider"], PROVIDER_REMOTE)

    def test_far_above_threshold_high_confidence(self) -> None:
        cfg = RouterConfig(threshold=0, max_confidence_distance=20)
        features = extract_features(
            "Analyze and evaluate and deduce the implications of quantum "
            "computing on cryptography. Given that Shor's algorithm must be "
            "considered, compare and contrast RSA versus lattice-based schemes, "
            "justifying each step with formal arguments. You must not omit "
            "edge cases or ignore post-quantum migration constraints."
        )
        result = route(features, config=cfg)
        self.assertGreater(result["confidence"], 0.6)

    def test_confidence_monotonically_increases_with_distance(self) -> None:
        """Scores farther from the threshold must have higher confidence."""
        cfg = RouterConfig(threshold=25, max_confidence_distance=20)
        # score=0 (distance=25) vs score=15 (distance=10)
        close_features = {
            "reasoning_score": 3,    # 3*5=15
            "estimated_input_tokens": 0,
            "complexity": "easy",
            "task_type": "general",
            "contains_code": False,
            "contains_math": False,
            "contains_json": False,
        }
        far_features = {
            "reasoning_score": 0,    # 0*5=0
            "estimated_input_tokens": 0,
            "complexity": "easy",
            "task_type": "general",
            "contains_code": False,
            "contains_math": False,
            "contains_json": False,
        }
        close_conf = route(close_features, config=cfg)["confidence"]
        far_conf = route(far_features, config=cfg)["confidence"]
        self.assertGreater(far_conf, close_conf)


# ---------------------------------------------------------------------------
# Reason list tests
# ---------------------------------------------------------------------------

class ReasonListTests(unittest.TestCase):
    """Reason list must always be non-empty and contain the decision summary."""

    def _has_decision_summary(self, result: dict) -> bool:
        return any("threshold" in r.lower() for r in result["reason"])

    def test_local_result_has_decision_summary(self) -> None:
        result = route_prompt("Hi")
        self.assertTrue(self._has_decision_summary(result))

    def test_remote_result_has_decision_summary(self) -> None:
        result = route_prompt(
            "Analyze, evaluate, compare, and deduce every aspect of the CAP "
            "theorem. You must justify why strong consistency conflicts with "
            "availability, explain each trade-off, and conclude with a formal "
            "argument. Do not ignore partition tolerance constraints."
        )
        self.assertTrue(self._has_decision_summary(result))

    def test_code_prompt_mentions_code(self) -> None:
        result = route_prompt("```python\ndef foo(): pass\n```")
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("code", reasons_text)

    def test_json_prompt_mentions_json(self) -> None:
        result = route_prompt('Parse this JSON: {"key": "value", "num": 42}')
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("json", reasons_text)

    def test_math_prompt_mentions_math(self) -> None:
        result = route_prompt("Solve the integral of x^2.")
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("math", reasons_text)

    def test_high_reasoning_score_narrated(self) -> None:
        result = route_prompt(
            "Analyze and evaluate and compare and contrast and deduce and "
            "justify every implication of the given argument. You must "
            "conclude with a formal logical proof that can be critiqued."
        )
        reasons_text = " ".join(result["reason"]).lower()
        self.assertIn("reasoning score", reasons_text)


if __name__ == "__main__":
    unittest.main()
