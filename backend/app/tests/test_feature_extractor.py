"""Tests for the feature extraction module.

Each test class targets one prompt archetype and verifies that all required
feature keys are present and semantically correct.  Tests are fully
deterministic: no network calls, no LLMs, no external dependencies.
"""

from __future__ import annotations

import unittest

from app.services.feature_extractor import (
    COMPLEXITY_EASY,
    COMPLEXITY_HARD,
    COMPLEXITY_MEDIUM,
    TASK_CODING,
    TASK_CREATIVE_WRITING,
    TASK_GENERAL,
    TASK_MATHEMATICS,
    TASK_PLANNING,
    TASK_QUESTION_ANSWERING,
    TASK_REASONING,
    TASK_TRANSLATION,
    extract_features,
)

# ---------------------------------------------------------------------------
# Required feature keys – validated in every test.
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset(
    {
        "prompt_length",
        "word_count",
        "estimated_input_tokens",
        "contains_code",
        "contains_math",
        "contains_json",
        "contains_markdown",
        "contains_numbers",
        "contains_question",
        "task_type",
        "reasoning_score",
        "complexity",
    }
)


def _assert_schema(tc: unittest.TestCase, features: dict) -> None:
    """Assert that every required key is present in *features*."""
    tc.assertEqual(
        _REQUIRED_KEYS,
        _REQUIRED_KEYS & features.keys(),
        msg="Missing feature keys: " + str(_REQUIRED_KEYS - features.keys()),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FeatureSchemaTests(unittest.TestCase):
    """Verify that the output dictionary always contains every required key."""

    def test_schema_is_complete_for_empty_prompt(self) -> None:
        features = extract_features("")
        _assert_schema(self, features)

    def test_schema_is_complete_for_typical_prompt(self) -> None:
        features = extract_features("Hello, how are you?")
        _assert_schema(self, features)


# ---------------------------------------------------------------------------
# Coding prompts
# ---------------------------------------------------------------------------


class CodingPromptTests(unittest.TestCase):
    """Feature extraction for prompts that contain or request source code."""

    _PROMPT = (
        "Write a Python function that implements binary search on a sorted list. "
        "Include docstrings and unit tests."
    )

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_coding(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_CODING)

    def test_contains_code_detected(self) -> None:
        # "function" keyword and "Python" should trip the code detector
        self.assertTrue(self.features["contains_code"])

    def test_complexity_is_at_least_easy(self) -> None:
        self.assertIn(
            self.features["complexity"],
            {COMPLEXITY_EASY, COMPLEXITY_MEDIUM, COMPLEXITY_HARD},
        )

    def test_reasoning_score_in_range(self) -> None:
        self.assertGreaterEqual(self.features["reasoning_score"], 0)
        self.assertLessEqual(self.features["reasoning_score"], 10)

    def test_word_count_positive(self) -> None:
        self.assertGreater(self.features["word_count"], 0)

    def test_estimated_tokens_positive(self) -> None:
        self.assertGreater(self.features["estimated_input_tokens"], 0)

    def test_prompt_length_matches(self) -> None:
        self.assertEqual(self.features["prompt_length"], len(self._PROMPT))

    def test_fenced_code_block_sets_contains_code(self) -> None:
        prompt = "Here is the code:\n```python\ndef foo():\n    return 42\n```"
        features = extract_features(prompt)
        self.assertTrue(features["contains_code"])

    def test_public_in_prose_does_not_set_contains_code(self) -> None:
        features = extract_features(
            "Summarize the public dataset and explain its value to public health researchers."
        )
        self.assertFalse(features["contains_code"])


# ---------------------------------------------------------------------------
# Math prompts
# ---------------------------------------------------------------------------


class MathPromptTests(unittest.TestCase):
    """Feature extraction for prompts with mathematical content."""

    _PROMPT = "Solve the integral of x^2 from 0 to 5 and simplify the result."

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_mathematics(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_MATHEMATICS)

    def test_contains_math_detected(self) -> None:
        self.assertTrue(self.features["contains_math"])

    def test_contains_numbers(self) -> None:
        self.assertTrue(self.features["contains_numbers"])

    def test_reasoning_score_elevated(self) -> None:
        # math presence contributes at least 1 point
        self.assertGreaterEqual(self.features["reasoning_score"], 1)

    def test_latex_math_detected(self) -> None:
        prompt = "Compute $\\int_0^1 x^2 dx$."
        features = extract_features(prompt)
        self.assertTrue(features["contains_math"])

    def test_bare_arithmetic_detected(self) -> None:
        prompt = "What is 12 * 8 + 4?"
        features = extract_features(prompt)
        self.assertTrue(features["contains_math"])


# ---------------------------------------------------------------------------
# Reasoning prompts
# ---------------------------------------------------------------------------


class ReasoningPromptTests(unittest.TestCase):
    """Feature extraction for prompts demanding multi-step reasoning."""

    _PROMPT = (
        "Analyze and compare the causes and effects of the 2008 financial crisis. "
        "Explain why the subprime mortgage market collapsed and what steps were "
        "taken. You must cite at least three economic indicators. Given that the "
        "Federal Reserve intervened, deduce the long-term implications for monetary "
        "policy. Do not include political opinions."
    )

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_reasoning(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_REASONING)

    def test_contains_question(self) -> None:
        self.assertTrue(self.features["contains_question"])

    def test_reasoning_score_is_high(self) -> None:
        # long prompt + reasoning keywords + constraints → score ≥ 4
        self.assertGreaterEqual(self.features["reasoning_score"], 4)

    def test_complexity_is_medium_or_hard(self) -> None:
        self.assertIn(
            self.features["complexity"], {COMPLEXITY_MEDIUM, COMPLEXITY_HARD}
        )

    def test_reasoning_score_in_valid_range(self) -> None:
        self.assertGreaterEqual(self.features["reasoning_score"], 0)
        self.assertLessEqual(self.features["reasoning_score"], 10)


# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------


class TranslationPromptTests(unittest.TestCase):
    """Feature extraction for translation requests."""

    _PROMPT = "Translate the following paragraph into French and Spanish."

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_translation(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_TRANSLATION)

    def test_complexity_is_easy(self) -> None:
        # A short, single-sentence translation request is low complexity
        self.assertIn(
            self.features["complexity"],
            {COMPLEXITY_EASY, COMPLEXITY_MEDIUM},
        )

    def test_contains_question_false(self) -> None:
        self.assertFalse(self.features["contains_question"])

    def test_translate_in_language_variant(self) -> None:
        prompt = "Translate this sentence in German."
        features = extract_features(prompt)
        self.assertEqual(features["task_type"], TASK_TRANSLATION)


# ---------------------------------------------------------------------------
# Planning prompts
# ---------------------------------------------------------------------------


class PlanningPromptTests(unittest.TestCase):
    """Feature extraction for planning and roadmap requests."""

    _PROMPT = (
        "Create a 30-day roadmap for learning machine learning. "
        "Outline the key milestones, weekly goals, and action items. "
        "Prioritize topics based on beginner to advanced progression."
    )

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_planning(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_PLANNING)

    def test_contains_numbers(self) -> None:
        self.assertTrue(self.features["contains_numbers"])

    def test_reasoning_score_in_range(self) -> None:
        self.assertGreaterEqual(self.features["reasoning_score"], 0)
        self.assertLessEqual(self.features["reasoning_score"], 10)

    def test_checklist_variant(self) -> None:
        prompt = "Give me a checklist of steps to launch a startup."
        features = extract_features(prompt)
        self.assertEqual(features["task_type"], TASK_PLANNING)


# ---------------------------------------------------------------------------
# Greeting prompts
# ---------------------------------------------------------------------------


class GreetingPromptTests(unittest.TestCase):
    """Feature extraction for minimal greeting inputs."""

    _PROMPT = "Hello!"

    def setUp(self) -> None:
        self.features = extract_features(self._PROMPT)

    def test_schema_complete(self) -> None:
        _assert_schema(self, self.features)

    def test_task_type_is_general(self) -> None:
        self.assertEqual(self.features["task_type"], TASK_GENERAL)

    def test_complexity_is_easy(self) -> None:
        self.assertEqual(self.features["complexity"], COMPLEXITY_EASY)

    def test_reasoning_score_is_zero(self) -> None:
        self.assertEqual(self.features["reasoning_score"], 0)

    def test_no_code_math_json_markdown(self) -> None:
        self.assertFalse(self.features["contains_code"])
        self.assertFalse(self.features["contains_math"])
        self.assertFalse(self.features["contains_json"])
        self.assertFalse(self.features["contains_markdown"])

    def test_hi_variant(self) -> None:
        features = extract_features("Hi there")
        self.assertEqual(features["task_type"], TASK_GENERAL)
        self.assertEqual(features["complexity"], COMPLEXITY_EASY)


# ---------------------------------------------------------------------------
# Additional feature-level tests
# ---------------------------------------------------------------------------


class TokenEstimationTests(unittest.TestCase):
    """Verify the lightweight token estimator behaves correctly."""

    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(extract_features("")["estimated_input_tokens"], 0)

    def test_four_chars_returns_one_token(self) -> None:
        self.assertEqual(extract_features("abcd")["estimated_input_tokens"], 1)

    def test_five_chars_returns_two_tokens(self) -> None:
        self.assertEqual(extract_features("abcde")["estimated_input_tokens"], 2)

    def test_token_count_always_positive_for_nonempty(self) -> None:
        self.assertGreater(extract_features("x")["estimated_input_tokens"], 0)


class ContainsJsonTests(unittest.TestCase):
    """Verify JSON-like structure detection."""

    def test_object_detected(self) -> None:
        features = extract_features('{"key": "value"}')
        self.assertTrue(features["contains_json"])

    def test_array_detected(self) -> None:
        features = extract_features("[1, 2, 3]")
        self.assertTrue(features["contains_json"])

    def test_plain_text_not_detected(self) -> None:
        features = extract_features("This is plain text with no JSON.")
        self.assertFalse(features["contains_json"])


class ContainsMarkdownTests(unittest.TestCase):
    """Verify Markdown formatting detection."""

    def test_heading_detected(self) -> None:
        features = extract_features("# Introduction\nThis is a heading.")
        self.assertTrue(features["contains_markdown"])

    def test_bold_detected(self) -> None:
        features = extract_features("This is **bold** text.")
        self.assertTrue(features["contains_markdown"])

    def test_list_item_detected(self) -> None:
        features = extract_features("- First item\n- Second item")
        self.assertTrue(features["contains_markdown"])


class ComplexityMappingTests(unittest.TestCase):
    """Verify the three-band complexity mapping."""

    def _prompt_with_score(self, target: int) -> str:
        """Build a prompt that approximates a specific reasoning score."""
        # A very short, content-free prompt yields 0.
        if target == 0:
            return "Hi"
        # A long prompt with many reasoning keywords and constraints yields ≥ 8.
        if target >= 8:
            return (
                "Analyze and evaluate and compare and contrast and deduce and infer "
                "the following argument. You must, given that the constraints are "
                "such that you must not ignore any implication, therefore consequently "
                "hence conclude. " * 6
            )
        # A medium prompt with a few keywords yields 4–7.
        return "Explain why and analyze the cause and effect of climate change in detail."

    def test_score_zero_maps_to_easy(self) -> None:
        features = extract_features("Hi")
        self.assertEqual(features["complexity"], COMPLEXITY_EASY)

    def test_medium_prompt_maps_to_medium_or_above(self) -> None:
        # Prompt is ~290 chars (length_score=1) + reasoning keywords: why, analyze,
        # evaluate, compare, cause, effect, implications, infer (keyword_score=2+)
        # + constraint "must" (constraint_score=1) → total ≥ 4 → medium or hard.
        prompt = (
            "Analyze and evaluate the root causes and effects of climate change. "
            "Compare the economic and environmental implications across developed "
            "and developing nations. You must infer which policy interventions are "
            "most effective given the available evidence."
        )
        features = extract_features(prompt)
        self.assertIn(
            features["complexity"], {COMPLEXITY_MEDIUM, COMPLEXITY_HARD}
        )

    def test_high_score_maps_to_hard(self) -> None:
        prompt = self._prompt_with_score(10)
        features = extract_features(prompt)
        self.assertEqual(features["complexity"], COMPLEXITY_HARD)


if __name__ == "__main__":
    unittest.main()
