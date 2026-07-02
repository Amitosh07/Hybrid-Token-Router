import unittest

from app.services import evaluator


def _run() -> dict:
    return {
        "prompt": "Explain retrieval augmented generation for search.",
        "local": {
            "model": "local-model",
            "response": "Retrieval augmented generation uses search results to ground an answer.",
            "latency_ms": 800,
        },
        "remote": {
            "model": "remote-model",
            "response": "RAG combines retrieval with generation.",
            "latency_ms": 1400,
        },
    }


class EvaluatorTests(unittest.TestCase):
    def test_compare_response_lengths_returns_deltas(self) -> None:
        metrics = evaluator.compare_response_lengths(_run()["local"], _run()["remote"])

        self.assertGreater(metrics["local_chars"], metrics["remote_chars"])
        self.assertGreater(metrics["local_words"], metrics["remote_words"])
        self.assertGreater(metrics["char_delta"], 0)
        self.assertGreater(metrics["word_delta"], 0)

    def test_compare_latency_returns_scores_without_winner(self) -> None:
        metrics = evaluator.compare_latency(_run()["local"], _run()["remote"])

        self.assertEqual(metrics["local_ms"], 800.0)
        self.assertEqual(metrics["remote_ms"], 1400.0)
        self.assertEqual(metrics["delta_ms"], -600.0)
        self.assertGreater(metrics["local_speed_score"], metrics["remote_speed_score"])
        self.assertNotIn("winner", metrics)

    def test_estimate_quality_penalizes_errors(self) -> None:
        success = evaluator.estimate_quality(_run()["prompt"], _run()["local"])
        failure = evaluator.estimate_quality(_run()["prompt"], {"error": "timeout"})

        self.assertGreater(success["score"], 0)
        self.assertLessEqual(success["score"], 1)
        self.assertEqual(failure["score"], 0)
        self.assertTrue(failure["has_error"])

    def test_evaluate_run_returns_metrics_only(self) -> None:
        result = evaluator.evaluate_run(_run())

        self.assertEqual(result["evaluation_method"], "rule")
        self.assertNotIn("winner", result)
        self.assertEqual(
            set(result["metrics"]),
            {
                "response_length",
                "latency",
                "structural_validity",
                "estimated_quality",
            },
        )
        self.assertTrue(result["metrics"]["structural_validity"]["local_valid"])
        self.assertTrue(result["metrics"]["structural_validity"]["remote_valid"])

    def test_evaluate_dataset_is_deterministic(self) -> None:
        first = evaluator.evaluate_dataset([_run()])
        second = evaluator.evaluate_dataset([_run()])

        self.assertEqual(first, second)

    def test_choose_winner_is_reserved_for_decision_engine(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "decision_engine"):
            evaluator.choose_winner({})


if __name__ == "__main__":
    unittest.main()
