"""Unit tests for the Benchmark Runner.

All model calls (ollama.generate, remote_llm.generate) are mocked so that no
real network requests are made.  Tests verify:

  - Prompt loading (happy path, bad data, missing files).
  - Single-prompt pipeline (correct shapes, error handling).
  - Aggregate statistics computation.
  - File persistence (results and summary serialisation).
  - Console output smoketest.
  - run_benchmark() integration (fully mocked pipeline).
"""

from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.benchmark import (
    ROUTER_VERSION,
    BenchmarkRecord,
    BenchmarkSummary,
    PromptEntry,
    _compute_summary,
    _parse_prompt_entry,
    _record_to_dict,
    _results_filename,
    _summaries_filename,
    _utc_now,
    benchmark_one,
    load_prompt_files,
    run_benchmark,
    save_results,
    save_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(**kwargs: Any) -> PromptEntry:
    defaults: dict[str, Any] = {
        "id": "coding_001",
        "category": "coding",
        "difficulty": "easy",
        "expected_reasoning": "low",
        "prompt": "Write hello world in Python.",
    }
    defaults.update(kwargs)
    return PromptEntry(**defaults)


def _make_record(**kwargs: Any) -> BenchmarkRecord:
    defaults: dict[str, Any] = {
        "id": "coding_001",
        "category": "coding",
        "difficulty": "easy",
        "expected_reasoning": "low",
        "prompt": "Write hello world in Python.",
        "features": {"reasoning_score": 0, "task_type": "coding"},
        "router": {
            "provider": "local",
            "routing_score": 9,
            "confidence": 0.97,
            "reason": ["Contains code", "Task type is coding"],
        },
        "local": {
            "model": "llama3",
            "response": "print('Hello, World!')",
            "latency_ms": 400,
            "estimated_input_tokens": 8,
            "estimated_output_tokens": 6,
        },
        "remote": {
            "model": "gpt-4o",
            "response": "print('Hello, World!')",
            "latency_ms": 800,
            "estimated_input_tokens": 8,
            "estimated_output_tokens": 6,
        },
        "evaluation": {
            "metrics": {},
            "evaluation_method": "rule",
        },
        "router_correct": None,
        "router_version": ROUTER_VERSION,
        "timestamp": "2026-07-03T00:00:00+00:00",
    }
    defaults.update(kwargs)
    return BenchmarkRecord(**defaults)


def _make_provider_result(response: str = "Hello!", latency: int = 500) -> dict[str, Any]:
    return {
        "model": "test-model",
        "response": response,
        "latency_ms": latency,
        "estimated_input_tokens": 10,
        "estimated_output_tokens": 5,
    }


# ---------------------------------------------------------------------------
# PromptEntry parsing tests
# ---------------------------------------------------------------------------

class ParsePromptEntryTests(unittest.TestCase):
    """_parse_prompt_entry validates required fields correctly."""

    def test_valid_entry_returns_prompt_entry(self) -> None:
        item = {
            "id": "coding_001",
            "category": "coding",
            "difficulty": "easy",
            "expected_reasoning": "low",
            "prompt": "Write hello world.",
        }
        entry = _parse_prompt_entry(item, "coding.json")
        self.assertIsInstance(entry, PromptEntry)
        self.assertEqual(entry.id, "coding_001")
        self.assertEqual(entry.category, "coding")

    def test_non_dict_returns_none(self) -> None:
        self.assertIsNone(_parse_prompt_entry("not a dict", "x.json"))
        self.assertIsNone(_parse_prompt_entry(42, "x.json"))
        self.assertIsNone(_parse_prompt_entry(None, "x.json"))

    def test_missing_id_returns_none(self) -> None:
        item = {
            "category": "coding",
            "difficulty": "easy",
            "expected_reasoning": "low",
            "prompt": "Write hello world.",
        }
        self.assertIsNone(_parse_prompt_entry(item, "coding.json"))

    def test_missing_prompt_returns_none(self) -> None:
        item = {
            "id": "coding_001",
            "category": "coding",
            "difficulty": "easy",
            "expected_reasoning": "low",
        }
        self.assertIsNone(_parse_prompt_entry(item, "coding.json"))

    def test_empty_prompt_returns_none(self) -> None:
        item = {
            "id": "coding_001",
            "category": "coding",
            "difficulty": "easy",
            "expected_reasoning": "low",
            "prompt": "",
        }
        self.assertIsNone(_parse_prompt_entry(item, "coding.json"))

    def test_all_fields_preserved(self) -> None:
        item = {
            "id": "math_007",
            "category": "mathematics",
            "difficulty": "hard",
            "expected_reasoning": "high",
            "prompt": "Prove Fermat's last theorem.",
        }
        entry = _parse_prompt_entry(item, "mathematics.json")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.difficulty, "hard")
        self.assertEqual(entry.expected_reasoning, "high")


# ---------------------------------------------------------------------------
# Prompt file loading tests
# ---------------------------------------------------------------------------

class LoadPromptFilesTests(unittest.TestCase):
    """load_prompt_files reads structured JSON files and skips bad ones."""

    def _write_json(self, directory: Path, filename: str, data: Any) -> None:
        (directory / filename).write_text(
            json.dumps(data, ensure_ascii=False), encoding="utf-8"
        )

    def test_missing_directory_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_prompt_files(Path("/nonexistent/path/that/cannot/exist"))

    def test_loads_valid_file(self, tmp_path=None) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_json(d, "coding.json", [
                {
                    "id": "coding_001",
                    "category": "coding",
                    "difficulty": "easy",
                    "expected_reasoning": "low",
                    "prompt": "Write hello world in Python.",
                }
            ])
            entries = load_prompt_files(d)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "coding_001")

    def test_skips_example_prompts(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # example_prompts.json has a plain string array — must be skipped
            self._write_json(d, "example_prompts.json", ["hello", "world"])
            entries = load_prompt_files(d)
        self.assertEqual(entries, [])

    def test_skips_invalid_json(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "broken.json").write_text("{ not valid json", encoding="utf-8")
            entries = load_prompt_files(d)
        self.assertEqual(entries, [])

    def test_skips_non_list_root(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_json(d, "coding.json", {"id": "coding_001"})
            entries = load_prompt_files(d)
        self.assertEqual(entries, [])

    def test_loads_multiple_files(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for cat in ("coding", "mathematics"):
                self._write_json(d, f"{cat}.json", [
                    {
                        "id": f"{cat}_001",
                        "category": cat,
                        "difficulty": "easy",
                        "expected_reasoning": "low",
                        "prompt": f"A {cat} prompt.",
                    }
                ])
            entries = load_prompt_files(d)
        self.assertEqual(len(entries), 2)
        ids = {e.id for e in entries}
        self.assertIn("coding_001", ids)
        self.assertIn("mathematics_001", ids)

    def test_skips_malformed_entries_continues(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            self._write_json(d, "mixed.json", [
                {
                    "id": "mixed_001",
                    "category": "general",
                    "difficulty": "easy",
                    "expected_reasoning": "low",
                    "prompt": "Good prompt.",
                },
                {"id": "mixed_002"},  # missing required fields
                "not a dict",
            ])
            entries = load_prompt_files(d)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].id, "mixed_001")


# ---------------------------------------------------------------------------
# benchmark_one tests
# ---------------------------------------------------------------------------

_MOCK_LOCAL_RESPONSE = "print('Hello, World!')"
_MOCK_REMOTE_RESPONSE = "# Hello world\nprint('Hello, World!')"


class BenchmarkOneTests(unittest.IsolatedAsyncioTestCase):
    """benchmark_one runs the full pipeline and returns a well-formed record."""

    async def _run_with_mocks(
        self,
        entry: PromptEntry | None = None,
        local_response: str = _MOCK_LOCAL_RESPONSE,
        remote_response: str = _MOCK_REMOTE_RESPONSE,
        local_error: Exception | None = None,
        remote_error: Exception | None = None,
    ) -> BenchmarkRecord:
        if entry is None:
            entry = _make_entry()

        async def fake_local(_prompt: str) -> str:
            if local_error:
                raise local_error
            return local_response

        async def fake_remote(_prompt: str) -> str:
            if remote_error:
                raise remote_error
            return remote_response

        with (
            patch("app.services.benchmark.ollama.generate", side_effect=fake_local),
            patch("app.services.benchmark.remote_llm.generate", side_effect=fake_remote),
        ):
            return await benchmark_one(entry)

    async def test_returns_benchmark_record(self) -> None:
        record = await self._run_with_mocks()
        self.assertIsInstance(record, BenchmarkRecord)

    async def test_id_matches_entry(self) -> None:
        entry = _make_entry(id="coding_042")
        record = await self._run_with_mocks(entry=entry)
        self.assertEqual(record.id, "coding_042")

    async def test_router_correct_is_none(self) -> None:
        record = await self._run_with_mocks()
        self.assertIsNone(record.router_correct)

    async def test_router_version_is_v1(self) -> None:
        record = await self._run_with_mocks()
        self.assertEqual(record.router_version, ROUTER_VERSION)

    async def test_timestamp_is_iso8601_string(self) -> None:
        record = await self._run_with_mocks()
        self.assertIsInstance(record.timestamp, str)
        # Must be parseable
        datetime.fromisoformat(record.timestamp)

    async def test_features_dict_present(self) -> None:
        record = await self._run_with_mocks()
        self.assertIsInstance(record.features, dict)
        self.assertIn("reasoning_score", record.features)

    async def test_router_dict_has_required_keys(self) -> None:
        record = await self._run_with_mocks()
        for key in ("provider", "routing_score", "confidence", "reason"):
            self.assertIn(key, record.router, msg=f"router missing key: {key}")

    async def test_router_provider_is_valid(self) -> None:
        record = await self._run_with_mocks()
        self.assertIn(record.router["provider"], {"local", "remote"})

    async def test_local_block_contains_response(self) -> None:
        record = await self._run_with_mocks(local_response="local answer")
        self.assertEqual(record.local.get("response"), "local answer")

    async def test_remote_block_contains_response(self) -> None:
        record = await self._run_with_mocks(remote_response="remote answer")
        self.assertEqual(record.remote.get("response"), "remote answer")

    async def test_evaluation_block_has_metrics(self) -> None:
        record = await self._run_with_mocks()
        self.assertIn("metrics", record.evaluation)
        metrics = record.evaluation["metrics"]
        self.assertIn("latency", metrics)
        self.assertIn("response_length", metrics)
        self.assertIn("structural_validity", metrics)

    async def test_local_error_gracefully_captured(self) -> None:
        """If local provider fails, the record still completes."""
        record = await self._run_with_mocks(
            local_error=ConnectionError("Ollama not reachable")
        )
        self.assertIsInstance(record, BenchmarkRecord)
        self.assertIn("error", record.local)

    async def test_remote_error_gracefully_captured(self) -> None:
        """If remote provider fails, the record still completes."""
        record = await self._run_with_mocks(
            remote_error=RuntimeError("Rate limit exceeded")
        )
        self.assertIsInstance(record, BenchmarkRecord)
        self.assertIn("error", record.remote)

    async def test_both_providers_always_executed(self) -> None:
        """Both providers must be called even when routing says local."""
        local_called = []
        remote_called = []

        async def fake_local(p: str) -> str:
            local_called.append(p)
            return "local"

        async def fake_remote(p: str) -> str:
            remote_called.append(p)
            return "remote"

        # Use a prompt that routes local
        entry = _make_entry(prompt="Hi there!")
        with (
            patch("app.services.benchmark.ollama.generate", side_effect=fake_local),
            patch("app.services.benchmark.remote_llm.generate", side_effect=fake_remote),
        ):
            record = await benchmark_one(entry)

        self.assertEqual(record.router["provider"], "local")  # router chose local
        self.assertEqual(len(local_called), 1, "local must be called once")
        self.assertEqual(len(remote_called), 1, "remote must be called even when router chose local")

    async def test_prompt_preserved_in_record(self) -> None:
        entry = _make_entry(prompt="What is the meaning of life?")
        record = await self._run_with_mocks(entry=entry)
        self.assertEqual(record.prompt, "What is the meaning of life?")


# ---------------------------------------------------------------------------
# _compute_summary tests
# ---------------------------------------------------------------------------

class ComputeSummaryTests(unittest.TestCase):
    """_compute_summary correctly aggregates records into statistics."""

    def _ts(self) -> datetime:
        return datetime(2026, 7, 3, 0, 0, 0, tzinfo=timezone.utc)

    def test_empty_records_returns_zero_summary(self) -> None:
        summary = _compute_summary([], [], 5.0, self._ts())
        self.assertEqual(summary.total_prompts, 0)
        self.assertEqual(summary.local_selected, 0)
        self.assertEqual(summary.remote_selected, 0)

    def test_total_prompts_correct(self) -> None:
        records = [_make_record(id=f"coding_{i:03d}") for i in range(5)]
        summary = _compute_summary(records, [], 10.0, self._ts())
        self.assertEqual(summary.total_prompts, 5)

    def test_local_and_remote_counts(self) -> None:
        r_local = _make_record(router={"provider": "local", "routing_score": 5, "confidence": 0.9, "reason": []})
        r_remote = _make_record(router={"provider": "remote", "routing_score": 40, "confidence": 0.95, "reason": []})
        summary = _compute_summary([r_local, r_remote, r_local], [], 3.0, self._ts())
        self.assertEqual(summary.local_selected, 2)
        self.assertEqual(summary.remote_selected, 1)

    def test_average_routing_score(self) -> None:
        records = [
            _make_record(router={"provider": "local", "routing_score": 10, "confidence": 0.8, "reason": []}),
            _make_record(router={"provider": "remote", "routing_score": 30, "confidence": 0.9, "reason": []}),
        ]
        summary = _compute_summary(records, [], 1.0, self._ts())
        self.assertAlmostEqual(summary.average_routing_score, 20.0)

    def test_average_confidence(self) -> None:
        records = [
            _make_record(router={"provider": "local", "routing_score": 5, "confidence": 0.8, "reason": []}),
            _make_record(router={"provider": "local", "routing_score": 5, "confidence": 0.6, "reason": []}),
        ]
        summary = _compute_summary(records, [], 1.0, self._ts())
        self.assertAlmostEqual(summary.average_confidence, 0.7)

    def test_categories_processed_are_sorted_unique(self) -> None:
        records = [
            _make_record(category="coding"),
            _make_record(category="mathematics"),
            _make_record(category="coding"),
        ]
        summary = _compute_summary(records, [], 1.0, self._ts())
        self.assertEqual(summary.categories_processed, ["coding", "mathematics"])

    def test_difficulty_distribution(self) -> None:
        records = [
            _make_record(difficulty="easy"),
            _make_record(difficulty="easy"),
            _make_record(difficulty="medium"),
            _make_record(difficulty="hard"),
        ]
        summary = _compute_summary(records, [], 1.0, self._ts())
        self.assertEqual(summary.difficulty_distribution["easy"], 2)
        self.assertEqual(summary.difficulty_distribution["medium"], 1)
        self.assertEqual(summary.difficulty_distribution["hard"], 1)

    def test_average_latencies(self) -> None:
        records = [
            _make_record(
                local={"model": "m", "response": "r", "latency_ms": 400, "estimated_input_tokens": 5, "estimated_output_tokens": 5},
                remote={"model": "m", "response": "r", "latency_ms": 800, "estimated_input_tokens": 5, "estimated_output_tokens": 5},
            ),
            _make_record(
                local={"model": "m", "response": "r", "latency_ms": 600, "estimated_input_tokens": 5, "estimated_output_tokens": 5},
                remote={"model": "m", "response": "r", "latency_ms": 1200, "estimated_input_tokens": 5, "estimated_output_tokens": 5},
            ),
        ]
        summary = _compute_summary(records, [], 1.0, self._ts())
        self.assertAlmostEqual(summary.average_local_latency_ms, 500.0)
        self.assertAlmostEqual(summary.average_remote_latency_ms, 1000.0)

    def test_errors_preserved(self) -> None:
        errors = ["coding_001: TimeoutError", "math_002: ConnectionError"]
        summary = _compute_summary([], errors, 1.0, self._ts())
        self.assertEqual(summary.errors, errors)

    def test_router_version(self) -> None:
        summary = _compute_summary([], [], 1.0, self._ts())
        self.assertEqual(summary.router_version, ROUTER_VERSION)

    def test_duration_preserved(self) -> None:
        summary = _compute_summary([], [], 42.7, self._ts())
        self.assertAlmostEqual(summary.benchmark_duration_seconds, 42.7)


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

class PersistenceTests(unittest.TestCase):
    """save_results and save_summary write correct JSON files."""

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.mkdtemp()
        self._tmp_path = Path(self._tmp)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _ts(self) -> datetime:
        return datetime(2026, 7, 3, 12, 30, 0, tzinfo=timezone.utc)

    def test_save_results_creates_file(self) -> None:
        records = [_make_record()]
        rd = self._tmp_path / "results"
        path = save_results(records, rd, self._ts())
        self.assertTrue(path.exists())

    def test_save_results_filename_format(self) -> None:
        rd = self._tmp_path / "results"
        path = save_results([_make_record()], rd, self._ts())
        self.assertEqual(path.name, "benchmark_2026-07-03T12-30-00.json")

    def test_save_results_valid_json(self) -> None:
        rd = self._tmp_path / "results"
        path = save_results([_make_record()], rd, self._ts())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)

    def test_save_results_record_schema(self) -> None:
        rd = self._tmp_path / "results"
        path = save_results([_make_record()], rd, self._ts())
        data = json.loads(path.read_text(encoding="utf-8"))
        record = data[0]
        for key in ("id", "category", "difficulty", "expected_reasoning",
                    "prompt", "features", "router", "local", "remote",
                    "evaluation", "router_correct", "router_version", "timestamp"):
            self.assertIn(key, record, msg=f"record missing key: {key}")

    def test_save_results_router_correct_is_null(self) -> None:
        rd = self._tmp_path / "results"
        path = save_results([_make_record()], rd, self._ts())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsNone(data[0]["router_correct"])

    def test_save_summary_creates_file(self) -> None:
        sd = self._tmp_path / "summaries"
        summary = _compute_summary([_make_record()], [], 1.0, self._ts())
        path = save_summary(summary, sd, self._ts())
        self.assertTrue(path.exists())

    def test_save_summary_filename_format(self) -> None:
        sd = self._tmp_path / "summaries"
        summary = _compute_summary([], [], 1.0, self._ts())
        path = save_summary(summary, sd, self._ts())
        self.assertEqual(path.name, "summary_2026-07-03T12-30-00.json")

    def test_save_summary_valid_json(self) -> None:
        sd = self._tmp_path / "summaries"
        summary = _compute_summary([], [], 1.0, self._ts())
        path = save_summary(summary, sd, self._ts())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)

    def test_save_summary_schema(self) -> None:
        sd = self._tmp_path / "summaries"
        summary = _compute_summary([], [], 1.0, self._ts())
        path = save_summary(summary, sd, self._ts())
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "total_prompts", "categories_processed", "difficulty_distribution",
            "router_version", "local_selected", "remote_selected",
            "average_routing_score", "average_confidence",
            "average_local_latency_ms", "average_remote_latency_ms",
            "average_local_tokens", "average_remote_tokens",
            "benchmark_duration_seconds", "timestamp",
        ):
            self.assertIn(key, data, msg=f"summary missing key: {key}")

    def test_save_results_creates_parent_dirs(self) -> None:
        deep_rd = self._tmp_path / "a" / "b" / "c" / "results"
        save_results([], deep_rd, self._ts())
        self.assertTrue(deep_rd.exists())

    def test_save_summary_creates_parent_dirs(self) -> None:
        deep_sd = self._tmp_path / "a" / "b" / "c" / "summaries"
        summary = _compute_summary([], [], 1.0, self._ts())
        save_summary(summary, deep_sd, self._ts())
        self.assertTrue(deep_sd.exists())


# ---------------------------------------------------------------------------
# Filename utility tests
# ---------------------------------------------------------------------------

class FilenameTests(unittest.TestCase):
    """_results_filename and _summaries_filename produce correct names."""

    def _ts(self) -> datetime:
        return datetime(2026, 7, 2, 18, 30, 0, tzinfo=timezone.utc)

    def test_results_filename(self) -> None:
        self.assertEqual(
            _results_filename(self._ts()),
            "benchmark_2026-07-02T18-30-00.json",
        )

    def test_summaries_filename(self) -> None:
        self.assertEqual(
            _summaries_filename(self._ts()),
            "summary_2026-07-02T18-30-00.json",
        )

    def test_utc_now_is_string(self) -> None:
        result = _utc_now()
        self.assertIsInstance(result, str)
        datetime.fromisoformat(result)


# ---------------------------------------------------------------------------
# _record_to_dict tests
# ---------------------------------------------------------------------------

class RecordToDictTests(unittest.TestCase):
    """_record_to_dict converts BenchmarkRecord to a serialisable dict."""

    def test_returns_dict(self) -> None:
        r = _make_record()
        d = _record_to_dict(r)
        self.assertIsInstance(d, dict)

    def test_router_correct_is_none(self) -> None:
        r = _make_record()
        d = _record_to_dict(r)
        self.assertIsNone(d["router_correct"])

    def test_all_required_keys_present(self) -> None:
        r = _make_record()
        d = _record_to_dict(r)
        for key in ("id", "category", "difficulty", "expected_reasoning",
                    "prompt", "features", "router", "local", "remote",
                    "evaluation", "router_correct", "router_version", "timestamp"):
            self.assertIn(key, d)


# ---------------------------------------------------------------------------
# run_benchmark integration test
# ---------------------------------------------------------------------------

class RunBenchmarkIntegrationTests(unittest.IsolatedAsyncioTestCase):
    """run_benchmark() integration test with fully mocked providers and file I/O."""

    def _write_prompt_file(self, directory: Path) -> None:
        prompts = [
            {
                "id": "coding_001",
                "category": "coding",
                "difficulty": "easy",
                "expected_reasoning": "low",
                "prompt": "Write hello world in Python.",
            },
            {
                "id": "mathematics_001",
                "category": "mathematics",
                "difficulty": "medium",
                "expected_reasoning": "medium",
                "prompt": "Solve x^2 - 4 = 0.",
            },
        ]
        (directory / "coding.json").write_text(
            json.dumps([prompts[0]]), encoding="utf-8"
        )
        (directory / "mathematics.json").write_text(
            json.dumps([prompts[1]]), encoding="utf-8"
        )

    async def test_run_benchmark_returns_summary(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)
            rd = d / "results"
            sd = d / "summaries"

            async def fake_local(_p: str) -> str:
                return "local response"

            async def fake_remote(_p: str) -> str:
                return "remote response"

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=fake_local),
                patch("app.services.benchmark.remote_llm.generate", side_effect=fake_remote),
            ):
                summary = await run_benchmark(pd, rd, sd)

        self.assertIsInstance(summary, BenchmarkSummary)
        self.assertEqual(summary.total_prompts, 2)

    async def test_run_benchmark_saves_results_file(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)
            rd = d / "results"
            sd = d / "summaries"

            async def fake_both(_p: str) -> str:
                return "answer"

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=fake_both),
                patch("app.services.benchmark.remote_llm.generate", side_effect=fake_both),
            ):
                await run_benchmark(pd, rd, sd)

            result_files = list(rd.glob("benchmark_*.json"))
        self.assertEqual(len(result_files), 1)

    async def test_run_benchmark_saves_summary_file(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)
            rd = d / "results"
            sd = d / "summaries"

            async def fake_both(_p: str) -> str:
                return "answer"

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=fake_both),
                patch("app.services.benchmark.remote_llm.generate", side_effect=fake_both),
            ):
                await run_benchmark(pd, rd, sd)

            summary_files = list(sd.glob("summary_*.json"))
        self.assertEqual(len(summary_files), 1)

    async def test_run_benchmark_continues_on_provider_error(self) -> None:
        """If one provider always fails, the benchmark still completes."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)
            rd = d / "results"
            sd = d / "summaries"

            async def good(_p: str) -> str:
                return "ok"

            async def bad(_p: str) -> str:
                raise ConnectionError("Ollama down")

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=bad),
                patch("app.services.benchmark.remote_llm.generate", side_effect=good),
            ):
                summary = await run_benchmark(pd, rd, sd)

        # Both prompts should still produce records (with error in local block)
        self.assertEqual(summary.total_prompts, 2)

    async def test_summary_has_correct_categories(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)
            rd = d / "results"
            sd = d / "summaries"

            async def fake(_p: str) -> str:
                return "answer"

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=fake),
                patch("app.services.benchmark.remote_llm.generate", side_effect=fake),
            ):
                summary = await run_benchmark(pd, rd, sd)

        self.assertIn("coding", summary.categories_processed)
        self.assertIn("mathematics", summary.categories_processed)

    async def test_router_version_in_summary(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd = d / "prompts"
            pd.mkdir()
            self._write_prompt_file(pd)

            async def fake(_p: str) -> str:
                return "answer"

            with (
                patch("app.services.benchmark.ollama.generate", side_effect=fake),
                patch("app.services.benchmark.remote_llm.generate", side_effect=fake),
            ):
                summary = await run_benchmark(pd, d / "r", d / "s")

        self.assertEqual(summary.router_version, ROUTER_VERSION)


if __name__ == "__main__":
    unittest.main()
