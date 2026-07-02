"""Project Health Check utility for the Hybrid Token Router.

This script performs read-only diagnostics. It checks configuration,
dependencies, local/remote provider reachability, prompt datasets, imports,
tests, and a one-prompt router pipeline.

It never modifies project files or starts a persistent server.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"

CHECK = "[OK]"
CROSS = "[FAIL]"
INFO = "[INFO]"

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
APP = BACKEND / "app"
ENV_PATH = BACKEND / ".env"
REQUIREMENTS = BACKEND / "requirements.txt"

REQUIRED_ENV_VARS = [
    "FRONTEND_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "REMOTE_API_KEY",
    "REMOTE_BASE_URL",
    "REMOTE_MODEL",
]

REQUIRED_FOLDERS = [
    BACKEND / "app" / "services",
    BACKEND / "app" / "tests",
    BACKEND / "app" / "data" / "prompts",
    BACKEND / "app" / "data" / "raw_runs",
    BACKEND / "app" / "data" / "evaluations",
    BACKEND / "app" / "data" / "benchmarks",
    BACKEND / "app" / "data" / "processed",
    BACKEND / "app" / "data" / "training",
    BACKEND / "docs",
]

PROMPT_FILES = [
    "coding.json",
    "mathematics.json",
    "reasoning.json",
    "planning.json",
    "translation.json",
    "summarization.json",
    "creative_writing.json",
    "general.json",
]

CORE_MODULES = [
    "feature_extractor.py",
    "router.py",
    "dataset_generator.py",
    "evaluator.py",
    "benchmark.py",
]


@dataclass
class CheckResult:
    """Result for one top-level health check."""

    name: str
    status: str = PASS
    details: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.status = FAIL
        self.details.append(f"{CROSS} {message}")

    def warn(self, message: str) -> None:
        if self.status != FAIL:
            self.status = WARN
        self.details.append(f"{INFO} {message}")

    def ok(self, message: str) -> None:
        self.details.append(f"{CHECK} {message}")


def add_backend_to_path() -> None:
    backend_str = str(BACKEND)
    if backend_str not in sys.path:
        sys.path.insert(0, backend_str)


def read_env_file(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        env[key.strip()] = value.strip().strip("\"'")
    return env


def http_json(url: str, headers: dict[str, str] | None = None, timeout: float = 5.0) -> tuple[int, Any]:
    request = Request(url, headers=headers or {}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
        if not payload:
            return response.status, None
        return response.status, json.loads(payload)


def check_python_environment() -> CheckResult:
    result = CheckResult("Python")
    version = sys.version_info
    result.ok(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version < (3, 10):
        result.fail("Python 3.10 or newer is recommended for this project.")

    if os.environ.get("VIRTUAL_ENV"):
        result.ok(f"Virtual environment detected: {Path(os.environ['VIRTUAL_ENV']).name}")
    else:
        result.warn("No active virtual environment detected. This is optional, but recommended.")

    if not REQUIREMENTS.exists():
        result.fail("backend/requirements.txt is missing.")
        return result

    try:
        raw_requirements = REQUIREMENTS.read_text(encoding="utf-8")
        result.ok("backend/requirements.txt is readable.")
    except OSError as exc:
        result.fail(f"backend/requirements.txt is not readable: {exc}")
        return result

    packages = parse_requirements(raw_requirements)
    if not packages:
        result.warn("backend/requirements.txt contains no package entries.")
        return result

    missing = []
    for package_name in packages:
        module_name = package_to_module(package_name)
        if importlib.util.find_spec(module_name) is None:
            missing.append(package_name)

    if missing:
        result.fail("Required packages not installed: " + ", ".join(missing))
    else:
        result.ok(f"Required packages installed: {len(packages)} checked.")

    return result


def parse_requirements(text: str) -> list[str]:
    packages: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("-"):
            continue
        package = re.split(r"[<>=!~\[]", stripped, maxsplit=1)[0].strip()
        if package:
            packages.append(package)
    return packages


def package_to_module(package_name: str) -> str:
    known = {
        "python-dotenv": "dotenv",
        "pydantic-settings": "pydantic_settings",
        "beautifulsoup4": "bs4",
        "scikit-learn": "sklearn",
    }
    return known.get(package_name.lower(), package_name.replace("-", "_"))


def check_environment_variables(env: dict[str, str]) -> CheckResult:
    result = CheckResult("Environment")
    if ENV_PATH.exists():
        result.ok("backend/.env exists.")
    else:
        result.fail("backend/.env is missing.")

    for key in REQUIRED_ENV_VARS:
        if env.get(key):
            result.ok(f"{key}: Present")
        else:
            result.fail(f"{key}: Missing")

    return result


def check_ollama(env: dict[str, str]) -> CheckResult:
    result = CheckResult("Ollama")
    executable = shutil.which("ollama")
    if executable:
        result.ok(f"Ollama executable found: {executable}")
    else:
        result.fail("Ollama executable not found on PATH.")

    base_url = env.get("OLLAMA_BASE_URL", "").rstrip("/")
    model = env.get("OLLAMA_MODEL", "")
    if not base_url:
        result.fail("OLLAMA_BASE_URL is missing, so server reachability cannot be checked.")
        return result

    try:
        status, data = http_json(f"{base_url}/api/tags")
        result.ok(f"Ollama server reachable at {base_url} (HTTP {status}).")
    except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
        result.fail(f"Ollama server is not reachable at {base_url}: {exc}")
        result.details.append(f"{INFO} Start Ollama with `ollama serve` and verify OLLAMA_BASE_URL.")
        return result

    if not model:
        result.fail("OLLAMA_MODEL is missing.")
        return result

    models = data.get("models", []) if isinstance(data, dict) else []
    names = {item.get("name", "") for item in models if isinstance(item, dict)}
    model_without_tag = model.split(":", 1)[0]
    if model in names or any(name.split(":", 1)[0] == model_without_tag for name in names):
        result.ok(f"Configured Ollama model exists: {model}")
    else:
        result.fail(f"Configured Ollama model was not found locally: {model}")
        result.details.append(f"{INFO} Download it with `ollama pull {model}`.")

    return result


def check_remote_provider(env: dict[str, str]) -> CheckResult:
    result = CheckResult("Remote Provider")
    api_key = env.get("REMOTE_API_KEY", "")
    base_url = env.get("REMOTE_BASE_URL", "").rstrip("/")
    model = env.get("REMOTE_MODEL", "")

    if api_key:
        result.ok("REMOTE_API_KEY: Present")
    else:
        result.fail("REMOTE_API_KEY: Missing")

    if not base_url:
        result.fail("REMOTE_BASE_URL: Missing")
        return result

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        result.fail(f"REMOTE_BASE_URL is not a valid HTTP URL: {base_url}")
        return result
    result.ok("REMOTE_BASE_URL is a valid URL.")

    if model:
        result.ok("REMOTE_MODEL: Present")
    else:
        result.fail("REMOTE_MODEL: Missing")

    if not api_key:
        result.warn("Skipping remote connectivity check because the API key is missing.")
        return result

    models_url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        status, _ = http_json(models_url, headers=headers)
        result.ok(f"Remote provider models endpoint reachable (HTTP {status}).")
    except HTTPError as exc:
        if exc.code in {401, 403}:
            result.fail(f"Remote provider rejected credentials at /models (HTTP {exc.code}).")
        elif exc.code == 404:
            result.warn("Remote provider does not expose /models; skipped token-free connectivity check.")
        else:
            result.fail(f"Remote provider connectivity check failed at /models (HTTP {exc.code}).")
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        result.fail(f"Remote provider connectivity check failed: {exc}")

    return result


def check_folder_structure() -> CheckResult:
    result = CheckResult("Folder Structure")
    for folder in REQUIRED_FOLDERS:
        rel = folder.relative_to(ROOT)
        if folder.exists() and folder.is_dir():
            result.ok(f"{rel} exists.")
        else:
            result.fail(f"{rel} is missing.")
    return result


def check_prompt_dataset() -> CheckResult:
    result = CheckResult("Prompt Dataset")
    prompts_dir = APP / "data" / "prompts"
    all_ids: set[str] = set()
    total_prompts = 0
    difficulty_distribution: Counter[str] = Counter()

    for filename in PROMPT_FILES:
        path = prompts_dir / filename
        category = path.stem
        if not path.exists():
            result.fail(f"{filename} is missing.")
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            result.fail(f"{filename} is not valid readable JSON: {exc}")
            continue

        if not isinstance(data, list):
            result.fail(f"{filename} must contain a JSON list.")
            continue

        file_ids: set[str] = set()
        file_difficulties: Counter[str] = Counter()
        for index, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                result.fail(f"{filename} item {index} is not an object.")
                continue

            prompt_id = str(item.get("id", "")).strip()
            if not prompt_id:
                result.fail(f"{filename} item {index} has no id.")
            elif prompt_id in file_ids or prompt_id in all_ids:
                result.fail(f"Duplicate prompt id found: {prompt_id}")
            else:
                file_ids.add(prompt_id)
                all_ids.add(prompt_id)

            item_category = item.get("category")
            if item_category != category:
                result.fail(
                    f"{filename} item {prompt_id or index} category is {item_category!r}, expected {category!r}."
                )

            difficulty = str(item.get("difficulty", "missing"))
            file_difficulties[difficulty] += 1

        total_prompts += len(data)
        difficulty_distribution.update(file_difficulties)
        result.ok(
            f"{filename}: {len(data)} prompts; difficulty distribution {dict(sorted(file_difficulties.items()))}"
        )

    if total_prompts:
        result.ok(f"Total prompt count: {total_prompts}")
        result.ok(f"Overall difficulty distribution: {dict(sorted(difficulty_distribution.items()))}")
    else:
        result.fail("No prompts were loaded from the required dataset files.")

    return result


def check_core_modules() -> CheckResult:
    result = CheckResult("Core Modules")
    services = APP / "services"
    for filename in CORE_MODULES:
        path = services / filename
        if path.exists() and path.is_file():
            result.ok(f"{filename} exists.")
        else:
            result.fail(f"{filename} is missing.")
    return result


def check_unit_tests() -> CheckResult:
    result = CheckResult("Tests")
    command = [sys.executable, "-m", "pytest", str(APP / "tests")]
    try:
        completed = subprocess.run(
            command,
            cwd=BACKEND,
            text=True,
            capture_output=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        result.fail("Python executable was not found.")
        return result
    except subprocess.TimeoutExpired:
        result.fail("pytest timed out after 120 seconds.")
        return result

    output = (completed.stdout or "") + "\n" + (completed.stderr or "")
    counts = parse_pytest_counts(output)
    result.details.append(f"{INFO} pytest exit code: {completed.returncode}")
    result.details.append(
        f"{INFO} total={counts['total']} passed={counts['passed']} failed={counts['failed']} skipped={counts['skipped']}"
    )

    if completed.returncode == 0:
        result.ok("pytest completed successfully.")
    else:
        result.fail("pytest reported failures or could not run. See pytest output when running manually.")

    return result


def parse_pytest_counts(output: str) -> dict[str, int]:
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for key in counts:
        match = re.search(rf"(\d+)\s+{key}", output)
        if match:
            counts[key] = int(match.group(1))
    counts["total"] = sum(counts.values())
    return counts


def check_fastapi_import() -> CheckResult:
    result = CheckResult("FastAPI")
    add_backend_to_path()
    try:
        module = importlib.import_module("app.main")
    except Exception as exc:  # noqa: BLE001
        result.fail(f"app.main import failed: {type(exc).__name__}: {exc}")
        return result

    app = getattr(module, "app", None)
    if app is None:
        result.fail("app.main imported, but no `app` object was found.")
        return result

    try:
        from fastapi import FastAPI
    except Exception as exc:  # noqa: BLE001
        result.fail(f"FastAPI package import failed: {type(exc).__name__}: {exc}")
        return result

    if isinstance(app, FastAPI):
        result.ok("FastAPI application imported successfully.")
    else:
        result.fail("app.main.app is not a FastAPI instance.")

    return result


def check_router_pipeline() -> CheckResult:
    result = CheckResult("Router")
    add_backend_to_path()
    sample_prompt = (
        "Analyze whether a short Python function that validates user input should "
        "run locally or remotely, and explain the routing decision."
    )

    try:
        from app.services.feature_extractor import extract_features
        from app.services.router import route

        features = extract_features(sample_prompt)
        routing = route(features)
    except Exception as exc:  # noqa: BLE001
        result.fail(f"Router pipeline failed: {type(exc).__name__}: {exc}")
        return result

    reason = routing.get("reason", [])
    if isinstance(reason, list):
        reason_text = "; ".join(str(item) for item in reason)
    else:
        reason_text = str(reason)

    result.ok("Sample prompt completed through Feature Extractor -> Router.")
    result.details.extend(
        [
            f"{INFO} Task Type: {features.get('task_type')}",
            f"{INFO} Reasoning Score: {features.get('reasoning_score')}",
            f"{INFO} Complexity: {features.get('complexity')}",
            f"{INFO} Routing Score: {routing.get('routing_score')}",
            f"{INFO} Confidence: {routing.get('confidence')}",
            f"{INFO} Chosen Provider: {routing.get('provider')}",
            f"{INFO} Reason: {reason_text}",
        ]
    )
    return result


def check_feature_extractor_import() -> CheckResult:
    result = CheckResult("Feature Extractor")
    add_backend_to_path()
    try:
        module = importlib.import_module("app.services.feature_extractor")
        if callable(getattr(module, "extract_features", None)):
            result.ok("feature_extractor imports successfully and exposes extract_features().")
        else:
            result.fail("feature_extractor imports but does not expose extract_features().")
    except Exception as exc:  # noqa: BLE001
        result.fail(f"feature_extractor import failed: {type(exc).__name__}: {exc}")
    return result


def check_benchmark_import() -> CheckResult:
    result = CheckResult("Benchmark")
    add_backend_to_path()
    try:
        module = importlib.import_module("app.services.benchmark")
        if callable(getattr(module, "run_benchmark", None)):
            result.ok("benchmark imports successfully and exposes run_benchmark().")
        else:
            result.fail("benchmark imports but does not expose run_benchmark().")
    except Exception as exc:  # noqa: BLE001
        result.fail(f"benchmark import failed: {type(exc).__name__}: {exc}")
    return result


def print_section(title: str, result: CheckResult) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for detail in result.details:
        print(detail)
    print(f"Status: {result.status}")


def summary_status(result: CheckResult) -> str:
    return PASS if result.status == PASS else FAIL


def print_overall_summary(results: dict[str, CheckResult]) -> None:
    overall = "READY" if all(result.status == PASS for result in results.values()) else "NOT READY"

    summary_rows = [
        ("Python", results["Python"]),
        ("Environment", results["Environment"]),
        ("Ollama", results["Ollama"]),
        ("Remote Provider", results["Remote Provider"]),
        ("Prompt Dataset", results["Prompt Dataset"]),
        ("Router", results["Router"]),
        ("Feature Extractor", results["Feature Extractor"]),
        ("Benchmark", results["Benchmark"]),
        ("Tests", results["Tests"]),
        ("FastAPI", results["FastAPI"]),
    ]

    print("\n======================================")
    print("Hybrid Token Router Health Report")
    print("======================================")
    for label, result in summary_rows:
        print(f"{label + ' ':.<20} {summary_status(result)}")
    print(f"{'Overall Status ':.<20} {overall}")
    print("======================================")

    failures = [result for result in results.values() if result.status != PASS]
    if failures:
        print("\nFailures and warnings:")
        for result in failures:
            print(f"- {result.name}: {result.status}")
            for detail in result.details:
                if detail.startswith(CROSS) or result.status == WARN:
                    print(f"  {detail}")


def main() -> int:
    print("Running Hybrid Token Router project health checks...")
    env = read_env_file()

    ordered_results = [
        check_python_environment(),
        check_environment_variables(env),
        check_ollama(env),
        check_remote_provider(env),
        check_folder_structure(),
        check_prompt_dataset(),
        check_core_modules(),
        check_unit_tests(),
        check_fastapi_import(),
        check_router_pipeline(),
        check_benchmark_import(),
        check_feature_extractor_import(),
    ]

    results = {result.name: result for result in ordered_results}
    for result in ordered_results:
        print_section(result.name, result)

    print_overall_summary(results)
    return 0 if all(result.status == PASS for result in ordered_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
