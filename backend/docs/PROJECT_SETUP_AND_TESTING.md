# Hybrid Token Router Project Setup and Testing

This guide explains how to prepare, run, test, benchmark, and diagnose the Hybrid Token Router backend.

## 1. Project prerequisites

- Python 3.10 or newer
- `pip`
- A terminal opened at the project root
- Ollama installed locally for the local provider
- A remote provider API key for the remote provider

Project root:

```powershell
C:\Users\Amitosh Nigam\Desktop\Hybrid-Token-Router
```

Backend root:

```powershell
C:\Users\Amitosh Nigam\Desktop\Hybrid-Token-Router\backend
```

## 2. Create the virtual environment

From the project root:

```powershell
cd backend
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then activate again.

## 3. Install dependencies

From `backend` with the virtual environment active:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `requirements.txt` is empty, install the packages used by the backend:

```powershell
python -m pip install fastapi uvicorn httpx pytest
```

## 4. Configure `.env`

Create:

```powershell
backend\.env
```

Expected variables:

```env
FRONTEND_URL=http://localhost:5173
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
REMOTE_API_KEY=your_remote_provider_key
REMOTE_BASE_URL=https://api.groq.com/openai/v1
REMOTE_MODEL=llama-3.1-8b-instant
```

Do not commit real API keys.

## 5. Install Ollama

Download and install Ollama from:

```text
https://ollama.com/download
```

Verify installation:

```powershell
ollama --version
```

Start the Ollama server if it is not already running:

```powershell
ollama serve
```

## 6. Download the required model

Use the model configured in `OLLAMA_MODEL`.

Example:

```powershell
ollama pull llama3.1
```

Verify local models:

```powershell
ollama list
```

## 7. Start the FastAPI server

From `backend`:

```powershell
python -m uvicorn app.main:app --reload
```

Expected output includes:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

## 8. Open Swagger UI

With the backend server running, open:

```text
http://127.0.0.1:8000/docs
```

The OpenAPI JSON is available at:

```text
http://127.0.0.1:8000/openapi.json
```

## 9. Run unit tests

From `backend`:

```powershell
python -m pytest app/tests
```

Expected successful output:

```text
================ test session starts ================
...
================ 12 passed in 1.23s =================
```

The exact test count and duration can vary.

## 10. Run the benchmark

From `backend`:

```powershell
python -m app.services.benchmark
```

The benchmark reads prompts from:

```text
backend/app/data/prompts
```

It writes results under:

```text
backend/app/data/benchmarks/results
backend/app/data/benchmarks/summaries
```

Expected console output includes:

```text
Hybrid Token Router Benchmark
Total prompts        : ...
Local selected       : ...
Remote selected      : ...
Benchmark completed successfully.
```

## 11. Run the project health check

From the project root:

```powershell
python backend/tools/project_health_check.py
```

Or from `backend`:

```powershell
python tools/project_health_check.py
```

The utility only performs checks. It does not edit files, start a permanent server, or run the full benchmark.

Expected successful summary:

```text
======================================
Hybrid Token Router Health Report
======================================
Python ............. PASS
Environment ........ PASS
Ollama ............. PASS
Remote Provider .... PASS
Prompt Dataset ..... PASS
Router ............. PASS
Feature Extractor .. PASS
Benchmark .......... PASS
Tests .............. PASS
FastAPI ............ PASS
Overall Status ..... READY
======================================
```

If a check fails, the summary shows `FAIL` and the detailed section above it explains why.

## 12. Common errors and fixes

### `ModuleNotFoundError`

Cause:

- Dependencies are missing.
- Command was run from the wrong directory.
- The virtual environment is not active.

Fix:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Missing `.env`

Cause:

- `backend/.env` does not exist.

Fix:

```powershell
New-Item -ItemType File .env
```

Then add the required variables listed in this guide.

### Ollama not running

Cause:

- The Ollama server is not reachable at `OLLAMA_BASE_URL`.

Fix:

```powershell
ollama serve
```

Then verify:

```powershell
curl http://localhost:11434/api/tags
```

### Missing model

Cause:

- `OLLAMA_MODEL` is configured, but the model has not been downloaded.

Fix:

```powershell
ollama pull llama3.1
```

Use the model name from your `.env`.

### API key missing

Cause:

- `REMOTE_API_KEY` is empty or absent in `backend/.env`.

Fix:

```env
REMOTE_API_KEY=your_remote_provider_key
```

The health check reports only `Present` or `Missing`; it never prints the secret.

### Import errors

Cause:

- Missing packages.
- Syntax errors.
- Running from a directory where `app` cannot be imported.

Fix:

```powershell
cd backend
python -c "import app.main; print('ok')"
```

If that fails, read the exception and install the missing package or correct the environment.

### Pytest failures

Cause:

- A test assertion failed.
- A dependency or environment variable is missing.
- A provider call was not mocked or configured.

Fix:

```powershell
cd backend
python -m pytest app/tests -vv
```

Read the first failing test and fix the underlying issue intentionally.

## 13. Expected folder structure

Required backend folders:

```text
backend/
  app/
    services/
    tests/
    data/
      prompts/
      raw_runs/
      evaluations/
      benchmarks/
      processed/
      training/
  docs/
  tools/
```

Required prompt files:

```text
backend/app/data/prompts/coding.json
backend/app/data/prompts/mathematics.json
backend/app/data/prompts/reasoning.json
backend/app/data/prompts/planning.json
backend/app/data/prompts/translation.json
backend/app/data/prompts/summarization.json
backend/app/data/prompts/creative_writing.json
backend/app/data/prompts/general.json
```

Required core service files:

```text
backend/app/services/feature_extractor.py
backend/app/services/router.py
backend/app/services/dataset_generator.py
backend/app/services/evaluator.py
backend/app/services/benchmark.py
```

## 14. Expected console outputs

Health check with missing configuration:

```text
Environment
-----------
[OK] backend/.env exists.
[OK] FRONTEND_URL: Present
[FAIL] REMOTE_API_KEY: Missing
Status: FAIL
```

Ollama unavailable:

```text
Ollama
------
[FAIL] Ollama executable not found on PATH.
[FAIL] Ollama server is not reachable at http://localhost:11434: ...
[INFO] Start Ollama with `ollama serve` and verify OLLAMA_BASE_URL.
Status: FAIL
```

Router pipeline success:

```text
Router
------
[OK] Sample prompt completed through Feature Extractor -> Router.
[INFO] Task Type: coding
[INFO] Reasoning Score: 2
[INFO] Complexity: easy
[INFO] Routing Score: 19
[INFO] Confidence: 0.3208
[INFO] Chosen Provider: local
[INFO] Reason: Moderate reasoning score (2/10); Task type is coding; Routing score 19 is below threshold 25 -> local
Status: PASS
```
