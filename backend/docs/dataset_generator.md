# Dataset Generator

The dataset generator builds raw comparison data for future routing-model training. It does not route prompts, train a model, extract classifier features, or modify any API endpoint.

## Architecture

`backend/app/services/dataset_generator.py` is a standalone service module. It uses the existing provider clients:

- `app.services.ollama.generate` for the local model
- `app.services.remote_llm.generate` for the remote model

For every prompt, both providers are called. Each provider result is measured independently and saved in a JSON-compatible structure.

## Workflow

1. Load prompts from a JSON file with `load_prompts(path)`.
2. Run one prompt with `run_prompt(prompt)`.
3. Run a full prompt sequence with `generate_dataset(prompts)`.
4. Save the result with `save_dataset(dataset, output_path)`.

`generate_dataset` processes prompts sequentially. For each individual prompt, the local and remote providers run concurrently so both are always attempted for that prompt.

If one provider fails, the other provider is still allowed to finish. The failed provider stores an `error` field and dataset generation continues.

## JSON Schema

The output is a list of entries:

```json
{
  "prompt": "Explain RAG",
  "metadata": {
    "prompt_length": 11,
    "word_count": 2,
    "category": null,
    "difficulty": null
  },
  "local": {
    "model": "qwen2.5:3b",
    "response": "...",
    "latency_ms": 830,
    "estimated_input_tokens": 3,
    "estimated_output_tokens": 96
  },
  "remote": {
    "model": "llama-3.1-8b-instant",
    "response": "...",
    "latency_ms": 1540,
    "estimated_input_tokens": 3,
    "estimated_output_tokens": 101
  },
  "created_at": "2026-07-03T10:15:00+00:00"
}
```

On provider failure, that provider block includes `error`:

```json
{
  "model": "qwen2.5:3b",
  "latency_ms": 12,
  "estimated_input_tokens": 3,
  "estimated_output_tokens": 0,
  "error": "Could not connect to Ollama."
}
```

Token counts use a replaceable approximation:

```text
ceil(len(text) / 4)
```

No tokenizer is called.

## How To Run

From the `backend` directory, run:

```powershell
python -m app.services.dataset_generator app/data/prompts/example_prompts.json app/data/raw_runs/generated_dataset.json
```

Required environment settings are the same provider settings used by the existing app:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `REMOTE_API_KEY`
- `REMOTE_BASE_URL`
- `REMOTE_MODEL`

## Future Improvements

- Replace the simple token estimate with a provider-aware tokenizer.
- Add optional prompt category and difficulty labels.
- Add retry and backoff controls for transient provider failures.
- Add batch progress logging for long prompt lists.
- Add schema validation before saving generated datasets.
