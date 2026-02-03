# Smoke Test Harness Outline (Minimal Integration)

This document defines a minimal smoke-test runner structure and the integration flow. It is intentionally generic and should be wired into the inference server's loader registry.

## Minimal Runner (Python)

```python
# smoke_test_runner.py
from pathlib import Path
import time


def run_smoke_test(manifest: dict, model_path: Path) -> None:
    start = time.time()
    test = manifest["smoke_test"]
    entry = manifest["entrypoint"]

    if entry["loader"] == "diffusers":
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_pretrained(model_path)
        _ = pipe(test.get("prompt", "a single red apple on a white table"))

    elif entry["loader"] == "transformers":
        # Example for text generation; real use depends on task type.
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        _ = model.generate(**tok(test.get("prompt", "hello"), return_tensors="pt"))

    elif entry["loader"] == "custom":
        # Call a registered, versioned loader entrypoint.
        raise NotImplementedError("Custom loader must be registered in code.")

    elapsed = time.time() - start
    if elapsed > test["max_seconds"]:
        raise TimeoutError(f"Smoke test exceeded {test['max_seconds']}s")
```

## Integration Notes

- Run after download + manifest validation.
- Mark model as "available" only if smoke test completes within `max_seconds`.
- Store output artifacts (e.g., images) for debugging and audit.
- If using Supabase, log smoke-test results and latency to a `model_smoke_tests` table.

## Suggested Supabase Table (Optional)

```sql
create table if not exists model_smoke_tests (
  id uuid primary key default gen_random_uuid(),
  model_id text not null,
  version text not null,
  profile text not null,
  status text not null,
  duration_ms integer not null,
  created_at timestamp with time zone default now()
);
```
