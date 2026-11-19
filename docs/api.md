# API

Start the local service with `uv run uvicorn tuneforge.api:app --host 127.0.0.1`. OpenAPI is available at `/docs`. Authentication and multi-tenant authorization are deployment responsibilities.

Dataset paths are confined to the service's `datasets` directory. Register a file with `POST /datasets/register`. Planning accepts its fingerprint as `dataset_id`. Invalid LoRA targets and unsupported batch settings return HTTP 422. A source that differs from its registration causes `POST /runs` to return HTTP 409 before creating a run.

## Cancelling a synchronous run

1. Start a run with `POST /runs`. The request stays open through execution.
2. On another connection to the same worker, call `GET /runs/active` to obtain the active run ID.
3. Call `POST /runs/{id}/cancel`. An accepted request allows the in-flight update and checkpoint save to finish. The trainer observes cancellation between updates or before finalization.
4. Read the terminal state from the original response or `GET /runs/{id}`.

Cancellation is cooperative and worker-local. An unknown, finished, or already-finalizing run returns HTTP 409. Planning also returns HTTP 409 while the local model worker is in use. After an interruption, use the CLI with the service root to resume from a durable checkpoint.

The API also provides runtime inspection, metrics, checkpoint and evidence lookup, supplied-prediction evaluation, and registry approval and rejection.
