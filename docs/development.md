# Development

Install and validate:

```bash
uv sync --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=tuneforge --cov-report=term-missing
uv run python scripts/check_repository.py
uv build
uv run tuneforge --help
```

The test suite covers dataset formats and schemas, duplicate/leakage behavior, deterministic invariants, model and license policy, target discovery, capability gates, real forward/backward/optimizer steps, full and LoRA checkpoints, state transitions, non-finite failure, evaluation, registry evidence, API, CLI, security, and all twelve acceptance scenarios.

No test downloads model weights or requires a Hugging Face/OpenAI token, paid API, or GPU. Generated artifacts belong outside Git. Do not relax a fail-closed guard to make a fixture pass.

Use conventional, focused commits. Changes to lifecycle transitions, manifests, checkpoint formats, capability gates, or evidence schemas require tests and a note in [decisions.md](decisions.md) or [CHANGELOG.md](../CHANGELOG.md).

