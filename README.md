# TuneForge

### Governed LLM & VLM Fine-Tuning Platform

TuneForge validates and fingerprints training datasets, builds reproducible training plans, executes governed fine-tuning workflows, tracks experiments and checkpoints, evaluates model candidates, and produces evidence-backed training artifacts.

TuneForge is an original implementation built against documented public APIs. It is not a fork and does not contain copied upstream code, prompts, tests, documentation, model cards, history, branding, or benchmark claims.

## What works in v0.1

- Strict JSON, JSONL, and CSV intake for instruction, chat, prompt/completion, and preference records.
- Canonical normalization, deterministic fingerprints and splits, duplicate reports, leakage checks, data-quality reports, and tokenization analysis.
- Typed model and dataset license metadata. Unknown licenses remain explicitly `unknown`.
- Model manifests, training plans, runtime inspection, resource estimates, trainability gates, and reproducibility controls.
- Real local CPU training from a randomly initialized tiny GPT-2 configuration. No model weights are downloaded.
- Real PEFT LoRA adapter injection, measured trainable-parameter reporting, forward/backward/optimizer steps, safetensors save, and adapter reload.
- SQLite experiments, run state, measured metrics, checkpoints, evaluations, registry entries, and redacted audit events.
- Checkpoint integrity, compatibility validation, retention, safe paths, deterministic evidence ZIPs, model-card drafts, and training reports.
- FastAPI endpoints, a Typer CLI, twelve executable acceptance scenarios, and offline routine CI.

QLoRA configuration is validated and gated on real CUDA and bitsandbytes capability. It is not executed or claimed by routine CPU CI. VLM-aware modality boundaries exist, but full VLM training is intentionally unsupported in v0.1. Adapter merging, distributed orchestration, remote pretrained downloads, and Hub publication are also intentionally unsupported.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev
uv run tuneforge --help
uv run tuneforge demo --root .tuneforge-demo
uv run tuneforge eval --root .tuneforge-eval
```

The demo generates its own ten-record fixture, constructs a tiny transformer locally, performs three real training steps, saves and reloads a LoRA adapter, measures held-out loss, and writes an evidence bundle. Generated data and artifacts are ignored by Git.

To run the service:

```bash
uv run uvicorn tuneforge.api:app --host 127.0.0.1 --port 8000
```

API datasets are confined to `.tuneforge/datasets`. Copy a local input there before registration. The CLI accepts explicit local paths.

## Workflow configuration

```json
{
  "dataset_path": "datasets/instructions.jsonl",
  "artifact_root": ".tuneforge",
  "method": "lora",
  "dataset_license": {"identifier": "unknown", "verified": false},
  "training": {
    "max_steps": 3,
    "max_sequence_length": 24,
    "checkpoint_frequency": 2,
    "seed": 42
  },
  "lora": {"r": 2, "alpha": 4, "target_modules": ["c_attn"]}
}
```

```bash
uv run tuneforge dataset validate datasets/instructions.jsonl
uv run tuneforge plan workflow.json
uv run tuneforge train workflow.json
```

## Architecture and governance

The lifecycle is dataset intake → normalization → fingerprint/split → manifests → plan → capability gate → trainer → checkpoint/evaluation → registry/evidence. A run cannot skip lifecycle states, an approval cannot exist without evaluation evidence, and a checkpoint cannot escape its run directory.

Read the detailed documentation:

- [Architecture](docs/architecture.md)
- [Datasets](docs/datasets.md)
- [Training](docs/training.md)
- [PEFT and LoRA](docs/peft.md)
- [Evaluation](docs/evaluation.md)
- [Checkpoints](docs/checkpoints.md)
- [Security](docs/security.md)
- [Reproducibility](docs/reproducibility.md)
- [Development](docs/development.md)
- [Architecture decisions](docs/decisions.md)
- [API](docs/api.md) and [CLI](docs/cli.md)
- [Research references and independence boundary](REFERENCES.md)

## Evidence boundaries

Training loss, evaluation loss, duration, parameter counts, and checkpoint hashes are recorded only when measured. The tiny fixture is a mechanical integration test, not a model-quality benchmark. Resource values are planning estimates, not guaranteed memory measurements. TuneForge never assumes that public availability implies commercial licensing permission and does not provide legal advice.

## License

TuneForge source is available under the [MIT License](LICENSE). Model, dataset, and generated-artifact licenses remain separate and must be verified by operators.
