# Changelog

## Unreleased

- Enforced the CPU training configuration at planning, execution and restart. New workflows default to explicit evaluation. Historical checkpoints retain their recorded configuration for held-out scoring. See the [training contract](docs/training.md#executable-configuration).

## 0.1.0 - 2026-09-26

- Added strict dataset intake, canonical normalization, deterministic fingerprints/splits, duplicate and leakage controls, quality reporting, and tokenization analysis.
- Added model/license manifests, LoRA and QLoRA configuration, runtime capability gates, resource estimates, and training plans.
- Added real local tiny full and PEFT LoRA training with measured metrics and safetensors checkpoint reload.
- Added SQLite tracking, checkpoint governance, deterministic evaluation/comparison, model registry, evidence bundles, training reports, model-card drafts, provenance, and redacted audit events.
- Added FastAPI, Typer, twelve executable scenarios, extensive tests, offline CI, and project documentation.
