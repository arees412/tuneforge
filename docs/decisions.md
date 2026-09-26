# Architecture decisions

## ADR-001: Original implementation

TuneForge is a new public repository, not a fork. Public libraries and their official documentation informed API compatibility. No upstream implementation, prompt, test, README, model card, branding, benchmark, or commit history was copied.

## ADR-002: Offline deterministic CI

Routine CI constructs a tiny causal transformer from local configuration and generated fixtures. This makes real full and LoRA training mechanically verifiable without credentials, network model downloads, paid services, or GPU infrastructure.

## ADR-003: Fail-closed QLoRA

QLoRA configuration is modeled, but execution requires genuine CUDA and bitsandbytes capability. CPU CI reports it as blocked and never emits fake quantized metrics.

## ADR-004: Direct PyTorch training backend

The backend uses Transformers model configuration, PEFT adapters, and a direct PyTorch optimizer loop. This keeps lifecycle, finite-value checks, metrics, and checkpoint policy explicit. A future Transformers/TRL backend must preserve the same typed contract and evidence rules.

## ADR-005: SQLite plus structured artifacts

SQLite stores queryable metadata while canonical JSON and safetensors hold portable run evidence. A schema-migration table establishes the v0.1 database version.

## ADR-006: Safe serialization boundary

Weights and adapters use safetensors. Optimizer pickle loading, model merging, remote code, automated Hub publication, distributed scheduling, and full VLM training are intentionally unsupported in v0.1.

