"""Evidence-backed reports, model cards, audit events, and reproducible ZIP bundles."""

from __future__ import annotations

import uuid
import zipfile
from pathlib import Path
from typing import Any

from tuneforge.checkpoints import run_directory
from tuneforge.domain import (
    AuditEvent,
    DatasetManifest,
    EvaluationResult,
    ModelManifest,
    RegistryStatus,
    TrainingEvidence,
    TrainingPlan,
    TrainingRun,
)
from tuneforge.security import confined_path, redact
from tuneforge.storage import MetadataStore
from tuneforge.utils import (
    canonical_json,
    file_sha256,
    stable_digest,
    utc_now,
    write_canonical_json,
)


def audit_event(event_type: str, subject_id: str, details: dict[str, Any]) -> AuditEvent:
    return AuditEvent(
        id=f"audit-{uuid.uuid4().hex}",
        event_type=event_type,
        subject_id=subject_id,
        created_at=utc_now(),
        details=redact(details),
    )


def build_training_evidence(
    run: TrainingRun,
    dataset: DatasetManifest,
    model: ModelManifest,
    plan: TrainingPlan,
    store: MetadataStore,
    evaluation: EvaluationResult | None = None,
    registry_status: RegistryStatus | None = None,
) -> TrainingEvidence:
    checkpoints = store.list_checkpoints(run.id)
    checkpoint_hashes = {
        checkpoint.checkpoint_id: stable_digest(checkpoint) for checkpoint in checkpoints
    }
    payload = {
        "run_id": run.id,
        "dataset_manifest_hash": stable_digest(dataset),
        "model_manifest_hash": model.manifest_hash,
        "training_plan_hash": plan.plan_hash,
        "runtime": plan.runtime,
        "metrics": store.list_metrics(run.id),
        "checkpoint_hashes": checkpoint_hashes,
        "evaluation": evaluation,
        "registry_status": registry_status,
        "final_state": run.state,
    }
    return TrainingEvidence(**payload, evidence_hash=stable_digest(payload))


def generate_training_report(
    run: TrainingRun,
    model: ModelManifest,
    dataset: DatasetManifest,
    plan: TrainingPlan,
    store: MetadataStore,
    evaluation: EvaluationResult | None = None,
) -> dict[str, Any]:
    metrics = store.list_metrics(run.id)
    checkpoints = store.list_checkpoints(run.id)
    duration = max(0.0, (run.updated_at - run.created_at).total_seconds())
    return {
        "run_id": run.id,
        "model": model.model_dump(mode="json"),
        "dataset": {
            "fingerprint": dataset.fingerprint,
            "records": dataset.quality.valid_records,
            "license": dataset.source.license.model_dump(mode="json"),
        },
        "method": plan.method.value,
        "hardware": plan.runtime.model_dump(mode="json"),
        "duration_seconds": duration,
        "steps": run.last_completed_step,
        "metrics": [item.model_dump(mode="json") for item in metrics],
        "checkpoints": [item.model_dump(mode="json") for item in checkpoints],
        "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
        "warnings": plan.warnings,
        "artifact_hashes": {item.checkpoint_id: stable_digest(item) for item in checkpoints},
    }


def generate_model_card(
    run: TrainingRun,
    model: ModelManifest,
    dataset: DatasetManifest,
    plan: TrainingPlan,
    evaluation: EvaluationResult | None,
) -> str:
    evaluation_text = (
        canonical_json(evaluation.metrics)
        if evaluation is not None
        else "No evaluation result was supplied; no performance claim is made."
    )
    return f"""---
license: {model.license.identifier}
base_model: {model.model_id}
library_name: tuneforge
---

# TuneForge training artifact: {run.id}

## Provenance

- Base model: `{model.model_id}` at revision `{model.revision}`
- Training method: `{plan.method.value}`
- Dataset fingerprint: `{dataset.fingerprint}`
- Dataset license metadata: `{dataset.source.license.identifier}`
- Dataset license metadata verified: {str(dataset.source.license.verified).lower()}
- Model manifest: `{model.manifest_hash}`
- Training plan: `{plan.plan_hash}`

## Measured evaluation

{evaluation_text}

## Intended use and limitations

This draft documents one governed training run.
The local tiny-model path validates mechanics only and does not establish general model quality.
Users must verify model and dataset licenses, privacy constraints, and intended-use fitness.

## Runtime and reproducibility

Runtime: `{plan.runtime.os}`.
Python: `{plan.runtime.python_version}`. PyTorch: `{plan.runtime.torch_version}`.
Seed: `{run.seed}`. Determinism is bounded by the recorded software and hardware environment.
"""


def write_evidence_bundle(
    artifact_root: Path,
    run: TrainingRun,
    evidence: TrainingEvidence,
    training_report: dict[str, Any],
    model_card: str,
) -> tuple[Path, str]:
    directory = run_directory(artifact_root, run.id)
    evidence_directory = confined_path(directory, "evidence")
    evidence_directory.mkdir(parents=True, exist_ok=True)
    files = {
        "evidence.json": canonical_json(evidence) + "\n",
        "model-card.md": model_card.replace("\r\n", "\n"),
        "training-report.json": canonical_json(training_report) + "\n",
    }
    for name, content in files.items():
        target = confined_path(evidence_directory, name)
        target.write_text(content, encoding="utf-8", newline="\n")
    archive = confined_path(evidence_directory, "evidence.zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name in sorted(files):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, files[name].encode("utf-8"))
    write_canonical_json(
        evidence_directory / "bundle-manifest.json",
        {
            "archive": archive.name,
            "sha256": file_sha256(archive),
            "members": sorted(files),
        },
    )
    return archive, file_sha256(archive)


__all__ = [
    "audit_event",
    "build_training_evidence",
    "generate_model_card",
    "generate_training_report",
    "write_evidence_bundle",
]
