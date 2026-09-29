"""Checkpoint creation, integrity validation, resumption, and bounded retention."""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterable
from pathlib import Path

from torch import nn

from tuneforge.domain import CheckpointManifest, ModelManifest, TrainingConfig, TrainingMethod
from tuneforge.security import confined_path, validate_artifact_name
from tuneforge.utils import file_sha256, stable_digest, utc_now, write_canonical_json


def run_directory(artifact_root: Path, run_id: str) -> Path:
    validate_artifact_name(run_id)
    path = confined_path(artifact_root, Path("runs") / run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _artifact_hashes(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): file_sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "checkpoint-manifest.json"
    }


def save_checkpoint(
    model: nn.Module,
    artifact_root: Path,
    run_id: str,
    step: int,
    epoch: float,
    method: TrainingMethod,
    model_manifest: ModelManifest,
    training_config: TrainingConfig,
    metrics: dict[str, float],
    *,
    manual: bool = False,
) -> CheckpointManifest:
    directory = confined_path(
        run_directory(artifact_root, run_id), Path("checkpoints") / f"step-{step:06d}"
    )
    if directory.exists():
        raise FileExistsError(f"checkpoint already exists: step {step}")
    directory.mkdir(parents=True)
    save_pretrained = getattr(model, "save_pretrained", None)
    if not callable(save_pretrained):
        raise TypeError("trainer model does not support safe save_pretrained serialization")
    save_pretrained(directory, safe_serialization=True)
    state = {
        "run_id": run_id,
        "step": step,
        "epoch": epoch,
        "method": method,
        "model_manifest_hash": model_manifest.manifest_hash,
        "training_config_hash": stable_digest(training_config),
    }
    write_canonical_json(directory / "training-state.json", state)
    hashes = _artifact_hashes(directory)
    checkpoint = CheckpointManifest(
        checkpoint_id=f"ckpt-{uuid.uuid4().hex}",
        run_id=run_id,
        step=step,
        epoch=epoch,
        created_at=utc_now(),
        method=method,
        base_model_reference=model_manifest.model_id,
        state_kind="adapter" if method in {TrainingMethod.LORA, TrainingMethod.QLORA} else "full",
        artifact_paths=sorted(hashes),
        file_hashes=hashes,
        metrics=metrics,
        configuration_digest=stable_digest(training_config),
        model_manifest_hash=model_manifest.manifest_hash,
        manual=manual,
    )
    write_canonical_json(directory / "checkpoint-manifest.json", checkpoint)
    return checkpoint


def checkpoint_directory(artifact_root: Path, manifest: CheckpointManifest) -> Path:
    return confined_path(
        run_directory(artifact_root, manifest.run_id),
        Path("checkpoints") / f"step-{manifest.step:06d}",
        must_exist=True,
    )


def validate_checkpoint(
    artifact_root: Path,
    manifest: CheckpointManifest,
    model_manifest_hash: str,
    training_config: TrainingConfig,
) -> Path:
    if manifest.model_manifest_hash != model_manifest_hash:
        raise ValueError("checkpoint model manifest does not match the requested run")
    if manifest.configuration_digest != stable_digest(training_config):
        raise ValueError("checkpoint training configuration does not match the requested run")
    directory = checkpoint_directory(artifact_root, manifest)
    for relative, expected in manifest.file_hashes.items():
        artifact = confined_path(directory, relative, must_exist=True)
        if file_sha256(artifact) != expected:
            raise ValueError(f"checkpoint integrity failure: {relative}")
    return directory


def apply_retention(
    artifact_root: Path,
    checkpoints: Iterable[CheckpointManifest],
    keep_last_n: int,
) -> list[CheckpointManifest]:
    ordered = sorted(checkpoints, key=lambda item: (item.step, item.checkpoint_id))
    protected = {item.checkpoint_id for item in ordered if item.manual}
    protected.update(item.checkpoint_id for item in ordered[-keep_last_n:])
    removed: list[CheckpointManifest] = []
    for checkpoint in ordered:
        if checkpoint.checkpoint_id in protected:
            continue
        directory = checkpoint_directory(artifact_root, checkpoint)
        shutil.rmtree(directory)
        removed.append(checkpoint)
    return removed


__all__ = [
    "apply_retention",
    "checkpoint_directory",
    "run_directory",
    "save_checkpoint",
    "validate_checkpoint",
]
