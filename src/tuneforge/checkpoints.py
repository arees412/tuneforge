"""Safe checkpoint publication, verification and journaled retention."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfile

from torch import nn

from tuneforge.domain import (
    CheckpointManifest,
    ModelManifest,
    RecordedTrainingConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.security import confined_path, validate_artifact_name
from tuneforge.storage import MetadataStore
from tuneforge.utils import (
    canonical_json,
    file_sha256,
    stable_digest,
    sync_directory,
    utc_now,
    write_canonical_json,
)

_MANIFEST = "checkpoint-manifest.json"


def run_directory(artifact_root: Path, run_id: str) -> Path:
    validate_artifact_name(run_id)
    path = confined_path(artifact_root, Path("runs") / run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _paths(root: Path, manifest: CheckpointManifest) -> tuple[Path, Path]:
    validate_artifact_name(manifest.run_id)
    validate_artifact_name(manifest.checkpoint_id)
    if manifest.step < 1:
        raise ValueError("invalid checkpoint step")
    parent = Path("runs") / manifest.run_id / "checkpoints"
    return (
        confined_path(root, parent / f"step-{manifest.step:06d}"),
        confined_path(root, parent / f".pending-{manifest.checkpoint_id}"),
    )


def checkpoint_directory(artifact_root: Path, manifest: CheckpointManifest) -> Path:
    directory, _ = _paths(artifact_root, manifest)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def _check_files(directory: Path, manifest: CheckpointManifest, *, partial: bool = False) -> None:
    required = {"training-state.json"}
    if manifest.method is TrainingMethod.LORA:
        required.update({"adapter_model.safetensors", "adapter_config.json"})
        optional = {"README.md"}
    elif manifest.method is TrainingMethod.FULL_FINE_TUNE:
        required.update({"model.safetensors", "config.json"})
        optional = {"generation_config.json"}
    else:
        raise ValueError("unsupported checkpoint method")
    hashed = set(manifest.file_hashes)
    if not required <= hashed <= required | optional or sorted(hashed) != sorted(
        manifest.artifact_paths
    ):
        raise ValueError("invalid checkpoint artifact set")
    actual = {p.name for p in directory.iterdir()}
    expected = hashed | {_MANIFEST}
    if actual - expected or (not partial and actual != expected):
        raise ValueError("checkpoint artifact set mismatch")
    # Retired checkpoints keep their identity file until all other members are gone.
    if actual:
        marker = confined_path(directory, _MANIFEST, must_exist=True)
        if CheckpointManifest.model_validate_json(marker.read_bytes()) != manifest:
            raise ValueError("checkpoint ownership or manifest mismatch")
    for name in actual - {_MANIFEST}:
        member = confined_path(directory, name, must_exist=True)
        if not member.is_file() or file_sha256(member) != manifest.file_hashes[name]:
            raise ValueError(f"checkpoint integrity failure: {name}")


def validate_checkpoint(
    artifact_root: Path,
    manifest: CheckpointManifest,
    model_manifest_hash: str,
    training_config: RecordedTrainingConfig,
) -> Path:
    if manifest.model_manifest_hash != model_manifest_hash:
        raise ValueError("checkpoint model manifest does not match the requested run")
    if manifest.configuration_digest != stable_digest(training_config):
        raise ValueError("checkpoint training configuration does not match the requested run")
    directory = checkpoint_directory(artifact_root, manifest)
    _check_files(directory, manifest)
    state = json.loads((directory / "training-state.json").read_bytes())
    if (
        state.get("dataset_fingerprint"),
        state["run_id"],
        state["step"],
        state["method"],
        state["training_config_hash"],
        state["model_manifest_hash"],
    ) != (
        manifest.dataset_fingerprint,
        manifest.run_id,
        manifest.step,
        manifest.method,
        manifest.configuration_digest,
        model_manifest_hash,
    ):
        raise ValueError("checkpoint training state does not match its owner")
    return directory


@contextmanager
def verified_checkpoint_files(
    root: Path,
    checkpoint: CheckpointManifest,
    model_hash: str,
    config: RecordedTrainingConfig,
    store: MetadataStore,
    owner: str,
) -> Iterator[Path]:
    run = store.get_run(owner)
    if (
        run is None
        or run.model_manifest_hash != model_hash
        or run.training_config_hash != stable_digest(config)
        or checkpoint.run_id != owner
        or checkpoint not in store.list_checkpoints(owner)
    ):
        raise ValueError("checkpoint is not registered to this run and configuration")
    if checkpoint.dataset_fingerprint != run.dataset_fingerprint:
        raise ValueError("checkpoint lacks matching dataset provenance for governed loading")
    directory = validate_checkpoint(root, checkpoint, model_hash, config)
    # Load only the exact bytes verified in a private directory.
    with tempfile.TemporaryDirectory() as work:
        snapshot = Path(work)
        for name in [*checkpoint.artifact_paths, _MANIFEST]:
            copyfile(confined_path(directory, name, must_exist=True), snapshot / name)
        _check_files(snapshot, checkpoint)
        yield snapshot


def recover_checkpoints(root: Path, store: MetadataStore, run_id: str) -> None:
    """Replay durable intents under the SQLite writer lock. Safe after a stopped worker."""
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT * FROM checkpoint_ops WHERE run_id=? ORDER BY rowid", (run_id,)
        ).fetchall()
        for row in rows:
            cp = CheckpointManifest.model_validate_json(row["payload"])
            if cp.run_id != run_id or cp.checkpoint_id != row["id"]:
                raise ValueError("checkpoint journal ownership mismatch")
            live, staging = _paths(root, cp)
            if row["operation"] == "publish":
                if staging.exists():
                    _check_files(staging, cp)
                    if live.exists():
                        raise ValueError("checkpoint destination already exists")
                    staging.rename(live)
                _check_files(live, cp)
                conn.execute(
                    "INSERT OR IGNORE INTO checkpoints VALUES (?,?,?,?)",
                    (cp.checkpoint_id, run_id, cp.step, canonical_json(cp)),
                )
            elif row["operation"] == "retire":
                if cp.manual:
                    raise ValueError("manual checkpoint is protected")
                if live.exists():
                    _check_files(live, cp)
                    if staging.exists():
                        raise ValueError("checkpoint staging collision")
                    live.rename(staging)
                if staging.exists():
                    _check_files(staging, cp, partial=True)
                    for name in cp.artifact_paths:
                        confined_path(staging, name).unlink(missing_ok=True)
                    (staging / _MANIFEST).unlink(missing_ok=True)
                    staging.rmdir()
            else:
                raise ValueError("unknown checkpoint journal operation")
            sync_directory(live.parent)
            conn.execute("DELETE FROM checkpoint_ops WHERE id=?", (cp.checkpoint_id,))


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
    store: MetadataStore,
    manual: bool = False,
) -> CheckpointManifest:
    owner = store.get_run(run_id)
    if (
        owner is None
        or owner.model_manifest_hash != model_manifest.manifest_hash
        or owner.training_config_hash != stable_digest(training_config)
        or method != model_manifest.training_method
        or step < 1
    ):
        raise ValueError("checkpoint ownership or configuration mismatch")
    parent = confined_path(run_directory(artifact_root, run_id), "checkpoints")
    parent.mkdir(exist_ok=True)
    if confined_path(parent, f"step-{step:06d}").exists():
        raise FileExistsError(f"checkpoint already exists: step {step}")
    identifier = f"ckpt-{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(dir=parent, prefix=".writing-") as work:
        directory = Path(work) / "files"
        directory.mkdir()
        save = getattr(model, "save_pretrained", None)
        if not callable(save):
            raise TypeError("model does not support safe checkpoint serialization")
        save(directory, safe_serialization=True)
        state = {
            "run_id": run_id,
            "step": step,
            "epoch": epoch,
            "dataset_fingerprint": owner.dataset_fingerprint,
            "method": method,
            "model_manifest_hash": model_manifest.manifest_hash,
            "training_config_hash": stable_digest(training_config),
        }
        write_canonical_json(directory / "training-state.json", state)
        hashes = {}
        for member in directory.iterdir():
            path = confined_path(directory, member.name, must_exist=True)
            with path.open("rb+") as stream:
                os.fsync(stream.fileno())
            hashes[member.name] = file_sha256(path)
        cp = CheckpointManifest(
            checkpoint_id=identifier,
            dataset_fingerprint=owner.dataset_fingerprint,
            run_id=run_id,
            step=step,
            epoch=epoch,
            created_at=utc_now(),
            method=method,
            base_model_reference=model_manifest.model_id,
            state_kind="adapter" if method is TrainingMethod.LORA else "full",
            artifact_paths=sorted(hashes),
            file_hashes=hashes,
            metrics=metrics,
            configuration_digest=stable_digest(training_config),
            model_manifest_hash=model_manifest.manifest_hash,
            manual=manual,
        )
        write_canonical_json(directory / _MANIFEST, cp)
        _check_files(directory, cp)
        pending = confined_path(parent, f".pending-{identifier}")
        moved = False
        try:
            with store.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                active = conn.execute(
                    "SELECT 1 FROM checkpoints WHERE run_id=? AND step=?", (run_id, step)
                ).fetchone()
                queued = conn.execute(
                    "SELECT payload FROM checkpoint_ops WHERE run_id=?", (run_id,)
                ).fetchall()
                if active or any(
                    CheckpointManifest.model_validate_json(row["payload"]).step == step
                    for row in queued
                ):
                    raise FileExistsError(
                        f"checkpoint step is already registered or pending: {step}"
                    )
                directory.rename(pending)
                moved = True
                sync_directory(parent)
                conn.execute(
                    "INSERT INTO checkpoint_ops VALUES (?,?,?,?)",
                    (identifier, run_id, "publish", canonical_json(cp)),
                )
        except Exception:
            if moved:
                pending.rename(directory)
            raise
    recover_checkpoints(artifact_root, store, run_id)
    return cp


def apply_retention(
    artifact_root: Path, store: MetadataStore, run_id: str, keep_last_n: int
) -> None:
    if keep_last_n < 1:
        raise ValueError("retention requires at least one checkpoint")
    recover_checkpoints(artifact_root, store, run_id)
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT payload FROM checkpoints WHERE run_id=? ORDER BY step DESC,id", (run_id,)
        ).fetchall()
        for row in rows[keep_last_n:]:
            cp = CheckpointManifest.model_validate_json(row["payload"])
            if cp.manual:
                continue
            if cp.run_id != run_id:
                raise ValueError("checkpoint run mismatch")
            _check_files(checkpoint_directory(artifact_root, cp), cp)
            conn.execute(
                "INSERT INTO checkpoint_ops VALUES (?,?,?,?)",
                (cp.checkpoint_id, run_id, "retire", canonical_json(cp)),
            )
            conn.execute(
                "DELETE FROM checkpoints WHERE id=? AND run_id=?", (cp.checkpoint_id, run_id)
            )
    recover_checkpoints(artifact_root, store, run_id)
