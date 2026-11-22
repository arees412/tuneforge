"""Checkpoint ownership, protected retention and crash replay on private fixture files."""

from __future__ import annotations

from pathlib import Path

import pytest

from tuneforge import checkpoints
from tuneforge.domain import TrainingConfig, TrainingMethod, TrainingRun
from tuneforge.modeling import build_model_manifest, create_tiny_model, tiny_model_spec
from tuneforge.security import PathSecurityError
from tuneforge.storage import MetadataStore
from tuneforge.utils import stable_digest, utc_now


@pytest.fixture
def checkpoint_factory(tmp_path):
    root = tmp_path / "artifacts"
    store = MetadataStore(tmp_path / "meta.sqlite3")
    model = create_tiny_model(seed=42)
    manifest = build_model_manifest(tiny_model_spec(model), TrainingMethod.FULL_FINE_TUNE)
    config = TrainingConfig()

    def make(owner, step, manual=False):
        if store.get_run(owner) is None:
            store.save_run(
                TrainingRun(
                    id=owner,
                    dataset_fingerprint="fixture",
                    model_manifest_hash=manifest.manifest_hash,
                    training_config_hash=stable_digest(config),
                    package_version="test",
                    seed=42,
                    environment={},
                    created_at=utc_now(),
                    updated_at=utc_now(),
                )
            )
        return checkpoints.save_checkpoint(
            model,
            root,
            owner,
            step,
            1.0,
            TrainingMethod.FULL_FINE_TUNE,
            manifest,
            config,
            {"training_loss": 1.0},
            store=store,
            manual=manual,
        )

    return root, store, make, manifest, config


def test_retention_leaves_manual_and_unrelated_checkpoints(checkpoint_factory):
    root, store, make, model, config = checkpoint_factory
    first = make("run-a", 1, manual=True)
    old = make("run-a", 2)
    newest = make("run-a", 3)
    other = make("run-b", 2)
    checkpoints.apply_retention(root, store, "run-a", 1)
    assert store.list_checkpoints("run-a") == [first, newest]
    assert store.list_checkpoints("run-b") == [other]
    for cp in (first, newest, other):
        assert checkpoints.validate_checkpoint(root, cp, model.manifest_hash, config).is_dir()
    with pytest.raises(FileNotFoundError):
        checkpoints.checkpoint_directory(root, old)
    with (
        pytest.raises(ValueError, match="registered"),
        checkpoints.verified_checkpoint_files(
            root, newest, model.manifest_hash, config, store, "run-b"
        ),
    ):
        pytest.fail("wrong owner accepted")


def test_unexpected_member_blocks_deletion(checkpoint_factory):
    root, store, make, _, _ = checkpoint_factory
    first = make("run-a", 1)
    second = make("run-a", 2)
    untouched = checkpoints.checkpoint_directory(root, first) / "operator-notes.txt"
    untouched.write_text("keep")
    with pytest.raises(ValueError, match="artifact set"):
        checkpoints.apply_retention(root, store, "run-a", 1)
    assert untouched.read_text() == "keep"
    assert store.list_checkpoints("run-a") == [first, second]


def test_reject_linked_directory_before_checkpoint_io(checkpoint_factory):
    root, store, make, model, config = checkpoint_factory
    cp = make("run-a", 1)
    other = make("run-a", 2)
    path = checkpoints.checkpoint_directory(root, cp)
    original = path.with_name("original")
    path.rename(original)
    path.symlink_to(original, target_is_directory=True)
    with pytest.raises(PathSecurityError):
        checkpoints.validate_checkpoint(root, cp, model.manifest_hash, config)
    with pytest.raises(PathSecurityError):
        checkpoints.apply_retention(root, store, "run-a", 1)
    assert original.is_dir() and store.list_checkpoints("run-a") == [cp, other]
    for invalid_id in ("../invalid", "/absolute", "C:drive"):
        with pytest.raises(ValueError):
            checkpoints.checkpoint_directory(root, cp.model_copy(update={"run_id": invalid_id}))


def test_retirement_interruption_replays_partial_deletion(checkpoint_factory, monkeypatch):
    root, store, make, _, _ = checkpoint_factory
    first = make("run-a", 1)
    newest = make("run-a", 2)
    original = Path.unlink
    removed = []

    def interrupt(path, *args, **kwargs):
        if path.parent.name.startswith(".pending-") and removed:
            raise OSError("simulated cleanup interruption")
        original(path, *args, **kwargs)
        removed.append(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(OSError, match="cleanup interruption"):
            checkpoints.apply_retention(root, store, "run-a", 1)
    assert store.list_checkpoints("run-a") == [newest]
    checkpoints.recover_checkpoints(root, store, "run-a")
    checkpoints.recover_checkpoints(root, store, "run-a")
    assert list(checkpoints.checkpoint_directory(root, newest).parent.iterdir()) == [
        checkpoints.checkpoint_directory(root, newest)
    ]
    with pytest.raises(FileNotFoundError):
        checkpoints.checkpoint_directory(root, first)


def test_publish_interruption_replays_after_rename(checkpoint_factory, monkeypatch):
    root, store, make, model, config = checkpoint_factory
    original = Path.rename

    def interrupt(path, target):
        result = original(path, target)
        if path.name.startswith(".pending-"):
            raise OSError("simulated interruption after rename")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", interrupt)
        with pytest.raises(OSError, match="after rename"):
            make("run-a", 1)
    assert store.list_checkpoints("run-a") == []
    checkpoints.recover_checkpoints(root, store, "run-a")
    [cp] = store.list_checkpoints("run-a")
    assert checkpoints.validate_checkpoint(root, cp, model.manifest_hash, config).is_dir()


def test_legacy_checkpoint_requires_fresh_provenance(checkpoint_factory):
    root, store, make, model, config = checkpoint_factory
    current = make("run-a", 1)
    legacy = current.model_copy(update={"dataset_fingerprint": None})
    store.record_checkpoint(legacy)
    with (
        pytest.raises(ValueError, match="dataset provenance"),
        checkpoints.verified_checkpoint_files(
            root, legacy, model.manifest_hash, config, store, "run-a"
        ),
    ):
        pytest.fail("legacy weights accepted for governed execution")
