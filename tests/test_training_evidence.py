from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tuneforge.checkpoints import apply_retention, checkpoint_directory, validate_checkpoint
from tuneforge.dataset import build_dataset_manifest
from tuneforge.domain import (
    EvaluationResult,
    LoRAConfig,
    RegistryStatus,
    RunState,
    SplitConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.evidence import (
    build_training_evidence,
    generate_model_card,
    generate_training_report,
    write_evidence_bundle,
)
from tuneforge.modeling import (
    apply_lora,
    build_model_manifest,
    create_tiny_model,
    parameter_report,
    tiny_model_spec,
)
from tuneforge.planning import build_training_plan
from tuneforge.runtime import detect_runtime
from tuneforge.storage import MetadataStore
from tuneforge.training import CancellationToken, DeterministicTrainerBackend
from tuneforge.utils import file_sha256


def execute(
    root: Path,
    dataset_path: Path,
    method: TrainingMethod,
    *,
    max_steps: int = 2,
    loss_hook=None,  # type: ignore[no-untyped-def]
):  # type: ignore[no-untyped-def]
    dataset, records = build_dataset_manifest(dataset_path, SplitConfig(seed=5))
    config = TrainingConfig(
        max_steps=max_steps,
        max_sequence_length=16,
        checkpoint_frequency=1,
        keep_last_n=1,
        seed=5,
    )
    base = create_tiny_model(seed=config.seed)
    adapter = LoRAConfig(r=2, alpha=4) if method is TrainingMethod.LORA else None
    model = build_model_manifest(tiny_model_spec(base), method, adapter)
    planned = apply_lora(base, adapter) if adapter else base
    plan = build_training_plan(
        dataset,
        model,
        config,
        detect_runtime(),
        trainable_parameters=int(parameter_report(planned)["trainable_parameters"]),
        adapter=adapter,
    )
    store = MetadataStore(root / "tuneforge.sqlite3")
    store.register_dataset(dataset.fingerprint, dataset)
    outcome = DeterministicTrainerBackend().execute(
        plan, model, config, records, root, store, loss_hook=loss_hook
    )
    return dataset, config, model, plan, store, outcome


@pytest.mark.training
@pytest.mark.parametrize("method", [TrainingMethod.FULL_FINE_TUNE, TrainingMethod.LORA])
def test_real_training_checkpoint_and_reload(
    tmp_path: Path, dataset_path: Path, method: TrainingMethod
) -> None:
    _, config, model, _, store, outcome = execute(tmp_path / method.value, dataset_path, method)
    assert outcome.run.state is RunState.COMPLETED
    assert len(outcome.metrics) == 2
    assert all(metric.training_loss > 0 for metric in outcome.metrics)
    assert outcome.adapter_reload_verified
    checkpoint = outcome.checkpoints[-1]
    directory = validate_checkpoint(
        tmp_path / method.value, checkpoint, model.manifest_hash, config
    )
    assert directory.is_dir()
    assert any(path.suffix == ".safetensors" for path in directory.iterdir())
    assert len(store.list_metrics(outcome.run.id)) == 2
    if method is TrainingMethod.LORA:
        assert (
            outcome.parameter_report["trainable_parameters"]
            < outcome.parameter_report["total_parameters"]
        )


@pytest.mark.training
def test_non_finite_loss_fails_without_checkpoint(tmp_path: Path, dataset_path: Path) -> None:
    with pytest.raises(FloatingPointError, match="non-finite"):
        execute(
            tmp_path,
            dataset_path,
            TrainingMethod.LORA,
            max_steps=1,
            loss_hook=lambda _step, loss: loss * torch.tensor(float("nan")),
        )
    store = MetadataStore(tmp_path / "tuneforge.sqlite3")
    with store.connect() as connection:
        states = [row["state"] for row in connection.execute("SELECT state FROM runs")]
        checkpoint_count = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    assert states == ["failed"]
    assert checkpoint_count == 0


def test_cancellation_before_first_step(tmp_path: Path, dataset_path: Path) -> None:
    dataset, records = build_dataset_manifest(dataset_path, SplitConfig(seed=5))
    config = TrainingConfig(max_steps=1, max_sequence_length=16)
    model = create_tiny_model(seed=5)
    manifest = build_model_manifest(tiny_model_spec(model), TrainingMethod.FULL_FINE_TUNE)
    plan = build_training_plan(
        dataset,
        manifest,
        config,
        detect_runtime(),
        trainable_parameters=int(parameter_report(model)["trainable_parameters"]),
    )
    store = MetadataStore(tmp_path / "meta.sqlite3")
    token = CancellationToken()
    token.cancel()
    outcome = DeterministicTrainerBackend().execute(
        plan, manifest, config, records, tmp_path, store, cancellation=token
    )
    assert outcome.run.state is RunState.CANCELLED
    assert not outcome.checkpoints


@pytest.mark.training
def test_checkpoint_integrity_retention_and_evidence(tmp_path: Path, dataset_path: Path) -> None:
    dataset, config, model, plan, store, outcome = execute(
        tmp_path, dataset_path, TrainingMethod.LORA
    )
    with pytest.raises(ValueError, match="model manifest"):
        validate_checkpoint(tmp_path, outcome.checkpoints[-1], "wrong", config)
    removed = apply_retention(tmp_path, outcome.checkpoints, keep_last_n=1)
    assert len(removed) == 1
    assert checkpoint_directory(tmp_path, outcome.checkpoints[-1]).is_dir()
    with pytest.raises(FileNotFoundError):
        checkpoint_directory(tmp_path, removed[0])
    evaluation = EvaluationResult(
        suite_id="suite",
        suite_version="1",
        subject_id=outcome.run.id,
        metrics={"loss": 1.0},
        case_results={"case": 1.0},
        passed=True,
    )
    evidence = build_training_evidence(
        outcome.run,
        dataset,
        model,
        plan,
        store,
        evaluation,
        RegistryStatus.EVALUATED,
    )
    report = generate_training_report(outcome.run, model, dataset, plan, store, evaluation)
    card = generate_model_card(outcome.run, model, dataset, plan, evaluation)
    archive, first_hash = write_evidence_bundle(tmp_path, outcome.run, evidence, report, card)
    assert archive.is_file()
    assert "no performance claim" not in card
    archive, second_hash = write_evidence_bundle(tmp_path, outcome.run, evidence, report, card)
    assert first_hash == second_hash == file_sha256(archive)
