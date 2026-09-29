"""Real, bounded, offline tiny-transformer training backends."""

from __future__ import annotations

import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch import nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel

from tuneforge import __version__
from tuneforge.checkpoints import run_directory, save_checkpoint, validate_checkpoint
from tuneforge.domain import (
    CanonicalTrainingRecord,
    CheckpointManifest,
    LoRAConfig,
    ModelManifest,
    RunState,
    TrainingConfig,
    TrainingMethod,
    TrainingMetric,
    TrainingPlan,
    TrainingRun,
)
from tuneforge.modeling import apply_lora, create_tiny_model, parameter_report
from tuneforge.storage import MetadataStore
from tuneforge.utils import stable_digest, utc_now, write_canonical_json

LossHook = Callable[[int, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainingOutcome:
    run: TrainingRun
    metrics: list[TrainingMetric]
    checkpoints: list[CheckpointManifest]
    parameter_report: dict[str, float | int]
    adapter_reload_verified: bool


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def requested(self) -> bool:
        return self._event.is_set()


class TrainerBackend(ABC):
    @abstractmethod
    def execute(
        self,
        plan: TrainingPlan,
        model_manifest: ModelManifest,
        training_config: TrainingConfig,
        records: list[CanonicalTrainingRecord],
        artifact_root: Path,
        store: MetadataStore,
        *,
        experiment_id: str | None = None,
        cancellation: CancellationToken | None = None,
        resume: CheckpointManifest | None = None,
        loss_hook: LossHook | None = None,
    ) -> TrainingOutcome:
        raise NotImplementedError


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _encode(text: str, max_length: int) -> list[int]:
    """Deterministic byte tokenizer reserved for the local tiny-model path."""

    payload = [1, *(3 + byte % 125 for byte in text.encode("utf-8")), 2][:max_length]
    if len(payload) < 2:
        payload = [1, 2]
    return payload


def _batch(record: CanonicalTrainingRecord, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    target = record.response or record.chosen or ""
    values = _encode(f"{record.prompt}\n{target}", max_length)
    input_ids = torch.tensor([values], dtype=torch.long)
    return input_ids, input_ids.clone()


def _create_run(
    plan: TrainingPlan,
    config: TrainingConfig,
    store: MetadataStore,
    experiment_id: str | None,
) -> TrainingRun:
    now = utc_now()
    run = TrainingRun(
        # Keep governed artifact paths below legacy Windows path-length limits.
        id=f"run-{uuid.uuid4().hex[:12]}",
        experiment_id=experiment_id,
        dataset_fingerprint=plan.dataset_fingerprint,
        model_manifest_hash=plan.model_manifest_hash,
        training_config_hash=stable_digest(config),
        package_version=__version__,
        seed=config.seed,
        environment=plan.runtime.model_dump(mode="json"),
        created_at=now,
        updated_at=now,
    )
    store.save_run(run)
    for state in (RunState.VALIDATED, RunState.PLANNED, RunState.QUEUED):
        run = store.transition_run(run.id, state, updated_at=utc_now())
    return run


def load_checkpoint_model(
    checkpoint: CheckpointManifest,
    directory: Path,
    config: TrainingConfig,
) -> nn.Module:
    if checkpoint.method is TrainingMethod.FULL_FINE_TUNE:
        return GPT2LMHeadModel.from_pretrained(directory, local_files_only=True)
    base = create_tiny_model(seed=config.seed)
    return PeftModel.from_pretrained(base, directory, is_trainable=True, local_files_only=True)


def verify_checkpoint_reload(
    artifact_root: Path,
    checkpoint: CheckpointManifest,
    model_manifest: ModelManifest,
    config: TrainingConfig,
) -> bool:
    directory = validate_checkpoint(artifact_root, checkpoint, model_manifest.manifest_hash, config)
    reloaded = load_checkpoint_model(checkpoint, directory, config)
    reloaded.eval()
    with torch.no_grad():
        output = reloaded(input_ids=torch.tensor([[1, 4, 5, 2]], dtype=torch.long))
    return bool(torch.isfinite(output.logits).all())


class DeterministicTrainerBackend(TrainerBackend):
    """Direct PyTorch backend used for real offline CPU validation and CI."""

    _active_lock = threading.Lock()

    def execute(
        self,
        plan: TrainingPlan,
        model_manifest: ModelManifest,
        training_config: TrainingConfig,
        records: list[CanonicalTrainingRecord],
        artifact_root: Path,
        store: MetadataStore,
        *,
        experiment_id: str | None = None,
        cancellation: CancellationToken | None = None,
        resume: CheckpointManifest | None = None,
        loss_hook: LossHook | None = None,
    ) -> TrainingOutcome:
        if not records:
            raise ValueError("training requires at least one normalized record")
        if training_config.max_sequence_length > 32:
            raise ValueError("local tiny model supports a maximum sequence length of 32")
        plan_payload = plan.model_dump(exclude={"plan_hash"}, exclude_none=True)
        if plan.plan_hash != stable_digest(plan_payload):
            raise ValueError("training plan integrity check failed")
        if not self._active_lock.acquire(blocking=False):
            raise RuntimeError("another local training run is active")
        run = _create_run(plan, training_config, store, experiment_id)
        cancellation = cancellation or CancellationToken()
        checkpoints: list[CheckpointManifest] = []
        metrics: list[TrainingMetric] = []
        report: dict[str, float | int] = {}
        reload_verified = False
        try:
            set_reproducibility(training_config.seed)
            if resume is not None:
                directory = validate_checkpoint(
                    artifact_root, resume, model_manifest.manifest_hash, training_config
                )
                if resume.run_id != run.id:
                    # A resumed execution is a new governed run linked by its checkpoint evidence.
                    run.environment["resumed_from_run_id"] = resume.run_id
                    run.environment["resumed_from_checkpoint_id"] = resume.checkpoint_id
                    store.save_run(run)
                model = load_checkpoint_model(resume, directory, training_config)
                start_step = resume.step
            else:
                model = create_tiny_model(seed=training_config.seed)
                if plan.method is TrainingMethod.LORA:
                    adapter = model_manifest.adapter_configuration
                    if not isinstance(adapter, LoRAConfig):
                        raise ValueError("LoRA execution requires a validated LoRA configuration")
                    model = apply_lora(model, adapter)
                elif plan.method is TrainingMethod.QLORA:
                    raise RuntimeError(
                        "QLoRA execution is capability-gated and not implemented by the CPU backend"
                    )
                start_step = 0
            report = parameter_report(model)
            if int(report["trainable_parameters"]) <= 0:
                raise RuntimeError("model exposes zero trainable parameters")
            if plan.method is TrainingMethod.LORA and int(report["trainable_parameters"]) >= int(
                report["total_parameters"]
            ):
                raise RuntimeError("LoRA did not reduce the trainable parameter count")

            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
            run = store.transition_run(run.id, RunState.RUNNING, updated_at=utc_now())
            model.train()
            final_step = start_step
            steps_to_execute = min(plan.estimated_steps, training_config.max_steps)
            for offset in range(1, steps_to_execute + 1):
                if cancellation.requested:
                    run = store.transition_run(
                        run.id,
                        RunState.CANCELLED,
                        updated_at=utc_now(),
                        last_completed_step=final_step,
                    )
                    return TrainingOutcome(run, metrics, checkpoints, report, False)
                step = start_step + offset
                final_step = step
                started = time.perf_counter()
                record = records[(step - 1) % len(records)]
                input_ids, labels = _batch(record, training_config.max_sequence_length)
                optimizer.zero_grad(set_to_none=True)
                output = model(input_ids=input_ids, labels=labels)
                loss = output.loss
                if loss_hook is not None:
                    loss = loss_hook(step, loss)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite training loss detected")
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ]
                if not gradients or not all(bool(torch.isfinite(item).all()) for item in gradients):
                    raise FloatingPointError("missing or non-finite trainable gradients detected")
                torch.nn.utils.clip_grad_norm_(
                    (parameter for parameter in model.parameters() if parameter.requires_grad),
                    training_config.gradient_clipping,
                )
                optimizer.step()
                duration = time.perf_counter() - started
                metric = TrainingMetric(
                    run_id=run.id,
                    step=step,
                    epoch=step / max(1, steps_to_execute),
                    training_loss=float(loss.detach()),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    duration_seconds=duration,
                    trainable_parameters=int(report["trainable_parameters"]),
                    total_parameters=int(report["total_parameters"]),
                )
                metrics.append(metric)
                store.record_metric(metric)
                should_save = (
                    step % training_config.checkpoint_frequency == 0 or offset == steps_to_execute
                )
                if should_save:
                    checkpoint = save_checkpoint(
                        model,
                        artifact_root,
                        run.id,
                        step,
                        metric.epoch,
                        plan.method,
                        model_manifest,
                        training_config,
                        {"training_loss": metric.training_loss},
                    )
                    checkpoints.append(checkpoint)
                    store.record_checkpoint(checkpoint)
            if not checkpoints:
                raise RuntimeError("training completed without a checkpoint")
            reload_verified = verify_checkpoint_reload(
                artifact_root, checkpoints[-1], model_manifest, training_config
            )
            if not reload_verified:
                raise RuntimeError("checkpoint reload verification failed")
            run = store.transition_run(
                run.id,
                RunState.EVALUATING,
                updated_at=utc_now(),
                last_completed_step=final_step,
            )
            run = store.transition_run(
                run.id,
                RunState.COMPLETED,
                updated_at=utc_now(),
                last_completed_step=final_step,
            )
            output_directory = run_directory(artifact_root, run.id)
            write_canonical_json(output_directory / "run.json", run)
            write_canonical_json(output_directory / "training-plan.json", plan)
            write_canonical_json(output_directory / "model-manifest.json", model_manifest)
            write_canonical_json(output_directory / "training-config.json", training_config)
            write_canonical_json(output_directory / "metrics.json", metrics)
            return TrainingOutcome(run, metrics, checkpoints, report, reload_verified)
        except Exception as exc:
            current = store.get_run(run.id)
            if current is not None and current.state not in {
                RunState.COMPLETED,
                RunState.FAILED,
                RunState.CANCELLED,
            }:
                store.transition_run(
                    run.id,
                    RunState.FAILED,
                    updated_at=utc_now(),
                    failure_category=type(exc).__name__,
                    last_completed_step=len(metrics),
                )
            raise
        finally:
            self._active_lock.release()


class TransformersTrainerBackend(DeterministicTrainerBackend):
    """Reserved public backend name; v0.1 delegates to the governed direct PyTorch loop."""


__all__ = [
    "CancellationToken",
    "DeterministicTrainerBackend",
    "TrainerBackend",
    "TrainingOutcome",
    "TransformersTrainerBackend",
    "load_checkpoint_model",
    "set_reproducibility",
    "verify_checkpoint_reload",
]
