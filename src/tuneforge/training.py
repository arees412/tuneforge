"""Real, bounded, offline tiny-transformer training backends."""

from __future__ import annotations

import random
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import PeftModel
from torch import nn
from torch.optim import AdamW
from transformers import GPT2LMHeadModel

from tuneforge import __version__
from tuneforge.checkpoints import (
    apply_retention,
    run_directory,
    save_checkpoint,
    verified_checkpoint_files,
)
from tuneforge.dataset import verified_split
from tuneforge.domain import (
    CanonicalTrainingRecord,
    CheckpointManifest,
    DatasetManifest,
    LoRAConfig,
    ModelManifest,
    RecordedTrainingConfig,
    RunState,
    TrainingConfig,
    TrainingMethod,
    TrainingMetric,
    TrainingPlan,
    TrainingRun,
)
from tuneforge.modeling import apply_lora, create_tiny_model, parameter_report
from tuneforge.planning import checkpoint_strategy, validate_cpu_config
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
        self._lock = threading.Lock()
        self._closed = False

    def cancel(self) -> bool:
        with self._lock:
            if self._closed:
                return False
            self._event.set()
            return True

    def finish(self) -> bool:
        with self._lock:
            self._closed = True
            return self._event.is_set()

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
        records: list[CanonicalTrainingRecord] | DatasetManifest,
        artifact_root: Path,
        store: MetadataStore,
        *,
        experiment_id: str | None = None,
        cancellation: CancellationToken | None = None,
        resume: CheckpointManifest | None = None,
        loss_hook: LossHook | None = None,
        on_started: Callable[[str], None] | None = None,
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
    artifact_root: Path,
    config: RecordedTrainingConfig,
    model_hash: str,
    store: MetadataStore,
    owner: str,
) -> nn.Module:
    with verified_checkpoint_files(
        artifact_root, checkpoint, model_hash, config, store, owner
    ) as directory:
        if checkpoint.method is TrainingMethod.FULL_FINE_TUNE:
            return GPT2LMHeadModel.from_pretrained(
                directory, local_files_only=True, use_safetensors=True
            )
        base = create_tiny_model(seed=config.seed)
        return PeftModel.from_pretrained(base, directory, is_trainable=True, local_files_only=True)


def verify_checkpoint_reload(
    artifact_root: Path,
    checkpoint: CheckpointManifest,
    model_manifest: ModelManifest,
    config: TrainingConfig,
    store: MetadataStore,
    owner: str,
) -> bool:
    reloaded = load_checkpoint_model(
        checkpoint, artifact_root, config, model_manifest.manifest_hash, store, owner
    )
    reloaded.eval()
    with torch.no_grad():
        output = reloaded(input_ids=torch.tensor([[1, 4, 5, 2]], dtype=torch.long))
    return bool(torch.isfinite(output.logits).all())


class WorkerBusyError(RuntimeError):
    pass


_worker_lock = threading.Lock()


@contextmanager
def local_worker() -> Iterator[None]:
    """Model planning and execution both use the process-wide PyTorch random state."""
    if not _worker_lock.acquire(blocking=False):
        raise WorkerBusyError("another local training run is active")
    try:
        yield
    finally:
        _worker_lock.release()


class DeterministicTrainerBackend(TrainerBackend):
    """Direct PyTorch backend used for real offline CPU validation and CI."""

    @local_worker()
    def execute(
        self,
        plan: TrainingPlan,
        model_manifest: ModelManifest,
        training_config: TrainingConfig,
        records: list[CanonicalTrainingRecord] | DatasetManifest,
        artifact_root: Path,
        store: MetadataStore,
        *,
        experiment_id: str | None = None,
        cancellation: CancellationToken | None = None,
        resume: CheckpointManifest | None = None,
        loss_hook: LossHook | None = None,
        on_started: Callable[[str], None] | None = None,
    ) -> TrainingOutcome:
        training_config = validate_cpu_config(training_config)
        if plan.checkpoint_strategy != checkpoint_strategy(training_config):
            raise ValueError("training plan checkpoint strategy differs from configuration")
        if plan.evaluation_plan != {"strategy": training_config.evaluation_strategy}:
            raise ValueError("training plan evaluation strategy differs from configuration")
        registered = store.get_dataset(plan.dataset_fingerprint)
        if registered is None:
            raise ValueError("training requires a registered dataset")
        if isinstance(records, DatasetManifest):
            if resume is None and records != registered:
                raise ValueError("registered dataset manifest mismatch")
            if records.fingerprint != plan.dataset_fingerprint:
                raise ValueError("run dataset fingerprint mismatch")
            registered = records
        train_records = verified_split(registered, "train")
        if isinstance(records, list):
            # Older callers may supply either all normalized records or the train partition.
            supplied = {record.id: record for record in records}
            if any(supplied.get(record.id) != record for record in train_records):
                raise ValueError("supplied records differ from the registered training snapshot")
        if plan.effective_batch_size != 1:
            raise ValueError("CPU training plan requires effective batch size 1")
        budget = min(training_config.max_steps, len(train_records) * training_config.epochs)
        if plan.training_config_hash != stable_digest(training_config):
            raise ValueError("training plan configuration mismatch; replan this workflow")
        if budget != plan.estimated_steps:
            raise ValueError("training plan step budget mismatch")
        if (
            plan.model_manifest_hash != model_manifest.manifest_hash
            or plan.method != model_manifest.training_method
            or stable_digest(
                {
                    k: getattr(model_manifest, k)
                    for k in ModelManifest.model_fields
                    if k != "manifest_hash"
                }
            )
            != model_manifest.manifest_hash
        ):
            raise ValueError("training plan model identity mismatch")
        plan_payload = plan.model_dump(exclude={"plan_hash"}, exclude_none=True)
        if plan.plan_hash != stable_digest(plan_payload):
            raise ValueError("training plan integrity check failed")
        run: TrainingRun | None = None
        cancellation = cancellation or CancellationToken()
        checkpoints: list[CheckpointManifest] = []
        metrics: list[TrainingMetric] = []
        report: dict[str, float | int] = {}
        reload_verified = False
        try:
            set_reproducibility(training_config.seed)
            if resume is not None:
                parent = store.get_run(resume.run_id)
                if parent is None or parent.dataset_fingerprint != registered.fingerprint:
                    raise ValueError("checkpoint dataset identity mismatch")
                if parent.environment.get("plan_hash", plan.plan_hash) != plan.plan_hash:
                    raise ValueError("checkpoint training plan mismatch")
                if parent.environment.get(
                    "dataset_manifest_hash", stable_digest(registered)
                ) != stable_digest(registered):
                    raise ValueError("checkpoint dataset manifest mismatch")
                model = load_checkpoint_model(
                    resume,
                    artifact_root,
                    training_config,
                    model_manifest.manifest_hash,
                    store,
                    resume.run_id,
                )
                start_step = resume.step
                if start_step > budget or (
                    start_step == budget and parent.state is RunState.COMPLETED
                ):
                    raise ValueError("checkpoint has exhausted the total step budget")
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

            run = _create_run(plan, training_config, store, experiment_id)
            run.environment["plan_hash"] = plan.plan_hash
            run.environment["dataset_manifest_hash"] = stable_digest(registered)
            if resume is not None:
                run.environment["resumed_from_run_id"] = resume.run_id
                run.environment["resumed_from_checkpoint_id"] = resume.checkpoint_id
            store.save_run(run)
            output_directory = run_directory(artifact_root, run.id)
            for filename, payload in (
                ("training-plan.json", plan),
                ("model-manifest.json", model_manifest),
                ("training-config.json", training_config),
                ("dataset-manifest.json", registered),
            ):
                write_canonical_json(output_directory / filename, payload)
            optimizer = AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=training_config.learning_rate,
                weight_decay=training_config.weight_decay,
            )
            run = store.transition_run(run.id, RunState.RUNNING, updated_at=utc_now())
            if on_started is not None:
                on_started(run.id)
            model.train()
            final_step = start_step
            for step in range(start_step + 1, budget + 1):
                if cancellation.requested:
                    run = store.transition_run(
                        run.id,
                        RunState.CANCELLED,
                        updated_at=utc_now(),
                        last_completed_step=final_step,
                    )
                    return TrainingOutcome(
                        run, metrics, store.list_checkpoints(run.id), report, False
                    )
                started = time.perf_counter()
                record = train_records[(step - 1) % len(train_records)]
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
                final_step = step
                duration = time.perf_counter() - started
                metric = TrainingMetric(
                    run_id=run.id,
                    step=step,
                    epoch=step / len(train_records),
                    training_loss=float(loss.detach()),
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    duration_seconds=duration,
                    trainable_parameters=int(report["trainable_parameters"]),
                    total_parameters=int(report["total_parameters"]),
                )
                metrics.append(metric)
                store.record_metric(metric)
                should_save = step % training_config.checkpoint_frequency == 0 or step == budget
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
                        store=store,
                    )
                    checkpoints.append(checkpoint)
                    apply_retention(artifact_root, store, run.id, training_config.keep_last_n)
            if cancellation.finish():
                run = store.transition_run(
                    run.id,
                    RunState.CANCELLED,
                    updated_at=utc_now(),
                    last_completed_step=final_step,
                )
                return TrainingOutcome(run, metrics, store.list_checkpoints(run.id), report, False)
            if resume is not None and start_step == budget:
                # Republish final weights when only finalization was interrupted.
                checkpoints.append(
                    save_checkpoint(
                        model,
                        artifact_root,
                        run.id,
                        start_step,
                        resume.epoch,
                        plan.method,
                        model_manifest,
                        training_config,
                        resume.metrics,
                        store=store,
                    )
                )
            checkpoints = store.list_checkpoints(run.id)
            if not checkpoints:
                raise RuntimeError("training completed without a checkpoint")
            reload_verified = verify_checkpoint_reload(
                artifact_root, checkpoints[-1], model_manifest, training_config, store, run.id
            )
            if not reload_verified:
                raise RuntimeError("checkpoint reload verification failed")
            run = store.transition_run(
                run.id,
                RunState.EVALUATING,
                updated_at=utc_now(),
                last_completed_step=final_step,
            )
            finished_at = utc_now()
            completed = run.model_copy(
                update={"state": RunState.COMPLETED, "updated_at": finished_at}
            )
            write_canonical_json(output_directory / "metrics.json", metrics)
            write_canonical_json(output_directory / "run.json", completed)
            run = store.transition_run(
                run.id,
                RunState.COMPLETED,
                updated_at=finished_at,
                last_completed_step=final_step,
            )
            return TrainingOutcome(run, metrics, checkpoints, report, reload_verified)
        except Exception as exc:
            current = store.get_run(run.id) if run is not None else None
            if current is not None and current.state not in {
                RunState.COMPLETED,
                RunState.FAILED,
                RunState.CANCELLED,
            }:
                if current.state is RunState.EVALUATING:
                    (run_directory(artifact_root, current.id) / "run.json").unlink(missing_ok=True)
                store.transition_run(
                    current.id,
                    RunState.FAILED,
                    updated_at=utc_now(),
                    failure_category=type(exc).__name__,
                    last_completed_step=metrics[-1].step
                    if metrics
                    else (resume.step if resume else 0),
                )
            raise
        finally:
            cancellation.finish()


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
