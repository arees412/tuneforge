"""Deterministic validation and construction of executable training plans."""

from __future__ import annotations

import math

from tuneforge.domain import (
    DatasetManifest,
    LoRAConfig,
    ModelManifest,
    QLoRAConfig,
    RuntimeCapabilities,
    TrainingConfig,
    TrainingMethod,
    TrainingPlan,
)
from tuneforge.runtime import estimate_resources, require_qlora_capability, validate_precision
from tuneforge.utils import stable_digest


def effective_batch_size(config: TrainingConfig, device_count: int = 1) -> int:
    return config.per_device_batch_size * config.gradient_accumulation * max(1, device_count)


def build_training_plan(
    dataset: DatasetManifest,
    model: ModelManifest,
    config: TrainingConfig,
    runtime: RuntimeCapabilities,
    *,
    trainable_parameters: int,
    adapter: LoRAConfig | QLoRAConfig | None = None,
) -> TrainingPlan:
    if dataset.quality.valid_records == 0:
        raise ValueError("training plan requires at least one valid record")
    if dataset.leakage_findings:
        raise ValueError("training plan rejected because cross-split leakage was detected")
    if model.parameter_count is None or model.parameter_count <= 0:
        raise ValueError("model parameter count is required for execution planning")
    if trainable_parameters <= 0:
        raise ValueError("training plan has zero trainable parameters")
    validate_precision(config.precision, runtime)
    if model.training_method is TrainingMethod.QLORA:
        require_qlora_capability(runtime)
        if not isinstance(adapter, QLoRAConfig):
            raise ValueError("QLoRA training requires a QLoRA configuration")
    if model.training_method is TrainingMethod.LORA and not isinstance(adapter, LoRAConfig):
        raise ValueError("LoRA training requires a LoRA configuration")
    train_count = dataset.split.record_counts["train"]
    batch = effective_batch_size(config, runtime.gpu_count or 1)
    estimated = min(config.max_steps, max(1, math.ceil(train_count / batch) * config.epochs))
    estimate = estimate_resources(
        model.parameter_count,
        trainable_parameters,
        model.training_method,
        config.max_sequence_length,
        config.per_device_batch_size,
    )
    warnings = [estimate.disclaimer]
    if runtime.gpu_count == 0:
        warnings.append("CPU execution selected; estimates do not imply accelerator availability.")
    payload = {
        "dataset_fingerprint": dataset.fingerprint,
        "model_manifest_hash": model.manifest_hash,
        "method": model.training_method,
        "runtime": runtime,
        "effective_batch_size": batch,
        "estimated_steps": estimated,
        "adapter_configuration": adapter,
        "checkpoint_strategy": {
            "save_strategy": config.save_strategy,
            "frequency": config.checkpoint_frequency,
            "keep_last_n": config.keep_last_n,
        },
        "evaluation_plan": {"strategy": config.evaluation_strategy},
        "warnings": warnings,
        "resource_estimate": estimate,
    }
    plan = TrainingPlan(**payload, plan_hash="pending")
    plan.plan_hash = stable_digest(plan.model_dump(exclude={"plan_hash"}, exclude_none=True))
    return plan


__all__ = ["build_training_plan", "effective_batch_size"]
