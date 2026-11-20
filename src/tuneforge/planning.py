"""Deterministic validation and construction of executable training plans."""

from __future__ import annotations

import torch

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
from tuneforge.runtime import estimate_resources, require_qlora_capability
from tuneforge.utils import stable_digest


def effective_batch_size(config: TrainingConfig, device_count: int = 1) -> int:
    return config.per_device_batch_size * config.gradient_accumulation * max(1, device_count)


def validate_cpu_config(config: TrainingConfig) -> TrainingConfig:
    # Revalidate even model_copy(update=...) and take a private snapshot for execution.
    config = TrainingConfig.model_validate(config)
    if config.per_device_batch_size != 1 or config.gradient_accumulation != 1:
        raise ValueError("CPU training requires batch size 1 and gradient accumulation 1")
    if config.precision != "fp32" or config.max_sequence_length > 32:
        raise ValueError("CPU training requires precision='fp32' and max_sequence_length <= 32")
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled("cpu"):
        raise ValueError(
            "CPU training requires PyTorch's FP32 default dtype with CPU autocast disabled"
        )
    return config


def checkpoint_strategy(config: TrainingConfig) -> dict[str, str | int]:
    return {
        "save_strategy": config.save_strategy,
        "frequency": config.checkpoint_frequency,
        "keep_last_n": config.keep_last_n,
    }


def build_training_plan(
    dataset: DatasetManifest,
    model: ModelManifest,
    config: TrainingConfig,
    runtime: RuntimeCapabilities,
    *,
    trainable_parameters: int,
    adapter: LoRAConfig | QLoRAConfig | None = None,
) -> TrainingPlan:
    config = validate_cpu_config(config)
    if dataset.quality.valid_records == 0:
        raise ValueError("training plan requires at least one valid record")
    if dataset.leakage_findings:
        raise ValueError("training plan rejected because cross-split leakage was detected")
    if model.parameter_count is None or model.parameter_count <= 0:
        raise ValueError("model parameter count is required for execution planning")
    if trainable_parameters <= 0:
        raise ValueError("training plan has zero trainable parameters")
    if model.training_method is TrainingMethod.QLORA:
        require_qlora_capability(runtime)
        if not isinstance(adapter, QLoRAConfig):
            raise ValueError("QLoRA training requires a QLoRA configuration")
    if model.training_method is TrainingMethod.LORA and not isinstance(adapter, LoRAConfig):
        raise ValueError("LoRA training requires a LoRA configuration")
    train_count = dataset.split.record_counts["train"]
    if not train_count:
        raise ValueError("required train split is empty")
    batch = effective_batch_size(config)
    estimated = min(config.max_steps, train_count * config.epochs)
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
        "training_config_hash": stable_digest(config),
        "model_manifest_hash": model.manifest_hash,
        "method": model.training_method,
        "runtime": runtime,
        "effective_batch_size": batch,
        "estimated_steps": estimated,
        "adapter_configuration": adapter,
        "checkpoint_strategy": checkpoint_strategy(config),
        "evaluation_plan": {"strategy": config.evaluation_strategy},
        "warnings": warnings,
        "resource_estimate": estimate,
    }
    plan = TrainingPlan(**payload, plan_hash="pending")
    plan.plan_hash = stable_digest(plan.model_dump(exclude={"plan_hash"}, exclude_none=True))
    return plan


__all__ = ["build_training_plan", "effective_batch_size"]
