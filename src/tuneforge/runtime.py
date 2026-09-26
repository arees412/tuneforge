"""Runtime capability detection and conservative resource estimates."""

from __future__ import annotations

import importlib.util
import platform
import sys

import torch

from tuneforge.domain import ResourceEstimate, RuntimeCapabilities, TrainingMethod


def detect_runtime() -> RuntimeCapabilities:
    cuda = torch.cuda.is_available()
    count = torch.cuda.device_count() if cuda else 0
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    memory = [torch.cuda.get_device_properties(index).total_memory for index in range(count)]
    mps_backend = getattr(torch.backends, "mps", None)
    mps = bool(mps_backend and mps_backend.is_available())
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    return RuntimeCapabilities(
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        cuda_available=cuda,
        cuda_version=torch.version.cuda,
        gpu_names=names,
        gpu_count=count,
        gpu_memory_bytes=memory,
        mps_available=mps,
        cpu=platform.processor() or platform.machine(),
        bf16_supported=bf16,
        fp16_supported=cuda or mps,
        bitsandbytes_available=importlib.util.find_spec("bitsandbytes") is not None,
        flash_attention_available=importlib.util.find_spec("flash_attn") is not None,
    )


def estimate_resources(
    parameter_count: int,
    trainable_parameter_count: int,
    method: TrainingMethod,
    sequence_length: int,
    batch_size: int,
) -> ResourceEstimate:
    base_bytes_per_parameter = 1 if method is TrainingMethod.QLORA else 4
    base = parameter_count * base_bytes_per_parameter
    trainable = trainable_parameter_count * 4
    checkpoint = trainable if method in {TrainingMethod.LORA, TrainingMethod.QLORA} else base
    workload = sequence_length * batch_size
    risk = "low" if workload <= 4096 else "medium" if workload <= 32768 else "high"
    return ResourceEstimate(
        estimated_base_model_bytes=base,
        estimated_trainable_parameter_bytes=trainable,
        optimizer_state_category="AdamW states are approximately 2x trainable FP32 parameters",
        activation_risk=risk,
        estimated_checkpoint_bytes=checkpoint,
        disclaimer=(
            "Planning estimate only; it is not a guaranteed measurement of host or accelerator memory."
        ),
    )


def require_qlora_capability(runtime: RuntimeCapabilities) -> None:
    missing: list[str] = []
    if not runtime.cuda_available:
        missing.append("CUDA")
    if not runtime.bitsandbytes_available:
        missing.append("bitsandbytes")
    if missing:
        raise RuntimeError(f"QLoRA execution unavailable; missing: {', '.join(missing)}")


def validate_precision(precision: str, runtime: RuntimeCapabilities) -> None:
    if precision == "bf16" and not runtime.bf16_supported:
        raise RuntimeError("bf16 requested but unsupported by this runtime")
    if precision == "fp16" and not runtime.fp16_supported:
        raise RuntimeError("fp16 requested but unsupported by this runtime")


__all__ = ["detect_runtime", "estimate_resources", "require_qlora_capability", "validate_precision"]

