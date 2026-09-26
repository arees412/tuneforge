"""Model manifests and documented PEFT/Transformers integration boundaries."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig as PeftLoraConfig
from peft import TaskType, get_peft_model
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from tuneforge.domain import (
    LoRAConfig,
    ModelManifest,
    ModelSpec,
    QLoRAConfig,
    TrainingMethod,
)
from tuneforge.utils import stable_digest


TINY_MODEL_ID = "tuneforge-local-tiny-gpt2"


def create_tiny_model(*, seed: int = 42) -> GPT2LMHeadModel:
    """Construct a randomly initialized transformer entirely from local configuration."""

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=128,
        n_positions=32,
        n_ctx=32,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def parameter_report(model: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / total if total else 0.0,
    }


def discover_target_modules(model: nn.Module) -> list[str]:
    supported = {"Linear", "Conv1D"}
    targets = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if name and module.__class__.__name__ in supported and name.rsplit(".", 1)[-1] != "lm_head"
    }
    return sorted(targets)


def validate_target_modules(model: nn.Module, configured: list[str]) -> None:
    discovered = set(discover_target_modules(model))
    missing = sorted(set(configured) - discovered)
    if missing:
        raise ValueError(f"LoRA target modules not found: {', '.join(missing)}")


def apply_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    validate_target_modules(model, config.target_modules)
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias=config.bias,
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=config.modules_to_save or None,
    )
    return get_peft_model(model, peft_config)


def build_model_manifest(
    spec: ModelSpec,
    method: TrainingMethod,
    adapter: LoRAConfig | QLoRAConfig | None = None,
) -> ModelManifest:
    config_payload: dict[str, Any] = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "architecture_family": spec.architecture_family,
        "task_type": spec.task_type,
        "context_length": spec.context_length,
    }
    configuration_hash = stable_digest(config_payload)
    payload = {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "configuration_hash": configuration_hash,
        "tokenizer_id": f"{spec.model_id}-byte-tokenizer",
        "tokenizer_revision": "1",
        "license": spec.license,
        "parameter_count": spec.parameter_count,
        "training_method": method,
        "adapter_configuration": adapter,
        "base_model_fingerprint": configuration_hash,
    }
    return ModelManifest(**payload, manifest_hash=stable_digest(payload))


def tiny_model_spec(model: nn.Module | None = None) -> ModelSpec:
    instance = model or create_tiny_model()
    return ModelSpec(
        model_id=TINY_MODEL_ID,
        architecture_family="GPT2LMHeadModel",
        source="local_config",
        parameter_count=int(parameter_report(instance)["total_parameters"]),
        dtype_support=["float32"],
        context_length=32,
        quantization_compatible=False,
    )


__all__ = [
    "TINY_MODEL_ID",
    "apply_lora",
    "build_model_manifest",
    "create_tiny_model",
    "discover_target_modules",
    "parameter_report",
    "tiny_model_spec",
    "validate_target_modules",
]

