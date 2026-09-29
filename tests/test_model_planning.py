from __future__ import annotations

import pytest
from pydantic import ValidationError

from tuneforge.domain import (
    DatasetManifest,
    LoRAConfig,
    ModelSpec,
    QLoRAConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.modeling import (
    apply_lora,
    build_model_manifest,
    create_tiny_model,
    discover_target_modules,
    parameter_report,
    tiny_model_spec,
    validate_target_modules,
)
from tuneforge.planning import build_training_plan, effective_batch_size
from tuneforge.runtime import detect_runtime, estimate_resources, require_qlora_capability


def test_tiny_model_forward_backward_and_parameters() -> None:
    model = create_tiny_model(seed=1)
    output = model(
        input_ids=__import__("torch").tensor([[1, 4, 5, 2]]),
        labels=__import__("torch").tensor([[1, 4, 5, 2]]),
    )
    output.loss.backward()
    report = parameter_report(model)
    assert output.logits.shape == (1, 4, 128)
    assert report["total_parameters"] == report["trainable_parameters"]


def test_model_spec_license_unknown_and_remote_code_denied() -> None:
    assert ModelSpec(model_id="local").license.identifier == "unknown"
    with pytest.raises(ValidationError, match="remote code"):
        ModelSpec(model_id="unsafe", trust_remote_code_required=True)
    allowed = ModelSpec(
        model_id="explicit", trust_remote_code_required=True, trust_remote_code_allowed=True
    )
    assert allowed.trust_remote_code_allowed


def test_lora_config_discovery_injection_and_validation() -> None:
    model = create_tiny_model()
    assert "c_attn" in discover_target_modules(model)
    with pytest.raises(ValueError, match="not found"):
        validate_target_modules(model, ["does_not_exist"])
    with pytest.raises(ValidationError):
        LoRAConfig(r=0)
    adapted = apply_lora(model, LoRAConfig(r=2, alpha=4))
    counts = parameter_report(adapted)
    assert 0 < counts["trainable_parameters"] < counts["total_parameters"]


def test_model_manifest_is_stable() -> None:
    spec = tiny_model_spec()
    config = LoRAConfig(r=2)
    first = build_model_manifest(spec, TrainingMethod.LORA, config)
    second = build_model_manifest(spec, TrainingMethod.LORA, config)
    assert first.manifest_hash == second.manifest_hash


def test_runtime_detection_and_qlora_gate() -> None:
    runtime = detect_runtime()
    assert runtime.python_version
    unsupported = runtime.model_copy(
        update={"cuda_available": False, "bitsandbytes_available": False}
    )
    with pytest.raises(RuntimeError, match="QLoRA"):
        require_qlora_capability(unsupported)


def test_effective_batch_resource_estimate_and_plan(dataset_manifest: DatasetManifest) -> None:
    config = TrainingConfig(
        per_device_batch_size=2, gradient_accumulation=4, max_sequence_length=16
    )
    assert effective_batch_size(config, 2) == 16
    model = create_tiny_model()
    adapter = LoRAConfig(r=2)
    adapted = apply_lora(model, adapter)
    manifest = build_model_manifest(tiny_model_spec(model), TrainingMethod.LORA, adapter)
    plan = build_training_plan(
        dataset_manifest,
        manifest,
        config,
        detect_runtime(),
        trainable_parameters=int(parameter_report(adapted)["trainable_parameters"]),
        adapter=adapter,
    )
    assert plan.effective_batch_size == 8
    assert plan.plan_hash
    estimate = estimate_resources(1000, 100, TrainingMethod.LORA, 32, 2)
    assert estimate.estimated_checkpoint_bytes == 400


def test_invalid_plan_zero_trainable_and_qlora(dataset_manifest: DatasetManifest) -> None:
    full = build_model_manifest(tiny_model_spec(), TrainingMethod.FULL_FINE_TUNE)
    with pytest.raises(ValueError, match="zero trainable"):
        build_training_plan(
            dataset_manifest, full, TrainingConfig(), detect_runtime(), trainable_parameters=0
        )
    qlora = build_model_manifest(tiny_model_spec(), TrainingMethod.QLORA, QLoRAConfig())
    runtime = detect_runtime().model_copy(
        update={"cuda_available": False, "bitsandbytes_available": False}
    )
    with pytest.raises(RuntimeError, match="QLoRA"):
        build_training_plan(
            dataset_manifest,
            qlora,
            TrainingConfig(),
            runtime,
            trainable_parameters=10,
            adapter=QLoRAConfig(),
        )
