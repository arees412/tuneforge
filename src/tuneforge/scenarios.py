"""Twelve deterministic acceptance scenarios defined by the TuneForge specification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from tuneforge.checkpoints import validate_checkpoint
from tuneforge.dataset import (
    build_dataset_manifest,
    detect_leakage,
    normalize_record,
    split_records,
    validate_dataset,
)
from tuneforge.demo import run_demo
from tuneforge.domain import (
    DatasetSchema,
    LoRAConfig,
    ModelManifest,
    ModelSpec,
    SplitConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.evaluation import compare_results
from tuneforge.modeling import (
    apply_lora,
    build_model_manifest,
    create_tiny_model,
    parameter_report,
    tiny_model_spec,
)
from tuneforge.planning import build_training_plan
from tuneforge.runtime import detect_runtime, require_qlora_capability
from tuneforge.security import PathSecurityError, confined_path
from tuneforge.storage import MetadataStore
from tuneforge.training import DeterministicTrainerBackend
from tuneforge.utils import write_canonical_json


def run_scenarios(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    valid_path = root / "scenario-valid.json"
    write_canonical_json(
        valid_path,
        [{"instruction": f"Prompt {index}", "output": f"Answer {index}"} for index in range(12)],
    )
    manifest, records = build_dataset_manifest(valid_path, SplitConfig(seed=101))
    results["01_valid_instruction_dataset"] = {
        "passed": manifest.quality.valid_records == 12,
        "manifest": manifest.fingerprint,
    }

    malformed = root / "scenario-malformed.json"
    malformed.write_text('[{"instruction": 7}]\n', encoding="utf-8")
    invalid = validate_dataset(malformed)
    results["02_malformed_dataset"] = {"passed": not invalid.valid, "run_created": False}

    left = normalize_record(
        {"instruction": "same", "output": "target"}, 0, DatasetSchema.INSTRUCTION
    )
    right = left.model_copy(update={"id": "record-copy", "source_record_id": "copy"})
    leakage = detect_leakage({"train": [left], "test": [right]})
    results["03_duplicate_leakage"] = {"passed": bool(leakage), "findings": len(leakage)}

    split_one = split_records(records, SplitConfig(seed=202))
    split_two = split_records(records, SplitConfig(seed=202))
    results["04_stable_split"] = {
        "passed": split_one.fingerprint == split_two.fingerprint,
        "digest": split_one.fingerprint,
    }

    full = run_demo(root / "full", method=TrainingMethod.FULL_FINE_TUNE)
    results["05_tiny_full_training"] = {
        "passed": full["state"] == "completed" and full["steps"] > 0,
        "run_id": full["run_id"],
        "losses": full["training_losses"],
    }

    lora_root = root / "lora"
    lora = run_demo(lora_root, method=TrainingMethod.LORA)
    results["06_tiny_lora_training"] = {
        "passed": lora["adapter_reload_verified"]
        and lora["parameters"]["trainable_parameters"] < lora["parameters"]["total_parameters"],
        "run_id": lora["run_id"],
        "parameters": lora["parameters"],
    }

    runtime = detect_runtime().model_copy(
        update={"cuda_available": False, "bitsandbytes_available": False}
    )
    try:
        require_qlora_capability(runtime)
        qlora_blocked = False
    except RuntimeError:
        qlora_blocked = True
    results["07_unsupported_qlora"] = {"passed": qlora_blocked, "executed": False}

    store = MetadataStore(lora_root / "tuneforge.sqlite3")
    lora_run_id = str(lora["run_id"])
    checkpoint = store.list_checkpoints(lora_run_id)[-1]
    run_root = lora_root / "runs" / lora_run_id
    model_manifest = ModelManifest.model_validate_json(
        (run_root / "model-manifest.json").read_text(encoding="utf-8")
    )
    config = TrainingConfig.model_validate_json(
        (run_root / "training-config.json").read_text(encoding="utf-8")
    )
    validate_checkpoint(lora_root, checkpoint, model_manifest.manifest_hash, config)
    try:
        validate_checkpoint(lora_root, checkpoint, "mismatched-manifest", config)
        mismatch_rejected = False
    except ValueError:
        mismatch_rejected = True
    results["08_resume_validation"] = {
        "passed": mismatch_rejected,
        "valid_checkpoint": checkpoint.checkpoint_id,
    }

    try:
        ModelSpec(model_id="unsafe", trust_remote_code_required=True)
        remote_denied = False
    except ValidationError:
        remote_denied = True
    results["09_remote_code_policy"] = {"passed": remote_denied}

    baseline_id = f"eval-{full['run_id']}"
    candidate_id = f"eval-{lora['run_id']}"
    from tuneforge.domain import EvaluationResult

    baseline = EvaluationResult(
        suite_id="scenario-loss",
        suite_version="1",
        subject_id=baseline_id,
        metrics={"loss": float(full["candidate_loss"])},
        case_results={"held-out": float(full["candidate_loss"])},
        passed=True,
    )
    candidate = EvaluationResult(
        suite_id="scenario-loss",
        suite_version="1",
        subject_id=candidate_id,
        metrics={"loss": float(lora["candidate_loss"])},
        case_results={"held-out": float(lora["candidate_loss"])},
        passed=True,
    )
    comparison = compare_results(baseline, candidate, allowed_regression=1.0)
    results["10_regression_comparison"] = {
        "passed": comparison.passed,
        "suite_id": comparison.suite_id,
    }

    try:
        confined_path(root, "../escape")
        path_rejected = False
    except PathSecurityError:
        path_rejected = True
    results["11_malicious_path"] = {"passed": path_rejected}

    failing_root = root / "non-finite"
    failing_store = MetadataStore(failing_root / "tuneforge.sqlite3")
    failure_manifest, failure_records = build_dataset_manifest(valid_path, SplitConfig(seed=101))
    failure_config = TrainingConfig(max_steps=1, max_sequence_length=16, checkpoint_frequency=1)
    base_model = create_tiny_model(seed=failure_config.seed)
    adapter = LoRAConfig(r=2, alpha=4)
    failure_model = build_model_manifest(tiny_model_spec(base_model), TrainingMethod.LORA, adapter)
    adapted = apply_lora(base_model, adapter)
    failure_plan = build_training_plan(
        failure_manifest,
        failure_model,
        failure_config,
        detect_runtime(),
        trainable_parameters=int(parameter_report(adapted)["trainable_parameters"]),
        adapter=adapter,
    )
    try:
        DeterministicTrainerBackend().execute(
            failure_plan,
            failure_model,
            failure_config,
            failure_records,
            failing_root,
            failing_store,
            loss_hook=lambda _step, loss: loss * torch.tensor(float("nan")),
        )
        non_finite_failed = False
    except FloatingPointError:
        non_finite_failed = True
    failed_runs = []
    with failing_store.connect() as connection:
        failed_runs = list(connection.execute("SELECT state FROM runs"))
    results["12_non_finite_loss"] = {
        "passed": non_finite_failed
        and bool(failed_runs)
        and all(row["state"] == "failed" for row in failed_runs),
        "checkpoint_count": sum(
            len(failing_store.list_checkpoints(row["id"]))
            for row in failing_store.connect().execute("SELECT id FROM runs")
        ),
    }

    results["all_passed"] = all(
        bool(value["passed"]) for key, value in results.items() if key != "all_passed"
    )
    return results


__all__ = ["run_scenarios"]
