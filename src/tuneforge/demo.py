"""Offline end-to-end demonstration using only generated local fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tuneforge.checkpoints import apply_retention, checkpoint_directory
from tuneforge.dataset import build_dataset_manifest
from tuneforge.domain import (
    EvaluationResult,
    LicenseMetadata,
    LoRAConfig,
    RegistryStatus,
    SplitConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.evaluation import compare_results, evaluate_language_model_loss
from tuneforge.evidence import (
    audit_event,
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
from tuneforge.registry import ModelRegistry
from tuneforge.runtime import detect_runtime
from tuneforge.storage import MetadataStore
from tuneforge.training import DeterministicTrainerBackend, load_checkpoint_model
from tuneforge.utils import stable_digest, write_canonical_json


def _write_fixture(path: Path) -> None:
    records = [
        {
            "id": f"example-{index}",
            "instruction": f"Return the number {index} as a word.",
            "input": "Use lowercase ASCII.",
            "output": value,
            "metadata": {"fixture": True},
        }
        for index, value in enumerate(
            ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        )
    ]
    write_canonical_json(path, records)


def run_demo(root: Path, *, method: TrainingMethod = TrainingMethod.LORA) -> dict[str, Any]:
    """Execute dataset -> plan -> real training -> reload -> evaluation -> evidence."""

    root.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "generated-demo-dataset.json"
    _write_fixture(dataset_path)
    store = MetadataStore(root / "tuneforge.sqlite3")
    dataset, records = build_dataset_manifest(
        dataset_path,
        SplitConfig(train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=17),
        license_metadata=LicenseMetadata(
            identifier="CC0-1.0",
            usage_restrictions="Generated deterministic test fixture only",
            verified=True,
        ),
        provenance={"generator": "tuneforge.demo", "contains_real_user_data": "false"},
    )
    store.register_dataset(dataset.fingerprint, dataset)
    config = TrainingConfig(
        max_steps=3,
        per_device_batch_size=1,
        max_sequence_length=24,
        checkpoint_frequency=2,
        keep_last_n=1,
        seed=17,
    )
    base = create_tiny_model(seed=config.seed)
    adapter = (
        LoRAConfig(r=2, alpha=4, target_modules=["c_attn"])
        if method is TrainingMethod.LORA
        else None
    )
    model_manifest = build_model_manifest(tiny_model_spec(base), method, adapter)
    planned_model = apply_lora(base, adapter) if adapter is not None else base
    counts = parameter_report(planned_model)
    plan = build_training_plan(
        dataset,
        model_manifest,
        config,
        detect_runtime(),
        trainable_parameters=int(counts["trainable_parameters"]),
        adapter=adapter,
    )
    outcome = DeterministicTrainerBackend().execute(
        plan, model_manifest, config, records, root, store
    )
    final_checkpoint = outcome.checkpoints[-1]
    final_directory = checkpoint_directory(root, final_checkpoint)
    trained_model = load_checkpoint_model(final_checkpoint, final_directory, config)
    evaluation_records = [records[-1]]
    baseline_loss = evaluate_language_model_loss(
        create_tiny_model(seed=config.seed),
        evaluation_records,
        max_length=config.max_sequence_length,
    )
    candidate_loss = evaluate_language_model_loss(
        trained_model, evaluation_records, max_length=config.max_sequence_length
    )
    baseline_result = EvaluationResult(
        suite_id="tiny-held-out-loss",
        suite_version="1",
        subject_id="local-random-baseline",
        metrics={"loss": baseline_loss},
        case_results={evaluation_records[0].id: baseline_loss},
        passed=True,
        warnings=["Tiny fixture loss is a pipeline check, not a general benchmark."],
    )
    candidate_result = EvaluationResult(
        suite_id="tiny-held-out-loss",
        suite_version="1",
        subject_id=outcome.run.id,
        metrics={"loss": candidate_loss},
        case_results={evaluation_records[0].id: candidate_loss},
        passed=True,
        warnings=["Tiny fixture loss is a pipeline check, not a general benchmark."],
    )
    comparison = compare_results(baseline_result, candidate_result, allowed_regression=1.0)
    evaluation_id = f"eval-{stable_digest(candidate_result)[:16]}"
    store.save_evaluation(evaluation_id, candidate_result)
    registry = ModelRegistry(store)
    entry = registry.register(
        outcome.run.id,
        "adapter" if method is TrainingMethod.LORA else "full",
        final_directory.as_posix(),
    )
    entry = registry.transition(entry.id, RegistryStatus.EVALUATED, evaluation_id=evaluation_id)
    report = generate_training_report(
        outcome.run, model_manifest, dataset, plan, store, candidate_result
    )
    card = generate_model_card(outcome.run, model_manifest, dataset, plan, candidate_result)
    evidence = build_training_evidence(
        outcome.run, dataset, model_manifest, plan, store, candidate_result, entry.status
    )
    archive, archive_hash = write_evidence_bundle(root, outcome.run, evidence, report, card)
    event = audit_event(
        "demo.completed",
        outcome.run.id,
        {"evidence_hash": evidence.evidence_hash, "archive_sha256": archive_hash},
    )
    store.record_audit(event)
    removed = apply_retention(root, outcome.checkpoints, config.keep_last_n)
    for checkpoint in removed:
        store.delete_checkpoint_metadata(checkpoint.checkpoint_id)
    result: dict[str, Any] = {
        "run_id": outcome.run.id,
        "state": outcome.run.state.value,
        "method": method.value,
        "steps": outcome.run.last_completed_step,
        "training_losses": [metric.training_loss for metric in outcome.metrics],
        "baseline_loss": baseline_loss,
        "candidate_loss": candidate_loss,
        "comparison": comparison.model_dump(mode="json"),
        "parameters": outcome.parameter_report,
        "adapter_reload_verified": outcome.adapter_reload_verified,
        "retained_checkpoints": len(store.list_checkpoints(outcome.run.id)),
        "evidence_bundle": archive.as_posix(),
        "evidence_bundle_sha256": archive_hash,
        "dataset_fingerprint": dataset.fingerprint,
        "model_manifest_hash": model_manifest.manifest_hash,
        "training_plan_hash": plan.plan_hash,
        "qlora_executed": False,
    }
    write_canonical_json(root / "demo-result.json", result)
    return result


__all__ = ["run_demo"]
