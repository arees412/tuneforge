from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuneforge.domain import (
    AuditEvent,
    EvaluationCase,
    EvaluationResult,
    EvaluationSuite,
    Experiment,
    RegistryStatus,
    RunState,
    TrainingMetric,
    TrainingRun,
)
from tuneforge.evaluation import compare_results, evaluate_suite
from tuneforge.publishing import HubPublisher
from tuneforge.registry import ModelRegistry
from tuneforge.security import PathSecurityError, confined_path, redact_text, validate_artifact_name
from tuneforge.storage import MetadataStore
from tuneforge.utils import stable_digest, utc_now


def make_run(identifier: str = "run-test") -> TrainingRun:
    now = utc_now()
    return TrainingRun(
        id=identifier,
        dataset_fingerprint="dataset",
        model_manifest_hash="model",
        training_config_hash="config",
        package_version="0.1.0",
        seed=42,
        environment={},
        created_at=now,
        updated_at=now,
    )


def test_storage_migration_run_transitions_and_metrics(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    store.save_experiment(Experiment(id="exp", name="test", created_at=utc_now()))
    run = make_run()
    store.save_run(run)
    for state in [RunState.VALIDATED, RunState.PLANNED, RunState.QUEUED, RunState.RUNNING]:
        run = store.transition_run(run.id, state, updated_at=utc_now())
    metric = TrainingMetric(
        run_id=run.id,
        step=1,
        epoch=1.0,
        training_loss=1.2,
        learning_rate=0.01,
        duration_seconds=0.1,
        trainable_parameters=10,
        total_parameters=100,
    )
    store.record_metric(metric)
    assert store.list_metrics(run.id) == [metric]
    with pytest.raises(ValueError, match="invalid run transition"):
        store.transition_run(run.id, RunState.VALIDATED, updated_at=utc_now())


def test_audit_redaction(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    event = AuditEvent(
        id="audit-1",
        event_type="test",
        subject_id="run",
        created_at=utc_now(),
        details={"authorization": "Bearer secret-token", "password": "password=secret"},
    )
    store.record_audit(event)
    persisted = store.list_audit("run")[0]
    assert "secret-token" not in json.dumps(persisted.details)
    assert "[REDACTED]" in redact_text("api_key=abc123")


def test_registry_requires_evaluation_evidence(tmp_path: Path) -> None:
    store = MetadataStore(tmp_path / "meta.sqlite3")
    registry = ModelRegistry(store)
    entry = registry.register("run", "adapter", "artifact")
    with pytest.raises(ValueError, match="evaluation evidence"):
        registry.transition(entry.id, RegistryStatus.EVALUATED)
    evaluation = EvaluationResult(
        suite_id="suite",
        suite_version="1",
        subject_id="run",
        metrics={"exact_match": 1.0},
        case_results={"case": 1.0},
        passed=True,
    )
    store.save_evaluation("eval", evaluation)
    entry = registry.transition(entry.id, RegistryStatus.EVALUATED, evaluation_id="eval")
    entry = registry.transition(entry.id, RegistryStatus.APPROVED)
    assert entry.status is RegistryStatus.APPROVED
    with pytest.raises(ValueError, match="invalid registry"):
        registry.transition(entry.id, RegistryStatus.REJECTED)


def test_evaluation_suite_formats_and_comparison() -> None:
    suite = EvaluationSuite(
        id="suite",
        version="1",
        cases=[
            EvaluationCase(id="exact", input="a", expected_output="yes"),
            EvaluationCase(
                id="json", input="b", expected_output="{}", metric="format_valid", threshold=1
            ),
        ],
    )
    outputs = {"a": "yes", "b": '{"ok": true}'}
    baseline = evaluate_suite(suite, "baseline", outputs.__getitem__)
    candidate = evaluate_suite(suite, "candidate", outputs.__getitem__)
    assert baseline.passed
    assert compare_results(baseline, candidate).passed
    incompatible = candidate.model_copy(update={"suite_version": "2"})
    with pytest.raises(ValueError, match="same evaluation"):
        compare_results(baseline, incompatible)


def test_path_confinement_artifact_names_and_publisher_boundary(tmp_path: Path) -> None:
    safe = confined_path(tmp_path, "runs/run-1")
    assert safe.is_relative_to(tmp_path)
    with pytest.raises(PathSecurityError):
        confined_path(tmp_path, "../escape")
    with pytest.raises(PathSecurityError):
        confined_path(tmp_path, Path("C:/absolute"))
    with pytest.raises(PathSecurityError):
        confined_path(tmp_path, r"..\escape")
    with pytest.raises(PathSecurityError):
        confined_path(tmp_path, r"\\server\share\artifact")
    with pytest.raises(PathSecurityError):
        confined_path(tmp_path, "C:drive-relative")
    assert validate_artifact_name("run-123") == "run-123"
    with pytest.raises(ValueError):
        validate_artifact_name("../run")
    publisher = HubPublisher()
    with pytest.raises(PermissionError):
        publisher.publish_model(tmp_path, "org/model")
    with pytest.raises(PermissionError):
        publisher.publish_dataset(tmp_path, "org/data")


def test_stable_digest_order_independent() -> None:
    assert stable_digest({"a": 1, "b": 2}) == stable_digest({"b": 2, "a": 1})
