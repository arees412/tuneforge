"""SQLite metadata persistence with explicit run state transitions."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from tuneforge.domain import (
    AuditEvent,
    CheckpointManifest,
    DatasetManifest,
    EvaluationResult,
    Experiment,
    ModelRegistryEntry,
    RunState,
    TrainingMetric,
    TrainingRun,
)
from tuneforge.security import redact
from tuneforge.utils import canonical_json

ModelT = TypeVar("ModelT", bound=BaseModel)


ALLOWED_TRANSITIONS: dict[RunState, set[RunState]] = {
    RunState.CREATED: {RunState.VALIDATED, RunState.CANCELLED, RunState.FAILED},
    RunState.VALIDATED: {RunState.PLANNED, RunState.CANCELLED, RunState.FAILED},
    RunState.PLANNED: {RunState.QUEUED, RunState.CANCELLED, RunState.FAILED},
    RunState.QUEUED: {RunState.RUNNING, RunState.CANCELLED, RunState.FAILED},
    RunState.RUNNING: {RunState.EVALUATING, RunState.CANCELLED, RunState.FAILED},
    RunState.EVALUATING: {RunState.COMPLETED, RunState.CANCELLED, RunState.FAILED},
    RunState.COMPLETED: set(),
    RunState.FAILED: set(),
    RunState.CANCELLED: set(),
}


class MetadataStore:
    """Small deterministic SQLite repository; each row stores a validated JSON document."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        statements = (
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)",
            "CREATE TABLE IF NOT EXISTS datasets (id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS experiments (id TEXT PRIMARY KEY, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS runs "
            "(id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS metrics "
            "(run_id TEXT NOT NULL, step INTEGER NOT NULL, payload TEXT NOT NULL, "
            "PRIMARY KEY (run_id, step))",
            "CREATE TABLE IF NOT EXISTS checkpoints "
            "(id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step INTEGER NOT NULL, "
            "payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS evaluations "
            "(id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS registry "
            "(id TEXT PRIMARY KEY, status TEXT NOT NULL, payload TEXT NOT NULL)",
            "CREATE TABLE IF NOT EXISTS audit_events "
            "(id TEXT PRIMARY KEY, subject_id TEXT NOT NULL, created_at TEXT NOT NULL, "
            "payload TEXT NOT NULL)",
        )
        with self.connect() as connection:
            for statement in statements:
                connection.execute(statement)
            connection.execute("INSERT OR IGNORE INTO schema_migrations(version) VALUES (1)")

    @staticmethod
    def _decode(payload: str, model: type[ModelT]) -> ModelT:
        return model.model_validate_json(payload)

    def register_dataset(self, identifier: str, manifest: DatasetManifest) -> None:
        self._upsert("datasets", identifier, manifest)

    def get_dataset(self, identifier: str) -> DatasetManifest | None:
        return self._get("datasets", identifier, DatasetManifest)

    def save_experiment(self, experiment: Experiment) -> None:
        self._upsert("experiments", experiment.id, experiment)

    def save_run(self, run: TrainingRun) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO runs(id, state, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state, payload=excluded.payload",
                (run.id, run.state.value, canonical_json(run)),
            )

    def get_run(self, identifier: str) -> TrainingRun | None:
        return self._get("runs", identifier, TrainingRun)

    def transition_run(
        self,
        identifier: str,
        target: RunState,
        *,
        updated_at: Any,
        failure_category: str | None = None,
        last_completed_step: int | None = None,
    ) -> TrainingRun:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload FROM runs WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {identifier}")
            run = TrainingRun.model_validate_json(row["payload"])
            if target not in ALLOWED_TRANSITIONS[run.state]:
                raise ValueError(f"invalid run transition: {run.state.value} -> {target.value}")
            run.state = target
            run.updated_at = updated_at
            run.failure_category = failure_category
            if last_completed_step is not None:
                run.last_completed_step = last_completed_step
            connection.execute(
                "UPDATE runs SET state = ?, payload = ? WHERE id = ?",
                (target.value, canonical_json(run), identifier),
            )
            return run

    def record_metric(self, metric: TrainingMetric) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO metrics(run_id, step, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(run_id, step) DO UPDATE SET payload=excluded.payload",
                (metric.run_id, metric.step, canonical_json(metric)),
            )

    def list_metrics(self, run_id: str) -> list[TrainingMetric]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM metrics WHERE run_id = ? ORDER BY step", (run_id,)
            ).fetchall()
        return [self._decode(row["payload"], TrainingMetric) for row in rows]

    def record_checkpoint(self, checkpoint: CheckpointManifest) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO checkpoints(id, run_id, step, payload) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (
                    checkpoint.checkpoint_id,
                    checkpoint.run_id,
                    checkpoint.step,
                    canonical_json(checkpoint),
                ),
            )

    def list_checkpoints(self, run_id: str) -> list[CheckpointManifest]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM checkpoints WHERE run_id = ? ORDER BY step", (run_id,)
            ).fetchall()
        return [self._decode(row["payload"], CheckpointManifest) for row in rows]

    def delete_checkpoint_metadata(self, checkpoint_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM checkpoints WHERE id = ?", (checkpoint_id,))

    def save_evaluation(self, identifier: str, result: EvaluationResult) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO evaluations(id, subject_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET subject_id=excluded.subject_id, "
                "payload=excluded.payload",
                (identifier, result.subject_id, canonical_json(result)),
            )

    def get_evaluation(self, identifier: str) -> EvaluationResult | None:
        return self._get("evaluations", identifier, EvaluationResult)

    def save_registry_entry(self, entry: ModelRegistryEntry) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO registry(id, status, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET status=excluded.status, payload=excluded.payload",
                (entry.id, entry.status.value, canonical_json(entry)),
            )

    def get_registry_entry(self, identifier: str) -> ModelRegistryEntry | None:
        return self._get("registry", identifier, ModelRegistryEntry)

    def list_registry(self) -> list[ModelRegistryEntry]:
        with self.connect() as connection:
            rows = connection.execute("SELECT payload FROM registry ORDER BY id").fetchall()
        return [self._decode(row["payload"], ModelRegistryEntry) for row in rows]

    def record_audit(self, event: AuditEvent) -> None:
        safe_event = event.model_copy(update={"details": redact(event.details)})
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO audit_events(id, subject_id, created_at, payload) VALUES (?, ?, ?, ?)",
                (
                    event.id,
                    event.subject_id,
                    event.created_at.isoformat(),
                    canonical_json(safe_event),
                ),
            )

    def list_audit(self, subject_id: str) -> list[AuditEvent]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM audit_events WHERE subject_id = ? ORDER BY created_at, id",
                (subject_id,),
            ).fetchall()
        return [self._decode(row["payload"], AuditEvent) for row in rows]

    def _upsert(self, table: str, identifier: str, value: BaseModel) -> None:
        if table not in {"datasets", "experiments"}:
            raise ValueError("unsupported table")
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO {table}(id, payload) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                (identifier, canonical_json(value)),
            )

    def _get(self, table: str, identifier: str, model: type[ModelT]) -> ModelT | None:
        if table not in {"datasets", "experiments", "runs", "evaluations", "registry"}:
            raise ValueError("unsupported table")
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT payload FROM {table} WHERE id = ?",
                (identifier,),
            ).fetchone()
        return None if row is None else self._decode(row["payload"], model)


__all__ = ["ALLOWED_TRANSITIONS", "MetadataStore"]
