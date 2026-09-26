"""Evidence-gated local model registry."""

from __future__ import annotations

import uuid
from typing import Literal

from tuneforge.domain import ModelRegistryEntry, RegistryStatus
from tuneforge.storage import MetadataStore
from tuneforge.utils import utc_now

ALLOWED_REGISTRY_TRANSITIONS: dict[RegistryStatus, set[RegistryStatus]] = {
    RegistryStatus.CANDIDATE: {RegistryStatus.EVALUATED, RegistryStatus.REJECTED},
    RegistryStatus.EVALUATED: {RegistryStatus.APPROVED, RegistryStatus.REJECTED},
    RegistryStatus.APPROVED: {RegistryStatus.ARCHIVED},
    RegistryStatus.REJECTED: {RegistryStatus.ARCHIVED},
    RegistryStatus.ARCHIVED: set(),
}


class ModelRegistry:
    def __init__(self, store: MetadataStore) -> None:
        self.store = store

    def register(
        self,
        run_id: str,
        artifact_kind: Literal["adapter", "full", "reference"],
        artifact_path: str,
    ) -> ModelRegistryEntry:
        now = utc_now()
        entry = ModelRegistryEntry(
            id=f"model-{uuid.uuid4().hex}",
            run_id=run_id,
            artifact_kind=artifact_kind,
            artifact_path=artifact_path,
            created_at=now,
            updated_at=now,
        )
        self.store.save_registry_entry(entry)
        return entry

    def transition(
        self,
        identifier: str,
        target: RegistryStatus,
        *,
        evaluation_id: str | None = None,
    ) -> ModelRegistryEntry:
        entry = self.store.get_registry_entry(identifier)
        if entry is None:
            raise KeyError(f"unknown registry entry: {identifier}")
        if target not in ALLOWED_REGISTRY_TRANSITIONS[entry.status]:
            raise ValueError(f"invalid registry transition: {entry.status.value} -> {target.value}")
        evidence_id = evaluation_id or entry.evaluation_id
        if target in {RegistryStatus.EVALUATED, RegistryStatus.APPROVED} and not evidence_id:
            raise ValueError("evaluation evidence is required for this registry transition")
        if evidence_id and self.store.get_evaluation(evidence_id) is None:
            raise ValueError("evaluation evidence does not exist")
        entry.status = target
        entry.evaluation_id = evidence_id
        entry.updated_at = utc_now()
        self.store.save_registry_entry(entry)
        return entry

    def list(self) -> list[ModelRegistryEntry]:
        return self.store.list_registry()


__all__ = ["ALLOWED_REGISTRY_TRANSITIONS", "ModelRegistry"]
