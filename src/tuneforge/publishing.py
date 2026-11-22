"""Explicitly disabled-by-default external publication boundary."""

from __future__ import annotations

from pathlib import Path


class HubPublisher:
    def __init__(self, *, enabled: bool = False) -> None:
        self.enabled = enabled

    def publish_model(self, artifact: Path, repository_id: str, *, approved: bool = False) -> None:
        if not self.enabled or not approved:
            raise PermissionError("Hub publication is disabled without explicit operator approval")
        raise NotImplementedError("Hub publication is intentionally unsupported in TuneForge v0.1")

    def publish_dataset(self, dataset: Path, repository_id: str, *, approved: bool = False) -> None:
        if not self.enabled or not approved:
            raise PermissionError(
                "dataset publication is disabled without explicit operator approval"
            )
        raise NotImplementedError(
            "dataset publication is intentionally unsupported in TuneForge v0.1"
        )


__all__ = ["HubPublisher"]
