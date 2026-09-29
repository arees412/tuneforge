from __future__ import annotations

from pathlib import Path

import pytest

from tuneforge.dataset import build_dataset_manifest
from tuneforge.domain import DatasetManifest, SplitConfig
from tuneforge.utils import write_canonical_json


@pytest.fixture
def instruction_records() -> list[dict[str, object]]:
    return [
        {
            "id": f"row-{index}",
            "instruction": f"Question {index}",
            "input": "fixture",
            "output": f"Answer {index}",
            "metadata": {"index": index},
        }
        for index in range(12)
    ]


@pytest.fixture
def dataset_path(tmp_path: Path, instruction_records: list[dict[str, object]]) -> Path:
    path = tmp_path / "dataset.json"
    write_canonical_json(path, instruction_records)
    return path


@pytest.fixture
def dataset_manifest(dataset_path: Path) -> DatasetManifest:
    manifest, _ = build_dataset_manifest(dataset_path, SplitConfig(seed=11))
    return manifest
