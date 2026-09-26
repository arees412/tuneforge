from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tuneforge.api import create_app
from tuneforge.cli import app


def test_api_health_dataset_plan_and_run(tmp_path: Path, dataset_path: Path) -> None:
    service_root = tmp_path / "service"
    dataset_root = service_root / "datasets"
    dataset_root.mkdir(parents=True)
    shutil.copyfile(dataset_path, dataset_root / "fixture.json")
    client = TestClient(create_app(service_root))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/runtime/capabilities").status_code == 200
    validation = client.post("/datasets/validate", json={"path": "fixture.json"})
    assert validation.status_code == 200
    assert validation.json()["valid"]
    registration = client.post(
        "/datasets/register",
        json={"path": "fixture.json", "license": {"identifier": "unknown"}},
    )
    assert registration.status_code == 200
    identifier = registration.json()["id"]
    assert client.get(f"/datasets/{identifier}").status_code == 200
    plan = client.post(
        "/training/plan",
        json={
            "dataset_id": identifier,
            "method": "lora",
            "training": {"max_steps": 1, "max_sequence_length": 16, "checkpoint_frequency": 1},
            "lora": {"r": 2, "alpha": 4},
        },
    )
    assert plan.status_code == 200
    assert plan.json()["training_plan"]["method"] == "lora"
    run = client.post(
        "/runs",
        json={
            "dataset_id": identifier,
            "method": "lora",
            "training": {"max_steps": 1, "max_sequence_length": 16, "checkpoint_frequency": 1},
            "lora": {"r": 2, "alpha": 4},
        },
    )
    assert run.status_code == 200
    run_id = run.json()["run"]["id"]
    assert client.get(f"/runs/{run_id}").json()["state"] == "completed"
    assert len(client.get(f"/runs/{run_id}/metrics").json()) == 1
    assert len(client.get(f"/runs/{run_id}/checkpoints").json()) == 1


def test_api_evaluation_registry_errors_and_path_escape(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "service"))
    response = client.post(
        "/evaluations",
        json={
            "subject_id": "candidate",
            "suite": {
                "id": "suite",
                "version": "1",
                "cases": [{"id": "case", "input": "hello", "expected_output": "world"}],
            },
            "predictions": {"case": "world"},
        },
    )
    assert response.status_code == 200
    evaluation_id = response.json()["id"]
    assert client.get(f"/evaluations/{evaluation_id}").json()["passed"]
    assert client.get("/registry").json() == []
    assert client.post("/datasets/validate", json={"path": "../escape.json"}).status_code == 400
    assert client.post("/runs/missing/cancel").status_code == 409


def test_cli_help_validate_split_and_capabilities(dataset_path: Path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    validation = runner.invoke(app, ["dataset", "validate", str(dataset_path)])
    assert validation.exit_code == 0
    assert '"valid":true' in validation.output
    assert runner.invoke(app, ["dataset", "split", str(dataset_path)]).exit_code == 0
    assert runner.invoke(app, ["capabilities"]).exit_code == 0
