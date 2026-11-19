"""Observe real optimizer updates and model evaluation at TuneForge's public boundaries."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tuneforge import cli, demo, training
from tuneforge.api import create_app
from tuneforge.dataset import build_dataset_manifest
from tuneforge.domain import SplitConfig
from tuneforge.storage import MetadataStore
from tuneforge.utils import write_canonical_json


@pytest.fixture
def workflow(tmp_path):
    root = tmp_path / "service"
    source = root / "datasets" / "records.json"
    write_canonical_json(source, [{"prompt": f"p{i}", "completion": f"a{i}"} for i in range(10)])
    config = {
        "dataset_path": str(source),
        "artifact_root": str(tmp_path / "cli"),
        "split": {
            "train_ratio": 0.6,
            "validation_ratio": 0.2,
            "test_ratio": 0.2,
            "seed": 42,
            "algorithm": "seeded_random",
        },
        "training": {"max_steps": 4, "epochs": 2, "checkpoint_frequency": 1, "keep_last_n": 2},
    }
    path = tmp_path / "workflow.json"
    write_canonical_json(path, config)
    return path, config


@pytest.fixture
def updates(monkeypatch):
    """Associate the input ID with a successful real AdamW step; retain no sample text."""
    batch = training._batch
    step = training.AdamW.step
    pending = []
    consumed = []

    def encode(record, max_length):
        pending.append(record.id)
        return batch(record, max_length)

    def update(optimizer, *args, **kwargs):
        result = step(optimizer, *args, **kwargs)
        consumed.append(pending.pop(0))
        return result

    monkeypatch.setattr(training, "_batch", encode)
    monkeypatch.setattr(training.AdamW, "step", update)
    return consumed


def service_for(config):
    source = Path(config["dataset_path"])
    app = create_app(source.parent.parent)
    client = TestClient(app)
    response = client.post(
        "/datasets/register", json={"path": source.name, "split": config["split"]}
    )
    response.raise_for_status()
    return (
        client,
        app.state.service,
        {"dataset_id": response.json()["id"], "training": config["training"]},
    )


def invoke(*args):
    result = CliRunner().invoke(cli.app, [str(a) for a in args])
    assert result.exit_code == 0, (result.output, result.exception)
    return json.loads(result.stdout)


def observe_evaluation(monkeypatch, module):
    evaluate = module.evaluate_language_model_loss
    consumed = []

    def record(model, records, **kwargs):
        consumed.append([record.id for record in records])
        return evaluate(model, records, **kwargs)

    monkeypatch.setattr(module, "evaluate_language_model_loss", record)
    return consumed


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_updates_and_evaluation_follow_manifest(workflow, entry, updates, monkeypatch):
    path, config = workflow
    manifest, _ = build_dataset_manifest(
        Path(config["dataset_path"]), SplitConfig(**config["split"])
    )
    held = observe_evaluation(monkeypatch, cli)
    if entry == "api":
        client, service, req = service_for(config)
        plan = client.post("/training/plan", json=req).json()["training_plan"]
        res = client.post("/runs", json=req)
        res.raise_for_status()
        run = res.json()["run"]
        root = service.root
    else:
        plan = invoke("plan", path)["training_plan"]
        run = invoke("train", path)["run"]
        root = Path(config["artifact_root"])
    invoke("evaluate", run["id"], "--root", root)
    assert plan["effective_batch_size"] == 1
    assert len(updates) == run["last_completed_step"] == plan["estimated_steps"] == 4
    assert updates == manifest.split.record_ids["train"][:4]
    assert held == [manifest.split.record_ids["test"]]
    assert set(updates).isdisjoint(held[0] + manifest.split.record_ids["validation"])
    assert run["dataset_fingerprint"] == manifest.fingerprint


def test_demo_uses_held_out_records_and_binds_evidence(tmp_path, updates, monkeypatch):
    observed = observe_evaluation(monkeypatch, demo)
    result = invoke("demo", "--root", tmp_path)
    run_dir = tmp_path / "runs" / result["run_id"]
    manifest = json.loads((run_dir / "dataset-manifest.json").read_text())
    assert updates == manifest["split"]["record_ids"]["train"][:3]
    assert observed == [manifest["split"]["record_ids"]["test"]] * 2
    evidence = json.loads((run_dir / "evidence" / "evidence.json").read_text())
    assert (
        result["dataset_fingerprint"] == manifest["fingerprint"] == evidence["dataset_fingerprint"]
    )


def test_changed_dataset_rejected_before_new_artifacts(workflow, updates):
    _, config = workflow
    client, service, request = service_for(config)
    source = Path(config["dataset_path"])
    source.write_bytes(source.read_bytes() + b" ")
    result = client.post("/runs", json=request)
    assert result.status_code == 409 and "fingerprint" in result.json()["detail"]
    assert updates == []
    assert not list(service.artifact_root.iterdir())
    with service.store.connect() as db:
        for table in ("runs", "metrics", "checkpoints", "evaluations"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_snapshot_is_not_reopened_by_optimizer(workflow, updates, monkeypatch):
    _, config = workflow
    client, _, req = service_for(config)
    source = Path(config["dataset_path"])
    manifest, _ = build_dataset_manifest(source, SplitConfig(**config["split"]))
    original = training._create_run

    def start(*args, **kwargs):
        write_canonical_json(source, [{"prompt": "new", "completion": "row"}])
        return original(*args, **kwargs)

    monkeypatch.setattr(training, "_create_run", start)
    response = client.post("/runs", json=req)
    response.raise_for_status()
    assert updates == manifest.split.record_ids["train"][:4]
    assert response.json()["run"]["dataset_fingerprint"] == manifest.fingerprint


@pytest.mark.parametrize("key", ["per_device_batch_size", "gradient_accumulation"])
def test_unsupported_batch_rejected_at_plan_and_run(workflow, updates, key):
    _, config = workflow
    client, _, req = service_for(config)
    req["training"][key] = 2
    for route in ["/training/plan", "/runs"]:
        response = client.post(route, json=req)
        assert response.status_code == 422 and "batch size" in response.json()["detail"]
    assert updates == []


def test_invalid_lora_targets_are_client_errors(workflow):
    _, config = workflow
    client, _, req = service_for(config)
    for route in ["/training/plan", "/runs"]:
        res = client.post(route, json={**req, "lora": {"target_modules": ["unavailable"]}})
        assert res.status_code == 422 and "target modules" in res.json()["detail"]


def test_active_http_run_can_be_cancelled_and_resumed_by_cli(workflow, updates, monkeypatch):
    _, config = workflow
    client, service, req = service_for(config)
    reached = threading.Event()
    proceed = threading.Event()
    step = training.AdamW.step

    def pause_after_update(optimizer, *args, **kwargs):
        value = step(optimizer, *args, **kwargs)
        reached.set()
        assert proceed.wait(15)
        return value

    monkeypatch.setattr(training.AdamW, "step", pause_after_update)
    with ThreadPoolExecutor(max_workers=1) as pool:
        task = pool.submit(client.post, "/runs", json=req)
        try:
            assert reached.wait(15)
            [run_id] = client.get("/runs/active").json()
            assert client.post("/training/plan", json=req).status_code == 409
            assert client.post(f"/runs/{run_id}/cancel").status_code == 200
        finally:
            proceed.set()
        result = task.result(timeout=15)
    result.raise_for_status()
    assert result.json()["run"]["state"] == "cancelled"
    assert service.store.list_checkpoints(run_id)[0].step == 1
    assert client.get("/runs/active").json() == []
    assert client.post(f"/runs/{run_id}/cancel").status_code == 409
    child = invoke("resume", run_id, "--root", service.root)["run"]
    manifest = service.store.get_dataset(req["dataset_id"])
    assert updates == manifest.split.record_ids["train"][:4]
    assert child["last_completed_step"] == 4
    assert [m.step for m in service.store.list_metrics(child["id"])] == [2, 3, 4]
    assert child["environment"]["resumed_from_run_id"] == run_id
    assert [c.step for c in service.store.list_checkpoints(child["id"])] == [3, 4]
    exhausted = CliRunner().invoke(cli.app, ["resume", child["id"], "--root", str(service.root)])
    assert "exhausted" in str(exhausted.exception)
    assert len(updates) == 4


@pytest.mark.parametrize("operation", ["resume", "evaluate"])
def test_changed_source_stops_recovery_and_evaluation(workflow, updates, operation):
    path, config = workflow
    run = invoke("train", path)["run"]
    root = Path(config["artifact_root"])
    before = {
        p.relative_to(root): p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file()
    }
    source = Path(config["dataset_path"])
    source.write_bytes(source.read_bytes() + b" ")
    response = CliRunner().invoke(cli.app, [operation, run["id"], "--root", str(root)])
    assert response.exit_code != 0 and "fingerprint" in str(response.exception)
    after = {p.relative_to(root): p.read_bytes() for p in (root / "runs").rglob("*") if p.is_file()}
    assert after == before and len(updates) == 4
    with MetadataStore(root / "tuneforge.sqlite3").connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0


@pytest.mark.parametrize("filename", ["metrics.json", "run.json"])
def test_final_artifact_failure_recovers_without_extra_update(
    workflow, updates, monkeypatch, filename
):
    path, config = workflow
    original = training.write_canonical_json

    def fail(path, payload):
        if path.name == filename:
            raise OSError("simulated final write failure")
        original(path, payload)

    with monkeypatch.context() as patch:
        patch.setattr(training, "write_canonical_json", fail)
        result = CliRunner().invoke(cli.app, ["train", str(path)])
    assert isinstance(result.exception, OSError)
    root = Path(config["artifact_root"])
    store = MetadataStore(root / "tuneforge.sqlite3")
    with store.connect() as db:
        [(parent, state)] = db.execute("SELECT id,state FROM runs").fetchall()
    assert state == "failed" and not (root / "runs" / parent / "run.json").exists()
    assert len(updates) == 4
    child = invoke("resume", parent, "--root", root)["run"]
    assert child["last_completed_step"] == 4 and child["state"] == "completed"
    assert len(updates) == 4 and store.list_metrics(child["id"]) == []
    assert store.list_checkpoints(child["id"])[0].step == 4


def test_empty_required_splits_are_rejected(workflow, updates):
    path, config = workflow
    # A seeded-random split assigns the only row at score 0.5.
    write_canonical_json(Path(config["dataset_path"]), [{"prompt": "p", "completion": "a"}])
    config["split"].update(train_ratio=0.4, validation_ratio=0.3, test_ratio=0.3)
    write_canonical_json(path, config)
    result = CliRunner().invoke(cli.app, ["train", str(path)])
    assert "train split is empty" in str(result.exception)
    assert updates == []
    config["split"].update(train_ratio=0.8, validation_ratio=0.2, test_ratio=0)
    write_canonical_json(path, config)
    run = invoke("train", path)["run"]
    result = CliRunner().invoke(cli.app, ["evaluate", run["id"], "--root", config["artifact_root"]])
    assert "test split is empty" in str(result.exception)


@pytest.mark.parametrize("operation", ["evaluate", "resume"])
def test_corrupted_checkpoint_fails_closed_at_public_commands(workflow, updates, operation):
    path, config = workflow
    run = invoke("train", path)["run"]
    root = Path(config["artifact_root"])
    state = root / "runs" / run["id"] / "checkpoints" / "step-000004" / "training-state.json"
    state.write_bytes(state.read_bytes() + b" ")
    result = CliRunner().invoke(cli.app, [operation, run["id"], "--root", str(root)])
    assert "checkpoint integrity" in str(result.exception)
    assert len(updates) == 4 and len(list((root / "runs").iterdir())) == 1


@pytest.mark.parametrize("name", ["training-config.json", "model-manifest.json"])
def test_recovery_rejects_incompatible_inputs(workflow, name, updates):
    path, config = workflow
    run = invoke("train", path)["run"]
    root = Path(config["artifact_root"])
    source = root / "runs" / run["id"] / name
    value = json.loads(source.read_bytes())
    if name == "training-config.json":
        value["learning_rate"] = 0.01
    else:
        value["revision"] = "different"
    write_canonical_json(source, value)
    r = CliRunner().invoke(cli.app, ["resume", run["id"], "--root", str(root)])
    assert r.exit_code != 0 and isinstance(r.exception, ValueError)
    assert "identity" in str(r.exception) or "configuration" in str(r.exception)
    assert len(updates) == 4 and len(list((root / "runs").iterdir())) == 1


def test_public_cli_recovers_after_worker_process_exit(workflow, updates):
    import os
    import subprocess
    import sys

    path, config = workflow
    # Exit only the fixture subprocess, after its first durable checkpoint.
    program = """
import os, sys
from tuneforge import training
from tuneforge.cli import app
original = training.apply_retention
def stop_after_checkpoint(*args, **kwargs):
    original(*args, **kwargs)
    os._exit(41)
training.apply_retention = stop_after_checkpoint
sys.argv = ["tuneforge", "train", sys.argv[1]]
app()
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(path)],
        check=False,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
        timeout=90,
    )
    assert result.returncode == 41
    root = Path(config["artifact_root"])
    store = MetadataStore(root / "tuneforge.sqlite3")
    with store.connect() as db:
        [(parent, state)] = db.execute("SELECT id,state FROM runs").fetchall()
    assert state == "running"
    assert store.list_checkpoints(parent)[0].step == 1
    child = invoke("resume", parent, "--root", root)["run"]
    manifest = store.get_dataset(child["dataset_fingerprint"])
    assert updates == manifest.split.record_ids["train"][1:4]
    assert [m.step for m in store.list_metrics(child["id"])] == [2, 3, 4]
