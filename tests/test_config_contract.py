"""Executable CPU configuration at the HTTP, CLI, planner and backend boundaries."""

import json

import pytest
import torch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from tuneforge import cli, training
from tuneforge.api import create_app
from tuneforge.domain import RecordedTrainingConfig, TrainingConfig, TrainingMethod
from tuneforge.modeling import build_model_manifest, tiny_model_spec
from tuneforge.planning import build_training_plan
from tuneforge.runtime import detect_runtime
from tuneforge.storage import MetadataStore
from tuneforge.utils import stable_digest, write_canonical_json

UNSUPPORTED = [
    ("scheduler", "linear"),
    ("warmup_ratio", 0.5),
    ("save_strategy", "epoch"),
    ("evaluation_strategy", "steps"),
    ("evaluation_strategy", "epoch"),
    ("early_stopping_patience", 2),
    ("logging_steps", 3),
    ("optimizer", "sgd"),
    ("seed", 2**64),
    ("seed", -(2**63) - 1),
]


@pytest.fixture
def service(tmp_path, dataset_path):
    app = create_app(tmp_path / "api")
    client = TestClient(app)
    runtime = app.state.service
    path = runtime.dataset_root / "records.json"
    path.write_bytes(dataset_path.read_bytes())
    response = client.post("/datasets/register", json={"path": path.name})
    response.raise_for_status()
    return client, runtime, response.json()["id"]


def assert_no_training(store, root):
    with store.connect() as db:
        for table in ("runs", "metrics", "checkpoints", "checkpoint_ops", "evaluations"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert not (root / "runs").exists()


@pytest.mark.parametrize(("field", "value"), UNSUPPORTED)
def test_public_config_rejection(service, dataset_path, tmp_path, field, value, monkeypatch):
    client, runtime, dataset_id = service
    monkeypatch.setattr(training.AdamW, "step", lambda *_: pytest.fail("unexpected update"))
    settings = {field: value}
    for route in ("/training/plan", "/runs"):
        response = client.post(route, json={"dataset_id": dataset_id, "training": settings})
        assert response.status_code == 422
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "training", field]
        assert error["msg"]
    assert_no_training(runtime.store, runtime.artifact_root)

    config = tmp_path / "workflow.json"
    root = tmp_path / "cli"
    write_canonical_json(
        config,
        {"dataset_path": str(dataset_path), "artifact_root": str(root), "training": settings},
    )
    for command in ("plan", "train"):
        result = CliRunner().invoke(cli.app, [command, str(config)], terminal_width=160)
        assert result.exit_code == 2
        assert field in result.output
    assert not root.exists()


@pytest.fixture
def full_plan(dataset_manifest):
    model = build_model_manifest(tiny_model_spec(), TrainingMethod.FULL_FINE_TUNE)
    runtime = detect_runtime()
    plan = build_training_plan(
        dataset_manifest,
        model,
        TrainingConfig(),
        runtime,
        trainable_parameters=model.parameter_count,
    )
    return model, runtime, plan


@pytest.mark.parametrize(
    ("field", "value"),
    [
        *UNSUPPORTED,
        ("precision", "bf16"),
        ("per_device_batch_size", 2),
        ("gradient_accumulation", 2),
        ("max_sequence_length", 33),
        ("gradient_clipping", float("inf")),
    ],
)
def test_direct_callers_revalidate_unchecked_input(
    tmp_path, dataset_manifest, full_plan, field, value, monkeypatch
):
    model, runtime, plan = full_plan
    # Pydantic's model_copy(update=...) deliberately skips validation.
    invalid = TrainingConfig().model_copy(update={field: value})
    store = MetadataStore(tmp_path / "state.sqlite3")
    store.register_dataset(dataset_manifest.fingerprint, dataset_manifest)
    root = tmp_path / "artifacts"
    monkeypatch.setattr(training.AdamW, "step", lambda *_: pytest.fail("unexpected update"))
    reason = "batch size" if field in ("per_device_batch_size", "gradient_accumulation") else field
    with pytest.raises(ValueError, match=reason):
        build_training_plan(
            dataset_manifest, model, invalid, runtime, trainable_parameters=model.parameter_count
        )
    with pytest.raises(ValueError, match=reason):
        training.DeterministicTrainerBackend().execute(
            plan, model, invalid, dataset_manifest, root, store
        )
    assert_no_training(store, root)


@pytest.mark.parametrize(
    ("field", "strategy"),
    [
        ("checkpoint_strategy", {"save_strategy": "epoch", "frequency": 2, "keep_last_n": 2}),
        ("evaluation_plan", {"strategy": "steps"}),
    ],
)
def test_backend_rejects_misrepresented_plan(
    tmp_path, dataset_manifest, full_plan, field, strategy
):
    model, _, plan = full_plan
    plan = plan.model_copy(update={field: strategy})
    plan.plan_hash = stable_digest(plan.model_dump(exclude={"plan_hash"}, exclude_none=True))
    store = MetadataStore(tmp_path / "state.sqlite3")
    store.register_dataset(dataset_manifest.fingerprint, dataset_manifest)
    root = tmp_path / "artifacts"
    with pytest.raises(ValueError, match="strategy differs"):
        training.DeterministicTrainerBackend().execute(
            plan, model, TrainingConfig(), dataset_manifest, root, store
        )
    assert_no_training(store, root)


@pytest.mark.parametrize("explicit", [False, True], ids=["defaults", "configured"])
def test_four_real_updates_match_configuration(service, monkeypatch, explicit):
    client, runtime, dataset_id = service
    settings = (
        {
            "scheduler": "constant",
            "warmup_ratio": 0,
            "evaluation_strategy": "none",
            "save_strategy": "steps",
            "logging_steps": 1,
            "early_stopping_patience": None,
            "epochs": 2,
            "max_steps": 4,
            "learning_rate": 0.003,
            "weight_decay": 0.2,
            "gradient_clipping": 0.25,
            "seed": 19,
            "max_sequence_length": 8,
            "checkpoint_frequency": 3,
            "keep_last_n": 2,
        }
        if explicit
        else {}
    )
    config = TrainingConfig(**settings)
    request = (
        {"dataset_id": dataset_id, "training": settings} if explicit else {"dataset_id": dataset_id}
    )
    plan = client.post("/training/plan", json=request).json()["training_plan"]
    assert plan["estimated_steps"] == 4 and plan["effective_batch_size"] == 1
    assert plan["evaluation_plan"] == {"strategy": "none"}
    assert plan["checkpoint_strategy"]["frequency"] == config.checkpoint_frequency
    step = training.AdamW.step
    clip = torch.nn.utils.clip_grad_norm_
    batch = training._batch
    observed, clips, shapes = [], [], []

    def update(optimizer, *args, **kwargs):
        group = optimizer.param_groups[0]
        observed.append(
            (group["lr"], group["weight_decay"], group["params"][0].dtype, torch.initial_seed())
        )
        return step(optimizer, *args, **kwargs)

    def clip_gradients(parameters, max_norm, *args, **kwargs):
        clips.append(max_norm)
        return clip(parameters, max_norm, *args, **kwargs)

    def encode(record, max_length):
        tokens, labels = batch(record, max_length)
        shapes.append(tuple(tokens.shape))
        return tokens, labels

    monkeypatch.setattr(training.AdamW, "step", update)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip_gradients)
    monkeypatch.setattr(training, "_batch", encode)
    response = client.post("/runs", json=request)
    response.raise_for_status()
    result = response.json()
    run = result["run"]
    assert run["state"] == "completed" and result["adapter_reload_verified"]
    assert observed == [(config.learning_rate, config.weight_decay, torch.float32, config.seed)] * 4
    assert clips == [config.gradient_clipping] * 4
    assert len(shapes) == 4 and all(n == 1 and t <= config.max_sequence_length for n, t in shapes)
    if explicit:
        assert shapes == [(1, 8)] * 4
    metrics = runtime.store.list_metrics(run["id"])
    assert [(m.step, m.learning_rate) for m in metrics] == [
        (i, config.learning_rate) for i in range(1, 5)
    ]
    assert all(m.evaluation_loss is None for m in metrics)
    with runtime.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 0
    assert [c.step for c in runtime.store.list_checkpoints(run["id"])] == (
        [3, 4] if explicit else [2, 4]
    )
    saved = runtime.artifact_root / "runs" / run["id"] / "training-config.json"
    assert json.loads(saved.read_bytes())["evaluation_strategy"] == "none"
    assert stable_digest(config) == plan["training_config_hash"] == run["training_config_hash"]


@pytest.mark.parametrize("old_strategy", ["steps", "epoch", None])
def test_resume_rejects_unsupported_or_absent_recorded_strategy(service, old_strategy, monkeypatch):
    client, runtime, dataset_id = service
    response = client.post("/runs", json={"dataset_id": dataset_id})
    response.raise_for_status()
    run_id = response.json()["run"]["id"]
    directory = runtime.artifact_root / "runs" / run_id
    path = directory / "training-config.json"
    config = json.loads(path.read_bytes())
    if old_strategy is None:
        del config["evaluation_strategy"]
    else:
        config["evaluation_strategy"] = old_strategy
    write_canonical_json(path, config)
    before = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    monkeypatch.setattr(training.AdamW, "step", lambda *_: pytest.fail("unexpected update"))
    result = CliRunner().invoke(
        cli.app, ["resume", run_id, "--root", str(runtime.root)], terminal_width=160
    )
    assert result.exit_code == 2 and "evaluation_strategy" in result.output
    assert {p: p.read_bytes() for p in directory.rglob("*") if p.is_file()} == before
    assert list(directory.parent.iterdir()) == [directory]
    with runtime.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1


def test_overflowing_json_number_has_a_field_level_422(service):
    client, runtime, dataset_id = service
    body = '{"dataset_id":"' + dataset_id + '","training":{"gradient_clipping":1e309}}'
    for route in ("/training/plan", "/runs"):
        response = client.post(route, content=body, headers={"content-type": "application/json"})
        assert response.status_code == 422
        [error] = response.json()["detail"]
        assert error["loc"] == ["body", "training", "gradient_clipping"]
        assert "finite" in error["msg"]
    assert_no_training(runtime.store, runtime.artifact_root)


@pytest.mark.parametrize("epochs", [1, 2])
def test_epoch_budget_limits_actual_updates(service, epochs, monkeypatch):
    client, runtime, _ = service
    dataset = client.post(
        "/datasets/register",
        json={
            "path": "records.json",
            "split": {
                "algorithm": "seeded_random",
                "train_ratio": 0.2,
                "validation_ratio": 0.4,
                "test_ratio": 0.4,
            },
        },
    ).json()
    train_ids = dataset["manifest"]["split"]["record_ids"]["train"]
    assert len(train_ids) == 2
    settings = {"dataset_id": dataset["id"], "training": {"epochs": epochs, "max_steps": 5}}
    plan = client.post("/training/plan", json=settings).json()["training_plan"]
    batch = training._batch
    consumed = []

    def observe(record, length):
        consumed.append(record.id)
        return batch(record, length)

    monkeypatch.setattr(training, "_batch", observe)
    run = client.post("/runs", json=settings).json()["run"]
    assert plan["estimated_steps"] == run["last_completed_step"] == epochs * len(train_ids)
    assert consumed == train_ids * epochs
    assert len(runtime.store.list_metrics(run["id"])) == len(consumed)


@pytest.mark.parametrize("ambient", ["float64", "autocast"])
def test_precision_rejection_respects_host_torch_settings(
    tmp_path, full_plan, dataset_manifest, ambient
):
    model, runtime, plan = full_plan
    config = TrainingConfig()
    store = MetadataStore(tmp_path / "state.sqlite3")
    root = tmp_path / "artifacts"
    previous = torch.get_default_dtype()
    try:
        if ambient == "float64":
            torch.set_default_dtype(torch.float64)
        with torch.autocast("cpu", enabled=ambient == "autocast"):
            with pytest.raises(ValueError, match="FP32"):
                build_training_plan(
                    dataset_manifest,
                    model,
                    config,
                    runtime,
                    trainable_parameters=model.parameter_count,
                )
            with pytest.raises(ValueError, match="FP32"):
                training.DeterministicTrainerBackend().execute(
                    plan, model, config, dataset_manifest, root, store
                )
    finally:
        torch.set_default_dtype(previous)
    assert_no_training(store, root)


def test_archived_config_keeps_its_original_digest():
    recorded = TrainingConfig().model_dump(exclude_none=True)
    recorded.update(
        scheduler="linear",
        warmup_ratio=0.5,
        save_strategy="epoch",
        evaluation_strategy="steps",
        logging_steps=3,
        early_stopping_patience=2,
    )
    archived = RecordedTrainingConfig.model_validate(recorded)
    assert stable_digest(archived) == stable_digest(recorded)
    with pytest.raises(ValueError, match="evaluation_strategy"):
        TrainingConfig.model_validate(recorded)
