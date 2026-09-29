"""TuneForge command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, Field

from tuneforge.checkpoints import checkpoint_directory
from tuneforge.dataset import (
    WhitespaceTokenizer,
    analyze_tokenization,
    build_dataset_manifest,
    build_quality_report,
    validate_dataset,
)
from tuneforge.demo import run_demo
from tuneforge.domain import (
    DatasetManifest,
    LicenseMetadata,
    LoRAConfig,
    ModelManifest,
    QLoRAConfig,
    SplitConfig,
    TrainingConfig,
    TrainingMethod,
    TrainingPlan,
)
from tuneforge.evaluation import compare_results, evaluate_language_model_loss
from tuneforge.modeling import (
    apply_lora,
    build_model_manifest,
    create_tiny_model,
    parameter_report,
    tiny_model_spec,
)
from tuneforge.planning import build_training_plan
from tuneforge.runtime import detect_runtime
from tuneforge.scenarios import run_scenarios
from tuneforge.storage import MetadataStore
from tuneforge.training import DeterministicTrainerBackend, load_checkpoint_model
from tuneforge.utils import canonical_json

app = typer.Typer(
    help="Governed, reproducible local model-training workflows.", no_args_is_help=True
)
dataset_app = typer.Typer(help="Validate, inspect, and split controlled local datasets.")
registry_app = typer.Typer(help="Inspect the local evidence-gated model registry.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(registry_app, name="registry")


class WorkflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: Path
    artifact_root: Path = Path(".tuneforge")
    method: TrainingMethod = TrainingMethod.LORA
    split: SplitConfig = Field(default_factory=SplitConfig)
    dataset_license: LicenseMetadata = Field(default_factory=LicenseMetadata)
    provenance: dict[str, str] = Field(default_factory=dict)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


def _echo(value: Any) -> None:
    typer.echo(canonical_json(value))


def _load_workflow(path: Path) -> WorkflowConfig:
    try:
        return WorkflowConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc), param_hint="config") from exc


def _plan_workflow(
    config: WorkflowConfig,
) -> tuple[DatasetManifest, list[Any], ModelManifest, TrainingPlan]:
    dataset, records = build_dataset_manifest(
        config.dataset_path,
        config.split,
        license_metadata=config.dataset_license,
        provenance=config.provenance,
    )
    base = create_tiny_model(seed=config.training.seed)
    adapter = (
        config.lora
        if config.method is TrainingMethod.LORA
        else config.qlora
        if config.method is TrainingMethod.QLORA
        else None
    )
    model = build_model_manifest(tiny_model_spec(base), config.method, adapter)
    planned = apply_lora(base, config.lora) if config.method is TrainingMethod.LORA else base
    counts = parameter_report(planned)
    plan = build_training_plan(
        dataset,
        model,
        config.training,
        detect_runtime(),
        trainable_parameters=int(counts["trainable_parameters"]),
        adapter=adapter,
    )
    return dataset, records, model, plan


@dataset_app.command("validate")
def dataset_validate(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    result = validate_dataset(path)
    _echo(result)
    if not result.valid:
        raise typer.Exit(code=1)


@dataset_app.command("inspect")
def dataset_inspect(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_length: Annotated[int, typer.Option(min=4, max=131_072)] = 2048,
) -> None:
    result = validate_dataset(path)
    if not result.valid:
        _echo(result)
        raise typer.Exit(code=1)
    _echo(
        {
            "validation": result,
            "quality": build_quality_report(result.records),
            "tokenization": analyze_tokenization(
                result.records, WhitespaceTokenizer(), max_length=max_length
            ),
        }
    )


@dataset_app.command("split")
def dataset_split(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    seed: Annotated[int, typer.Option()] = 42,
) -> None:
    manifest, _ = build_dataset_manifest(path, SplitConfig(seed=seed))
    _echo(manifest.split)


@app.command("plan")
def plan(config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    workflow = _load_workflow(config)
    dataset, _, model, training_plan = _plan_workflow(workflow)
    _echo({"dataset_manifest": dataset, "model_manifest": model, "training_plan": training_plan})


@app.command("train")
def train(config: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]) -> None:
    workflow = _load_workflow(config)
    dataset, records, model, training_plan = _plan_workflow(workflow)
    store = MetadataStore(workflow.artifact_root / "tuneforge.sqlite3")
    store.register_dataset(dataset.fingerprint, dataset)
    outcome = DeterministicTrainerBackend().execute(
        training_plan,
        model,
        workflow.training,
        records,
        workflow.artifact_root,
        store,
    )
    _echo(
        {
            "run": outcome.run,
            "parameters": outcome.parameter_report,
            "adapter_reload_verified": outcome.adapter_reload_verified,
        }
    )


@app.command("resume")
def resume(
    run_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path(".tuneforge"),
) -> None:
    store = MetadataStore(root / "tuneforge.sqlite3")
    run_value = store.get_run(run_id)
    checkpoints = store.list_checkpoints(run_id)
    if run_value is None or not checkpoints:
        raise typer.BadParameter("run or checkpoint not found", param_hint="run_id")
    run_root = root / "runs" / run_id
    model = ModelManifest.model_validate_json(
        (run_root / "model-manifest.json").read_text(encoding="utf-8")
    )
    training_config = TrainingConfig.model_validate_json(
        (run_root / "training-config.json").read_text(encoding="utf-8")
    )
    training_plan = TrainingPlan.model_validate_json(
        (run_root / "training-plan.json").read_text(encoding="utf-8")
    )
    dataset = store.get_dataset(run_value.dataset_fingerprint)
    if dataset is None:
        raise typer.BadParameter("dataset metadata is missing", param_hint="run_id")
    _, records = build_dataset_manifest(
        dataset.source.path,
        SplitConfig(
            train_ratio=dataset.split.ratios["train"],
            validation_ratio=dataset.split.ratios["validation"],
            test_ratio=dataset.split.ratios["test"],
            seed=dataset.split.seed,
            algorithm=dataset.split.algorithm,
        ),
        license_metadata=dataset.source.license,
        provenance=dataset.source.provenance,
    )
    outcome = DeterministicTrainerBackend().execute(
        training_plan,
        model,
        training_config,
        records,
        root,
        store,
        resume=checkpoints[-1],
    )
    _echo({"run": outcome.run, "resumed_from": checkpoints[-1].checkpoint_id})


@app.command("evaluate")
def evaluate(
    run_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path(".tuneforge"),
) -> None:
    store = MetadataStore(root / "tuneforge.sqlite3")
    run_value = store.get_run(run_id)
    checkpoints = store.list_checkpoints(run_id)
    if run_value is None or not checkpoints:
        raise typer.BadParameter("run or checkpoint not found", param_hint="run_id")
    dataset = store.get_dataset(run_value.dataset_fingerprint)
    if dataset is None:
        raise typer.BadParameter("dataset metadata is missing", param_hint="run_id")
    config = TrainingConfig.model_validate_json(
        (root / "runs" / run_id / "training-config.json").read_text(encoding="utf-8")
    )
    _, records = build_dataset_manifest(dataset.source.path, SplitConfig(seed=dataset.split.seed))
    checkpoint = checkpoints[-1]
    model = load_checkpoint_model(checkpoint, checkpoint_directory(root, checkpoint), config)
    loss = evaluate_language_model_loss(model, records[-1:], max_length=config.max_sequence_length)
    _echo({"run_id": run_id, "measured_loss": loss, "scope": "local tiny fixture"})


@app.command("compare")
def compare(
    baseline: Annotated[str, typer.Argument()],
    candidate: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path(".tuneforge"),
) -> None:
    store = MetadataStore(root / "tuneforge.sqlite3")
    left = store.get_evaluation(baseline)
    right = store.get_evaluation(candidate)
    if left is None or right is None:
        raise typer.BadParameter("evaluation not found")
    _echo(compare_results(left, right))


@app.command("checkpoints")
def checkpoints(
    run_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path(".tuneforge"),
) -> None:
    _echo(MetadataStore(root / "tuneforge.sqlite3").list_checkpoints(run_id))


@registry_app.command("list")
def registry_list(root: Annotated[Path, typer.Option()] = Path(".tuneforge")) -> None:
    _echo(MetadataStore(root / "tuneforge.sqlite3").list_registry())


@app.command("evidence")
def evidence(
    run_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path(".tuneforge"),
) -> None:
    manifest = root / "runs" / run_id / "evidence" / "bundle-manifest.json"
    if not manifest.is_file():
        raise typer.BadParameter("evidence bundle not found", param_hint="run_id")
    _echo(json.loads(manifest.read_text(encoding="utf-8")))


@app.command("capabilities")
def capabilities() -> None:
    _echo(detect_runtime())


@app.command("demo")
def demo(root: Annotated[Path, typer.Option()] = Path(".tuneforge-demo")) -> None:
    _echo(run_demo(root))


@app.command("eval")
def eval_harness(root: Annotated[Path, typer.Option()] = Path(".tuneforge-eval")) -> None:
    """Run all twelve deterministic specification acceptance scenarios."""

    result = run_scenarios(root)
    _echo(result)
    if not result["all_passed"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
