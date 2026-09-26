"""Typed FastAPI surface for local governed TuneForge workflows."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from tuneforge.dataset import (
    build_dataset_manifest,
    validate_dataset,
)
from tuneforge.domain import (
    EvaluationSuite,
    LicenseMetadata,
    LoRAConfig,
    QLoRAConfig,
    RegistryStatus,
    SplitConfig,
    TrainingConfig,
    TrainingMethod,
)
from tuneforge.evaluation import evaluate_suite
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
from tuneforge.security import PathSecurityError, confined_path
from tuneforge.storage import MetadataStore
from tuneforge.training import CancellationToken, DeterministicTrainerBackend
from tuneforge.utils import file_sha256, stable_digest


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetPathRequest(APIModel):
    path: str


class DatasetRegisterRequest(DatasetPathRequest):
    split: SplitConfig = Field(default_factory=SplitConfig)
    license: LicenseMetadata = Field(default_factory=LicenseMetadata)
    provenance: dict[str, str] = Field(default_factory=dict)


class PlanRequest(APIModel):
    dataset_id: str
    method: TrainingMethod = TrainingMethod.LORA
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    qlora: QLoRAConfig = Field(default_factory=QLoRAConfig)


class RunRequest(PlanRequest):
    pass


class EvaluationRequest(APIModel):
    subject_id: str
    suite: EvaluationSuite
    predictions: dict[str, str]


class TuneForgeService:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.dataset_root = root / "datasets"
        self.artifact_root = root / "artifacts"
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.store = MetadataStore(root / "tuneforge.sqlite3")
        self.registry = ModelRegistry(self.store)
        self.cancellations: dict[str, CancellationToken] = {}

    def dataset_path(self, value: str) -> Path:
        try:
            return confined_path(self.dataset_root, value, must_exist=True)
        except (PathSecurityError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    def plan(self, request: PlanRequest):  # type: ignore[no-untyped-def]
        dataset = self.store.get_dataset(request.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        base = create_tiny_model(seed=request.training.seed)
        adapter = (
            request.lora
            if request.method is TrainingMethod.LORA
            else request.qlora
            if request.method is TrainingMethod.QLORA
            else None
        )
        model_manifest = build_model_manifest(tiny_model_spec(base), request.method, adapter)
        trainable = int(
            parameter_report(
                apply_lora(base, request.lora) if request.method is TrainingMethod.LORA else base
            )["trainable_parameters"]
        )
        try:
            plan = build_training_plan(
                dataset,
                model_manifest,
                request.training,
                detect_runtime(),
                trainable_parameters=trainable,
                adapter=adapter,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return dataset, model_manifest, plan


def create_app(root: Path | None = None) -> FastAPI:
    service = TuneForgeService(root or Path(".tuneforge"))
    app = FastAPI(title="TuneForge", version="0.1.0")
    app.state.service = service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runtime/capabilities")
    def capabilities():  # type: ignore[no-untyped-def]
        return detect_runtime()

    @app.post("/datasets/validate")
    def validate(request: DatasetPathRequest):  # type: ignore[no-untyped-def]
        return validate_dataset(service.dataset_path(request.path))

    @app.post("/datasets/register")
    def register_dataset(request: DatasetRegisterRequest):  # type: ignore[no-untyped-def]
        path = service.dataset_path(request.path)
        try:
            manifest, _ = build_dataset_manifest(
                path,
                request.split,
                license_metadata=request.license,
                provenance=request.provenance,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        service.store.register_dataset(manifest.fingerprint, manifest)
        return {"id": manifest.fingerprint, "manifest": manifest}

    @app.get("/datasets/{identifier}")
    def get_dataset(identifier: str):  # type: ignore[no-untyped-def]
        manifest = service.store.get_dataset(identifier)
        if manifest is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        return manifest

    @app.post("/training/plan")
    def plan(request: PlanRequest):  # type: ignore[no-untyped-def]
        _, model_manifest, training_plan = service.plan(request)
        return {"model_manifest": model_manifest, "training_plan": training_plan}

    @app.post("/runs")
    def run(request: RunRequest):  # type: ignore[no-untyped-def]
        dataset, model_manifest, training_plan = service.plan(request)
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
            model_manifest,
            request.training,
            records,
            service.artifact_root,
            service.store,
        )
        return {
            "run": outcome.run,
            "parameters": outcome.parameter_report,
            "adapter_reload_verified": outcome.adapter_reload_verified,
        }

    @app.get("/runs/{identifier}")
    def get_run(identifier: str):  # type: ignore[no-untyped-def]
        run_value = service.store.get_run(identifier)
        if run_value is None:
            raise HTTPException(status_code=404, detail="run not found")
        return run_value

    @app.post("/runs/{identifier}/cancel")
    def cancel_run(identifier: str) -> dict[str, str]:
        token = service.cancellations.get(identifier)
        if token is None:
            raise HTTPException(status_code=409, detail="run is not active on this worker")
        token.cancel()
        return {"status": "cancellation_requested"}

    @app.get("/runs/{identifier}/metrics")
    def metrics(identifier: str):  # type: ignore[no-untyped-def]
        if service.store.get_run(identifier) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return service.store.list_metrics(identifier)

    @app.get("/runs/{identifier}/checkpoints")
    def checkpoints(identifier: str):  # type: ignore[no-untyped-def]
        if service.store.get_run(identifier) is None:
            raise HTTPException(status_code=404, detail="run not found")
        return service.store.list_checkpoints(identifier)

    @app.get("/runs/{identifier}/evidence")
    def evidence(identifier: str):  # type: ignore[no-untyped-def]
        run_value = service.store.get_run(identifier)
        if run_value is None:
            raise HTTPException(status_code=404, detail="run not found")
        dataset = service.store.get_dataset(run_value.dataset_fingerprint)
        if dataset is None:
            raise HTTPException(status_code=409, detail="run dataset metadata is missing")
        # Stored bundles are authoritative; the endpoint never fabricates missing manifests.
        path = service.artifact_root / "runs" / identifier / "evidence" / "evidence.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="evidence has not been generated")
        return {"path": path.as_posix(), "sha256": file_sha256(path)}

    @app.post("/evaluations")
    def evaluate(request: EvaluationRequest):  # type: ignore[no-untyped-def]
        missing = sorted({case.id for case in request.suite.cases} - set(request.predictions))
        if missing:
            raise HTTPException(
                status_code=422, detail=f"missing predictions: {', '.join(missing)}"
            )
        case_ids = {case.input: case.id for case in request.suite.cases}
        result = evaluate_suite(
            request.suite,
            request.subject_id,
            lambda value: request.predictions[case_ids[value]],
        )
        identifier = f"eval-{stable_digest(result)[:16]}"
        service.store.save_evaluation(identifier, result)
        return {"id": identifier, "result": result}

    @app.get("/evaluations/{identifier}")
    def get_evaluation(identifier: str):  # type: ignore[no-untyped-def]
        result = service.store.get_evaluation(identifier)
        if result is None:
            raise HTTPException(status_code=404, detail="evaluation not found")
        return result

    @app.get("/registry")
    def registry():  # type: ignore[no-untyped-def]
        return service.registry.list()

    def registry_transition(identifier: str, target: RegistryStatus):  # type: ignore[no-untyped-def]
        try:
            return service.registry.transition(identifier, target)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/registry/{identifier}/approve")
    def approve(identifier: str):  # type: ignore[no-untyped-def]
        return registry_transition(identifier, RegistryStatus.APPROVED)

    @app.post("/registry/{identifier}/reject")
    def reject(identifier: str):  # type: ignore[no-untyped-def]
        return registry_transition(identifier, RegistryStatus.REJECTED)

    return app


app = create_app()


__all__ = ["app", "create_app"]
