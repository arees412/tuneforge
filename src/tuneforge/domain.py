"""Typed domain models for governed training workflows."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class DatasetFormat(StrEnum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"


class DatasetSchema(StrEnum):
    INSTRUCTION = "instruction"
    CHAT = "chat"
    PROMPT_COMPLETION = "prompt_completion"
    PREFERENCE = "preference"


class Modality(StrEnum):
    TEXT = "text"
    IMAGE_TEXT = "image_text"


class TrainingMethod(StrEnum):
    FULL_FINE_TUNE = "full_fine_tune"
    LORA = "lora"
    QLORA = "qlora"


class RunState(StrEnum):
    CREATED = "created"
    VALIDATED = "validated"
    PLANNED = "planned"
    QUEUED = "queued"
    RUNNING = "running"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RegistryStatus(StrEnum):
    CANDIDATE = "candidate"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class LicenseMetadata(StrictModel):
    identifier: str = "unknown"
    source_url: str | None = None
    usage_restrictions: str | None = None
    redistribution_notes: str | None = None
    verified: bool = False


class DatasetSource(StrictModel):
    path: Path
    format: DatasetFormat
    content_hash: str
    record_count: int = Field(ge=0)
    schema_name: DatasetSchema
    license: LicenseMetadata = Field(default_factory=LicenseMetadata)
    provenance: dict[str, str] = Field(default_factory=dict)


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class CanonicalTrainingRecord(StrictModel):
    id: str
    modality: Modality = Modality.TEXT
    system: str | None = None
    prompt: str
    response: str | None = None
    messages: list[Message] = Field(default_factory=list)
    chosen: str | None = None
    rejected: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_record_id: str


class ValidationIssue(StrictModel):
    record_index: int | None = None
    code: str
    message: str


class DatasetValidationResult(StrictModel):
    valid: bool
    schema_name: DatasetSchema | None = None
    records: list[CanonicalTrainingRecord] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)


class DuplicateFinding(StrictModel):
    kept_record_id: str
    duplicate_record_id: str
    reason: Literal["exact", "normalized_text"]
    decision: Literal["keep_first"] = "keep_first"


class LeakageFinding(StrictModel):
    left_split: str
    right_split: str
    left_record_id: str
    right_record_id: str
    reason: Literal["identical_record", "identical_prompt", "identical_target"]


class SplitConfig(StrictModel):
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    validation_ratio: float = Field(default=0.1, ge=0, lt=1)
    test_ratio: float = Field(default=0.1, ge=0, lt=1)
    seed: int = 42
    algorithm: Literal["stable_hash", "seeded_random"] = "stable_hash"

    @model_validator(mode="after")
    def validate_ratios(self) -> "SplitConfig":
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-9:
            raise ValueError("split ratios must sum to 1.0")
        return self


class DatasetSplit(StrictModel):
    algorithm: str
    seed: int
    ratios: dict[str, float]
    record_ids: dict[str, list[str]]
    record_counts: dict[str, int]
    fingerprint: str


class DataQualityReport(StrictModel):
    total_records: int
    valid_records: int
    invalid_records: int
    duplicates: int
    empty_targets: int
    average_input_length: float
    average_output_length: float
    max_length: int
    schema_violations: int
    split_leakage: int
    sensitivity_warnings: int
    license_metadata_available: bool


class TokenizationReport(StrictModel):
    tokenizer: str
    input_tokens: list[int]
    target_tokens: list[int]
    combined_tokens: list[int]
    truncation_rate: float
    over_limit_records: list[str]


class DatasetManifest(StrictModel):
    manifest_version: str = "1"
    normalization_version: str = "1"
    source: DatasetSource
    split: DatasetSplit
    fingerprint: str
    duplicate_findings: list[DuplicateFinding] = Field(default_factory=list)
    leakage_findings: list[LeakageFinding] = Field(default_factory=list)
    quality: DataQualityReport


class ModelSpec(StrictModel):
    model_id: str
    architecture_family: str | None = None
    revision: str = "local"
    source: Literal["local_config", "huggingface", "filesystem"] = "local_config"
    parameter_count: int | None = Field(default=None, ge=0)
    dtype_support: list[str] = Field(default_factory=list)
    trust_remote_code_required: bool = False
    trust_remote_code_allowed: bool = False
    license: LicenseMetadata = Field(default_factory=LicenseMetadata)
    context_length: int | None = Field(default=None, gt=0)
    quantization_compatible: bool | None = None
    task_type: str = "causal_lm"
    modality: Modality = Modality.TEXT

    @model_validator(mode="after")
    def reject_remote_code(self) -> "ModelSpec":
        if self.trust_remote_code_required and not self.trust_remote_code_allowed:
            raise ValueError("model requires remote code but operator opt-in is disabled")
        return self


class LoRAConfig(StrictModel):
    r: int = Field(default=4, gt=0, le=1024)
    alpha: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["c_attn"])
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM"] = "CAUSAL_LM"
    modules_to_save: list[str] = Field(default_factory=list)

    @field_validator("target_modules")
    @classmethod
    def target_modules_are_present(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("target_modules must contain non-empty module names")
        return value


class QLoRAConfig(StrictModel):
    load_in_4bit: bool = True
    quantization_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    double_quantization: bool = True


class ModelManifest(StrictModel):
    model_id: str
    revision: str
    configuration_hash: str
    tokenizer_id: str
    tokenizer_revision: str
    license: LicenseMetadata
    parameter_count: int | None
    training_method: TrainingMethod
    adapter_configuration: LoRAConfig | QLoRAConfig | None = None
    base_model_fingerprint: str | None = None
    manifest_hash: str


class TrainingConfig(StrictModel):
    epochs: int = Field(default=1, gt=0, le=100)
    max_steps: int = Field(default=4, gt=0, le=100_000)
    learning_rate: float = Field(default=5e-4, gt=0, le=1)
    weight_decay: float = Field(default=0.0, ge=0, le=1)
    warmup_ratio: float = Field(default=0.0, ge=0, lt=1)
    per_device_batch_size: int = Field(default=2, gt=0, le=1024)
    gradient_accumulation: int = Field(default=1, gt=0, le=1024)
    max_sequence_length: int = Field(default=32, ge=4, le=131_072)
    optimizer: Literal["adamw"] = "adamw"
    scheduler: Literal["constant", "linear"] = "constant"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    gradient_clipping: float = Field(default=1.0, gt=0)
    seed: int = 42
    logging_steps: int = Field(default=1, gt=0)
    evaluation_strategy: Literal["none", "steps", "epoch"] = "steps"
    save_strategy: Literal["steps", "epoch"] = "steps"
    checkpoint_frequency: int = Field(default=2, gt=0)
    early_stopping_patience: int | None = Field(default=None, gt=0)
    keep_last_n: int = Field(default=2, gt=0, le=100)


class RuntimeCapabilities(StrictModel):
    os: str
    python_version: str
    torch_version: str | None
    cuda_available: bool
    cuda_version: str | None
    gpu_names: list[str]
    gpu_count: int
    gpu_memory_bytes: list[int]
    mps_available: bool
    cpu: str
    bf16_supported: bool
    fp16_supported: bool
    bitsandbytes_available: bool
    flash_attention_available: bool


class ResourceEstimate(StrictModel):
    estimated_base_model_bytes: int
    estimated_trainable_parameter_bytes: int
    optimizer_state_category: str
    activation_risk: Literal["low", "medium", "high", "unknown"]
    estimated_checkpoint_bytes: int
    disclaimer: str


class TrainingPlan(StrictModel):
    dataset_fingerprint: str
    model_manifest_hash: str
    method: TrainingMethod
    runtime: RuntimeCapabilities
    effective_batch_size: int
    estimated_steps: int
    adapter_configuration: LoRAConfig | QLoRAConfig | None = None
    checkpoint_strategy: dict[str, Any]
    evaluation_plan: dict[str, Any]
    warnings: list[str]
    resource_estimate: ResourceEstimate
    plan_hash: str


class Experiment(StrictModel):
    id: str
    name: str
    created_at: datetime


class TrainingRun(StrictModel):
    id: str
    experiment_id: str | None = None
    state: RunState = RunState.CREATED
    dataset_fingerprint: str
    model_manifest_hash: str
    training_config_hash: str
    package_version: str
    seed: int
    environment: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    failure_category: str | None = None
    last_completed_step: int = 0


class TrainingMetric(StrictModel):
    run_id: str
    step: int
    epoch: float
    training_loss: float
    evaluation_loss: float | None = None
    learning_rate: float
    duration_seconds: float
    trainable_parameters: int
    total_parameters: int


class CheckpointManifest(StrictModel):
    checkpoint_id: str
    run_id: str
    step: int
    epoch: float
    created_at: datetime
    method: TrainingMethod
    base_model_reference: str
    state_kind: Literal["adapter", "full"]
    artifact_paths: list[str]
    file_hashes: dict[str, str]
    metrics: dict[str, float]
    configuration_digest: str
    model_manifest_hash: str
    manual: bool = False


class EvaluationCase(StrictModel):
    id: str
    input: str
    expected_output: str
    metric: Literal["exact_match", "format_valid", "loss"] = "exact_match"
    threshold: float = 1.0
    tags: list[str] = Field(default_factory=list)


class EvaluationSuite(StrictModel):
    id: str
    version: str
    cases: list[EvaluationCase]


class EvaluationResult(StrictModel):
    suite_id: str
    suite_version: str
    subject_id: str
    metrics: dict[str, float]
    case_results: dict[str, float]
    passed: bool
    warnings: list[str] = Field(default_factory=list)


class ModelComparison(StrictModel):
    baseline_id: str
    candidate_id: str
    suite_id: str
    metric_deltas: dict[str, float]
    regressions: list[str]
    passed: bool
    warnings: list[str] = Field(default_factory=list)


class ModelRegistryEntry(StrictModel):
    id: str
    run_id: str
    artifact_kind: Literal["adapter", "full", "reference"]
    artifact_path: str
    evaluation_id: str | None = None
    status: RegistryStatus = RegistryStatus.CANDIDATE
    created_at: datetime
    updated_at: datetime


class AuditEvent(StrictModel):
    id: str
    event_type: str
    subject_id: str
    created_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)


class TrainingEvidence(StrictModel):
    run_id: str
    dataset_manifest_hash: str
    model_manifest_hash: str
    training_plan_hash: str
    runtime: RuntimeCapabilities
    metrics: list[TrainingMetric]
    checkpoint_hashes: dict[str, str]
    evaluation: EvaluationResult | None = None
    registry_status: RegistryStatus | None = None
    final_state: RunState
    evidence_hash: str

