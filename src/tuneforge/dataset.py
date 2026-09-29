"""Controlled dataset ingestion, normalization, splitting, and quality checks."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from tuneforge.domain import (
    CanonicalTrainingRecord,
    DataQualityReport,
    DatasetFormat,
    DatasetManifest,
    DatasetSchema,
    DatasetSource,
    DatasetSplit,
    DatasetValidationResult,
    DuplicateFinding,
    LeakageFinding,
    LicenseMetadata,
    Message,
    SplitConfig,
    TokenizationReport,
    ValidationIssue,
)
from tuneforge.utils import file_sha256, sha256_text, stable_digest

MAX_DATASET_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 100_000
MAX_RECORD_CHARS = 100_000

_ALLOWED_FIELDS: dict[DatasetSchema, set[str]] = {
    DatasetSchema.INSTRUCTION: {"id", "instruction", "input", "output", "system", "metadata"},
    DatasetSchema.CHAT: {"id", "messages", "metadata"},
    DatasetSchema.PROMPT_COMPLETION: {"id", "prompt", "completion", "metadata"},
    DatasetSchema.PREFERENCE: {"id", "prompt", "chosen", "rejected", "metadata"},
}


class TokenizerLike(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...


class WhitespaceTokenizer:
    """Deterministic local tokenizer for offline analysis and tests."""

    name = "tuneforge-whitespace-v1"

    def encode(self, text: str) -> list[int]:
        return [int(sha256_text(token)[:8], 16) for token in re.findall(r"\w+|[^\w\s]", text)]


def detect_format(path: Path) -> DatasetFormat:
    try:
        return DatasetFormat(path.suffix.lower().lstrip("."))
    except ValueError as exc:
        raise ValueError(f"unsupported dataset format: {path.suffix or '<none>'}") from exc


def read_records(path: Path, *, max_bytes: int = MAX_DATASET_BYTES) -> list[dict[str, Any]]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("dataset source must be a file")
    if resolved.stat().st_size > max_bytes:
        raise ValueError(f"dataset exceeds {max_bytes} byte limit")
    fmt = detect_format(resolved)
    try:
        if fmt is DatasetFormat.JSON:
            data = json.loads(resolved.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                data = data["records"]
            if not isinstance(data, list):
                raise ValueError("JSON dataset must be an array or contain a records array")
            records = data
        elif fmt is DatasetFormat.JSONL:
            records = [
                json.loads(line)
                for line in resolved.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            with resolved.open("r", encoding="utf-8", newline="") as stream:
                records = list(csv.DictReader(stream))
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
        raise ValueError(f"unable to decode dataset: {exc}") from exc
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset exceeds {MAX_RECORDS} record limit")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every dataset record must be an object")
    return [dict(record) for record in records]


def detect_schema(record: dict[str, Any]) -> DatasetSchema:
    keys = set(record)
    matches: list[DatasetSchema] = []
    if {"instruction", "output"} <= keys:
        matches.append(DatasetSchema.INSTRUCTION)
    if "messages" in keys:
        matches.append(DatasetSchema.CHAT)
    if {"prompt", "completion"} <= keys:
        matches.append(DatasetSchema.PROMPT_COMPLETION)
    if {"prompt", "chosen", "rejected"} <= keys:
        matches.append(DatasetSchema.PREFERENCE)
    if len(matches) != 1:
        raise ValueError("record schema is missing or conflicts with another supported schema")
    return matches[0]


def _require_text(record: dict[str, Any], field: str, *, allow_empty: bool = False) -> str:
    value = record.get(field)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    if not value and not allow_empty:
        raise ValueError(f"{field} must not be empty")
    if len(value) > MAX_RECORD_CHARS:
        raise ValueError(f"{field} exceeds the record size limit")
    return value


def _normalize_messages(value: Any) -> list[Message]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError("messages must be a list containing at least user and assistant messages")
    messages = [Message.model_validate(item) for item in value]
    conversation = messages[1:] if messages[0].role == "system" else messages
    if any(message.role == "system" for message in conversation):
        raise ValueError("system message is only allowed at the beginning")
    expected = "user"
    for message in conversation:
        if message.role != expected:
            raise ValueError("messages must alternate user and assistant roles")
        expected = "assistant" if expected == "user" else "user"
    if conversation[-1].role != "assistant":
        raise ValueError("chat records must end with an assistant response")
    return messages


def normalize_record(
    record: dict[str, Any], index: int, schema_name: DatasetSchema
) -> CanonicalTrainingRecord:
    unknown = set(record) - _ALLOWED_FIELDS[schema_name]
    if unknown:
        raise ValueError(f"unsupported fields: {', '.join(sorted(unknown))}")
    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    source_id = str(record.get("id", index))
    if schema_name is DatasetSchema.INSTRUCTION:
        instruction = _require_text(record, "instruction")
        input_text = _require_text(record, "input", allow_empty=True) if "input" in record else ""
        prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
        response = _require_text(record, "output")
        system = _require_text(record, "system", allow_empty=True) if "system" in record else None
        messages: list[Message] = []
        chosen = rejected = None
    elif schema_name is DatasetSchema.CHAT:
        messages = _normalize_messages(record.get("messages"))
        system = messages[0].content if messages[0].role == "system" else None
        prompt = "\n".join(message.content for message in messages if message.role == "user")
        response = messages[-1].content
        chosen = rejected = None
    elif schema_name is DatasetSchema.PROMPT_COMPLETION:
        prompt = _require_text(record, "prompt")
        response = _require_text(record, "completion")
        system = None
        messages = []
        chosen = rejected = None
    else:
        prompt = _require_text(record, "prompt")
        chosen = _require_text(record, "chosen")
        rejected = _require_text(record, "rejected")
        if chosen == rejected:
            raise ValueError("preference chosen and rejected responses must differ")
        response = None
        system = None
        messages = []
    identity_basis = {"index": index, "source_record_id": source_id, "record": record}
    return CanonicalTrainingRecord(
        id=f"record-{index:06d}-{stable_digest(identity_basis)[:12]}",
        system=system or None,
        prompt=prompt,
        response=response,
        messages=messages,
        chosen=chosen,
        rejected=rejected,
        metadata=metadata,
        source_record_id=source_id,
    )


def validate_records(
    records: list[dict[str, Any]], declared_schema: DatasetSchema | None = None
) -> DatasetValidationResult:
    issues: list[ValidationIssue] = []
    normalized: list[CanonicalTrainingRecord] = []
    detected: DatasetSchema | None = declared_schema
    for index, record in enumerate(records):
        try:
            schema_name = detect_schema(record)
            if detected is None:
                detected = schema_name
            if schema_name is not detected:
                raise ValueError(f"mixed dataset schemas: expected {detected}, found {schema_name}")
            normalized.append(normalize_record(record, index, schema_name))
        except (ValueError, TypeError) as exc:
            issues.append(
                ValidationIssue(record_index=index, code="invalid_record", message=str(exc))
            )
    if not records:
        issues.append(ValidationIssue(code="empty_dataset", message="dataset contains no records"))
    return DatasetValidationResult(
        valid=not issues,
        schema_name=detected,
        records=normalized if not issues else [],
        issues=issues,
    )


def validate_dataset(
    path: Path, declared_schema: DatasetSchema | None = None
) -> DatasetValidationResult:
    try:
        return validate_records(read_records(path), declared_schema)
    except ValueError as exc:
        return DatasetValidationResult(
            valid=False,
            schema_name=declared_schema,
            issues=[ValidationIssue(code="invalid_dataset", message=str(exc))],
        )


def _normalized_text(record: CanonicalTrainingRecord) -> str:
    text = "\n".join(
        part
        for part in [record.system, record.prompt, record.response, record.chosen, record.rejected]
        if part
    )
    return " ".join(text.casefold().split())


def find_duplicates(records: list[CanonicalTrainingRecord]) -> list[DuplicateFinding]:
    findings: list[DuplicateFinding] = []
    exact_seen: dict[str, str] = {}
    normalized_seen: dict[str, str] = {}
    for record in records:
        exact_key = stable_digest(record.model_dump(exclude={"id", "source_record_id"}))
        normalized_key = sha256_text(_normalized_text(record))
        if exact_key in exact_seen:
            findings.append(
                DuplicateFinding(
                    kept_record_id=exact_seen[exact_key],
                    duplicate_record_id=record.id,
                    reason="exact",
                )
            )
        elif normalized_key in normalized_seen:
            findings.append(
                DuplicateFinding(
                    kept_record_id=normalized_seen[normalized_key],
                    duplicate_record_id=record.id,
                    reason="normalized_text",
                )
            )
        else:
            exact_seen[exact_key] = record.id
            normalized_seen[normalized_key] = record.id
    return findings


def deduplicate(
    records: list[CanonicalTrainingRecord], findings: list[DuplicateFinding] | None = None
) -> list[CanonicalTrainingRecord]:
    duplicate_ids = {item.duplicate_record_id for item in (findings or find_duplicates(records))}
    return [record for record in records if record.id not in duplicate_ids]


def split_records(records: list[CanonicalTrainingRecord], config: SplitConfig) -> DatasetSplit:
    buckets: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    if config.algorithm == "stable_hash":
        scored = [
            (int(sha256_text(f"{config.seed}:{record.id}")[:16], 16) / 16**16, record.id)
            for record in records
        ]
    else:
        record_ids = [record.id for record in records]
        random.Random(config.seed).shuffle(record_ids)
        scored = [
            ((index + 0.5) / max(1, len(record_ids)), record_id)
            for index, record_id in enumerate(record_ids)
        ]
    validation_edge = config.train_ratio + config.validation_ratio
    for score, record_id in scored:
        bucket = (
            "train"
            if score < config.train_ratio
            else "validation"
            if score < validation_edge
            else "test"
        )
        buckets[bucket].append(record_id)
    for values in buckets.values():
        values.sort()
    payload = {
        "algorithm": config.algorithm,
        "seed": config.seed,
        "ratios": {
            "train": config.train_ratio,
            "validation": config.validation_ratio,
            "test": config.test_ratio,
        },
        "record_ids": buckets,
    }
    return DatasetSplit(
        **payload,
        record_counts={name: len(values) for name, values in buckets.items()},
        fingerprint=stable_digest(payload),
    )


def detect_leakage(
    split_records_map: dict[str, list[CanonicalTrainingRecord]],
    *,
    check_prompts: bool = True,
    check_targets: bool = True,
) -> list[LeakageFinding]:
    findings: list[LeakageFinding] = []
    indexes: dict[str, dict[str, list[tuple[str, str]]]] = {
        "identical_record": defaultdict(list),
        "identical_prompt": defaultdict(list),
        "identical_target": defaultdict(list),
    }
    for split_name, records in split_records_map.items():
        for record in records:
            indexes["identical_record"][stable_digest(record.model_dump(exclude={"id"}))].append(
                (split_name, record.id)
            )
            if check_prompts:
                indexes["identical_prompt"][
                    sha256_text(" ".join(record.prompt.casefold().split()))
                ].append((split_name, record.id))
            target = record.response or record.chosen
            if check_targets and target:
                indexes["identical_target"][
                    sha256_text(" ".join(target.casefold().split()))
                ].append((split_name, record.id))
    for reason, index in indexes.items():
        for occurrences in index.values():
            for left_index, left in enumerate(occurrences):
                for right in occurrences[left_index + 1 :]:
                    if left[0] != right[0]:
                        findings.append(
                            LeakageFinding(
                                left_split=left[0],
                                right_split=right[0],
                                left_record_id=left[1],
                                right_record_id=right[1],
                                reason=reason,
                            )
                        )
    return findings


def analyze_tokenization(
    records: list[CanonicalTrainingRecord], tokenizer: TokenizerLike, max_length: int
) -> TokenizationReport:
    input_tokens = [len(tokenizer.encode(record.prompt)) for record in records]
    target_tokens = [
        len(tokenizer.encode(record.response or record.chosen or "")) for record in records
    ]
    combined = [left + right for left, right in zip(input_tokens, target_tokens, strict=True)]
    over_limit = [
        record.id for record, length in zip(records, combined, strict=True) if length > max_length
    ]
    return TokenizationReport(
        tokenizer=tokenizer.name,
        input_tokens=input_tokens,
        target_tokens=target_tokens,
        combined_tokens=combined,
        truncation_rate=len(over_limit) / max(1, len(records)),
        over_limit_records=over_limit,
    )


def build_quality_report(
    records: list[CanonicalTrainingRecord],
    *,
    invalid_records: int = 0,
    duplicate_findings: list[DuplicateFinding] | None = None,
    leakage_findings: list[LeakageFinding] | None = None,
    license_metadata: LicenseMetadata | None = None,
) -> DataQualityReport:
    input_lengths = [len(record.prompt) for record in records]
    output_lengths = [len(record.response or record.chosen or "") for record in records]
    sensitive_pattern = re.compile(r"(?i)(api[_-]?key|bearer\s+[a-z0-9._-]+|password\s*[:=])")
    sensitivity_warnings = sum(
        bool(sensitive_pattern.search(_normalized_text(record))) for record in records
    )
    return DataQualityReport(
        total_records=len(records) + invalid_records,
        valid_records=len(records),
        invalid_records=invalid_records,
        duplicates=len(duplicate_findings or []),
        empty_targets=sum(length == 0 for length in output_lengths),
        average_input_length=fmean(input_lengths) if input_lengths else 0.0,
        average_output_length=fmean(output_lengths) if output_lengths else 0.0,
        max_length=max(
            (left + right for left, right in zip(input_lengths, output_lengths, strict=True)),
            default=0,
        ),
        schema_violations=invalid_records,
        split_leakage=len(leakage_findings or []),
        sensitivity_warnings=sensitivity_warnings,
        license_metadata_available=bool(
            license_metadata and license_metadata.identifier != "unknown"
        ),
    )


def build_dataset_manifest(
    path: Path,
    split_config: SplitConfig,
    *,
    license_metadata: LicenseMetadata | None = None,
    provenance: dict[str, str] | None = None,
) -> tuple[DatasetManifest, list[CanonicalTrainingRecord]]:
    raw_records = read_records(path)
    validation = validate_records(raw_records)
    if not validation.valid or validation.schema_name is None:
        messages = "; ".join(issue.message for issue in validation.issues)
        raise ValueError(f"dataset validation failed: {messages}")
    duplicates = find_duplicates(validation.records)
    clean_records = deduplicate(validation.records, duplicates)
    split = split_records(clean_records, split_config)
    lookup = {record.id: record for record in clean_records}
    by_split = {
        name: [lookup[record_id] for record_id in ids] for name, ids in split.record_ids.items()
    }
    leakage = detect_leakage(by_split)
    license_value = license_metadata or LicenseMetadata()
    source = DatasetSource(
        path=path,
        format=detect_format(path),
        content_hash=file_sha256(path),
        record_count=len(raw_records),
        schema_name=validation.schema_name,
        license=license_value,
        provenance=provenance or {},
    )
    quality = build_quality_report(
        clean_records,
        duplicate_findings=duplicates,
        leakage_findings=leakage,
        license_metadata=license_value,
    )
    fingerprint_payload = {
        "source_hash": source.content_hash,
        "normalization_version": "1",
        "schema_version": "1",
        "record_count": len(clean_records),
        "split": split.model_dump(mode="json"),
    }
    manifest = DatasetManifest(
        source=source,
        split=split,
        fingerprint=stable_digest(fingerprint_payload),
        duplicate_findings=duplicates,
        leakage_findings=leakage,
        quality=quality,
    )
    return manifest, clean_records


def schema_counts(records: list[dict[str, Any]]) -> Counter[DatasetSchema]:
    return Counter(detect_schema(record) for record in records)


__all__ = [
    "WhitespaceTokenizer",
    "analyze_tokenization",
    "build_dataset_manifest",
    "build_quality_report",
    "deduplicate",
    "detect_format",
    "detect_leakage",
    "detect_schema",
    "find_duplicates",
    "normalize_record",
    "read_records",
    "split_records",
    "validate_dataset",
    "validate_records",
]
