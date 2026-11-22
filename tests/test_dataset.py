from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuneforge.dataset import (
    WhitespaceTokenizer,
    analyze_tokenization,
    build_dataset_manifest,
    build_quality_report,
    deduplicate,
    detect_format,
    detect_leakage,
    detect_schema,
    find_duplicates,
    normalize_record,
    read_records,
    split_records,
    validate_dataset,
    validate_records,
)
from tuneforge.domain import DatasetFormat, DatasetSchema, LicenseMetadata, SplitConfig
from tuneforge.utils import write_canonical_json


def test_json_jsonl_and_csv_intake(
    tmp_path: Path, instruction_records: list[dict[str, object]]
) -> None:
    json_path = tmp_path / "data.json"
    jsonl_path = tmp_path / "data.jsonl"
    csv_path = tmp_path / "data.csv"
    write_canonical_json(json_path, {"records": instruction_records})
    jsonl_path.write_text(
        "\n".join(json.dumps(row) for row in instruction_records) + "\n", encoding="utf-8"
    )
    csv_path.write_text(
        "instruction,input,output\nSay hi,,hello\nSay bye,,goodbye\n", encoding="utf-8"
    )
    assert len(read_records(json_path)) == 12
    assert len(read_records(jsonl_path)) == 12
    assert len(read_records(csv_path)) == 2
    assert detect_format(csv_path) is DatasetFormat.CSV


def test_invalid_format_and_json(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("not supported", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        read_records(path)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    assert not validate_dataset(broken).valid


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({"instruction": "Do", "output": "Done"}, DatasetSchema.INSTRUCTION),
        (
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Yo"},
                ]
            },
            DatasetSchema.CHAT,
        ),
        ({"prompt": "P", "completion": "C"}, DatasetSchema.PROMPT_COMPLETION),
        ({"prompt": "P", "chosen": "C", "rejected": "R"}, DatasetSchema.PREFERENCE),
    ],
)
def test_schema_detection_and_normalization(
    record: dict[str, object], expected: DatasetSchema
) -> None:
    assert detect_schema(record) is expected
    normalized = normalize_record(record, 0, expected)
    assert normalized.prompt
    assert normalized.response or normalized.chosen


def test_strict_schema_rejects_unknown_and_mixed_fields() -> None:
    result = validate_records([{"instruction": "Do", "output": "Done", "mystery": 1}])
    assert not result.valid
    with pytest.raises(ValueError, match="conflicts"):
        detect_schema({"instruction": "x", "output": "y", "prompt": "p", "completion": "c"})


def test_chat_role_rules_and_preference_rules() -> None:
    invalid_chat = validate_records(
        [
            {
                "messages": [
                    {"role": "assistant", "content": "wrong"},
                    {"role": "user", "content": "order"},
                ]
            }
        ]
    )
    invalid_preference = validate_records([{"prompt": "p", "chosen": "x", "rejected": "x"}])
    assert not invalid_chat.valid
    assert not invalid_preference.valid


def test_duplicates_deduplication_and_quality() -> None:
    rows = [
        {"instruction": "Hello", "output": "World"},
        {"instruction": "Hello", "output": "World"},
        {"instruction": " hello ", "output": " WORLD "},
    ]
    result = validate_records(rows)
    assert result.valid
    findings = find_duplicates(result.records)
    assert [item.reason for item in findings] == ["exact", "normalized_text"]
    cleaned = deduplicate(result.records, findings)
    quality = build_quality_report(cleaned, duplicate_findings=findings)
    assert len(cleaned) == 1
    assert quality.duplicates == 2


def test_deterministic_split_fingerprint_and_manifest(dataset_path: Path) -> None:
    first, records = build_dataset_manifest(
        dataset_path,
        SplitConfig(seed=9),
        license_metadata=LicenseMetadata(identifier="CC0-1.0", verified=True),
    )
    second, _ = build_dataset_manifest(dataset_path, SplitConfig(seed=9))
    assert first.fingerprint == second.fingerprint
    assert split_records(records, SplitConfig(seed=9)) == split_records(
        records, SplitConfig(seed=9)
    )
    assert first.source.license.verified


def test_leakage_and_tokenization() -> None:
    record = normalize_record(
        {"instruction": "same prompt", "output": "same target"}, 0, DatasetSchema.INSTRUCTION
    )
    copied = record.model_copy(update={"id": "copy", "source_record_id": "copy"})
    findings = detect_leakage({"train": [record], "test": [copied]})
    assert {item.reason for item in findings} >= {"identical_prompt", "identical_target"}
    analysis = analyze_tokenization([record], WhitespaceTokenizer(), max_length=2)
    assert analysis.truncation_rate == 1.0


def test_size_limit_and_empty_dataset(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    assert not validate_dataset(path).valid
    path.write_text("[] " * 10, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds"):
        read_records(path, max_bytes=2)
