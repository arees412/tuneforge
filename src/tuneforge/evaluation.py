"""Deterministic fixture evaluation and baseline/candidate regression comparison."""

from __future__ import annotations

import json
import math
from collections.abc import Callable

import torch
from torch import nn

from tuneforge.domain import (
    CanonicalTrainingRecord,
    EvaluationResult,
    EvaluationSuite,
    ModelComparison,
)

Predictor = Callable[[str], str]


def evaluate_suite(
    suite: EvaluationSuite,
    subject_id: str,
    predictor: Predictor,
) -> EvaluationResult:
    results: dict[str, float] = {}
    warnings: list[str] = []
    for case in suite.cases:
        predicted = predictor(case.input)
        if case.metric == "exact_match":
            score = float(predicted.strip() == case.expected_output.strip())
        elif case.metric == "format_valid":
            try:
                json.loads(predicted)
                score = 1.0
            except json.JSONDecodeError:
                score = 0.0
        else:
            try:
                score = float(predicted)
            except ValueError:
                score = math.inf
                warnings.append(f"case {case.id} did not return a numeric loss")
        results[case.id] = score
    aggregate: dict[str, list[float]] = {}
    for case in suite.cases:
        aggregate.setdefault(case.metric, []).append(results[case.id])
    metrics = {name: sum(values) / len(values) for name, values in aggregate.items()}
    passed = all(
        results[case.id] <= case.threshold
        if case.metric == "loss"
        else results[case.id] >= case.threshold
        for case in suite.cases
    )
    return EvaluationResult(
        suite_id=suite.id,
        suite_version=suite.version,
        subject_id=subject_id,
        metrics=metrics,
        case_results=results,
        passed=passed,
        warnings=warnings,
    )


def compare_results(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    allowed_regression: float = 0.0,
) -> ModelComparison:
    if (baseline.suite_id, baseline.suite_version) != (
        candidate.suite_id,
        candidate.suite_version,
    ):
        raise ValueError("baseline and candidate must use the same evaluation suite version")
    common = sorted(set(baseline.metrics) & set(candidate.metrics))
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for metric in common:
        direction = -1.0 if metric == "loss" else 1.0
        delta = candidate.metrics[metric] - baseline.metrics[metric]
        deltas[metric] = delta
        if direction * delta < -allowed_regression:
            regressions.append(metric)
    warnings = [] if common else ["no common metrics were available for comparison"]
    return ModelComparison(
        baseline_id=baseline.subject_id,
        candidate_id=candidate.subject_id,
        suite_id=baseline.suite_id,
        metric_deltas=deltas,
        regressions=regressions,
        passed=bool(common) and not regressions,
        warnings=warnings,
    )


def evaluate_language_model_loss(
    model: nn.Module,
    records: list[CanonicalTrainingRecord],
    *,
    max_length: int = 32,
) -> float:
    if not records:
        raise ValueError("loss evaluation requires records")
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for record in records:
            text = f"{record.prompt}\n{record.response or record.chosen or ''}"
            token_ids = [1, *(3 + byte % 125 for byte in text.encode("utf-8")), 2][:max_length]
            tensor = torch.tensor([token_ids], dtype=torch.long)
            loss = model(input_ids=tensor, labels=tensor).loss
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite evaluation loss detected")
            losses.append(float(loss))
    return sum(losses) / len(losses)


def assert_no_evaluation_contamination(
    training: list[CanonicalTrainingRecord], evaluation: list[CanonicalTrainingRecord]
) -> None:
    training_values = {
        " ".join(f"{item.prompt}\n{item.response or item.chosen or ''}".casefold().split())
        for item in training
    }
    contaminated = [
        item.id
        for item in evaluation
        if " ".join(f"{item.prompt}\n{item.response or item.chosen or ''}".casefold().split())
        in training_values
    ]
    if contaminated:
        raise ValueError(f"evaluation contamination detected: {', '.join(contaminated)}")


__all__ = [
    "assert_no_evaluation_contamination",
    "compare_results",
    "evaluate_language_model_loss",
    "evaluate_suite",
]
