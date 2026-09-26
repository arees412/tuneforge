"""Security controls shared by dataset, artifact, logging, and API layers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class PathSecurityError(ValueError):
    """Raised when a requested path leaves its configured root."""


_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[a-z0-9._~+/=-]+"),
    re.compile(r"(?i)(hf_)[a-z0-9]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|password|token)\s*[:=]\s*)[^\s,;]+"),
)


def confined_path(root: Path, relative: str | Path, *, must_exist: bool = False) -> Path:
    """Resolve a relative artifact path and reject absolute paths and traversal."""

    candidate = Path(relative)
    if candidate.is_absolute():
        raise PathSecurityError("absolute artifact paths are not allowed")
    root_resolved = root.resolve()
    resolved = (root_resolved / candidate).resolve(strict=must_exist)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PathSecurityError("artifact path escapes configured root") from exc
    return resolved


def redact_text(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", redacted)
    return redacted


def redact(value: Any) -> Any:
    """Recursively redact credentials without including raw training samples."""

    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {str(key): redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def validate_artifact_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}", value):
        raise ValueError("artifact identifier contains unsafe characters")
    return value

