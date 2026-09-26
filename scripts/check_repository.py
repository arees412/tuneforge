"""Fail CI on unsafe tracked artifacts, credential signatures, or broken local docs links."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROHIBITED_SUFFIXES = {".bin", ".pt", ".pth", ".ckpt", ".safetensors"}
MAX_TRACKED_BYTES = 1_000_000
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
LINK_PATTERN = re.compile(r"\[[^]]+\]\(([^)]+)\)")


def tracked_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / value.decode() for value in completed.stdout.split(b"\0") if value]


def check_files(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() in PROHIBITED_SUFFIXES:
            errors.append(f"prohibited model artifact: {relative}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {relative}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"possible {label} in {relative}")
        if path.suffix.lower() == ".md":
            for target in LINK_PATTERN.findall(text):
                if target.startswith(("http://", "https://", "mailto:", "#")):
                    continue
                clean_target = target.split("#", 1)[0]
                if clean_target and not (path.parent / clean_target).resolve().exists():
                    errors.append(f"broken local documentation link in {relative}: {target}")
    return errors


def main() -> int:
    errors = check_files(tracked_files())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("repository safety and documentation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
