#!/usr/bin/env python3
"""Fail when publication-unsafe or generated files enter the repository."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
MAX_TRACKED_BYTES = 10 * 1024 * 1024
GENERATED_SUFFIXES = {
    ".ckpt",
    ".feather",
    ".joblib",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
}
CONTENT_SUFFIXES = {
    ".cff",
    ".csv",
    ".example",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".sql",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
LOCAL_PATH_PATTERNS = [
    re.compile(r"/home/"),
    re.compile(r"/mnt/"),
    re.compile(r"/Users/"),
    re.compile(r"[A-Za-z]:\\\\Users\\\\"),
]
TOKEN_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@", re.IGNORECASE),
]
NAMED_SECRET = re.compile(
    r"(?im)^\s*[\"']?"
    r"(api_key|access_token|client_secret|password|AZURE_OPENAI_API_KEY|DB_PASSWORD)"
    r"[\"']?\s*[:=]\s*(.*?)\s*,?\s*$"
)
EXPECTED_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE.md",
    "CITATION.cff",
    "pyproject.toml",
}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def contains_possible_secret(path: Path, text: str) -> bool:
    if any(pattern.search(text) for pattern in TOKEN_PATTERNS):
        return True

    safe_values = {"", "null", "none", "replace-me", "str"}
    for match in NAMED_SECRET.finditer(text):
        raw_value = match.group(2).strip().rstrip(",").strip()
        value = raw_value.strip("\"'")
        if value.lower() in safe_values:
            continue
        if value.startswith("$" + "{") and value.endswith("}"):
            continue
        if value.startswith("<") and value.endswith(">"):
            continue
        if path.suffix == ".py" and not raw_value.startswith(("\"", "'")):
            continue
        return True
    return False


def publication_files() -> list[Path]:
    """Return tracked and publishable untracked files, excluding ignored files."""
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "-z",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]

    return [
        ROOT / relative_path.decode("utf-8")
        for relative_path in result.stdout.split(b"\0")
        if relative_path
    ]


def check_tree() -> list[str]:
    errors: list[str] = []

    for expected in sorted(EXPECTED_FILES):
        if not (ROOT / expected).is_file():
            errors.append(f"missing required file: {expected}")

    files = publication_files()
    for path in sorted(files):

        rel = relative(path)
        if "__pycache__" in path.parts:
            errors.append(f"generated cache file: {rel}")
        if path.suffix.lower() in GENERATED_SUFFIXES:
            errors.append(f"generated artifact: {rel}")
        if "|" in path.name:
            errors.append(f"non-portable filename: {rel}")
        if path.stat().st_size > MAX_TRACKED_BYTES:
            errors.append(f"file exceeds 10 MiB publication limit: {rel}")

        name = path.name
        if name.startswith(".env") and name != ".env.example":
            errors.append(f"unapproved environment file: {rel}")
        if name.endswith(".env"):
            errors.append(f"unapproved environment file: {rel}")

        if path.resolve() == SELF or path.suffix.lower() not in CONTENT_SUFFIXES:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"non-text content in source/config file: {rel}")
            continue

        for pattern in LOCAL_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"machine-specific absolute path: {rel}")
                break

        if not name.endswith(".example") and contains_possible_secret(path, text):
            errors.append(f"possible literal secret: {rel}")

        try:
            if path.suffix == ".py":
                ast.parse(text, filename=rel)
            elif path.suffix == ".json":
                json.loads(text)
            elif path.suffix in {".yaml", ".yml"}:
                yaml.safe_load(text)
        except Exception as exc:
            errors.append(f"parse failure in {rel}: {type(exc).__name__}: {exc}")

    return sorted(set(errors))


def main() -> int:
    errors = check_tree()
    if errors:
        print("Repository check failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    file_count = len(publication_files())
    print(f"Repository check passed for {file_count} files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
