"""Deterministic AIONS CI diagnostic primitive.

Captures a process result, classifies common failure families, and emits a
bounded JSON object suitable for a repair agent. No source mutation occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Literal

FailureKind = Literal["format", "compile", "test", "lint", "benchmark", "unknown"]

@dataclass(frozen=True)
class Diagnostic:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    kind: FailureKind


def classify(command: str, output: str, exit_code: int) -> FailureKind:
    text = output.lower()
    cmd = command.lower()
    if "cargo fmt" in cmd or "rustfmt" in text:
        return "format" if exit_code else "unknown"
    if "cargo clippy" in cmd or "clippy" in text:
        return "lint" if exit_code else "unknown"
    if "cargo bench" in cmd or "criterion" in text:
        return "benchmark" if exit_code else "unknown"
    if "cargo test" in cmd or re.search(r"test .* failed|failures:", text):
        return "test" if exit_code else "unknown"
    if "cargo build" in cmd or "error[e" in text or "could not compile" in text:
        return "compile" if exit_code else "unknown"
    return "unknown"


def capture(command: str, exit_code: int, stdout: str, stderr: str, limit: int = 8000) -> Diagnostic:
    combined = (stdout + "\n" + stderr)[-limit:]
    return Diagnostic(command, exit_code, stdout[-limit:], stderr[-limit:], classify(command, combined, exit_code))


def to_json(diagnostic: Diagnostic) -> str:
    return json.dumps(asdict(diagnostic), sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    # Smoke contract: stdin supplies {command, exit_code, stdout, stderr}.
    payload = json.load(__import__("sys").stdin)
    print(to_json(capture(**payload)))
