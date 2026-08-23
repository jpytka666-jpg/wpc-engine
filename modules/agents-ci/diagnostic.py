"""Bounded diagnostics parsing and failure classification for AIONS Local CI."""

from __future__ import annotations

import re

_REPAIRABLE_STAGES = {"format", "compile", "test", "lint", "benchmark"}
_NON_REPAIRABLE_STAGES = {"contract"}
_ERROR_RE = re.compile(r"(?:error(?:\[[A-Z]\d+\])?|fatal error)[: ]", re.IGNORECASE)
_WARNING_RE = re.compile(r"warning(?:\[[A-Z]\d+\])?:", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:-->|at )?([A-Za-z0-9_./\\-]+\.(?:rs|toml|json|py|yml|yaml))(?::\d+(?::\d+)?)?")


def classify_failure(exit_code: int, stage: str, summary: str) -> tuple[str, bool]:
    """Return the canonical failure class and whether automated repair is allowed."""
    del summary

    if stage in _NON_REPAIRABLE_STAGES:
        return "contract", False

    if stage in _REPAIRABLE_STAGES:
        classification = "formatting" if stage == "format" else stage
        return classification, exit_code != 0

    return "unknown", False


def parse_rust_diagnostics(stdout: str, stderr: str, affected_paths: list[str], max_items: int = 20) -> list[str]:
    """Extract a small deterministic diagnostic context from Rust command output."""
    candidates: list[str] = []
    seen: set[str] = set()

    for raw in (stdout + "\n" + stderr).splitlines():
        line = raw.strip()
        if not line:
            continue
        if _ERROR_RE.search(line) or _WARNING_RE.search(line):
            candidates.append(line)
        path_match = _PATH_RE.search(line)
        if path_match:
            candidates.append(f"path:{path_match.group(1)}")

    for path in affected_paths:
        if path:
            candidates.append(f"path:{path}")

    result: list[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= max_items:
            break

    return result
