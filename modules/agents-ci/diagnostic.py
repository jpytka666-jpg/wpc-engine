"""Failure classification for the AIONS Agents / Local CI contract."""

from __future__ import annotations


_REPAIRABLE_STAGES = {"format", "compile", "test", "lint", "benchmark"}
_NON_REPAIRABLE_STAGES = {"contract"}


def classify_failure(exit_code: int, stage: str, summary: str) -> tuple[str, bool]:
    """Return the canonical failure class and whether automated repair is allowed.

    Contract failures are authoritative interface violations and therefore remain
    non-repairable. Operational failures are classified by their stage and may be
    repaired when the stage is supported by the local repair loop.
    """
    del summary  # Classification is contract-driven; free-text is bounded context.

    if stage in _NON_REPAIRABLE_STAGES:
        return "contract", False

    if stage in _REPAIRABLE_STAGES:
        classification = "formatting" if stage == "format" else stage
        return classification, exit_code != 0

    return "unknown", False
