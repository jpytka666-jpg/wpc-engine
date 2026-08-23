# Agents / Local CI — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Turn the parallel CI contract into a bounded diagnostic and repair pipeline.

## Milestones
- [x] Parallel matrix gate with fail-fast disabled.
- [x] Module contract and safety boundary.
- [ ] Stable diagnostic JSON schema.
- [ ] Capture exit code/stdout/stderr and affected paths.
- [ ] Failure classifier for format/build/test/lint/benchmark.
- [ ] Bounded repair proposal interface.
- [ ] Isolated apply + verify loop.
- [ ] Coding-agent command interface behind explicit approval.

## Rules
No automatic push/merge. Repairs are proposed, isolated, verified, and bounded. GitHub CI is authoritative.
