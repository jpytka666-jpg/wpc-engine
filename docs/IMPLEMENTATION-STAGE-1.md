# Memory / System Graph — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Create a real graph model based on stable AIONS entity IDs rather than UI-only nodes.

## Milestones
- [x] Operational graph contract.
- [ ] Canonical node/edge identity model.
- [ ] Entity types for memory, code symbol, process, agent, module and dependency.
- [ ] Event ingestion contract.
- [ ] Snapshot/export format.
- [ ] Integrity checks for orphaned or duplicate IDs.
- [ ] Query API for neighbourhood, ancestry and dependency paths.
- [ ] Test fixtures generated from deterministic synthetic AIONS state.

## Rules
The graph observes subsystem state but never becomes its owner. Visualization comes after the model and query contracts are verified.
