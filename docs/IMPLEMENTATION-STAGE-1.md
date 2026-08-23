# Memory / System Graph — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Create a real graph model based on **stable ID** AIONS entity identities rather than UI-only nodes.

## Milestones
- [x] Operational graph contract.
- [x] Canonical node/edge identity model.
- [x] Entity types for memory, code symbol, process, agent, module and dependency.
- [x] Event ingestion contract.
- [x] Snapshot/export format.
- [x] Integrity checks for orphaned or duplicate IDs.
- [x] Query API for neighbourhood, ancestry and dependency paths.
- [x] Test fixtures generated from deterministic synthetic AIONS state.

## Rules
The graph observes subsystem state but never becomes its owner. Visualization comes after the model and query contracts are verified.
