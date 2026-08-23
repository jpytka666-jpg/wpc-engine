# WPC Runtime — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Promote the design contract into a small, testable runtime-facing API without changing unrelated crates.

## Milestones
- [x] Module contract and interface sketch.
- [x] Dedicated CI gate.
- [x] Typed runtime request/response boundary.
- [x] Deterministic resident lifecycle test.
- [x] KV adapter trait consumed without owning KV storage.
- [ ] Benchmark smoke test for the public boundary.
- [ ] Cross-module integration test with Agents/CI diagnostics.

## Rules
No local deployment. No modification of existing AIONS workspaces. Every implementation step lands on this branch and must pass its own CI before promotion.
