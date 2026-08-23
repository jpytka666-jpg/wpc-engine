# Phase 1 Resident Runtime Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification after every step.

**Goal:** Close the remaining Phase 1 WPC Runtime gates by proving and documenting the resident runtime, load-once weights, reusable KV/mmap storage, and agent-session persistence already present in Rust.

**Architecture:** Keep Rust as the system-of-record and correctness reference. Reuse the existing `ResidentEngine`, `ResidentSession`, `MoeKvCache`, mmap-backed WPC v4 model data, and `BatchEngine`; do not introduce a second workspace/memory subsystem unless profiling later proves a hot-path allocation problem.

**Tech Stack:** Rust, existing WPC runtime, `memmap2`, `Arc`, existing KV/cache code, GitHub Actions.

**Spec:** `docs/AIONS-INTEGRATION-MAP.md` Phase 1; `docs/UNIFIED_STACK.md` runtime/performance sections.

## Global Constraints

- Work only on `feature/phase1-resident-runtime` and integrate through PR #23.
- Keep existing Rust implementation as the correctness reference.
- Do not introduce Mojo/CUDA/ASM in Phase 1.
- Do not use CBMS as hot-path KV storage.
- Preserve numerical behaviour within existing documented tolerances.
- Do not add a new allocation/workspace subsystem without profiler evidence.

---

### Task 1: Verify resident agent/session lifetime — COMPLETE

- [x] `ResidentEngine::load` happens before the agent turn loop.
- [x] One `ResidentSession` is retained across turns.
- [x] Shared prompt-prefix reuse and cache truncation semantics are verified from the implementation.

### Task 2: Verify load-once WPC weight storage — COMPLETE

- [x] Regression test proves multiple linear layers share the same `Arc` backing model data.
- [x] mmap-backed model storage is the load-once mechanism.

### Task 3: Verify reusable KV/mmap allocation behaviour — COMPLETE

- [x] Regression tests cover mmap capacity reuse.
- [x] Regression tests cover KV growth, preservation, and reuse after growth.

### Task 4: CI verification — COMPLETE

- [x] Full workspace build passes.
- [x] Full workspace tests pass.
- [x] Formatting passes.
- [x] Runtime Clippy passes with `-D warnings`.
- [x] Existing benchmark compile/smoke gates pass.

Verified by GitHub Actions CI run `32669214449` on PR #23.

### Task 5: Close Phase 1 roadmap gates — COMPLETE

- [x] Resident runtime marked complete in `docs/AIONS-INTEGRATION-MAP.md`.
- [x] Load-once/reuse marked complete after CI verification.
- [x] CBMS is documented as outside the real-time token path.
- [x] PR #23 prepared for merge into `integration/full-organism-v2`.

## Design decision

A separate `ResidentWorkspace` is intentionally **not** introduced in this phase. The current implementation already provides resident session state, mmap-backed load-once weights, and reusable KV/mmap storage. Allocation reuse inside individual forward-token scratch vectors remains a profiling-driven performance investigation for a later phase.
