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

### Task 1: Verify resident agent/session lifetime

**Files:**
- Inspect: `wpc-runtime/src/bin/aions-agent.rs`
- Inspect: `wpc-runtime/src/resident.rs`

**Interfaces:**
- `ResidentEngine::load` creates the model once.
- `ResidentEngine::start_session` creates one session before the agent turn loop.
- `ResidentSession::generate` reuses the shared prompt prefix and KV cache.

- [x] Confirm `ResidentEngine::load` happens before the agent turn loop.
- [x] Confirm one `ResidentSession` is retained across turns.
- [x] Confirm prefix reuse and cache truncation semantics.

### Task 2: Verify load-once WPC weight storage

**Files:**
- Inspect: `wpc-runtime/src/wpc_weights_v4.rs`
- Test: `wpc-runtime/tests/resident_weight_sharing.rs`

**Interfaces:**
- `WpcModelDataV4::open` returns `Arc<WpcModelDataV4>` over an mmap of the packed model.
- `WpcLinearV4` clones the same `Arc` for each tensor layer.

- [x] Add a regression test proving multiple linear layers share the same `Arc` backing model data.
- [x] Keep mmap-backed model storage as the load-once mechanism.

### Task 3: Verify reusable KV/mmap allocation behaviour

**Files:**
- Inspect: `wpc-runtime/src/forward_batch.rs`
- Test: `wpc-runtime/tests/resident_memory.rs`

**Interfaces:**
- `MmapF32::ensure_capacity` retains existing allocation when capacity is sufficient.
- `KvLayer` grows capacity when required and preserves existing rows/data.
- Rust `Vec` truncation in the resident KV cache retains allocated capacity.

- [x] Add regression tests for mmap capacity reuse.
- [x] Add regression tests for KV growth/preservation and reuse after growth.

### Task 4: CI verification

**Files:**
- No runtime source changes required.

- [ ] Full workspace build passes.
- [ ] Full workspace tests pass.
- [ ] Formatting passes.
- [ ] Runtime Clippy passes with `-D warnings`.
- [ ] Existing benchmark compile/smoke gates pass.

### Task 5: Close Phase 1 roadmap gates

**Files:**
- Modify: `docs/AIONS-INTEGRATION-MAP.md`
- Modify: `docs/UNIFIED_STACK.md` only if the documented runtime boundary needs correction.

- [ ] Mark resident runtime complete only after CI confirms the implementation.
- [ ] Mark load-once/reuse complete only after CI confirms the new regression tests.
- [ ] Record that CBMS remains outside the real-time token path.
- [ ] Merge PR #23 into `integration/full-organism-v2` once all required checks are green.

## Design decision

A separate `ResidentWorkspace` is intentionally **not** being introduced in this phase. The current implementation already provides resident session state, mmap-backed load-once weights, and reusable KV/mmap storage. Allocation reuse inside individual forward-token scratch vectors is a separate performance investigation and should be driven by profiling after Phase 1, not assumed to be a blocker without measurements.
