# Phase 1 Resident Runtime Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification after every step.

**Goal:** Close the remaining Phase 1 WPC Runtime gaps by making one resident runtime/session hold model state, reusing working buffers, and proving the behaviour with correctness and performance gates.

**Architecture:** Keep Rust as the reference/system-of-record implementation. Extend the existing `ResidentEngine`, `ResidentSession`, `MoeKvCache`, and `BatchEngine` rather than creating a second memory/runtime subsystem. CBMS remains outside the real-time token path.

**Tech Stack:** Rust, existing WPC runtime, existing KV cache, existing batch attention code, GitHub Actions.

**Spec:** `docs/AIONS-INTEGRATION-MAP.md` Phase 1; `docs/UNIFIED_STACK.md` runtime/performance sections.

## Global Constraints

- Work only on `feature/phase1-resident-runtime` and integrate later through PR.
- Keep existing Rust implementation as the correctness reference.
- Do not introduce Mojo/CUDA/ASM in Phase 1.
- Do not use CBMS as hot-path KV storage.
- Preserve numerical behaviour within existing documented tolerances.
- Every task ends with a focused test or CI verification before the next task.

---

### Task 1: Establish resident-session correctness baseline

**Files:**
- Test: `wpc-runtime/tests/resident_runtime.rs` (create)
- Modify: `wpc-runtime/src/resident.rs` only if required by the tests

**Interfaces:**
- Consume `ResidentEngine::load`, `start_session`, `ResidentSession::generate`.
- Produce regression tests proving repeated session calls reuse prompt state and reset safely on a changed prefix.

- [ ] **Step 1: Write failing tests** for repeated generation with the same prompt prefix and for reset after prefix divergence.
- [ ] **Step 2: Run the focused resident tests and record the current result.**
- [ ] **Step 3: Make the smallest Rust change required for correctness.**
- [ ] **Step 4: Re-run the focused tests until green.**
- [ ] **Step 5: Commit** with `test: establish resident runtime session baseline`.

### Task 2: Introduce `ResidentWorkspace`

**Files:**
- Create: `wpc-runtime/src/resident_workspace.rs`
- Modify: `wpc-runtime/src/lib.rs`
- Test: `wpc-runtime/tests/resident_workspace.rs`

**Interfaces:**
- `ResidentWorkspace::new(hidden, heads, kv_heads, head_dim, moe_intermediate, vocab, experts)`
- Reusable buffers exposed through mutable accessors needed by `forward_token`.
- `reset_for_token()` clears logical lengths without dropping capacity.

- [ ] **Step 1: Write failing capacity-reuse tests.**
- [ ] **Step 2: Implement workspace with reusable `Vec<f32>` buffers and logical-length reset.**
- [ ] **Step 3: Run focused workspace tests.**
- [ ] **Step 4: Wire the workspace into `ResidentSession` construction.**
- [ ] **Step 5: Commit** with `feat: add resident workspace buffers`.

### Task 3: Reuse hot-path buffers in Qwen3-MoE forward

**Files:**
- Modify: `wpc-runtime/src/qwen3_moe_model.rs`
- Modify: `wpc-runtime/src/resident.rs`
- Test: `wpc-runtime/tests/resident_workspace.rs`

**Interfaces:**
- `Qwen3MoeModel::forward_token_with_workspace(token_id, cache, workspace)`.
- Existing `forward_token` remains available as the reference path.

- [ ] **Step 1: Add a correctness test comparing reference `forward_token` and workspace-backed execution on the same deterministic fixture.**
- [ ] **Step 2: Implement workspace-backed temporary buffers without changing model math.**
- [ ] **Step 3: Run correctness tests.**
- [ ] **Step 4: Update `ResidentSession` to use the workspace-backed path.**
- [ ] **Step 5: Run full Rust runtime tests and Clippy.**
- [ ] **Step 6: Commit** with `perf: reuse resident forward buffers`.

### Task 4: Verify KV residency and session state

**Files:**
- Test: `wpc-runtime/tests/resident_runtime.rs`
- Modify: `wpc-runtime/src/resident.rs` only if required
- Modify: `wpc-runtime/src/qwen3_moe_model.rs` only if required

**Interfaces:**
- Existing `ResidentSession` owns cache lifetime.
- Cache truncation returns to the prompt boundary after generation.

- [ ] **Step 1: Add tests for KV length before generation, during generation, and after truncation.**
- [ ] **Step 2: Add a test that repeated generation with the same prompt prefix does not re-run the prefix tokens.**
- [ ] **Step 3: Run the focused tests.**
- [ ] **Step 4: Commit** with `test: verify resident kv lifecycle`.

### Task 5: Add allocation/reuse benchmark

**Files:**
- Create: `wpc-runtime/benches/resident_runtime.rs`
- Modify: `wpc-runtime/Cargo.toml` only if benchmark registration is needed

**Interfaces:**
- Benchmark cold load vs resident repeated session calls.
- Benchmark reference forward path vs workspace-backed path on the same synthetic fixture.

- [ ] **Step 1: Add benchmark cases.**
- [ ] **Step 2: Run benchmarks and capture baseline numbers.**
- [ ] **Step 3: Verify the workspace-backed path is not slower beyond measurement noise; if it regresses, profile before proceeding.**
- [ ] **Step 4: Commit** with `bench: measure resident workspace reuse`.

### Task 6: Close Phase 1 roadmap gates

**Files:**
- Modify: `docs/AIONS-INTEGRATION-MAP.md`
- Modify: `docs/UNIFIED_STACK.md` if the implementation changes the documented runtime boundary

- [ ] **Step 1: Require full workspace build, test, fmt and Clippy to pass on the feature branch.**
- [ ] **Step 2: Require resident runtime and benchmark verification to pass.**
- [ ] **Step 3: Change Phase 1 resident-runtime and allocation-reuse checkboxes to `[x]` only after evidence is green.**
- [ ] **Step 4: Commit** with `docs: close Phase 1 resident runtime roadmap gates`.
- [ ] **Step 5: Open a PR from `feature/phase1-resident-runtime` to `integration/full-organism-v2` with the verification evidence.**
