# AIONS OS Full Build Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current WPC/AIONS Rust core and the seven remaining architectural lanes into a bootable, offline-first AIONS operating environment with stable cross-layer interfaces and verifiable end-to-end behavior.

**Architecture:** Maintain eight independent architectural lanes with explicit contracts. Integrate upward only after each lane passes its own build, correctness, functional, performance and security gates. Use `arch/os-integration` only for final wiring, not as a development dumping ground.

**Tech Stack:** Rust, Cargo, GitHub Actions, WPC v4 runtime, MCP, CBMS, hot/warm KV, Rust kernel, isolated gateway VM, native AIONS Studio, live graph/event bus.

**Spec:** `AIONS_MASTER_BUILD_PLAN.md`

## Global Constraints

- GitHub is the engineering source of truth; local disks are read-only.
- Keep the eight architectural branches isolated until their contracts and CI gates pass.
- Do not disable Clippy or tests to obtain green CI.
- Keep KV on the real-time token path; CBMS remains persistent/warm storage.
- Keep Ghost Gate outside the core runtime as an isolated network boundary.
- Keep cloud-specific services optional; the OS must remain useful offline.
- Every completion claim requires fresh verification evidence.

---

### Task 1: Stabilize current unified runtime

**Files:**
- Modify: runtime files identified by the current `cargo clippy` output only.
- Test: existing workspace and runtime correctness tests.

**Interfaces:** Preserve existing `Linear`, `EmbeddingTable`, attention/KV and agent interfaces.

- [ ] Run `cargo build --workspace --all-targets` on the current integration branch.
- [ ] Run `cargo test --workspace` and record every failed test.
- [ ] Run the repository's configured full Clippy command and record every error.
- [ ] Fix only the reported runtime lint/compile issues without changing semantics.
- [ ] Run `cargo fmt --all` and confirm a clean diff.
- [ ] Re-run build, tests and Clippy from a clean checkout.
- [ ] Run fused-kernel and attention correctness tests, including the existing 7/7 attention correctness suite.
- [ ] Commit only validated fixes with one focused commit.

### Task 2: Close unified WPC/AIONS organism gate

**Files:**
- Modify: `.github/workflows/*` only when a workflow contract is demonstrably wrong.
- Test: integration/full-organism test suites and agent diagnostics.

**Interfaces:** WPC runtime ↔ AIONS agent ↔ MCP service boundaries remain stable.

- [ ] Run `integration/full-organism-v2` CI after Task 1.
- [ ] Verify Build, Test, Format, Clippy and benchmark smoke all execute on the same commit lineage.
- [ ] Verify no formatter-generated commit is being skipped by push-trigger behavior without an explicit follow-up trigger.
- [ ] Confirm the Full Organism Gate consumes the actual tested commit.
- [ ] Confirm `aions-agent` diagnostics remain green.
- [ ] Commit workflow corrections separately from code fixes.

### Task 3: Resident WPC runtime

**Files:**
- Modify: `wpc-runtime` runtime launcher/session code.
- Test: resident-session tests and multi-turn agent smoke tests.

**Interfaces:** Introduce an explicit long-lived runtime session API that owns model state and exposes inference calls without reloading weights between turns.

- [ ] Write a failing test proving two agent turns currently rebuild state.
- [ ] Add a resident runtime session object that owns loaded model resources.
- [ ] Route successive agent turns through the same resident session.
- [ ] Verify model weights remain resident across turns.
- [ ] Verify prompt/output correctness remains unchanged.
- [ ] Add a regression test for session teardown and restart.
- [ ] Measure load overhead before and after.
- [ ] Run full workspace verification and commit.

### Task 4: Batched forward execution

**Files:**
- Modify: `wpc-runtime` attention/forward path.
- Test: batch-vs-token reference tests and benchmark harness.

**Interfaces:** Add a batch forward API that preserves the single-token reference semantics and returns one output state per input token/sequence position.

- [ ] Write reference tests comparing batch size 1 against the current scalar path.
- [ ] Extend tests to batch size 2, 4 and 8 with identical token sequences.
- [ ] Implement the minimal batched attention/MLP path.
- [ ] Validate masks, GQA, RoPE, residuals and sampling-visible logits.
- [ ] Measure prompt prefill throughput and memory traffic.
- [ ] Reject any result that changes token outputs unexpectedly.
- [ ] Commit only after functional and performance checks pass.

### Task 5: Persistent hot KV and warm-memory substrate

**Files:**
- Modify: `modules/memory-kv` and `arch/memory-kv` implementation files.
- Test: existing sequence-gate suites plus new persistence/recovery tests.

**Interfaces:** Define hot-KV API for active generation and a separate warm/CBMS API for persistence.

- [ ] Lock hot-KV ownership and lifetime semantics.
- [ ] Add sequence append/read/truncate tests.
- [ ] Add snapshot serialization tests for warm storage.
- [ ] Add recovery tests after process restart.
- [ ] Verify CBMS is never placed on the per-token hot path.
- [ ] Measure hit/miss latency and memory footprint.
- [ ] Pass every memory-KV sequence gate before exposing the interface to the runtime.

### Task 6: Kernel foundation

**Files:**
- Modify: `arch/aions-kernel` only on its own branch.
- Test: kernel boot, memory, IPC, scheduler and capability tests.

**Interfaces:** Publish a minimal userspace ABI for processes, memory, IPC and capabilities.

- [ ] Define kernel/userspace contract tests before implementation.
- [ ] Implement minimal boot path.
- [ ] Implement page/address-space management.
- [ ] Implement scheduler and process lifecycle.
- [ ] Implement capability-based IPC.
- [ ] Start a minimal userspace init process.
- [ ] Add deterministic fault and recovery tests.
- [ ] Produce a bootable development image and gate it independently.

### Task 7: Userspace system services

**Files:**
- Modify: userspace service trees on `arch/os-integration` or the dedicated service branch.
- Test: service startup, IPC, storage, graphics and input smoke suites.

**Interfaces:** Consume only the published kernel ABI.

- [ ] Implement process service.
- [ ] Implement storage/filesystem service.
- [ ] Implement graphics/input service boundaries.
- [ ] Implement system API service.
- [ ] Run service startup under the real kernel image.
- [ ] Inject service failure and verify isolation/restart behavior.

### Task 8: Ghost Gate

**Files:**
- Modify: `arch/ghost-gate`.
- Test: gateway policy, isolation, routing and recovery suites.

**Interfaces:** Expose a narrow authenticated gateway API to AIONS userspace; keep physical network details behind the gateway.

- [ ] Define default-deny gateway contract.
- [ ] Implement isolated gateway VM.
- [ ] Implement firewall policy.
- [ ] Implement VPN and DNS policy hooks.
- [ ] Add optional Tor route without making it mandatory for normal offline operation.
- [ ] Test AIONS behavior when the gateway disappears.
- [ ] Test that unauthorized network paths are blocked.

### Task 9: AIONS Studio and system interface

**Files:**
- Modify: `arch/studio`.
- Test: shell startup, editor/build/debugger, agent and system-control integration tests.

**Interfaces:** Studio consumes stable service, agent, memory and graph APIs rather than internal kernel/runtime structs.

- [ ] Define Studio service contracts.
- [ ] Implement primary shell/window lifecycle.
- [ ] Add editor and terminal.
- [ ] Add compiler/test/debugger controls.
- [ ] Add AI agent panel and approval flow.
- [ ] Add memory/KV inspection view.
- [ ] Add system service controls.
- [ ] Add Git integration.
- [ ] Verify Studio can perform a complete build/debug/agent task offline.

### Task 10: Live system and memory graph

**Files:**
- Modify: `arch/memory-graph`.
- Test: event ingestion, graph consistency and load tests.

**Interfaces:** Event-based graph API for projects, dependencies, processes, memory, agents and services.

- [ ] Define stable node/edge identifiers.
- [ ] Connect project/code dependency events.
- [ ] Connect process/service lifecycle events.
- [ ] Connect agent/MCP events.
- [ ] Connect CBMS/memory events without copying hot KV on every token.
- [ ] Add live update tests and bounded-memory tests.
- [ ] Integrate graph view with Studio.

### Task 11: Final OS integration

**Files:**
- Modify: `arch/os-integration` integration wiring only.
- Test: full boot-to-agent end-to-end suite.

**Interfaces:** Only stable published interfaces from Tasks 5–10.

- [ ] Verify every lane's own CI gate is green before merging it.
- [ ] Integrate kernel/userspace ABI.
- [ ] Integrate Ghost Gate API.
- [ ] Integrate memory substrate with resident runtime.
- [ ] Integrate agent/tool platform.
- [ ] Integrate Studio and graph.
- [ ] Build a bootable AIONS image.
- [ ] Run offline end-to-end agent task from Studio.
- [ ] Run failure injection and recovery tests.
- [ ] Run security/isolation tests.
- [ ] Run performance regression suite.
- [ ] Create a release candidate only after every gate is green.

## Review Gates

After every task: inspect the Git diff, run the focused tests, then run the relevant lane CI. Do not merge a task merely because an agent reports success.

## Current starting point

The active `integration/full-organism-v2` line is in runtime/agent validation. The immediate blocker is runtime Clippy; the next architectural milestone after stabilisation is resident WPC execution and batched forward. Kernel, Ghost Gate, memory, Studio and graph remain independent lanes until their own contracts and gates pass.
