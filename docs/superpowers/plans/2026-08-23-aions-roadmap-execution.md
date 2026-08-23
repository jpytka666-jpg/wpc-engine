# AIONS Roadmap Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the AIONS master roadmap in dependency order using GitHub as the only writable workspace, with every subsystem independently gated before integration.

**Architecture:** WPC remains the resident inference engine; Agents/Local CI owns diagnostics and bounded repair; KV/CBMS provide hot and persistent memory without putting persistent storage on the generation-critical path. Studio, Graph, Kernel, Ghost Gate, and OS Integration are integrated only after their Stage-2 contracts and CI gates are green.

**Tech Stack:** Rust, Cargo, GitHub Actions, JSON/JSON Schema, Python contract tests, WPC runtime, CBMS/KV interfaces, AIONS MCP, Ghost Gate VM boundary.

**Spec:** `AIONS_MASTER_ROADMAP.md` on the `docs/aions-master-roadmap` branch, plus the existing Stage-2 module contracts and PRs.

## Global Constraints

- GitHub is the only writable workspace for this execution phase.
- Local disks may be read for inspection only; no local writes, builds, generated files, or destructive actions.
- Do not merge a subsystem until its required CI/contract gate is green.
- Keep experimental KV compression outside the hot generation path until measured evidence is positive.
- Keep kernel responsibilities minimal and prefer userspace services.
- Ghost Gate is fail-closed and remains a separate network boundary.
- Preserve rollback points and auditable Git history.

---

### Task 1: Close Agents / Local CI Stage 2

**Files:**
- Create: `modules/agents-ci/diagnostic.py` on `stage2/agents-ci-diagnostics` (already added in commit `3993b6a`)
- Test: `modules/agents-ci/diagnostic_contract_test.py`
- CI: `.github/workflows/agents-ci-stage2.yml`

**Interfaces:**
- Consumes: diagnostic stage/exit-code inputs.
- Produces: `classify_failure(exit_code: int, stage: str, summary: str) -> tuple[str, bool>` and a green Stage-2 diagnostic gate.

- [ ] Verify PR #17 head contains commit `3993b6a`.
- [ ] Verify the fresh Agents CI run executes `python3 modules/agents-ci/diagnostic_contract_test.py` successfully.
- [ ] Verify Stage 2 Contract Gate remains green.
- [ ] Verify the repository-wide CI run remains green.
- [ ] Mark Task 3 in the execution tracker as GREEN only after all three checks pass.
- [ ] Merge PR #17 only after the gate is green and the PR remains mergeable.

### Task 2: Validate and integrate WPC resident runtime

**Files:**
- Review only: `stage2/wpc-runtime-contracts` and PR #18
- Review only: `integration/aions-unified-stack` and PR #2
- Test/CI: existing WPC runtime and resident-session workflows

**Interfaces:**
- Consumes: WPC resident-load contract linking model ID, scheme, weight source, and KV policy.
- Produces: resident runtime lifetime across multiple agent turns, with no per-turn model reload.

- [ ] Verify PR #18 contract tests are green.
- [ ] Compare PR #18 against its Stage-1 base and confirm it changes only the resident contract boundary.
- [ ] Validate PR #2 remains compatible with the new contract.
- [ ] Run/inspect the complete CI gate for resident execution.
- [ ] Merge the resident contract only after CI verification.
- [ ] Keep the older integration PR isolated until the current contract gate proves it can be safely integrated.

### Task 3: Implement Memory/KV substrate

**Files:**
- Review/modify only inside `stage2/memory-kv-contracts` and its implementation branch.
- Target contract: `arch/memory-kv`.

**Interfaces:**
- Produces a typed KV handle lifecycle and deterministic fixture.
- Persistent storage stays explicitly outside the generation-critical path.

- [ ] Verify PR #11 Stage-2 contract is green.
- [ ] Freeze canonical hot-KV lifecycle semantics before adding persistence.
- [ ] Add the minimum standalone implementation required by the contract, using TDD.
- [ ] Add warm/cold tier boundaries without making SSD/CBMS a token-generation dependency.
- [ ] Add KV replay/load benchmark coverage.
- [ ] Keep WPC-compressed KV experimental and unintegrated until measured evidence is positive.
- [ ] Merge only after standalone tests and CI are green.

### Task 4: Implement System/Memory Graph

**Files:**
- Target: `arch/memory-graph` and `stage2/memory-graph-contracts`.

**Interfaces:**
- Canonical snapshot with stable node IDs and typed edges.
- Later consumers: Studio, OS Integration, and operational tooling.

- [ ] Verify PR #12 contract gate.
- [ ] Implement entity/relationship storage behind the canonical snapshot contract.
- [ ] Add graph integrity tests.
- [ ] Add query/navigation tests for agents, processes, services, drivers, CBMS memories, KV objects, and WPC objects.
- [ ] Keep visualization out of the core graph data model.
- [ ] Merge only after independent CI is green.

### Task 5: Implement AIONS Studio

**Files:**
- Target: `arch/studio` and `stage2/studio-contracts`.

**Interfaces:**
- Stable control surface with explicit command approval states.
- Depends on runtime, memory, graph, and system contracts but should remain independently testable.

- [ ] Verify PR #15 Stage-2 command approval contract.
- [ ] Implement approval-state transitions and rejection/fail-closed behavior.
- [ ] Add tests for system/developer control boundaries.
- [ ] Keep the UI shell isolated from kernel internals.
- [ ] Add integration adapters only after the individual contracts pass.
- [ ] Merge only after CI is green.

### Task 6: Implement AIONS Kernel prototype

**Files:**
- Target: `arch/aions-kernel` and `stage2/aions-kernel-contracts`.

**Interfaces:**
- Explicit mechanism-level capability rights.
- Userspace owns service logic.
- IPC and service boundaries remain explicit.

- [ ] Verify PR #14 capability-rights contract.
- [ ] Implement the minimal kernel mechanism surface, not AIONS application logic.
- [ ] Add IPC/capability tests.
- [ ] Add minimal memory/process primitives only behind explicit interfaces.
- [ ] Keep storage, GPU, input, and network logic in userspace services where practical.
- [ ] Add a bootable minimal image only after the contract tests are stable.
- [ ] Merge only after independent CI is green.

### Task 7: Implement Ghost Gate

**Files:**
- Target: `arch/ghost-gate` and `stage2/ghost-gate-contracts`.

**Interfaces:**
- Typed egress request with explicit mode, decision, and DNS policy.
- OFFLINE is structurally deny-only.

- [ ] Verify PR #13 fail-closed contract.
- [ ] Implement the minimal Debian/Linux VM boundary.
- [ ] Add default-deny firewall behavior.
- [ ] Add DNS policy/filtering.
- [ ] Add explicit VPN mode.
- [ ] Add optional Tor mode without treating anonymity as a guarantee.
- [ ] Add audit/observability events.
- [ ] Test offline/fail-closed behavior.
- [ ] Merge only after independent CI is green.

### Task 8: AIONS OS Integration

**Files:**
- Target: `arch/os-integration` and `stage2/os-integration-contracts`.
- Consumers: resident WPC, Agents/CI, KV/CBMS, Graph, Studio, Kernel, Ghost Gate.

**Interfaces:**
- Canonical module manifest covering version, CI, health, lifecycle, permissions, boot/recovery, and reproducible release state.

- [ ] Verify PR #16 manifest contract.
- [ ] Define service startup ordering from kernel boundary upward.
- [ ] Start resident WPC as a managed service.
- [ ] Start CBMS/memory services.
- [ ] Start graph subsystem.
- [ ] Start Studio.
- [ ] Connect Agents/Local CI.
- [ ] Connect Ghost Gate.
- [ ] Add health/lifecycle checks for every module.
- [ ] Run end-to-end smoke tests.
- [ ] Produce a reproducible system build/image definition.
- [ ] Merge only after all subsystem gates are green.

### Task 9: Full integration gate

**Files:**
- Modify only integration CI/manifests created by the OS Integration stage.
- No subsystem implementation changes unless a failing integration contract identifies an actual boundary defect.

**Interfaces:**
- Single integration gate over all accepted subsystem contracts.

- [ ] Verify all subsystem branches/PRs used by the integration branch are green.
- [ ] Add a full-stack smoke workflow.
- [ ] Test resident WPC over multiple agent turns.
- [ ] Test tool discovery and tool calls through AIONS MCP.
- [ ] Test hot KV lifecycle and persistent memory separation.
- [ ] Test graph updates from live subsystem events.
- [ ] Test Studio control approvals.
- [ ] Test kernel capability and IPC boundaries.
- [ ] Test Ghost Gate default-deny and offline mode.
- [ ] Fail the integration gate on any missing health/lifecycle contract.
- [ ] Merge only when the complete gate is green.

### Task 10: Full organism milestone

**Files:**
- Integration tests and release documentation only.

**Interfaces:**
- AIONS behaves as one reproducible organism while preserving subsystem boundaries.

- [ ] Verify M8 milestone criteria from `AIONS_MASTER_ROADMAP.md`.
- [ ] Run clean end-to-end bootstrap from the reproducible artifact.
- [ ] Verify recovery/restart behavior for resident WPC and managed services.
- [ ] Verify graph state reflects operational services and AI memory state.
- [ ] Verify Studio can inspect and control the system without bypassing capability boundaries.
- [ ] Record benchmark and verification evidence.
- [ ] Tag the accepted integration point.

### Task 11: Local deployment

**Files:**
- GitHub deployment/release definitions only during this phase.
- Local machine remains read-only.

**Interfaces:**
- Reproducible deployment artifact for later local installation.

- [ ] Generate the deployment artifact in GitHub Actions.
- [ ] Verify checksums and release metadata.
- [ ] Verify deployment instructions are complete and reproducible.
- [ ] Do not install or mutate local disks in this phase.
- [ ] Treat local installation as a separate future change that explicitly lifts the read-only constraint.

## Execution Rules

1. Work one task at a time in the order above.
2. Never skip a failing gate by advancing downstream.
3. Prefer fixing the smallest boundary defect rather than broad refactors.
4. Every implementation change gets a failing test first, then minimal code, then green verification.
5. Every task ends with a GitHub commit and a visible CI result.
6. If a task exposes a cross-subsystem design defect, stop that task and update the relevant contract before implementing more code.
