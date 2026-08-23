# AIONS Organism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Build the complete modular AIONS organism on GitHub first, prove every subsystem independently, then prove cross-module integration before any local deployment.

**Architecture:** Eight bounded modules remain independently owned: WPC Runtime, Agents CI, Memory/KV, Studio, Memory Graph, AIONS Kernel, Ghost Gate, and OS Integration. Stage-2 branches extend their corresponding `arch/*` branches; they merge into `arch/*` only after module CI is green, and `arch/*` integrates into `main` only after the full integration gate is green.

**Tech Stack:** Rust workspace, JSON Schema contracts, GitHub Actions, GitHub-hosted isolated runners, deterministic fixtures, Redox-inspired capability boundaries, typed module manifests.

**Spec:** module contracts and `docs/IMPLEMENTATION-STAGE-1.md` files on each `arch/*` branch.

## Global Constraints

- GitHub-first: no local AIONS implementation until complete integration CI is green.
- Do not modify existing local AIONS workspaces during this project phase.
- Each module owns its state and exposes typed boundaries.
- Durable memory must not silently enter the generation-critical token path.
- Ghost Gate is the only network boundary; default is fail-closed.
- Kernel owns privileged mechanisms only; AIONS services and drivers remain userspace-first.
- Studio is a control surface and never owns subsystem state.
- Independent CI jobs run in isolated GitHub-hosted runners with `fail-fast: false` where parallelism is useful.
- Every implementation task has a deterministic verification step and a small commit.

---

### Task 1: Stage-2 CI gate

**Files:**
- Create/update: `.github/workflows/stage2-contract.yml` on each Stage-2 branch.
- Test: schema/contract validation steps for each module.

**Interfaces:** Stage-2 branches consume their parent `arch/*` branch and produce validated contract artifacts.

- [ ] Step 1: Add a reusable contract gate that validates required docs and JSON schemas.
- [ ] Step 2: Run schema parsing and deterministic fixture checks independently per module.
- [ ] Step 3: Run the gate on PRs targeting `arch/*` and on pushes to `stage2/*`.
- [ ] Step 4: Verify every Stage-2 PR gets a distinct run.
- [ ] Step 5: Commit and leave each PR Draft until green.

### Task 2: WPC Runtime minimal resident-load contract

**Files:**
- Modify: `modules/wpc-runtime/runtime-load.schema.json`.
- Create: `modules/wpc-runtime/resident-load.fixture.json`.
- Test: schema validation and fixture round-trip.

**Interfaces:** Produces stable resident-load metadata for model ID, WPC scheme, weight source and KV policy.

- [ ] Step 1: Validate the schema against the fixture.
- [ ] Step 2: Add explicit residency lifecycle values.
- [ ] Step 3: Verify incompatible empty/unknown combinations are rejected.
- [ ] Step 4: Commit.

### Task 3: Agents CI diagnostic core

**Files:**
- Modify: `modules/agents-ci/diagnostic.schema.json`.
- Create: deterministic diagnostic fixtures for compile/test/lint/benchmark failures.
- Test: parser/classifier contract checks.

**Interfaces:** Produces machine-readable diagnostic envelopes with exit code, category, bounded context and repair permission.

- [ ] Step 1: Add deterministic examples for each failure class.
- [ ] Step 2: Validate the schema and classification vocabulary.
- [ ] Step 3: Verify bounded repair permission semantics.
- [ ] Step 4: Commit.

### Task 4: Memory/KV lifecycle

**Files:**
- Modify: `modules/memory-kv/kv_handle.schema.json`.
- Modify: `modules/memory-kv/kv_roundtrip.fixture.json`.
- Test: lifecycle, sequence ownership and boundary cases.

**Interfaces:** Produces a typed KV handle and append/read lifecycle metadata.

- [ ] Step 1: Define explicit lifecycle states.
- [ ] Step 2: Add sequence ownership and token-range metadata.
- [ ] Step 3: Validate round-trip and invalid sequence cases.
- [ ] Step 4: Commit.

### Task 5: Memory Graph canonical model

**Files:**
- Modify: `modules/memory-graph/graph_snapshot.schema.json`.
- Create: deterministic graph fixture.
- Test: stable-ID, duplicate-ID and orphan-edge checks.

**Interfaces:** Produces canonical node/edge snapshots independent of UI.

- [ ] Step 1: Define stable node IDs and typed node kinds.
- [ ] Step 2: Define edge identity and endpoint rules.
- [ ] Step 3: Reject duplicate IDs and orphaned edges.
- [ ] Step 4: Commit.

### Task 6: Studio control model

**Files:**
- Modify: `modules/studio/command.schema.json`.
- Create: deterministic approval-state fixtures.
- Test: command approval and denial semantics.

**Interfaces:** Produces typed commands with explicit approval states and source provenance.

- [ ] Step 1: Define `pending`, `approved`, `denied`, `executed` states.
- [ ] Step 2: Define which command classes require confirmation.
- [ ] Step 3: Validate invalid state transitions.
- [ ] Step 4: Commit.

### Task 7: AIONS Kernel capability boundary

**Files:**
- Modify: `modules/aions-kernel/capability.schema.json`.
- Create: capability denial fixtures.
- Test: rights validation and isolation semantics.

**Interfaces:** Produces mechanism-level capabilities while keeping service logic in userspace.

- [ ] Step 1: Define a closed vocabulary for privileged rights.
- [ ] Step 2: Validate owner and device scoping.
- [ ] Step 3: Add denial fixtures for undeclared rights.
- [ ] Step 4: Commit.

### Task 8: Ghost Gate fail-closed policy

**Files:**
- Modify: `modules/ghost-gate/egress_request.schema.json`.
- Create: OFFLINE/VPN/TOR policy fixtures.
- Test: default-deny, DNS policy and invalid-mode cases.

**Interfaces:** Produces typed egress requests and policy decisions with auditable mode semantics.

- [ ] Step 1: Define mode and decision fields.
- [ ] Step 2: Make OFFLINE structurally deny-only.
- [ ] Step 3: Add DNS and audit metadata.
- [ ] Step 4: Validate forbidden direct-egress fixtures.
- [ ] Step 5: Commit.

### Task 9: OS Integration control plane

**Files:**
- Modify: `modules/os-integration/manifest.schema.json`.
- Modify: `modules/os-integration/stage2-manifest.json`.
- Create: health/lifecycle fixtures.
- Test: module registration and dependency ordering.

**Interfaces:** Consumes the seven subsystem manifests and produces reproducible orchestration metadata.

- [ ] Step 1: Validate all eight module identities and branch mappings.
- [ ] Step 2: Add health and lifecycle state fields.
- [ ] Step 3: Add dependency ordering validation.
- [ ] Step 4: Verify no subsystem implementation leaks into integration ownership.
- [ ] Step 5: Commit.

### Task 10: Integration gate

**Files:**
- Create: `.github/workflows/aions-integration.yml`.
- Create: `modules/os-integration/integration-fixture.json`.
- Test: all module gates, manifest consistency, schema compatibility.

**Interfaces:** Consumes green `arch/*` module gates and produces one integration verdict.

- [ ] Step 1: Collect module gate results without sharing mutable workspaces.
- [ ] Step 2: Validate OS Integration manifest against all module heads.
- [ ] Step 3: Run cross-module schema compatibility checks.
- [ ] Step 4: Add reproducible integration fixture.
- [ ] Step 5: Commit and require the integration gate for `main`.

### Task 11: Full system implementation

**Rule:** Execute only after Tasks 1-10 are green.

- [ ] Implement real WPC resident runtime path behind the accepted contract.
- [ ] Implement diagnostic/repair loop behind the accepted diagnostic schema.
- [ ] Implement KV lifecycle with CBMS adapter outside token-critical path.
- [ ] Implement Graph model and query API.
- [ ] Implement Studio headless command/event API before UI.
- [ ] Implement kernel capability/IPC primitives before drivers.
- [ ] Implement Ghost Gate VM integration after security gate.
- [ ] Implement OS lifecycle/orchestration and reproducible release manifest.
- [ ] Run complete system CI repeatedly until green.

### Task 12: Deployment gate

- [ ] Produce a versioned release artifact from GitHub.
- [ ] Verify full integration CI green on the release commit.
- [ ] Only then create the local deployment workspace.
- [ ] Run local smoke tests against the exact GitHub release artifact.
