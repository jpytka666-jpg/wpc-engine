# AIONS OS — Master Build Plan

**Status:** ACTIVE — canonical architecture and sequencing document
**Date:** 2026-08-23
**Owner:** Marcin Szul
**Repository:** `jpytka666-jpg/wpc-engine`

## Mission

Build AIONS as an offline-first AI operating environment with a small Rust kernel, userspace system services, isolated network boundary, persistent memory substrate, resident WPC inference runtime, autonomous agent/tool platform, native Studio interface, live system/knowledge graph, and a final cross-layer integration image.

## Non-negotiable architecture rules

1. Keep the eight architectural lanes isolated until their contracts and gates pass.
2. Prefer stable interfaces/adapters over copying historical repositories into `wpc-engine`.
3. Keep the kernel, KV, Ghost Gate and experimental system work out of production integration until independently verified.
4. WPC Engine is the current Rust engineering core, not the final OS repository layout.
5. Reusable MCP/tool capabilities belong behind stable AIONS service interfaces.
6. Cloud services remain optional; the core OS must remain useful offline.
7. Every major lane owns its branch and CI gate.
8. No milestone is complete from code existence alone: contract, build, tests, functional verification and security/performance gates must pass.
9. Local disks are read-only for engineering work in this workflow; GitHub is the source of truth for changes.
10. Never disable Clippy or tests merely to obtain a green CI result.

## Architectural layers

### Layer 0 — AIONS Kernel
Branch: `arch/aions-kernel`

Rust/Redox-inspired microkernel boundary: boot, address spaces, memory management, scheduler, IPC, capability/security model, interrupts and minimal hardware-facing primitives. Drivers remain userspace wherever practical.

**Exit gate:** boots a minimal image, starts userspace init, passes memory/IPC/scheduler/capability tests, and exposes stable service interfaces.

### Layer 1 — System Services
Branch: `arch/os-integration`

Userspace drivers, storage, process services, graphics, input and system APIs. Kernel dependencies remain narrow and explicit.

**Exit gate:** services boot under the kernel, IPC failures are contained, storage/process/graphics/input smoke tests pass.

### Layer 2 — Ghost Gate
Branch: `arch/ghost-gate`

Isolated VM providing the network boundary: firewall, VPN, DNS policy and optional Tor routing. AIONS talks to a narrow gateway API rather than owning the physical network path.

**Exit gate:** default-deny boundary, deterministic gateway API, network-policy tests, isolation tests and recovery tests pass.

### Layer 3 — Memory Substrate
Branch: `arch/memory-kv`

Hot KV cache, warm/compressed KV, CBMS, memory indexes, event/history storage and memory graph interfaces. KV stays on the real-time token path; CBMS remains persistent/warm storage outside that hot path.

**Exit gate:** KV correctness, sequence/gate tests, persistence/recovery, bounded latency, and explicit hot/warm/cold data movement tests pass.

### Layer 4 — AI Runtime
Branch: `arch/wpc-runtime`

WPC compiler/format/runtime, resident model runtime, attention/KV execution, model loading, inference services and batching.

Current target order:
`Clippy clean → full integration tests → resident runtime → batched forward → persistent KV reuse → speculative verification → expert-grouped execution`.

**Exit gate:** supported model loads, deterministic correctness tests pass, resident runtime survives multi-turn sessions, batch path is measured, and performance regression gates pass.

### Layer 5 — Agent / Tool Platform
Branch: `arch/agents-ci`

AIONS agent, MCP registry/discovery, tool execution, router/scheduler, workflow orchestration, browser/desktop automation, project scanner, Git integration and Local CI repair loop.

**Exit gate:** live `tools/list`, approval mode, tool-call execution, transcript continuity, failure recovery and authenticated GitHub repair cycle pass.

### Layer 6 — AIONS Studio / Interface
Branch: `arch/studio`

Native system shell/IDE: editor, terminal, compiler, debugger, AI agent, memory view, system controls, Git and visual system graph.

**Exit gate:** Studio boots as the primary shell, can inspect/edit/build/debug a project, call agents/tools, inspect memory and control system services through stable APIs.

### Layer 7 — Graph / System View
Branch: `arch/memory-graph`

Interactive graph of projects, code dependencies, processes, memory, agents, services and system relationships.

**Exit gate:** graph updates from live events, links entities back to source/service identifiers, and remains usable under realistic system activity.

## Cross-layer integration

Branch: `arch/os-integration`

Integration order:

1. Kernel + system-service ABI
2. System services + Ghost Gate API
3. Memory substrate + AI runtime API
4. Agent platform + runtime + MCP services
5. Studio + all service APIs
6. Graph + event sources
7. Boot/persistence/recovery/security integration
8. Full offline end-to-end scenario

**Final gate:** one bootable AIONS image reaches Studio, starts the agent, loads the resident WPC runtime, uses hot KV, persists warm memory through CBMS, communicates through Ghost Gate, renders the live graph, and passes end-to-end recovery/security tests.

## Development sequence

### Milestone 1 — Stabilize current Rust core
- Resolve remaining runtime Clippy findings.
- Run complete workspace build/test/format/Clippy.
- Run fused-kernel correctness and attention tests.
- Run benchmark compile/smoke.
- Verify no accidental parser corruption or unrelated file rewrites.

### Milestone 2 — Unified WPC + AIONS organism
- Keep `integration/full-organism-v2` green.
- Confirm WPC runtime and AIONS agent contracts.
- Verify MCP discovery and tool execution.
- Verify Local CI repair loop without false-positive success.

### Milestone 3 — Resident runtime and batched execution
- Make WPC runtime long-lived across an agent session.
- Add persistent model state and reusable KV cache.
- Implement batched forward execution.
- Measure prefill, decode, memory traffic and latency.
- Only then implement speculative verification and expert-grouped scheduling.

### Milestone 4 — Memory substrate
- Stabilize hot KV.
- Add warm/compressed KV.
- Integrate CBMS persistence and recovery.
- Add memory graph events and indexes.

### Milestone 5 — Kernel + system services
- Implement minimal bootable kernel.
- Establish capability/IPC ABI.
- Start userspace init and drivers.
- Add storage/process/graphics/input services.

### Milestone 6 — Ghost Gate
- Produce isolated gateway VM.
- Enforce network policy outside the core OS.
- Expose a minimal authenticated gateway API.
- Test offline operation when the gateway is unavailable.

### Milestone 7 — Studio + graph
- Build native Studio shell.
- Connect editor/build/debugger to agent tooling.
- Add memory/system views.
- Integrate live graph and service controls.

### Milestone 8 — Full OS integration
- Integrate all stable interfaces only after their lane gates pass.
- Build a bootable image.
- Run offline end-to-end agent task.
- Run failure injection, recovery and security tests.
- Establish release/recovery artifacts.

## Verification doctrine

For every milestone:

`contract → failing test → minimal implementation → focused test → integration test → functional test → performance/security check → CI gate → review → merge`

A green agent report is not evidence. Git diff, CI output and functional tests are evidence.

## Current position — 2026-08-23

`integration/full-organism-v2` is the active integration workstream. Build and workspace tests are passing; remaining work is dominated by runtime Clippy cleanup and final validation. The architecture is intentionally not yet merged into the kernel, Ghost Gate, Studio or final OS integration lanes.

The next architectural objective after current validation is **resident WPC runtime + batched forward execution**, while the other lanes continue independently behind their contracts.

## Source architecture

The canonical layer/branch mapping is also recorded in `docs/AIONS-INTEGRATION-MAP.md`. Historical repositories remain sources for reusable components and ideas; they are not to be blindly merged into the Rust core.
