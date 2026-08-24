# AIONS Integration Map

This document is the **top-level AIONS implementation roadmap**. `docs/UNIFIED_STACK.md` contains the detailed runtime architecture; this file defines the module boundaries, development order, milestones, and promotion gates.

The project is intentionally polyglot. Existing Rust components remain the stable reference implementation. New languages are added only as interface or acceleration layers when they provide measured gains in speed, quality, stability, precision, or hardware coverage without creating duplicate ownership.

## 1. Eight AIONS architectural modules

1. **WPC Runtime** — compressed model compilation and inference, resident runtime, batching, attention, MoE, and sampling.
2. **Agents / Local CI** — deterministic diagnostics, repair loop, coding agent, and verification orchestration.
3. **Memory / KV** — resident KV cache, memory management, compressed-KV research, CBMS persistence, and retrieval outside the real-time token path.
4. **AIONS Studio** — desktop developer/system environment: editor, terminal, graph views, compiler, debugger, AI controls, and observability.
5. **Memory / System Graph** — graph of code, memory, processes, agents, sessions, and dependencies.
6. **AIONS Kernel** — Rust system layer: IPC, capabilities, scheduling, memory primitives, and userspace-first drivers/services.
7. **Ghost Gate** — isolated network boundary for firewall, VPN, DNS, and optional Tor routing.
8. **OS Integration** — packaging, boot, supervision, permissions, observability, deployment, and release integration.

## 2. Historical repository mapping

Historical repositories are **source material until promoted through current interfaces and CI**.

- `wpc-engine` → active WPC/runtime and AIONS system centre.
- `aions-mcp-server` → MCP tools, CBMS access, and automation capability layer.
- `aions-server-wiedzy` → knowledge, memory, and historical AIONS consolidation.
- `mcp-integration-system` → integration/reference patterns; not a second production MCP authority.
- `fresh-start` → historical Studio/developer-environment source material.
- `polip-agi` → historical agent architecture and experiments.
- `super-system` → historical system integration experiments.

Do **not** merge historical repositories wholesale. Extract specific components behind stable interfaces, add tests, and promote only verified code.

## 3. Language and layer strategy

AIONS is deliberately **multi-language** because different layers have different optimisation targets.

| Layer | Preferred technology | Responsibility |
|---|---|---|
| AIONS Studio UI | TypeScript + React/Svelte | editor, dashboards, graph UI, terminal, and developer tooling |
| Desktop/native shell | Tauri + Rust | windows, filesystem, process control, IPC, and OS integration |
| Core/runtime | Rust | lifecycle, ownership, sessions, scheduler, security, stable APIs, runtime state, and KV orchestration |
| WPC compiler/format | Rust | safetensors, quantisation, packing, validation, and deterministic artifact generation |
| Reference compute path | Rust | correctness baseline and fallback implementation |
| Portable AI acceleration | Mojo | candidate batched GEMM, attention, dequantisation, and SIMD-heavy kernels |
| NVIDIA production backend | CUDA C++ | CUDA kernels, Tensor Cores, and CUDA libraries |
| GPU micro-optimisation | PTX | only measured bottlenecks that justify lower-level control |
| CPU micro-optimisation | C/C++ intrinsics + assembly | only measured hot paths where higher-level code is insufficient |
| Research/offline tooling | Python | experiments, analysis, and benchmarks; not the production hot path |

### Core rule

**Do not rewrite working Rust merely to change language.** Rust remains the system-of-record and reference implementation. Acceleration layers sit behind explicit backend interfaces and are promoted only after benchmark and correctness evidence.

## 4. Development order

Every phase must leave the previous phase usable and testable.

### Phase 0 — Foundation / CI

- [x] Unified Rust workspace exists.
- [x] Full-organism module layout exists.
- [x] CI verifies build, tests, formatting, and Clippy.
- [x] Repair Agent verifies changes before persisting them.
- [x] Repair Agent explicitly dispatches post-repair verification workflows.
- [ ] Historical Git secret scan before declaring public repositories clean.

### Phase 1 — WPC Runtime

**Status: COMPLETE — resident runtime and load-once storage are implemented and verified by PR #23 CI.**

- [x] WPC compiler and on-disk formats.
- [x] v3/v4 quantisation and reconstruction paths.
- [x] Qwen/Gemma runtime support.
- [x] Attention, RoPE, norms, MoE routing, and sampling.
- [x] Fused/reference kernel correctness tests.
- [x] Long-lived resident runtime per agent task/session.
- [x] Load compressed weights once and reuse allocations at the model/KV storage layer.

**Phase 1 implementation evidence:** `aions-agent` creates one `ResidentEngine` and one `ResidentSession` outside the agent-turn loop; WPC v4 model data is mmap-backed and shared through `Arc`; prompt prefixes and hot KV state are retained across turns; reusable mmap/KV capacity is covered by regression tests. Fine-grained scratch-buffer reuse inside `forward_token` remains a separate profiling-driven optimisation task and is not a Phase 1 correctness gate.

### Phase 2 — Agents / Local CI

**Status: implementation present; verification loop is now the protected path.**

- [x] AIONS agent executable.
- [x] Dynamic MCP `initialize` + `tools/list` discovery.
- [x] Tool-call transcript loop.
- [x] Rust diagnostics and repair classification.
- [x] Automated Clippy/build/test repair workflow.
- [x] Verified repair-before-push gate.
- [ ] Broader automated recovery for non-Clippy integration failures.

### Phase 3 — Memory / KV

**Status: active — contracts and WPC-KV compression gate are implemented; persistence/integration work is next.**

- [x] Memory/KV crates and contract tests.
- [x] KV persistence and lifecycle contracts.
- [x] mmap-backed / structured memory work where already implemented.
- [x] WPC vector-KV compression gate using the existing WPC PatternDict + ResidualDict engine.
- [x] WPC-KV reconstruction/attention correctness gate on deterministic synthetic tensors.
- [x] WPC-KV compression gate CI is independently runnable on the feature branch.
- [ ] Promote auditable WPC-KV metrics to the canonical CI evidence record.
- [ ] Validate WPC-KV against production attention tensors and real model distributions.
- [ ] Resident KV cache across the broader Memory/KV subsystem.
- [ ] Reusable model state across MCP tool calls.
- [ ] Explicit hot-path versus CBMS persistence boundary.
- [ ] Measure memory growth, reuse, eviction, and cache locality.

**WPC-KV checkpoint:** PR #21 remains open/Draft intentionally. It proves the WPC-derived vector-KV path and its correctness/compression gate, but it is **not generation-critical** and is not yet promoted into the resident hot KV path. The next Phase 3 work must build persistence/integration around the verified reference path rather than silently replacing it.

### Phase 4 — Batched execution

**Status: next performance-critical implementation step after the remaining Phase 3 gates.**

- [ ] BatchEngine as the primary prefill/forward path.
- [ ] Batched prompt prefill.
- [ ] Batched GEMM/attention.
- [ ] Benchmark the single-token reference against the batch path.
- [ ] Preserve reference numerical results within documented tolerances.
- [ ] Add scheduling for variable batch sizes and sequence lengths.

### Phase 5 — Compute acceleration layers

Only after the Rust/reference path is correct and benchmarked:

1. **Mojo** — portable high-performance CPU/GPU kernel candidates.
2. **CUDA C++** — NVIDIA production backend.
3. **AVX2/AVX-512 / C++ intrinsics** — CPU specialisations.
4. **PTX / assembly** — surgical optimisation only where profiling proves it necessary.

All acceleration layers must satisfy the same correctness, precision, determinism, stability, and performance gates.

### Phase 6 — AIONS Studio

- [ ] TypeScript UI framework and design system.
- [ ] Tauri + Rust desktop shell.
- [ ] Live runtime/KV/agent observability.
- [ ] Integrated terminal and task control.
- [ ] Code/editor/compiler/debugger integration.
- [ ] Memory and system graph views.

### Phase 7 — Memory / System Graph

- [ ] Unified graph model for code, memory, agents, processes, and dependencies.
- [ ] Stable graph API from the Rust core.
- [ ] Interactive Studio visualisation.
- [ ] Snapshot/restore and provenance.

### Phase 8 — AIONS Kernel

- [ ] Stable Rust kernel contracts.
- [ ] IPC and capability model.
- [ ] Scheduling and memory primitives.
- [ ] Userspace-first services/drivers.
- [ ] Hardware-facing interfaces only after contract tests exist.

### Phase 9 — Ghost Gate

- [ ] Isolated network boundary.
- [ ] Firewall/VPN/DNS policy engine.
- [ ] Optional Tor routing.
- [ ] Fail-closed tests and observability.

### Phase 10 — OS Integration

- [ ] Packaging and artifact layout.
- [ ] Boot and service supervision.
- [ ] Permissions and capability enforcement.
- [ ] Release/upgrade/rollback path.
- [ ] End-to-end system health and observability.

## 5. Promotion gates

A component is promoted into the active stack only when it has:

1. **One owner** and no duplicate production implementation.
2. **A documented interface** between layers.
3. **Automated correctness tests** or a justified smoke test.
4. **Deterministic behaviour** where promised.
5. **Measured performance evidence** for any acceleration claim.
6. **Numerical/semantic equivalence** to the reference path within an explicit tolerance unless a new model or quantisation scheme is intentionally introduced.
7. **Full CI compatibility**: build, tests, formatting, and Clippy.
8. **A practical fallback** to the reference implementation where feasible.
9. **Observability** for backend selection, timing, failures, and resource use.

Performance is never accepted by silently sacrificing correctness, precision, or stability.

## 6. Target execution architecture

```text
AIONS Studio
 TypeScript + React/Svelte
        |
        v
Tauri desktop shell
        |
        v
AIONS Core / Runtime
 Rust
 sessions | scheduler | agents | IPC | security | KV
        |
   +----+-------------------+
   |                        |
   v                        v
AIONS MCP               Compute backends
 tools / CBMS           Mojo / CUDA / C++ / ASM
   |                        |
   +------------+-----------+
                v
       Resident WPC Runtime
       weights + batching + attention + MoE + sampling
                |
         +------+------+
         |             |
        CPU           GPU
      AVX2/AVX512   CUDA / PTX
```

Language boundaries should be **coarse-grained**. Submit batches and tensor views to compute backends; do not perform per-token Rust↔FFI chatter when a whole kernel or batch can cross the boundary.

## 7. Security rule

Public source is assumed observable. Secrets must never be committed. Credentials belong in environment variables, local secret stores, or managed secret systems. Historical Git history requires a dedicated secret scan before a public-clean declaration.

## 8. Immediate priority after Phase 1

**1. Memory/KV contract and persistence integration → 2. WPC-KV validation on real model-derived tensors → 3. Resident/persistent KV boundary and reuse/eviction metrics → 4. Batched prefill/forward → 5. Compute acceleration → 6. Studio → 7. Graph → 8. Kernel → 9. Ghost Gate → 10. OS integration.**

The goal is one coherent AIONS system with a stable Rust core and specialised acceleration and interface layers, not one language forced onto every subsystem.
