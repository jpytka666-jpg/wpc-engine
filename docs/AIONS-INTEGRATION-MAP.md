# AIONS Integration Map

This document is the **top-level AIONS implementation roadmap**. `docs/UNIFIED_STACK.md` contains the detailed canonical runtime architecture; this file defines the module boundaries, build order and promotion gates.

The project is intentionally polyglot. Existing Rust components remain the stable reference implementation. New languages are added as acceleration or interface layers only where they improve measured speed, quality, stability, precision or hardware coverage without creating duplicate ownership.

## 1. Eight AIONS architectural modules

1. **WPC Runtime** — compressed model compilation and inference, resident runtime, batching, attention, MoE and sampling.
2. **Agents / Local CI** — deterministic diagnostics, repair loop, coding agent and verification orchestration.
3. **Memory / KV** — resident KV cache, memory management, compressed-KV research, CBMS persistence and retrieval outside the real-time token path.
4. **AIONS Studio** — desktop developer/system environment: editor, terminal, graph views, compiler, debugger, AI controls and observability.
5. **Memory / System Graph** — graph of code, memory, processes, agents, sessions and dependencies.
6. **AIONS Kernel** — Rust system layer: IPC, capabilities, scheduling, memory primitives and userspace-first drivers/services.
7. **Ghost Gate** — isolated network boundary for firewall, VPN, DNS and optional Tor routing.
8. **OS Integration** — packaging, boot, supervision, permissions, observability, deployment and release integration.

## 2. Historical repository mapping

Historical repositories are **source material until promoted through current interfaces and CI**.

- `wpc-engine` → active WPC/runtime and AIONS system centre.
- `aions-mcp-server` → MCP tools, CBMS access and automation capability layer.
- `aions-server-wiedzy` → knowledge, memory and historical AIONS consolidation.
- `mcp-integration-system` → integration/reference patterns; not a second production MCP authority.
- `fresh-start` → historical Studio/developer-environment source material.
- `polip-agi` → historical agent architecture and experiments.
- `super-system` → historical system integration experiments.

Do **not** merge historical repositories wholesale. Extract specific components behind stable interfaces, add tests, and promote only verified code.

## 3. Language and layer strategy

AIONS is deliberately **multi-language** because different layers have different optimisation targets.

| Layer | Preferred technology | Responsibility |
|---|---|---|
| AIONS Studio UI | TypeScript + React/Svelte | editor, dashboards, graph UI, terminal and developer tooling |
| Desktop/native shell | Tauri + Rust | windows, filesystem, process control, IPC and OS integration |
| Core/runtime | Rust | lifecycle, ownership, sessions, scheduler, security, stable APIs, runtime state and KV orchestration |
| WPC compiler/format | Rust | safetensors, quantisation, packing, validation and deterministic artifact generation |
| Reference compute path | Rust | correctness baseline and fallback implementation |
| Portable AI acceleration | Mojo | candidate batched GEMM, attention, dequantisation and SIMD-heavy kernels |
| NVIDIA production backend | CUDA C++ | CUDA kernels, Tensor Cores and CUDA libraries |
| GPU micro-optimisation | PTX | only measured bottlenecks that justify lower-level control |
| CPU micro-optimisation | C/C++ intrinsics + assembly | only measured hot paths where higher-level code is insufficient |
| Research/offline tooling | Python | experiments, analysis and benchmarks; not production hot path |

### Rule
**Do not rewrite working Rust merely to change language.** Rust remains the system-of-record and reference implementation. Acceleration layers sit behind explicit backend interfaces and are promoted only after benchmark and correctness evidence.

## 4. Development order

The roadmap is staged so that every stage leaves the previous layer usable and testable.

### Phase 0 — Foundation / CI

- [x] Unified Rust workspace exists.
- [x] Full-organism module layout exists.
- [x] CI verifies build, tests, formatting and Clippy.
- [x] Repair Agent verifies changes before persisting them.
- [x] Repair Agent explicitly dispatches post-repair verification workflows.
- [ ] Historical Git secret scan before declaring public repositories clean.

### Phase 1 — WPC Runtime

**Status: substantially implemented; now move from correctness baseline to production runtime.**

- [x] WPC compiler and on-disk formats.
- [x] v3/v4 quantisation and reconstruction paths.
- [x] Qwen/Gemma runtime support.
- [x] Attention, RoPE, norms, MoE routing and sampling.
- [x] Fused/reference kernel correctness tests.
- [ ] Long-lived resident runtime per agent task/session.
- [ ] Load compressed weights once and reuse allocations.

### Phase 2 — Agents / Local CI

**Status: implementation present; verification loop is being hardened.**

- [x] AIONS agent executable.
- [x] Dynamic MCP `initialize` + `tools/list` discovery.
- [x] Tool-call transcript loop.
- [x] Rust diagnostics and repair classification.
- [x] Automated Clippy/build/test repair workflow.
- [x] Verified repair-before-push gate.
- [ ] Broader automated recovery for non-Clippy integration failures.

### Phase 3 — Memory / KV

**Status: existing crates/tests; next major architectural milestone.**

- [x] Memory/KV crates and contract tests.
- [x] KV persistence and lifecycle contracts.
- [x] mmap-backed / structured memory work where already implemented.
- [ ] Resident KV cache across model turns.
- [ ] Reusable model state across MCP tool calls.
- [ ] Explicit hot-path versus CBMS persistence boundary.
- [ ] Measure memory growth, reuse and eviction behaviour.

### Phase 4 — Batched execution

**Status: next performance-critical implementation step.**

- [ ] BatchEngine as the primary prefill/forward path.
- [ ] Batched prompt prefill.
- [ ] Batched GEMM/attention.
- [ ] Benchmark single-token reference versus batch path.
- [ ] Preserve reference numerical results within documented tolerances.
- [ ] Add scheduling for variable batch sizes and sequence lengths.

### Phase 5 — Compute acceleration layers

Only after the Rust/reference path is correct and benchmarked:

1. **Mojo** — portable high-performance CPU/GPU kernel candidates.
2. **CUDA C++** — NVIDIA production backend.
3. **AVX2/AVX-512 / C++ intrinsics** — CPU specialisations.
4. **PTX / assembly** — surgical optimisation only where profiling proves it necessary.

All acceleration layers must satisfy the same correctness, precision, determinism, stability and performance gates.

### Phase 6 — AIONS Studio

- [ ] TypeScript UI framework and design system.
- [ ] Tauri + Rust desktop shell.
- [ ] Live runtime/KV/agent observability.
- [ ] Integrated terminal and task control.
- [ ] Code/editor/compiler/debugger integration.
- [ ] Memory and system graph views.

### Phase 7 — Memory / System Graph

- [ ] Unified graph model for code, memory, agents, processes and dependencies.
- [ ] Stable graph API from Rust core.
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
6. **Numerical/semantic equivalence** to the reference path within an explicit tolerance unless a new model/quantisation scheme is intentionally introduced.
7. **Full CI compatibility**: build, tests, formatting and Clippy.
8. **A practical fallback** to the reference implementation where feasible.
9. **Observability** for backend selection, timing, failures and resource use.

Performance is never accepted by silently sacrificing correctness, precision or stability.

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

Language boundaries should be **coarse-grained**. Submit batches/tensor views to compute backends; do not perform per-token Rust↔FFI chatter when a whole kernel/batch can cross the boundary.

## 7. Security rule

Public source is assumed observable. Secrets must never be committed. Credentials belong in environment variables, local secret stores or managed secret systems. Historical Git history requires a dedicated secret scan before a public-clean declaration.

## 8. Immediate priority after CI stabilises

**1. Resident WPC runtime → 2. Persistent KV across agent turns → 3. Batched prefill/forward → 4. Benchmark → 5. Mojo/CUDA/CPU acceleration → 6. Studio → 7. Graph → 8. Kernel → 9. Ghost Gate → 10. OS integration.**

The goal is one coherent AIONS system with a stable Rust core and specialised acceleration/interface layers, not one language forced onto every subsystem.
