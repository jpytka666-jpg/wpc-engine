# AIONS + WPC Unified Stack

This document is the **canonical architecture and implementation roadmap** for the current project. The repositories are **not** merged blindly: each keeps one responsibility and WPC Engine remains the inference/runtime centre.

## 1. System boundary

```text
                              USER / TASK
                                   |
                                   v
                    +----------------------------+
                    |      AIONS STUDIO           |
                    | TypeScript + React/Svelte   |
                    | Tauri desktop shell        |
                    +-------------+--------------+
                                  |
                                  v
                    +----------------------------+
                    |       AIONS CORE            |
                    |            Rust              |
                    | sessions / scheduler        |
                    | agents / IPC / security    |
                    | memory / KV / runtime state |
                    +-------------+--------------+
                                  |
                    +-------------+--------------+
                    |                            |
                    v                            v
          +------------------+          +------------------+
          |  AIONS MCP       |          |  COMPUTE BACKEND |
          | tools / CBMS     |          | Mojo / CUDA /    |
          | filesystem       |          | C++ / ASM        |
          | browser / Docker |          +---------+--------+
          +------------------+                    |
                                                  v
                                   +---------------------------+
                                   | WPC RUNTIME / MODEL CORE  |
                                   | resident weights + KV     |
                                   | batching / attention      |
                                   | MoE / sampling            |
                                   +-------------+--------------+
                                                 |
                                                 v
                                      CPU / AVX2 / AVX-512
                                      NVIDIA CUDA / PTX

Historical repositories remain source material, not hidden runtime dependencies.
```

## 2. Canonical responsibilities

### `wpc-engine` — runtime, model and system centre

This is the active centre of the stack.

- WPC compilation and on-disk model format
- compressed model execution
- Qwen/Gemma architecture support
- attention, RoPE, norms, MoE routing and sampling
- long-lived runtime/session state
- resident KV cache and memory management
- AIONS agent executable
- correctness, determinism and performance tests

Rust remains the **system-of-record implementation** for runtime state, ownership, lifecycle, scheduling, IPC and safety. Existing Rust code is not to be rewritten merely to change language; new layers are added only where measurement shows a real benefit in throughput, latency, memory behaviour or hardware coverage.

### `aions-mcp-server` — capability layer

This remains a separate service. It exposes the live MCP tool catalogue and executes actions requested by the agent.

The WPC agent must treat this server as the authority for available tools: it discovers them through `initialize` and `tools/list` instead of compiling a static tool list.

### `aions-server-wiedzy` — knowledge/history layer

This repository is the AIONS knowledge and consolidation archive. Its useful output for the current stack is knowledge, memory and historical design context.

It is **not** another inference runtime.

### Historical/reference repositories

`mcp-integration-system`, `super-system`, `polip-agi` and `fresh-start` remain reference or historical material. Promote individual components only after explicit revalidation against current interfaces and CI.

## 3. Language and execution-layer policy

AIONS is intentionally **polyglot**. The language is selected by the responsibility of the layer, not by ideology.

| Layer | Preferred technology | Rule |
|---|---|---|
| Desktop UI | TypeScript + React/Svelte | UI, editor, graph, dashboards and developer tooling |
| Desktop/native shell | Tauri + Rust | OS integration, windows, IPC, filesystem and process control |
| Core/runtime | Rust | lifecycle, ownership, sessions, scheduler, KV, memory, security and stable APIs |
| WPC compiler/format | Rust | parsing, quantisation, packaging and deterministic model handling |
| Portable AI kernels | Mojo | candidate backend for batched GEMM, attention, dequantisation and SIMD-heavy compute |
| NVIDIA GPU backend | CUDA C++ | production NVIDIA kernels, Tensor Cores and CUDA libraries |
| GPU micro-optimisation | PTX | only for measured hot paths where CUDA is insufficient |
| CPU micro-optimisation | C/C++ intrinsics and assembly | only for measured hot paths where generated code is insufficient |
| Research/tooling | Python | experiments, model analysis and offline benchmarks; not the production hot path |

### Compute boundary

Language boundaries must occur at **coarse-grained compute boundaries**, not inside every token operation.

Preferred shape:

```text
Rust runtime
   -> submit batch / tensor view
   -> Mojo or CUDA backend
   -> dequant + GEMM + attention + KV update
   -> return results / metadata
   -> Rust continues scheduling
```

Avoid per-token Rust -> FFI -> Rust chatter. The boundary must carry enough work to amortise FFI, synchronisation and launch overhead.

### Mojo policy

Mojo is an **acceleration layer**, not a replacement for the Rust runtime. Start with isolated kernels and backend interfaces. Promote Mojo code only when it is faster, equally correct, sufficiently stable and easier to maintain than the current Rust/C++ implementation for that workload.

### CUDA/PTX policy

CUDA is the preferred NVIDIA production backend. PTX and assembly are surgical tools, not default implementation languages. Do not hand-write assembly before profiling proves the higher-level implementation is the bottleneck.

## 4. Runtime contract

The production loop is designed to become:

1. Start one `aions-agent` session.
2. Start or connect to the configured AIONS MCP server.
3. Perform MCP `initialize`.
4. Perform `tools/list` and build a compact manifest.
5. Attach the agent to **one resident WPC runtime** for the task/session.
6. Load compressed weights once.
7. Keep the active KV cache resident.
8. Use batched prefill/forward where possible.
9. If the model emits `TOOL_CALL`, execute it through MCP.
10. Append the tool result to the same transcript and continue from the same runtime/KV state.
11. Repeat until `FINAL`, cancellation or `max_turns`.

The current project may still use subprocess runtime startup while the resident runtime is being implemented, but that is a temporary boundary, not the target architecture.

## 5. Performance path

The target execution path is:

```text
model artifact
   -> mmap / packed weights
   -> resident WPC runtime
   -> batched prefill / forward
   -> dequant backend
   -> GEMM / attention backend
   -> resident KV cache
   -> sampling
   -> next batch / token
```

The architecture is layered so performance improvements do not destabilise the system:

**Layer 0 — correctness baseline**

Keep the existing Rust implementation as the reference path. Preserve numerical contracts and deterministic tests.

**Layer 1 — runtime residency**

Make the WPC runtime long-lived per session. Load weights once, reuse allocations and keep KV state alive across agent turns.

**Layer 2 — batching**

Promote `BatchEngine`/batched forward and prompt prefill. Benchmark latency, throughput, memory use and correctness against the single-token reference.

**Layer 3 — compute backends**

Move only measured hot kernels behind backend interfaces:

```text
CPU reference -> Rust
CPU accelerated -> Mojo / C++ intrinsics / AVX2 / AVX-512
NVIDIA -> CUDA C++
NVIDIA surgical -> PTX / assembly where justified
```

**Layer 4 — scheduling and locality**

Add expert grouping, memory locality, kernel fusion, asynchronous transfers and better batch scheduling only when benchmarks justify the complexity.

## 6. Quality, precision and stability gates

Every promoted acceleration layer must satisfy all of the following before replacing the reference implementation:

- **Correctness:** outputs match the reference within an explicitly documented numerical tolerance.
- **Precision:** quantisation/dequantisation and model semantics remain unchanged unless the change is explicitly part of a new scheme.
- **Determinism:** repeated runs under the same configuration remain reproducible where deterministic behaviour is promised.
- **Stability:** no regression in the full Rust workspace build/test/Clippy gates.
- **Performance:** a benchmark demonstrates a material benefit on the target hardware.
- **Isolation:** backend failures can fall back to the reference path where practical.
- **Observability:** backend choice, timing and failures are measurable.

Performance is never accepted by sacrificing silent correctness or model quality.

## 7. What is deliberately NOT merged

Do not copy all historical AIONS files into WPC Engine merely to create the appearance of one project. That would create duplicate runtimes, conflicting memory implementations and unclear ownership.

The unification is architectural:

- **WPC = brain / inference and model execution**
- **Rust = system state / lifecycle / safety / orchestration**
- **AIONS Agent = control loop**
- **AIONS MCP = hands / tools**
- **CBMS / knowledge server = long-term knowledge**
- **TypeScript + Tauri = Studio / human interface**
- **Mojo / CUDA / C++ / ASM = compute acceleration layers**
- **historical repos = source material**

## 8. Promotion rule

A component moves into the active stack only when it has:

1. a single defined owner;
2. a documented interface;
3. an automated test or smoke test;
4. no duplicate production implementation;
5. a measurable reason to exist;
6. a clear rollback/reference implementation where the component is performance-critical.

## 9. Master implementation roadmap

### Phase A — Foundation and CI

- [x] Unified Rust workspace and module contracts
- [x] WPC runtime/compiler/format baseline
- [x] AIONS Agent MCP integration baseline
- [x] Unified organism diagnostics and contract checks
- [x] CI repair loop with pre-push verification and explicit workflow dispatch
- [ ] Historical Git secret scan before declaring public repositories clean

### Phase B — Resident inference runtime

- [ ] Make `wpc-runtime` long-lived for an entire agent task/session
- [ ] Keep compressed model weights resident across turns
- [ ] Keep KV cache resident and reusable across turns
- [ ] Reuse allocator/buffer pools instead of rebuilding execution state
- [ ] Expose a stable Rust session API for the resident runtime

### Phase C — Batched compute

- [ ] Complete batched prefill/forward path
- [ ] Compare batched path against the single-token reference for correctness
- [ ] Benchmark latency, throughput and memory footprint
- [ ] Add batch scheduling and prompt-prefill optimisations
- [ ] Only then enable speculative decoding and expert-grouped execution

### Phase D — Compute backends

- [ ] Define backend interface at coarse tensor/batch boundaries
- [ ] Keep Rust reference kernels as correctness baseline
- [ ] Prototype Mojo kernels for the hottest CPU/GPU-portable operations
- [ ] Add AVX2/AVX-512 specialisations where profiling justifies them
- [ ] Add CUDA C++ backend for NVIDIA hardware
- [ ] Use PTX/assembly only for measured remaining hot spots
- [ ] Benchmark every backend against the same reference vectors and numerical tolerances

### Phase E — AIONS Studio

- [ ] TypeScript UI shell and component system
- [ ] Tauri + Rust native bridge
- [ ] model/runtime controls
- [ ] live agent transcript and tool inspection
- [ ] performance/health dashboard
- [ ] memory/KV inspection
- [ ] integrated graph/developer tooling

### Phase F — Memory and system graph

- [ ] Unified memory graph for code, processes, agents and dependencies
- [ ] CBMS integration for persistent knowledge/transcripts
- [ ] clear separation between hot KV state and persistent/offline memory
- [ ] visual inspection and diagnostics in Studio

### Phase G — Kernel and system boundaries

- [ ] AIONS kernel contracts and userspace-first services
- [ ] IPC/capability/scheduling primitives
- [ ] driver/service boundaries that do not contaminate application runtime
- [ ] Ghost Gate isolated network boundary

### Phase H — OS integration

- [ ] packaging and release pipeline
- [ ] boot/service supervision
- [ ] permissions and security model
- [ ] observability and recovery
- [ ] end-to-end release validation

## 10. Current priority order

The immediate order is intentionally strict:

1. Finish the current CI verification cycle on `integration/full-organism-v2`.
2. Treat the verified Rust workspace as the stable reference baseline.
3. Build the resident runtime/session layer.
4. Make KV state persistent across agent turns.
5. Finish and benchmark batched prefill/forward.
6. Introduce backend interfaces and benchmark Mojo/CUDA/CPU specialisations.
7. Build AIONS Studio on TypeScript + Tauri.
8. Expand Memory/Graph, then Kernel, Ghost Gate and OS integration.

The project should optimise **from the inside out**: preserve a correct Rust core, add acceleration behind stable interfaces, and promote only changes that improve speed, quality or hardware coverage without creating an unacceptable stability or maintenance cost.
