# AIONS — Master Roadmap

## Vision

AIONS is a Rust-first AI operating environment built around a small, capability-oriented kernel, isolated userspace services, a persistent AI memory substrate, a resident WPC inference runtime, an integrated development environment, and an explicit network security boundary.

The goal is not to build everything at once. Each layer must have a working, testable milestone before the next layer depends on it.

## Architecture

```text
                         AIONS STUDIO
          editor · compiler · debugger · AI · graph
                              │
                       AIONS SYSTEM API
                              │
       ┌──────────────────────┼──────────────────────┐
       │                      │                      │
   AI / MEMORY             RUNTIME                SYSTEM
       │                      │                      │
   CBMS · KV · WPC      resident WPC          services/drivers
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              │
                        AIONS KERNEL
                         Rust / IPC
                              │
                           hardware
                              │
                        GHOST GATE VM
                    firewall · VPN · DNS · Tor
                              │
                           network
```

## Track 1 — WPC Runtime

- [x] WPC format/compiler foundations
- [x] Batch attention / GEMM foundation
- [ ] Make resident AIONS runtime fully green
- [ ] Validate resident model lifetime across multiple agent turns
- [ ] Benchmark startup vs resident execution
- [ ] Merge integration only after full CI verification

## Track 2 — KV Cache + Memory Substrate

- [ ] Define canonical KV block format
- [ ] Implement hot KV interface
- [ ] Define warm/cold KV tiers
- [ ] Build CBMS memory index for persistent AI context
- [ ] Add KV replay/load benchmarks
- [ ] Prototype WPC-compressed KV blocks
- [ ] Measure compression ratio
- [ ] Measure reconstruction latency
- [ ] Measure output/attention quality impact
- [ ] Integrate only after standalone evidence is positive

## Track 3 — AIONS Local CI / Coding Agent

- [x] Deterministic local runner foundation
- [x] Failure classifier foundation
- [ ] Robust Rust diagnostics parser
- [ ] Context builder
- [ ] Local LLM integration
- [ ] Isolated patch application
- [ ] Bounded repair loop
- [ ] `check` / `repair` / `implement` tools
- [ ] Git checkpointing and PR preparation
- [ ] CBMS repair history

## Track 4 — AIONS Studio

The graphical face of the OS: an IDE-like environment that is also the system shell.

- [ ] Native Rust UI shell
- [ ] Code editor
- [ ] Integrated compiler/build panel
- [ ] Debugger
- [ ] AI coding/chat panel
- [ ] Git interface
- [ ] System monitor
- [ ] Memory inspector
- [ ] WPC/KV inspector
- [ ] Integrated terminal as an advanced escape hatch, not the primary interface

## Track 5 — System / Memory Graph

Interactive graph rather than a decorative dashboard.

- [ ] Define graph entities and relationships
- [ ] Visualize agents, processes, services and drivers
- [ ] Visualize CBMS memories and links
- [ ] Visualize KV/WPC objects
- [ ] Navigate from graph node to source/code/state
- [ ] Live system graph updates
- [ ] Graph becomes an operational interface, not only visualization

## Track 6 — AIONS Kernel

Rust-first kernel inspired by the Redox separation model, but with an AIONS-specific architecture.

- [ ] Freeze userspace/kernel contract
- [ ] Define IPC and capability model
- [ ] Minimal memory management
- [ ] Scheduler/process model
- [ ] Userspace scheme/service model
- [ ] Storage service
- [ ] GPU/display service
- [ ] Input service
- [ ] Network service boundary
- [ ] Hardware driver model in userspace where practical
- [ ] Bootable minimal AIONS image

## Track 7 — Ghost Gate

A separate small VM acts as the network boundary. AIONS does not directly own the external network path.

- [ ] Define AIONS ↔ Ghost Gate protocol
- [ ] Minimal Debian/Linux VM prototype
- [ ] Default-deny firewall
- [ ] DNS policy/filtering
- [ ] VPN mode
- [ ] Optional Tor mode
- [ ] Network observability and audit log
- [ ] Fail-closed behavior
- [ ] AIONS offline mode

Ghost Gate is a security boundary, not an anonymity guarantee. VPN/Tor configuration must be treated as separate, explicit modes.

## Track 8 — AIONS OS Integration

- [ ] Boot kernel + userspace services
- [ ] Start resident WPC runtime as a managed service
- [ ] Start CBMS/memory services
- [ ] Start AIONS Studio
- [ ] Start graph subsystem
- [ ] Connect Local CI/Coding Agent
- [ ] Connect Ghost Gate
- [ ] End-to-end system smoke test
- [ ] Reproducible build/image

## Track 9 — TempleOS-inspired UX (not architecture)

Borrow the useful idea: development and system operation live in one environment.

- [ ] Immediate compile/run workflow
- [ ] System APIs visible as programmable objects
- [ ] AI-assisted interactive development
- [ ] Inspectable system state
- [ ] Fast feedback loop

Do not copy TempleOS's lack of modern memory/process isolation. Security and isolation follow the AIONS/Redox-inspired model instead.

## Milestones

### M0 — Working WPC foundation
Batch attention + WPC runtime work independently and are tested.

### M1 — Resident AI runtime
AIONS can keep a model resident across multiple agent turns and the complete CI gate is green.

### M2 — AI developer loop
AIONS Local CI can detect, diagnose, patch and verify bounded coding repairs.

### M3 — Persistent AI memory
Hot KV + CBMS form a measured memory substrate; compressed KV remains experimental until benchmarks justify integration.

### M4 — AIONS Studio
The IDE becomes the primary interface for code, AI, memory, system state and Git.

### M5 — System graph
A live graph exposes and controls relationships across code, agents, memory and system services.

### M6 — Kernel prototype
Minimal bootable Rust kernel with userspace services and explicit IPC/capability boundaries.

### M7 — Ghost Gate
Network access is isolated behind a separately managed VM boundary.

### M8 — AIONS OS prototype
Kernel + userspace services + memory + resident AI + Studio + graph + Ghost Gate operate as one reproducible system.

## Engineering rules

1. A passing build is not enough; required verification gates must be green.
2. Experimental subsystems stay isolated until measured.
3. AI-generated code is accepted only after independent verification.
4. Security boundaries fail closed by default.
5. Keep kernel responsibilities minimal; prefer userspace services.
6. Preserve rollback points and auditable Git history.
7. Benchmark before replacing a working subsystem with an experimental one.
