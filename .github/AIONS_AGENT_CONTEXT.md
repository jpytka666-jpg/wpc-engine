# AIONS Agent Context

> This file is the short operational memory for coding agents working on AIONS. Read it before changing architecture or CI.

## Mission
AIONS is an offline-first AI operating environment. The Rust/WPC engine is the current core; it is not yet the final OS.

## Source of truth
- Architecture: `AIONS_MASTER_BUILD_PLAN.md`
- Integration map: `docs/AIONS-INTEGRATION-MAP.md`
- Active integration branch: `integration/full-organism-v2`
- GitHub is the engineering source of truth. Local disks are read-only.

## Eight architectural lanes
1. `arch/wpc-runtime` — resident WPC inference and performance
2. `arch/agents-ci` — agents, MCP, Local CI and repair loop
3. `arch/memory-kv` — hot/warm KV and CBMS substrate
4. `arch/studio` — native AIONS Studio/interface
5. `arch/memory-graph` — system and memory graph
6. `arch/aions-kernel` — Rust/Redox-inspired kernel
7. `arch/ghost-gate` — isolated network boundary VM
8. `arch/os-integration` — final cross-layer integration

## Current gate
Current work is unified organism validation. Do not confuse CI cleanup with an architectural milestone. Build/test correctness must be established before performance work advances.

Current immediate sequence:
1. Clean runtime Clippy.
2. Full workspace test/format/Clippy.
3. Fused kernel and attention correctness.
4. Benchmark compile/smoke.
5. Validate resident runtime.
6. Implement batched forward.
7. Reuse persistent KV across agent turns.
8. Then pursue speculative verification and expert-grouped execution.

## Coding rules
- Do not disable Clippy to make CI green.
- Do not rewrite unrelated files.
- Do not treat agent success messages as proof; inspect diff and CI.
- Keep unsafe pointer contracts explicit and documented.
- Prefer minimal fixes that preserve behavior.
- Every architectural lane keeps its own CI gate until integration.
- Do not merge kernel/Ghost Gate/Studio/Graph into the runtime branch just to make one demo work.

## Functional architecture
WPC = compressed model storage/execution.
Router/scheduler = selects specialists/tools and execution strategy.
AIONS MCP = external capability/tool boundary.
KV = hot real-time model state.
CBMS = persistent/warm memory outside token-hot path.
Studio = primary system shell/interface.
Graph = live relationship view of projects, processes, agents, services and memory.
Ghost Gate = network boundary, not part of the model runtime.

## Required evidence before claiming completion
- Build exit 0.
- Tests report zero failures.
- Clippy reports zero errors under the repository's configured warning policy.
- Functional smoke test exercises the actual interface between the changed modules.
- Performance claims have fresh measurements.
- Security/isolation claims have fresh tests.
