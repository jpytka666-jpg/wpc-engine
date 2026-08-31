# WPC Runtime Module Contract

## Mission
The WPC runtime is the execution layer for compressed model weights and resident model state used by AIONS.

## Owns
- WPC model loading and validation
- Resident model lifecycle
- Batch execution, attention and GEMM runtime primitives
- Runtime-facing KV interfaces
- Performance benchmarks and correctness tests

## Does not own
- AIONS kernel or device drivers
- Network access or Ghost Gate policy
- AIONS Studio UI
- Long-term CBMS policy
- GitHub automation or coding-agent policy

## Required boundaries
Cross-module behavior uses explicit Rust APIs. Other modules must not reach into private runtime implementation details.

## Acceptance gates
1. Supported Rust toolchain builds successfully.
2. Workspace tests pass.
3. `cargo fmt --all --check` passes.
4. Runtime Clippy with warnings denied passes.
5. Benchmark target compiles and smoke benchmark completes.
6. Resident load/use/release lifecycle is deterministic.
7. Benchmark code cannot silently change production semantics.
8. Public APIs needed by Memory/KV and Agents/CI are documented.

## Promotion protocol
GitHub branch is the design and verification laboratory. CI-green changes are promoted to the isolated local workspace only after review. Existing AIONS local workspaces remain read-only until explicit integration.
