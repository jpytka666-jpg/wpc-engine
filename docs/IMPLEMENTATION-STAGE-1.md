# Memory / KV — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Define the boundary between generation-critical KV and durable/compressed memory.

## Milestones
- [x] Hot-KV versus persistent-memory contract.
- [ ] Typed KV handle and lifecycle interface.
- [ ] Append/read batch semantics with sequence ownership.
- [ ] Memory residency metrics.
- [ ] Compression experiment interface kept outside the token-critical path.
- [ ] CBMS adapter contract.
- [ ] Correctness tests for round-trip and sequence boundaries.
- [ ] Benchmark matrix for latency, RAM and compression ratio.

## Rules
The model owns execution; this module owns KV state policy. Durable storage must never silently enter the critical generation path.
