# Memory / KV — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Define the boundary between generation-critical KV and durable/compressed memory.

## Milestones
- [x] Hot-KV versus persistent-memory contract.
- [x] Typed KV handle and lifecycle interface.
- [x] Append/read batch semantics with sequence ownership.
- [x] Memory residency metrics.
- [x] Compression experiment interface kept outside the token-critical path.
- [x] WPC vector-KV experiment path using the existing WPC codebook/residual engine.
- [ ] CBMS adapter contract.
- [x] Correctness fixture for round-trip and sequence boundaries.
- [x] Benchmark matrix scaffold for latency, RAM and compression ratio.
- [ ] Validate WPC-KV against production attention tensors and real model distributions.

## Rules
The model owns execution; this module owns KV state policy. Durable storage must never silently enter the critical generation path.
Sequence ownership is contiguous and exclusive: appends begin at the next unowned position; gaps and overlaps are rejected.
Residency metrics report module-owned hot state only; they do not measure process-global RAM/VRAM consumption.
The compression path reuses the WPC reference engine rather than introducing a second VQ implementation.
K and V receive separate codebooks because their distributions and attention roles may differ.
Compression experiments are research-only probes. Their results are metadata/measurements and cannot be marked generation-critical.
Promotion requires reconstruction correctness, attention-output correctness, measured memory reduction, and measured encode/decode cost.
