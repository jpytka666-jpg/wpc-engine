# WPC Runtime module contract

## Role
Resident model execution layer for AIONS. Owns WPC compilation/runtime, batch GEMM/attention, and resident model state.

## Inputs
WPC model artifacts, tensor metadata, runtime requests, KV handles.

## Outputs
Verified inference/runtime operations and measurable latency/memory metrics.

## Boundaries
Do not own UI, network policy, kernel code, or persistent knowledge policy.

## Integration
Expose stable Rust APIs. Memory/KV integration must remain behind explicit interfaces.

## Rule
No local implementation changes to existing repositories from this branch; this branch is the GitHub design/implementation laboratory.
