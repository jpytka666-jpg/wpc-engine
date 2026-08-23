# Memory / KV module contract

## Role
Tiered AI memory: hot KV, compressed/persistent KV experiments, and CBMS-backed durable memory.

## Design
Keep token-critical hot KV in fast memory. Persistent or compressed forms stay outside the critical generation path until benchmarks prove useful.

## Measures
Compression ratio, reconstruction latency, RAM/VRAM use, and output-quality impact.

## Boundaries
Do not own the model executor, UI, kernel, or network gateway.

## Rule
Research and interfaces land on GitHub first; local implementation follows only after CI and benchmark acceptance.
