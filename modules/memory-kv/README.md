# Memory-KV module

Purpose: define the stable boundary between resident inference and reusable KV state.

Stage 1 owns metadata, lifecycle, compatibility and persistence contracts.
It does not implement model-specific attention kernels.

## Contract

- session identity is explicit;
- KV entries have model/config fingerprints;
- incompatible state is rejected, never silently reused;
- hot state and durable state are separate layers;
- serialization is deterministic;
- callers can evict, snapshot and restore without knowing storage internals.

## Next implementation gate

Add a typed envelope, compatibility validator, bounded snapshot format and
property tests before wiring this module into `wpc-runtime`.
