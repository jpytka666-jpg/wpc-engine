# Memory-KV module

Purpose: define the stable boundary between resident inference and reusable KV state.

Stage 1 owns metadata, lifecycle, compatibility, sequence ownership, and persistence contracts.
It does not implement model-specific attention kernels.

## Contract

- session identity is explicit;
- KV entries have model/config fingerprints;
- incompatible state is rejected, never silently reused;
- hot state and durable state are separate layers;
- sequence ownership is contiguous and exclusive;
- serialization is deterministic;
- callers can read owned ranges without knowing storage internals.

## Current implementation gate

The module now provides a typed KV envelope plus a hot in-memory batch buffer. Appends must begin at the next unowned sequence position; gaps and overlaps are rejected. Reads are restricted to already-owned half-open ranges.

Persistent storage, CBMS integration, compressed KV, and WPC runtime wiring remain outside this gate.
