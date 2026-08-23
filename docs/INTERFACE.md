# WPC Runtime Interface Sketch

This document is a design contract, not yet a production API.

## Runtime lifecycle

```text
load(model_artifact) -> ResidentModel
ResidentModel::execute(request, kv_handle) -> Output
ResidentModel::release() -> Result
```

## Design rules

- Loading is explicit and observable.
- Resident state must not depend on process-global hidden state.
- KV is passed through an interface owned by the Memory/KV module.
- Execution APIs must expose errors rather than silently falling back.
- Metrics are collected without changing inference semantics.

## Integration adapters

- **Agents/CI:** build/test/benchmark runner invokes the public runtime surface.
- **Memory/KV:** supplies KV handles and lifecycle operations.
- **AIONS Studio:** consumes diagnostics and runtime metrics; it does not own runtime state.
- **AIONS OS:** eventually hosts the runtime as a userspace service/process.

## Not implemented here yet

No new runtime implementation is introduced by this document. The next code milestone will be created as a focused PR with tests and a green CI gate.
