# WPC Runtime module

Purpose: own compressed-model execution and expose a long-lived resident runtime
contract to the AIONS agent.

## Stage 1 boundary

- model loading is explicit and validated;
- resident state owns weights and session-local execution state;
- KV ownership is delegated to the Memory-KV contract;
- tool execution is delegated to MCP/capability services;
- runtime failures are structured and recoverable;
- no UI, network policy or kernel code enters this module.

## Resident lifecycle

```text
load artifact -> validate model -> warm runtime -> serve turns -> flush/evict
```

## Next gate

Define the resident session interface, KV handoff contract and deterministic
smoke tests for load-once / multi-turn reuse before further optimisation.
