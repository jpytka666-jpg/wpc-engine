# Memory Graph module

Purpose: expose relationships between code, memory, tasks, tools and runtime state
without replacing CBMS or becoming another inference engine.

## Stage 1 contract

- nodes have stable IDs and typed kinds;
- edges have explicit direction and relationship type;
- provenance is mandatory for derived facts;
- graph mutations are append/audit oriented;
- queries are bounded and deterministic;
- graph views never become the canonical memory store.

## Next gate

Define node/edge schemas, provenance rules and a query API, then add a small
in-memory implementation with round-trip tests.
