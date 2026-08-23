# AIONS Studio module

Purpose: provide the human-facing control surface for the AIONS system.

The Studio is an IDE-like shell around the system, not a second runtime.
It consumes explicit capabilities from the agent and kernel layers.

## Stage 1 contract

- command palette maps to typed commands;
- diagnostics are inspectable as structured events;
- AI actions are visible and auditable;
- terminal access is optional, not the primary control path;
- the visual graph is a view over system state, not the source of truth.

## Next gate

Define the command schema, event stream and capability discovery handshake.
No UI implementation is accepted until those interfaces are stable.
