# Agents / Local CI module contract

## Role
Deterministic verification, failure classification, bounded repair, and later coding-agent orchestration.

## Pipeline
check -> classify -> build context -> propose patch -> isolated apply -> verify -> report.

## Safety
No push or merge by default. Bounded repair attempts. Independent CI is the source of truth.

## Integration
Expose machine-readable diagnostics to AIONS. Do not own model runtime, kernel, network, or persistent memory.

## Rule
GitHub-first development. Local deployment follows only after the module passes its defined gates.
