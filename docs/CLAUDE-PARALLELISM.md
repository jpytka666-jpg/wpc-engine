# Claude parallel-work protocol

GitHub is the source of truth during architecture development.

## Safe parallelism

- Each module uses its own `arch/*` branch.
- Each GitHub Actions job runs on an isolated runner.
- Matrix jobs use `fail-fast: false` so independent checks continue after one failure.
- Never share mutable working directories between jobs.
- Never edit an existing local AIONS workspace during this phase.
- Cross-module changes require an explicit integration branch and integration CI.

## Claude handoff

Before editing, identify the module branch, contract, roadmap and dependencies. Only modify files owned by that module. Report commits, tests and unresolved dependencies for the next agent.
