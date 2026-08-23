# AIONS Project Agent Instructions

## Identity

Project owner: **Marcin Szul**.
Repository: `jpytka666-jpg/wpc-engine`.

## Startup rule

Before changing architecture, CI, or cross-layer interfaces, read:

1. `AIONS_MASTER_BUILD_PLAN.md`
2. `.github/AIONS_AGENT_CONTEXT.md`
3. `docs/AIONS-INTEGRATION-MAP.md`
4. `docs/superpowers/plans/2026-08-23-aions-os-build.md`

These files are the persistent project context and source of truth for the AIONS OS build.

## Engineering source of truth

GitHub is authoritative. Local disks are **read-only** for inspection; implementation changes belong on GitHub.

## Current architecture

The project is built through eight isolated architectural lanes:

- `arch/wpc-runtime`
- `arch/agents-ci`
- `arch/memory-kv`
- `arch/studio`
- `arch/memory-graph`
- `arch/aions-kernel`
- `arch/ghost-gate`
- `arch/os-integration`

Do not merge these lanes blindly. Each lane must expose stable contracts and pass its own verification gates before final integration.

## Verification rule

Never disable Clippy, tests, security checks, or functional gates just to obtain green CI. A green result is valid only after the relevant verification actually runs and passes.

## Current mission

Build an offline-first AIONS operating environment consisting of:

Rust kernel → userspace services/drivers → Ghost Gate → memory substrate → resident WPC runtime → agent/tool platform → AIONS Studio → live graph → final OS integration.
