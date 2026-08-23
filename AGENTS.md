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

## Current milestone state

- Milestones 1–2: complete on the integration branch after full workspace build/test/format/Clippy/benchmark gates passed.
- Milestone 3: active.
- Resident WPC weights remain loaded across agent turns.
- Resident prompt/KV reuse is implemented in commit `7605c94c` and must remain green after formatting.
- Stage 3A batched Qwen3-MoE prefill is isolated in draft PR `#22` (`feature/qwen3-moe-batched-prefill`) and must pass its Rust correctness/CI gate before merge.
- Do not merge PR #22 until the integration branch and the PR checks are both green.

## Work sequencing

Finish and verify resident KV reuse first. Then validate Stage 3A batched prefill. Only after both are green move to deeper batched linear/MoE execution, speculative verification, expert-grouped scheduling, and the later kernel/Ghost Gate/Studio/OS integration milestones.
