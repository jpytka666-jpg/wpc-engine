# AIONS Local CI

AIONS Local CI is the planned local verification and repair layer for the WPC/AIONS workspace.

## Goal

Turn a coding request into a verified repository state, not merely a generated patch:

`request -> inspect -> implement -> format -> build -> test -> clippy -> benchmark -> repair -> verify`

## Safety model

The agent must operate inside an explicitly selected repository/worktree. It must:

- keep an auditable diff;
- use an allowlisted command set;
- never push or merge by default;
- run verification after every repair;
- stop after a bounded number of repair attempts;
- require explicit approval before destructive operations or publishing changes.

## Planned components

1. **Runner** — executes deterministic checks and captures exit codes/stdout/stderr.
2. **Failure classifier** — groups failures into formatting, compile, test, lint, and benchmark classes.
3. **Context builder** — gives the coding model the relevant files, diagnostics, and diff rather than the entire repository by default.
4. **Repair agent** — proposes and applies a minimal patch.
5. **Verification loop** — reruns the failing check and then the full gate after a successful repair.
6. **Git controller** — produces commits and optionally prepares a PR; publishing remains opt-in.
7. **AIONS tool interface** — exposes commands such as `check`, `repair`, and eventually `implement`.

## Initial gate

The first implementation should mirror the repository's GitHub CI:

- `cargo build --workspace --release`
- `cargo test --workspace --release`
- `cargo fmt --all --check`
- `cargo clippy -p wpc-runtime --all-targets --release -- -D warnings`
- benchmark compilation
- bounded benchmark smoke test

The local agent should report `GREEN` only when the complete gate passes.
