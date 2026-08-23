# AIONS Local CI — Roadmap

## Phase 0 — Foundation
- [x] Define safety model and deterministic verification gates
- [x] Add standalone Rust runner
- [x] Add failure classification
- [x] Keep push/merge disabled by default

## Phase 1 — Context + diagnostics
- [ ] Capture stdout/stderr and exit status for every gate
- [ ] Parse Rust compiler diagnostics, test failures, Clippy output and rustfmt diffs
- [ ] Identify relevant files and bounded source context
- [ ] Produce a machine-readable repair request

## Phase 2 — AI repair
- [ ] Connect an approved local LLM
- [ ] Give the model only the diagnostic, relevant diff and required source context
- [ ] Generate a minimal patch
- [ ] Apply patch in an isolated worktree
- [ ] Re-run the failing gate
- [ ] Limit repair attempts and detect repeated failures

## Phase 3 — Coding agent
- [ ] Add `implement` mode for explicit feature requests
- [ ] Let the model inspect architecture and plan changes
- [ ] Generate code across multiple files when required
- [ ] Verify with the complete CI gate
- [ ] Produce a human-readable change summary

## Phase 4 — Git automation
- [ ] Create checkpoint commits
- [ ] Generate branch/PR metadata
- [ ] Push only when explicitly enabled
- [ ] Never merge automatically by default
- [ ] Preserve rollback points

## Phase 5 — AIONS integration
- [ ] Expose `check`, `repair`, and `implement` as AIONS tools
- [ ] Add policy/permission controls
- [ ] Stream progress and diagnostics to AIONS
- [ ] Store repair history in CBMS

## Definition of done
AIONS can receive a coding task, implement it, detect failures, repair them within a bounded loop, and report GREEN only after the independent verification gate passes.
