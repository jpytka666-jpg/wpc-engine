# Agents / Local CI roadmap

## Phase 1 — deterministic runner
- [x] Contract and safety model
- [x] Parallel GitHub Actions gate structure
- [ ] Capture exit code/stdout/stderr
- [ ] Machine-readable diagnostic schema

## Phase 2 — diagnosis
- [ ] Classify formatting, compile, test, lint and benchmark failures
- [ ] Select minimal source context
- [ ] Produce bounded repair request

## Phase 3 — repair
- [ ] Generate minimal patch with approved model
- [ ] Apply only in isolated worktree
- [ ] Re-run failing gate
- [ ] Stop after bounded attempts

## Phase 4 — coding agent
- [ ] Implement explicit feature requests
- [ ] Verify through complete CI gate
- [ ] Produce commit/PR proposal

## Parallelism rule
Independent checks should run as separate matrix jobs with `fail-fast: false`. Jobs must use isolated GitHub-hosted runners and must not share mutable workspace state. Cross-module integration happens only in a dedicated integration job after module gates pass.
