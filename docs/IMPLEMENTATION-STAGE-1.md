# AIONS Studio — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Define Studio as the system-facing control surface without taking ownership of subsystem state.

## Milestones
- [x] Studio contract and OS-facing role.
- [ ] Stable command/event model for editor, build, debug and Git actions.
- [ ] AI assistant session boundary and approval states.
- [ ] Runtime diagnostics panel contract.
- [ ] Memory/graph navigation contract.
- [ ] Kernel/service inspection contract.
- [ ] Ghost Gate status and network-mode display contract.
- [ ] Headless integration tests before UI implementation.

## Rules
Studio calls subsystem APIs; it never reaches into private runtime or kernel internals. Local UI implementation waits for green interface tests.
