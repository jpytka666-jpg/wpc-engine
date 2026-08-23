# AIONS Kernel — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Define the minimum privileged mechanism surface while keeping drivers and AIONS services in userspace.

## Milestones
- [x] Kernel module contract.
- [ ] Boot contract and panic/recovery policy.
- [ ] Capability model and object rights.
- [ ] IPC message envelope and endpoint identity.
- [ ] Minimal scheduler-facing service contract.
- [ ] Userspace driver boundary.
- [ ] Memory-management primitive contract.
- [ ] Security tests for capability denial and IPC isolation.

## Rules
No driver pile-in. Kernel owns mechanisms, not AIONS business logic. Redox-inspired userspace service boundaries remain the default.
