# AIONS OS Integration — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Create the integration control plane that consumes the seven module contracts without absorbing subsystem logic.

## Milestones
- [x] Integration contract and dependency rule.
- [x] Module manifest with version and CI status.
- [x] Health-check envelope.
- [ ] Service lifecycle contract.
- [ ] Permission/profile declaration.
- [ ] Boot/recovery state machine.
- [ ] Reproducible release artifact manifest.
- [ ] Full integration gate combining all module checks.

## Rules
Integration owns orchestration and verification only. Subsystems keep their own state and implementation. Local deployment begins only after the complete GitHub integration gate is green.
