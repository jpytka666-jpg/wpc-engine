# Ghost Gate — Implementation Stage 1

Status: active GitHub-only implementation stage.

## Goal
Define a narrow, auditable network gateway contract before building the VM.

## Milestones
- [x] Default-deny gateway contract.
- [ ] Typed egress request and policy decision model.
- [ ] Modes: OFFLINE, VPN, VPN+firewall, optional TOR.
- [ ] DNS policy and leak-prevention contract.
- [ ] Route/audit event format.
- [ ] Health and fail-closed semantics.
- [ ] Container/VM integration test plan.
- [ ] Security gate for forbidden direct egress.

## Rules
AIONS talks to Ghost Gate; Ghost Gate is the only network boundary in this design. Local VM deployment waits for security and CI acceptance.
