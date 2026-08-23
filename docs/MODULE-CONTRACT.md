# Ghost Gate module contract

## Role
Isolated network boundary VM between AIONS and external networks.

## Functions
Firewall policy, VPN transport, DNS policy, optional Tor mode, and controlled egress.

## Architecture
AIONS services request network access through an explicit gateway interface. The host/network stack is not exposed directly to AIONS by default.

## Security
Default deny, least privilege, auditable routes, explicit modes: OFFLINE, VPN, VPN+firewall, optional TOR.

## Boundary
Ghost Gate does not own AIONS memory, runtime, Studio, or kernel state.

## Rule
GitHub-first design/implementation. Local VM deployment follows only after tests and security checks pass.
