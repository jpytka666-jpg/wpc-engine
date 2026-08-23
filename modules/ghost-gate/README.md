# Ghost Gate module

Purpose: isolate AIONS from the public network behind a small hardened VM.

```text
AIONS services -> Ghost Gate VM -> firewall / VPN / optional Tor -> network
```

The AIONS runtime must not own raw public-network configuration directly.

## Stage 1 contract

- one narrow egress boundary;
- explicit allow/deny policy;
- DNS and routing policy live inside the gate;
- host services remain unaware of external network credentials;
- telemetry records connection intent without storing secrets;
- VM lifecycle is controlled independently from AIONS inference.

## Next gate

Define the IPC/request envelope, egress policy schema, health probe and
failure-closed behaviour before implementing VPN/Tor integration.
