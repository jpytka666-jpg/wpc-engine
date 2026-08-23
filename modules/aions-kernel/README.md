# AIONS Kernel module

Purpose: define the future Redox/Rust kernel boundary while keeping AIONS
services and drivers in user space.

## Architecture

```text
Redox kernel
  |-- scheduling / memory / IPC / hardware primitives
  +-- capability boundary
        |
        +-- AIONS user-space services
              |-- runtime
              |-- agents
              |-- drivers
              +-- tools
```

## Stage 1 contract

- no AIONS application logic in kernel space;
- device access is capability-controlled;
- IPC contracts are explicit and versioned;
- every driver has a user-space owner;
- kernel integration is tested behind a feature boundary.

## Next gate

Specify capability IDs, IPC envelopes and the user-space driver ABI before any
kernel code is promoted from prototype to integration.
