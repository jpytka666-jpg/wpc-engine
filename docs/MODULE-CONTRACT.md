# AIONS Kernel module contract

## Role
Minimal Rust kernel foundation: memory primitives, scheduling, IPC, capabilities and hardware-facing primitives.

## Architecture
Prefer userspace services and drivers. Kernel owns only mechanisms that require kernel privilege.

## Security
Capability-oriented isolation, explicit IPC boundaries, least privilege, deterministic boot/runtime policy.

## Boundaries
Userspace AIONS services, WPC, memory, Studio and Ghost Gate remain outside the kernel.

## Rule
No local changes to existing AIONS installations. GitHub-first design and CI; local implementation only after acceptance.
