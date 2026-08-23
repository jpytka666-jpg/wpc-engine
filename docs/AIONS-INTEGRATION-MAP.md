# AIONS Integration Map

## Purpose
This document maps the historical repositories into the modular AIONS OS architecture. Historical code is treated as source material until it passes current interfaces and CI.

## Eight architectural modules

1. **WPC Runtime** — model execution, resident runtime, batch GEMM/attention.
2. **Agents / Local CI** — deterministic verification, diagnostics, repair loop, coding agent.
3. **Memory / KV** — hot KV, compressed KV research, CBMS persistence and retrieval.
4. **AIONS Studio** — native developer/system environment: editor, compiler, debugger, AI and system controls.
5. **Memory/System Graph** — interactive graph of code, memory, processes, agents and dependencies.
6. **AIONS Kernel** — Rust kernel, IPC, capabilities, scheduling and memory primitives; userspace-first drivers/services.
7. **Ghost Gate** — isolated network VM boundary for firewall/VPN/DNS and optional Tor routing.
8. **OS Integration** — packaging, boot, service supervision, permissions, observability and release integration.

## Historical repository mapping

- `wpc-engine` → primary WPC/runtime source.
- `aions-mcp-server` → tools, MCP integration, CBMS and automation source material.
- `aions-server-wiedzy` → knowledge/memory source material.
- `mcp-integration-system` → orchestration, workflow, security and integration source material.
- `fresh-start` → candidate Studio/developer-environment source material.
- `polip-agi` → historical agent architecture and experiments.
- `super-system` → historical system integration experiments.

## Integration rule
Do not merge historical repositories wholesale. Extract useful components behind stable AIONS interfaces, add tests, and migrate only verified modules.

## Security rule
Public source is assumed observable. Secrets must never be committed. Environment variables, local secret stores and secret managers are required for credentials. Historical Git history still needs a dedicated secret scan before declaring the public repositories clean.

## Development order

`WPC runtime → Local CI → Memory/KV → Studio → Graph → Kernel → Ghost Gate → OS integration`

Each module gets an independent implementation branch and CI gate. Cross-module integration happens through explicit interfaces rather than direct repository coupling.
