# AIONS Integration Map

This repository is the current Rust/WPC engineering core. The other public repositories are treated as historical and service-layer sources, not as independent competing architectures.

## Source repositories

| Repository | Historical role | Planned destination |
|---|---|---|
| `aions-server-wiedzy` | AIONS knowledge/server work | CBMS / memory services |
| `aions-mcp-server` | 40+ MCP tools, memory, project scanning, Docker/WSL/system tooling | AIONS services + Studio tool layer |
| `mcp-integration-system` | MCP registry, workflow orchestration, codegen, security, monitoring, cloud integration | AIONS agent/orchestration + optional external/cloud adapters |
| `fresh-start` | Claude/Kiro/MCP/VS Code integration and workflow automation | AIONS Studio developer environment |
| `wpc-engine` | WPC format/compiler/runtime, resident runtime, KV/attention work | Core Rust runtime |
| `polip-agi` | Earlier experimental AGI/system direction | Historical research; recover useful ideas selectively |
| `super-system` | Earlier system-level experiments | Historical research; audit before integration |

## Modular architecture

### Layer 0 — Hardware / kernel
Rust microkernel architecture inspired by Redox: memory, scheduling, IPC, capabilities and minimal hardware-facing primitives.

### Layer 1 — System services
Userspace drivers, storage, process services, graphics, input and system APIs. Drivers remain outside the kernel wherever practical.

### Layer 2 — Ghost Gate
An isolated VM providing the network boundary: firewall, VPN, DNS policy and optional Tor routing. AIONS talks to a narrow gateway API instead of directly owning the physical network path.

### Layer 3 — Memory substrate
CBMS, hot KV, warm/compressed KV, WPC model representations, indexes and event/history storage.

### Layer 4 — AI runtime
WPC resident runtime, model loading, attention/KV execution, inference services and agent runtime.

### Layer 5 — Agent/tool platform
MCP registry, tool discovery, workflow orchestration, browser/desktop automation, project scanner, Git and AIONS Local CI.

### Layer 6 — AIONS Studio
Native IDE-like system shell: editor, compiler, debugger, AI agent, memory view, system controls, Git and visual graph.

### Layer 7 — Graph / system view
Interactive graph representing projects, code dependencies, processes, memory, agents, services and system relationships.

## Integration rules

1. Preserve working historical components until their replacement is tested.
2. Prefer adapters/interfaces over copying entire repositories into WPC Engine.
3. Keep experimental kernel, KV and Ghost Gate work isolated from production runtime until benchmarks/tests pass.
4. Treat the WPC Engine as the current Rust core, not as the final OS repository layout.
5. Move reusable MCP/tool capabilities behind stable AIONS service interfaces.
6. Keep cloud-specific components optional; the core AIONS OS must remain useful offline.
7. Every major module gets its own branch and CI gate.

## Historical timeline anchors

- 2025-12-06: AIONS MCP server and AIONS context/tool ecosystem already existed.
- 2025-12-10 to 2025-12-15: MCP integration system grew into a large tested orchestration/cloud/security platform; commits record 749 tests and 40 validated properties.
- 2026-08: WPC/runtime/KV work became the active Rust engineering core and resident runtime integration began.
- 2026-08-23: architecture expanded into an explicit AIONS OS plan: Rust kernel, userspace services/drivers, Ghost Gate, memory substrate, Studio, graph and autonomous coding/CI.

## Eight development branches

1. `arch/wpc-runtime` — resident WPC runtime and inference core
2. `arch/agents-ci` — Local CI, repair loop and coding agent
3. `arch/memory-kv` — KV/CBMS memory substrate
4. `arch/studio` — AIONS Studio system shell/IDE
5. `arch/memory-graph` — system and knowledge graph
6. `arch/aions-kernel` — Rust kernel/userspace architecture
7. `arch/ghost-gate` — isolated network gateway VM
8. `arch/os-integration` — final cross-layer integration

The eight branches are architectural lanes. They should not be merged together blindly; each lane must expose stable interfaces and pass its own verification before integration.
