# Repository roles

| Repository | Role now | Production dependency? |
|---|---|---|
| `wpc-engine` | WPC compiler, runtime, agent, benchmarks | **YES — core** |
| `aions-mcp-server` | MCP tool server and capability execution | **YES — service** |
| `aions-server-wiedzy` | CBMS/knowledge/history and consolidation material | **YES — knowledge source; not inference** |
| `mcp-integration-system` | MCP integration/reference implementation | No — reference/lab |
| `super-system` | hands-on learning examples | No |
| `polip-agi` | historical AGI experiment | No |
| `fresh-start` | historical/experimental workspace | No |

## Rule

There is one production inference runtime: `wpc-engine`.

There is one production MCP tool authority: `aions-mcp-server`.

Knowledge and memory remain services/data, not competing inference engines.

Everything else is either a laboratory or history. A useful component can be promoted later, but promotion must be explicit and tested.
