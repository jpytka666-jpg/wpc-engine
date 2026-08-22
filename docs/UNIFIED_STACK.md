# AIONS + WPC Unified Stack

This document is the canonical architecture for the current project. The repositories are **not** merged blindly: each keeps one responsibility and WPC Engine becomes the inference/runtime center.

## 1. System boundary

```text
                         USER / TASK
                              |
                              v
                    +-------------------+
                    |   AIONS AGENT     |
                    | task loop/router  |
                    +---------+---------+
                              |
                 MCP initialize + tools/list
                              |
                              v
                    +-------------------+
                    |  AIONS MCP SERVER |
                    | tools / filesystem |
                    | browser / docker   |
                    | memory / CBMS      |
                    +---------+---------+
                              |
                              |
                              v
+-------------------+   +-------------------+   +-------------------+
| AIONS Knowledge   |<->|   WPC RUNTIME     |<->| Model Artifact    |
| / CBMS / Chroma   |   | attention / MoE   |   | v4 4.25-bit       |
+-------------------+   | KV / sampling     |   +-------------------+
                        +---------+---------+
                                  |
                                  v
                         CPU / optional GPU

Supporting repositories remain separate:
- mcp-integration-system: integration/reference patterns
- super-system: learning/experimental material
- polip-agi and fresh-start: historical work, not runtime dependencies
```

## 2. Canonical responsibilities

### `wpc-engine` — runtime and model layer

This is the active centre of the stack.

- WPC compilation and on-disk model format
- compressed model execution
- Qwen/Gemma architecture support
- attention, RoPE, norms, MoE routing and sampling
- KV storage and the new batched-forward path
- AIONS agent executable
- performance and correctness tests

The current feature branch adds `BatchEngine`, mmap-backed KV storage, GEMM attention and benchmarks. That work directly addresses the documented single-token bottleneck.

### `aions-mcp-server` — capability layer

This remains a separate service. It exposes the live MCP tool catalogue and executes actions requested by the agent.

Current tool families include memory/CBMS, file search, project scanning, browser automation, Docker/WSL/system operations, conversation logging, MCP discovery and web fetching.

The WPC agent must treat this server as the authority for available tools: it discovers them through `initialize` and `tools/list` instead of compiling a static tool list.

### `aions-server-wiedzy` — knowledge/history layer

This repository is the AIONS knowledge and consolidation archive. Its consolidation documents describe the historical AIONS versions, CBMS material, recovery snapshots, reports and predecessors.

It is **not** another inference runtime. Its useful output for the current stack is knowledge, memory and historical design context.

### `mcp-integration-system` — reference/integration lab

This contains a TypeScript MCP integration/server implementation and should not become a second production MCP server by accident. Reuse proven integration patterns from it, but keep the production tool authority in `aions-mcp-server`.

### `super-system` — learning lab

This is an educational/experimental repository. It is not a dependency of the production runtime.

### `polip-agi` and `fresh-start` — historical projects

Keep them intact as historical experiments. Do not pull their code into the runtime unless a specific component is deliberately revalidated and promoted.

## 3. Runtime contract

The production loop is:

1. Start `aions-agent`.
2. Start the configured AIONS MCP server as a child process.
3. Perform MCP `initialize`.
4. Perform `tools/list` and build a compact manifest.
5. Run the WPC model for the current turn.
6. If the model emits `TOOL_CALL`, execute it through MCP.
7. Append the tool result to the transcript.
8. Repeat until `FINAL` or `max_turns`.

The MCP command currently documented by the project is:

```text
docker exec -i aions-mcp python -m src stdio
```

The agent deliberately launches the program directly rather than through a shell.

## 4. Performance path

The immediate performance path is now explicit:

```text
model artifact
   -> mmap / packed weights
   -> BatchEngine
   -> GEMM attention
   -> KV layer
   -> sampling
   -> next token
```

The next major optimisation is **long-lived runtime state**. The current agent launches `wpc-runtime` for each model turn. The target design is one resident runtime per task/session so that:

- compressed weights are loaded once;
- KV cache remains resident;
- model/runtime initialisation is amortised;
- MCP tool calls return to the same model state;
- batched prefill can operate on the complete prompt instead of one token at a time.

## 5. What is deliberately NOT merged

Do not copy all historical AIONS files into WPC Engine merely to create the appearance of one project. That would create duplicate runtimes, conflicting memory implementations and unclear ownership.

The unification is architectural:

- **WPC = brain/inference**
- **AIONS Agent = control loop**
- **AIONS MCP = hands/tools**
- **CBMS/knowledge server = long-term knowledge**
- **historical repos = source material**

## 6. Promotion rule

A component moves from a historical/reference repository into the active stack only when it has:

1. a single defined owner;
2. a documented interface;
3. an automated test or smoke test;
4. no duplicate production implementation;
5. a measurable reason to exist.

This keeps the current project from becoming another pile of partially overlapping AIONS versions.

## 7. Current priority order

1. Make the `feature/forward-batch-gemm-bench` path pass CI and correctness tests.
2. Keep the AIONS MCP contract live and dynamically discovered.
3. Make `wpc-runtime` long-lived across agent turns.
4. Make KV cache resident and reusable.
5. Benchmark batched prefill versus the current single-token path.
6. Only then consider deeper GPU execution work.

The objective is one coherent local AI system, not six repositories pretending to be one executable.
