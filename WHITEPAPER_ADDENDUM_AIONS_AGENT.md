# WPC/AIONS Agent Integration — Addendum to WHITEPAPER.md

**21 August 2026 — post-revision-2 engineering update**

## 1. Purpose

The WPC runtime now has an AIONS-facing autonomous agent loop. The design is intentionally separate from the compression format: WPC remains the inference engine, while the agent provides planning, tool selection, execution, and transcript management.

## 2. Dynamic tool discovery

The agent connects to an AIONS MCP server and performs the standard initialization sequence followed by `tools/list`. The complete live tool catalogue — names, descriptions, and input schemas — is inserted into the model's system context.

This is deliberately dynamic. The agent does not assume that AIONS has a particular number of tools. If the MCP server exposes a new tool, the agent can discover it without recompilation.

## 3. Execution loop

```text
Task
  -> live MCP tool catalogue
  -> WPC Qwen3-Coder-30B-A3B v4
  -> TOOL_CALL {name, arguments}
  -> AIONS MCP tools/call
  -> TOOL_RESULT
  -> transcript
  -> next model turn
```

The model is constrained to emit either a `TOOL_CALL` record or a `FINAL` record. Tool results are appended to the transcript and the loop continues until a final response or the configured turn limit is reached.

## 4. Current implementation boundary

The MCP connection is persistent across turns, but the WPC runtime is currently launched as a subprocess for each model turn. This avoids coupling the agent to the runtime's internal Rust model API and makes the integration easy to test.

The next performance step is to make the WPC runtime long-lived across the entire agent session. That would keep model weights resident and, more importantly, make persistent KV-cache reuse possible between tool turns instead of rebuilding the model execution state for every turn.

## 5. Relation to the main performance roadmap

The agent layer does not solve the principal runtime limitation identified in revision 2: one forward pass per token. The next architectural optimisation remains batched forward execution. Once that exists, the same agent loop can benefit from fast prompt prefill, speculative verification, and expert-grouped execution.

The important separation is therefore:

- **WPC** — compressed model storage and execution.
- **Router/scheduler** — decides which specialists/tools are required.
- **AIONS MCP** — exposes the available external capabilities.
- **KV cache** — keeps active model state on the hot path; it should not be replaced by SSD/CBMS during token generation.
- **CBMS** — persistent storage for knowledge, transcripts, and potentially reusable KV snapshots outside the real-time token path.

This addendum records the integration as an engineering milestone without claiming that the remaining throughput work has already been measured.
