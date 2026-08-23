# WPC/AIONS Agent Integration — Addendum to WHITEPAPER.md

**23 August 2026 — resident runtime milestone**

## 1. Purpose

The WPC runtime provides an AIONS-facing autonomous agent loop. The design intentionally separates model execution from agent control: WPC remains the inference engine, while the agent provides planning, tool selection, execution, and transcript management.

## 2. Dynamic tool discovery

The agent connects to an AIONS MCP server and performs the standard initialization sequence followed by `tools/list`. The complete live tool catalogue — names, descriptions, and input schemas — is inserted into the model's system context.

This is deliberately dynamic. The agent does not assume that AIONS has a particular number of tools. If the MCP server exposes a new tool, the agent can discover it without recompilation.

## 3. Execution loop

```text
Task
  -> live MCP tool catalogue
  -> resident WPC Qwen3-Coder-30B-A3B v4 session
  -> TOOL_CALL {name, arguments}
  -> AIONS MCP tools/call
  -> TOOL_RESULT
  -> transcript
  -> next model turn using the same resident session/KV state
```

The model is constrained to emit either a `TOOL_CALL` record or a `FINAL` record. Tool results are appended to the transcript and the loop continues until a final response or the configured turn limit is reached.

## 4. Resident runtime milestone

The previous per-model-turn subprocess boundary has been replaced by a resident Rust runtime/session path.

- `ResidentEngine` loads the WPC v4 model once for the agent task.
- `ResidentSession` remains alive across agent turns.
- The session reuses a shared prompt prefix when a later turn extends the same prefix.
- Active KV state remains resident in the session and is truncated back to the prompt boundary after generation.
- WPC v4 packed weights are mmap-backed and shared through `Arc`, so model data is loaded once and reused by the model layers.
- Reusable KV/mmap capacity is covered by regression tests and full CI verification.

This closes the Phase 1 resident-runtime milestone. Fine-grained scratch-buffer pooling inside individual token execution remains a separate profiling-driven optimisation and is not required for resident correctness.

## 5. Relation to the main performance roadmap

The resident runtime does not remove the principal compute limitation: the current hot path still has a single-token reference path, while batched prefill/forward is the next major throughput step.

The intended sequence is:

```text
resident runtime
   -> resident KV
   -> batched prefill / forward
   -> benchmark against Rust reference
   -> Mojo / CUDA / CPU backend specialisation
```

Any acceleration layer must preserve correctness, precision, determinism and stability before replacing or becoming the preferred backend.

## 6. Memory boundary

The important separation remains:

- **WPC** — compressed model storage and execution.
- **Rust runtime/session** — lifecycle, scheduling, active model state and hot execution orchestration.
- **Router/scheduler** — decides which specialists/tools are required.
- **AIONS MCP** — exposes the available external capabilities.
- **KV cache** — active model state on the real-time hot path; it stays resident during token generation.
- **CBMS** — persistent storage for knowledge, transcripts, and optional reusable snapshots outside the real-time token path.

CBMS is therefore not used as a replacement for the live KV cache during token generation.

## 7. Current engineering status

Phase 1 is considered complete only after the resident-runtime regression tests, full workspace build/tests, formatting, runtime Clippy, and benchmark compile/smoke gates are green. The remaining throughput work is intentionally deferred to the batched execution and compute-backend phases.
