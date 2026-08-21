# wpc-engine

**Weight-Pattern Compression — a tensor compilation and inference engine in pure Rust.**

Runs large language models on ordinary CPUs by compressing their weights to as few as 4.25 bits each.

## AIONS agent integration

The repository includes `aions-agent`, a thin autonomous loop connecting the WPC runtime to AIONS through MCP. It dynamically calls `initialize` and `tools/list` at startup, so it does not hard-code a fixed number of AIONS tools.

Execution path:

```text
user task
  -> AIONS agent / router
  -> live MCP tools/list
  -> WPC Qwen3-Coder-30B-A3B v4
  -> TOOL_CALL
  -> AIONS MCP server
  -> TOOL_RESULT
  -> next agent turn
```

Adding a new AIONS MCP tool therefore does not require changing the agent binary.

Example:

```text
AIONS_MCP_COMMAND=/path/to/aions-mcp-server cargo run --release --bin aions-agent -- --task "inspect the repository and fix the failing test" --model /home/aions/qwen3-coder-model --wpc /home/aions/qwen3-coder-wpc4 --scheme v4 --max-turns 6
```

The MCP connection stays alive across agent turns. The current implementation launches `wpc-runtime` for each model turn; the next performance step is a long-lived runtime so the model and KV cache remain resident throughout the tool loop.

## Current WPC results

Qwen3-Coder-30B-A3B: 57.0 GB source -> 15.10 GB v4 artifact, 4.25 bits/weight, measured 2.35 tok/s on the 2016 quad-core CPU used in the study. Tensor ordering alone gave +45% throughput with bit-identical reconstruction. The 2.25-bit build is smaller but slower and degenerate.

See `WHITEPAPER.md` for the full revision-2 measurements.

## Usage

```text
wpc-compiler --input <bf16 model dir> --output <artifact dir> --scheme v4
wpc-runtime --model <tokenizer + norms dir> --wpc <artifact dir> --scheme v4 --prompt "def binary_search(arr, target):" --max-tokens 60
```

## Known limitation

The runtime still performs one forward pass per token. Batched prefill, speculative decoding, and expert-grouped execution remain the main performance work. The AIONS agent is therefore an orchestration proof, not yet the final high-throughput agent runtime.

## Repository layout

- `wpc-core` — quantisation and safetensors primitives
- `wpc-format` — on-disk formats
- `wpc-compiler` — model compiler
- `wpc-runtime` — inference engine
- `wpc-runtime/src/bin/aions-agent.rs` — AIONS MCP-aware agent
- `wpc-eval` — SIMD kernels and tests

## Building

```text
cargo build --release
cargo test
```

Requires AVX2 + FMA.
