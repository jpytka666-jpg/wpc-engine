# wpc-engine

**Weight-Pattern Compression — a tensor compilation and inference engine in pure Rust.**

Runs large language models on ordinary CPUs by compressing their weights to as few as
4.25 bits each, with no GPU, no Python runtime, and no external inference dependencies.

---

## What it does

A 30-billion-parameter mixture-of-experts model — Qwen3-Coder-30B-A3B — compresses from
**57.0 GB to 15.10 GB** and generates correct code on a 2016 quad-core laptop CPU, at
**2.35 tokens per second — roughly 120 words per minute**.

| Model | Source | v3 (6.25 bit) | v4 (4.25 bit) | Best ratio |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B (MoE) | 57.0 GB | 22.21 GB | **15.10 GB** | 3.77x |
| Qwen3-4B (dense) | — | 3.0 GB | **2 038 MiB** | — |
| Gemma-12B-it (dense) | 23.0 GB | 8.70 GB | — | 2.64x |
| Qwen2.5-0.5B (dense) | 0.95 GB | 0.37 GB | — | 2.57x |

**v4 is the recommended scheme, and the obvious choice whenever memory is tight** — a small
GPU, a laptop, or anything that has to hold the model in a limited amount of RAM. It is not a
quality trade: on the dense control model, the 4.25-bit build emits **token ids identical to
the 6.25-bit build**, and on the coder model it produces correct code, correct tool calls, and
correct end-of-turn termination.

Going below four bits does not pay. A 2.25-bit build is 47% smaller and *slower*, and its
output degenerates — below roughly 2 GB the model stops being limited by memory bandwidth, so
further compression buys nothing and costs everything.

Full measurements, parameter accounting, and performance analysis are in
[WHITEPAPER.md](WHITEPAPER.md).

---

## How it works

Weights are quantised per block of 128 values using an affine map:

```
w = zero_point + code * scale
```

Each block stores a 16-bit `zero_point`, a 16-bit `scale`, and the codes themselves:

| Scheme | Codes | Bytes per block | Bits per weight |
|---|---|---|---|
| v3 | 128 × 6-bit, bit-packed | 100 | 6.25 |
| **v4** | **128 × 4-bit, two per byte** | **68** | **4.25** |

At six bits, how the codes are arranged inside the block matters as much as the density: a
**two-plane layout** — a 64-byte low plane holding bits 0–3 and a 32-byte high plane holding bits 4–5, permuted at pack time — lets the decode path recover eight weights per AVX2 instruction group with no cross-lane shuffles. On a dense model this is the difference between the packing costing 65% of decode throughput and costing nothing.

At four bits the problem disappears entirely. A 4-bit code divides a byte exactly, so extraction is a shift and a mask — no shuffles, no planes, no pack-time permutation.

**Layout matters outside the block, too.** Sorting tensors so that each expert occupies one contiguous run — instead of scattering its three projections across the artifact — is worth **+45% throughput on its own**, with every weight value left untouched and reconstruction bit-identical.

Routers in mixture-of-experts models are deliberately left uncompressed: expert selection is a discrete argmax, where quantisation error would change *which* experts run rather than merely by how much.

---

## Supported architectures

| Architecture | Status |
|---|---|
| Qwen2 | dense, compressed |
| Qwen3 | dense, compressed |
| Qwen3-MoE | sparse, compressed (router + top-k expert routing) |
| Gemma4 | dense, compressed |

Architecture is auto-detected from `config.json`.

---

## Usage

Compile a model:

```
wpc-compiler --input <bf16 model dir> --output <artifact dir> --scheme v4
```

Run it:

```
wpc-runtime --model <tokenizer + norms dir> \
            --wpc <artifact dir> \
            --scheme v4 \
            --prompt "def binary_search(arr, target):" \
            --max-tokens 60
```

Compression schemes:

| Scheme | Bits/weight | Use |
|---|---|---|
| `v1` | codebook | superseded — see the whitepaper for why the premise failed |
| `v2` | 8.25 | superseded, byte-aligned affine |
| `v3` | 6.25 | supported |
| `v4` | **4.25** | **recommended, and required when memory is tight** |
| `v5` | 2.25 | implemented and rejected; output degenerates |

---

## AIONS agent integration

The repository ships `aions-agent`, a thin autonomous loop connecting the WPC runtime to AIONS over MCP. At startup it performs the standard `initialize` handshake followed by `tools/list`, so the live tool catalogue is discovered at run time rather than compiled in.

The important runtime property is now **resident model state**: the Qwen3-MoE WPC v4 model is loaded once when `aions-agent` starts and remains resident across agent turns. The previous implementation launched `wpc-runtime` for every turn, repeatedly paying model-load cost. The resident API keeps the model weights in memory and creates a fresh KV cache per task turn.

The AIONS MCP server is documented to expose stdio through Docker with:

```
docker exec -i aions-mcp python -m src stdio
```

Use the command as program + arguments because `aions-agent` intentionally does not invoke a shell for the MCP child:

```
cargo run --release --bin aions-agent -- \
  --mcp-command docker \
  --mcp-arg exec \
  --mcp-arg -i \
  --mcp-arg aions-mcp \
  --mcp-arg python \
  --mcp-arg -m \
  --mcp-arg src \
  --mcp-arg stdio \
  --task "inspect the repository and fix the failing test" \
  --model /home/aions/qwen3-coder-run \
  --wpc /home/aions/qwen3-coder-wpc4 \
  --scheme v4 \
  --max-turns 6
```

Runtime flow:

```
user task
  -> AIONS agent / router
  -> live MCP tools/list
  -> resident WPC Qwen3-Coder-30B-A3B v4
  -> TOOL_CALL
  -> AIONS MCP server
  -> TOOL_RESULT
  -> same resident model
  -> next agent turn
```

The standalone `wpc-resident` binary exposes the same resident model through JSONL for integration and testing. The reusable implementation lives in `wpc-runtime::resident::ResidentEngine`.

---

## Known performance work

The runtime still has important optimisation work ahead. The current standard inference path processes one token at a time. The `integration/aions-unified-stack` branch additionally contains the `BatchEngine`, mmap-backed KV layer, GEMM attention path, correctness tests and benchmarks from `feature/forward-batch-gemm-bench`.

The next step after resident weights is to make KV state reusable across compatible turns and to route batched prompt prefill through `BatchEngine`, so the full prompt is not processed as a sequence of independent single-token forwards.

---

## Repository layout

| Crate / component | Purpose |
|---|---|
| `wpc-core` | Quantisation encoders, codebooks, safetensors reader |
| `wpc-format` | On-disk block formats and packing/unpacking |
| `wpc-compiler` | Command-line model compiler |
| `wpc-runtime` | Inference engine: attention, RoPE, norms, MoE routing, sampling |
| `wpc-runtime/src/resident.rs` | Long-lived Qwen3-MoE WPC runtime API |
| `wpc-runtime/src/forward_batch.rs` | Batched attention, mmap KV storage and GEMM path |
| `wpc-runtime/src/bin/aions-agent.rs` | AIONS MCP-aware resident agent loop |
| `wpc-runtime/src/bin/wpc-resident.rs` | JSONL resident-runtime process |
| `wpc-eval` | Fused AVX2 kernels and correctness tests |

---

## Building and testing

```
cargo build --workspace --release
cargo test --workspace --release
cargo fmt --all --check
cargo clippy -p wpc-runtime --all-targets --release -- -D warnings
```

The integration branch CI also compiles the attention benchmark and runs a short benchmark smoke test.

---

## Licence and a request

**Free and Open Source. However, if you monetize this project, you are kindly asked to donate 1% of your profits to a charity supporting neurodivergent individuals, honoring the project author's request.**

This is a request, not a licence condition. It is made in good faith and left to yours.
