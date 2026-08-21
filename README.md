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
**two-plane layout** — a 64-byte low plane holding bits 0–3 and a 32-byte high plane holding
bits 4–5, permuted at pack time — lets the decode path recover eight weights per AVX2
instruction group with no cross-lane shuffles. On a dense model this is the difference between
the packing costing 65% of decode throughput and costing nothing.

At four bits the problem disappears entirely. A 4-bit code divides a byte exactly, so
extraction is a shift and a mask — no shuffles, no planes, no pack-time permutation.

**Layout matters outside the block, too.** Sorting tensors so that each expert occupies one
contiguous run — instead of scattering its three projections across the artifact — is worth
**+45% throughput on its own**, with every weight value left untouched and reconstruction
bit-identical.

Routers in mixture-of-experts models are deliberately left uncompressed: expert selection is a
discrete argmax, where quantisation error would change *which* experts run rather than merely
by how much.

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

The repository also ships `aions-agent`, a thin autonomous loop that connects the WPC runtime
to AIONS over MCP. At startup it performs the standard `initialize` handshake followed by
`tools/list`, so the live tool catalogue — names, descriptions and input schemas — is
discovered at run time rather than compiled in. Adding a new AIONS tool therefore requires no
change to the agent binary.

The AIONS MCP server in `jpytka666-jpg/aions-mcp-server` is documented to expose stdio through
Docker with:

```
docker exec -i aions-mcp python -m src stdio
```

Use the command as program + arguments because `aions-agent` intentionally does not invoke a
shell for the MCP child:

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

For repeat use, the same server command can be supplied by whatever launcher or wrapper you
use to populate `--mcp-command` and repeated `--mcp-arg` values. The agent keeps the MCP
connection alive across turns and can optionally require approval before every tool call with
`--ask`.

```
user task
  -> AIONS agent / router
  -> live MCP tools/list
  -> WPC Qwen3-Coder-30B-A3B v4
  -> TOOL_CALL
  -> AIONS MCP server
  -> TOOL_RESULT
  -> next agent turn
```

The current implementation launches `wpc-runtime` afresh for each model turn. Making the
runtime long-lived — so that weights and the KV cache stay resident across the whole tool loop —
is the next performance step. Details are in
[WHITEPAPER_ADDENDUM_AIONS_AGENT.md](WHITEPAPER_ADDENDUM_AIONS_AGENT.md).

---

## Known limitation

The engine processes **one token at a time**. There is no batched forward pass, so reading a
prompt costs the same per token as writing a reply, and batched prefill, speculative decoding,
and expert-grouped execution are all blocked behind it. This is the principal outstanding item
of work, and the agent loop above does not change it.

---

## Repository layout

| Crate | Purpose |
|---|---|
| `wpc-core` | Quantisation encoders, codebooks, safetensors reader |
| `wpc-format` | On-disk block formats and packing/unpacking |
| `wpc-compiler` | Command-line model compiler |
| `wpc-runtime` | Inference engine: attention, RoPE, norms, MoE routing, sampling |
| `wpc-runtime/src/bin/aions-agent.rs` | AIONS MCP-aware agent loop |
| `wpc-eval` | Fused AVX2 kernels and correctness tests |

---

## Building

```
cargo build --release
cargo test
```

93 tests pass. Requires a CPU with AVX2 and FMA. No other dependencies.

---

## Licence and a request

**Free and Open Source. However, if you monetize this project, you are kindly asked to
donate 1% of your profits to a charity supporting neurodivergent individuals, honoring
the project author's request.**

This is a request, not a licence condition. It is made in good faith and left to yours.
