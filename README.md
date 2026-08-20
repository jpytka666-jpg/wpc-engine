# wpc-engine

**Weight-Pattern Compression — a tensor compilation and inference engine in pure Rust.**

Runs large language models on ordinary CPUs by compressing their weights to 6.25 bits
each, with no GPU, no Python runtime, and no external inference dependencies.

---

## What it does

A 30-billion-parameter mixture-of-experts model — Qwen3-Coder-30B-A3B — compresses from
**57.0 GB to 22.21 GB** and generates correct code on a 2016 quad-core laptop CPU.

| Model | Source | Compressed | Ratio |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (MoE) | 57.0 GB | 22.21 GB | 2.57x |
| Gemma-12B-it (dense) | 23.0 GB | 8.70 GB | 2.64x |
| Qwen2.5-0.5B (dense) | 0.95 GB | 0.37 GB | 2.57x |

Quality is preserved: at 6.25 bits/weight the coder model produces textbook-correct
implementations, follows instructions, emits well-formed tool calls, and terminates
correctly on end-of-turn tokens.

Full measurements, parameter accounting, and performance analysis are in
[WHITEPAPER.md](WHITEPAPER.md).

---

## How it works

Weights are quantised per block of 128 values using an affine map:

```
w = zero_point + code * scale
```

Each block stores a 16-bit `zero_point`, a 16-bit `scale`, and 128 six-bit codes packed
into 96 bytes — **100 bytes per 128 weights, exactly 6.25 bits per weight**.

How those codes are arranged inside the 96 bytes matters as much as the density. A
**two-plane layout** — a 64-byte low plane holding bits 0–3 and a 32-byte high plane
holding bits 4–5, permuted at pack time so eight consecutive output weights come from
eight consecutive bytes of each plane — lets the decode path recover eight weights per
AVX2 instruction group with no cross-lane shuffles. On a dense model this is the
difference between the packing costing 65% of decode throughput and costing nothing.

The two-plane layout is implemented and validated on a branch; the current default build
still uses the earlier packing.

Routers in mixture-of-experts models are deliberately left uncompressed: expert selection
is a discrete argmax, where quantisation error would change *which* experts run rather
than merely by how much.

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
wpc-compiler --input <bf16 model dir> --output <artifact dir> --scheme v3
```

Run it:

```
wpc-runtime --model <tokenizer + norms dir> \
            --wpc <artifact dir> \
            --scheme v3 \
            --prompt "def binary_search(arr, target):" \
            --max-tokens 60
```

Compression schemes: `v1` (vector-quantised codebook, superseded), `v2` (byte-aligned
affine, 8.25 bits/weight), `v3` (bit-packed affine, 6.25 bits/weight — recommended).

---

## Repository layout

| Crate | Purpose |
|---|---|
| `wpc-core` | Quantisation encoders, codebooks, safetensors reader |
| `wpc-format` | On-disk block formats and packing/unpacking |
| `wpc-compiler` | Command-line model compiler |
| `wpc-runtime` | Inference engine: attention, RoPE, norms, MoE routing, sampling |
| `wpc-eval` | Fused AVX2 kernels and correctness tests |

---

## Building

```
cargo build --release
cargo test
```

Requires a CPU with AVX2 and FMA. No other dependencies.

---

## Licence and a request

**Free and Open Source. However, if you monetize this project, you are kindly asked to
donate 1% of your profits to a charity supporting neurodivergent individuals, honoring
the project author's request.**

This is a request, not a licence condition. It is made in good faith and left to yours.
