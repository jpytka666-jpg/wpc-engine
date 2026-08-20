# WPC: Weight-Pattern Compression for Sparse Mixture-of-Experts Inference on Commodity CPUs

**Technical Report — 20 August 2026**

Author: Marcin Szul
Engine: `wpc-engine` (Rust, no external inference dependencies)

---

## Abstract

We report the first end-to-end evaluation of WPC (Weight-Pattern Compression) applied to
a sparse Mixture-of-Experts (MoE) language model, Qwen3-Coder-30B-A3B, executed entirely
on a 2016-vintage quad-core mobile CPU with no GPU acceleration.

The compression scheme reduces the model from 57.0 GB (bfloat16) to 22.21 GB at an
effective rate of 6.25 bits per weight — a 2.57x reduction — while preserving generation
quality sufficient for correct code synthesis and correct tool-call emission.

The principal finding is negative and actionable: although MoE sparsity reduces per-token
weight traffic by 3.66x relative to a comparable dense model, the measured decode throughput
is *lower* than that dense baseline. We attribute this inversion to three quantified causes —
thread under-utilisation, prefetch-hostile expert access patterns, and the absence of weight
reuse across the cache hierarchy — and we identify the corresponding optimisation headroom.

---

## 1. System Under Test

### 1.1 Hardware

| Component | Specification |
|---|---|
| CPU | Intel Core i7-6820HQ (Skylake-H, 2016) |
| Cores / threads | 4 physical / 8 logical |
| Base clock | 2.70 GHz |
| SIMD | AVX2 + FMA (256-bit) |
| L1d cache | 32 KiB per core (128 KiB total) |
| L2 cache | 256 KiB per core (1 MiB total) |
| L3 cache | 8 MiB shared |
| System memory | 39 GiB available to the runtime |
| Memory subsystem | DDR4 dual-channel; ~10 GB/s measured effective streaming bandwidth |
| GPU used | None |

### 1.2 Model

Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`), bfloat16 source distribution.

| Hyperparameter | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 48 |
| `num_attention_heads` | 32 (`head_dim` 128) |
| `num_key_value_heads` | 4 (grouped-query attention) |
| `num_experts` | 128 |
| `num_experts_per_tok` | 8 |
| `moe_intermediate_size` | 768 |
| `vocab_size` | 151 936 |
| `tie_word_embeddings` | false |
| Sparse layers | all 48 (`mlp_only_layers` empty, `decoder_sparse_step` = 1) |

### 1.3 Compression Scheme (WPC v3)

WPC v3 applies per-block affine quantisation with bit-packed codes. Each block of 128
consecutive fp32 weights is encoded as:

- `zero_point` : fp16 (2 bytes)
- `scale` : fp16 (2 bytes)
- 128 × 6-bit codes, bit-packed into 96 bytes

Total: **100 bytes per 128 weights = 6.25 bits/weight**.

Reconstruction is `w = zero_point + code * scale`.

WPC v3 is bit-identical to the earlier byte-aligned v2 encoding (8.25 bits/weight) by
construction, while occupying 24.2% less space.

**Codeword layout — status.** The measurements in this report use the *four-codes-per-three-bytes*
packing, in which extracting eight consecutive weights requires cross-lane byte shuffles.

A revised **two-plane layout** — a 64-byte low plane (bits 0–3, two codes/byte) and a 32-byte
high plane (bits 4–5, four codes/byte), permuted at pack time so eight consecutive output
weights derive from eight consecutive bytes of each plane — permits a shuffle-free AVX2 decode
path. It is implemented and validated on a separate branch but is **not present in the build
measured here**, and the Qwen3-Coder artifact was compiled with the original layout.

Validation of the two-plane layout on Qwen2.5-0.5B (16 tests passing, identical token ids):

| Layout | 30 tokens |
|---|---|
| v2 (8.25 bits, byte-aligned) | 3.43 s |
| v3, original packing | 7.28 s |
| v3, two-plane packing | **3.31 s** |

The original packing paid a 65% decode penalty that erased the density gain; the two-plane
layout removes it. Its effect on the MoE workload is **unmeasured** — §5 shows that model is
bounded by memory access, not by decode arithmetic, so the gain there is expected to be small.
Merging it and re-measuring is outstanding work.

**Router weights are held uncompressed.** Expert selection is a discrete argmax over 128
logits; quantisation error there changes *which* experts run, not merely by how much, and
the routers are numerically negligible (0.26 M of 623 M parameters per layer).

---

## 2. Compression Results

| Metric | Value |
|---|---|
| Source (bfloat16), 16 shards | 57.0 GB |
| WPC v3 compiled artifact | **22.21 GB** |
| WPC v2 equivalent (for reference) | 29.31 GB |
| v3 saving over v2 | 7.11 GB (24.2%) |
| Overall reduction vs. source | 2.57x |
| Tensors encoded | 18 626 |
| Reconstruction error (per-block affine, measured on Gemma-12B `k_proj`) | 2.78% |

---

## 3. Parameter and Traffic Accounting

Derived analytically from the configuration in §1.2.

### 3.1 Per-layer parameter budget

| Component | Parameters |
|---|---|
| `q_proj` (2048 × 4096) | 8.389 M |
| `k_proj` (2048 × 512) | 1.049 M |
| `v_proj` (2048 × 512) | 1.049 M |
| `o_proj` (4096 × 2048) | 8.389 M |
| Attention subtotal | **18.874 M** |
| Router (2048 × 128) | 0.262 M |
| One expert (3 × 2048 × 768) | 4.719 M |
| All 128 experts | **604.0 M** |
| **Layer total** | **623.1 M** |

### 3.2 Whole-model totals

| Quantity | Value |
|---|---|
| 48 decoder layers | 29.91 B |
| `embed_tokens` (151 936 × 2048) | 0.311 B |
| `lm_head` (untied) | 0.311 B |
| **Total parameters** | **30.53 B** |

### 3.3 Activated parameters per decoded token

| Component | Parameters |
|---|---|
| Attention (dense, all layers) | 18.874 M × 48 = 0.906 B |
| Routers | 0.262 M × 48 = 0.013 B |
| Experts (8 of 128, all layers) | 37.749 M × 48 = 1.812 B |
| `lm_head` (full vocabulary projection) | 0.311 B |
| **Total activated** | **3.042 B** |

**Sparsity ratio: 3.042 / 30.53 = 9.96%.** Approximately one tenth of the model participates
in each token.

### 3.4 Weight traffic per token

At 6.25 bits/weight (0.78125 bytes):

**3.042 B × 0.78125 = 2.377 GB of weight traffic per decoded token.**

---

## 4. Measured Performance

All figures from instrumented runs (`/usr/bin/time -v`), greedy decoding, batch size 1.

| Trial | Workload | Prefill | Decode | Throughput | CPU | Peak RSS |
|---|---|---|---|---|---|---|
| 1 | Code synthesis | 8 tok / 118.04 s (cold) | 60 tok / 86.02 s | **0.70 tok/s** | 158% | 14.93 GB |
| 2 | Technical exposition | 19 tok / 23.87 s (warm) | 60 tok / 66.60 s | **0.90 tok/s** | 267% | 12.86 GB |
| 3 | Tool-call emission | 29 tok / 28.76 s (warm) | 9 tok / 8.85 s | **1.02 tok/s** | — | — |

Model load time is 0.23–0.49 s: the compiled artifact is memory-mapped, not copied.

Trial 1's 118 s prefill reflects a cold page cache; trials 2 and 3 execute against a warm
cache and are the representative figures.

### 4.1 Effective bandwidth utilisation

Steady-state decode (trial 2): 2.377 GB/token ÷ 1.11 s/token = **2.14 GB/s effective**.

Against ~10 GB/s of practically attainable streaming bandwidth, the engine achieves
**21% of the memory subsystem's capability**.

### 4.2 Generation quality

**Code synthesis.** Prompt `def binary_search(arr, target):` produced a textbook-correct
implementation — correct half-open interval handling, correct midpoint computation,
correct branch structure. Truncated only by the 60-token budget.

**Technical exposition.** The model correctly characterised a hash map and, without
prompting for it, correctly enumerated the cases where it is the wrong structure
(order maintenance, range queries, memory pressure).

**Tool-call emission.** The model emitted `read_file('config.json')` and nothing else,
then **terminated on token 151645 (end-of-turn)**. Instruction-following and stop-token
handling are both correct — a prerequisite for agentic use.

No degeneration, repetition loops, or token-space artifacts were observed at 6.25 bits/weight.

---

## 5. Analysis: Why Sparsity Does Not Translate into Speed

### 5.1 The inversion

| Model | Traffic/token | Measured | Effective bandwidth |
|---|---|---|---|
| Gemma-12B (dense, WPC v3) | 8.70 GB | 1.06 tok/s | 9.22 GB/s |
| Qwen3-Coder-30B-A3B (sparse, WPC v3) | 2.377 GB | 0.90 tok/s | 2.14 GB/s |

The MoE model reads **3.66x less data per token** yet decodes **15% slower**. The entire
sparsity advantage is consumed — and then some — by a **4.3x collapse in effective
bandwidth**. This is the central result.

### 5.2 Cause 1 — Thread under-utilisation

Observed CPU occupancy is 158–267% of an available 800%. Between 2.0 and 3.3 of 8 logical
threads are doing work. The dense path, streaming contiguous rows, parallelises naturally;
the MoE path evidently serialises around expert dispatch.

*Estimated headroom: 2.4–4.0x.* This is the largest single recoverable factor and requires
no change to the compression format.

### 5.3 Cause 2 — Prefetch-hostile access pattern

Each layer selects 8 of 128 experts by runtime argmax. The selected expert weights occupy
8 × 3.69 MB = **29.5 MB scattered across a 22.21 GB mapping**, at offsets not known until
the router has executed.

Hardware prefetchers detect strides; they cannot anticipate a data-dependent gather across
gigabyte distances. Every expert access therefore begins with a cold TLB and cache miss, and
the memory controller services a random-access pattern at a fraction of its sequential rate.
The dense baseline, by contrast, walks its weights linearly and is prefetched perfectly.

*Mitigations:* expert-major memory layout with 2 MiB huge pages to reduce TLB pressure;
speculative prefetch issued immediately after the router argmax, overlapping fetch latency
with the attention block of the same layer.

### 5.4 Cause 3 — No weight reuse in the cache hierarchy

The working set of one layer's activated experts (29.5 MB) exceeds the 8 MiB L3 by 3.7x,
and the next token selects a *different* expert subset. Consequently every weight byte is
fetched from DRAM, consumed by a single fused multiply-add, and evicted.

Arithmetic intensity is therefore **2 FLOPs per byte** — firmly in the memory-bound regime
of the roofline model, with no reuse available to amortise the fetch.

The hierarchy is used correctly where it can be: the 2048-element activation vector
(8 KiB fp32) resides in L1d (32 KiB) and is reused across all eight expert evaluations
within a layer. The problem is not cache management; it is that MoE weights are
fundamentally single-use at batch size 1.

**The structural remedy is batching.** At batch size *B*, one expert fetch serves *B* tokens,
raising arithmetic intensity to 2*B* FLOPs/byte. This converts the workload from
memory-bound toward compute-bound and is the only technique that attacks the root cause
rather than its symptoms.

### 5.5 Prefill is not batched

Warm prefill costs 0.99–1.26 s/token — indistinguishable from decode. Prompt tokens are
being processed sequentially, forfeiting the one situation in which all tokens are known
in advance and weight fetches could be amortised across them.

*Estimated headroom: large.* A batched prefill would make long prompts nearly free relative
to the current linear cost, which presently dominates short interactions (trial 3 spent
28.76 s on prefill to produce 8.85 s of output).

---

## 6. Viability Assessment for AIONS Integration

**Current state: not viable for interactive dispatch.**
At 0.90 tok/s (~40 words/minute) the model is below the threshold for conversational use.
Correctness, instruction-following, and tool-call formation are all adequate; throughput
is the sole blocker.

**Projected state after the identified optimisations:**

| Optimisation | Factor | Cumulative |
|---|---|---|
| Baseline | — | 0.90 tok/s |
| Full thread utilisation | 2.4–4.0x | 2.2–3.6 tok/s |
| Expert-major layout + huge pages + speculative prefetch | 1.3–1.8x | 2.8–6.5 tok/s |
| Batched prefill | (prompt-side only) | — |

A sustained 3–5 tok/s (130–220 words/minute) is the realistic target and would place the
model within interactive range. None of these optimisations require modifying the
compression format, retraining, or additional hardware.

**Memory footprint is not a constraint:** peak RSS of 14.93 GB against 39 GB available
leaves ample headroom for extended context.

---

## 7. Conclusions

1. WPC v3 compresses a 30-billion-parameter MoE model by **2.57x** (57.0 GB → 22.21 GB)
   at 6.25 bits/weight with **no observable degradation** in code synthesis, technical
   exposition, or tool-call emission.

2. Bit-packing to 6.25 bits/weight yields a 24.2% density gain over byte-aligned encoding.
   Realising that gain without a decode-throughput penalty requires a layout designed for
   shuffle-free AVX2 extraction: the original packing cost 65% of decode throughput on a
   dense model, while the two-plane layout recovers it in full (3.31 s vs 3.43 s for v2).
   The two-plane layout is validated but not yet merged into the measured build.

3. A 30B-parameter model runs on a 2016 quad-core mobile CPU with no GPU. This is the
   qualitative result.

4. **MoE sparsity is currently a liability rather than an asset in this engine.** Despite
   3.66x less weight traffic per token, the sparse model decodes 15% slower than a dense
   baseline, because scattered expert access collapses effective bandwidth by 4.3x.

5. The performance deficit is **implementation-bound, not format-bound**. Thread
   utilisation of 2–3 cores out of 8, unbatched prefill, and prefetch-hostile expert
   layout account for the gap. All three are addressable within the existing architecture.

---

## 8. Reproduction

```
wpc-compiler --input <bf16 model dir> --output <artifact dir> --scheme v3

wpc-runtime --model <norms + tokenizer dir> \
            --wpc <artifact dir> \
            --scheme v3 \
            --arch qwen3-moe \
            --prompt "def binary_search(arr, target):" \
            --max-tokens 60
```

Architecture is auto-detected from `model_type: "qwen3_moe"`; `--arch` is optional.

---

*All figures in this report are measured on the system described in §1.1 unless explicitly
labelled as derived. Derived quantities (§3) follow analytically from the published model
configuration and the compression rate, and are consistent with the measured artifact size
to within 0.1%.*
