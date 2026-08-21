# WPC: Weight-Pattern Compression for Sparse Mixture-of-Experts Inference on Commodity CPUs

**Technical Report — 21 August 2026 (revision 2)**

Author: Marcin Szul
Engine: `wpc-engine` (Rust, no external inference dependencies)

---

## Abstract

We report an end-to-end evaluation of WPC (Weight-Pattern Compression) applied to a sparse
Mixture-of-Experts (MoE) language model, Qwen3-Coder-30B-A3B, executed entirely on a
2016-vintage quad-core mobile CPU with no GPU acceleration.

Revision 1 established that the model compresses from 57.0 GB (bfloat16) to 22.21 GB at
6.25 bits per weight with no observable quality loss, and reported a negative headline result:
despite 3.66x less weight traffic per token than a comparable dense model, the sparse model
decoded *slower*.

This revision reports five further results.

1. **A 4.25-bit encoding (v4).** The artifact falls to **15.10 GB** — a 3.77x reduction from
   source — and decode throughput rises from 1.00 to **2.35 tok/s**, roughly 51 to 120 words
   per minute. Quality was checked on twelve tasks; tool-call emission was correct in all three
   cases tested.

2. **Tensor ordering is worth 45% on its own.** Storing each expert as one contiguous run,
   rather than scattering its three projections across the artifact, raised throughput from
   1.16 to 1.68 tok/s at an unchanged bit width. **No weight value is altered; reconstruction
   is bit-identical.** The gain was predicted at 3–4x and measured at 1.45x; §5.4 records why
   the original estimate was wrong.

3. **A negative result at 2.25 bits (v5).** The artifact is 47% smaller than v4 and *slower*,
   and its output degenerates. The point at which further compression stops buying speed is
   therefore locatable, and on this hardware it lies at roughly 2 GB. Four bits per weight is
   an optimum, not a compromise.

4. **A post-mortem of the abandoned v1 codebook scheme.** With every known implementation bug
   repaired, v1 still reconstructs at 56.0% error against v3's 2.39%. Three independent lines
   of evidence show the failure is a property of the weight distribution, not of the code.

5. **Three model families in this study share a byte-identical tokenizer**
   (151 643 entries, SHA-256 prefix `63a10eac44df16bb`), which is the precondition for
   speculative decoding with a small drafter.

The throughput inversion of revision 1 is resolved — the sparse model now decodes 2.2x faster
than the dense baseline — but the underlying efficiency deficit is not: the engine still
attains only 38% of practically available memory bandwidth. The principal remaining limitation
is architectural and stated plainly in §10: the engine performs **one forward pass per token**,
which blocks batched prefill, speculative decoding, and expert-grouped execution alike.

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

### 1.2 Models

The primary subject is Qwen3-Coder-30B-A3B (`Qwen3MoeForCausalLM`), bfloat16 source
distribution.

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

Two dense models serve as controls, because a dense model isolates the effect of a change from
the confounding influence of expert routing:

| Control model | Role |
|---|---|
| Qwen3-4B | Bit-width scaling study (§6) — small enough to rebuild in every scheme |
| Qwen2.5-0.5B | Encoder-error measurements (§7) — small enough to measure a single tensor exhaustively |

### 1.3 Compression Schemes

All WPC schemes from v2 onward apply **per-block affine quantisation**: a block of 128
consecutive weights is reconstructed as `w = zero_point + code * scale`, where `zero_point` and
`scale` are fp16 values stored once per block. The schemes differ only in how many bits each
code occupies and how the codes are packed.

| Scheme | Bytes per 128 weights | Bits/weight | Status |
|---|---|---|---|
| v1 | (variable, codebook) | ~3.00 effective | Abandoned — see §7 |
| v2 | 132 | 8.25 | Superseded, byte-aligned |
| v3 | 100 | 6.25 | Supported |
| **v4** | **68** | **4.25** | **Recommended** |
| v5 | 36 | 2.25 | Built, measured, **rejected** — see §6 |

#### 1.3.1 WPC v3 — 6.25 bits/weight

- `zero_point` : fp16 (2 bytes)
- `scale` : fp16 (2 bytes)
- 128 × 6-bit codes, bit-packed into 96 bytes

Total: **100 bytes per 128 weights = 6.25 bits/weight**. v3 is bit-identical to v2 by
construction while occupying 24.2% less space.

**Codeword layout — status.** The v3 measurements in this report use the
*four-codes-per-three-bytes* packing, in which extracting eight consecutive weights requires
cross-lane byte shuffles.

A revised **two-plane layout** — a 64-byte low plane (bits 0–3, two codes/byte) and a 32-byte
high plane (bits 4–5, four codes/byte), permuted at pack time so eight consecutive output
weights derive from eight consecutive bytes of each plane — permits a shuffle-free AVX2 decode
path. It is implemented and validated on a separate branch but is **not present in the build
measured here**.

Validation of the two-plane layout on Qwen2.5-0.5B (identical token ids):

| Layout | 30 tokens |
|---|---|
| v2 (8.25 bits, byte-aligned) | 3.43 s |
| v3, original packing | 7.28 s |
| v3, two-plane packing | **3.31 s** |

The original packing paid a 65% decode penalty that erased the density gain; the two-plane
layout removes it.

#### 1.3.2 WPC v4 — 4.25 bits/weight

- `zero_point` : fp16 (2 bytes)
- `scale` : fp16 (2 bytes)
- 128 × 4-bit codes, **two codes per byte**, 64 bytes

Total: **68 bytes per 128 weights = 4.25 bits/weight**.

The packing problem that §1.3.1 spends four paragraphs on **does not arise at four bits**.
A 4-bit code divides a byte exactly, so extracting a code is a shift and a mask with no
cross-lane movement, no plane permutation, and no pack-time reordering. The layout question
that dominated v3 engineering is dissolved by the width itself rather than solved.

The header cost is unchanged at 4 bytes per block, so it rises from 0.25 to 0.25 bits/weight in
absolute terms but from 4.0% to 5.9% of the budget — still negligible, and worth contrasting
with v1, which spent half its budget on headers (§7.4).

#### 1.3.3 WPC v5 — 2.25 bits/weight

- `zero_point` : fp16 (2 bytes)
- `scale` : fp16 (2 bytes)
- 128 × 2-bit codes, four codes per byte, 32 bytes

Total: **36 bytes per 128 weights = 2.25 bits/weight**. Four representable levels per block.
The scheme is implemented and measured; §6 reports why it is rejected.

#### 1.3.4 Routers

**Router weights are held uncompressed in every scheme.** Expert selection is a discrete argmax
over 128 logits; quantisation error there changes *which* experts run, not merely by how much,
and the routers are numerically negligible (0.26 M of 623 M parameters per layer).

---

## 2. Compression Results

### 2.1 Qwen3-Coder-30B-A3B

| Metric | WPC v3 | WPC v4 |
|---|---|---|
| Source (bfloat16), 16 shards | 57.0 GB | 57.0 GB |
| Compiled artifact | 22.21 GB | **15.10 GB** (15 462 MiB) |
| Bits per weight | 6.25 | 4.25 |
| Reduction vs. source | 2.57x | **3.77x** |
| Tensors encoded | 18 626 | 18 626 |

For reference, the superseded v2 encoding of the same model occupies 29.31 GB; v3 saves 7.11 GB
(24.2%) against it, and v4 saves a further 7.11 GB (32.0%) against v3.

### 2.2 Across models

| Model | Source | v3 (6.25 bit) | v4 (4.25 bit) |
|---|---|---|---|
| Qwen3-Coder-30B-A3B (MoE) | 57.0 GB | 22.21 GB | **15.10 GB** |
| Gemma-12B-it (dense) | 23.0 GB | 8.70 GB | not built |
| Qwen3-4B (dense) | — | 3.0 GB | 2 038 MiB |
| Qwen2.5-0.5B (dense) | 0.95 GB | 0.37 GB | not built |

### 2.3 Reconstruction error

Measured per-tensor, as relative reconstruction error against the bfloat16 source:

| Scheme | Tensor | Error |
|---|---|---|
| v3 (per-block affine) | Gemma-12B `k_proj` | 2.78% |
| v3 (per-block affine) | Qwen2.5-0.5B `layers.0.self_attn.k_proj.weight` | **2.39%** |
| v1 (codebook, all bugs fixed) | same Qwen2.5-0.5B tensor | **56.0%** |

The last row is the subject of §7.

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

| Scheme | Bytes/weight | Traffic per decoded token |
|---|---|---|
| v3 | 0.78125 | 2.377 GB |
| **v4** | **0.53125** | **1.616 GB** |

One expert therefore occupies 3.69 MB under v3 and **2.51 MB under v4**; the eight experts
activated in a layer occupy 29.5 MB and **20.1 MB** respectively. Both remain well above the
8 MiB L3 (§9.4).

---

## 4. Measured Performance

All figures from instrumented runs (`/usr/bin/time -v`), greedy decoding, batch size 1.

### 4.1 WPC v3 baseline, source tensor order

| Trial | Workload | Prefill | Decode | Throughput | CPU | Peak RSS |
|---|---|---|---|---|---|---|
| 1 | Code synthesis | 8 tok / 118.04 s (cold) | 60 tok / 86.02 s | **0.70 tok/s** | 158% | 14.93 GB |
| 2 | Technical exposition | 19 tok / 23.87 s (warm) | 60 tok / 66.60 s | **0.90 tok/s** | 267% | 12.86 GB |
| 3 | Tool-call emission | 29 tok / 28.76 s (warm) | 9 tok / 8.85 s | **1.02 tok/s** | — | — |

Model load time is 0.23–0.49 s: the compiled artifact is memory-mapped, not copied.

Trial 1's 118 s prefill reflects a cold page cache; trials 2 and 3 execute against a warm
cache and are the representative figures.

### 4.2 WPC v4, expert-contiguous tensor order

| Configuration | Artifact | Throughput |
|---|---|---|
| v3, 6.25 bits, source tensor order | 22.21 GB | 1.00 tok/s |
| **v4, 4.25 bits, expert-contiguous order** | **15.10 GB** | **2.33 tok/s peak, 2.35 tok/s steady state** |

**2.35x faster on an artifact 32% smaller — from roughly 51 to roughly 120 words per minute.**

Two changes are combined in that comparison. Their contributions were also separated:
the ordering change alone, measured at unchanged bit width, accounts for +45% (§5.3); the
remainder is attributable to the narrower encoding and the simpler decode path it permits.

### 4.3 Effective bandwidth utilisation

Derived from §3.4 and §4.2.

| Configuration | Traffic/token | s/token | Effective bandwidth | Share of ~10 GB/s |
|---|---|---|---|---|
| v3, source order | 2.377 GB | 1.11 | 2.14 GB/s | 21% |
| **v4, expert order** | **1.616 GB** | **0.426** | **3.80 GB/s** | **38%** |

The engine has gone from wasting four fifths of the memory subsystem to wasting three fifths.
The remaining deficit is analysed in §9.

### 4.4 Generation quality at 4.25 bits

Twelve tasks were run, spanning code synthesis, general knowledge, arithmetic, multi-step
reasoning, and tool use.

**Tool-call emission — 3 of 3 correct.** The model emitted, in each case, the call and nothing
else, terminating on the end-of-turn token:

- `read_file('README.md')`
- `search_web(query="current price of bitcoin")`
- `list_files(src)`, correctly identified as the *first* step of a two-step plan

Instruction-following and stop-token handling are both intact at 4.25 bits. This is the
prerequisite for agentic use and it survives the compression.

**Code synthesis.** Prompt `def binary_search(arr, target):` produced a textbook-correct
implementation — correct half-open interval handling, correct midpoint computation, correct
branch structure. On the dense control model the evidence is stronger still: at 4.25 bits
Qwen3-4B emitted **token ids identical to the 6.25-bit build** over a 40-token completion
(§6). Where that holds, the compression is not merely acceptable — it is invisible.

**Technical exposition.** The model correctly characterised a hash map and, without prompting,
enumerated the cases where it is the wrong structure (order maintenance, range queries, memory
pressure).

**Weakest observed area — translation into Polish.** In a translation task the model rendered
*tomorrow* as *pojutrze* ("the day after tomorrow") and appended a sentence that was not present
in the source. Both are the kind of error that a reader who does not know the source language
cannot detect. Non-English generation should not be treated as unsupervised-safe at this bit
width; we have not established whether the fault originates in the compression or in the base
model.

No degeneration, repetition loops, or token-space artifacts were observed at 6.25 or 4.25 bits.
They were observed immediately at 2.25 bits (§6.3).

---

## 5. Tensor Ordering: Locality Without Touching a Weight

### 5.1 The defect

The compiler emitted tensors in whatever order the source safetensors shards happened to
enumerate them. In the compiled artifact, layer 0 began:

```
experts.99.down_proj
experts.109.up_proj
experts.77.up_proj
```

Two problems are visible in three lines. Expert indices are in arbitrary order, and — more
damaging — the three projections belonging to any single expert (`gate_proj`, `up_proj`,
`down_proj`) were stored far apart from each other. Evaluating one expert therefore meant three
separate excursions into different regions of a 22.21 GB mapping.

### 5.2 The change

Tensors are sorted by the tuple **(layer, section, expert, projection)**. One expert becomes one
contiguous run of approximately 3.5 MB, and the experts of a layer follow one another in index
order.

**No weight value is modified.** Only the order in which tensors are laid out in the file
changes; the encoder, the block format, and every code are untouched. Reconstruction was
verified **bit-identical** before and after the reordering.

### 5.3 Measured effect

Isolated on WPC v3, so that bit width is not a confounder:

| Tensor order | Throughput |
|---|---|
| Source order | 1.16 tok/s |
| Expert-contiguous order | **1.68 tok/s** |

**+45% from a sort.** No arithmetic changed, no format changed, no accuracy was traded.

### 5.4 A corrected estimate

The gain was predicted at **3–4x** before it was measured. It came in at **1.45x**. The
prediction was wrong, and the reason is instructive enough to record.

The mental model behind 3–4x was of weights scattered byte-by-byte, with the prefetcher unable
to establish any stride. That was not the situation. Each individual projection matrix was
**already stored contiguously** — 1.573 M weights, about 1.17 MiB at 6.25 bits, laid out in one
unbroken run. The access pattern before the fix was therefore not random; it was a sequence of
long sequential runs with jumps between them.

Hardware prefetching already worked *within* each run, and a 1.17 MiB run is long enough to
amortise the cost of the jump that precedes it. Sorting removes jumps — three per expert become
one — but it never had a byte-scattered pattern to repair, because none existed. The available
gain was the jump cost, not the streaming cost, and the jump cost was the smaller of the two.

The correct general form of the lesson: **estimate the size of the contiguous runs before
estimating the value of making them contiguous.** A layout fix is worth what the discontinuities
cost, and discontinuities between megabyte-scale runs cost far less than discontinuities between
bytes.

---

## 6. Bit-Width Scaling: Where Compression Stops Paying

The 30B MoE model is too slow to rebuild in every scheme, so the scaling question was settled on
the dense Qwen3-4B control, where a full rebuild and rerun costs minutes.

### 6.1 Measurements

Identical prompt, 40 tokens, greedy decoding:

| Scheme | Bits/weight | Artifact | 40 tokens | Output |
|---|---|---|---|---|
| v3 | 6.25 | 3.0 GB | 18.03 s | correct code |
| **v4** | **4.25** | **2 038 MiB** | **11.84 s** | correct code, **token ids identical to v3** |
| v5 | 2.25 | 1 079 MiB | 12.40 s | **degenerate** |

### 6.2 The key observation

**v5 is 47% smaller than v4 and 4.7% slower.**

That single inequality is the whole finding. Every step of compression up to this point bought
speed, because the workload was bounded by how fast weights could be moved from DRAM. If halving
the bytes no longer halves the time — if it does not reduce the time at all — then the bytes
were no longer the binding constraint.

On this hardware the transition lies at roughly **2 GB of artifact**. Below it the model is no
longer memory-bandwidth-bound, and the cost shifts to per-weight decode and arithmetic work,
which 2-bit codes do not reduce and may slightly increase.

The consequence is a design conclusion rather than an engineering one:
**4 bits per weight is an optimum, not a compromise.** It is the last width at which compression
still buys speed, and the last width at which it costs no quality. Going further gives up the
model in exchange for disk space that was not scarce.

The same reasoning predicts where the boundary sits for the 30B MoE model: its v4 artifact is
15.10 GB, an order of magnitude above the 2 GB transition, so it remains firmly
bandwidth-bound — consistent with §4.3, where it still runs at 38% of available bandwidth. For
that model further compression *would* still buy speed, if a scheme existed that did not destroy
quality. §7 explains why the obvious candidate does not exist.

### 6.3 What degeneration looks like

Verbatim v5 output, same prompt as the v3/v4 runs above:

~~~
def binary_search(arr, target):md.'``` `.``` `cer`` ``` ```` between`````` economist
~~~

A single token id, 63, accounts for **25 of the 40 emitted tokens**. This is not a model
producing worse code; it is a model whose output distribution has collapsed onto a small number
of high-frequency tokens because four representable levels per block cannot preserve the
geometry of the weight matrices. The failure mode is unambiguous and requires no metric to
detect.

---

## 7. Post-mortem: Why the v1 Pattern Dictionary Failed

WPC v1 was the scheme the project is named after: instead of quantising each weight
independently, it catalogued recurring *patterns* across blocks of weights and stored an index
into that catalogue. It was abandoned during development because its implementation was known to
be defective. That justification was never satisfying — an abandoned idea with known bugs
teaches nothing — so v1 was repaired and measured properly.

### 7.1 Measurements

All figures on Qwen2.5-0.5B, tensor `model.layers.0.self_attn.k_proj.weight`, relative
reconstruction error:

| Configuration | Error |
|---|---|
| v1, all bugs repaired, blocks **held out** of the dictionary's training pool | **56.0%** |
| v1, all bugs repaired, blocks **inside** the training pool | 41.1% |
| v1, one historical bug reintroduced | 58.0% |
| v1, both historical bugs reintroduced | 73.3% |
| **v3 per-block affine, same tensor** | **2.39%** |

The held-out figure is the honest one; the in-pool figure measures how well a dictionary
memorises data it was built from, which is not a compression result.

Repairing every known bug moved v1 from 73.3% to 56.0% error. **It remains worse than v3 by a
factor of 23.** The bugs were real and they were not the problem.

### 7.2 Evidence 1 — the data has no structure to catalogue

A pattern dictionary is profitable exactly when the energy of the data concentrates in a few
directions, so that a modest number of prototypes cover most of the space. The covariance of
16-dimensional weight blocks was decomposed to test this. Its eigenvalues, as shares of total
variance:

```
6.9%  6.9%  6.8%  ...  6.4%
```

Perfectly flat. **Fourteen of sixteen axes are required to capture 90% of the variance.**
There is no low-dimensional subspace to exploit. After the block mean absorbs one dimension, the
remainder behaves as **isotropic noise in fifteen dimensions** — and isotropic noise is, by
definition, the case in which no dictionary of prototypes can help, because there are no
recurring patterns to be recorded.

### 7.3 Evidence 2 — enlarging the dictionary fails at a predictable rate

If the dictionary is merely too small, growing it should fix the problem. It does not, and the
rate at which it fails to is itself diagnostic:

| Dictionary size | Error |
|---|---|
| k = 16 | 85.5% |
| k = 256 | 71.2% |
| k = 4096 | 58.2% |

Each additional bit of index multiplies the error by **0.9535**. Rate–distortion theory gives
the distortion of an optimal quantiser for a *d*-dimensional isotropic source as proportional to
2^(−R/d); for d = 15 that predicts a per-bit factor of 2^(−1/15) = 0.9548. The measured 0.9535
matches the prediction for fifteen-dimensional noise, confirming §7.2 from an entirely
independent direction.

Extrapolating that factor from 58.2% at 12 bits down to v3's 2.39% requires **2^79 dictionary
entries** — a number without physical meaning. The scheme does not need a bigger dictionary; it
needs different data.

### 7.4 Evidence 3 — at equal budget, the simple method wins

The decisive comparison holds the bit budget fixed:

| Budget | v1 codebook | Per-block affine |
|---|---|---|
| 3.00 bits/weight | 56.0% | **39.6%** |
| 3.25 bits/weight | — | **21.6%** |

At the same cost, plain affine quantisation is substantially more accurate, and it improves
sharply with a quarter of a bit more, which the codebook does not.

Part of the gap is pure overhead. **v1 spends half of its budget on the block header** — 24 of
48 bits per block describe the block rather than its contents. v3 and v4 spend 4 bytes per
128 weights, or 0.25 bits/weight, on the same duty.

### 7.5 Conclusion

**Abandoning v1 was correct. The reason recorded at the time was not.**

It was dropped because it was buggy. The bugs are now fixed, and it still loses to a scheme
that is simpler, faster, and older. The failure is a property of the weight distribution —
transformer weight blocks, viewed sixteen dimensions at a time, contain no repeating patterns to
catalogue — and no amount of implementation quality would have changed that.

This is worth stating plainly because the opposite error is common and expensive: an idea gets
abandoned for an incidental reason, then returns years later because nobody wrote down whether
it was the execution or the premise that failed. Here it was the premise.

---

## 8. Tokenizer Identity Across Model Families

An incidental measurement with a disproportionate consequence.

All three models in this study — Qwen3-Coder-30B-A3B, Qwen3-4B, and Qwen2.5-0.5B — carry
**byte-identical tokenizers**:

| Property | Value |
|---|---|
| Vocabulary entries | 151 643 |
| SHA-256 fingerprint (prefix) | `63a10eac44df16bb` |
| Models sharing it | 3 of 3 |

(The 151 936 in §1.2 is the model's padded embedding dimension; the tokenizer itself defines
151 643 entries. The padding does not affect the token space.)

**Why this matters.** Speculative decoding requires a small, fast *drafter* to propose several
tokens which a large *verifier* then accepts or rejects in a single pass. The technique is
usually complicated by token-space mismatch between the two models, requiring a remapping layer
that costs accuracy and engineering effort. Here there is nothing to remap: Qwen2.5-0.5B, whose
v3 artifact is 0.37 GB, emits exactly the token ids that the 30B model consumes.

The drafter is roughly 60x smaller than the verifier and correspondingly faster, so a high draft
acceptance rate would translate directly into throughput. **This is a projection, not a
measurement** — no speculative decoding has been implemented or benchmarked. It is blocked on
exactly one thing, which is the subject of the next section: verification requires evaluating
*k* draft tokens in one forward pass, and the engine cannot yet evaluate two.

---

## 9. Analysis: Why Sparsity Still Does Not Pay Its Full Rate

### 9.1 The inversion, and its partial resolution

Revision 1 reported that the sparse model read 3.66x less data per token than a dense baseline
and nevertheless decoded slower. That inversion is now resolved:

| Configuration | Traffic/token | Measured | Effective bandwidth |
|---|---|---|---|
| Gemma-12B (dense, v3) | 8.70 GB | 1.06 tok/s | 9.22 GB/s |
| Qwen3-Coder-30B-A3B (v3, source order) | 2.377 GB | 0.90–1.00 tok/s | 2.14–2.38 GB/s |
| **Qwen3-Coder-30B-A3B (v4, expert order)** | **1.616 GB** | **2.35 tok/s** | **3.80 GB/s** |

The 30B sparse model now decodes **2.2x faster** than the 12B dense one, which is what sparsity
was supposed to deliver from the start.

**The efficiency deficit is not resolved.** The dense model still extracts 9.22 GB/s from the
memory subsystem where the sparse model extracts 3.80 GB/s. The gap has narrowed from 4.3x to
**2.4x**, but it is the same gap, and it means the sparse path is still leaving more than half of
its theoretical advantage unclaimed. Two of the three causes identified in revision 1 remain
untouched.

### 9.2 Cause 1 — thread under-utilisation (unaddressed)

Observed CPU occupancy on the v3 build was 158–267% of an available 800%: between 2.0 and 3.3 of
8 logical threads doing work. The dense path, streaming contiguous rows, parallelises naturally;
the MoE path serialises around expert dispatch.

Occupancy has **not been re-measured on the v4 build**, and it should be, because the ordering
change may have altered it. Assuming it persists, this remains the largest single recoverable
factor and requires no change to the compression format.

*Estimated headroom: 2.4–4.0x, unverified for v4.*

### 9.3 Cause 2 — prefetch-hostile access pattern (partly addressed)

Each layer selects 8 of 128 experts by runtime argmax. Under v4 the selected expert weights
occupy 8 × 2.51 MB = **20.1 MB scattered across a 15.10 GB mapping**, at offsets not known until
the router has executed.

The tensor-ordering change of §5 addressed the *intra-expert* half of this: an expert is now one
contiguous run rather than three distant ones, and that was worth 45%. The *inter-expert* half
remains — which experts are wanted is still not known until the router runs, and hardware
prefetchers cannot anticipate a data-dependent gather across gigabyte distances.

*Remaining mitigations:* 2 MiB huge pages to reduce TLB pressure across a 15 GB mapping, and
speculative prefetch issued immediately after the router argmax, overlapping fetch latency with
the attention block of the same layer.

### 9.4 Cause 3 — no weight reuse in the cache hierarchy (unaddressed)

The working set of one layer's activated experts is 20.1 MB under v4 — still **2.5x the 8 MiB
L3** — and the next token selects a different expert subset. Every weight byte is therefore
fetched from DRAM, consumed by a single fused multiply-add, and evicted.

Arithmetic intensity is **2 FLOPs per byte**, firmly in the memory-bound regime of the roofline
model, with no reuse available to amortise the fetch.

The hierarchy is used correctly where it can be: the 2048-element activation vector (8 KiB fp32)
resides in L1d (32 KiB) and is reused across all eight expert evaluations within a layer. The
problem is not cache management; it is that MoE weights are fundamentally single-use at batch
size 1.

**The structural remedy is batching.** At batch size *B*, one expert fetch serves *B* tokens,
raising arithmetic intensity to 2*B* FLOPs/byte. Better still for this architecture is
*expert-grouped* batching: gather the tokens of a batch that route to the same expert and
evaluate them together, so the 2.51 MB fetch is amortised over all of them. This is the only
technique that attacks the root cause rather than its symptoms, and §10 explains why none of it
is currently possible.

---

## 10. The Principal Limitation: One Token per Forward Pass

The engine's inference interface is `forward(token)`: one token in, one distribution out. There
is no batched path anywhere in the runtime.

**Direct evidence.** Reading a prompt costs the same as writing a reply. From an instrumented
run:

```
prefill (28 tokens) in 65.2s
```

Prompt tokens are being pushed through the network one at a time, at the same per-token cost as
generated tokens — roughly a second or more each — even though every prompt token is known in
advance and could be processed together with its neighbours. The one situation in which batching
is trivially available is being forfeited.

**What this blocks.** Three separate improvements identified in this report reduce to the same
missing capability:

| Wanted | Requires |
|---|---|
| Fast prefill — long prompts becoming nearly free (§4.1, §11) | Batched forward over prompt tokens |
| Speculative decoding with a 0.37 GB drafter (§8) | Verifying *k* draft tokens in one pass |
| Expert-grouped execution — one weight fetch serving many tokens (§9.4) | Batch of tokens available at once |

Each of these is independently worth a large multiple. None of them can be attempted while the
forward pass accepts exactly one token.

**This is the principal outstanding item of work.** It is not a tuning parameter or a format
question; it is a shape change to the runtime's inner interface, and everything else identified
in §9 is downstream of it.

---

## 11. Viability Assessment for AIONS Integration

**Current state: usable for supervised generation, not yet for interactive dispatch.**

At 2.35 tok/s the model produces roughly **120 words per minute** — approximately the pace of
ordinary speech, and within a factor of two of comfortable silent reading. Revision 1's figure of
0.90 tok/s (~40 wpm) was below the threshold at which a person will wait for output; 120 wpm is
not. Correctness, instruction-following, and tool-call formation are all adequate.

**The blocker has moved from decode to prefill.** At the prefill rate logged in §10, a 500-token
prompt costs on the order of nineteen minutes before the first output token appears. For agentic
use — where prompts carry tool definitions, file contents and history — this now dominates the
interaction entirely, and it is exactly the case that batching makes nearly free.

**Projected state after the identified optimisations** (projections, not measurements):

| Optimisation | Factor | Cumulative |
|---|---|---|
| Baseline (v4, expert-contiguous order) | — | 2.35 tok/s |
| Batched prefill (§10) | prompt-side; removes the dominant latency | 2.35 tok/s decode |
| Full thread utilisation (§9.2, if the v3 deficit persists) | 2.4–4.0x | 5.6–9.4 tok/s |
| Huge pages + speculative expert prefetch (§9.3) | 1.3–1.8x | 7.3–17 tok/s |
| Speculative decoding with Qwen2.5-0.5B drafter (§8) | acceptance-rate dependent | — |

A sustained 5–10 tok/s (250–500 words per minute) is the realistic target and would place the
model well inside interactive range on 2016 hardware. **None of these optimisations require
modifying the compression format, retraining, or additional hardware.**

**Memory footprint is not a constraint.** The v4 artifact is 15.10 GB against 39 GB available and
is memory-mapped rather than copied, leaving ample headroom for extended context. Peak RSS on
the v3 build was 14.93 GB; it has not been re-measured for v4.

---

## 12. Conclusions

1. **A 30B-parameter model runs on a 2016 quad-core mobile CPU with no GPU, at conversational
   pace.** WPC v4 compresses it by **3.77x** (57.0 GB → 15.10 GB) at 4.25 bits/weight and
   decodes at 2.35 tok/s, or roughly 120 words per minute. This is the qualitative result.

2. **Four bits per weight is an optimum rather than a compromise.** At 4.25 bits the dense
   control model produces token ids *identical* to the 6.25-bit build, and the 30B model emits
   correct code and correct tool calls. At 2.25 bits the artifact is smaller, no faster, and
   degenerate. The point at which compression stops buying speed is locatable and, on this
   hardware, sits near 2 GB of artifact.

3. **Layout is worth as much as arithmetic, and it is free.** Sorting tensors so that each
   expert occupies one contiguous run raised throughput 45% without altering a single weight
   value; reconstruction is bit-identical. The estimate that motivated the work was 3–4x and
   the truth was 1.45x, because the individual matrices were already contiguous — a correction
   worth carrying forward: measure the size of your contiguous runs before valuing the removal
   of discontinuities between them.

4. **The pattern-dictionary premise is dead, and now demonstrably so.** With every known bug
   repaired, v1 reconstructs at 56.0% error against v3's 2.39%. Flat covariance spectra,
   a dictionary-scaling law matching 2^(−R/15) exactly, and a direct loss to plain affine
   quantisation at equal budget all say the same thing from different directions: transformer
   weight blocks contain no repeating patterns to catalogue. The idea failed on its premise,
   not on its execution.

5. **MoE sparsity now pays, but only partially.** The 30B sparse model decodes 2.2x faster than
   the 12B dense baseline, resolving revision 1's inversion. Yet it still extracts 3.80 GB/s
   from a subsystem that yields 9.22 GB/s to the dense model — a 2.4x efficiency deficit, down
   from 4.3x but unfinished.

6. **The performance deficit remains implementation-bound, not format-bound.** Thread
   utilisation of 2–3 cores out of 8, an unbatched prefill, and a data-dependent expert gather
   account for the remaining gap. All three are addressable within the existing architecture,
   and none require touching the compression scheme.

7. **One limitation dominates all others: the engine performs one forward pass per token.**
   Batched prefill, speculative decoding, and expert-grouped execution are three names for the
   same missing capability. Until the forward pass accepts more than one token, each of them is
   unavailable, and the improvements they would deliver are the largest remaining on the table.

8. **The tokenizer identity across all three model families** (151 643 entries, SHA-256 prefix
   `63a10eac44df16bb`) removes the usual obstacle to speculative decoding. A 0.37 GB drafter and
   a 15.10 GB verifier share a token space exactly. The infrastructure is in place; only the
   batched forward pass is missing.

---

## 13. Verification and Reproduction

### 13.1 Test coverage

**93 unit tests pass**, up from 46 at revision 1. The additions cover the v4 and v5 block
formats, the tensor-ordering pass, and bit-exactness of reconstruction across the reordering —
the last of these being what licenses the claim in §5.2 that no weight value changed.

### 13.2 Commands

```
wpc-compiler --input <bf16 model dir> --output <artifact dir> --scheme v4

wpc-runtime --model <norms + tokenizer dir> \
            --wpc <artifact dir> \
            --scheme v4 \
            --arch qwen3-moe \
            --prompt "def binary_search(arr, target):" \
            --max-tokens 60
```

Architecture is auto-detected from `model_type: "qwen3_moe"`; `--arch` is optional. Substitute
`--scheme v3` to reproduce the revision 1 figures.

---

*All figures in this report are measured on the system described in §1.1 unless explicitly
labelled as derived or projected. Derived quantities (§3, §4.3) follow analytically from the
published model configuration and the compression rate, and are consistent with the measured
artifact sizes to within 0.1%. Projections (§8, §11) are labelled as such and are not
measurements.*
