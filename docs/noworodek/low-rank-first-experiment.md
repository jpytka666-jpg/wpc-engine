# Noworodek — Low-Rank First Representation Experiment

## Decision

Low-rank is the first new representation to evaluate after the already-proven fused WPC representation. Do not add sinusoidal, polynomial, recursive, or general procedural generators until low-rank has been measured on real WPC tensors.

## Proven baseline

The current WPC fused decode-and-multiply path has already demonstrated the key execution property: the compressed weight is consumed inside the matrix/vector multiplication and the fully expanded tensor is never materialized.

Recorded experiment:

- tensor: `model.layers.12.self_attn.o_proj.weight`
- source shape: `2560 x 4096`
- compressed representation: `5,570,560 B` (5.3 MiB)
- expanded representation: `40.0 MiB`
- expanded tensor: **never materialized by the kernel**
- fused kernel: `1.043 ms`, `10,049.7 M weights/s`, `20.09 GFLOP/s`
- observed CPU comparison: `16–18x` faster
- error: GPU `2.235e-04` vs CPU `1.592e-03`
- recorded result: `PASS`
- run log: `gpu/wpc4-decode/runs/gemv_2026-08-25_1522.log`

This is an engineering result supplied by the project record. The numbers must remain tagged as measured evidence from that run; future work must not generalize them to all tensors without a new benchmark.

## What changes with Noworodek

Current WPC baseline:

```text
one predetermined representation
+ one fused execution kernel
```

Target architecture:

```text
per-tensor representation choice
+ observable/editable representation parameters
+ representation-specific execution backend
```

This is a qualitative extension, not merely another compression ratio tweak.

## Why low-rank comes first

For a matrix `W` represented as `A * B`, with rank `r`:

```text
storage:  (n + m) * r
instead of n * m
```

For `2560 x 4096` and `r = 64`, parameter storage is approximately `12x` smaller than dense storage (before metadata and any quantization details).

The execution form is also standard:

```text
y = A * (B * x)
```

No new mathematical generator is required. Existing GEMM/GEMV primitives can be reused or fused later. Gradients are straightforward:

```text
dW = dA * B + A * dB
```

The important claim is **not** that low-rank is automatically better. It is that storage, training, and execution are all measurable without inventing a new kernel family first.

## Required experiment

Run the same real tensor through a rank sweep, at minimum:

```text
r = 8, 16, 32, 64, 128, 256
```

For each rank record:

1. reconstruction error against the original tensor;
2. representation bytes;
3. compression ratio;
4. materialized reconstruction time, if any;
5. execution time using the low-rank form without full materialization;
6. CPU/GPU comparison;
7. numerical stability during training;
8. downstream task quality where the tensor is part of a runnable model.

The first target tensor is the already-measured `model.layers.12.self_attn.o_proj.weight` so that the baseline is directly comparable.

## Acceptance rule

A rank is considered viable only when all of the following are measured:

- acceptable reconstruction error for the target workload;
- smaller representation than the proven WPC baseline or a clearly justified latency/memory advantage;
- execution does not require materializing the full dense tensor in the hot path;
- training remains numerically stable;
- evaluation quality remains within the experiment's declared tolerance.

No rank is selected by intuition alone.

## Representation lifecycle

Every representation remains continuously editable:

```text
WeightSet
   -> representation manifest
   -> representation parameters
   -> live execution
   -> Observatory
   -> edit
   -> snapshot/version
   -> rollback or continue training
```

The representation itself is first-class state. Observatory must record `parameter delta` for low-rank factors (`dA`, `dB`) separately from any optional dense reconstruction diff.

## Engineering constraint

Each fundamentally new representation requires a compatible execution path. Until that path is benchmarked, the representation is experimental and must not become the default selector output.

This rule prevents representation proliferation from outrunning kernel engineering.
