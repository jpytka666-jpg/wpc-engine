# Noworodek — Compact, Fast, Powerful, Continuously Editable

## Primary objective

Noworodek is designed to remain four things simultaneously:

1. **Small** — minimize persistent parameter storage and runtime memory.
2. **Fast** — optimize the executed representation, memory movement and kernel path, not merely the file size.
3. **Powerful** — preserve enough representational capacity for coding, tool use, language and operator tasks.
4. **Continuously editable** — trainable parameters and their representations remain addressable and replaceable throughout the model lifecycle.

## Weight representation is hybrid

A WeightSet is not required to be a dense array of floats. A tensor may use one of several equivalent representations:

```text
Dense       -> literal tensor
Quantized   -> compact numerical tensor
Pattern     -> pattern/codebook representation
LowRank     -> factorized tensor
Procedural  -> compact parameterized generator
Hybrid      -> representation selected per tensor/block
```

The runtime must be able to switch representation versions without changing the stable tensor identity.

## Procedural weights

A procedural representation stores a compact computation and its editable parameters rather than materializing every value. Examples include affine transforms, low-rank products and future restricted compute graphs.

The important distinction is:

```text
storage compression != execution speed
```

A formula that saves storage can be slower to execute than a dense tensor. Therefore every procedural representation must be benchmarked for:

- storage bytes
- materialization cost
- direct execution cost
- memory traffic
- numerical error
- model quality

No representation is accepted as a performance improvement without measurements.

## Continuous editability contract

Every trainable tensor keeps:

- stable tensor ID
- WeightSet ID/version
- representation type
- shape/dtype
- provenance
- edit history
- observability hooks

Edits may target literal values, representation parameters, or a representation swap. The model must observe the resulting parameter delta and associate it with an experience/training event when applicable.

## Training contract

The canonical early training representation is FP32 master parameters with FP32 gradients/optimizer state. Compact representations are introduced only when the model has passed a numerical/quality baseline.

Later, the representation optimizer may choose per tensor:

```text
Dense FP32 -> Dense low precision -> LowRank -> Pattern -> Procedural
```

subject to quality and latency constraints.

## Performance principle

The desired optimization is not to make the model "think" by replacing binary arithmetic with formulas. The processor still executes numerical operations. The target is instead to reduce unnecessary memory traffic and compute by representing learned structure compactly and executing that structure directly when beneficial.

## Evaluation gate

A representation change is considered successful only when an evaluator can compare the old and new representation on the same tasks and report:

```text
quality
latency
memory
storage
training stability
```

The system must never trade correctness for an unverified compression or speed claim.
