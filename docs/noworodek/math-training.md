# Noworodek Mathematics Training

## Goal

Mathematics is the first explicit domain curriculum for Noworodek. The objective is not memorization of final answers. Training examples must preserve a verifiable relationship between problem, mathematical operation, intermediate expression, final result, and evaluator evidence.

## Curriculum order

1. Arithmetic
2. Algebra
3. Linear algebra
4. Calculus
5. Discrete mathematics
6. Geometry

Difficulty increases only after held-out evaluation shows stable performance on the current domain.

## Example representation

```text
problem
  -> structured mathematical steps
  -> intermediate expressions
  -> final answer
  -> deterministic evaluator
  -> score/evidence
```

`MathTrace` stores explicit, externally defined solution steps. It is training data, not private model reasoning. The trace is valid only when its step indices, expressions, and final answer satisfy the schema.

## Evaluation rules

- Exact answers use exact-match evaluation where the answer representation is canonical.
- Numerical problems must eventually support tolerance-aware numeric evaluation instead of string matching.
- Symbolic problems should canonicalize equivalent expressions before scoring.
- A high score requires independent evidence; a matching string alone is insufficient for difficult problems.
- Training and held-out evaluation datasets must be separated.

## Training signals

The normalized mathematical experience should expose:

```text
problem
operation sequence
intermediate state
result
verification evidence
score
```

This allows the Observatory to correlate training experiences with changes in the external WeightSets.

## Representation experiments

The proven WPC fused decode-and-multiply path remains the primary compressed execution baseline. Low-Rank is the first new representation experiment.

For each mathematical benchmark, compare:

- dense/reference representation;
- existing WPC fused representation;
- Low-Rank at multiple ranks.

Record storage, reconstruction error, execution latency, GPU memory traffic where available, and task quality. Do not claim a representation is better merely because it uses fewer bytes.

## Next implementation stages

1. Expand the deterministic starter curriculum.
2. Add canonicalization and tolerance-aware numeric evaluators.
3. Add generated problem families with independent seeds.
4. Add held-out mathematical evaluation.
5. Connect `MathTrace` to normalized training experiences.
6. Train a small Noworodek checkpoint on GPU.
7. Compare dense/WPC/Low-Rank WeightSet representations for the same trained tensors.
8. Add specialist `MathSet` hot-swapping and rollback.
