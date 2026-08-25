# Noworodek Weight Observatory & Weight Lab

## Purpose

Noworodek must make parameter evolution observable and experimentally editable without pretending that one tensor element corresponds to one human-readable fact. Learning is represented as a relationship between an experience, a training update, distributed parameter deltas, and measured behaviour.

## Data flow

```text
experience
    -> training step
    -> before/after parameter snapshot
    -> tensor deltas
    -> statistical summaries
    -> evaluator result
    -> accepted/rejected experimental WeightSet
```

The `TrainingObservatory` records the experience identifier, training step, WeightSet identity, optional loss, and per-tensor delta summaries.

A delta summary contains:

- tensor name;
- number of changed elements;
- L1 magnitude;
- L2 magnitude;
- maximum absolute change.

This is deliberately an attribution record, not a claim that a tensor or parameter has a single semantic meaning.

## Weight Lab API

The initial editor layer provides controlled parameter surgery:

- snapshot a tensor;
- compute before/after diffs;
- scale a tensor;
- add an explicit delta;
- replace tensor contents with shape-checked data.

All edits operate through `WeightBackend`, so future mmap and WPC implementations can reuse the same editor contract.

## Experimental lifecycle

```text
current WeightSet
       |
    snapshot
       |
experimental clone
       |
weight edit / training update
       |
evaluator
   /       \
 keep     rollback
```

No editor operation is treated as knowledge extraction by itself. A change becomes useful evidence only when an independent evaluation demonstrates a behavioural effect.

## Future GUI

The eventual GUI should present WeightSets, tensor metadata, training-experience timeline, tensor diffs, and evaluator outcomes. It should not expose millions of raw values as the primary UX. Raw tensors remain available for expert inspection, while the default view focuses on deltas, patterns, provenance, and measurable behavioural changes.

## Future WPC integration

The editor and observatory must remain backend-agnostic. WPC is expected to implement the same weight backend contract later, enabling the same observation/diff/edit/evaluate workflow over WPC-managed parameter storage.
