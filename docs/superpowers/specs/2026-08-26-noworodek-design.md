# Noworodek — Modular AIONS Operator Model Design

**Date:** 2026-08-26  
**Branch:** `Noworodek`  
**Status:** Approved design

## 1. Purpose

Noworodek is a from-scratch, small decoder-only language model designed first as an AIONS operator rather than as a general-purpose foundation model.

The defining architectural requirement is modular parameter ownership: the model must support independently versioned, attachable and detachable `WeightSet`s from the first implementation. Examples include general language, coding, AIONS operation, and tool-use specialisations.

WPC integration is deliberately deferred until the modular weight contract is proven. WPC will become an interchangeable weight backend rather than a special case embedded throughout the model.

## 2. Architectural principles

1. **Model core and parameter storage are separate.** The model consumes a `WeightBackend`/`WeightSetManager` interface rather than assuming one monolithic checkpoint.
2. **WeightSets are first-class.** A WeightSet has a manifest, version, tensor inventory, compatibility information, provenance, checksums, and capability metadata.
3. **Hot swapping is explicit.** A WeightSet can be mounted, unmounted, replaced, snapshotted, and restored subject to compatibility and lifecycle checks.
4. **Core parameters and specialist parameters are distinct.** Shared core weights remain resident where appropriate; specialist WeightSets provide targeted parameter specialisation without requiring multiple complete models by default.
5. **Observability is built in.** Training can inspect weights, gradients, loss, and optimizer state without changing the model implementation.
6. **AIONS is an environment, not merely a dataset.** Training data will represent tasks, context, available tools, actions, observations, outcomes, and evaluation signals.
7. **Safety and reproducibility precede autonomous execution.** Initial tools run in constrained/read-only evaluation paths; destructive capabilities require explicit later integration.
8. **Evidence over claims.** Every milestone has executable tests and recorded measurements/artifacts.

## 3. Target architecture

```text
                    NOWORODEK
                        │
                 ┌──────▼──────┐
                 │  MODEL CORE │
                 └──────┬──────┘
                        │
                WeightSetManager
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
       GENERAL       CODING        TOOLUSE
       WeightSet     WeightSet     WeightSet
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                  WeightBackend
                 ┌──────┼──────┐
                 ▼      ▼      ▼
               memory  mmap   WPC*

* later milestone; not part of the initial implementation
```

## 4. WeightSet contract

A WeightSet must be independently addressable and must expose:

- stable identifier and semantic name;
- version;
- architecture/model compatibility identifier;
- tensor manifest and shapes/dtypes;
- checksum/integrity information;
- training provenance;
- declared capability/specialisation metadata;
- lifecycle state sufficient to determine mounted/unmounted status.

The manager must reject incompatible sets rather than silently applying tensors to the wrong architecture.

The initial backend implementations are in-memory and file/mmap oriented. The interface must not leak their storage details into the Transformer layers.

## 5. Model and training

The initial model is intentionally small and trainable from scratch. The first objective is architectural validation, not competitive language-model quality.

Required training capabilities:

- tokenisation and causal language modelling;
- forward and backward passes;
- optimizer and checkpoint/reload;
- deterministic smoke training where practical;
- parameter/gradient instrumentation;
- WeightSet attach/detach tests during controlled execution.

The first specialist training targets are:

- general language behaviour;
- Rust/code behaviour;
- AIONS/tool operation;
- tool-use variants as experiments rather than assumptions.

## 6. AIONS operator training

Training examples use an interaction schema conceptually equivalent to:

```text
TASK
CONTEXT
AVAILABLE_TOOLS
ACTION
TOOL_RESULT
NEXT_ACTION
FINAL_RESULT
EVALUATION
```

The evaluator measures whether the operator selected useful actions, respected tool contracts, reached the requested outcome, and passed available tests/evidence.

The intended progression is supervised fine-tuning first, followed only after a reliable evaluator exists by tool-use/reinforcement training.

## 7. WPC integration boundary

WPC remains an implementation of a `WeightBackend`, not a requirement spread through the model core.

The eventual flow is:

```text
WeightSet
   -> WeightBackend
   -> WPC artifact
   -> materialised tensors as required
```

The WPC milestone measures correctness, load/unload cost, memory use, materialisation cost, and inference/training impact before any claim of benefit is made.

## 8. Error handling

The system must fail closed for:

- incompatible architecture/version;
- missing tensors;
- shape/dtype mismatch;
- checksum/integrity failure;
- illegal lifecycle transitions;
- unsupported backend operations.

Tool execution must surface explicit errors to the training/evaluation loop rather than converting failures into successful examples.

## 9. Testing strategy

Each milestone must have focused unit/integration tests plus a runnable checkpoint.

Minimum evidence includes:

1. WeightSet manifest validation;
2. attach/detach and replacement;
3. incompatible-set rejection;
4. checkpoint/reload;
5. forward/backward smoke training;
6. observable parameter/gradient statistics;
7. AIONS tool-loop evaluation;
8. later WPC backend equivalence tests.

## 10. Milestone sequence

1. Recon and specification.
2. WeightSet contract.
3. WeightSetManager and lifecycle/hot-swap mechanism.
4. Minimal decoder-only Transformer.
5. Training loop and checkpoints.
6. Weight/gradient observability.
7. AIONS-oriented dataset format.
8. Constrained AIONS tool environment.
9. Automated evaluator.
10. SFT operator training.
11. Tool-use/reinforcement experiments.
12. WPC backend.
13. End-to-end benchmarks and evidence.

No milestone is considered complete without passing its defined verification checkpoint.

## 11. Explicit non-goals for the first implementation

- No attempt to reproduce Qwen/Gemma scale.
- No immediate replacement of the existing WPC runtime.
- No assumption that every specialist requires a full duplicate model.
- No autonomous destructive modification of the host AIONS environment during early training.
- No performance claims before measurements exist.
