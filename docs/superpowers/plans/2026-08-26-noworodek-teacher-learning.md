# Noworodek — Teacher Observation Learning Implementation Plan

**Date:** 2026-08-26
**Branch:** `Noworodek`
**Status:** Approved implementation plan
**Architecture source:** `WHITEPAPER_ADDENDUM_AIONS_AGENT.md` §6 and `docs/architecture/noworodek-teacher-observatory.md`

## Goal

Turn the teacher-observation concept into a working Rust pipeline in which Noworodek can learn from the observable work of an external teacher agent (initially Claude Code), while all Noworodek trainable tensors remain externalized, addressable, editable, and observable.

The pipeline must preserve raw provenance, normalize observations into evaluator-backed experiences, and keep teacher access read-only.

## Non-goals

- No capture or training on private chain-of-thought.
- No direct teacher-parameter mutation.
- No assumption that teacher and student architectures match.
- No automatic teacher-to-student tensor transplantation.
- No claim of competence without evaluator evidence.
- No WPC backend integration until the student WeightSet and observation pipeline are proven.

## Global architecture

```text
Human + Teacher Agent
        |
        v
ObservationBus
        |
        +--> RawTraceStore
        |
        +--> TeacherObserver
        |
        v
ExperienceNormalizer
        |
        v
Evaluator
        |
        v
TrainingSample
        |
        +--> behaviour target
        +--> representation target (optional)
        +--> parameter-pattern experiment (optional)
        |
        v
Noworodek Trainer
        |
        v
External WeightSets
        |
        v
TrainingObservatory + WeightEditor
```

## Task 1 — Harden the teacher protocol

**Files:**
- Modify: `noworodek/src/teacher.rs`
- Test: `noworodek/src/teacher.rs`

### RED
Add tests proving:

- stable IDs are non-empty;
- teacher manifests preserve architecture and tensor metadata;
- snapshots contain provenance;
- observations are immutable/read-only data;
- an observation can be linked to an experience and step range.

Run:

```bash
cargo test -p noworodek teacher
```

Expected: FAIL for any missing contract behavior.

### GREEN
Implement validation and explicit observation metadata. Do not add provider-specific Claude code yet.

### Checkpoint

`TeacherObserver` types are validated and provider-neutral.

## Task 2 — Build the Raw Trace event schema

**Files:**
- Create: `noworodek/src/trace.rs`
- Modify: `noworodek/src/lib.rs`
- Test: `noworodek/src/trace.rs`

Define events for:

```text
SessionStarted
UserIntent
ContextSnapshot
ToolCall
ToolResult
FileRead
FileWrite
PatchApplied
CommandExecuted
TestResult
CiResult
GitState
TeacherMessage
SessionFinished
```

Every event carries:

- session ID;
- monotonic sequence number;
- timestamp;
- source/provider ID;
- payload;
- optional parent event ID;
- optional experience ID.

Raw trace events must be append-only in the logical model.

### Checkpoint

A synthetic Claude-like session can be represented losslessly as raw events.

## Task 3 — RawTraceStore

**Files:**
- Create: `noworodek/src/trace_store.rs`
- Test: `noworodek/src/trace_store.rs`

Implement a deterministic, file-friendly event store abstraction. Start with in-memory and JSONL-compatible storage; do not introduce a database dependency yet.

Required behaviors:

- append event;
- retrieve session;
- retrieve experience slice;
- preserve ordering;
- detect malformed sequence numbers;
- expose provenance without mutation.

### Checkpoint

A complete session can be recorded and replayed identically from the raw trace.

## Task 4 — ObservationBus

**Files:**
- Create: `noworodek/src/observation_bus.rs`
- Test: `noworodek/src/observation_bus.rs`

Provide a single typed ingress for teacher observations and environment events.

Conceptually:

```rust
pub trait ObservationSink {
    fn accept(&mut self, event: ObservationEvent) -> Result<(), ObservationError>;
}
```

The bus must support multiple sinks without giving sinks mutable access to the teacher.

Initial sinks:

- raw trace sink;
- training observatory sink;
- diagnostics sink.

### Checkpoint

One event stream feeds raw provenance and training observability without duplicated caller logic.

## Task 5 — ExperienceNormalizer

**Files:**
- Create: `noworodek/src/experience.rs`
- Test: `noworodek/src/experience.rs`

Transform raw traces into normalized experiences:

```text
TASK
CONTEXT
OBSERVATION
ACTION
TOOL_RESULT
CORRECTION
OUTCOME
EVIDENCE
SCORE
```

Normalization rules must be deterministic and preserve references to the original event IDs.

Important: an action is not labelled correct until evaluator evidence exists.

### Checkpoint

Given the same trace, normalization produces the same experience record and provenance links.

## Task 6 — Evaluator interface

**Files:**
- Create: `noworodek/src/evaluator.rs`
- Test: `noworodek/src/evaluator.rs`

Define:

```rust
pub trait Evaluator {
    fn evaluate(&self, experience: &NormalizedExperience) -> EvaluationResult;
}
```

Support at minimum:

- success;
- partial success;
- failure;
- evidence references;
- numerical score.

Initial evaluators should use observable evidence: tests, exit codes, CI status, expected file changes, and task completion predicates.

### Checkpoint

Synthetic teacher trajectories receive stable scores and evidence.

## Task 7 — Claude Code adapter boundary

**Files:**
- Create: `noworodek/src/teachers/claude_code.rs`
- Create: `noworodek/src/teachers/mod.rs`
- Tests: `noworodek/src/teachers/claude_code.rs`

Do not depend on undocumented/private Claude internals.

Implement the adapter around an **observable event interface**. The initial adapter may consume exported/session-visible events supplied by an external integration layer.

Required mappings:

- user message → `UserIntent`;
- visible assistant/tool action → `ToolCall`/`TeacherMessage`;
- tool result → `ToolResult`;
- repository changes → `PatchApplied`/`GitState`;
- command/test results → `CommandExecuted`/`TestResult`;
- CI checks → `CiResult`.

### Checkpoint

A real or fixture Claude Code session can be converted to RawTrace events without provider-specific logic leaking into the core model.

## Task 8 — Externalized Transformer parameter contract

**Files:**
- Create/modify: `noworodek/src/model/`
- Test: model unit/integration tests

Before implementing the full Transformer, require every trainable tensor to be registered through the WeightSet subsystem and have a stable ID.

Examples:

```text
model.embeddings.token.weight
model.layers.03.attention.q_proj.weight
model.layers.03.attention.k_proj.weight
model.layers.03.attention.v_proj.weight
model.layers.03.attention.o_proj.weight
model.layers.03.mlp.up_proj.weight
model.layers.03.mlp.down_proj.weight
model.layers.03.norm.weight
model.lm_head.weight
```

The model core may reference handles/views, but it must not privately own an unregistered trainable parameter tensor.

### Checkpoint

The full parameter inventory of the tiny Transformer can be enumerated from WeightSets alone.

## Task 9 — Weight Observatory integration with real Transformer training

**Files:**
- Modify: `noworodek/src/observatory.rs`
- Modify: `noworodek/src/training.rs`
- Test: integration tests

For each training step capture, where enabled:

```text
experience ID
step
WeightSet version
W_before reference/snapshot metadata
W_after reference/snapshot metadata
delta statistics
loss
gradient statistics
evaluator score (when available)
```

Use selective tensor capture to avoid copying the whole model at every step.

### Checkpoint

A tiny model trained on synthetic data produces real per-tensor deltas linked to training experiences.

## Task 10 — Teacher behaviour learning

**Files:**
- Create: `noworodek/src/learning.rs`
- Test: end-to-end fixtures

Implement a first supervised path where Noworodek learns to predict the next useful observable action from normalized teacher experiences.

Do not begin with RL. First establish a reproducible SFT-like training target.

### Checkpoint

On held-out synthetic tasks, the student reproduces a measurable subset of the teacher's action policy better than an untrained baseline.

## Task 11 — Representation observation hooks

Add optional read-only hooks for teacher activations/logits when the teacher explicitly exposes them.

Keep these separate from behavioural targets so the evaluator can determine whether any improvement comes from representation distillation.

### Checkpoint

Representation observations can be added to an experience without changing its behavioural trace.

## Task 12 — Parameter-pattern research lane

Create a separate experimental namespace for:

```text
Teacher tensor snapshot
        ↓
Teacher delta/pattern analysis
        ↓
Mapping proposal
        ↓
Student experimental WeightSet
        ↓
Evaluator
        ↓
KEEP / ROLLBACK
```

No automatic transfer. Every mapped WeightSet is versioned and reversible.

### Checkpoint

At least one controlled teacher/student parameter-pattern experiment can be reproduced and evaluated.

## Task 13 — Specialist WeightSet training

Split normalized experiences into specialist streams:

- GeneralSet
- CodingSet
- RustSet
- DebuggingSet
- ToolUseSet
- AIONSOperatorSet
- VerificationSet
- PlanningSet

Train and benchmark specialist sets independently and verify hot-swap behaviour.

### Checkpoint

At least two specialist WeightSets improve their target task family without destroying the shared-core smoke benchmark.

## Task 14 — Full Claude-with-human observation experiment

Only after Tasks 1–13 pass:

1. run a real Claude Code session with the human;
2. capture observable events;
3. store raw trace;
4. normalize experiences;
5. evaluate teacher outcomes;
6. train a small Noworodek model;
7. evaluate on unseen tasks;
8. compare against an untrained baseline;
9. inspect correlated WeightSet deltas.

### Success criteria

We require measurable evidence for:

- task success rate;
- tool-selection accuracy;
- test/CI success;
- generalization to unseen tasks;
- training loss improvement;
- WeightSet delta statistics;
- reproducibility from the raw trace.

## Task 15 — WPC backend integration

Only after the external WeightSet + Transformer + teacher-learning system is proven.

Replace or supplement memory/file backends with WPC while preserving the same model-facing contract.

Measure:

- memory footprint;
- load/unload latency;
- tensor materialisation cost;
- training overhead;
- inference overhead;
- correctness against the non-WPC backend.

## Checkpoint policy

Every task follows:

```text
RED
  ↓
failing focused test
  ↓
GREEN
  ↓
minimal implementation
  ↓
full relevant test suite
  ↓
CI
  ↓
checkpoint evidence
```

No task is marked complete on local intuition alone.

## Final architecture target

```text
Human
  ↓
Claude Code / Teacher
  ↓
ObservationBus
  ├── RawTraceStore
  ├── TeacherObserver
  └── Diagnostics
  ↓
ExperienceNormalizer
  ↓
Evaluator
  ↓
Training targets
  ↓
Noworodek Transformer
  ↓
WeightSetManager
  ├── GeneralSet
  ├── CodingSet
  ├── ToolUseSet
  ├── AIONSOperatorSet
  └── experimental sets
  ↓
TrainingObservatory
  ↓
WeightEditor / Weight Surgery
  ↓
Future WPC backend
```

## Explicit evidence rule

The project may say **"observed"** when a raw trace was captured, **"trained"** when an optimizer changed student parameters, and **"learned"** only when an independent evaluator demonstrates improved behaviour on defined tasks, preferably including held-out tasks. This distinction is mandatory in documentation and benchmarks.
