# Noworodek — Teacher Observatory

## Purpose

Noworodek must be able to learn from another model through an explicit observation interface. Observation is not limited to text outputs: when the teacher exposes compatible instrumentation, Noworodek may inspect parameter metadata, weight snapshots/deltas, activations, logits, and behavioural/tool trajectories.

The protocol deliberately separates **observation** from **parameter copying**. A teacher observation is evidence that can be transformed into student training signals; it is not assumed that a teacher tensor maps directly to a student tensor.

## Observation layers

### 1. Structure

- architecture identifier
- layer/module names
- tensor names, shapes, dtypes
- attention/MLP topology

### 2. Parameters

- immutable tensor snapshots where permitted
- tensor statistics
- parameter deltas between teacher checkpoints
- provenance and training-step metadata

### 3. Runtime representations

- selected activations
- hidden states
- attention outputs
- logits

The protocol must support selective capture so that observation does not require materialising the entire teacher model state.

### 4. Behaviour

- task/input
- available tools
- selected action
- tool invocation
- tool result
- final output
- evaluator result

### 5. Training events

When the teacher training loop is instrumented, an observation may associate a training experience with `W_before`, `W_after`, and the resulting delta. This establishes an evidence trail rather than claiming that an individual parameter represents an individual concept.

## Core protocol concepts

```text
TeacherIdentity
TeacherArchitecture
TeacherTensorManifest
TeacherSnapshot
TeacherDelta
ActivationObservation
BehaviourTrace
TrainingExperience
ObservationRecord
```

Every observation must contain enough provenance to reproduce what was observed: teacher identity/version, architecture identifier, observation type, source checkpoint/step, tensor or module identifiers where applicable, and a stable experience identifier when associated with training.

## Student learning boundary

```text
Teacher
  |
  v
ObservationRecord
  |
  +--> behaviour/distillation signal
  +--> representation signal
  +--> parameter-pattern signal
  |
  v
Student training
  |
  v
Student WeightSet
```

The student may learn from observations using behavioural cloning, logit/representation distillation, supervised targets, or future parameter-pattern transfer experiments.

## Cross-architecture rule

Teacher and student parameters are not assumed to be directly compatible. A future mapping layer must explicitly describe transformations between teacher and student representations. Direct tensor transplantation is an experimental operation and must be validated by the evaluator.

## Safety and integrity

Teacher observation is read-only by default. The protocol must never grant the student write access to the teacher's parameters. Any experimental export is represented as a versioned artifact and never as an implicit mutation of the teacher.

## Planned implementation order

1. Define observation data types and stable IDs in Rust.
2. Add a read-only `TeacherObserver` interface.
3. Capture teacher structure/parameter metadata.
4. Capture controlled tensor snapshots and deltas.
5. Add activation/logit observation hooks.
6. Add behaviour/tool traces.
7. Feed observations into the student training loop.
8. Add cross-architecture mapping experiments.
9. Evaluate parameter-pattern transfer separately from behavioural distillation.

The teacher protocol is a first-class architectural boundary and must exist before the Transformer implementation is coupled to a particular teacher model.