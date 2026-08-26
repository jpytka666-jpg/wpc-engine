# WPC/AIONS Agent Integration — Addendum to WHITEPAPER.md

**21 August 2026 — post-revision-2 engineering update**

## 1. Purpose

The WPC runtime now has an AIONS-facing autonomous agent loop. The design is intentionally separate from the compression format: WPC remains the inference engine, while the agent provides planning, tool selection, execution, and transcript management.

## 2. Dynamic tool discovery

The agent connects to an AIONS MCP server and performs the standard initialization sequence followed by `tools/list`. The complete live tool catalogue — names, descriptions, and input schemas — is inserted into the model's system context.

This is deliberately dynamic. The agent does not assume that AIONS has a particular number of tools. If the MCP server exposes a new tool, the agent can discover it without recompilation.

## 3. Execution loop

```text
Task
  -> live MCP tool catalogue
  -> WPC Qwen3-Coder-30B-A3B v4
  -> TOOL_CALL {name, arguments}
  -> AIONS MCP tools/call
  -> TOOL_RESULT
  -> transcript
  -> next model turn
```

The model is constrained to emit either a `TOOL_CALL` record or a `FINAL` record. Tool results are appended to the transcript and the loop continues until a final response or the configured turn limit is reached.

## 4. Current implementation boundary

The MCP connection is persistent across turns, but the WPC runtime is currently launched as a subprocess for each model turn. This avoids coupling the agent to the runtime's internal Rust model API and makes the integration easy to test.

The next performance step is to make the WPC runtime long-lived across the entire agent session. That would keep model weights resident and, more importantly, make persistent KV-cache reuse possible between tool turns instead of rebuilding the model execution state for every turn.

## 5. Relation to the main performance roadmap

The agent layer does not solve the principal runtime limitation identified in revision 2: one forward pass per token. The next architectural optimisation remains batched forward execution. Once that exists, the same agent loop can benefit from fast prompt prefill, speculative verification, and expert-grouped execution.

The important separation is therefore:

- **WPC** — compressed model storage and execution.
- **Router/scheduler** — decides which specialists/tools are required.
- **AIONS MCP** — exposes the available external capabilities.
- **KV cache** — keeps active model state on the hot path; it should not be replaced by SSD/CBMS during token generation.
- **CBMS** — persistent storage for knowledge, transcripts, and potentially reusable KV snapshots outside the real-time token path.

This addendum records the integration as an engineering milestone without claiming that the remaining throughput work has already been measured.

---

# 6. Noworodek — Learning by Observing an External Teacher Agent

**26 August 2026 — approved architectural extension**

Noworodek is a from-scratch Rust operator model whose defining property is that trainable parameters are externalized, addressable, observable, editable, versioned, and grouped into interchangeable `WeightSet`s. This extension adds a second defining capability: Noworodek can learn from an externally observed teacher agent such as Claude Code working with a human operator.

The objective is not to capture or reproduce a teacher's private chain-of-thought. The training signal is the **observable trajectory of work**: user intent, visible context, tool calls, tool results, code changes, tests, CI outcomes, explicit rationale supplied by the teacher, corrections, and final outcomes.

## 6.1 System concept

```text
                         HUMAN
                           |
                           v
                      TEACHER AGENT
                       (e.g. Claude)
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
        REPO             TOOLS            OUTPUT
          |                |                |
          +----------------+----------------+
                           v
                    OBSERVATION BUS
                           |
               +-----------+-----------+
               |           |           |
               v           v           v
            ACTIONS     OUTCOMES      STATE
               |           |           |
               +-----------+-----------+
                           v
                   EXPERIENCE STORE
                           |
                           v
                       NOWORODEK
                           |
                 +---------+---------+
                 |                   |
                 v                   v
              TRAINING           EVALUATION
                 |                   |
                 +---------+---------+
                           v
                      WeightSets
```

The teacher is therefore treated as an **observable environment participant**, not as a source for indiscriminate model-weight copying.

## 6.2 What Noworodek observes

The observation protocol is divided into independent layers:

### A. User intent and task context

- task statement;
- relevant visible context;
- declared constraints;
- repository/branch/commit state where exposed.

### B. Teacher actions

- file inspection;
- code/search operations;
- tool calls and arguments;
- test/build commands;
- edits and patches;
- Git operations;
- CI inspection;
- explicit decision/rationale text when the teacher provides it.

### C. Environment observations

- tool results;
- stdout/stderr;
- exit codes;
- test results;
- CI results;
- changed files and diffs;
- resulting repository state.

### D. Behaviour and outcomes

- action sequence;
- tool trajectory;
- final result;
- evaluator evidence;
- success/failure/partial-success score.

### E. Optional model-internal observations

When the teacher model explicitly exposes them through a compatible instrumentation interface, the system may record:

- parameter manifests;
- tensor statistics;
- weight snapshots;
- weight deltas;
- selected activations;
- hidden states;
- logits.

These observations are **read-only**. The student is never granted implicit write access to the teacher.

## 6.3 Raw trace versus normalized experience

Two representations are required:

```text
RAW TRACE
   |
   v
NORMALIZED EXPERIENCE
   |
   v
TRAINING TARGETS
```

The raw trace preserves provenance and allows reprocessing. The normalized experience is a stable training schema:

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

This separation permits old sessions to be re-evaluated or converted into multiple specialist datasets without changing the captured source record.

## 6.4 Learning principle

Noworodek must not learn merely because the teacher performed an action. It must learn from **action + observation + outcome**.

For example:

```text
Teacher:
inspect -> hypothesize -> patch -> test
                         |
                         v
                     TEST FAIL
                         |
                         v
                     revise -> test
                         |
                         v
                      PASS
```

The evaluator provides the evidence signal. This prevents the training system from treating every teacher action as correct by default.

## 6.5 Weight observability and learning

All trainable Noworodek tensors are externalized from the model core and identified by stable names such as:

```text
model.layers.03.attention.q_proj.weight
model.layers.03.attention.k_proj.weight
model.layers.03.attention.v_proj.weight
model.layers.03.mlp.up_proj.weight
```

Training observation records may then associate:

```text
experience
   -> training step
   -> WeightSet version
   -> W_before
   -> W_after
   -> delta statistics
   -> behaviour/evaluation result
```

The system must not claim that a single tensor or scalar parameter corresponds to a single human concept. Knowledge and behaviour are represented distributively. The purpose of observation is to measure parameter changes correlated with experiences and resulting behaviour.

## 6.6 Teacher model observation protocol

Teacher integration is defined by a generic `TeacherObserver` boundary rather than a provider-specific interface:

```text
TeacherObserver
   +--> ClaudeCodeTeacher
   +--> HumanTeacher
   +--> OtherLLMTeacher
   +--> ScriptedTeacher
```

The protocol records teacher identity, architecture, tensor metadata when available, observation IDs, training steps, experiences, deltas, and behaviour traces. Teacher parameters remain read-only.

## 6.7 Cross-architecture learning

Teacher and student models are not assumed to have identical parameter shapes or layer counts. Direct tensor transplantation is therefore an explicit experimental operation, never an implicit part of ordinary learning.

The learning stack is expected to support three distinct signal classes:

1. **Behavioural learning** — learn useful actions and tool trajectories.
2. **Representation learning** — learn from selected teacher activations/logits where exposed.
3. **Parameter-pattern experiments** — analyse teacher weight patterns and experimentally map them into compatible student WeightSets.

These classes are evaluated separately so that a gain from behavioural imitation is not incorrectly attributed to parameter transfer.

## 6.8 Specialist WeightSets produced from experience

Experiences may be classified into specialist training streams such as:

```text
GeneralSet
CodingSet
RustSet
DebuggingSet
ToolUseSet
AIONSOperatorSet
VerificationSet
PlanningSet
```

The WeightSet manager remains responsible for versioning, mounting, replacing, rollback, and compatibility checks. Specialist sets do not imply duplicate full models; shared core parameters may remain common while specialist parameter sets provide targeted adaptation.

## 6.9 Safety and integrity boundary

- Teacher observation is read-only by default.
- Private chain-of-thought is not a required training source.
- Teacher access is never converted into arbitrary mutation authority.
- Raw traces preserve provenance.
- Normalized experiences carry evaluator evidence.
- Weight edits are versioned and reversible.
- No claim of learned competence is accepted without an evaluator result.

## 6.10 Research questions

The implementation will allow measurable experiments on:

- whether tool-use competence can be learned from observed trajectories;
- how much behaviour transfers without copying teacher weights;
- whether selected activations/logits improve transfer;
- whether parameter deltas correlate with specific training experiences;
- whether specialist WeightSets can be learned independently and hot-swapped;
- whether experimentally mapped teacher parameter patterns improve student benchmarks;
- whether WPC can later serve as a backend for these externalized WeightSets.

---

# 7. Representation strategy — proven fused WPC first, Low-Rank second

**26 August 2026 — evidence-driven architecture update**

The project now distinguishes between what has been **measured** and what remains a research hypothesis.

## 7.1 Proven result: execution without full materialization

The current fused WPC decode-and-multiply kernel has already demonstrated the core execution idea: the compressed weight is consumed inside the multiply and the fully expanded matrix is never materialized.

Recorded run:

```text
Tensor                  : model.layers.12.self_attn.o_proj.weight
Shape                   : 2560 x 4096
Compressed              : 5,570,560 B (5.3 MiB)
Expanded dense size     : 40.0 MiB
Expanded tensor         : never materialized by this kernel
Fused kernel            : 1.043 ms
Throughput              : 10,049.7 M weights/s
Compute                 : 20.09 GFLOP/s
CPU comparison          : 16–18x faster (reported run)
Error GPU               : 2.235e-04
Error CPU               : 1.592e-03
Result                  : PASS
Log                     : gpu/wpc4-decode/runs/gemv_2026-08-25_1522.log
```

This is an **experiment-specific measured result**, not a universal performance claim. The key architectural fact established by the run is the absence of full dense materialization in the hot path.

The repository also contains an earlier design comment that anticipated this execution form: a fused decode-and-multiply kernel should consume the compressed representation directly instead of first expanding the whole tensor.

## 7.2 What remains different in Noworodek

The current WPC representation is predetermined:

```text
one representation
+ one optimized execution kernel
```

Noworodek proposes:

```text
representation selected per tensor
+ editable representation parameters
+ observable representation changes
```

That is a qualitative architecture extension. It must not be presented as already solved.

## 7.3 Low-Rank is the first alternative representation

The first new representation is **Low-Rank**:

```text
W = A * B
```

with small rank `r`.

Storage changes from:

```text
n * m
```

to:

```text
(n + m) * r
```

For `2560 x 4096` and `r = 64`, factor storage is about twelve times smaller than dense FP32 storage before metadata and other representation costs.

Execution can use the standard associative form:

```text
y = A * (B * x)
```

and the trainable factors have a direct gradient path. This makes low-rank the most controlled first experiment across storage, execution, and training complexity.

## 7.4 Required rank sweep

No rank is chosen by intuition. The first benchmark must evaluate the real tensor above at:

```text
r = 8, 16, 32, 64, 128, 256
```

For every rank measure:

- reconstruction error;
- representation bytes;
- compression ratio;
- execution latency without full dense materialization;
- GPU versus CPU latency;
- training stability;
- downstream model quality.

The result becomes evidence for a future representation selector.

## 7.5 Representation-kernel rule

Every fundamentally new representation requires an execution implementation that is at least benchmarked before it is promoted. A new storage format without an efficient execution path is not treated as a performance improvement.

This rule explicitly blocks representation proliferation from outrunning kernel engineering.

## 7.6 Continuously editable representation

The representation remains part of the editable model state:

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

For low-rank, Observatory must distinguish factor changes (`dA`, `dB`) from optional dense reconstruction diffs.

## 7.7 Research order

1. Preserve the proven fused WPC path as the baseline.
2. Measure low-rank ranks on the proven target tensor.
3. Compare quality, storage, and execution.
4. Add low-rank training and observable factor deltas.
5. Only then consider additional procedural representations.

Sinusoids, polynomials, recursive generators, and arbitrary learned programs remain research candidates until their execution cost and gradient stability are demonstrated.
