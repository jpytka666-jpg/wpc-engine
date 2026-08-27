# Code Atom Registry V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an external, observable registry for reusable Rust/C++ code units (functions, patches, debug fixes) without turning every full function into a literal tokenizer vocabulary token.

**Architecture:** Keep lexical tokenization separate from structural code memory. A `CodeAtom` stores canonical source metadata, stable identity, language/kind, parent lineage, and optional patch/debug linkage. `CodeAtomRegistry` provides deterministic insert/retrieve/lineage operations; later curriculum and Observatory layers can associate `ExperienceId → CodeAtomId → ΔW`.

**Tech Stack:** Rust, existing `noworodek` crate, deterministic FNV-1a hashing, serde-free core registry for V1, existing WeightSet/Observatory APIs.

**Spec:** `docs/noworodek/code-atom-registry-v1.md`

## Global Constraints

- Project: WPC-ENGINE / Noworodek.
- Workstream branch: `noworodek-code-atoms-v1`.
- Rust implementation only.
- Code atoms are external metadata/memory units, not literal vocabulary entries.
- Existing Transformer forward/backprop contracts remain unchanged.
- Every persistent code-memory object must be deterministic and observable.

---

## Task 1 — Write the spec

- [ ] Define `CodeAtom`, `CodeAtomId`, `CodeAtomKind`, `CodeAtomRegistry`, and lineage invariants.
- [ ] Define deterministic ID derivation from language + kind + canonical source + parent + version.
- [ ] Define retrieval and patch/debug lineage semantics.
- [ ] Define V1 non-goals: no AST parser dependency yet; no Qwen tokenizer mutation yet.

## Task 2 — RED tests

- [ ] Add unit/integration tests for stable atom IDs.
- [ ] Add tests that duplicate registration is idempotent.
- [ ] Add tests for parent/child lineage and patch lineage.
- [ ] Add tests for lookup by ID and language/kind filtering.
- [ ] Add a test showing a debug patch points from failing atom to repaired atom.

## Task 3 — Minimal implementation

- [ ] Implement the core atom types and deterministic ID.
- [ ] Implement registry insert/get/contains/list/children.
- [ ] Implement patch lineage and debug lineage.
- [ ] Export the module through `noworodek::code_atoms`.

## Task 4 — Observatory bridge

- [ ] Add an `experience_id` field to atom observations without coupling atom storage to WeightSet internals.
- [ ] Add an event constructor for `experience → atom → verification`.
- [ ] Keep the event payload read-only and replayable.

## Task 5 — Runner

- [ ] Add `noworodek-code-atom-demo` that creates a Rust function atom, a failing/debug atom, and a repaired child atom.
- [ ] Print deterministic IDs, lineage, and registry counts.
- [ ] Add a small held-out retrieval check.

## Task 6 — Verification and metadata

- [ ] Run the targeted tests and full `cargo test` locally.
- [ ] Record outcome and provenance in `docs/noworodek/code-atom-registry-v1.md`.
- [ ] Commit RED and GREEN separately with project/branch/workstream metadata.
- [ ] Open a PR from `noworodek-code-atoms-v1` to `Noworodek` after GREEN.
