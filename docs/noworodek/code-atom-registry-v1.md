# Code Atom Registry V1

## Status
Prototype specification — workstream `noworodek-code-atoms-v1`.

## Goal
Represent reusable Rust/C++ code units as external, deterministic, editable memory objects that can be linked to experiences, patches, failures, and repairs without turning each whole function into a literal tokenizer vocabulary token.

## Core model

A `CodeAtom` contains:
- deterministic `CodeAtomId`;
- language (`Rust` or `Cpp` in V1);
- kind (`Function`, `Block`, `Patch`, `DebugFix`);
- canonical source text;
- optional parent atom;
- optional related/failure atom;
- optional experience identifier;
- semantic version string.

`CodeAtomRegistry` is an external index. It does not own Transformer weights and does not mutate the tokenizer vocabulary.

## Deterministic identity

`CodeAtomId` is derived from:
`language + kind + canonical source + parent id + version` using deterministic FNV-1a 64-bit hashing.

Whitespace is canonicalized only at the outer edge (trim); internal source text is preserved in V1. A byte-identical input therefore produces a byte-identical ID.

## Lineage

- `original -> patch` uses `parent_id`.
- `failure -> debug fix` uses `related_id` plus `parent_id` when the repaired atom replaces an original.
- An atom can have at most one parent in V1.
- Registry insertion is idempotent for the same ID.

## V1 non-goals

- No AST parser dependency yet.
- No modification of Qwen3-Coder tokenizer vocabulary.
- No automatic semantic equivalence proof.
- No claim that an atom is a token inside the model.

## Next phase

V2 adds a real code tokenizer/parser bridge, structural extraction of functions and AST subtrees, and retrieval of atoms as compressed structural units alongside lexical token sequences.
