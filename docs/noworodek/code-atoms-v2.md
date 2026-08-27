# Noworodek Code Atom Registry V2

## Metadata

- project: WPC-ENGINE / Noworodek
- branch: noworodek-code-atoms-v1
- workstream: Code Atom Registry V2
- provenance: M.Szul via GPT-5.6 Luna
- scope: Qwen3-Coder tokenization + deterministic structural function spans

## Contract

V2 composes the existing pinned Qwen3-Coder tokenizer with deterministic function-span extraction. The extractor produces an external `TokenizedCodeAtom` containing the stable `CodeAtom`, Qwen token IDs, and source byte span.

## Explicit limitation

V2 is NOT a full Rust/C++ AST parser. Function extraction is delimiter-aware and language-aware but heuristic. Full AST parsing is reserved for V3.

## Purpose

This establishes the representation required for function-level memory without turning every full function into a permanent vocabulary token. A later registry/graph layer can retrieve the entire function, patch, or debug lineage as one external memory atom.

## Verification

Expected local result:

`RESULT qwen_tokenized_function_atoms=true structural_parser=heuristic_v2 ast_engine=false`
