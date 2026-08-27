# Noworodek Code Dictionary Ingest V1

- Project: WPC-ENGINE / Noworodek
- Branch: noworodek-code-atoms-v1
- Workstream: Code Memory / Bulk FunctionAtom ingestion
- Provenance: M.Szul via GPT-5.6 Luna
- Purpose: ingest large Rust/C++ source trees into external FunctionAtom memory using the pinned Qwen3-Coder tokenizer.
- Input: recursive source tree with `.rs`, `.c`, `.h`, `.cc`, `.cpp`, `.cxx`, `.hpp`, `.hh`, `.hxx` files.
- Output: deterministic CodeAtomRegistry statistics; duplicate atoms are ignored idempotently.
- Training boundary: this command does NOT update Transformer weights. It only builds external code memory.
- Next stage: sample/retrieve atoms into curriculum experiences, then run real Transformer backprop + Observatory.
- Validation command: `cargo run --release --manifest-path .\\noworodek\\Cargo.toml --bin noworodek-code-dictionary-ingest -- <source-root> <qwen-tokenizer.json>`
