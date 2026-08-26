# NOWORODEK CODER SCHOOL V1

- Project: WPC-ENGINE / Noworodek
- Branch: Noworodek
- Workstream: Coder School / Rust + C++
- Owner/provenance: M.Szul via GPT-5.6 Luna
- Status: implementation committed; local runtime verification pending
- Runner: `noworodek/src/bin/noworodek-code-school.rs`
- WeightSet: `coder-school-v1`
- Architecture: `noworodek-decoder-v0`
- Curriculum: 16 micro-lessons (8 Rust, 8 C++)
- Observation fields: ExperienceID, train loss before/after, held-out loss, external WeightSet mutation
- Current limitation: the V1 student uses an 8-symbol deterministic hash encoding and a tiny 1-layer decoder. It is a curriculum/training-pipeline prototype, not evidence of semantic Rust/C++ competence.
- Next gate: replace hashed encoding with a real code tokenizer and substantially larger student model, while retaining the same ExperienceID -> DeltaW -> Observatory instrumentation.
