// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:20:03
// REASON FOR CREATION: CBMS was described in several places as a compression scheme and
//   measured, on 2026-08-26, as something narrower and more useful: a writing system. This
//   crate is the first implementation of it that can be executed rather than described.
// MECHANICS: Two halves. `book` parses the code book M. Szul maintains - concepts to
//   symbols - and refuses one whose symbols collide. `codec` holds the grammar: how a root
//   and its inflection combine into an encoded word, and how to read one back.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: Storage and retrieval layer today; the vocabulary source for a
//   CBMS-native tokenizer later. The two share this crate deliberately, so the symbols a
//   model would be trained on are the same ones the store is written in.
// DEPENDENCIES/LINKS: none at runtime; sits beside wpc-runtime in the same workspace
//   because the eventual tokenizer must run inside the inference loop
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! CBMS: a writing system in which one symbol carries one concept.
//!
//! Text is canonicalised into Esperanto first, because Esperanto spells one concept one
//! way. Polish spells it many: `wiedza`, `wiedzy`, `wiedzą`, `wiedzę` are four entries a
//! vocabulary would have to hold and four token sequences a model would have to learn.
//! Measured on a 340 091-character corpus of this project's own text, 28.9% of the
//! distinct vocabulary is repeated forms of words already present.
//!
//! What the encoding buys, measured rather than assumed:
//!
//! | | |
//! |---|---|
//! | characters | 0.57x - nearly half |
//! | tokens, borrowed tokenizer | 1.14x - **worse**, the symbols are not in its vocabulary |
//! | tokens, own vocabulary | 0.80x |
//! | vocabulary needed | ~618 against Qwen3's 151 936 |
//!
//! The third row is the point and the second row is the warning: encoding into a
//! tokenizer that has never seen these codepoints costs more than plain text, because
//! each symbol is split into byte fragments. CBMS pays off only when CBMS is the
//! vocabulary.

pub mod book;
pub mod build;
pub mod codec;
pub mod container;
pub mod huffman;
pub mod vocab;

pub use container::{peek, read, write, ContainerError, Header};

pub use build::{extend, survey, BuildReport, CorpusStats, WordCount};

pub use book::{Book, BookError, Collision, Entry, Section};
pub use codec::{Codec, Coverage};
pub use vocab::Vocabulary;
