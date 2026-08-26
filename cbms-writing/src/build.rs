// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:34:15
// REASON FOR CREATION: The book is the bottleneck and the measurement says so: 747 ids
//   cover 0.56% of the AIONS corpus, so 70.7% of encoded ids are literal bytes. A book
//   written by hand cannot catch up with a corpus; it has to be built from one.
// MECHANICS: Counts words across a corpus, drops those the base book already encodes,
//   and mints a symbol for the most frequent of what remains. Minting happens HERE and
//   never during encoding: an encoder that invents symbols produces a book the receiver
//   does not have, and ids that shift between runs. New symbols come from codepoint
//   ranges the hand-written book does not use, so machine-minted entries are visibly
//   distinct from M. Szul's own.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The offline half. Produces a frozen, versioned book; everything
//   else consumes one. Extension is additive - existing entries keep their symbols, so
//   ids already issued keep their meaning and stored data stays readable.
// DEPENDENCIES/LINKS: book::Book, codec::Codec
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! Building a code book from a corpus, additively and offline.

use crate::book::Book;
use crate::codec::Codec;
use std::collections::{HashMap, HashSet};

/// Codepoint ranges for machine-minted entries, chosen because the hand-written book
/// uses none of them. Keeping them separate means a glance at a symbol says whether a
/// person chose it or a frequency count did.
const MINT_RANGES: [(u32, u32); 4] = [
    (0x1400, 0x167F), // Unified Canadian Aboriginal Syllabics - 640 codepoints
    (0x13A0, 0x13F5), // Cherokee - 86
    (0x10A0, 0x10FA), // Georgian - 91
    (0xA000, 0xA48C), // Yi Syllables - 1165
];

#[derive(Debug, Clone)]
pub struct WordCount {
    pub word: String,
    pub count: usize,
}

#[derive(Debug, Default)]
pub struct CorpusStats {
    pub total_words: usize,
    pub distinct_words: usize,
    /// Occurrences the base book can already encode.
    pub covered_occurrences: usize,
}

impl CorpusStats {
    pub fn coverage(&self) -> f64 {
        if self.total_words == 0 { 0.0 } else { self.covered_occurrences as f64 / self.total_words as f64 }
    }
}

/// Split text into candidate words. Deliberately crude and deliberately fixed: the
/// same rule has to run when the book is built and when text is encoded, or the book
/// will contain entries the encoder can never look up.
pub fn words_of(text: &str) -> Vec<String> {
    text.split(|c: char| c.is_whitespace())
        .map(|w| w.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

/// Count words and note how many occurrences the base book already handles.
pub fn survey(book: &Book, texts: &[String]) -> (Vec<WordCount>, CorpusStats) {
    let codec = Codec::new(book);
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut stats = CorpusStats::default();

    for text in texts {
        for w in words_of(text) {
            stats.total_words += 1;
            *counts.entry(w).or_default() += 1;
        }
    }
    stats.distinct_words = counts.len();

    if let Some(codec) = &codec {
        for (w, n) in &counts {
            if codec.encode_word(w).is_some() {
                stats.covered_occurrences += n;
            }
        }
    }

    let mut ranked: Vec<WordCount> =
        counts.into_iter().map(|(word, count)| WordCount { word, count }).collect();
    // Frequency first, then alphabetically, so the same corpus always yields the same
    // book. A book that depends on hash order cannot be rebuilt or reviewed.
    ranked.sort_by(|a, b| b.count.cmp(&a.count).then(a.word.cmp(&b.word)));
    (ranked, stats)
}

/// Codepoints free for minting: in a mint range and not already used by the book.
fn available(book: &Book) -> Vec<char> {
    let used: HashSet<char> = book.entries().iter().flat_map(|e| e.symbol.chars()).collect();
    let mut out = Vec::new();
    for (lo, hi) in MINT_RANGES {
        for cp in lo..=hi {
            if let Some(ch) = char::from_u32(cp) {
                if !used.contains(&ch) {
                    out.push(ch);
                }
            }
        }
    }
    out
}

#[derive(Debug)]
pub struct BuildReport {
    pub added: usize,
    pub skipped_already_known: usize,
    pub ran_out_of_symbols: usize,
    pub coverage_before: f64,
    pub coverage_after: f64,
    pub book_text: String,
}

/// Extend a book with the most frequent words it cannot encode.
///
/// `min_count` keeps one-off noise out: a word seen once is a typo, an identifier or a
/// hash, and spending a symbol on it makes the vocabulary bigger without making the
/// corpus shorter.
pub fn extend(book: &Book, texts: &[String], max_new: usize, min_count: usize) -> BuildReport {
    let (ranked, stats) = survey(book, texts);
    let codec = Codec::new(book);
    let mut pool = available(book).into_iter();

    let mut added_lines = Vec::new();
    let mut added = 0usize;
    let mut skipped = 0usize;
    let mut exhausted = 0usize;
    let mut newly_covered = 0usize;

    for wc in &ranked {
        if added >= max_new || wc.count < min_count {
            break;
        }
        if codec.as_ref().is_some_and(|c| c.encode_word(&wc.word).is_some()) {
            skipped += 1;
            continue;
        }
        match pool.next() {
            Some(symbol) => {
                added_lines.push(format!("{}={}", wc.word, symbol));
                added += 1;
                newly_covered += wc.count;
            }
            None => {
                exhausted += 1;
            }
        }
    }

    let mut text = book.to_text();
    if !added_lines.is_empty() {
        text.push_str("\n\nCODEBOOK_CBMS_ES\n");
        text.push_str("# minted from corpus frequency, not chosen by hand\n");
        for line in &added_lines {
            text.push_str(line);
            text.push('\n');
        }
    }

    BuildReport {
        added,
        skipped_already_known: skipped,
        ran_out_of_symbols: exhausted,
        coverage_before: stats.coverage(),
        coverage_after: if stats.total_words == 0 {
            0.0
        } else {
            (stats.covered_occurrences + newly_covered) as f64 / stats.total_words as f64
        },
        book_text: text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BASE: &str = "CODEBOOK_CBMS_ES\n\
        homo=Ա\n\
        libro=չ\n\
        \n\
        CBMS-Eo-v1.1-EXT\n\
        -o=U+25CB\n-a=U+25CF\n-e=U+25C7\n-i=U+25C6\n\
        -as=U+25B6\n-is=U+25C0\n-os=U+25B2\n-us=U+25BC\n-u=U+25A0\n\
        -n=U+25A1\n-j=U+25AA\n-jn=U+25A3\n\
        MORPH-SEP=U+00B7\n";

    fn corpus() -> Vec<String> {
        vec![
            "status status status chunk chunk libro".to_string(),
            "status chunk commit".to_string(),
        ]
    }

    #[test]
    fn frequent_unknown_words_get_symbols_and_rare_ones_do_not() {
        let base = Book::parse(BASE).unwrap();
        // status 4, chunk 3, commit 1, libro 1
        let r = extend(&base, &corpus(), 100, 2);
        assert_eq!(r.added, 2, "status and chunk clear min_count; commit does not");
    }

    #[test]
    fn extending_is_additive_so_existing_symbols_do_not_move() {
        let base = Book::parse(BASE).unwrap();
        let r = extend(&base, &corpus(), 100, 2);
        let grown = Book::parse(&r.book_text).expect("built book parses");
        assert_eq!(grown.symbol_for("homo"), Some("Ա"), "hand-written entry untouched");
        assert_eq!(grown.symbol_for("libro"), Some("չ"));
        assert!(grown.symbol_for("status").is_some(), "new entry present");
    }

    #[test]
    fn minted_symbols_come_from_ranges_the_hand_written_book_does_not_use() {
        let base = Book::parse(BASE).unwrap();
        let r = extend(&base, &corpus(), 100, 2);
        let grown = Book::parse(&r.book_text).unwrap();
        let sym = grown.symbol_for("status").unwrap().chars().next().unwrap() as u32;
        assert!(
            MINT_RANGES.iter().any(|(lo, hi)| (*lo..=*hi).contains(&sym)),
            "U+{sym:04X} is outside the mint ranges"
        );
    }

    #[test]
    fn the_built_book_has_no_collisions() {
        // Book::parse refuses a colliding book, so this passing IS the check.
        let base = Book::parse(BASE).unwrap();
        let r = extend(&base, &corpus(), 500, 1);
        Book::parse(&r.book_text).expect("minting must not reuse a symbol");
    }

    #[test]
    fn coverage_is_reported_and_goes_up() {
        let base = Book::parse(BASE).unwrap();
        let r = extend(&base, &corpus(), 100, 2);
        assert!(r.coverage_after > r.coverage_before,
                "{} should exceed {}", r.coverage_after, r.coverage_before);
    }

    #[test]
    fn the_same_corpus_always_produces_the_same_book() {
        // Ties break alphabetically rather than by hash order, or the book could not
        // be rebuilt, reviewed in a diff, or agreed on by two machines.
        let base = Book::parse(BASE).unwrap();
        let a = extend(&base, &corpus(), 100, 1);
        let b = extend(&base, &corpus(), 100, 1);
        assert_eq!(a.book_text, b.book_text);
    }

    #[test]
    fn words_are_split_the_same_way_the_encoder_will_see_them() {
        assert_eq!(words_of("Status: chunk, commit."), vec!["status", "chunk", "commit"]);
    }
}
