// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 00:00:00
// REASON FOR CREATION: The code book maps concepts to symbols but says nothing about
//   inflection, and inflection is where the saving actually is: measured on a real corpus,
//   28.9% of the distinct vocabulary is repeated forms of words already present. Writing
//   `skribas` as root-plus-tense instead of as its own entry is what keeps the book small
//   enough to be a tokenizer vocabulary.
// MECHANICS: A word is normalised onto the diacritic spelling the book uses, then matched
//   whole; failing that, its Esperanto ending is stripped and the bare root looked up, and
//   the ending re-emitted as a separate mark after a separator. Plural and accusative
//   attach as their own marks. Decoding runs the same rules backwards, matching the longest
//   symbol first so multi-character symbols survive. Words with no entry pass through
//   unchanged and are counted, never silently dropped.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The grammar half of the writing system. book.rs holds what a
//   symbol means; this holds how symbols combine into words.
// DEPENDENCIES/LINKS: book::Book; consumed by the encoding binaries and, later, by the
//   tokenizer that feeds the runtime
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! Encoding and decoding between Esperanto text and CBMS symbols.

use crate::book::Book;

/// Esperanto is written three ways in the wild: with its diacritics, in the x-system
/// (`cx` for `ĉ`), and in bare ascii with the diacritic simply dropped. The book uses
/// diacritics, so the other two have to be folded onto it before any lookup.
const X_SYSTEM: [(&str, char); 6] = [
    ("cx", 'ĉ'), ("gx", 'ĝ'), ("hx", 'ĥ'), ("jx", 'ĵ'), ("sx", 'ŝ'), ("ux", 'ŭ'),
];
const BARE_ASCII: [(char, char); 6] = [
    ('c', 'ĉ'), ('g', 'ĝ'), ('h', 'ĥ'), ('j', 'ĵ'), ('s', 'ŝ'), ('u', 'ŭ'),
];

/// Grammatical endings, longest first so `-as` is tried before `-a`.
const ENDINGS: [(&str, &str); 9] = [
    ("-as", "as"), ("-is", "is"), ("-os", "os"), ("-us", "us"),
    ("-u", "u"), ("-i", "i"), ("-o", "o"), ("-a", "a"), ("-e", "e"),
];

/// Endings a bare root may already carry, stripped before a different ending is applied.
const ROOT_ENDINGS: [char; 4] = ['o', 'a', 'e', 'i'];

const PLURAL: &str = "-j";
const ACCUSATIVE: &str = "-n";
/// Esperanto has exactly three number/case combinations: `j`, `n`, `jn`. Giving the
/// third its own mark is what holds every word to three symbols instead of four.
/// Optional: a book without it falls back to writing the two marks separately.
const PLURAL_ACCUSATIVE: &str = "-jn";
const SEPARATOR: &str = "MORPH-SEP";

pub struct Codec<'b> {
    book: &'b Book,
    /// Only used when reading older text. Encoding never emits it: the grammar marks
    /// live in Geometric Shapes (U+25A0..U+25CF) and no lexical symbol in the book does,
    /// so a decoder can tell a mark from a root by its codepoint alone. Verified against
    /// the real book: 11 marks against 441 lexical characters, zero overlap.
    sep: String,
    plural: String,
    accusative: String,
    plural_accusative: String,
}

#[derive(Debug, Default, Clone)]
pub struct Coverage {
    pub words: usize,
    pub encoded: usize,
    /// Words with no entry, in first-seen order. These are what a bigger book would take.
    pub missing: Vec<String>,
}

impl Coverage {
    pub fn ratio(&self) -> f64 {
        if self.words == 0 { 0.0 } else { self.encoded as f64 / self.words as f64 }
    }
}

impl<'b> Codec<'b> {
    /// The separator is optional now that marks are recognised by codepoint block.
    pub fn new(book: &'b Book) -> Option<Self> {
        Some(Codec {
            sep: book.symbol_for(SEPARATOR).unwrap_or_default().to_string(),
            plural: book.symbol_for(PLURAL).unwrap_or_default().to_string(),
            accusative: book.symbol_for(ACCUSATIVE).unwrap_or_default().to_string(),
            plural_accusative: book.symbol_for(PLURAL_ACCUSATIVE).unwrap_or_default().to_string(),
            book,
        })
    }

    /// Is this symbol a grammatical mark rather than a concept?
    fn is_mark(&self, sym: &str) -> bool {
        self.book
            .root_for(sym)
            .is_some_and(|root| root.starts_with('-'))
    }

    /// Fold a word onto the spelling the book uses.
    fn normalise(&self, word: &str) -> String {
        let mut w = word.to_lowercase();
        // The book's own spelling wins. Some of its keys are themselves written in the
        // x-system, and rewriting those into diacritics would walk away from the entry.
        if self.book.contains_root(&w) {
            return w;
        }
        for (pair, ch) in X_SYSTEM {
            if w.contains(pair) {
                w = w.replace(pair, &ch.to_string());
            }
        }
        if self.book.contains_root(&w) {
            return w;
        }
        // Bare ascii: restore a diacritic where doing so produces a known root.
        let chars: Vec<char> = w.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            if let Some((_, accented)) = BARE_ASCII.iter().find(|(plain, _)| *plain == ch) {
                let mut candidate: String = chars[..i].iter().collect();
                candidate.push(*accented);
                candidate.extend(chars[i + 1..].iter());
                if self.book.contains_root(&candidate) {
                    return candidate;
                }
            }
        }
        w
    }

    /// One word to symbols. Returns `None` when the book has no entry, so the caller
    /// decides what to do rather than receiving a silent passthrough it cannot detect.
    pub fn encode_word(&self, word: &str) -> Option<String> {
        let w = self.normalise(word);
        let mut stem = w.as_str();
        let mut plural = false;
        let mut accusative = false;

        // Case and number come off first, but only when what remains is still a word.
        if !self.book.contains_root(stem) {
            if let Some(rest) = stem.strip_suffix('n') {
                if rest.chars().count() > 1 {
                    accusative = true;
                    stem = rest;
                }
            }
        }
        if !self.book.contains_root(stem) {
            if let Some(rest) = stem.strip_suffix('j') {
                if rest.chars().count() > 1 {
                    plural = true;
                    stem = rest;
                }
            }
        }

        let mut out = String::new();
        if let Some(sym) = self.book.symbol_for(stem) {
            out.push_str(sym);
        } else {
            // No separator: the mark's own codepoint block says it is a mark.
            let (sym, mark) = self.split_ending(stem)?;
            out.push_str(sym);
            out.push_str(mark);
        }
        match (plural, accusative) {
            (true, true) if !self.plural_accusative.is_empty() => {
                out.push_str(&self.plural_accusative)
            }
            (true, true) => {
                out.push_str(&self.plural);
                out.push_str(&self.accusative);
            }
            (true, false) => out.push_str(&self.plural),
            (false, true) => out.push_str(&self.accusative),
            (false, false) => {}
        }
        Some(out)
    }

    /// Longest symbol at `pos`, and how many characters it took.
    fn symbol_at(&self, chars: &[char], pos: usize) -> Option<(String, usize)> {
        for &len in self.book.symbol_lengths() {
            if pos + len > chars.len() {
                continue;
            }
            let candidate: String = chars[pos..pos + len].iter().collect();
            if self.book.root_for(&candidate).is_some() {
                return Some((candidate, len));
            }
        }
        None
    }

    /// Strip an Esperanto ending and find the bare root under it.
    fn split_ending(&self, word: &str) -> Option<(&str, &str)> {
        for (tag, suffix) in ENDINGS {
            let Some(stem) = word.strip_suffix(suffix) else { continue };
            if stem.chars().count() < 2 {
                continue;
            }
            let mark = self.book.symbol_for(tag)?;
            // The book stores roots in their citation form, which already carries an
            // ending: `skribi`, `homo`. Try the bare stem and each citation form.
            for candidate in [
                stem.to_string(),
                format!("{stem}i"),
                format!("{stem}o"),
                format!("{stem}a"),
                format!("{stem}e"),
            ] {
                if let Some(sym) = self.book.symbol_for(&candidate) {
                    return Some((sym, mark));
                }
            }
        }
        None
    }

    /// Whole text. Unknown words are kept verbatim and counted.
    pub fn encode_text(&self, text: &str) -> (String, Coverage) {
        let mut out = Vec::new();
        let mut cov = Coverage::default();
        for token in text.split_whitespace() {
            let trimmed = token.trim_matches(|c: char| c.is_ascii_punctuation());
            if trimmed.is_empty() {
                out.push(token.to_string());
                continue;
            }
            cov.words += 1;
            match self.encode_word(trimmed) {
                Some(sym) => {
                    cov.encoded += 1;
                    out.push(sym);
                }
                None => {
                    cov.missing.push(trimmed.to_string());
                    out.push(trimmed.to_string());
                }
            }
        }
        (out.join(" "), cov)
    }

    /// One encoded word back to Esperanto.
    ///
    /// Reads a root, then whatever marks follow it. Marks are recognised by their entry
    /// name starting with `-`, so no separator is needed; one is skipped if present, so
    /// text written by the earlier separator-emitting version still reads.
    pub fn decode_word(&self, encoded: &str) -> Option<String> {
        let chars: Vec<char> = encoded.chars().collect();
        let (root_sym, taken) = self.symbol_at(&chars, 0)?;
        if self.is_mark(&root_sym) {
            return None; // a word cannot begin with a grammatical mark
        }
        let mut word = self.book.root_for(&root_sym)?.to_string();
        let mut pos = taken;
        let mut plural = false;
        let mut accusative = false;

        while pos < chars.len() {
            // Tolerate the separator the earlier format wrote.
            if !self.sep.is_empty() {
                let n = self.sep.chars().count();
                if pos + n <= chars.len() && chars[pos..pos + n].iter().collect::<String>() == self.sep {
                    pos += n;
                    continue;
                }
            }
            let Some((sym, len)) = self.symbol_at(&chars, pos) else { break };
            let Some(tag) = self.book.root_for(&sym) else { break };
            if !tag.starts_with('-') {
                break; // another concept: not part of this word
            }
            match tag {
                PLURAL => plural = true,
                ACCUSATIVE => accusative = true,
                PLURAL_ACCUSATIVE => {
                    plural = true;
                    accusative = true;
                }
                _ => {
                    // A part-of-speech or tense ending replaces the citation form's own.
                    let Some((_, suffix)) = ENDINGS.iter().find(|(t, _)| *t == tag) else { break };
                    if word.chars().last().is_some_and(|c| ROOT_ENDINGS.contains(&c)) {
                        word.pop();
                    }
                    word.push_str(suffix);
                }
            }
            pos += len;
        }

        if plural {
            word.push('j');
        }
        if accusative {
            word.push('n');
        }
        Some(word)
    }

    pub fn decode_text(&self, encoded: &str) -> String {
        encoded
            .split_whitespace()
            .map(|w| self.decode_word(w).unwrap_or_else(|| w.to_string()))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::Book;

    const BOOK: &str = "CODEBOOK_CBMS_ES\n\
        mi=Α\n\
        ni=Ζ\n\
        homo=Ա\n\
        libro=չ\n\
        skribi=б\n\
        legi=в\n\
        esti=К\n\
        hodiaŭ=ᵧ\n\
        instruisto=Մ\n\
        longo=ታ\n\
        \n\
        CBMS-Eo-v1.1-EXT\n\
        -o=U+25CB\n\
        -a=U+25CF\n\
        -e=U+25C7\n\
        -i=U+25C6\n\
        -as=U+25B6\n\
        -is=U+25C0\n\
        -os=U+25B2\n\
        -us=U+25BC\n\
        -u=U+25A0\n\
        -n=U+25A1\n\
        -j=U+25AA\n\
        -jn=U+25A3\n\
        MORPH-SEP=U+00B7\n";

    fn codec(book: &Book) -> Codec<'_> {
        Codec::new(book).expect("book carries a separator")
    }

    #[test]
    fn a_bare_root_is_one_symbol() {
        let b = Book::parse(BOOK).unwrap();
        assert_eq!(codec(&b).encode_word("homo").as_deref(), Some("Ա"));
    }

    #[test]
    fn tense_becomes_a_separate_mark_instead_of_its_own_entry() {
        // This is the whole point: `skribas`, `skribis`, `skribos` share one book entry.
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        assert_eq!(c.encode_word("skribas").as_deref(), Some("б\u{25B6}"));
        assert_eq!(c.encode_word("skribis").as_deref(), Some("б\u{25C0}"));
        assert_eq!(c.encode_word("skribos").as_deref(), Some("б\u{25B2}"));
    }

    #[test]
    fn no_separator_is_written_because_the_codepoint_block_already_says_it_is_a_mark() {
        let b = Book::parse(BOOK).unwrap();
        let enc = codec(&b).encode_word("skribas").unwrap();
        assert!(!enc.contains('\u{00B7}'), "separator is dead weight: {enc:?}");
        assert_eq!(enc.chars().count(), 2);
    }

    #[test]
    fn text_written_with_the_old_separator_still_reads() {
        let b = Book::parse(BOOK).unwrap();
        assert_eq!(codec(&b).decode_word("б\u{00B7}\u{25B6}").as_deref(), Some("skribas"));
    }

    #[test]
    fn case_and_number_attach_as_their_own_marks() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        assert_eq!(c.encode_word("libron").as_deref(), Some("չ\u{25A1}"));
        assert_eq!(c.encode_word("libroj").as_deref(), Some("չ\u{25AA}"));
    }

    #[test]
    fn plural_accusative_is_one_mark_when_the_book_offers_one() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        // ▣ is □ containing ▪ - the combination written as one symbol.
        assert_eq!(c.encode_word("librojn").as_deref(), Some("չ\u{25A3}"));
        assert_eq!(c.decode_word("չ\u{25A3}").as_deref(), Some("librojn"));
    }

    #[test]
    fn a_book_without_the_combined_mark_falls_back_to_two() {
        let without = BOOK.replace("-jn=U+25A3\n", "");
        let b = Book::parse(&without).unwrap();
        let c = codec(&b);
        assert_eq!(c.encode_word("librojn").as_deref(), Some("չ\u{25AA}\u{25A1}"));
        assert_eq!(c.decode_word("չ\u{25AA}\u{25A1}").as_deref(), Some("librojn"));
    }

    #[test]
    fn no_word_ever_needs_more_than_three_symbols() {
        // The constraint M. Szul set: one word, one to three characters. The worst case
        // in Esperanto is a plural accusative adjective - root, part of speech, and the
        // combined number/case mark.
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        for word in ["homo", "homon", "homoj", "homojn",
                     "skribi", "skribas", "skribis", "skribos", "skribus", "skribu",
                     "longa", "longan", "longaj", "longajn", "longe"] {
            let enc = c.encode_word(word).unwrap_or_else(|| panic!("{word} encodes"));
            let n = enc.chars().count();
            assert!(n <= 3, "{word} -> {enc:?} is {n} symbols, over the limit of 3");
            assert_eq!(
                c.decode_word(&enc).as_deref(),
                Some(word),
                "{word} must survive the round trip at {n} symbols"
            );
        }
    }

    #[test]
    fn the_x_system_and_bare_ascii_reach_the_same_entry() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        assert_eq!(c.encode_word("hodiaŭ").as_deref(), Some("ᵧ"));
        assert_eq!(c.encode_word("hodiaux").as_deref(), Some("ᵧ"), "x-system");
        assert_eq!(c.encode_word("hodiau").as_deref(), Some("ᵧ"), "bare ascii");
    }

    #[test]
    fn a_key_the_book_spells_in_the_x_system_is_not_normalised_away() {
        // Found by round-tripping the real 457-entry book: `mesagxfluo` is stored in the
        // x-system, so folding gx->ĝ before lookup walked away from its own entry.
        let b = Book::parse("CODEBOOK_CBMS_ES\nmesagxfluo=ቿ\nMORPH-SEP=U+00B7\n").unwrap();
        assert_eq!(codec(&b).encode_word("mesagxfluo").as_deref(), Some("ቿ"));
    }

    #[test]
    fn a_word_the_book_does_not_have_is_reported_not_invented() {
        let b = Book::parse(BOOK).unwrap();
        assert_eq!(codec(&b).encode_word("kvantumkomputilo"), None);
    }

    #[test]
    fn round_trip_returns_the_original_words() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        // The correctness gate. If this fails the writing system loses information and
        // nothing built on top of it can be trusted.
        for word in ["homo", "libro", "libron", "libroj", "librojn",
                     "skribas", "skribis", "legas", "estas", "instruisto"] {
            let enc = c.encode_word(word).unwrap_or_else(|| panic!("{word} encodes"));
            let dec = c.decode_word(&enc).unwrap_or_else(|| panic!("{enc} decodes"));
            assert_eq!(dec, word, "round trip failed for {word} via {enc}");
        }
    }

    #[test]
    fn whole_sentences_round_trip_and_report_coverage() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        let (enc, cov) = c.encode_text("mi legas libron hodiaŭ");
        assert_eq!(cov.words, 4);
        assert_eq!(cov.encoded, 4);
        assert_eq!(c.decode_text(&enc), "mi legas libron hodiaŭ");
    }

    #[test]
    fn unknown_words_survive_encoding_untouched() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        let (enc, cov) = c.encode_text("mi uzas kvantumkomputilon");
        assert_eq!(cov.words, 3);
        assert_eq!(cov.encoded, 1, "only `mi` is in this small book");
        assert_eq!(cov.missing, vec!["uzas", "kvantumkomputilon"]);
        assert!(enc.contains("kvantumkomputilon"), "unknown text is preserved: {enc}");
    }

    #[test]
    fn encoding_is_shorter_in_characters_than_what_it_replaces() {
        let b = Book::parse(BOOK).unwrap();
        let c = codec(&b);
        let source = "la instruisto skribas libron";
        let (enc, _) = c.encode_text(source);
        assert!(
            enc.chars().count() < source.chars().count(),
            "{enc:?} ({}) should be shorter than {source:?} ({})",
            enc.chars().count(),
            source.chars().count()
        );
    }
}
