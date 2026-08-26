// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:20:03
// REASON FOR CREATION: The codec turns words it knows into symbols and passes the rest
//   through as text. That is a dictionary, not a tokenizer: measured coverage of the
//   AIONS corpus by the current book is 0.56%, so "everything becomes an id" was not true.
//   A tokenizer has to encode ANY input, including a word nobody has ever written down.
// MECHANICS: Assigns a stable id to every book entry, then reserves 256 ids for raw bytes
//   and a handful for control. A word the book knows costs one to three ids; a word it does
//   not is escaped and spelled out in UTF-8 bytes, so nothing is ever lost and nothing has
//   to be guessed. Ids pack into fixed-width bit fields sized from the vocabulary, which is
//   what turns the symbols from odd-looking text into a machine format.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The layer that makes CBMS total. Below it the codec deals in
//   concepts; above it everything is integers, which is what a model, a wire format and a
//   memory store all need.
// DEPENDENCIES/LINKS: book::Book, codec::Codec
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! Stable integer ids for CBMS symbols, with a byte fallback so any input encodes.

use crate::book::Book;
use crate::codec::Codec;
use std::collections::HashMap;

/// Control ids come first so that adding book entries never moves them.
pub const ID_PAD: u16 = 0;
/// Reserved. Kept so the control block keeps its shape as things are added to it;
/// removing it would shift every id after it and make stored data unreadable.
pub const ID_RESERVED: u16 = 1;
/// Separates words. Whitespace is structure, not a concept.
pub const ID_SPACE: u16 = 2;
pub const ID_NEWLINE: u16 = 3;
/// Marks the start of a run of literal bytes; the id after it is the run length.
pub const ID_RAW_RUN: u16 = 4;
/// The next word starts with a capital. Canonicalisation lowercases, so without these
/// every sentence-initial word would fall to the byte fallback in order to keep its
/// capital - paying one id per byte for a word that is otherwise one id, to record one bit.
pub const ID_CAPITAL: u16 = 5;
/// The next word is in capitals throughout.
pub const ID_UPPER: u16 = 6;
const CONTROL_COUNT: u16 = 8;
/// 256 ids for raw bytes, immediately after the control block.
const BYTE_BASE: u16 = CONTROL_COUNT;
const SYMBOL_BASE: u16 = BYTE_BASE + 256;

pub struct Vocabulary<'b> {
    codec: Codec<'b>,
    book: &'b Book,
    symbol_to_id: HashMap<String, u16>,
    id_to_symbol: Vec<String>,
}

impl<'b> Vocabulary<'b> {
    pub fn new(book: &'b Book) -> Option<Self> {
        let codec = Codec::new(book)?;
        let mut symbol_to_id = HashMap::new();
        let mut id_to_symbol = Vec::new();
        // Order follows the book, so a book that only grows keeps every existing id.
        for entry in book.entries() {
            if symbol_to_id.contains_key(&entry.symbol) {
                continue;
            }
            let id = SYMBOL_BASE + id_to_symbol.len() as u16;
            symbol_to_id.insert(entry.symbol.clone(), id);
            id_to_symbol.push(entry.symbol.clone());
        }
        Some(Vocabulary { codec, book, symbol_to_id, id_to_symbol })
    }

    /// Total ids: control, bytes, and one per distinct symbol.
    pub fn len(&self) -> usize {
        SYMBOL_BASE as usize + self.id_to_symbol.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_symbol.is_empty()
    }

    /// Bits needed per id if packed at fixed width.
    pub fn bits_per_id(&self) -> u32 {
        let n = self.len().max(2) as u32;
        32 - (n - 1).leading_zeros()
    }

    pub fn symbol_count(&self) -> usize {
        self.id_to_symbol.len()
    }

    fn id_of_symbol(&self, sym: &str) -> Option<u16> {
        self.symbol_to_id.get(sym).copied()
    }

    fn symbol_of_id(&self, id: u16) -> Option<&str> {
        if id < SYMBOL_BASE {
            return None;
        }
        self.id_to_symbol.get((id - SYMBOL_BASE) as usize).map(|s| s.as_str())
    }

    /// Split an encoded word into its symbols. They are all in the book, so longest
    /// match against known symbols is exact rather than a guess.
    fn symbols_of(&self, encoded: &str) -> Option<Vec<String>> {
        let chars: Vec<char> = encoded.chars().collect();
        let mut out = Vec::new();
        let mut pos = 0;
        'outer: while pos < chars.len() {
            for &len in self.book.symbol_lengths() {
                if pos + len > chars.len() {
                    continue;
                }
                let cand: String = chars[pos..pos + len].iter().collect();
                if self.symbol_to_id.contains_key(&cand) {
                    out.push(cand);
                    pos += len;
                    continue 'outer;
                }
            }
            return None;
        }
        Some(out)
    }

    /// Text to ids. Anything the book does not have is spelled out in bytes, so this
    /// never fails and never invents.
    pub fn encode(&self, text: &str) -> Vec<u16> {
        let mut out = Vec::new();
        let mut first = true;
        for line in text.split('\n') {
            if !first {
                out.push(ID_NEWLINE);
            }
            first = false;
            let mut first_word = true;
            for word in line.split(' ') {
                if !first_word {
                    out.push(ID_SPACE);
                }
                first_word = false;
                if word.is_empty() {
                    continue;
                }
                // Case is recorded as a mark and stripped before lookup, so `Homo` and
                // `homo` reach the same entry without either losing its spelling.
                let (case, folded) = split_case(word);
                match self.codec.encode_word(&folded).and_then(|e| self.symbols_of(&e)) {
                    Some(syms) => {
                        if let Some(mark) = case {
                            out.push(mark);
                        }
                        for s in syms {
                            // symbols_of only returns symbols that are in the map
                            out.push(self.id_of_symbol(&s).expect("symbol has an id"));
                        }
                    }
                    None => self.push_literal(&mut out, word),
                }
            }
        }
        out
    }

    /// Re-apply a case mark to a decoded word.
    fn apply_case(mark: Option<u16>, word: &str) -> String {
        match mark {
            Some(ID_UPPER) => word.to_uppercase(),
            Some(ID_CAPITAL) => {
                let mut c = word.chars();
                match c.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            }
            _ => word.to_string(),
        }
    }

    /// A word with no entry: escape, length, then its UTF-8 bytes.
    fn push_literal(&self, out: &mut Vec<u16>, word: &str) {
        let bytes = word.as_bytes();
        for chunk in bytes.chunks(255) {
            out.push(ID_RAW_RUN);
            out.push(chunk.len() as u16);
            for &b in chunk {
                out.push(BYTE_BASE + b as u16);
            }
        }
    }

    /// Ids back to text. Exact for anything `encode` produced.
    // The final flush! clears `case` on the way out and nothing reads it afterwards.
    #[allow(unused_assignments)]
    pub fn decode(&self, ids: &[u16]) -> String {
        let mut out = String::new();
        let mut word = String::new();
        let mut case: Option<u16> = None;
        let mut i = 0;

        // Encoded symbols accumulate into a word so the codec can read the grammar.
        macro_rules! flush {
            () => {
                if !word.is_empty() {
                    let plain = self.codec.decode_word(&word).unwrap_or_else(|| word.clone());
                    out.push_str(&Self::apply_case(case, &plain));
                    word.clear();
                }
                case = None;
            };
        }

        while i < ids.len() {
            let id = ids[i];
            match id {
                ID_SPACE => {
                    flush!();
                    out.push(' ');
                    i += 1;
                }
                ID_NEWLINE => {
                    flush!();
                    out.push('\n');
                    i += 1;
                }
                ID_RAW_RUN => {
                    flush!();
                    let len = ids.get(i + 1).copied().unwrap_or(0) as usize;
                    let mut bytes = Vec::with_capacity(len);
                    for k in 0..len {
                        match ids.get(i + 2 + k) {
                            Some(&b) if (BYTE_BASE..SYMBOL_BASE).contains(&b) => {
                                bytes.push((b - BYTE_BASE) as u8)
                            }
                            _ => break,
                        }
                    }
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    i += 2 + len;
                }
                ID_CAPITAL | ID_UPPER => {
                    flush!();
                    case = Some(id);
                    i += 1;
                }
                ID_PAD | ID_RESERVED => i += 1,
                _ => {
                    if let Some(sym) = self.symbol_of_id(id) {
                        word.push_str(sym);
                    }
                    i += 1;
                }
            }
        }
        flush!();
        out
    }

    /// Fixed-width bit packing. This is the step that stops CBMS being text.
    pub fn pack(&self, ids: &[u16]) -> Vec<u8> {
        let width = self.bits_per_id();
        let mut out = Vec::with_capacity((ids.len() * width as usize).div_ceil(8));
        let mut acc: u32 = 0;
        let mut bits = 0u32;
        for &id in ids {
            acc |= (id as u32) << bits;
            bits += width;
            while bits >= 8 {
                out.push((acc & 0xFF) as u8);
                acc >>= 8;
                bits -= 8;
            }
        }
        if bits > 0 {
            out.push((acc & 0xFF) as u8);
        }
        out
    }

    pub fn unpack(&self, bytes: &[u8], count: usize) -> Vec<u16> {
        let width = self.bits_per_id();
        let mask = ((1u32 << width) - 1) as u16;
        let mut out = Vec::with_capacity(count);
        let mut acc: u32 = 0;
        let mut bits = 0u32;
        let mut idx = 0;
        while out.len() < count {
            while bits < width {
                let byte = bytes.get(idx).copied().unwrap_or(0);
                acc |= (byte as u32) << bits;
                bits += 8;
                idx += 1;
            }
            out.push((acc as u16) & mask);
            acc >>= width;
            bits -= width;
        }
        out
    }
}

/// Split a word into a case mark and its folded form. Only the two shapes that carry
/// no information beyond case are marked; anything mixed keeps its own spelling and
/// takes the byte fallback, because a mark cannot describe it.
fn split_case(word: &str) -> (Option<u16>, String) {
    let has_upper = word.chars().any(|c| c.is_uppercase());
    if !has_upper {
        return (None, word.to_string());
    }
    let lower = word.to_lowercase();
    if word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase()) && word.chars().any(|c| c.is_alphabetic()) {
        // Round trip must hold: `to_uppercase` of the folded form has to give it back.
        if lower.to_uppercase() == word {
            return (Some(ID_UPPER), lower);
        }
    }
    let mut chars = word.chars();
    if let Some(first) = chars.next() {
        if first.is_uppercase() && chars.all(|c| !c.is_uppercase()) {
            let capitalised = {
                let mut c = lower.chars();
                match c.next() {
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                    None => String::new(),
                }
            };
            if capitalised == word {
                return (Some(ID_CAPITAL), lower);
            }
        }
    }
    (None, word.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BOOK: &str = "CODEBOOK_CBMS_ES\n\
        mi=Α\n\
        homo=Ա\n\
        libro=չ\n\
        skribi=б\n\
        legi=в\n\
        longo=ታ\n\
        komputilo=տ\n\
        \n\
        CBMS-Eo-v1.1-EXT\n\
        -o=U+25CB\n-a=U+25CF\n-e=U+25C7\n-i=U+25C6\n\
        -as=U+25B6\n-is=U+25C0\n-os=U+25B2\n-us=U+25BC\n-u=U+25A0\n\
        -n=U+25A1\n-j=U+25AA\n-jn=U+25A3\n\
        MORPH-SEP=U+00B7\n";

    fn book() -> Book {
        Book::parse(BOOK).unwrap()
    }

    #[test]
    fn known_words_cost_one_id_per_symbol() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        assert_eq!(v.encode("homo").len(), 1);
        assert_eq!(v.encode("skribas").len(), 2, "root + tense");
        assert_eq!(v.encode("longajn").len(), 3, "root + adjective + plural-accusative");
    }

    #[test]
    fn the_vocabulary_is_small_enough_to_pack_tightly() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        // Control ids, 256 byte ids, and one per symbol. The byte fallback dominates a
        // small book, which is the point: it is what makes any input encodable.
        assert!(v.len() < 1024, "vocabulary is {}", v.len());
        let width = v.bits_per_id();
        assert!(width <= 10, "{} ids needed {width} bits", v.len());
        assert!(
            (1usize << width) >= v.len() && (1usize << (width - 1)) < v.len(),
            "{width} bits is not the tightest width for {} ids",
            v.len()
        );
        // The real 482-entry book lands at 746 ids, still ten bits. Both are far under
        // the 151 936 a general-purpose tokenizer carries.
    }

    #[test]
    fn a_word_the_book_has_never_seen_still_encodes() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        let ids = v.encode("kvantumkomputilo");
        assert!(!ids.is_empty());
        assert_eq!(v.decode(&ids), "kvantumkomputilo");
    }

    #[test]
    fn any_text_at_all_round_trips_exactly() {
        // This is what makes it a tokenizer rather than a dictionary. Nothing may be
        // dropped, including scripts and punctuation the book knows nothing about.
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        for text in [
            "mi legas libron",
            "kvantumkomputilo",
            "mi legas kvantumlibron hodiaux",
            "Marcin Szul, Leeds 2026",
            "fn main() { println!(\"hi\"); }",
            "日本語 и кириллица",
            "",
            "   ",
            "line one\nline two\n\nline four",
            // Found on the real corpus: `see` loses its ending to `se`, a real root,
            // and decodes one letter short unless the encoder verifies itself.
            "see how some code is written",
            // Case must survive without the word falling to the byte fallback.
            "Homo Libro HOMO libro McDonald iPhone",
        ] {
            let ids = v.encode(text);
            assert_eq!(v.decode(&ids), text, "round trip failed for {text:?}");
        }
    }

    #[test]
    fn a_capital_costs_one_id_rather_than_the_whole_word_in_bytes() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        assert_eq!(v.encode("homo").len(), 1);
        assert_eq!(v.encode("Homo").len(), 2, "case mark plus symbol");
        assert_eq!(v.encode("HOMO").len(), 2);
        assert_eq!(v.decode(&v.encode("Homo")), "Homo");
        assert_eq!(v.decode(&v.encode("HOMO")), "HOMO");
    }

    #[test]
    fn a_word_whose_ending_lands_on_a_different_real_root_is_not_encoded() {
        // `see` -> strip -e -> `se`, which exists, and decodes back as `se`.
        // The encoder checks its own round trip and hands this to the byte fallback.
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        assert_eq!(v.decode(&v.encode("see")), "see");
    }

    #[test]
    fn packing_and_unpacking_returns_the_same_ids() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        let ids = v.encode("mi legas longajn librojn kaj kvantumkomputilon");
        let packed = v.pack(&ids);
        assert_eq!(v.unpack(&packed, ids.len()), ids);
    }

    #[test]
    fn packed_form_is_smaller_than_the_utf8_of_the_symbols() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        let source = "mi legas longajn librojn";
        let ids = v.encode(source);
        let packed = v.pack(&ids);
        // The comparison that matters is against the ORIGINAL text, not against the
        // UTF-8 of the symbols, which is a baseline nobody would ever transmit.
        assert!(
            packed.len() < source.len(),
            "packed {} bytes vs source {} bytes",
            packed.len(),
            source.len()
        );
    }

    #[test]
    fn ids_are_stable_when_the_book_only_grows() {
        let b1 = book();
        let v1 = Vocabulary::new(&b1).unwrap();
        let before = v1.encode("homo");

        let grown = BOOK.replace("mi=Α\n", "mi=Α\n") .replace("komputilo=տ\n", "komputilo=տ\nurbo=զ\n");
        let b2 = Book::parse(&grown).unwrap();
        let v2 = Vocabulary::new(&b2).unwrap();
        // `urbo` is appended after the entries that already existed, so ids assigned
        // before it keep their meaning and previously encoded data stays readable.
        assert_eq!(v2.encode("homo"), before);
    }
}
