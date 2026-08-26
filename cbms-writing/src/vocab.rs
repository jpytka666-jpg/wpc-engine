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
use crate::build::split_affixes;
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
/// The same two, with a leading space folded in. A capitalised word usually starts a
/// sentence, so without these the commonest word in the language to follow a space would
/// be the one that still had to pay an id for it.
pub const ID_CAPITAL_SPACED: u16 = 7;
pub const ID_UPPER_SPACED: u16 = 8;
const CONTROL_COUNT: u16 = 16;
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
        //
        // The pair is INTERLEAVED - a symbol's plain id and its spaced twin sit next to
        // each other - and that is what makes the promise above true for both halves.
        // Blocking them instead (all plain, then all spaced) makes every spaced id depend
        // on the total symbol count, so appending one entry silently shifts the whole
        // second half. Measured before this change: appending a single word moved
        // `urbo` in "homo urbo" from 5276 to 5277 while `homo` stayed at 468. That is
        // what forced a full retrain instead of carrying the previous one forward, and
        // it would have quietly invalidated every stored block ever written.
        //
        // Nothing about the space folding itself changes here - the space still rides on
        // the word's first symbol and still costs no id of its own. Only the arithmetic
        // that names that id is different.
        for entry in book.entries() {
            if symbol_to_id.contains_key(&entry.symbol) {
                continue;
            }
            // Refuse rather than wrap. A book past this size would alias new symbols onto
            // ids that already mean something, which is the one failure that cannot be
            // detected downstream - the data would decode, into the wrong words.
            let index = id_to_symbol.len();
            let id = SYMBOL_BASE.checked_add(2u16.checked_mul(u16::try_from(index).ok()?)?)?;
            symbol_to_id.insert(entry.symbol.clone(), id);
            id_to_symbol.push(entry.symbol.clone());
        }
        Some(Vocabulary { codec, book, symbol_to_id, id_to_symbol })
    }

    /// Total ids: control, bytes, and TWO per distinct symbol - the symbol itself and
    /// the same symbol with a space in front. Doubling the symbol range is what buys the
    /// space ids back; it is exactly what BPE does with `Ġword` against `word`, and the
    /// result is still tiny beside a general tokenizer's vocabulary.
    pub fn len(&self) -> usize {
        SYMBOL_BASE as usize + 2 * self.id_to_symbol.len()
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

    /// The symbol an id names, and whether that id also carries a leading space.
    fn symbol_of_id(&self, id: u16) -> Option<(&str, bool)> {
        if id < SYMBOL_BASE {
            return None;
        }
        // Interleaved: even offset is the plain symbol, odd is the same symbol with a
        // leading space. Reading the twin off the low bit costs nothing and, unlike the
        // old block layout, does not consult the book's size - so an id decoded today
        // means the same word it meant before the book grew.
        let offset = (id - SYMBOL_BASE) as usize;
        let spaced = offset % 2 == 1;
        self.id_to_symbol.get(offset / 2).map(|s| (s.as_str(), spaced))
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
                // A single space before a word is folded INTO that word's first symbol
                // rather than spent as an id of its own. Measured on prose, separate
                // space ids were 27.7% of the whole stream - one id per space, where BPE
                // pays nothing because its `Ġword` and `word` are simply different
                // tokens. Same trick here: the symbol range is doubled, and the second
                // half means "with a space in front".
                let leading_space = !first_word;
                first_word = false;
                if word.is_empty() {
                    // Two spaces in a row: nothing to attach the second one to.
                    if leading_space {
                        out.push(ID_SPACE);
                    }
                    continue;
                }
                let mut pending = leading_space;
                // Punctuation is split off before lookup. The book is built from words
                // with punctuation trimmed, so leaving it attached here would make every
                // entry unreachable for any word that happens to end a sentence - which
                // is what kept 59.5% of ids as literal bytes at 100% reported coverage.
                let (lead, core, tail) = split_affixes(word);
                self.push_literal(&mut out, lead, &mut pending);
                self.push_word(&mut out, core, &mut pending);
                self.push_literal(&mut out, tail, &mut pending);
                // Nothing in the token could carry it - a token of pure punctuation
                // whose bytes all went out unspaced cannot happen, but be exact.
                if pending {
                    out.push(ID_SPACE);
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

    /// The id meaning "this symbol, with one space in front of it".
    ///
    /// The twin is the next id up, never an offset by the book's size - see the note in
    /// `new`. This is what lets a book grow without renumbering anything already written.
    fn spaced(&self, id: u16) -> u16 {
        id + 1
    }

    /// One word through the code book, or spelled out if the book cannot hold it.
    ///
    /// `pending` carries a leading space that has not been spent yet: if the word
    /// encodes, the space rides on its first symbol and costs nothing.
    fn push_word(&self, out: &mut Vec<u16>, word: &str, pending: &mut bool) {
        if word.is_empty() {
            return;
        }
        // Case is recorded as a mark and stripped before lookup, so `Homo` and `homo`
        // reach the same entry without either losing its spelling. A word whose case no
        // mark can describe must not reach the codec at all, because canonicalisation
        // lowercases and nothing would put the capitals back.
        // A verbatim entry needs no case handling at all, and is the only way a mixed
        // shape like `AarSvc_6e9d9` can be encoded: no mark describes it, so the book
        // must hold it exactly as written or it goes out byte by byte.
        // Take the space if there is one; the first id pushed below carries it.
        let take = |out: &mut Vec<u16>, id: u16, pending: &mut bool, first: &mut bool| {
            let id = if *first && *pending {
                *pending = false;
                self.spaced(id)
            } else {
                id
            };
            *first = false;
            out.push(id);
        };

        if let Some(sym) = self.book.symbol_for(word) {
            if let Some(id) = self.id_of_symbol(sym) {
                let mut first = true;
                take(out, id, pending, &mut first);
                return;
            }
        }
        if let Some((case, folded)) = split_case(word) {
            if let Some(syms) = self.codec.encode_word(&folded).and_then(|e| self.symbols_of(&e)) {
                let mut first = true;
                if let Some(mark) = case {
                    // Control ids have their own spaced twins; the generic offset is
                    // only valid inside the symbol range.
                    out.push(if *pending {
                        *pending = false;
                        first = false;
                        if mark == ID_UPPER { ID_UPPER_SPACED } else { ID_CAPITAL_SPACED }
                    } else {
                        first = false;
                        mark
                    });
                }
                for s in syms {
                    let id = self.id_of_symbol(&s).expect("symbols_of only returns known symbols");
                    take(out, id, pending, &mut first);
                }
                return;
            }
        }
        self.push_literal(out, word, pending);
    }

    /// Spelled out, one id per byte.
    ///
    /// No framing: byte ids occupy their own range, so a decoder recognises them by
    /// value. The run marker this used to write cost two ids per literal word to say
    /// something the id already said.
    fn push_literal(&self, out: &mut Vec<u16>, text: &str, pending: &mut bool) {
        if text.is_empty() {
            return;
        }
        let mut rest = text;
        let mut first = true;
        while !rest.is_empty() {
            // Longest listed run first. A book that holds `", "` spends one id where
            // spelling it out spends two, which is the whole of BPE's advantage on
            // punctuation and costs nothing to take.
            let mut matched = 0usize;
            let mut id = None;
            for end in (1..=rest.chars().count().min(8)).rev() {
                let cut: usize = rest.char_indices().nth(end).map_or(rest.len(), |(i, _)| i);
                if let Some(sym) = self.book.symbol_for(&rest[..cut]) {
                    // Lexical entries only. The grammar section holds `-u`, `-a` and
                    // friends, and matching those against literal text turned
                    // `Warm-up` into `Warmup`: the hyphen and `u` were read as an
                    // imperative ending and folded into the word before it.
                    if self.book.section_of_symbol(sym) == Some(crate::book::Section::Extension) {
                        continue;
                    }
                    if let Some(found) = self.id_of_symbol(sym) {
                        matched = cut;
                        id = Some(found);
                        break;
                    }
                }
            }
            match id {
                Some(id) => {
                    let id = if first && *pending {
                        *pending = false;
                        self.spaced(id)
                    } else {
                        id
                    };
                    out.push(id);
                    rest = &rest[matched..];
                }
                None => {
                    // Byte ids are a fixed 256 with no spaced twin, so an unspent space
                    // here costs an id of its own.
                    if first && *pending {
                        out.push(ID_SPACE);
                        *pending = false;
                    }
                    let ch = rest.chars().next().expect("rest is not empty");
                    let cut = ch.len_utf8();
                    for &b in rest[..cut].as_bytes() {
                        out.push(BYTE_BASE + b as u16);
                    }
                    rest = &rest[cut..];
                }
            }
            first = false;
        }

    }

    /// Ids back to text. Exact for anything `encode` produced.
    // The final flush! clears `case` on the way out and nothing reads it afterwards.
    #[allow(unused_assignments)]
    pub fn decode(&self, ids: &[u16]) -> String {
        let mut out = String::new();
        let mut word = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let mut case: Option<u16> = None;

        // Symbols accumulate into a word so the codec can read the grammar off it;
        // byte ids accumulate separately so a multi-byte character is not split.
        macro_rules! flush_word {
            () => {
                if !word.is_empty() {
                    let plain = self.codec.decode_word(&word).unwrap_or_else(|| word.clone());
                    out.push_str(&Self::apply_case(case, &plain));
                    word.clear();
                    case = None;
                }
            };
        }
        macro_rules! flush_bytes {
            () => {
                if !bytes.is_empty() {
                    out.push_str(&String::from_utf8_lossy(&bytes));
                    bytes.clear();
                }
            };
        }

        for &id in ids {
            match id {
                ID_SPACE | ID_NEWLINE => {
                    flush_word!();
                    flush_bytes!();
                    out.push(if id == ID_SPACE { ' ' } else { '\n' });
                }
                ID_CAPITAL | ID_UPPER => {
                    flush_word!();
                    flush_bytes!();
                    case = Some(id);
                }
                ID_CAPITAL_SPACED | ID_UPPER_SPACED => {
                    flush_word!();
                    flush_bytes!();
                    out.push(' ');
                    case = Some(if id == ID_UPPER_SPACED { ID_UPPER } else { ID_CAPITAL });
                }
                ID_PAD | ID_RESERVED | ID_RAW_RUN => {}
                _ if (BYTE_BASE..SYMBOL_BASE).contains(&id) => {
                    flush_word!();
                    bytes.push((id - BYTE_BASE) as u8);
                }
                _ => {
                    if let Some((sym, spaced)) = self.symbol_of_id(id) {
                        // A spaced id both ends the previous word and opens a new one,
                        // so the boundary needs no id of its own either.
                        if spaced {
                            flush_word!();
                            flush_bytes!();
                            out.push(' ');
                        } else {
                            flush_bytes!();
                            // Only a grammatical mark continues the word in progress.
                            // Any other symbol starts its own - without this, `homo,`
                            // encoded as two symbols would be read as one word and the
                            // comma would be dropped on the floor.
                            let is_mark = self.book.section_of_symbol(sym)
                                == Some(crate::book::Section::Extension);
                            if !is_mark {
                                flush_word!();
                            }
                        }
                        word.push_str(sym);
                    } else {
                        flush_bytes!();
                    }
                }
            }
        }
        flush_word!();
        flush_bytes!();
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

/// Split a word into a case mark and its folded form.
///
/// Returns `None` when no mark can describe the word's case - mixed shapes like
/// `C:\Users\User\CLAUDE.md` or `iPhone`. Those must take the byte fallback: the codec
/// lowercases before lookup, and with nothing recording where the capitals were, the
/// text would come back changed. Losing a byte saving is cheap; losing the text is not.
pub(crate) fn split_case(word: &str) -> Option<(Option<u16>, String)> {
    if !word.chars().any(|c| c.is_uppercase()) {
        return Some((None, word.to_string()));
    }
    let lower = word.to_lowercase();

    // Every letter capital, and folding back up returns exactly what came in.
    if word.chars().any(|c| c.is_alphabetic())
        && word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase())
        && lower.to_uppercase() == word
    {
        return Some((Some(ID_UPPER), lower));
    }

    // First letter capital, nothing else, and capitalising the folded form returns it.
    let mut chars = word.chars();
    if let Some(first) = chars.next() {
        if first.is_uppercase() && chars.all(|c| !c.is_uppercase()) {
            let mut c = lower.chars();
            let capitalised = match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            };
            if capitalised == word {
                return Some((Some(ID_CAPITAL), lower));
            }
        }
    }
    None
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
    fn a_space_before_a_word_costs_nothing() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        assert_eq!(v.encode("homo").len(), 1);
        assert_eq!(v.encode("homo homo").len(), 2, "the space rides on the second word");
        assert_eq!(v.encode("homo homo homo").len(), 3);
        assert_eq!(v.decode(&v.encode("homo homo homo")), "homo homo homo");
    }

    #[test]
    fn folding_the_space_works_on_any_script_because_it_works_on_ids() {
        // `Ġword` against `word` is a BPE trick tied to its own vocabulary. Doing it at
        // the id layer instead means the symbol underneath can be Latin, Cyrillic,
        // Armenian or a CJK ideograph minted from a corpus - the spaced twin is the same
        // arithmetic either way.
        let grown = BOOK.replace(
            "homo=Ա\n",
            "homo=Ա\nsłowo=一\nword=ᐁ\n한국=ᄀ\n",
        );
        let b = Book::parse(&grown).unwrap();
        let v = Vocabulary::new(&b).unwrap();
        for text in ["słowo słowo", "word word", "한국 한국", "słowo word 한국 homo"] {
            let ids = v.encode(text);
            assert_eq!(v.decode(&ids), text, "{text} must survive");
            assert_eq!(
                ids.len(),
                text.split(' ').count(),
                "{text} -> {ids:?}: one id per word, spaces folded in"
            );
        }
    }

    #[test]
    fn a_capitalised_word_after_a_space_still_pays_nothing_for_it() {
        // A capital usually starts a sentence, so this is the commonest case there is.
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        // A capitalised word is two ids of its own - the mark and the symbol. The space
        // rides on the mark, so the pair costs two rather than three.
        assert_eq!(v.encode("Homo").len(), 2);
        assert_eq!(v.encode("homo Homo").len(), 3, "1 + 2, with no id spent on the space");
        assert_eq!(v.decode(&v.encode("homo Homo HOMO")), "homo Homo HOMO");
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
    fn mixed_case_takes_the_byte_fallback_rather_than_losing_its_capitals() {
        // Found on the real corpus with a corpus-built book: a whole file path had been
        // minted as one entry, matched, and came back lowercased. No mark describes a
        // mixed shape, so it must not reach canonicalisation at all.
        let pathish = ["c:", "users", "user", "claude.md"].join("\\");
        let mixed = ["C:", "Users", "User", "CLAUDE.md"].join("\\");
        let b = Book::parse(&BOOK.replace("homo=Ա\n", &format!("homo=Ա\n{pathish}=Ք\n"))).unwrap();
        let v = Vocabulary::new(&b).unwrap();
        assert_eq!(v.decode(&v.encode(&pathish)), pathish, "lowercase form still encodes");
        for word in [mixed.as_str(), "iPhone", "McDonald", "aBc"] {
            assert_eq!(v.decode(&v.encode(word)), word, "{word} must survive unchanged");
        }
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
        // A SECOND word matters here. The first word of a line carries no leading space,
        // so testing one word alone only ever exercised the half that was already stable
        // - which is how the block layout survived this test while every spaced id moved
        // whenever the book grew.
        let before_plain = v1.encode("homo");
        let before_spaced = v1.encode("homo komputilo");

        let grown = BOOK.replace("komputilo=տ\n", "komputilo=տ\nurbo=զ\n");
        let b2 = Book::parse(&grown).unwrap();
        let v2 = Vocabulary::new(&b2).unwrap();

        // `urbo` is appended after the entries that already existed, so ids assigned
        // before it keep their meaning and previously encoded data stays readable.
        assert_eq!(v2.encode("homo"), before_plain, "plain id moved");
        assert_eq!(
            v2.encode("homo komputilo"),
            before_spaced,
            "spaced id moved - anything written before the book grew now decodes wrong"
        );

        // And the growth is genuinely visible, so the test cannot pass by comparing two
        // identical books.
        assert!(v2.symbol_count() > v1.symbol_count());
    }

    #[test]
    fn a_spaced_id_decodes_to_the_word_before_it() {
        let b = book();
        let v = Vocabulary::new(&b).unwrap();
        // The space costs no id of its own: two words, and the second one's leading
        // space rides on its first symbol. This is the measured win the interleaving had
        // to preserve - prose went from 1.45x to 1.16x of BPE on exactly this.
        let two = v.encode("homo komputilo");
        let one = v.encode("homo");
        assert_eq!(two.len(), one.len() + v.encode("komputilo").len());
        assert_eq!(v.decode(&two), "homo komputilo");
    }
}
