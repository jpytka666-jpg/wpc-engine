// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:20:02
// REASON FOR CREATION: The CBMS code book existed as a hand-maintained text file with no
//   reader. Everything that wants to encode, decode, size a vocabulary or build a
//   tokenizer needs the same parsed view of it, and each of those writing its own parser
//   is how two of them end up disagreeing about what a symbol means.
// MECHANICS: Reads the `root=symbol` line format M. Szul already writes, in two sections:
//   lexical roots and the grammar/protocol extension. Accepts either a literal character
//   or `U+XXXX` notation on the right-hand side, since the existing file uses one in each
//   section. Builds the reverse map at load time and refuses a book whose symbols collide,
//   because a collision makes decoding ambiguous and there is no later point at which that
//   becomes detectable.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The single source of truth for what a symbol means. codec.rs
//   holds the grammar; this holds the vocabulary.
// DEPENDENCIES/LINKS: reads CODEBOOK_CBMS_ES format files; used by codec.rs and by the
//   book-building and encoding binaries
// TECH STACK: Rust 2021, standard library only. The eventual consumer is the tokenizer
//   inside the inference loop, which is Rust; a Python implementation would have to be
//   written twice and kept in agreement.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! The CBMS code book: Esperanto roots and grammatical marks, each mapped to one symbol.

use std::collections::HashMap;
use std::fmt;

/// Section markers as they appear in the file M. Szul maintains.
const LEXICAL_HEADER: &str = "CODEBOOK_CBMS_ES";
const EXTENSION_PREFIX: &str = "CBMS-Eo-v1.1";
/// Frozen code lengths, one per id, written by `seal`.
///
/// The table lives with the book rather than inside every message because both ends
/// already share the book. A message that carried its own table would pay twice for
/// something both sides have, and for short messages the table would dwarf the content.
const CODES_HEADER: &str = "CBMS-CODES-v1";

/// One symbol claimed by two different roots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collision {
    pub symbol: String,
    pub first: String,
    pub second: String,
}

#[derive(Debug)]
pub enum BookError {
    /// Symbols claimed twice. Decoding could not tell them apart.
    ///
    /// All of them are reported together: a book is hand-edited, and finding one
    /// collision per run means editing it once per collision.
    Collisions(Vec<Collision>),
    /// A `U+XXXX` value that is not a codepoint.
    BadCodepoint { line: usize, text: String },
    /// An entry with an empty root or an empty symbol.
    Empty { line: usize },
}

impl fmt::Display for BookError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BookError::Collisions(list) => {
                writeln!(f, "{} symbol(s) claimed twice; decoding would be ambiguous:", list.len())?;
                for c in list {
                    writeln!(
                        f,
                        "  {:?} (U+{:04X}) <- {:?} and {:?}",
                        c.symbol,
                        c.symbol.chars().next().map(|ch| ch as u32).unwrap_or(0),
                        c.first,
                        c.second
                    )?;
                }
                Ok(())
            }
            BookError::BadCodepoint { line, text } => {
                write!(f, "line {line}: {text:?} is not a valid U+XXXX codepoint")
            }
            BookError::Empty { line } => write!(f, "line {line}: empty root or symbol"),
        }
    }
}

impl std::error::Error for BookError {}

/// Which half of the book an entry belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    /// A concept: `homo`, `skribi`, `komputilo`.
    Lexical,
    /// A grammatical ending, separator or protocol marker: `-as`, `@START`, `MORPH-SEP`.
    Extension,
}

#[derive(Debug, Clone)]
pub struct Entry {
    pub root: String,
    pub symbol: String,
    pub section: Section,
}

#[derive(Debug, Default)]
pub struct Book {
    /// Frozen Huffman code lengths, one per id, or empty if the book is unsealed.
    code_lengths: Vec<u8>,
    entries: Vec<Entry>,
    by_root: HashMap<String, usize>,
    by_symbol: HashMap<String, usize>,
    /// Symbol lengths present, longest first. Decoding must try long symbols before
    /// short ones or `二十` is read as `二` followed by `十`.
    symbol_lengths: Vec<usize>,
}

impl Book {
    /// Parse the `root=symbol` line format. Blank lines and section headers are skipped;
    /// a line without `=` is ignored rather than fatal, because the file is hand-edited
    /// and a stray note in it should not stop the system loading.
    pub fn parse(text: &str) -> Result<Self, BookError> {
        let (book, collisions) = Book::parse_lenient(text)?;
        if collisions.is_empty() {
            Ok(book)
        } else {
            Err(BookError::Collisions(collisions))
        }
    }

    /// Parse and report collisions instead of refusing, so a tool can show the whole
    /// list at once. The returned book keeps the first claimant of each symbol.
    pub fn parse_lenient(text: &str) -> Result<(Self, Vec<Collision>), BookError> {
        let mut book = Book::default();
        let mut collisions = Vec::new();
        let mut section = Section::Lexical;
        let mut in_codes = false;

        for (i, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line == LEXICAL_HEADER {
                section = Section::Lexical;
                continue;
            }
            if line.starts_with(EXTENSION_PREFIX) {
                section = Section::Extension;
                continue;
            }
            if line.starts_with(CODES_HEADER) {
                in_codes = true;
                continue;
            }
            if in_codes {
                // One length per id, whitespace separated. Anything unparseable ends
                // the section rather than corrupting the table.
                let mut ok = true;
                let mut lens = Vec::new();
                for tok in line.split_whitespace() {
                    match tok.parse::<u8>() {
                        Ok(v) => lens.push(v),
                        Err(_) => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    book.code_lengths.extend(lens);
                    continue;
                }
                in_codes = false;
            }
            let Some((raw_root, value)) = split_on_separator(line) else {
                continue;
            };
            let root = unescape_root(raw_root.trim());
            let root = root.as_str();
            let value = value.trim();
            if root.is_empty() || value.is_empty() {
                return Err(BookError::Empty { line: i + 1 });
            }
            let symbol = decode_value(value).ok_or_else(|| BookError::BadCodepoint {
                line: i + 1,
                text: value.to_string(),
            })?;
            if let Some(c) = book.insert(Entry { root: root.to_string(), symbol, section }) {
                collisions.push(c);
            }
        }
        book.finish();
        Ok((book, collisions))
    }

    /// Returns the collision when the symbol is already claimed by a different root.
    fn insert(&mut self, entry: Entry) -> Option<Collision> {
        if let Some(&prev) = self.by_symbol.get(&entry.symbol) {
            let first = self.entries[prev].root.clone();
            // A root repeated with the same symbol is a harmless duplicate line.
            if first != entry.root {
                return Some(Collision { symbol: entry.symbol, first, second: entry.root });
            }
            return None;
        }
        let idx = self.entries.len();
        self.by_root.insert(entry.root.clone(), idx);
        self.by_symbol.insert(entry.symbol.clone(), idx);
        self.entries.push(entry);
        None
    }

    fn finish(&mut self) {
        let mut lens: Vec<usize> = self.entries.iter().map(|e| e.symbol.chars().count()).collect();
        lens.sort_unstable();
        lens.dedup();
        lens.reverse();
        self.symbol_lengths = lens;
    }

    pub fn symbol_for(&self, root: &str) -> Option<&str> {
        self.by_root.get(root).map(|&i| self.entries[i].symbol.as_str())
    }

    pub fn root_for(&self, symbol: &str) -> Option<&str> {
        self.by_symbol.get(symbol).map(|&i| self.entries[i].root.as_str())
    }

    /// Which half of the book a SYMBOL belongs to.
    ///
    /// Grammatical marks were once recognised by their root starting with `-`, which
    /// held until a corpus-built book gained `-` itself as a punctuation run and every
    /// hyphen in the text started being read as a verb ending. The section is recorded;
    /// asking it is not a guess.
    pub fn section_of_symbol(&self, symbol: &str) -> Option<Section> {
        self.by_symbol.get(symbol).map(|&i| self.entries[i].section)
    }

    pub fn section_of(&self, root: &str) -> Option<Section> {
        self.by_root.get(root).map(|&i| self.entries[i].section)
    }

    pub fn contains_root(&self, root: &str) -> bool {
        self.by_root.contains_key(root)
    }

    /// Symbol lengths in characters, longest first, for longest-match decoding.
    pub fn symbol_lengths(&self) -> &[usize] {
        &self.symbol_lengths
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Frozen code lengths, empty when the book has not been sealed.
    pub fn code_lengths(&self) -> &[u8] {
        &self.code_lengths
    }

    pub fn set_code_lengths(&mut self, lengths: Vec<u8>) {
        self.code_lengths = lengths;
    }

    pub fn is_sealed(&self) -> bool {
        !self.code_lengths.is_empty()
    }

    /// A fingerprint over what decoding depends on: every root, its symbol, its section,
    /// in order, plus the code table.
    ///
    /// A message names this, so a receiver holding a different book finds out at once
    /// instead of silently producing different text. Order matters because ids are
    /// assigned in it - two books with the same entries in a different order are NOT
    /// interchangeable, and the hash has to say so.
    pub fn fingerprint(&self) -> u64 {
        // FNV-1a. Not a security hash and not meant to be: this catches the wrong book,
        // not an attacker, and a dependency for that would be a poor trade.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut eat = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for e in &self.entries {
            eat(e.root.as_bytes());
            eat(b"=");
            eat(e.symbol.as_bytes());
            eat(if e.section == Section::Lexical { b"L" } else { b"X" });
        }
        eat(b"|codes|");
        eat(&self.code_lengths);
        h
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn count(&self, section: Section) -> usize {
        self.entries.iter().filter(|e| e.section == section).count()
    }

    /// Every distinct codepoint the book needs. This is the vocabulary a CBMS-native
    /// tokenizer would have to carry, and the reason the whole exercise is worth doing.
    pub fn codepoints(&self) -> Vec<char> {
        let mut set: Vec<char> = self.entries.iter().flat_map(|e| e.symbol.chars()).collect();
        set.sort_unstable();
        set.dedup();
        set
    }

    /// Render back to the file format, so a built book is editable by hand afterwards.
    pub fn to_text(&self) -> String {
        // Entries are written in the order they were read, with a section header
        // wherever the section changes. Grouping them instead would reorder the book,
        // and ids are assigned in order - so a saved book would load as a DIFFERENT
        // book with the same entries. The fingerprint check at seal time caught exactly
        // that, which is what it is for.
        let mut out = String::new();
        let mut current: Option<Section> = None;
        for e in &self.entries {
            if current != Some(e.section) {
                if current.is_some() {
                    out.push('\n');
                }
                out.push_str(match e.section {
                    Section::Lexical => LEXICAL_HEADER,
                    Section::Extension => "CBMS-Eo-v1.1-EXT",
                });
                out.push('\n');
                current = Some(e.section);
            }
            out.push_str(&escape_root(&e.root));
            out.push('=');
            if e.section == Section::Extension {
                // Conventionally written as codepoints: several of these marks are
                // invisible or easy to mistake for one another.
                for ch in e.symbol.chars() {
                    out.push_str(&format!("U+{:04X}", ch as u32));
                }
            } else {
                out.push_str(&e.symbol);
            }
            out.push('\n');
        }
        if !self.code_lengths.is_empty() {
            out.push_str("\n\n");
            out.push_str(CODES_HEADER);
            out.push('\n');
            // Wrapped so the file stays readable and diffable rather than one huge line.
            for row in self.code_lengths.chunks(32) {
                let line: Vec<String> = row.iter().map(|v| v.to_string()).collect();
                out.push_str(&line.join(" "));
                out.push('\n');
            }
        }
        out
    }
}

/// Split a line at its separator, honouring escapes.
///
/// A corpus-built book holds punctuation runs as roots, and one of them can be `=` or
/// `==` itself. Splitting on the first `=` turns that into an empty root and the book
/// refuses to load - which is how this was found.
fn split_on_separator(line: &str) -> Option<(&str, &str)> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2, // whatever follows is literal, separator included
            b'=' => return Some((&line[..i], &line[i + 1..])),
            _ => i += 1,
        }
    }
    None
}

/// Undo `escape_root`.
fn unescape_root(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Escape a root so the line format survives it. Only the separator and the escape
/// character itself need it; everything else is written as it is, so a book stays
/// readable by eye.
pub(crate) fn escape_root(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c == '\\' || c == '=' {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// `U+25CB` or a literal character. The existing book uses the first form in the
/// extension section and the second in the lexical section.
fn decode_value(value: &str) -> Option<String> {
    if let Some(hex) = value.strip_prefix("U+") {
        // Several marks concatenated, e.g. "U+25CBU+25A1", are accepted.
        let mut out = String::new();
        for part in hex.split("U+") {
            let cp = u32::from_str_radix(part, 16).ok()?;
            out.push(char::from_u32(cp)?);
        }
        Some(out)
    } else {
        Some(value.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "CODEBOOK_CBMS_ES\n\
        \n\
        homo=Ա\n\
        skribi=б\n\
        dudek=二十\n\
        \n\
        CBMS-Eo-v1.1-EXT\n\
        -as=U+25B6\n\
        MORPH-SEP=U+00B7\n";

    #[test]
    fn reads_both_sections_and_both_value_notations() {
        let book = Book::parse(SAMPLE).expect("sample book parses");
        assert_eq!(book.count(Section::Lexical), 3);
        assert_eq!(book.count(Section::Extension), 2);
        assert_eq!(book.symbol_for("homo"), Some("Ա"));
        assert_eq!(book.symbol_for("-as"), Some("\u{25B6}"));
        assert_eq!(book.symbol_for("MORPH-SEP"), Some("\u{00B7}"));
    }

    #[test]
    fn maps_back_from_symbol_to_root() {
        let book = Book::parse(SAMPLE).unwrap();
        assert_eq!(book.root_for("Ա"), Some("homo"));
        assert_eq!(book.root_for("二十"), Some("dudek"));
    }

    #[test]
    fn longest_symbol_first_so_multi_char_symbols_survive_decoding() {
        let book = Book::parse(SAMPLE).unwrap();
        // 二十 is two characters; without trying length 2 before length 1 a decoder
        // would read it as two separate symbols and lose the word.
        assert_eq!(book.symbol_lengths().first(), Some(&2));
    }

    #[test]
    fn a_symbol_claimed_twice_is_refused_rather_than_silently_shadowed() {
        let clashing = "CODEBOOK_CBMS_ES\nhomo=Ա\nviro=Ա\n";
        let err = Book::parse(clashing).expect_err("collision must not parse");
        assert!(matches!(err, BookError::Collisions(_)), "got {err:?}");
    }

    #[test]
    fn lenient_parsing_reports_every_collision_at_once() {
        // A hand-edited book with several clashes should be fixable in one pass,
        // not one run per clash.
        let many = "CODEBOOK_CBMS_ES\nhomo=Ա\nviro=Ա\nlibro=չ\ngazeto=չ\n";
        let (book, clashes) = Book::parse_lenient(many).expect("lenient parse succeeds");
        assert_eq!(clashes.len(), 2);
        assert_eq!(book.root_for("Ա"), Some("homo"), "first claimant is kept");
    }

    #[test]
    fn a_repeated_identical_line_is_not_a_collision() {
        let dup = "CODEBOOK_CBMS_ES\nhomo=Ա\nhomo=Ա\n";
        let book = Book::parse(dup).expect("identical duplicate is harmless");
        assert_eq!(book.len(), 1);
    }

    #[test]
    fn a_root_that_is_the_separator_itself_survives_the_file_format() {
        // Found by building a book from a corpus: punctuation runs become roots, and one
        // of them is `=`. Splitting on the first `=` gave an empty root and the whole
        // book refused to load.
        let mut b = Book::parse("CODEBOOK_CBMS_ES\nhomo=Ա\n").unwrap();
        b.insert(Entry { root: "=".into(), symbol: "ᐁ".into(), section: Section::Lexical });
        b.insert(Entry { root: "==".into(), symbol: "ᐂ".into(), section: Section::Lexical });
        b.insert(Entry { root: "a\\b".into(), symbol: "ᐃ".into(), section: Section::Lexical });
        b.finish();

        let again = Book::parse(&b.to_text()).expect("escaped book re-parses");
        assert_eq!(again.symbol_for("="), Some("ᐁ"));
        assert_eq!(again.symbol_for("=="), Some("ᐂ"));
        assert_eq!(again.symbol_for("a\\b"), Some("ᐃ"));
        assert_eq!(again.symbol_for("homo"), Some("Ա"));
    }

    #[test]
    fn a_saved_book_loads_back_as_the_same_book_not_a_reordered_one() {
        // Ids are assigned in entry order, so the file must preserve it. Writing all
        // lexical entries then all extension ones reordered a book that interleaved
        // them, and it loaded as a different book with the same contents - which the
        // fingerprint check found at seal time.
        let interleaved = "CODEBOOK_CBMS_ES\nhomo=Ա\n\
            CBMS-Eo-v1.1-EXT\n-as=U+25B6\nMORPH-SEP=U+00B7\n\
            CODEBOOK_CBMS_ES\nlibro=չ\n";
        let book = Book::parse(interleaved).unwrap();
        let again = Book::parse(&book.to_text()).expect("saved book reloads");
        assert_eq!(again.fingerprint(), book.fingerprint(), "order must survive the file");
        let roots: Vec<&str> = again.entries().iter().map(|e| e.root.as_str()).collect();
        assert_eq!(roots, vec!["homo", "-as", "MORPH-SEP", "libro"]);
    }

    #[test]
    fn a_sealed_book_carries_its_code_table_through_the_file() {
        let mut book = Book::parse(SAMPLE).unwrap();
        assert!(!book.is_sealed());
        book.set_code_lengths(vec![0, 1, 2, 3, 4, 5]);
        let again = Book::parse(&book.to_text()).expect("sealed book reloads");
        assert!(again.is_sealed());
        assert_eq!(again.code_lengths(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(again.fingerprint(), book.fingerprint());
    }

    #[test]
    fn the_fingerprint_notices_a_changed_book() {
        let a = Book::parse(SAMPLE).unwrap();
        let b = Book::parse(&SAMPLE.replace("homo=Ա\n", "homo=Ա\nurbo=զ\n")).unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
    }

    #[test]
    fn round_trips_through_the_file_format() {
        let book = Book::parse(SAMPLE).unwrap();
        let again = Book::parse(&book.to_text()).expect("rendered book re-parses");
        assert_eq!(again.len(), book.len());
        assert_eq!(again.symbol_for("dudek"), Some("二十"));
        assert_eq!(again.symbol_for("MORPH-SEP"), Some("\u{00B7}"));
    }

    #[test]
    fn counts_the_codepoints_a_tokenizer_would_need() {
        let book = Book::parse(SAMPLE).unwrap();
        // Ա б 二 十 ▶ ·  -> six distinct codepoints across five entries
        assert_eq!(book.codepoints().len(), 6);
    }
}
