// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 13:16:30
// REASON FOR CREATION: Everything under this point produces a bare stream of bits, which
//   is fine while one process both writes and reads it and useless the moment AIONS wants
//   to store a block on disk or send one to another agent. A receiver holding a different
//   code book would decode the same bits into different text and never know. This is the
//   envelope that makes a CBMS payload a thing that can be written, read back and sent.
// MECHANICS: A short header - magic, format version, flags, book fingerprint, id count -
//   in front of the coded payload. The fingerprint is checked before decoding, so the
//   wrong book is an error rather than quiet corruption. The code table is NOT carried
//   here: it lives in the book, because both ends already share the book and a table
//   inside every message would dwarf a short one.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The write, read and transfer format for AIONS memory. Below it
//   is coding; above it is storage and messaging, which need to know what they are
//   holding and whether they can read it.
// DEPENDENCIES/LINKS: book::Book, vocab::Vocabulary, huffman::Code
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

//! The on-disk and on-wire form of a CBMS payload.

use crate::book::Book;
use crate::huffman::Code;
use crate::vocab::Vocabulary;
use std::fmt;

pub const MAGIC: [u8; 4] = *b"CBMS";
pub const FORMAT_VERSION: u8 = 1;
/// magic + version + flags + fingerprint + id count
pub const HEADER_LEN: usize = 4 + 1 + 1 + 8 + 4;

/// Payload is Huffman coded; without it, fixed-width.
pub const FLAG_HUFFMAN: u8 = 0b0000_0001;

#[derive(Debug)]
pub enum ContainerError {
    /// Not a CBMS payload at all.
    NotCbms,
    /// Written by a newer format than this build understands.
    Version { found: u8, supported: u8 },
    /// The right format, the wrong book. Decoding would produce different text.
    WrongBook { expected: u64, found: u64 },
    /// Header claims more than the file holds.
    Truncated { need: usize, have: usize },
    /// The book has no frozen code table but the payload says it is coded.
    UnsealedBook,
}

impl fmt::Display for ContainerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainerError::NotCbms => write!(f, "not a CBMS payload (bad magic)"),
            ContainerError::Version { found, supported } => {
                write!(f, "format version {found}, this build reads up to {supported}")
            }
            ContainerError::WrongBook { expected, found } => write!(
                f,
                "written with book {expected:016x}, reader holds {found:016x} - \
                 decoding would produce different text, so it is refused"
            ),
            ContainerError::Truncated { need, have } => {
                write!(f, "truncated: header needs {need} bytes, {have} present")
            }
            ContainerError::UnsealedBook => write!(
                f,
                "payload is frequency-coded but the book carries no code table; \
                 seal the book first"
            ),
        }
    }
}

impl std::error::Error for ContainerError {}

/// What a header says, without decoding the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub version: u8,
    pub flags: u8,
    pub book: u64,
    pub ids: u32,
}

impl Header {
    pub fn huffman(&self) -> bool {
        self.flags & FLAG_HUFFMAN != 0
    }
}

/// Read the header alone. Useful for deciding whether a stored block is readable at all
/// before spending anything on it.
pub fn peek(bytes: &[u8]) -> Result<Header, ContainerError> {
    if bytes.len() < HEADER_LEN {
        return Err(ContainerError::Truncated { need: HEADER_LEN, have: bytes.len() });
    }
    if bytes[..4] != MAGIC {
        return Err(ContainerError::NotCbms);
    }
    let version = bytes[4];
    if version > FORMAT_VERSION {
        return Err(ContainerError::Version { found: version, supported: FORMAT_VERSION });
    }
    let book = u64::from_le_bytes(bytes[6..14].try_into().expect("8 bytes"));
    let ids = u32::from_le_bytes(bytes[14..18].try_into().expect("4 bytes"));
    Ok(Header { version, flags: bytes[5], book, ids })
}

/// Text to a self-describing payload.
pub fn write(book: &Book, vocab: &Vocabulary, text: &str) -> Result<Vec<u8>, ContainerError> {
    let ids = vocab.encode(text);
    let sealed = book.is_sealed();
    let payload = if sealed {
        Code::from_lengths(book.code_lengths().to_vec()).encode(&ids)
    } else {
        vocab.pack(&ids)
    };

    let mut out = Vec::with_capacity(HEADER_LEN + payload.len());
    out.extend_from_slice(&MAGIC);
    out.push(FORMAT_VERSION);
    out.push(if sealed { FLAG_HUFFMAN } else { 0 });
    out.extend_from_slice(&book.fingerprint().to_le_bytes());
    out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Payload back to text, refusing anything this book cannot faithfully read.
pub fn read(book: &Book, vocab: &Vocabulary, bytes: &[u8]) -> Result<String, ContainerError> {
    let header = peek(bytes)?;
    let mine = book.fingerprint();
    if header.book != mine {
        return Err(ContainerError::WrongBook { expected: header.book, found: mine });
    }
    let payload = &bytes[HEADER_LEN..];
    let ids = if header.huffman() {
        if !book.is_sealed() {
            return Err(ContainerError::UnsealedBook);
        }
        Code::from_lengths(book.code_lengths().to_vec()).decode(payload, header.ids as usize)
    } else {
        vocab.unpack(payload, header.ids as usize)
    };
    Ok(vocab.decode(&ids))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::Code;
    use std::collections::HashMap;

    const BOOK: &str = "CODEBOOK_CBMS_ES\n\
        mi=Α\nhomo=Ա\nlibro=չ\nskribi=б\nlegi=в\nlongo=ታ\nkomputilo=տ\n\
        \n\
        CBMS-Eo-v1.1-EXT\n\
        -o=U+25CB\n-a=U+25CF\n-e=U+25C7\n-i=U+25C6\n\
        -as=U+25B6\n-is=U+25C0\n-os=U+25B2\n-us=U+25BC\n-u=U+25A0\n\
        -n=U+25A1\n-j=U+25AA\n-jn=U+25A3\n\
        MORPH-SEP=U+00B7\n";

    const SAMPLE: &str = "mi legas longajn librojn kaj kvantumkomputilon";

    fn sealed_book() -> Book {
        let mut book = Book::parse(BOOK).unwrap();
        let lengths = {
            let vocab = Vocabulary::new(&book).unwrap();
            let ids = vocab.encode(SAMPLE);
            let mut freq: HashMap<u16, usize> = HashMap::new();
            for &id in &ids {
                *freq.entry(id).or_default() += 1;
            }
            Code::from_frequencies(&freq, vocab.len()).lengths().to_vec()
        };
        book.set_code_lengths(lengths);
        book
    }

    #[test]
    fn a_payload_round_trips_through_the_container() {
        let book = sealed_book();
        let vocab = Vocabulary::new(&book).unwrap();
        let bytes = write(&book, &vocab, SAMPLE).unwrap();
        assert_eq!(read(&book, &vocab, &bytes).unwrap(), SAMPLE);
    }

    #[test]
    fn an_unsealed_book_still_writes_and_reads_at_fixed_width() {
        // Sealing is an optimisation, not a precondition. A book that has never met a
        // corpus must still be usable.
        let book = Book::parse(BOOK).unwrap();
        let vocab = Vocabulary::new(&book).unwrap();
        let bytes = write(&book, &vocab, SAMPLE).unwrap();
        assert_eq!(peek(&bytes).unwrap().huffman(), false);
        assert_eq!(read(&book, &vocab, &bytes).unwrap(), SAMPLE);
    }

    #[test]
    fn the_header_says_what_it_holds_without_decoding_anything() {
        let book = sealed_book();
        let vocab = Vocabulary::new(&book).unwrap();
        let bytes = write(&book, &vocab, SAMPLE).unwrap();
        let h = peek(&bytes).unwrap();
        assert_eq!(h.version, FORMAT_VERSION);
        assert!(h.huffman());
        assert_eq!(h.book, book.fingerprint());
        assert_eq!(h.ids as usize, vocab.encode(SAMPLE).len());
    }

    #[test]
    fn a_different_book_is_refused_rather_than_decoded_into_different_text() {
        // The failure this exists to prevent: the same bits mean different things under
        // a different book, and without the check nobody would ever find out.
        let writer = sealed_book();
        let vocab_w = Vocabulary::new(&writer).unwrap();
        let bytes = write(&writer, &vocab_w, SAMPLE).unwrap();

        let other = Book::parse(&BOOK.replace("homo=Ա\n", "homo=Ա\nurbo=զ\n")).unwrap();
        let vocab_o = Vocabulary::new(&other).unwrap();
        let err = read(&other, &vocab_o, &bytes).expect_err("wrong book must be refused");
        assert!(matches!(err, ContainerError::WrongBook { .. }), "got {err}");
    }

    #[test]
    fn something_that_is_not_a_payload_is_rejected_on_sight() {
        let book = sealed_book();
        let vocab = Vocabulary::new(&book).unwrap();
        let err = read(&book, &vocab, b"just some text, honestly").unwrap_err();
        assert!(matches!(err, ContainerError::NotCbms), "got {err}");
    }

    #[test]
    fn a_truncated_payload_says_so_instead_of_panicking() {
        let err = peek(b"CBM").unwrap_err();
        assert!(matches!(err, ContainerError::Truncated { .. }), "got {err}");
    }

    #[test]
    fn a_newer_format_is_reported_not_guessed_at() {
        let mut bytes = vec![0u8; HEADER_LEN];
        bytes[..4].copy_from_slice(&MAGIC);
        bytes[4] = FORMAT_VERSION + 7;
        let err = peek(&bytes).unwrap_err();
        assert!(matches!(err, ContainerError::Version { .. }), "got {err}");
    }

    #[test]
    fn the_header_costs_eighteen_bytes() {
        // Small enough that per-block storage is not dominated by it. Stated as a test
        // so a future field addition is a deliberate decision rather than a surprise.
        assert_eq!(HEADER_LEN, 18);
        let book = sealed_book();
        let vocab = Vocabulary::new(&book).unwrap();
        let bytes = write(&book, &vocab, SAMPLE).unwrap();
        assert!(bytes.len() < SAMPLE.len(), "{} vs {}", bytes.len(), SAMPLE.len());
    }
}
