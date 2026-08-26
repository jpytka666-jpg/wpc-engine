// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 15:08:08
// REASON FOR CREATION: The container format works on one block. AIONS holds hundreds, and
//   the question that decides whether CBMS becomes its storage layer is whether EVERY one
//   of them survives the round trip - not a sample, not the easy ones.
// MECHANICS: Walks the live memory directory read-only, writes each chunk's content as a
//   `.cbms` payload into a SEPARATE directory, then reads every payload back and compares
//   it to the live chunk byte for byte. Nothing in the live store is opened for writing.
//   A parallel store can be measured, diffed and thrown away; a mutated one cannot.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The step between "the format works" and "AIONS uses the format".
//   It produces the evidence needed to decide the second, without performing it.
// DEPENDENCIES/LINKS: reads aions_core/memory/chunks/*.json; writes a parallel directory
//   of .cbms payloads; cbms_writing::{Book, Vocabulary, container}
// TECH STACK: Rust 2021, standard library only. JSON is read by a minimal scanner for the
//   one field that matters rather than by pulling in a parser - the store's shape is
//   fixed and known, and a dependency here would be carried into the runtime later.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

use cbms_writing::{container, Book, Vocabulary};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

/// Pull one string field out of a chunk. The store writes `"content": "..."` with the
/// usual JSON escapes; nothing here needs a general parser.
fn json_string_field(text: &str, field: &str) -> Option<String> {
    let key = format!("\"{field}\"");
    let at = text.find(&key)?;
    let rest = &text[at + key.len()..];
    let start = rest.find('"')? + 1;
    let bytes = rest.as_bytes();
    let mut out = String::new();
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => return Some(out),
            b'\\' => {
                i += 1;
                match *bytes.get(i)? {
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let hex = rest.get(i + 1..i + 5)?;
                        let cp = u32::from_str_radix(hex, 16).ok()?;
                        // Surrogate pairs: the store writes them for anything above the
                        // basic plane, and dropping one silently corrupts the text.
                        if (0xD800..0xDC00).contains(&cp) {
                            let low = rest.get(i + 7..i + 11)?;
                            let lo = u32::from_str_radix(low, 16).ok()?;
                            let combined = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            out.push(char::from_u32(combined)?);
                            i += 10;
                        } else {
                            out.push(char::from_u32(cp)?);
                            i += 4;
                        }
                    }
                    other => out.push(other as char),
                }
                i += 1;
            }
            _ => {
                let ch = rest[i..].chars().next()?;
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    None
}

fn chunk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    out.sort();
    out
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!(
            "cbms-store <sealed-book> <memory-dir> <out-dir>\n\
             \n\
             Reads every chunk in <memory-dir>/chunks, writes a .cbms payload per chunk\n\
             into <out-dir>, then reads them all back and compares. The memory directory\n\
             is never opened for writing."
        );
        return ExitCode::from(2);
    }
    let Ok(book_text) = std::fs::read_to_string(&args[0]) else {
        eprintln!("cannot read book {}", args[0]);
        return ExitCode::FAILURE;
    };
    let book = match Book::parse(&book_text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("book will not load: {e}");
            return ExitCode::FAILURE;
        }
    };
    let Some(vocab) = Vocabulary::new(&book) else {
        eprintln!("cannot build a vocabulary from this book");
        return ExitCode::FAILURE;
    };
    let chunks_dir = Path::new(&args[1]).join("chunks");
    let out_dir = Path::new(&args[2]);
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("cannot create {}: {e}", out_dir.display());
        return ExitCode::FAILURE;
    }

    let files = chunk_files(&chunks_dir);
    if files.is_empty() {
        eprintln!("no chunks under {}", chunks_dir.display());
        return ExitCode::FAILURE;
    }

    let mut source_bytes = 0usize;
    let mut stored_bytes = 0usize;
    let mut written = 0usize;
    let mut unreadable: Vec<String> = Vec::new();
    let mut mismatched: Vec<String> = Vec::new();

    for path in &files {
        let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or("?").to_string();
        let Ok(raw) = std::fs::read_to_string(path) else {
            unreadable.push(name);
            continue;
        };
        let Some(content) = json_string_field(&raw, "content") else {
            unreadable.push(name);
            continue;
        };
        if content.is_empty() {
            continue;
        }
        let Ok(payload) = container::write(&book, &vocab, &content) else {
            unreadable.push(name);
            continue;
        };
        let dest = out_dir.join(format!("{name}.cbms"));
        if std::fs::write(&dest, &payload).is_err() {
            unreadable.push(name);
            continue;
        }
        source_bytes += content.len();
        stored_bytes += payload.len();
        written += 1;

        // Read it back from disk. Verifying the value still in memory would only prove
        // the encoder agrees with itself.
        match std::fs::read(&dest).ok().and_then(|b| container::read(&book, &vocab, &b).ok()) {
            Some(back) if back == content => {}
            _ => mismatched.push(name),
        }
    }

    println!("book         : {:016x}{}", book.fingerprint(),
             if book.is_sealed() { " (sealed)" } else { " (unsealed)" });
    println!("chunks found : {}", files.len());
    println!("written      : {written}");
    println!("source       : {source_bytes:>9} bytes");
    println!("stored       : {stored_bytes:>9} bytes   {:.2}x",
             stored_bytes as f64 / source_bytes.max(1) as f64);
    println!();
    println!("read back identical : {}", written - mismatched.len());
    println!("MISMATCHED          : {}", mismatched.len());
    if !mismatched.is_empty() {
        for id in mismatched.iter().take(10) {
            println!("  {id}");
        }
    }
    if !unreadable.is_empty() {
        println!("could not process   : {} ({:?})",
                 unreadable.len(), &unreadable[..unreadable.len().min(5)]);
    }
    println!();
    println!("the live store was opened read-only; {} holds the parallel copy",
             out_dir.display());

    if mismatched.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
}
