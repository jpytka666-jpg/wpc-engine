// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 00:00:00
// REASON FOR CREATION: A codec whose only exercise is its own unit tests has been tested
//   against the author's assumptions, not against the book. This runs it over the real
//   457-entry code book and reports what it finds, including what it cannot encode.
// MECHANICS: `check` round-trips every root in the book through encode and decode and
//   fails loudly on any that does not return; `stats` reports the vocabulary a CBMS-native
//   tokenizer would need; `encode`/`decode` handle text from a file or the command line.
// SYSTEM PART: cbms-writing - the CBMS writing system
// ARCHITECTURE FUNCTION: The measuring instrument for the codec, and the tool that will
//   later rewrite a corpus into CBMS for training.
// DEPENDENCIES/LINKS: cbms_writing::{Book, Codec}
// TECH STACK: Rust 2021, standard library only.
// LOCAL WORKSPACE: C:\temp\aions-cbms-2026-08-26\cbms-writing
// GIT COMMIT: PENDING
// GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/cbms-writing
// ==========================================

use cbms_writing::{Book, Codec, Section};
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "cbms <book> <command> [text]\n\
         \n\
         check            round-trip every root in the book; non-zero exit on any loss\n\
         stats            vocabulary size and what a tokenizer would need\n\
         encode <text>    Esperanto to CBMS\n\
         decode <text>    CBMS to Esperanto\n\
         file <path>      encode a whole file, report coverage"
    );
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        return usage();
    }
    let text = match std::fs::read_to_string(&args[0]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read book {}: {e}", args[0]);
            return ExitCode::FAILURE;
        }
    };
    // Lenient on purpose: a collision is a finding to report, not a reason to show
    // nothing. The first claimant of each symbol is kept so the rest still runs.
    let (book, collisions) = match Book::parse_lenient(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("book will not load: {e}");
            return ExitCode::FAILURE;
        }
    };
    if !collisions.is_empty() {
        eprintln!("WARNING: {} symbol(s) claimed twice - these words cannot be told apart\n\
                   when decoding, and the second claimant is unreachable:", collisions.len());
        for c in &collisions {
            eprintln!(
                "  {:?} (U+{:04X})  {:?} keeps it, {:?} is lost",
                c.symbol,
                c.symbol.chars().next().map(|ch| ch as u32).unwrap_or(0),
                c.first,
                c.second
            );
        }
        eprintln!();
    }
    let Some(codec) = Codec::new(&book) else {
        eprintln!("book has no MORPH-SEP entry; encoded words could not be separated");
        return ExitCode::FAILURE;
    };

    match args[1].as_str() {
        "stats" => {
            let cps = book.codepoints();
            println!("lexical roots        : {}", book.count(Section::Lexical));
            println!("grammar and protocol : {}", book.count(Section::Extension));
            println!("entries total        : {}", book.len());
            println!("distinct codepoints  : {}", cps.len());
            println!();
            println!("a CBMS-native tokenizer would carry roughly {} ids,", book.len() + 128);
            println!("against Qwen3's 151 936 - about {}x smaller.", 151_936 / (book.len() + 128).max(1));
            ExitCode::SUCCESS
        }
        "check" => {
            let mut checked = 0usize;
            let mut lost: Vec<(String, String, String)> = Vec::new();
            let mut unencodable: Vec<String> = Vec::new();
            for entry in book.entries() {
                if entry.section != Section::Lexical {
                    continue;
                }
                checked += 1;
                match codec.encode_word(&entry.root) {
                    None => unencodable.push(entry.root.clone()),
                    Some(enc) => match codec.decode_word(&enc) {
                        Some(dec) if dec == entry.root => {}
                        Some(dec) => lost.push((entry.root.clone(), enc, dec)),
                        None => lost.push((entry.root.clone(), enc, "<no decode>".into())),
                    },
                }
            }
            // The constraint: one word must fit in one to three symbols. Checked over
            // every root in its inflected forms, not over a chosen sentence.
            let mut hist = [0usize; 8];
            let mut worst: Vec<(String, String)> = Vec::new();
            for entry in book.entries() {
                if entry.section != Section::Lexical {
                    continue;
                }
                let stem: String = {
                    let r = &entry.root;
                    match r.chars().last() {
                        Some(c) if "oaei".contains(c) => r[..r.len() - c.len_utf8()].to_string(),
                        _ => r.clone(),
                    }
                };
                for form in [
                    entry.root.clone(),
                    format!("{stem}on"), format!("{stem}oj"), format!("{stem}ojn"),
                    format!("{stem}a"), format!("{stem}an"), format!("{stem}ajn"),
                    format!("{stem}as"), format!("{stem}is"), format!("{stem}os"),
                ] {
                    if let Some(enc) = codec.encode_word(&form) {
                        let n = enc.chars().count();
                        hist[n.min(7)] += 1;
                        if n > 3 {
                            worst.push((form, enc));
                        }
                    }
                }
            }

            println!("roots checked  : {checked}");
            println!("round-tripped  : {}", checked - lost.len() - unencodable.len());
            println!("not encodable  : {}", unencodable.len());
            println!("LOST IN TRANSIT: {}", lost.len());
            println!();
            println!("symbols per word, over every root in its inflected forms:");
            for (n, &count) in hist.iter().enumerate() {
                if count > 0 {
                    let flag = if n > 3 { "  <- OVER THE LIMIT" } else { "" };
                    println!("  {n} symbol(s): {count:>6}{flag}");
                }
            }
            if worst.is_empty() {
                println!("  every form fits in 3 symbols or fewer");
            } else {
                println!("  {} form(s) need four or more:", worst.len());
                for (form, enc) in worst.iter().take(10) {
                    println!("    {form} -> {enc}");
                }
                println!("  (adding a `-jn` entry to the book collapses these to three)");
            }
            println!();
            for (root, enc, dec) in lost.iter().take(20) {
                println!("  {root} -> {enc} -> {dec}");
            }
            if !unencodable.is_empty() {
                println!("\nnot encodable (first 20): {:?}",
                         &unencodable[..unencodable.len().min(20)]);
            }
            if lost.is_empty() { ExitCode::SUCCESS } else { ExitCode::FAILURE }
        }
        "encode" => {
            let (enc, cov) = codec.encode_text(&args[2..].join(" "));
            println!("{enc}");
            eprintln!("coverage {}/{} ({:.0}%)", cov.encoded, cov.words, 100.0 * cov.ratio());
            if !cov.missing.is_empty() {
                eprintln!("missing: {:?}", cov.missing);
            }
            ExitCode::SUCCESS
        }
        "decode" => {
            println!("{}", codec.decode_text(&args[2..].join(" ")));
            ExitCode::SUCCESS
        }
        "file" => {
            let Ok(body) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read {}", args[2]);
                return ExitCode::FAILURE;
            };
            let (enc, cov) = codec.encode_text(&body);
            println!("source chars  : {}", body.chars().count());
            println!("encoded chars : {}", enc.chars().count());
            println!("ratio         : {:.2}x", enc.chars().count() as f64 / body.chars().count().max(1) as f64);
            println!("coverage      : {}/{} ({:.1}%)", cov.encoded, cov.words, 100.0 * cov.ratio());
            let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
            for m in &cov.missing {
                *counts.entry(m.as_str()).or_default() += 1;
            }
            let mut top: Vec<_> = counts.into_iter().collect();
            top.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            println!("\nmost frequent words the book lacks - these are what to add next:");
            for (w, n) in top.iter().take(25) {
                println!("  {n:>5}x  {w}");
            }
            ExitCode::SUCCESS
        }
        _ => usage(),
    }
}
