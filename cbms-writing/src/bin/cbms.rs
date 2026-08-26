// ==========================================
// AUTHOR: M. SZUL
// AI MODEL: Claude Opus 5
// TIMESTAMP: 2026-08-26 11:20:03
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
         file <path>      encode a whole file, report coverage\n\
         pack <corpus>    measure packing and frequency coding\n\
         build <corpus> <out> [max] [min]   mint entries from a corpus\n\
         seal <corpus> <out>                freeze a code table into the book\n\
         write <text> <out.cbms>            store or send\n\
         read <in.cbms> [out]               read back"
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
        "pack" => {
            // The comparison that matters is against what would otherwise be stored or
            // sent: the source text. Comparing packed ids to the UTF-8 of the symbols
            // flatters the result against a baseline nobody would ever transmit.
            let Ok(body) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read {}", args[2]);
                return ExitCode::FAILURE;
            };
            let Some(vocab) = cbms_writing::Vocabulary::new(&book) else {
                eprintln!("cannot build a vocabulary from this book");
                return ExitCode::FAILURE;
            };
            let ids = vocab.encode(&body);
            let packed = vocab.pack(&ids);
            let back = vocab.decode(&ids);

            let source_bytes = body.len();
            println!("vocabulary    : {} ids, {} bits each", vocab.len(), vocab.bits_per_id());
            println!("symbols in it : {}", vocab.symbol_count());
            println!();
            println!("source UTF-8  : {source_bytes:>9} bytes");
            println!("CBMS ids      : {:>9}", ids.len());
            println!("packed        : {:>9} bytes   {:.2}x of source",
                     packed.len(), packed.len() as f64 / source_bytes.max(1) as f64);
            println!();
            println!("LOSSLESS      : {}", if back == body { "yes" } else { "NO - ENCODING LOSES DATA" });
            if back != body {
                // Say exactly where, so the defect is a location rather than a mood.
                let a: Vec<char> = body.chars().collect();
                let b: Vec<char> = back.chars().collect();
                let at = a.iter().zip(b.iter()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                let from = at.saturating_sub(40);
                let to_a = (at + 40).min(a.len());
                let to_b = (at + 40).min(b.len());
                println!();
                println!("first difference at character {at} (source {} chars, decoded {} chars)",
                         a.len(), b.len());
                println!("  source : {:?}", a[from..to_a].iter().collect::<String>());
                println!("  decoded: {:?}", b[from..to_b].iter().collect::<String>());
                return ExitCode::FAILURE;
            }

            // Fixed-width packing spends the same bits on the 1367th `status` as on a
            // one-off hash. Order-0 entropy is what a frequency-weighted code would
            // spend instead, and the gap between the two is the money left on the table.
            let mut freq: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
            for &id in &ids {
                *freq.entry(id).or_default() += 1;
            }
            let n = ids.len() as f64;
            let entropy: f64 = freq
                .values()
                .map(|&c| {
                    let p = c as f64 / n;
                    -p * p.log2()
                })
                .sum();
            let ideal_bytes = (n * entropy / 8.0).ceil() as usize;
            println!();
            println!("distinct ids used  : {}", freq.len());
            println!("order-0 entropy    : {entropy:.2} bits/id  (fixed width spends {})",
                     vocab.bits_per_id());
            println!("theoretical floor  : {ideal_bytes:>9} bytes   {:.2}x of source",
                     ideal_bytes as f64 / source_bytes.max(1) as f64);

            // And what the actual coder achieves, which is the number that counts.
            let code = cbms_writing::huffman::Code::from_frequencies(&freq, vocab.len());
            let coded = code.encode(&ids);
            let table_bytes = code.lengths().len(); // one byte per id, shipped with the book
            println!("huffman-coded      : {:>9} bytes   {:.2}x of source",
                     coded.len(), coded.len() as f64 / source_bytes.max(1) as f64);
            println!("  code table       : {table_bytes} bytes, travels with the book, not per message");
            println!("  saved vs fixed   : {:.0}%",
                     100.0 * (1.0 - coded.len() as f64 / packed.len().max(1) as f64));
            let back_ids = code.decode(&coded, ids.len());
            println!("  LOSSLESS         : {}",
                     if back_ids == ids { "yes" } else { "NO - CODING LOSES IDS" });
            if back_ids != ids {
                return ExitCode::FAILURE;
            }

            // How much of the saving is the writing system, and how much is just that
            // any repetitive text compresses? Without this the number means little.
            let literal_ids = ids.iter().filter(|&&i| i >= 8 && i < 8 + 256).count();
            println!();
            println!("ids that are literal bytes : {literal_ids} ({:.1}%)",
                     100.0 * literal_ids as f64 / ids.len().max(1) as f64);
            println!("ids that are CBMS symbols  : {} ({:.1}%)",
                     ids.len() - literal_ids,
                     100.0 * (ids.len() - literal_ids) as f64 / ids.len().max(1) as f64);
            println!();
            println!("A high literal share means the book does not cover this corpus and");
            println!("the packing is spelling it out byte by byte. Fix the book, not the packer.");
            ExitCode::SUCCESS
        }
        "build" => {
            // <book> build <corpus> <out> [max_new] [min_count]
            if args.len() < 4 {
                eprintln!("cbms <book> build <corpus> <out-book> [max_new=2000] [min_count=3]");
                return ExitCode::from(2);
            }
            let Ok(corpus) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read corpus {}", args[2]);
                return ExitCode::FAILURE;
            };
            let max_new: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2000);
            let min_count: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);

            let texts = vec![corpus];
            let (ranked, stats) = cbms_writing::survey(&book, &texts);
            println!("corpus         : {} words, {} distinct",
                     stats.total_words, stats.distinct_words);
            println!("coverage now   : {:.2}%", 100.0 * stats.coverage());
            println!();
            println!("most frequent words the book cannot encode:");
            let codec_ref = &codec;
            for wc in ranked.iter().filter(|w| codec_ref.encode_word(&w.word).is_none()).take(15) {
                println!("  {:>6}x  {}", wc.count, wc.word);
            }
            println!();

            let report = cbms_writing::extend(&book, &texts, max_new, min_count);
            println!("minted         : {} new entries", report.added);
            println!("already known  : {} skipped", report.skipped_already_known);
            if report.ran_out_of_symbols > 0 {
                println!("NO SYMBOLS LEFT: {} words went unminted - widen MINT_RANGES",
                         report.ran_out_of_symbols);
            }
            println!("coverage before: {:.2}%", 100.0 * report.coverage_before);
            println!("coverage after : {:.2}%", 100.0 * report.coverage_after);

            // A book that will not load is not a book. Check before writing it out.
            match cbms_writing::Book::parse(&report.book_text) {
                Ok(b) => println!("\nbuilt book     : {} entries, loads clean", b.len()),
                Err(e) => {
                    eprintln!("\nbuilt book will not load: {e}");
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = std::fs::write(&args[3], &report.book_text) {
                eprintln!("cannot write {}: {e}", args[3]);
                return ExitCode::FAILURE;
            }
            println!("written        : {}", args[3]);
            ExitCode::SUCCESS
        }
        "grow" => {
            // <book> grow <corpus> [max_new] [min_count]
            //
            // `build` writes a NEW book somewhere else and leaves the caller to decide
            // what to do with it. That is the right shape for experiments and the wrong
            // one for the book everything shares, because nothing checks that the new
            // file is a continuation of the old one rather than a replacement.
            //
            // This grows the shared book IN PLACE and refuses if any entry that already
            // existed would move. An id is nothing but an entry's position, so an entry
            // that moves silently redefines every block ever written against it - the
            // data still decodes, into different words. That is the one failure with no
            // downstream symptom, so it is checked here and nowhere else.
            if args.len() < 3 {
                eprintln!("cbms <book> grow <corpus> [max_new=2000] [min_count=2]");
                eprintln!("  extends the shared book in place; refuses to renumber");
                return ExitCode::from(2);
            }
            let Ok(corpus) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read corpus {}", args[2]);
                return ExitCode::FAILURE;
            };
            let max_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);
            let min_count: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);

            let before = book.len();
            let report = cbms_writing::extend(&book, &vec![corpus], max_new, min_count);
            let grown = match cbms_writing::Book::parse(&report.book_text) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("grown book will not load: {e}");
                    eprintln!("book on disk NOT touched");
                    return ExitCode::FAILURE;
                }
            };

            // The guarantee, checked entry by entry rather than on raw bytes: writing the
            // book out may reflow comments, but an entry's ROOT and SYMBOL at a given
            // position are what an id means.
            let old = book.entries();
            let new = grown.entries();
            if new.len() < old.len() {
                eprintln!("REFUSED: book would shrink, {} entries to {}", old.len(), new.len());
                eprintln!("book on disk NOT touched");
                return ExitCode::FAILURE;
            }
            for (i, (o, n)) in old.iter().zip(new.iter()).enumerate() {
                if o.root != n.root || o.symbol != n.symbol {
                    eprintln!("REFUSED: entry {i} changed, {}={} became {}={}",
                              o.root, o.symbol, n.root, n.symbol);
                    eprintln!("  everything written before this point would decode wrong");
                    eprintln!("book on disk NOT touched");
                    return ExitCode::FAILURE;
                }
            }

            // Write beside, then rename. A half-written shared book is worse than an old
            // one, and rename is the only step here that cannot land partially.
            //
            // args[0] is the book: this vector already has the program name removed, so
            // args[1] is the COMMAND. Taking args[1] here wrote the shared book to a file
            // called `grow` and reported success - which is exactly the kind of quiet
            // wrong that the entry check above exists to prevent, arriving by a different
            // door. The path is echoed at the end so the next one is visible immediately.
            let book_path = &args[0];
            let tmp = format!("{book_path}.tmp");
            if let Err(e) = std::fs::write(&tmp, &report.book_text) {
                eprintln!("cannot write {tmp}: {e}");
                return ExitCode::FAILURE;
            }
            if let Err(e) = std::fs::rename(&tmp, book_path) {
                eprintln!("cannot replace {book_path}: {e}");
                return ExitCode::FAILURE;
            }

            println!("wpisow przedtem: {before}");
            println!("dopisano       : {} nowych", report.added);
            println!("juz znanych    : {} pominietych", report.skipped_already_known);
            if report.ran_out_of_symbols > 0 {
                println!("BRAK ZNAKOW    : {} slow bez symbolu - poszerz MINT_RANGES",
                         report.ran_out_of_symbols);
            }
            println!("wpisow teraz   : {}", grown.len());
            println!("pokrycie       : {:.2}% -> {:.2}%",
                     100.0 * report.coverage_before, 100.0 * report.coverage_after);
            println!("ZADEN stary numer sie nie przesunal - sprawdzone wpis po wpisie");
            println!("ksiazka        : {book_path}");
            ExitCode::SUCCESS
        }
        "mark" => {
            // <book> mark [n]  - the book's lineage at a given size.
            //
            // Anything trained or stored against this book records the pair it prints.
            // Later, `mark <n>` on a GROWN book recomputes the same value if and only if
            // the first n entries are untouched - which is exactly the condition under
            // which those ids still mean what they meant, and therefore the condition
            // under which weights or stored blocks may be carried forward.
            //
            // Non-zero exit when the book has fewer entries than asked about: that means
            // it shrank, and answering with a hash of whatever is there would call a
            // shrunken book a valid ancestor.
            let n = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(book.len());
            match book.fingerprint_prefix(n) {
                Some(mark) => {
                    println!("wpisow  : {}", book.len());
                    println!("dla     : {n}");
                    println!("znak    : {mark:016x}");
                    ExitCode::SUCCESS
                }
                None => {
                    eprintln!("ksiazka ma {} wpisow, pytano o {n} - ksiazka SIE SKURCZYLA",
                              book.len());
                    eprintln!("nic co zapisano przy {n} wpisach nie jest juz czytelne");
                    ExitCode::FAILURE
                }
            }
        }
        "ids" => {
            // <book> ids <text-file> <out.u16> - the interface to anything that trains.
            //
            // Deliberately the dumbest possible format: little-endian u16, no header, no
            // framing. A trainer should not have to understand the writing system to
            // consume it, and a format with opinions is a format two programs can
            // disagree about. The vocabulary size is printed, because that is the one
            // number the consumer cannot recover from the file itself.
            if args.len() < 4 {
                eprintln!("cbms <book> ids <text-file> <out.u16>");
                return ExitCode::from(2);
            }
            let Ok(text) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read {}", args[2]);
                return ExitCode::FAILURE;
            };
            let Some(vocab) = cbms_writing::Vocabulary::new(&book) else {
                eprintln!("cannot build a vocabulary from this book");
                return ExitCode::FAILURE;
            };
            let ids = vocab.encode(&text);

            // Verify before writing. A training corpus that does not decode back is a
            // corpus of quiet corruption, and nothing downstream would ever notice.
            if vocab.decode(&ids) != text {
                eprintln!("REFUSED: these ids do not decode back to the source text");
                return ExitCode::FAILURE;
            }

            let mut bytes = Vec::with_capacity(ids.len() * 2);
            for id in &ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
            if let Err(e) = std::fs::write(&args[3], &bytes) {
                eprintln!("cannot write {}: {e}", args[3]);
                return ExitCode::FAILURE;
            }

            let mut hi = 0u16;
            let mut distinct = std::collections::HashSet::new();
            for &id in &ids {
                hi = hi.max(id);
                distinct.insert(id);
            }
            println!("source chars   : {}", text.chars().count());
            println!("ids            : {}", ids.len());
            println!("vocab size     : {}   <- the model needs this", vocab.len());
            println!("highest id used: {hi}");
            println!("distinct used  : {}", distinct.len());
            println!("bytes written  : {} -> {}", bytes.len(), args[3]);
            println!("round trip     : exact");
            ExitCode::SUCCESS
        }
        "seal" => {
            // <book> seal <corpus> <out-book> - freeze a code table into the book, so
            // messages carry no table of their own.
            if args.len() < 4 {
                eprintln!("cbms <book> seal <corpus> <out-book>");
                return ExitCode::from(2);
            }
            let Ok(corpus) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read corpus {}", args[2]);
                return ExitCode::FAILURE;
            };
            let Some(vocab) = cbms_writing::Vocabulary::new(&book) else {
                eprintln!("cannot build a vocabulary from this book");
                return ExitCode::FAILURE;
            };
            let ids = vocab.encode(&corpus);
            let mut freq: std::collections::HashMap<u16, usize> = std::collections::HashMap::new();
            for &id in &ids {
                *freq.entry(id).or_default() += 1;
            }
            let code = cbms_writing::huffman::Code::from_frequencies(&freq, vocab.len());

            let mut sealed = book;
            sealed.set_code_lengths(code.lengths().to_vec());
            let text = sealed.to_text();
            match cbms_writing::Book::parse(&text) {
                Ok(again) => {
                    if again.fingerprint() != sealed.fingerprint() {
                        eprintln!("sealed book does not survive a round trip through the file");
                        return ExitCode::FAILURE;
                    }
                }
                Err(e) => {
                    eprintln!("sealed book will not load: {e}");
                    return ExitCode::FAILURE;
                }
            }
            if let Err(e) = std::fs::write(&args[3], &text) {
                eprintln!("cannot write {}: {e}", args[3]);
                return ExitCode::FAILURE;
            }
            println!("sealed         : {}", args[3]);
            println!("fingerprint    : {:016x}", sealed.fingerprint());
            println!("code table     : {} lengths", sealed.code_lengths().len());
            ExitCode::SUCCESS
        }
        "write" => {
            // <book> write <text-file> <out.cbms>
            if args.len() < 4 {
                eprintln!("cbms <book> write <text-file> <out.cbms>");
                return ExitCode::from(2);
            }
            let Ok(text) = std::fs::read_to_string(&args[2]) else {
                eprintln!("cannot read {}", args[2]);
                return ExitCode::FAILURE;
            };
            let Some(vocab) = cbms_writing::Vocabulary::new(&book) else {
                eprintln!("cannot build a vocabulary from this book");
                return ExitCode::FAILURE;
            };
            match cbms_writing::write(&book, &vocab, &text) {
                Ok(bytes) => {
                    if let Err(e) = std::fs::write(&args[3], &bytes) {
                        eprintln!("cannot write {}: {e}", args[3]);
                        return ExitCode::FAILURE;
                    }
                    println!("source  : {:>9} bytes", text.len());
                    println!("written : {:>9} bytes   {:.2}x   -> {}",
                             bytes.len(), bytes.len() as f64 / text.len().max(1) as f64, args[3]);
                    println!("book    : {:016x}{}", book.fingerprint(),
                             if book.is_sealed() { " (sealed)" } else { " (unsealed, fixed width)" });
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        "read" => {
            // <book> read <in.cbms> [out-text]
            if args.len() < 3 {
                eprintln!("cbms <book> read <in.cbms> [out-text]");
                return ExitCode::from(2);
            }
            let Ok(bytes) = std::fs::read(&args[2]) else {
                eprintln!("cannot read {}", args[2]);
                return ExitCode::FAILURE;
            };
            let Some(vocab) = cbms_writing::Vocabulary::new(&book) else {
                eprintln!("cannot build a vocabulary from this book");
                return ExitCode::FAILURE;
            };
            match cbms_writing::read(&book, &vocab, &bytes) {
                Ok(text) => {
                    match args.get(3) {
                        Some(path) => {
                            if let Err(e) = std::fs::write(path, &text) {
                                eprintln!("cannot write {path}: {e}");
                                return ExitCode::FAILURE;
                            }
                            println!("{} bytes -> {path}", text.len());
                        }
                        None => print!("{text}"),
                    }
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("{e}");
                    ExitCode::FAILURE
                }
            }
        }
        _ => usage(),
    }
}
