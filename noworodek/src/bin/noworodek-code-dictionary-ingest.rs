use std::{env, fs, path::{Path, PathBuf}};

use noworodek::{extract_functions, CodeAtomRegistry, CodeLanguage, Qwen3CoderTokenizer};

fn language_for(path: &Path) -> Option<CodeLanguage> {
    match path.extension().and_then(|x| x.to_str()).unwrap_or_default() {
        "rs" => Some(CodeLanguage::Rust),
        "c" | "h" | "cc" | "cpp" | "cxx" | "hpp" | "hh" | "hxx" => Some(CodeLanguage::Cpp),
        _ => None,
    }
}

fn collect_files(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, out)?;
        } else if language_for(&path).is_some() {
            out.push(path);
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = env::args().nth(1).or_else(|| env::var("CODE_DICTIONARY_ROOT").ok())
        .ok_or("usage: noworodek-code-dictionary-ingest <source-root> (or CODE_DICTIONARY_ROOT)")?;
    let tokenizer_path = env::args().nth(2).or_else(|| env::var("QWEN_TOKENIZER_JSON").ok())
        .ok_or("usage: noworodek-code-dictionary-ingest <source-root> <qwen-tokenizer.json> (or QWEN_TOKENIZER_JSON)")?;

    let root = PathBuf::from(root);
    let tokenizer = Qwen3CoderTokenizer::from_file(tokenizer_path)?;
    let mut files = Vec::new();
    collect_files(&root, &mut files)?;
    files.sort();

    let mut registry = CodeAtomRegistry::new();
    let mut scanned = 0usize;
    let mut parsed = 0usize;
    let mut functions = 0usize;
    let mut inserted = 0usize;
    let mut duplicates = 0usize;
    let mut failures = 0usize;
    let mut qwen_tokens = 0usize;

    println!("NOWORODEK CODE DICTIONARY INGEST V1");
    println!("root={}", root.display());
    println!("tokenizer={} revision={} vocab={}", tokenizer.model_id(), tokenizer.revision(), tokenizer.vocab_size());

    for path in files {
        scanned += 1;
        let source = match fs::read_to_string(&path) {
            Ok(x) => x,
            Err(_) => { failures += 1; continue; }
        };
        let Some(language) = language_for(&path) else { continue };
        let atoms = match extract_functions(&tokenizer, language, &source) {
            Ok(x) => x,
            Err(_) => { failures += 1; continue; }
        };
        parsed += 1;
        functions += atoms.len();
        for atom in atoms {
            qwen_tokens += atom.token_ids.len();
            match registry.insert(atom.atom) {
                Ok(true) => inserted += 1,
                Ok(false) => duplicates += 1,
                Err(_) => failures += 1,
            }
        }
    }

    println!("files_scanned={}", scanned);
    println!("files_parsed={}", parsed);
    println!("functions_detected={}", functions);
    println!("atoms_inserted={}", inserted);
    println!("duplicates={}", duplicates);
    println!("registry_len={}", registry.len());
    println!("qwen_tokens_total={}", qwen_tokens);
    println!("failures={}", failures);
    println!("RESULT bulk_external_code_memory=true training_not_run=true next_stage=curriculum_sampling+backprop");
    Ok(())
}
