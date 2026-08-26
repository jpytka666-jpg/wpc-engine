use std::env;
use noworodek::{extract_functions, CodeLanguage, Qwen3CoderTokenizer};

fn main() {
    let path = env::args().nth(1).or_else(|| env::var("QWEN_TOKENIZER_JSON").ok())
        .expect("usage: noworodek-code-atom-v2-demo <path-to-qwen-tokenizer.json> (or QWEN_TOKENIZER_JSON)");
    let tokenizer = Qwen3CoderTokenizer::from_file(&path).unwrap();
    let rust = "fn add(a: i32, b: i32) -> i32 { a + b }\nfn square(x: i32) -> i32 { x * x }";
    let cpp = "int add(int a, int b) { return a + b; }\nint square(int x) { return x * x; }";
    println!("NOWORODEK CODE ATOM V2");
    println!("tokenizer={} revision={} vocab={}", tokenizer.model_id(), tokenizer.revision(), tokenizer.vocab_size());
    for (language, source) in [(CodeLanguage::Rust, rust), (CodeLanguage::Cpp, cpp)] {
        let atoms = extract_functions(&tokenizer, language, source).unwrap();
        println!("language={:?} functions={}", language, atoms.len());
        for atom in atoms {
            println!("  id={} span={}..{} qwen_tokens={} source={}", atom.atom.id(), atom.start_byte, atom.end_byte, atom.token_ids.len(), atom.atom.source().replace('\n', "\\n"));
        }
    }
    println!("RESULT qwen_tokenized_function_atoms=true structural_parser=heuristic_v2 ast_engine=false");
}
