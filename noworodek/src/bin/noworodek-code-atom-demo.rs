use noworodek::{CodeAtom, CodeAtomKind, CodeAtomRegistry, CodeLanguage};

fn main() {
    println!("NOWORODEK CODE ATOM REGISTRY V1");
    let original = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn div(a:i32,b:i32)->i32{a/b}", "1.0.0")
        .with_experience("rust.division.learn.001");
    let patch = CodeAtom::patch(CodeLanguage::Rust, original.id(), "guard divisor and return Result", "1.0.0")
        .with_experience("rust.division.debug.002");
    let repaired = CodeAtom::derived(
        CodeLanguage::Rust,
        CodeAtomKind::DebugFix,
        patch.id(),
        "fn div(a:i32,b:i32)->Result<i32,String>{if b==0{Err(\"zero\".into())}else{Ok(a/b)}}",
        "1.0.0",
    );
    let mut registry = CodeAtomRegistry::new();
    for atom in [original.clone(), patch.clone(), repaired.clone()] {
        println!("insert id={} kind={:?} inserted={}", atom.id(), atom.kind(), registry.insert(atom).unwrap());
    }
    println!("registry_len={}", registry.len());
    println!("original_id={}", original.id());
    println!("patch_parent={:?}", registry.parent_of(&patch.id()).map(|x| x.to_string()));
    println!("repaired_parent={:?}", registry.parent_of(&repaired.id()).map(|x| x.to_string()));
    println!("rust_functions={}", registry.by_language_kind(CodeLanguage::Rust, CodeAtomKind::Function).len());
    println!("rust_debug_fixes={}", registry.by_language_kind(CodeLanguage::Rust, CodeAtomKind::DebugFix).len());
    println!("RESULT code_atom_lineage=true external_memory_unit=true vocab_token=false");
}
