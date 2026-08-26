use noworodek::code_atoms::{CodeAtom, CodeAtomKind, CodeLanguage, CodeAtomRegistry};

#[test]
fn atom_id_is_deterministic() {
    let a = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn add(a:i32,b:i32)->i32{a+b}", "1.0.0");
    let b = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn add(a:i32,b:i32)->i32{a+b}", "1.0.0");
    assert_eq!(a.id(), b.id());
}

#[test]
fn duplicate_registration_is_idempotent() {
    let atom = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn add(a:i32,b:i32)->i32{a+b}", "1.0.0");
    let mut registry = CodeAtomRegistry::new();
    assert!(registry.insert(atom.clone()).expect("insert"));
    assert!(!registry.insert(atom.clone()).expect("duplicate insert"));
    assert_eq!(registry.len(), 1);
}

#[test]
fn patch_lineage_links_original_to_repaired_atom() {
    let original = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn div(a:i32,b:i32)->i32{a/b}", "1.0.0");
    let patch = CodeAtom::patch(CodeLanguage::Rust, original.id(), "guard divisor", "1.0.0");
    let repaired = CodeAtom::derived(CodeLanguage::Rust, CodeAtomKind::DebugFix, patch.id(), "fn div(a:i32,b:i32)->Result<i32,String>{if b==0{Err(\"zero\".into())}else{Ok(a/b)}}", "1.0.0");
    let mut registry = CodeAtomRegistry::new();
    registry.insert(original.clone()).unwrap();
    registry.insert(patch.clone()).unwrap();
    registry.insert(repaired.clone()).unwrap();
    assert_eq!(registry.parent_of(&patch.id()), Some(original.id()));
    assert_eq!(registry.parent_of(&repaired.id()), Some(patch.id()));
    assert_eq!(registry.children_of(&original.id()).len(), 1);
}

#[test]
fn lookup_filters_language_and_kind() {
    let rust_fn = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Function, "fn add(a:i32,b:i32)->i32{a+b}", "1.0.0");
    let cpp_fn = CodeAtom::new(CodeLanguage::Cpp, CodeAtomKind::Function, "int add(int a,int b){return a+b;}", "1.0.0");
    let rust_block = CodeAtom::new(CodeLanguage::Rust, CodeAtomKind::Block, "if x > 0 { 1 } else { 0 }", "1.0.0");
    let mut registry = CodeAtomRegistry::new();
    for atom in [rust_fn, cpp_fn, rust_block] { registry.insert(atom).unwrap(); }
    assert_eq!(registry.by_language_kind(CodeLanguage::Rust, CodeAtomKind::Function).len(), 1);
    assert_eq!(registry.by_language_kind(CodeLanguage::Cpp, CodeAtomKind::Function).len(), 1);
    assert_eq!(registry.by_language_kind(CodeLanguage::Rust, CodeAtomKind::Block).len(), 1);
}
