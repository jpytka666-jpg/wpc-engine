use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CodeLanguage { Rust, Cpp }

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CodeAtomKind { Function, Block, Patch, DebugFix }

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CodeAtomId(String);

impl CodeAtomId { pub fn as_str(&self) -> &str { &self.0 } }
impl fmt::Display for CodeAtomId { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.0) } }

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodeAtom {
    id: CodeAtomId,
    language: CodeLanguage,
    kind: CodeAtomKind,
    source: String,
    version: String,
    parent: Option<CodeAtomId>,
    related: Option<CodeAtomId>,
    experience_id: Option<String>,
}

impl CodeAtom {
    pub fn new(language: CodeLanguage, kind: CodeAtomKind, source: impl Into<String>, version: impl Into<String>) -> Self {
        Self::build(language, kind, source.into().trim().to_string(), version.into(), None, None, None)
    }
    pub fn patch(language: CodeLanguage, parent: CodeAtomId, description: impl Into<String>, version: impl Into<String>) -> Self {
        Self::build(language, CodeAtomKind::Patch, description.into().trim().to_string(), version.into(), Some(parent), None, None)
    }
    pub fn derived(language: CodeLanguage, kind: CodeAtomKind, parent: CodeAtomId, source: impl Into<String>, version: impl Into<String>) -> Self {
        Self::build(language, kind, source.into().trim().to_string(), version.into(), Some(parent), None, None)
    }
    pub fn with_experience(mut self, experience_id: impl Into<String>) -> Self { self.experience_id = Some(experience_id.into()); self }
    pub fn with_related(mut self, related: CodeAtomId) -> Self { self.related = Some(related); self.id = make_id(self.language, self.kind, &self.source, &self.version, self.parent.as_ref(), self.related.as_ref()); self }
    fn build(language: CodeLanguage, kind: CodeAtomKind, source: String, version: String, parent: Option<CodeAtomId>, related: Option<CodeAtomId>, experience_id: Option<String>) -> Self { let id = make_id(language, kind, &source, &version, parent.as_ref(), related.as_ref()); Self { id, language, kind, source, version, parent, related, experience_id } }
    pub fn id(&self) -> CodeAtomId { self.id.clone() }
    pub fn language(&self) -> CodeLanguage { self.language }
    pub fn kind(&self) -> CodeAtomKind { self.kind }
    pub fn source(&self) -> &str { &self.source }
    pub fn version(&self) -> &str { &self.version }
    pub fn parent(&self) -> Option<&CodeAtomId> { self.parent.as_ref() }
    pub fn related(&self) -> Option<&CodeAtomId> { self.related.as_ref() }
    pub fn experience_id(&self) -> Option<&str> { self.experience_id.as_deref() }
}

#[derive(Clone, Debug, Default)]
pub struct CodeAtomRegistry { atoms: BTreeMap<CodeAtomId, CodeAtom> }
impl CodeAtomRegistry {
    pub fn new() -> Self { Self::default() }
    pub fn len(&self) -> usize { self.atoms.len() }
    pub fn insert(&mut self, atom: CodeAtom) -> Result<bool, String> { if let Some(existing)=self.atoms.get(&atom.id) { if existing != &atom { return Err("atom id collision with different payload".into()); } return Ok(false); } self.atoms.insert(atom.id.clone(), atom); Ok(true) }
    pub fn get(&self, id: &CodeAtomId) -> Option<&CodeAtom> { self.atoms.get(id) }
    pub fn parent_of(&self, id: &CodeAtomId) -> Option<CodeAtomId> { self.get(id).and_then(|a| a.parent().cloned()) }
}

fn make_id(language: CodeLanguage, kind: CodeAtomKind, source: &str, version: &str, parent: Option<&CodeAtomId>, related: Option<&CodeAtomId>) -> CodeAtomId {
    let mut h=0xcbf29ce484222325u64;
    let mut feed=|s:&str| { for b in s.as_bytes(){ h^=u64::from(*b); h=h.wrapping_mul(0x100000001b3);} h^=0xff; };
    feed(match language { CodeLanguage::Rust=>"rust", CodeLanguage::Cpp=>"cpp" });
    feed(match kind { CodeAtomKind::Function=>"function", CodeAtomKind::Block=>"block", CodeAtomKind::Patch=>"patch", CodeAtomKind::DebugFix=>"debugfix" });
    feed(source); feed(version); feed(parent.map(CodeAtomId::as_str).unwrap_or("")); feed(related.map(CodeAtomId::as_str).unwrap_or(""));
    CodeAtomId(format!("codeatom-{h:016x}"))
}
