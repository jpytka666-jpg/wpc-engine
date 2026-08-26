use crate::{CodeAtom, CodeAtomKind, CodeLanguage, Qwen3CoderTokenizer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizedCodeAtom { pub atom: CodeAtom, pub token_ids: Vec<u32>, pub start_byte: usize, pub end_byte: usize }

#[derive(Debug)] pub enum CodeAtomV2Error { Tokenizer(String), InvalidUtf8Boundary }
impl std::fmt::Display for CodeAtomV2Error { fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { match self {Self::Tokenizer(e)=>write!(f,"tokenizer: {e}"),Self::InvalidUtf8Boundary=>write!(f,"invalid UTF-8 byte boundary")} } }
impl std::error::Error for CodeAtomV2Error {}

pub fn extract_functions(tokenizer:&Qwen3CoderTokenizer, language:CodeLanguage, source:&str)->Result<Vec<TokenizedCodeAtom>,CodeAtomV2Error>{
 let mut out=Vec::new(); let mut search=0usize;
 while let Some(start)=find_marker(language,source,search){ let Some(open_rel)=source[start..].find('{') else{break}; let open=start+open_rel; let Some(close)=matching_brace(source,open) else{break}; let end=close+1; if !source.is_char_boundary(start)||!source.is_char_boundary(end){return Err(CodeAtomV2Error::InvalidUtf8Boundary)}; let body=&source[start..end]; let token_ids=tokenizer.encode(body,false).map_err(|e|CodeAtomV2Error::Tokenizer(e.to_string()))?; let atom=CodeAtom::new(language,CodeAtomKind::Function,body,"v2"); out.push(TokenizedCodeAtom{atom,token_ids,start_byte:start,end_byte:end}); search=end; }
 Ok(out)
}
fn find_marker(language:CodeLanguage,source:&str,from:usize)->Option<usize>{ match language { CodeLanguage::Rust=>source[from..].find("fn ").map(|r|from+r), CodeLanguage::Cpp=>{ let mut p=from; while let Some(r)=source[p..].find('{'){let open=p+r;let s=source[..open].rfind(&[';','}','{'][..]).map(|i|i+1).unwrap_or(0);if source[s..open].contains('('){return Some(s)} p=open+1;}None}}}
fn matching_brace(source:&str,open:usize)->Option<usize>{let mut depth=0usize;let mut string=false;let mut escaped=false;for (off,ch) in source[open..].char_indices(){if string{if escaped{escaped=false;continue} if ch=='\\'{escaped=true;continue} if ch=='"'{string=false} continue} if ch=='"'{string=true;continue} match ch{'{'=>depth+=1,'}'=>{depth=depth.saturating_sub(1);if depth==0{return Some(open+off)}},_=>{}}}None}

#[cfg(test)] mod tests { use super::*; #[test] fn brace_matching_handles_nested(){let s="fn x(){if true { 1 } else { 2 }}";let o=s.find('{').unwrap();assert_eq!(matching_brace(s,o),Some(s.len()-1));} }
