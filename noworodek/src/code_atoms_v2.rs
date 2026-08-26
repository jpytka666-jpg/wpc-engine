use crate::{CodeAtom, CodeAtomKind, CodeLanguage, Qwen3CoderTokenizer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizedCodeAtom {
    pub atom: CodeAtom,
    pub token_ids: Vec<u32>,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug)]
pub enum CodeAtomV2Error {
    Tokenizer(String),
    InvalidUtf8Boundary,
}

impl std::fmt::Display for CodeAtomV2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tokenizer(e) => write!(f, "tokenizer: {e}"),
            Self::InvalidUtf8Boundary => write!(f, "invalid UTF-8 byte boundary"),
        }
    }
}
impl std::error::Error for CodeAtomV2Error {}

/// Structural V2 prototype: lexical Qwen tokenization + delimiter-aware
/// function extraction. This deliberately does not claim full AST parsing.
pub fn extract_functions(
    tokenizer: &Qwen3CoderTokenizer,
    language: CodeLanguage,
    source: &str,
) -> Result<Vec<TokenizedCodeAtom>, CodeAtomV2Error> {
    let mut out = Vec::new();
    let bytes = source.as_bytes();
    let mut search = 0usize;
    while let Some(rel) = find_function_marker(language, source, search) {
        let start = rel;
        let Some(open_rel) = source[start..].find('{') else { break };
        let open = start + open_rel;
        let Some(close) = matching_brace(source, open) else { break };
        let end = close + 1;
        if !source.is_char_boundary(start) || !source.is_char_boundary(end) {
            return Err(CodeAtomV2Error::InvalidUtf8Boundary);
        }
        let body = &source[start..end];
        let token_ids = tokenizer
            .encode(body, false)
            .map_err(|e| CodeAtomV2Error::Tokenizer(e.to_string()))?;
        let atom = CodeAtom::new(language, CodeAtomKind::Function, body, "v2")
            .with_experience(format!("codeatom.extract.{}.{}", language_name(language), start));
        out.push(TokenizedCodeAtom { atom, token_ids, start_byte: start, end_byte: end });
        search = end;
    }
    let _ = bytes;
    Ok(out)
}

fn language_name(language: CodeLanguage) -> &'static str {
    match language { CodeLanguage::Rust => "rust", CodeLanguage::Cpp => "cpp" }
}

fn find_function_marker(language: CodeLanguage, source: &str, from: usize) -> Option<usize> {
    let marker = match language {
        CodeLanguage::Rust => "fn ",
        CodeLanguage::Cpp => "{",
    };
    if matches!(language, CodeLanguage::Cpp) {
        // C++ V2 heuristic: identify a likely function by a signature prefix
        // ending immediately before an opening brace and containing '('.
        let mut pos = from;
        while let Some(rel) = source[pos..].find('{') {
            let open = pos + rel;
            let prefix_start = source[..open].rfind(&[';', '}', '{'][..]).map(|i| i + 1).unwrap_or(0);
            if source[prefix_start..open].contains('(') {
                return Some(prefix_start + source[prefix_start..open].find(|c: char| c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '&' || c == '*' || c.is_whitespace()).unwrap_or(0));
            }
            pos = open + 1;
        }
        None
    } else {
        source[from..].find(marker).map(|r| from + r)
    }
}

fn matching_brace(source: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (offset, ch) in source[open..].char_indices() {
        if in_string {
            if escaped { escaped = false; continue; }
            if ch == '\\' { escaped = true; continue; }
            if ch == '"' { in_string = false; }
            continue;
        }
        if ch == '"' { in_string = true; continue; }
        match ch {
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 { return Some(open + offset); }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_marker_finds_function_span() {
        let source = "fn add(a: i32, b: i32) -> i32 { a + b }\nfn x() { let y = 1; }";
        assert_eq!(find_function_marker(CodeLanguage::Rust, source, 0), Some(0));
        assert_eq!(matching_brace(source, source.find('{').unwrap()), source.find('}'));
    }
}
