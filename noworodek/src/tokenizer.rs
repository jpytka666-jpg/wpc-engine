//! Qwen3-Coder-30B-A3B-Instruct tokenizer contract.
//!
//! The tokenizer is an input/serialization dependency of Noworodek. No model
//! weights are imported from Qwen. The reference tokenizer revision is pinned
//! so token IDs remain reproducible across experiments.

use std::path::Path;

use tokenizers::Tokenizer;

pub const MODEL_ID: &str = "Qwen/Qwen3-Coder-30B-A3B-Instruct";
pub const MODEL_REVISION: &str = "b2cff646eb4bb1d68355c01b18ae02e7cf42d120";
pub const VOCAB_SIZE: u32 = 151_936;
pub const MAX_POSITION_TOKENS: usize = 1_048_576;
pub const EOS_TOKEN: &str = "<|im_end|>";
pub const PAD_TOKEN: &str = "<|endoftext|>";
pub const EOS_ID: u32 = 151_645;
pub const PAD_ID: u32 = 151_643;
pub const IM_START_ID: u32 = 151_644;
pub const IM_END_ID: u32 = 151_645;

#[derive(Debug)]
pub enum TokenizerError {
    Load(String),
    VocabMismatch { expected: u32, actual: usize },
}

impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Load(message) => write!(f, "tokenizer load failed: {message}"),
            Self::VocabMismatch { expected, actual } => {
                write!(f, "tokenizer vocab mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for TokenizerError {}

pub struct Qwen3CoderTokenizer {
    inner: Tokenizer,
}

impl Qwen3CoderTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = Tokenizer::from_file(path).map_err(|error| TokenizerError::Load(error.to_string()))?;
        let actual = inner.get_vocab_size(false);
        if actual != VOCAB_SIZE as usize {
            return Err(TokenizerError::VocabMismatch { expected: VOCAB_SIZE, actual });
        }
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|error| TokenizerError::Load(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, TokenizerError> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|error| TokenizerError::Load(error.to_string()))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    pub fn model_id(&self) -> &'static str { MODEL_ID }
    pub fn revision(&self) -> &'static str { MODEL_REVISION }

    pub fn inner(&self) -> &Tokenizer { &self.inner }
}

/// Minimal chat serialization contract for the Qwen3-Coder special tokens.
/// Full Jinja rendering remains a separate adapter concern; the tokenizer
/// itself stays responsible only for text <-> token IDs.
pub fn format_chat_turn(role: &str, content: &str) -> String {
    format!("<|im_start|>{role}\n{content}<|im_end|>\n")
}

pub fn format_tool_call(name: &str, arguments: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n{arguments}\n</function>\n</tool_call>"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contract_matches_pinned_qwen_metadata() {
        assert_eq!(MODEL_ID, "Qwen/Qwen3-Coder-30B-A3B-Instruct");
        assert_eq!(VOCAB_SIZE, 151_936);
        assert_eq!(EOS_ID, 151_645);
        assert_eq!(PAD_ID, 151_643);
        assert_eq!(IM_START_ID, 151_644);
        assert_eq!(IM_END_ID, 151_645);
    }

    #[test]
    fn chat_turn_uses_qwen_message_boundaries() {
        assert_eq!(
            format_chat_turn("user", "hello"),
            "<|im_start|>user\nhello<|im_end|>\n"
        );
    }

    #[test]
    fn tool_call_contract_is_stable() {
        let rendered = format_tool_call("inspect_code", "path=/tmp/a.rs");
        assert!(rendered.starts_with("<tool_call>\n<function=inspect_code>"));
        assert!(rendered.contains("path=/tmp/a.rs"));
        assert!(rendered.ends_with("</function>\n</tool_call>"));
    }
}
