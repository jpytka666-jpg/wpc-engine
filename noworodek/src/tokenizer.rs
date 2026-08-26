//! Qwen3-Coder-30B-A3B-Instruct tokenizer contract.
use std::path::Path;
use tokenizers::Tokenizer;
pub const MODEL_ID:&str="Qwen/Qwen3-Coder-30B-A3B-Instruct";
pub const MODEL_REVISION:&str="573fa3901e5799703b1e60825b0ec024a4c0f1d3";
pub const VOCAB_SIZE:u32=151_936;
pub const MAX_POSITION_TOKENS:usize=1_048_576;
pub const EOS_TOKEN:&str="<|im_end|>"; pub const PAD_TOKEN:&str="<|endoftext|>";
pub const EOS_ID:u32=151_645; pub const PAD_ID:u32=151_643; pub const IM_START_ID:u32=151_644; pub const IM_END_ID:u32=151_645;
#[derive(Debug)] pub enum TokenizerError{Load(String),VocabMismatch{expected:u32,actual:usize}}
impl std::fmt::Display for TokenizerError{fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{match self{Self::Load(m)=>write!(f,"tokenizer load failed: {m}"),Self::VocabMismatch{expected,actual}=>write!(f,"tokenizer vocab exceeds model: model={expected}, tokenizer={actual}")}}}
impl std::error::Error for TokenizerError{}
pub struct Qwen3CoderTokenizer{inner:Tokenizer}
impl Qwen3CoderTokenizer{
 pub fn from_file(path:impl AsRef<Path>)->Result<Self,TokenizerError>{let inner=Tokenizer::from_file(path).map_err(|e|TokenizerError::Load(e.to_string()))?;let actual=inner.get_vocab_size(true);if actual>VOCAB_SIZE as usize{return Err(TokenizerError::VocabMismatch{expected:VOCAB_SIZE,actual})}Ok(Self{inner})}
 pub fn encode(&self,text:&str,add_special_tokens:bool)->Result<Vec<u32>,TokenizerError>{self.inner.encode(text,add_special_tokens).map(|e|e.get_ids().to_vec()).map_err(|e|TokenizerError::Load(e.to_string()))}
 pub fn decode(&self,ids:&[u32],skip_special_tokens:bool)->Result<String,TokenizerError>{self.inner.decode(ids,skip_special_tokens).map_err(|e|TokenizerError::Load(e.to_string()))}
 pub fn vocab_size(&self)->usize{self.inner.get_vocab_size(true)} pub fn base_vocab_size(&self)->usize{self.inner.get_vocab_size(false)} pub fn model_vocab_size(&self)->usize{VOCAB_SIZE as usize}
 pub fn model_id(&self)->&'static str{MODEL_ID} pub fn revision(&self)->&'static str{MODEL_REVISION} pub fn inner(&self)->&Tokenizer{&self.inner}
}
pub fn format_chat_turn(role:&str,content:&str)->String{format!("<|im_start|>{role}\n{content}<|im_end|>\n")}
pub fn format_tool_call(name:&str,arguments:&str)->String{format!("<tool_call>\n<function={name}>\n{arguments}\n</function>\n</tool_call>")}
#[cfg(test)]mod tests{use super::*;#[test]fn contract_matches_pinned_qwen_metadata(){assert_eq!(MODEL_ID,"Qwen/Qwen3-Coder-30B-A3B-Instruct");assert_eq!(MODEL_REVISION,"573fa3901e5799703b1e60825b0ec024a4c0f1d3");assert_eq!(VOCAB_SIZE,151_936);assert_eq!(EOS_ID,151_645);assert_eq!(PAD_ID,151_643);assert_eq!(IM_START_ID,151_644);assert_eq!(IM_END_ID,151_645);}#[test]fn configured_model_vocab_can_exceed_tokenizer_entry_count(){assert!(151_669usize<=VOCAB_SIZE as usize);}}
