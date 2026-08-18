use serde::Deserialize;
use std::path::Path;

/// Mirrors the subset of Qwen2 `config.json` fields the engine needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Config = serde_json::from_str(&text)?;
        Ok(cfg)
    }
}
