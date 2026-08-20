use serde::Deserialize;
use std::path::Path;

/// Mirrors the subset of `config.json` fields the engine needs for the Qwen
/// family. Qwen2 and Qwen3 share this struct: every Qwen3-only knob is an
/// `Option` that defaults to `None`, which reproduces the Qwen2 behaviour
/// exactly, so a Qwen2 `config.json` parses and behaves as it always did.
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
    /// `model_type` verbatim from config.json ("qwen2", "qwen3", ...). Used
    /// only for architecture auto-detection; absent in hand-written configs.
    #[serde(default)]
    pub model_type: Option<String>,
    /// Explicit attention head dimension. Qwen3 states it in config.json and
    /// it is NOT `hidden_size / num_attention_heads` there (2560/32 = 80, but
    /// the real value is 128, so q_proj is `[4096, 2560]` and o_proj is
    /// `[2560, 4096]`). Qwen2 omits the key, so this stays `None` and
    /// [`Config::head_dim`] keeps deriving it as before.
    #[serde(default, rename = "head_dim")]
    pub head_dim_override: Option<usize>,
    /// Qwen3-MoE only. `num_experts` is how many expert MLPs each layer holds
    /// (128 on Qwen3-Coder-30B-A3B), `num_experts_per_tok` how many of them
    /// actually run for a given token (8), `moe_intermediate_size` the width of
    /// one expert (768, against `intermediate_size` 6144 for the dense variant),
    /// and `norm_topk_prob` whether the top-k routing weights are renormalised
    /// to sum to 1 after selection. Every dense Qwen config omits all four, so
    /// they stay `None` and the dense path is untouched.
    #[serde(default)]
    pub num_experts: Option<usize>,
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    #[serde(default)]
    pub moe_intermediate_size: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,
}

fn default_true() -> bool {
    true
}

impl Config {
    /// Attention head dimension: the value stated in config.json when there is
    /// one (Qwen3), otherwise derived from hidden size (Qwen2, unchanged).
    pub fn head_dim(&self) -> usize {
        self.head_dim_override
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// True when config.json declares `model_type: "qwen3"`, i.e. the
    /// checkpoint carries `self_attn.q_norm`/`k_norm` and has no q/k/v biases.
    pub fn is_qwen3(&self) -> bool {
        self.model_type.as_deref() == Some("qwen3")
    }

    /// True when config.json declares `model_type: "qwen3_moe"`. Such a
    /// checkpoint has no `mlp.{gate,up,down}_proj` at all: each layer carries a
    /// router (`mlp.gate.weight`) plus `num_experts` separate expert MLPs, and
    /// `lm_head.weight` is a tensor of its own rather than the tied embedding.
    pub fn is_qwen3_moe(&self) -> bool {
        self.model_type.as_deref() == Some("qwen3_moe")
    }

    /// Attention behaves exactly as on dense Qwen3 (q_norm/k_norm, no biases)
    /// for the MoE variant too, so anything gated on "is this Qwen3-shaped
    /// attention" must accept both.
    pub fn has_qwen3_attention(&self) -> bool {
        self.is_qwen3() || self.is_qwen3_moe()
    }

    pub fn from_json_str(text: &str) -> anyhow::Result<Config> {
        let cfg: Config = serde_json::from_str(text)?;
        Ok(cfg)
    }

    pub fn load(path: &Path) -> anyhow::Result<Config> {
        let text = std::fs::read_to_string(path)?;
        Config::from_json_str(&text)
    }

    /// Read just `model_type` out of a config.json, tolerating files this
    /// struct could not fully parse (e.g. a Gemma4-shaped config). Returns
    /// `None` when the file is unreadable, is not JSON, or has no such key.
    pub fn model_type_of(path: &Path) -> Option<String> {
        let text = std::fs::read_to_string(path).ok()?;
        let v: serde_json::Value = serde_json::from_str(&text).ok()?;
        v.get("model_type")?.as_str().map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Qwen2-shaped config: no `head_dim` key, no `model_type` we care about.
    const QWEN2_JSON: &str = r#"{
        "model_type": "qwen2",
        "vocab_size": 151936,
        "hidden_size": 1536,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "intermediate_size": 8960,
        "rope_theta": 1000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true
    }"#;

    /// Qwen3-shaped config: `head_dim` stated explicitly and deliberately
    /// different from hidden_size / num_attention_heads.
    const QWEN3_JSON: &str = r#"{
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 2560,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 9728,
        "head_dim": 128,
        "attention_bias": false,
        "rope_theta": 5000000.0,
        "rms_norm_eps": 1e-06,
        "tie_word_embeddings": true
    }"#;

    #[test]
    fn qwen2_derives_head_dim_from_hidden_size() {
        let cfg = Config::from_json_str(QWEN2_JSON).expect("qwen2 config parses");
        assert_eq!(cfg.head_dim_override, None, "qwen2 states no head_dim");
        assert_eq!(cfg.head_dim(), 1536 / 12);
        assert_eq!(cfg.head_dim(), 128);
        assert!(!cfg.is_qwen3());
    }

    #[test]
    fn qwen3_reads_head_dim_from_the_file() {
        let cfg = Config::from_json_str(QWEN3_JSON).expect("qwen3 config parses");
        assert_eq!(cfg.head_dim_override, Some(128));
        assert_eq!(cfg.head_dim(), 128);
        // The derived Qwen2 formula would give 80 here; the file wins.
        assert_ne!(cfg.head_dim(), cfg.hidden_size / cfg.num_attention_heads);
        assert!(cfg.is_qwen3());
        // 32 heads * 128 != hidden 2560, so the attention block is wider than
        // the residual stream.
        assert_eq!(cfg.num_attention_heads * cfg.head_dim(), 4096);
    }

    #[test]
    fn unknown_config_keys_are_ignored() {
        // `attention_bias`, `sliding_window` etc. are present in real files and
        // must not break parsing.
        let cfg = Config::from_json_str(QWEN3_JSON).expect("extra keys tolerated");
        assert_eq!(cfg.num_hidden_layers, 36);
    }
}
