//! Config for the Gemma4 text architecture (`model_type: "gemma4_unified"`,
//! text sub-config at `config.json`'s `text_config` key). Verified against
//! the real fields present in the target checkpoint's `config.json` and
//! against HF `transformers` reference source
//! (`src/transformers/models/gemma4/{modeling,configuration}_gemma4.py`,
//! fetched from the `main` branch and from the `transformers==5.15.0` wheel
//! for `modeling_rope_utils.py`'s `_compute_proportional_rope_parameters`).
//!
//! This is a deliberately separate, parallel config/model path from the
//! existing Qwen2 `Config`/`Model` (see module docs in `model.rs`) — Gemma4
//! differs enough (per-layer-type head_dim/num_kv_heads, sandwich norms,
//! q/k/v-norm, partial RoPE, softcapping) that sharing the Qwen2 structs
//! would require plumbing options through every call site for no benefit.

use serde::Deserialize;
use std::path::Path;

fn default_true() -> bool {
    true
}
fn default_global_head_dim() -> usize {
    512
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeLayerParams {
    pub rope_theta: f64,
    #[serde(default)]
    pub partial_rotary_factor: Option<f64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub full_attention: RopeLayerParams,
    pub sliding_attention: RopeLayerParams,
}

/// Mirrors the subset of Gemma4's `text_config` fields the engine needs.
/// Extra JSON fields (e.g. `enable_moe_block`, `num_experts`, vision-only
/// knobs) are ignored by serde by default.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Sliding-attention-layer KV head count (full-attention layers use
    /// `num_global_key_value_heads` instead, see [`Gemma4Config::layer_spec`]).
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    /// Sliding-attention-layer head_dim (full-attention layers use
    /// `global_head_dim` instead — confirmed from actual tensor shapes in
    /// `model.meta`: full-attention layers' q_proj is `[16*512, 3840]`, not
    /// `[16*256, 3840]`).
    pub head_dim: usize,
    #[serde(default = "default_global_head_dim")]
    pub global_head_dim: usize,
    pub num_global_key_value_heads: usize,
    #[serde(default = "default_true")]
    pub attention_k_eq_v: bool,
    pub rms_norm_eps: f64,
    pub layer_types: Vec<String>,
    pub sliding_window: usize,
    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,
    #[serde(default = "default_true")]
    #[allow(dead_code)]
    pub tie_word_embeddings: bool,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
}

/// Per-layer attention shape, derived from [`Gemma4Config`] + `layer_types`.
/// See `configuration_gemma4.py`'s `Gemma4TextConfig.__post_init__`: for
/// `layer_type == "full_attention"`, `head_dim` and (when
/// `attention_k_eq_v`) `num_key_value_heads` are overridden to
/// `global_head_dim` / `num_global_key_value_heads`; sliding-attention
/// layers use the top-level config values unmodified.
pub struct LayerSpec {
    pub is_full: bool,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    /// `false` for full-attention layers when `attention_k_eq_v` is true:
    /// there is no `v_proj` weight at all (verified absent from both the
    /// dense `model.safetensors` and the WPC `model.meta`); value states
    /// reuse the pre-norm/pre-RoPE `k_proj` output instead (`v = k` before
    /// `k_norm`/RoPE are applied to the key copy), then get `v_norm` applied.
    pub has_v_proj: bool,
    pub rope_theta: f64,
    /// Number of leading `(x[i], x[i+head_dim/2])` frequency pairs that
    /// actually rotate; the remainder pass through unchanged (implicit
    /// zero rotation angle). `head_dim/2` reproduces full rotation.
    pub rotary_pairs: usize,
    pub sliding_window: Option<usize>,
}

impl Gemma4Config {
    pub fn layer_spec(&self, layer_idx: usize) -> LayerSpec {
        let is_full = self.layer_types[layer_idx] == "full_attention";
        if is_full {
            let head_dim = self.global_head_dim;
            let num_kv_heads = if self.attention_k_eq_v {
                self.num_global_key_value_heads
            } else {
                self.num_key_value_heads
            };
            let rp = &self.rope_parameters.full_attention;
            let factor = rp.partial_rotary_factor.unwrap_or(1.0);
            let rotary_pairs = ((factor * head_dim as f64) / 2.0).floor() as usize;
            LayerSpec {
                is_full: true,
                head_dim,
                num_kv_heads,
                has_v_proj: !self.attention_k_eq_v,
                rope_theta: rp.rope_theta,
                rotary_pairs,
                sliding_window: None,
            }
        } else {
            let head_dim = self.head_dim;
            let rp = &self.rope_parameters.sliding_attention;
            let factor = rp.partial_rotary_factor.unwrap_or(1.0);
            let rotary_pairs = ((factor * head_dim as f64) / 2.0).floor() as usize;
            LayerSpec {
                is_full: false,
                head_dim,
                num_kv_heads: self.num_key_value_heads,
                has_v_proj: true,
                rope_theta: rp.rope_theta,
                rotary_pairs,
                sliding_window: Some(self.sliding_window),
            }
        }
    }

    /// Load from a Gemma4 `config.json` (the whole file; `text_config` is
    /// extracted from it). Returns `Ok(None)` if the file has no
    /// `text_config` key (i.e. it's not a Gemma4-shaped config), so callers
    /// can auto-detect architecture.
    pub fn try_load(path: &Path) -> anyhow::Result<Option<Gemma4Config>> {
        let text = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let Some(text_config) = v.get("text_config") else {
            return Ok(None);
        };
        let cfg: Gemma4Config = serde_json::from_value(text_config.clone())?;
        Ok(Some(cfg))
    }
}
