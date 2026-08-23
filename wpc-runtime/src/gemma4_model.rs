//! Gemma4 text-architecture decoder stack. Parallel to `model.rs`'s Qwen2
//! path (see that module's docs for why this isn't unified into one
//! generic implementation) — reuses the same `Linear`/`EmbeddingTable`
//! trait objects and `WpcLinear`/`WpcEmbedding`/`SafetensorsFile` backends,
//! just wires them up per Gemma4's decoder-layer shape:
//!
//!   residual = h
//!   h = input_layernorm(h)
//!   h = self_attn(h)              # q_proj->q_norm->RoPE, k_proj->k_norm->RoPE
//!                                  # (or v=k_proj-pre-norm on full-attn layers,
//!                                  # which have no v_proj at all), v_proj->v_norm
//!                                  # (v_norm has no learned weight), GQA, o_proj
//!   h = post_attention_layernorm(h)
//!   h = residual + h
//!   residual = h
//!   h = pre_feedforward_layernorm(h)
//!   h = mlp(h)                    # down(gelu_tanh(gate(h)) * up(h))
//!   h = post_feedforward_layernorm(h)
//!   h = residual + h
//!
//! Sandwich norms, q/k/v-norm, per-layer-type head_dim/num_kv_heads/RoPE,
//! attention scaling=1.0, sliding-window masking, and final logit
//! softcapping are all verified against HF `transformers`' real
//! `modeling_gemma4.py`/`configuration_gemma4.py` (see `gemma4_config.rs`
//! module docs for exactly what was fetched).

use crate::gemma4_config::{Gemma4Config, LayerSpec};
use crate::norm::{rms_norm, rms_norm_no_weight, softmax_inplace};
use crate::rope::apply_rope_partial;
use crate::weights::{EmbeddingTable, Linear, SafetensorsFile};
use crate::wpc_weights::{WpcEmbedding, WpcLinear, WpcModelData};
use crate::wpc_weights_v2::{WpcEmbeddingV2, WpcLinearV2, WpcModelDataV2};
use crate::wpc_weights_v3::{WpcEmbeddingV3, WpcLinearV3, WpcModelDataV3};
use rayon::prelude::*;
use std::path::Path;

pub struct Gemma4Layer {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub pre_feedforward_layernorm: Vec<f32>,
    pub post_feedforward_layernorm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q_proj: Box<dyn Linear>,
    pub k_proj: Box<dyn Linear>,
    pub v_proj: Option<Box<dyn Linear>>,
    pub o_proj: Box<dyn Linear>,
    pub gate_proj: Box<dyn Linear>,
    pub up_proj: Box<dyn Linear>,
    pub down_proj: Box<dyn Linear>,
    pub layer_scalar: f32,
    pub spec: LayerSpec,
}

pub struct Gemma4Model {
    pub config: Gemma4Config,
    pub embed: Box<dyn EmbeddingTable>,
    pub layers: Vec<Gemma4Layer>,
    pub final_norm: Vec<f32>,
}

struct LayerCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

pub struct Gemma4KvCache {
    layers: Vec<LayerCache>,
    pub len: usize,
}

impl Gemma4KvCache {
    fn new(layer_specs: &[LayerSpec]) -> Self {
        Gemma4KvCache {
            layers: layer_specs
                .iter()
                .map(|s| LayerCache {
                    k: vec![Vec::new(); s.num_kv_heads],
                    v: vec![Vec::new(); s.num_kv_heads],
                })
                .collect(),
            len: 0,
        }
    }
}

const LANG_PREFIX: &str = "model.language_model";

// The full implementation on this branch is restored from the last clean
// commit below by a targeted cleanup pass. This placeholder content is never
// intended for execution.
