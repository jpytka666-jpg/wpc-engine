//! Qwen3 weight loading. The decoder stack itself is Qwen2's — same
//! `Model`/`DecoderLayer`/`forward_token` from `model.rs`, same SwiGLU MLP,
//! same GQA, same full RoPE — so this module only builds the layers; it does
//! not re-implement the forward pass.
//!
//! Three things differ from Qwen2, all of them handled here or by a `None`
//! default that leaves the Qwen2 path untouched:
//!
//! 1. `head_dim` comes from config.json instead of `hidden_size /
//!    num_attention_heads`. For Qwen3-4B those disagree (128 vs 80), so the
//!    attention block is *wider* than the residual stream: q_proj is
//!    `[num_heads * head_dim, hidden]` = `[4096, 2560]` and o_proj is
//!    `[hidden, num_heads * head_dim]` = `[2560, 4096]`. See
//!    [`crate::config::Config::head_dim`].
//! 2. No q/k/v biases. Qwen2 reads `self_attn.{q,k,v}_proj.bias`; Qwen3
//!    checkpoints contain no `.bias` tensor at all (`attention_bias: false`),
//!    so every `Linear` here is built with `None`.
//! 3. `self_attn.q_norm` / `self_attn.k_norm`: an RMSNorm of length `head_dim`
//!    applied per head after the projection and before RoPE — the same
//!    arrangement Gemma4 uses (`gemma4_model.rs`'s `forward_token`). These are
//!    stored in [`DecoderLayer::q_norm`]/[`DecoderLayer::k_norm`], which Qwen2
//!    leaves as `None`.
//!
//! Weights are read through [`ShardedSafetensors`] rather than a single
//! `model.safetensors`, because Qwen3-4B ships as three shards plus an index.

use crate::config::Config;
use crate::model::{DecoderLayer, Model};
use crate::weights::{DenseEmbedding, DenseLinear, EmbeddingTable, Linear, ShardedSafetensors};
use crate::wpc_weights::{WpcEmbedding, WpcLinear, WpcModelData};
use crate::wpc_weights_v2::{WpcEmbeddingV2, WpcLinearV2, WpcModelDataV2};
use crate::wpc_weights_v3::{WpcEmbeddingV3, WpcLinearV3, WpcModelDataV3};
use crate::wpc_weights_v4::{WpcEmbeddingV4, WpcLinearV4, WpcModelDataV4};
use std::path::Path;

/// Build the decoder stack. `make_linear(tensor_name, out_features,
/// in_features)` supplies the backend-specific `Linear` (dense or WPC v1/v2/v3);
/// 1D tensors (the four norms) always come from `st`, since WPC only compresses
/// 2D matrices. `backend` is a label for the per-layer progress log.
fn build_layers<F>(
    st: &ShardedSafetensors,
    config: &Config,
    mut make_linear: F,
    backend: &str,
) -> Vec<DecoderLayer>
where
    F: FnMut(&str, usize, usize) -> Box<dyn Linear>,
{
    let h = config.hidden_size;
    let head_dim = config.head_dim();
    let q_dim = config.num_attention_heads * head_dim;
    let kv_dim = config.num_key_value_heads * head_dim;
    let inter = config.intermediate_size;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for l in 0..config.num_hidden_layers {
        let p = format!("model.layers.{l}");

        let q_norm = st.read_f32(&format!("{p}.self_attn.q_norm.weight"));
        let k_norm = st.read_f32(&format!("{p}.self_attn.k_norm.weight"));
        assert_eq!(
            q_norm.len(),
            head_dim,
            "layer {l}: q_norm has {} weights, expected head_dim = {head_dim}",
            q_norm.len()
        );
        assert_eq!(
            k_norm.len(),
            head_dim,
            "layer {l}: k_norm has {} weights, expected head_dim = {head_dim}",
            k_norm.len()
        );

        layers.push(DecoderLayer {
            input_layernorm: st.read_f32(&format!("{p}.input_layernorm.weight")),
            post_attention_layernorm: st.read_f32(&format!("{p}.post_attention_layernorm.weight")),
            q_norm: Some(q_norm),
            k_norm: Some(k_norm),
            q_proj: make_linear(&format!("{p}.self_attn.q_proj.weight"), q_dim, h),
            k_proj: make_linear(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h),
            v_proj: make_linear(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h),
            o_proj: make_linear(&format!("{p}.self_attn.o_proj.weight"), h, q_dim),
            gate_proj: make_linear(&format!("{p}.mlp.gate_proj.weight"), inter, h),
            up_proj: make_linear(&format!("{p}.mlp.up_proj.weight"), inter, h),
            down_proj: make_linear(&format!("{p}.mlp.down_proj.weight"), h, inter),
        });
        eprintln!("loaded layer {l}/{} ({backend})", config.num_hidden_layers);
    }
    layers
}

/// Load every weight densely from the checkpoint's safetensors (single file or
/// shards). Costs roughly `4 * parameter_count` bytes of RAM, since bf16 is
/// expanded to f32 — about 16 GB for Qwen3-4B.
pub fn load(model_dir: &Path, config: Config) -> anyhow::Result<Model> {
    let st = ShardedSafetensors::open(model_dir)?;
    let embed: Box<dyn EmbeddingTable> = Box::new(DenseEmbedding::new(
        config.vocab_size,
        config.hidden_size,
        st.read_f32("model.embed_tokens.weight"),
    ));
    let layers = build_layers(
        &st,
        &config,
        |name: &str, out, inp| -> Box<dyn Linear> {
            Box::new(DenseLinear::new(out, inp, st.read_f32(name), None))
        },
        "Qwen3 dense",
    );
    let final_norm = st.read_f32("model.norm.weight");
    Ok(Model { config, embed, layers, final_norm })
}

/// Load through the WPC v1 (VQ-codebook) backend. Norms still come from the
/// dense checkpoint in `model_dir`, as in [`Model::load_wpc`].
pub fn load_wpc(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
    let st = ShardedSafetensors::open(model_dir)?;
    let wpc = WpcModelData::open(wpc_dir)?;
    let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbedding::new(
        wpc.clone(),
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    ));
    let layers = build_layers(
        &st,
        &config,
        |name: &str, out, inp| -> Box<dyn Linear> {
            Box::new(WpcLinear::new(wpc.clone(), name, out, inp, None))
        },
        "Qwen3 WPC v1",
    );
    let final_norm = st.read_f32("model.norm.weight");
    Ok(Model { config, embed, layers, final_norm })
}

/// Load through the WPC v2 (affine 6-bit) backend.
pub fn load_wpc_v2(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
    let st = ShardedSafetensors::open(model_dir)?;
    let wpc = WpcModelDataV2::open(wpc_dir)?;
    let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV2::new(
        wpc.clone(),
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    ));
    let layers = build_layers(
        &st,
        &config,
        |name: &str, out, inp| -> Box<dyn Linear> {
            Box::new(WpcLinearV2::new(wpc.clone(), name, out, inp, None))
        },
        "Qwen3 WPC v2",
    );
    let final_norm = st.read_f32("model.norm.weight");
    Ok(Model { config, embed, layers, final_norm })
}

/// Load through the WPC v3 backend (v2's quantization, 6-bit codes bit-packed).
/// Reconstruction is identical to v2 by construction; only the reader differs.
pub fn load_wpc_v3(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
    let st = ShardedSafetensors::open(model_dir)?;
    let wpc = WpcModelDataV3::open(wpc_dir)?;
    let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV3::new(
        wpc.clone(),
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    ));
    let layers = build_layers(
        &st,
        &config,
        |name: &str, out, inp| -> Box<dyn Linear> {
            Box::new(WpcLinearV3::new(wpc.clone(), name, out, inp, None))
        },
        "Qwen3 WPC v3",
    );
    let final_norm = st.read_f32("model.norm.weight");
    Ok(Model { config, embed, layers, final_norm })
}

/// Load through the WPC v4 backend (affine 4-bit, two codes per byte).
///
/// Same decode formula as v2/v3 with 16 levels per block instead of 64: the
/// model is ~32% smaller than v3 and measurably coarser. Norms still come from
/// the dense checkpoint in `model_dir`.
pub fn load_wpc_v4(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
    let st = ShardedSafetensors::open(model_dir)?;
    let wpc = WpcModelDataV4::open(wpc_dir)?;
    let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV4::new(
        wpc.clone(),
        "model.embed_tokens.weight",
        config.vocab_size,
        config.hidden_size,
    ));
    let layers = build_layers(
        &st,
        &config,
        |name: &str, out, inp| -> Box<dyn Linear> {
            Box::new(WpcLinearV4::new(wpc.clone(), name, out, inp, None))
        },
        "Qwen3 WPC v4",
    );
    let final_norm = st.read_f32("model.norm.weight");
    Ok(Model { config, embed, layers, final_norm })
}
