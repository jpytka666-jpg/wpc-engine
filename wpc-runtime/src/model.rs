use crate::config::Config;
use crate::norm::{rms_norm, softmax_inplace};
use crate::rope::apply_rope;
use crate::weights::{DenseEmbedding, DenseLinear, EmbeddingTable, Linear, SafetensorsFile};
use crate::wpc_weights::{WpcEmbedding, WpcLinear, WpcModelData};
use crate::wpc_weights_v2::{WpcEmbeddingV2, WpcLinearV2, WpcModelDataV2};
use crate::wpc_weights_v3::{WpcEmbeddingV3, WpcLinearV3, WpcModelDataV3};
use rayon::prelude::*;
use std::path::Path;

pub struct DecoderLayer {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    /// Qwen3's `self_attn.q_norm.weight`: an RMSNorm of length `head_dim`,
    /// applied per head to the q_proj output, after the projection and before
    /// RoPE. `None` on Qwen2, whose checkpoints have no such tensor — the
    /// forward pass then skips the step entirely and is bit-for-bit what it
    /// was before this field existed.
    pub q_norm: Option<Vec<f32>>,
    /// Same for `self_attn.k_norm.weight`, applied per KV head.
    pub k_norm: Option<Vec<f32>>,
    pub q_proj: Box<dyn Linear>,
    pub k_proj: Box<dyn Linear>,
    pub v_proj: Box<dyn Linear>,
    pub o_proj: Box<dyn Linear>,
    pub gate_proj: Box<dyn Linear>,
    pub up_proj: Box<dyn Linear>,
    pub down_proj: Box<dyn Linear>,
}

pub struct Model {
    pub config: Config,
    pub embed: Box<dyn EmbeddingTable>,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: Vec<f32>,
}

/// Per-layer KV cache. `k[kv_head]` / `v[kv_head]` are position-major,
/// each position contributing `head_dim` contiguous floats.
struct LayerCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

pub struct KvCache {
    layers: Vec<LayerCache>,
    pub len: usize, // number of positions appended so far
}

impl KvCache {
    fn new(num_layers: usize, num_kv_heads: usize) -> Self {
        KvCache {
            layers: (0..num_layers)
                .map(|_| LayerCache {
                    k: vec![Vec::new(); num_kv_heads],
                    v: vec![Vec::new(); num_kv_heads],
                })
                .collect(),
            len: 0,
        }
    }
}

impl Model {
    pub fn load(model_dir: &Path, config: Config) -> anyhow::Result<Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;

        let h = config.hidden_size;
        let embed_table = st.read_f32("model.embed_tokens.weight");
        let embed = Box::new(DenseEmbedding::new(config.vocab_size, h, embed_table));

        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let _ = n_rep; // used in forward, silence unused in some build configs

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));

            let q_w = st.read_f32(&format!("{p}.self_attn.q_proj.weight"));
            let q_b = st.read_f32(&format!("{p}.self_attn.q_proj.bias"));
            let k_w = st.read_f32(&format!("{p}.self_attn.k_proj.weight"));
            let k_b = st.read_f32(&format!("{p}.self_attn.k_proj.bias"));
            let v_w = st.read_f32(&format!("{p}.self_attn.v_proj.weight"));
            let v_b = st.read_f32(&format!("{p}.self_attn.v_proj.bias"));
            let o_w = st.read_f32(&format!("{p}.self_attn.o_proj.weight"));

            let gate_w = st.read_f32(&format!("{p}.mlp.gate_proj.weight"));
            let up_w = st.read_f32(&format!("{p}.mlp.up_proj.weight"));
            let down_w = st.read_f32(&format!("{p}.mlp.down_proj.weight"));

            let intermediate = config.intermediate_size;

            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                q_norm: None,
                k_norm: None,
                q_proj: Box::new(DenseLinear::new(
                    config.num_attention_heads * head_dim,
                    h,
                    q_w,
                    Some(q_b),
                )),
                k_proj: Box::new(DenseLinear::new(kv_dim, h, k_w, Some(k_b))),
                v_proj: Box::new(DenseLinear::new(kv_dim, h, v_w, Some(v_b))),
                o_proj: Box::new(DenseLinear::new(
                    h,
                    config.num_attention_heads * head_dim,
                    o_w,
                    None,
                )),
                gate_proj: Box::new(DenseLinear::new(intermediate, h, gate_w, None)),
                up_proj: Box::new(DenseLinear::new(intermediate, h, up_w, None)),
                down_proj: Box::new(DenseLinear::new(h, intermediate, down_w, None)),
            });
            eprintln!("loaded layer {l}/{}", config.num_hidden_layers);
        }

        let final_norm = st.read_f32("model.norm.weight");

        Ok(Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    /// Load using the WPC-compressed weight backend for every Linear layer
    /// and the embedding/lm_head table. 1D tensors (norms, biases) are not
    /// part of the WPC compression (wpc-compiler only compresses 2D
    /// matrices >= 4096 elements), so those are still read from the
    /// original dense `model.safetensors` in `model_dir`. `wpc_dir` must
    /// contain `model.meta`, `model.wpc`, `global_patterns.bin`, and
    /// `global_residuals.bin` as written by wpc-full-compiler.
    pub fn load_wpc(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelData::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbedding::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            h,
        ));

        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let _ = n_rep; // used in forward, silence unused in some build configs

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));

            let q_b = st.read_f32(&format!("{p}.self_attn.q_proj.bias"));
            let k_b = st.read_f32(&format!("{p}.self_attn.k_proj.bias"));
            let v_b = st.read_f32(&format!("{p}.self_attn.v_proj.bias"));

            let intermediate = config.intermediate_size;
            let num_attn_out = config.num_attention_heads * head_dim;

            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                q_norm: None,
                k_norm: None,
                q_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.q_proj.weight"),
                    num_attn_out,
                    h,
                    Some(q_b),
                )),
                k_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    h,
                    Some(k_b),
                )),
                v_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    Some(v_b),
                )),
                o_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    num_attn_out,
                    None,
                )),
                gate_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.mlp.gate_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                up_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.mlp.up_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                down_proj: Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    intermediate,
                    None,
                )),
            });
            eprintln!("loaded layer {l}/{} (WPC)", config.num_hidden_layers);
        }

        let final_norm = st.read_f32("model.norm.weight");

        Ok(Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    /// Load using the WPC v2-compressed weight backend for every Linear layer
    /// and the embedding/lm_head table. Similar to `load_wpc`, but using the
    /// simpler affine 6-bit quantization format (model_v2.wpc + model_v2.meta).
    pub fn load_wpc_v2(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelDataV2::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbeddingV2::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            h,
        ));

        let n_rep = config.num_attention_heads / config.num_key_value_heads;
        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;
        let _ = n_rep; // used in forward, silence unused in some build configs

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));

            let q_b = st.read_f32(&format!("{p}.self_attn.q_proj.bias"));
            let k_b = st.read_f32(&format!("{p}.self_attn.k_proj.bias"));
            let v_b = st.read_f32(&format!("{p}.self_attn.v_proj.bias"));

            let intermediate = config.intermediate_size;
            let num_attn_out = config.num_attention_heads * head_dim;

            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                q_norm: None,
                k_norm: None,
                q_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.q_proj.weight"),
                    num_attn_out,
                    h,
                    Some(q_b),
                )),
                k_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    h,
                    Some(k_b),
                )),
                v_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    Some(v_b),
                )),
                o_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    num_attn_out,
                    None,
                )),
                gate_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.mlp.gate_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                up_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.mlp.up_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                down_proj: Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    intermediate,
                    None,
                )),
            });
            eprintln!("loaded layer {l}/{} (WPC v2)", config.num_hidden_layers);
        }

        let final_norm = st.read_f32("model.norm.weight");

        Ok(Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    /// Load using the WPC v3 backend: v2's quantization with the 6-bit codes
    /// bit-packed (model_v3.wpc + model_v3.meta). Reconstruction is identical
    /// to v2 by construction; the file is 24% smaller, and since decoding is
    /// memory-bandwidth bound, so is most of the read time.
    pub fn load_wpc_v3(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelDataV3::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbeddingV3::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            h,
        ));

        let head_dim = config.head_dim();
        let kv_dim = config.num_key_value_heads * head_dim;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let p = format!("model.layers.{l}");
            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));

            let q_b = st.read_f32(&format!("{p}.self_attn.q_proj.bias"));
            let k_b = st.read_f32(&format!("{p}.self_attn.k_proj.bias"));
            let v_b = st.read_f32(&format!("{p}.self_attn.v_proj.bias"));

            let intermediate = config.intermediate_size;
            let num_attn_out = config.num_attention_heads * head_dim;

            layers.push(DecoderLayer {
                input_layernorm,
                post_attention_layernorm,
                q_norm: None,
                k_norm: None,
                q_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.q_proj.weight"),
                    num_attn_out,
                    h,
                    Some(q_b),
                )),
                k_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.k_proj.weight"),
                    kv_dim,
                    h,
                    Some(k_b),
                )),
                v_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    Some(v_b),
                )),
                o_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    num_attn_out,
                    None,
                )),
                gate_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.mlp.gate_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                up_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.mlp.up_proj.weight"),
                    intermediate,
                    h,
                    None,
                )),
                down_proj: Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    intermediate,
                    None,
                )),
            });
            eprintln!("loaded layer {l}/{} (WPC v3)", config.num_hidden_layers);
        }

        let final_norm = st.read_f32("model.norm.weight");

        Ok(Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
        )
    }

    /// Run one token through the full stack, updating `cache` in place, and
    /// return the logits over the vocabulary for the *next* token.
    pub fn forward_token(&self, token_id: u32, cache: &mut KvCache) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let n_rep = num_heads / num_kv_heads;
        let pos = cache.len;
        let eps = cfg.rms_norm_eps as f32;

        let mut residual = vec![0.0f32; h];
        self.embed.embed(token_id, &mut residual);

        for (li, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            let mut normed = vec![0.0f32; h];
            rms_norm(&residual, &layer.input_layernorm, eps, &mut normed);

            let mut q = vec![0.0f32; num_heads * head_dim];
            let mut k = vec![0.0f32; num_kv_heads * head_dim];
            let mut v = vec![0.0f32; num_kv_heads * head_dim];
            layer.q_proj.matvec(&normed, &mut q);
            layer.k_proj.matvec(&normed, &mut k);
            layer.v_proj.matvec(&normed, &mut v);

            // Qwen3 only: RMSNorm each head's q/k vector (length head_dim)
            // after the projection and before RoPE, exactly as Gemma4 does.
            // Qwen2 carries `None` here and drops straight through to RoPE.
            if let Some(q_norm) = &layer.q_norm {
                let mut tmp = vec![0.0f32; head_dim];
                for hd in 0..num_heads {
                    let slice = &mut q[hd * head_dim..(hd + 1) * head_dim];
                    rms_norm(slice, q_norm, eps, &mut tmp);
                    slice.copy_from_slice(&tmp);
                }
            }
            if let Some(k_norm) = &layer.k_norm {
                let mut tmp = vec![0.0f32; head_dim];
                for hd in 0..num_kv_heads {
                    let slice = &mut k[hd * head_dim..(hd + 1) * head_dim];
                    rms_norm(slice, k_norm, eps, &mut tmp);
                    slice.copy_from_slice(&tmp);
                }
            }

            for hd in 0..num_heads {
                apply_rope(
                    &mut q[hd * head_dim..(hd + 1) * head_dim],
                    pos,
                    cfg.rope_theta,
                );
            }
            for hd in 0..num_kv_heads {
                apply_rope(
                    &mut k[hd * head_dim..(hd + 1) * head_dim],
                    pos,
                    cfg.rope_theta,
                );
            }

            let lc = &mut cache.layers[li];
            for hd in 0..num_kv_heads {
                lc.k[hd].extend_from_slice(&k[hd * head_dim..(hd + 1) * head_dim]);
                lc.v[hd].extend_from_slice(&v[hd * head_dim..(hd + 1) * head_dim]);
            }
            let seq_len = pos + 1;
            let scale = 1.0f32 / (head_dim as f32).sqrt();

            let mut attn_out = vec![0.0f32; num_heads * head_dim];
            attn_out
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(qh, out_slice)| {
                    let kv_head = qh / n_rep;
                    let q_head = &q[qh * head_dim..(qh + 1) * head_dim];
                    let k_cache = &lc.k[kv_head];
                    let v_cache = &lc.v[kv_head];

                    let mut scores = vec![0.0f32; seq_len];
                    for t in 0..seq_len {
                        let k_t = &k_cache[t * head_dim..(t + 1) * head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_head[d] * k_t[d];
                        }
                        scores[t] = dot * scale;
                    }
                    softmax_inplace(&mut scores);

                    for x in out_slice.iter_mut().take(head_dim) {
                        *x = 0.0;
                    }
                    for t in 0..seq_len {
                        let v_t = &v_cache[t * head_dim..(t + 1) * head_dim];
                        let w = scores[t];
                        for d in 0..head_dim {
                            out_slice[d] += w * v_t[d];
                        }
                    }
                });

            let mut attn_proj = vec![0.0f32; h];
            layer.o_proj.matvec(&attn_out, &mut attn_proj);
            for i in 0..h {
                residual[i] += attn_proj[i];
            }

            // --- MLP block (SwiGLU) ---
            let mut normed2 = vec![0.0f32; h];
            rms_norm(
                &residual,
                &layer.post_attention_layernorm,
                eps,
                &mut normed2,
            );

            let inter = cfg.intermediate_size;
            let mut gate = vec![0.0f32; inter];
            let mut up = vec![0.0f32; inter];
            layer.gate_proj.matvec(&normed2, &mut gate);
            layer.up_proj.matvec(&normed2, &mut up);
            for i in 0..inter {
                let g = gate[i];
                let silu = g / (1.0 + (-g).exp());
                gate[i] = silu * up[i];
            }
            let mut mlp_out = vec![0.0f32; h];
            layer.down_proj.matvec(&gate, &mut mlp_out);
            for i in 0..h {
                residual[i] += mlp_out[i];
            }
        }

        cache.len += 1;

        let mut final_normed = vec![0.0f32; h];
        rms_norm(&residual, &self.final_norm, eps, &mut final_normed);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.embed.logits(&final_normed, &mut logits);
        logits
    }
}
