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
use crate::kv_probe::KvProbeHandle;
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
    /// `None` for full-attention layers (attention_k_eq_v): value states
    /// reuse the pre-norm/pre-RoPE k_proj output instead.
    pub v_proj: Option<Box<dyn Linear>>,
    pub o_proj: Box<dyn Linear>,
    pub gate_proj: Box<dyn Linear>,
    pub up_proj: Box<dyn Linear>,
    pub down_proj: Box<dyn Linear>,
    /// Learned per-layer scalar (checkpoint tensor `{prefix}.layer_scalar`,
    /// BF16 [1]) that rescales the ENTIRE residual stream at the end of
    /// every decoder layer (`hidden_states *= self.layer_scalar` in
    /// modeling_gemma4.py's `Gemma4TextDecoderLayer.forward`, after both
    /// the attention and MLP residual additions). Not a no-op default of
    /// 1.0 -- real trained values range ~0.05-0.52 across layers 0-47.
    /// Omitting this compounds a magnitude error through all 48 layers and
    /// was the root cause of immediate garbage output.
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
    k: Vec<Vec<f32>>, // [kv_head][position * head_dim + d]
    v: Vec<Vec<f32>>,
}

pub struct Gemma4KvCache {
    layers: Vec<LayerCache>,
    pub len: usize,
    probe: Option<KvProbeHandle>,
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
            probe: None,
        }
    }

    /// Install or remove an opt-in K/V observation hook.
    pub fn set_kv_probe(&mut self, probe: Option<KvProbeHandle>) {
        self.probe = probe;
    }
}

const LANG_PREFIX: &str = "model.language_model";

impl Gemma4Model {
    /// `model_dir` must contain the original dense `config.json` (not
    /// loaded for weights here) and `model.safetensors` (mmap'd, only 1D
    /// norm tensors are ever read out of it — see `weights::SafetensorsFile`).
    /// `wpc_dir` must contain the WPC-compressed `model.language_model.*`
    /// tensors (`model.meta`, `model.wpc`, `global_patterns.bin`,
    /// `global_residuals.bin`); multimodal tensors
    /// (`embed_vision`/`embed_audio`/`vision_embedder`) present in the same
    /// `model.meta` are simply never referenced.
    pub fn load_wpc(
        model_dir: &Path,
        wpc_dir: &Path,
        config: Gemma4Config,
    ) -> anyhow::Result<Gemma4Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelData::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbedding::new(
            wpc.clone(),
            &format!("{LANG_PREFIX}.embed_tokens.weight"),
            config.vocab_size,
            h,
        ));

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let spec = config.layer_spec(l);
            let p = format!("{LANG_PREFIX}.layers.{l}");

            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));
            let pre_feedforward_layernorm =
                st.read_f32(&format!("{p}.pre_feedforward_layernorm.weight"));
            let post_feedforward_layernorm =
                st.read_f32(&format!("{p}.post_feedforward_layernorm.weight"));
            let q_norm = st.read_f32(&format!("{p}.self_attn.q_norm.weight"));
            let k_norm = st.read_f32(&format!("{p}.self_attn.k_norm.weight"));
            let layer_scalar = st.read_f32(&format!("{p}.layer_scalar"))[0];

            let num_q_out = config.num_attention_heads * spec.head_dim;
            let kv_dim = spec.num_kv_heads * spec.head_dim;

            let q_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.self_attn.q_proj.weight"),
                num_q_out,
                h,
                None,
            ));
            let k_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
                None,
            ));
            let v_proj: Option<Box<dyn Linear>> = if spec.has_v_proj {
                Some(Box::new(WpcLinear::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    None,
                )))
            } else {
                None
            };
            let o_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                num_q_out,
                None,
            ));

            let inter = config.intermediate_size;
            let gate_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
                None,
            ));
            let up_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.mlp.up_proj.weight"),
                inter,
                h,
                None,
            ));
            let down_proj = Box::new(WpcLinear::new(
                wpc.clone(),
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
                None,
            ));

            layers.push(Gemma4Layer {
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                q_norm,
                k_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                layer_scalar,
                spec,
            });
            eprintln!(
                "loaded layer {l}/{} (Gemma4 WPC, {})",
                config.num_hidden_layers,
                if layers.last().unwrap().spec.is_full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            );
        }

        let final_norm = st.read_f32(&format!("{LANG_PREFIX}.norm.weight"));

        Ok(Gemma4Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    /// Load using WPC v2 (affine 6-bit) compressed weights.
    pub fn load_wpc_v2(
        model_dir: &Path,
        wpc_dir: &Path,
        config: Gemma4Config,
    ) -> anyhow::Result<Gemma4Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelDataV2::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbeddingV2::new(
            wpc.clone(),
            &format!("{LANG_PREFIX}.embed_tokens.weight"),
            config.vocab_size,
            h,
        ));

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let spec = config.layer_spec(l);
            let p = format!("{LANG_PREFIX}.layers.{l}");

            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));
            let pre_feedforward_layernorm =
                st.read_f32(&format!("{p}.pre_feedforward_layernorm.weight"));
            let post_feedforward_layernorm =
                st.read_f32(&format!("{p}.post_feedforward_layernorm.weight"));
            let q_norm = st.read_f32(&format!("{p}.self_attn.q_norm.weight"));
            let k_norm = st.read_f32(&format!("{p}.self_attn.k_norm.weight"));
            let layer_scalar = st.read_f32(&format!("{p}.layer_scalar"))[0];

            let num_q_out = config.num_attention_heads * spec.head_dim;
            let kv_dim = spec.num_kv_heads * spec.head_dim;

            let q_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.self_attn.q_proj.weight"),
                num_q_out,
                h,
                None,
            ));
            let k_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
                None,
            ));
            let v_proj: Option<Box<dyn Linear>> = if spec.has_v_proj {
                Some(Box::new(WpcLinearV2::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    None,
                )))
            } else {
                None
            };
            let o_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                num_q_out,
                None,
            ));

            let inter = config.intermediate_size;
            let gate_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
                None,
            ));
            let up_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.mlp.up_proj.weight"),
                inter,
                h,
                None,
            ));
            let down_proj = Box::new(WpcLinearV2::new(
                wpc.clone(),
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
                None,
            ));

            layers.push(Gemma4Layer {
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                q_norm,
                k_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                layer_scalar,
                spec,
            });
            eprintln!(
                "loaded layer {l}/{} (Gemma4 WPC v2, {})",
                config.num_hidden_layers,
                if layers.last().unwrap().spec.is_full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            );
        }

        let final_norm = st.read_f32(&format!("{LANG_PREFIX}.norm.weight"));

        Ok(Gemma4Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    /// Load using WPC v3: v2's affine quantization with the 6-bit codes
    /// bit-packed (model_v3.wpc + model_v3.meta).
    ///
    /// Reconstruction is identical to v2 by construction -- v3 blocks are a
    /// re-layout of v2 blocks, not a requantization -- so this differs from
    /// `load_wpc_v2` only in which backend reads the bytes. The file is 24%
    /// smaller, and since a token costs one full pass over the weights and the
    /// cores idle waiting for memory, most of that is off the clock too.
    pub fn load_wpc_v3(
        model_dir: &Path,
        wpc_dir: &Path,
        config: Gemma4Config,
    ) -> anyhow::Result<Gemma4Model> {
        let st_path = model_dir.join("model.safetensors");
        let st = SafetensorsFile::open(&st_path)?;
        let wpc = WpcModelDataV3::open(wpc_dir)?;

        let h = config.hidden_size;
        let embed = Box::new(WpcEmbeddingV3::new(
            wpc.clone(),
            &format!("{LANG_PREFIX}.embed_tokens.weight"),
            config.vocab_size,
            h,
        ));

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for l in 0..config.num_hidden_layers {
            let spec = config.layer_spec(l);
            let p = format!("{LANG_PREFIX}.layers.{l}");

            let input_layernorm = st.read_f32(&format!("{p}.input_layernorm.weight"));
            let post_attention_layernorm =
                st.read_f32(&format!("{p}.post_attention_layernorm.weight"));
            let pre_feedforward_layernorm =
                st.read_f32(&format!("{p}.pre_feedforward_layernorm.weight"));
            let post_feedforward_layernorm =
                st.read_f32(&format!("{p}.post_feedforward_layernorm.weight"));
            let q_norm = st.read_f32(&format!("{p}.self_attn.q_norm.weight"));
            let k_norm = st.read_f32(&format!("{p}.self_attn.k_norm.weight"));
            let layer_scalar = st.read_f32(&format!("{p}.layer_scalar"))[0];

            let num_q_out = config.num_attention_heads * spec.head_dim;
            let kv_dim = spec.num_kv_heads * spec.head_dim;

            let q_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.self_attn.q_proj.weight"),
                num_q_out,
                h,
                None,
            ));
            let k_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.self_attn.k_proj.weight"),
                kv_dim,
                h,
                None,
            ));
            let v_proj: Option<Box<dyn Linear>> = if spec.has_v_proj {
                Some(Box::new(WpcLinearV3::new(
                    wpc.clone(),
                    &format!("{p}.self_attn.v_proj.weight"),
                    kv_dim,
                    h,
                    None,
                )))
            } else {
                None
            };
            let o_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                num_q_out,
                None,
            ));

            let inter = config.intermediate_size;
            let gate_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
                None,
            ));
            let up_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.mlp.up_proj.weight"),
                inter,
                h,
                None,
            ));
            let down_proj = Box::new(WpcLinearV3::new(
                wpc.clone(),
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
                None,
            ));

            layers.push(Gemma4Layer {
                input_layernorm,
                post_attention_layernorm,
                pre_feedforward_layernorm,
                post_feedforward_layernorm,
                q_norm,
                k_norm,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                gate_proj,
                up_proj,
                down_proj,
                layer_scalar,
                spec,
            });
            eprintln!(
                "loaded layer {l}/{} (Gemma4 WPC v3, {})",
                config.num_hidden_layers,
                if layers.last().unwrap().spec.is_full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            );
        }

        let final_norm = st.read_f32(&format!("{LANG_PREFIX}.norm.weight"));

        Ok(Gemma4Model {
            config,
            embed,
            layers,
            final_norm,
        })
    }

    pub fn new_cache(&self) -> Gemma4KvCache {
        let specs: Vec<LayerSpec> = (0..self.config.num_hidden_layers)
            .map(|l| self.config.layer_spec(l))
            .collect();
        Gemma4KvCache::new(&specs)
    }

    /// Run one token through the full stack, updating `cache` in place, and
    /// return the (softcapped) logits over the vocabulary for the *next*
    /// token.
    pub fn forward_token(&self, token_id: u32, cache: &mut Gemma4KvCache) -> Vec<f32> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let pos = cache.len;
        let eps = cfg.rms_norm_eps as f32;

        let mut residual = vec![0.0f32; h];
        self.embed.embed(token_id, &mut residual);
        // Gemma4TextScaledWordEmbedding: multiply by sqrt(hidden_size) right
        // after the lookup (lm_head/logits path does NOT rescale — only the
        // input embedding lookup does).
        let embed_scale = (h as f32).sqrt();
        for x in residual.iter_mut() {
            *x *= embed_scale;
        }

        for (li, layer) in self.layers.iter().enumerate() {
            let spec = &layer.spec;
            let head_dim = spec.head_dim;
            let n_rep = num_heads / spec.num_kv_heads;

            // --- attention block ---
            let mut normed = vec![0.0f32; h];
            rms_norm(&residual, &layer.input_layernorm, eps, &mut normed);

            let mut q = vec![0.0f32; num_heads * head_dim];
            layer.q_proj.matvec(&normed, &mut q);
            // q_norm + RoPE, per head.
            for hd in 0..num_heads {
                let slice = &mut q[hd * head_dim..(hd + 1) * head_dim];
                let mut tmp = vec![0.0f32; head_dim];
                rms_norm(slice, &layer.q_norm, eps, &mut tmp);
                slice.copy_from_slice(&tmp);
                apply_rope_partial(slice, pos, spec.rope_theta, spec.rotary_pairs);
            }

            // Raw (pre-norm, pre-RoPE) k_proj output. For full-attention
            // layers this doubles as the value source (attention_k_eq_v).
            let kv_dim = spec.num_kv_heads * head_dim;
            let mut k_raw = vec![0.0f32; kv_dim];
            layer.k_proj.matvec(&normed, &mut k_raw);

            let mut v = vec![0.0f32; kv_dim];
            if let Some(v_proj) = &layer.v_proj {
                v_proj.matvec(&normed, &mut v);
            } else {
                v.copy_from_slice(&k_raw);
            }
            // v_norm: RMS-normalize (no learned weight), no RoPE.
            for hd in 0..spec.num_kv_heads {
                let slice = &mut v[hd * head_dim..(hd + 1) * head_dim];
                let mut tmp = vec![0.0f32; head_dim];
                rms_norm_no_weight(slice, eps, &mut tmp);
                slice.copy_from_slice(&tmp);
            }

            // k_norm + RoPE, per kv head, on the (now norm-consumed) k_raw buffer.
            let mut k = k_raw;
            for hd in 0..spec.num_kv_heads {
                let slice = &mut k[hd * head_dim..(hd + 1) * head_dim];
                let mut tmp = vec![0.0f32; head_dim];
                rms_norm(slice, &layer.k_norm, eps, &mut tmp);
                slice.copy_from_slice(&tmp);
                apply_rope_partial(slice, pos, spec.rope_theta, spec.rotary_pairs);
            }

            let probe = cache.probe.as_ref().cloned();
            let lc = &mut cache.layers[li];
            for hd in 0..spec.num_kv_heads {
                lc.k[hd].extend_from_slice(&k[hd * head_dim..(hd + 1) * head_dim]);
                lc.v[hd].extend_from_slice(&v[hd * head_dim..(hd + 1) * head_dim]);
            }
            if let Some(probe) = probe.as_ref() {
                for hd in 0..spec.num_kv_heads {
                    probe.observe(
                        li,
                        pos,
                        hd,
                        &k[hd * head_dim..(hd + 1) * head_dim],
                        &v[hd * head_dim..(hd + 1) * head_dim],
                    );
                }
            }
            let seq_len = pos + 1;
            // Gemma4TextAttention.scaling = 1.0 explicitly (NOT 1/sqrt(head_dim));
            // q_norm/k_norm already bound the dot-product magnitude. Verified
            // directly in modeling_gemma4.py, not assumed.
            let scale = 1.0f32;
            let t_start = match spec.sliding_window {
                Some(w) => pos.saturating_sub(w.saturating_sub(1)),
                None => 0,
            };

            let mut attn_out = vec![0.0f32; num_heads * head_dim];
            attn_out
                .par_chunks_mut(head_dim)
                .enumerate()
                .for_each(|(qh, out_slice)| {
                    let kv_head = qh / n_rep;
                    let q_head = &q[qh * head_dim..(qh + 1) * head_dim];
                    let k_cache = &lc.k[kv_head];
                    let v_cache = &lc.v[kv_head];

                    let n_scored = seq_len - t_start;
                    let mut scores = vec![0.0f32; n_scored];
                    for (si, t) in (t_start..seq_len).enumerate() {
                        let k_t = &k_cache[t * head_dim..(t + 1) * head_dim];
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_head[d] * k_t[d];
                        }
                        scores[si] = dot * scale;
                    }
                    softmax_inplace(&mut scores);

                    for x in out_slice.iter_mut().take(head_dim) {
                        *x = 0.0;
                    }
                    for (si, t) in (t_start..seq_len).enumerate() {
                        let v_t = &v_cache[t * head_dim..(t + 1) * head_dim];
                        let w = scores[si];
                        for d in 0..head_dim {
                            out_slice[d] += w * v_t[d];
                        }
                    }
                });

            let mut attn_proj = vec![0.0f32; h];
            layer.o_proj.matvec(&attn_out, &mut attn_proj);
            let mut attn_normed = vec![0.0f32; h];
            rms_norm(
                &attn_proj,
                &layer.post_attention_layernorm,
                eps,
                &mut attn_normed,
            );
            for i in 0..h {
                residual[i] += attn_normed[i];
            }

            // --- MLP block (GeGLU-tanh sandwich) ---
            let mut normed2 = vec![0.0f32; h];
            rms_norm(
                &residual,
                &layer.pre_feedforward_layernorm,
                eps,
                &mut normed2,
            );

            let inter = cfg.intermediate_size;
            let mut gate = vec![0.0f32; inter];
            let mut up = vec![0.0f32; inter];
            layer.gate_proj.matvec(&normed2, &mut gate);
            layer.up_proj.matvec(&normed2, &mut up);
            const SQRT_2_OVER_PI: f32 = 0.797_884_6;
            for i in 0..inter {
                let g = gate[i];
                let gelu_tanh =
                    0.5 * g * (1.0 + (SQRT_2_OVER_PI * (g + 0.044715 * g * g * g)).tanh());
                gate[i] = gelu_tanh * up[i];
            }
            let mut mlp_out = vec![0.0f32; h];
            layer.down_proj.matvec(&gate, &mut mlp_out);
            let mut mlp_normed = vec![0.0f32; h];
            rms_norm(
                &mlp_out,
                &layer.post_feedforward_layernorm,
                eps,
                &mut mlp_normed,
            );
            for i in 0..h {
                residual[i] += mlp_normed[i];
            }

            // Learned per-layer output gate (checkpoint `{prefix}.layer_scalar`):
            // rescales the WHOLE residual stream, not just this layer's delta.
            // See modeling_gemma4.py `Gemma4TextDecoderLayer.forward`:
            // `hidden_states *= self.layer_scalar` after both residual adds.
            for x in residual.iter_mut().take(h) {
                *x *= layer.layer_scalar;
            }
        }

        cache.len += 1;

        let mut final_normed = vec![0.0f32; h];
        rms_norm(&residual, &self.final_norm, eps, &mut final_normed);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.embed.logits(&final_normed, &mut logits);

        if let Some(cap) = cfg.final_logit_softcapping {
            let cap = cap as f32;
            for l in logits.iter_mut() {
                *l = cap * (*l / cap).tanh();
            }
        }
        logits
    }
}
