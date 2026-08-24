//! Qwen3-MoE (`model_type: "qwen3_moe"`), e.g. Qwen3-Coder-30B-A3B-Instruct.
//!
//! 2026-08-24 — maintained by ChatGPT in this session.
//! Reason: activate the existing read-only KV probe at the resident-cache
//! boundary for real Qwen statistics. No model-file write path is introduced.
//!
//! Deliberately a separate model type rather than a variant of [`crate::model::Model`],
//! the same way `gemma4_model` is: nothing here can change the behaviour of the
//! dense Qwen2/Qwen3 path, which is already working and compressed-model-tested.
//!
//! Attention is *identical* to dense Qwen3 — explicit `head_dim`, per-head
//! `q_norm`/`k_norm` before RoPE, no q/k/v biases, GQA, full RoPE. Only two
//! things differ, and both are in this file:
//!
//! 1. **The MLP is sparse.** A dense layer holds one `mlp.{gate,up,down}_proj`.
//!    A MoE layer holds a router (`mlp.gate.weight`, `[num_experts, hidden]`)
//!    and `num_experts` independent expert MLPs
//!    (`mlp.experts.{e}.{gate,up,down}_proj.weight`, each `[moe_intermediate, hidden]`).
//!    Per token the router scores all experts, the top `num_experts_per_tok`
//!    run, and their outputs are summed weighted by the routing probabilities.
//!    On Qwen3-Coder-30B-A3B that is 8 of 128 — which is why a 30B model moves
//!    roughly a 3B model's worth of weights per token.
//!
//! 2. **`lm_head` is its own tensor.** `tie_word_embeddings` is false here, so
//!    the output projection is a real `lm_head.weight` and not the embedding
//!    table reused, as it is on Gemma4 and small Qwen2 checkpoints.
//!
//! Routing follows the reference implementation exactly: softmax over *all*
//! experts first, then take top-k, then (when `norm_topk_prob`) renormalise the
//! k kept weights to sum to 1. Softmaxing after the top-k instead would give
//! different weights and is the easy mistake here.
//!
//! **The router is never compressed.** Every other matrix in the model computes
//! a value, where quantization error averages out across thousands of terms.
//! `mlp.gate.weight` instead makes a *discrete choice*: when two experts score
//! within quantization noise of each other, a 6-bit rounding decides which of
//! them runs, and the token takes a different computation path — an error no
//! downstream arithmetic can dilute. All 48 routers together are 12.6M
//! parameters, 0.04% of the model, so keeping them exact costs about 50 MB
//! against a ~24 GB compressed model. This is the same lesson as Gemma4's tied
//! embedding table: a small, decisive tensor quantized like a big, averaging
//! one. `wpc-full-compiler` skips these tensors; they are read from the dense
//! checkpoint in `--model`, alongside the 1D norms.

use crate::config::Config;
use crate::kv_probe::{KvProbeHandle, StatsKvProbe};
use crate::norm::{rms_norm, softmax_inplace};
use crate::rope::apply_rope;
use crate::weights::{DenseEmbedding, DenseLinear, EmbeddingTable, Linear, ShardedSafetensors};
use crate::wpc_weights_v2::{WpcEmbeddingV2, WpcLinearV2, WpcModelDataV2};
use crate::wpc_weights_v3::{WpcEmbeddingV3, WpcLinearV3, WpcModelDataV3};
use crate::wpc_weights_v4::{WpcEmbeddingV4, WpcLinearV4, WpcModelDataV4};
use rayon::prelude::*;
use std::path::Path;

// -----------------------------------------------------------------------------
// KV cache
// -----------------------------------------------------------------------------

/// Private per-layer key/value store. Separate from `model::KvCache` only
/// because that one's fields are module-private; the layout is the same.
struct LayerCache {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
}

pub struct MoeKvCache {
    layers: Vec<LayerCache>,
    head_dim: usize,
    /// Number of positions appended so far.
    pub len: usize,
    probe: Option<KvProbeHandle>,
    expected_vocab_size: usize,
}

impl MoeKvCache {
    fn new(num_layers: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        MoeKvCache {
            layers: (0..num_layers)
                .map(|_| LayerCache {
                    k: vec![Vec::new(); num_kv_heads],
                    v: vec![Vec::new(); num_kv_heads],
                })
                .collect(),
            head_dim,
            len: 0,
            probe: None,
            expected_vocab_size: 0,
        }
    }

    /// Install or remove the optional read-only K/V observation hook.
    pub fn set_kv_probe(&mut self, probe: Option<KvProbeHandle>) {
        self.probe = probe;
    }

    pub fn truncate(&mut self, len: usize) -> anyhow::Result<()> {
        if len > self.len {
            anyhow::bail!("cannot extend KV cache with truncate");
        }
        let elems = len
            .checked_mul(self.head_dim)
            .ok_or_else(|| anyhow::anyhow!("KV truncate size overflow"))?;
        for layer in &mut self.layers {
            for head in &mut layer.k {
                head.truncate(elems);
            }
            for head in &mut layer.v {
                head.truncate(elems);
            }
        }
        self.len = len;
        Ok(())
    }

    /// Restore a complete resident-KV snapshot plus the final logits for that prefix.
    /// KV alone cannot reconstruct the final residual state after a process restart,
    /// so the matching logits sidecar is required for exact continuation.
    pub fn restore_from_files(&mut self, snapshot_path: impl AsRef<Path>, logits_path: impl AsRef<Path>) -> anyhow::Result<Vec<f32>> {
        let snapshot = crate::kv_probe::KvSnapshot::read_from_path(snapshot_path)?;
        anyhow::ensure!(!snapshot.truncated, "KV snapshot is truncated and cannot be restored");
        anyhow::ensure!(!snapshot.records.is_empty(), "KV snapshot contains no records");
        let num_layers = self.layers.len();
        let num_kv_heads = self.layers.first().map(|l| l.k.len()).unwrap_or(0);
        let per_position = num_layers.checked_mul(num_kv_heads).ok_or_else(|| anyhow::anyhow!("KV restore count overflow"))?;
        let sequence_length = snapshot.records.iter().map(|r| r.position as usize + 1).max().unwrap_or(0);
        anyhow::ensure!(sequence_length > 0, "KV snapshot has zero sequence length");
        anyhow::ensure!(snapshot.records.len() == per_position * sequence_length, "KV snapshot record count does not match runtime geometry");
        for layer in &mut self.layers {
            for head in &mut layer.k { head.clear(); head.resize(sequence_length * self.head_dim, 0.0); }
            for head in &mut layer.v { head.clear(); head.resize(sequence_length * self.head_dim, 0.0); }
        }
        let mut seen = std::collections::HashSet::with_capacity(snapshot.records.len());
        for record in snapshot.records {
            let layer = record.layer as usize; let head = record.kv_head as usize; let position = record.position as usize;
            anyhow::ensure!(layer < num_layers, "KV snapshot layer out of range");
            anyhow::ensure!(head < num_kv_heads, "KV snapshot head out of range");
            anyhow::ensure!(position < sequence_length, "KV snapshot position out of range");
            anyhow::ensure!(record.key.len() == self.head_dim && record.value.len() == self.head_dim, "KV snapshot dimension mismatch");
            anyhow::ensure!(seen.insert((layer, head, position)), "duplicate KV snapshot record");
            let start = position * self.head_dim;
            self.layers[layer].k[head][start..start+self.head_dim].copy_from_slice(&record.key);
            self.layers[layer].v[head][start..start+self.head_dim].copy_from_slice(&record.value);
        }
        anyhow::ensure!(seen.len() == per_position * sequence_length, "KV snapshot has missing records");
        self.len = sequence_length;
        let bytes = std::fs::read(logits_path)?;
        anyhow::ensure!(bytes.len() % 4 == 0, "logits sidecar length is not f32-aligned");
        let logits: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        anyhow::ensure!(logits.len() == self.expected_vocab_size(), "logits sidecar vocabulary size mismatch");
        Ok(logits)
    }

}

// -----------------------------------------------------------------------------
// Layers
// -----------------------------------------------------------------------------

/// One expert: a standard SwiGLU MLP of width `moe_intermediate_size`.
pub struct Expert {
    pub gate_proj: Box<dyn Linear>,
    pub up_proj: Box<dyn Linear>,
    pub down_proj: Box<dyn Linear>,
}

pub struct MoeLayer {
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub q_proj: Box<dyn Linear>,
    pub k_proj: Box<dyn Linear>,
    pub v_proj: Box<dyn Linear>,
    pub o_proj: Box<dyn Linear>,
    /// `mlp.gate.weight`: scores every expert for this token.
    pub router: Box<dyn Linear>,
    pub experts: Vec<Expert>,
}

pub struct Qwen3MoeModel {
    pub config: Config,
    pub embed: Box<dyn EmbeddingTable>,
    /// Untied output projection (`lm_head.weight`).
    pub lm_head: Box<dyn Linear>,
    pub layers: Vec<MoeLayer>,
    pub final_norm: Vec<f32>,
    top_k: usize,
    norm_topk: bool,
    moe_inter: usize,
}

/// Pull the four MoE knobs out of the config, failing loudly rather than
/// silently guessing: a wrong expert count would load garbage weights.
fn moe_dims(config: &Config) -> anyhow::Result<(usize, usize, usize, bool)> {
    let n_exp = config.num_experts.ok_or_else(|| {
        anyhow::anyhow!("config.json has no `num_experts`; not a Qwen3-MoE model")
    })?;
    let top_k = config
        .num_experts_per_tok
        .ok_or_else(|| anyhow::anyhow!("config.json has no `num_experts_per_tok`; cannot route"))?;
    let moe_inter = config.moe_intermediate_size.ok_or_else(|| {
        anyhow::anyhow!("config.json has no `moe_intermediate_size`; expert width unknown")
    })?;
    if top_k == 0 || top_k > n_exp {
        anyhow::bail!("num_experts_per_tok = {top_k} is not in 1..={n_exp}");
    }
    Ok((
        n_exp,
        top_k,
        moe_inter,
        config.norm_topk_prob.unwrap_or(true),
    ))
}

/// Build the decoder stack. `make_linear(tensor_name, out_features, in_features)`
/// supplies the backend-specific `Linear`; the four norms are 1D and always come
/// from `st`, since WPC only ever compresses 2D matrices.
fn build_layers<F>(
    st: &ShardedSafetensors,
    config: &Config,
    n_exp: usize,
    moe_inter: usize,
    mut make_linear: F,
    backend: &str,
) -> Vec<MoeLayer>
where
    F: FnMut(&str, usize, usize) -> Box<dyn Linear>,
{
    let h = config.hidden_size;
    let head_dim = config.head_dim();
    let q_dim = config.num_attention_heads * head_dim;
    let kv_dim = config.num_key_value_heads * head_dim;

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for l in 0..config.num_hidden_layers {
        let p = format!("model.layers.{l}");

        let mut experts = Vec::with_capacity(n_exp);
        for e in 0..n_exp {
            let ep = format!("{p}.mlp.experts.{e}");
            experts.push(Expert {
                gate_proj: make_linear(&format!("{ep}.gate_proj.weight"), moe_inter, h),
                up_proj: make_linear(&format!("{ep}.up_proj.weight"), moe_inter, h),
                down_proj: make_linear(&format!("{ep}.down_proj.weight"), h, moe_inter),
            });
        }

        layers.push(MoeLayer {
            input_layernorm: st.read_f32(&format!("{p}.input_layernorm.weight")),
            post_attention_layernorm: st.read_f32(&format!("{p}.post_attention_layernorm.weight")),
            q_norm: st.read_f32(&format!("{p}.self_attn.q_norm.weight")),
            k_norm: st.read_f32(&format!("{p}.self_attn.k_norm.weight")),
            q_proj: make_linear(&format!("{p}.self_attn.q_proj.weight"), q_dim, h),
            k_proj: make_linear(&format!("{p}.self_attn.k_proj.weight"), kv_dim, h),
            v_proj: make_linear(&format!("{p}.self_attn.v_proj.weight"), kv_dim, h),
            o_proj: make_linear(&format!("{p}.self_attn.o_proj.weight"), h, q_dim),
            // The router is read DENSE and never compressed — see the note at
            // the top of this file. `wpc-full-compiler` skips `mlp.gate.weight`
            // for the same reason, so it is simply not present in the .wpc.
            router: Box::new(DenseLinear::new(
                n_exp,
                h,
                st.read_f32(&format!("{p}.mlp.gate.weight")),
                None,
            )),
            experts,
        });
        eprintln!(
            "loaded layer {l}/{} ({backend}, {n_exp} experts)",
            config.num_hidden_layers
        );
    }
    layers
}

impl Qwen3MoeModel {
    pub fn load(model_dir: &Path, config: Config) -> anyhow::Result<Self> {
        let (n_exp, top_k, moe_inter, norm_topk) = moe_dims(&config)?;
        let st = ShardedSafetensors::open(model_dir)?;
        let embed: Box<dyn EmbeddingTable> = Box::new(DenseEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            st.read_f32("model.embed_tokens.weight"),
        ));
        let lm_head: Box<dyn Linear> = Box::new(DenseLinear::new(
            config.vocab_size,
            config.hidden_size,
            st.read_f32("lm_head.weight"),
            None,
        ));
        let layers = build_layers(
            &st,
            &config,
            n_exp,
            moe_inter,
            |name: &str, out, inp| -> Box<dyn Linear> {
                Box::new(DenseLinear::new(out, inp, st.read_f32(name), None))
            },
            "Qwen3-MoE dense",
        );
        let final_norm = st.read_f32("model.norm.weight");
        Ok(Qwen3MoeModel {
            config,
            embed,
            lm_head,
            layers,
            final_norm,
            top_k,
            norm_topk,
            moe_inter,
        })
    }

    pub fn load_wpc_v2(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Self> {
        let (n_exp, top_k, moe_inter, norm_topk) = moe_dims(&config)?;
        let st = ShardedSafetensors::open(model_dir)?;
        let wpc = WpcModelDataV2::open(wpc_dir)?;
        let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV2::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        ));
        let lm_head: Box<dyn Linear> = Box::new(WpcLinearV2::new(
            wpc.clone(),
            "lm_head.weight",
            config.vocab_size,
            config.hidden_size,
            None,
        ));
        let layers = build_layers(
            &st,
            &config,
            n_exp,
            moe_inter,
            |name: &str, out, inp| -> Box<dyn Linear> {
                Box::new(WpcLinearV2::new(wpc.clone(), name, out, inp, None))
            },
            "Qwen3-MoE WPC v2",
        );
        let final_norm = st.read_f32("model.norm.weight");
        Ok(Qwen3MoeModel {
            config,
            embed,
            lm_head,
            layers,
            final_norm,
            top_k,
            norm_topk,
            moe_inter,
        })
    }

    pub fn load_wpc_v3(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Self> {
        let (n_exp, top_k, moe_inter, norm_topk) = moe_dims(&config)?;
        let st = ShardedSafetensors::open(model_dir)?;
        let wpc = WpcModelDataV3::open(wpc_dir)?;
        let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV3::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        ));
        let lm_head: Box<dyn Linear> = Box::new(WpcLinearV3::new(
            wpc.clone(),
            "lm_head.weight",
            config.vocab_size,
            config.hidden_size,
            None,
        ));
        let layers = build_layers(
            &st,
            &config,
            n_exp,
            moe_inter,
            |name: &str, out, inp| -> Box<dyn Linear> {
                Box::new(WpcLinearV3::new(wpc.clone(), name, out, inp, None))
            },
            "Qwen3-MoE WPC v3",
        );
        let final_norm = st.read_f32("model.norm.weight");
        Ok(Qwen3MoeModel {
            config,
            embed,
            lm_head,
            layers,
            final_norm,
            top_k,
            norm_topk,
            moe_inter,
        })
    }

    pub fn load_wpc_v4(model_dir: &Path, wpc_dir: &Path, config: Config) -> anyhow::Result<Self> {
        let (n_exp, top_k, moe_inter, norm_topk) = moe_dims(&config)?;
        let st = ShardedSafetensors::open(model_dir)?;
        let wpc = WpcModelDataV4::open(wpc_dir)?;
        let embed: Box<dyn EmbeddingTable> = Box::new(WpcEmbeddingV4::new(
            wpc.clone(),
            "model.embed_tokens.weight",
            config.vocab_size,
            config.hidden_size,
        ));
        let lm_head: Box<dyn Linear> = Box::new(WpcLinearV4::new(
            wpc.clone(),
            "lm_head.weight",
            config.vocab_size,
            config.hidden_size,
            None,
        ));
        let layers = build_layers(
            &st,
            &config,
            n_exp,
            moe_inter,
            |name: &str, out, inp| -> Box<dyn Linear> {
                Box::new(WpcLinearV4::new(wpc.clone(), name, out, inp, None))
            },
            "Qwen3-MoE WPC v4",
        );
        let final_norm = st.read_f32("model.norm.weight");
        Ok(Qwen3MoeModel {
            config,
            embed,
            lm_head,
            layers,
            final_norm,
            top_k,
            norm_topk,
            moe_inter,
        })
    }

    pub fn new_cache(&self) -> MoeKvCache {
        let mut cache = MoeKvCache::new(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            self.config.head_dim(),
        );
        cache.expected_vocab_size = self.config.vocab_size;
        if matches!(
            std::env::var("AIONS_KV_PROBE").as_deref(),
            Ok("1") | Ok("sample")
        ) {
            cache.set_kv_probe(Some(StatsKvProbe::from_env()));
        }
        cache
    }

    fn route(&self, scores: &[f32], chosen: &mut Vec<(usize, f32)>) {
        chosen.clear();
        let mut taken = vec![false; scores.len()];
        for _ in 0..self.top_k {
            let mut best = usize::MAX;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &s) in scores.iter().enumerate() {
                if !taken[i] && s > best_v {
                    best_v = s;
                    best = i;
                }
            }
            if best == usize::MAX {
                break;
            }
            taken[best] = true;
            chosen.push((best, best_v));
        }
        if self.norm_topk {
            let sum: f32 = chosen.iter().map(|&(_, w)| w).sum();
            if sum > 0.0 {
                for entry in chosen.iter_mut() {
                    entry.1 /= sum;
                }
            }
        }
    }

    pub fn forward_token(&self, token_id: u32, cache: &mut MoeKvCache) -> Vec<f32> {
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
        let mut chosen: Vec<(usize, f32)> = Vec::with_capacity(self.top_k);

        for (li, layer) in self.layers.iter().enumerate() {
            let mut normed = vec![0.0f32; h];
            rms_norm(&residual, &layer.input_layernorm, eps, &mut normed);

            let mut q = vec![0.0f32; num_heads * head_dim];
            let mut k = vec![0.0f32; num_kv_heads * head_dim];
            let mut v = vec![0.0f32; num_kv_heads * head_dim];
            layer.q_proj.matvec(&normed, &mut q);
            layer.k_proj.matvec(&normed, &mut k);
            layer.v_proj.matvec(&normed, &mut v);

            let mut tmp = vec![0.0f32; head_dim];
            for hd in 0..num_heads {
                let slice = &mut q[hd * head_dim..(hd + 1) * head_dim];
                rms_norm(slice, &layer.q_norm, eps, &mut tmp);
                slice.copy_from_slice(&tmp);
            }
            for hd in 0..num_kv_heads {
                let slice = &mut k[hd * head_dim..(hd + 1) * head_dim];
                rms_norm(slice, &layer.k_norm, eps, &mut tmp);
                slice.copy_from_slice(&tmp);
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

            let probe = cache.probe.as_ref().cloned();
            let lc = &mut cache.layers[li];
            for hd in 0..num_kv_heads {
                lc.k[hd].extend_from_slice(&k[hd * head_dim..(hd + 1) * head_dim]);
                lc.v[hd].extend_from_slice(&v[hd * head_dim..(hd + 1) * head_dim]);
            }
            if let Some(probe) = probe.as_ref() {
                for hd in 0..num_kv_heads {
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

            let mut normed2 = vec![0.0f32; h];
            rms_norm(
                &residual,
                &layer.post_attention_layernorm,
                eps,
                &mut normed2,
            );

            let mut router_scores = vec![0.0f32; layer.experts.len()];
            layer.router.matvec(&normed2, &mut router_scores);
            softmax_inplace(&mut router_scores);
            self.route(&router_scores, &mut chosen);

            let mi = self.moe_inter;
            let mut mlp_out = vec![0.0f32; h];
            let mut gate = vec![0.0f32; mi];
            let mut up = vec![0.0f32; mi];
            let mut expert_out = vec![0.0f32; h];
            for &(e, weight) in chosen.iter() {
                let expert = &layer.experts[e];
                expert.gate_proj.matvec(&normed2, &mut gate);
                expert.up_proj.matvec(&normed2, &mut up);
                for i in 0..mi {
                    let g = gate[i];
                    let silu = g / (1.0 + (-g).exp());
                    gate[i] = silu * up[i];
                }
                expert.down_proj.matvec(&gate, &mut expert_out);
                for i in 0..h {
                    mlp_out[i] += weight * expert_out[i];
                }
            }
            for i in 0..h {
                residual[i] += mlp_out[i];
            }
        }

        cache.len += 1;

        let mut final_normed = vec![0.0f32; h];
        rms_norm(&residual, &self.final_norm, eps, &mut final_normed);

        let mut logits = vec![0.0f32; cfg.vocab_size];
        self.lm_head.matvec(&final_normed, &mut logits);
        if let Some(probe) = cache.probe.as_ref() {
            probe.record_logits(&logits);
        }
        logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_probe_can_be_disabled() {
        let mut cache = MoeKvCache::new(1, 1, 2);
        cache.set_kv_probe(None);
    }

    fn cfg_with(n_exp: usize, top_k: usize, norm: bool) -> Config {
        let json = format!(
            r#"{{
                "model_type": "qwen3_moe",
                "vocab_size": 100, "hidden_size": 8,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "intermediate_size": 16,
                "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
                "num_experts": {n_exp}, "num_experts_per_tok": {top_k},
                "moe_intermediate_size": 4, "norm_topk_prob": {norm}
            }}"#
        );
        Config::from_json_str(&json).expect("test config parses")
    }

    fn model_for(config: Config) -> Qwen3MoeModel {
        let (_, top_k, moe_inter, norm_topk) = moe_dims(&config).unwrap();
        Qwen3MoeModel {
            config,
            embed: Box::new(DenseEmbedding::new(100, 8, vec![0.0; 800])),
            lm_head: Box::new(DenseLinear::new(100, 8, vec![0.0; 800], None)),
            layers: Vec::new(),
            final_norm: vec![1.0; 8],
            top_k,
            norm_topk,
            moe_inter,
        }
    }

    #[test]
    fn cache_truncate_preserves_requested_prefix() {
        let mut c = MoeKvCache::new(1, 1, 2);
        c.layers[0].k[0] = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        c.layers[0].v[0] = vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        c.len = 3;
        c.truncate(2).unwrap();
        assert_eq!(c.len, 2);
        assert_eq!(c.layers[0].k[0], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.layers[0].v[0], vec![6.0, 5.0, 4.0, 3.0]);
        assert!(c.truncate(3).is_err());
    }

    #[test]
    fn moe_config_fields_parse() {
        let c = cfg_with(128, 8, true);
        assert!(c.is_qwen3_moe());
        assert!(c.has_qwen3_attention());
        assert_eq!(c.num_experts, Some(128));
        assert_eq!(c.num_experts_per_tok, Some(8));
        assert_eq!(c.moe_intermediate_size, Some(4));
    }

    #[test]
    fn dense_config_has_no_moe_fields() {
        let dense = r#"{
            "model_type": "qwen3", "vocab_size": 10, "hidden_size": 8,
            "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "intermediate_size": 16,
            "rope_theta": 10000.0, "rms_norm_eps": 1e-6
        }"#;
        let c = Config::from_json_str(dense).unwrap();
        assert!(!c.is_qwen3_moe());
        assert_eq!(c.num_experts, None);
        assert!(moe_dims(&c).is_err(), "dense config must not pass as MoE");
    }

    #[test]
    fn route_picks_the_highest_scores() {
        let m = model_for(cfg_with(5, 2, false));
        let mut chosen = Vec::new();
        m.route(&[0.1, 0.5, 0.2, 0.9, 0.05], &mut chosen);
        assert_eq!(chosen.len(), 2);
        assert_eq!(chosen[0].0, 3, "highest first");
        assert_eq!(chosen[1].0, 1);
    }

    #[test]
    fn route_renormalises_when_asked() {
        let m = model_for(cfg_with(5, 2, true));
        let mut chosen = Vec::new();
        m.route(&[0.1, 0.5, 0.2, 0.9, 0.05], &mut chosen);
        let sum: f32 = chosen.iter().map(|&(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-6, "top-k weights sum to 1, got {sum}");
        assert!((chosen[0].1 - 0.642_857).abs() < 1e-4);
    }

    #[test]
    fn route_leaves_weights_alone_when_not_asked() {
        let m = model_for(cfg_with(5, 2, false));
        let mut chosen = Vec::new();
        m.route(&[0.1, 0.5, 0.2, 0.9, 0.05], &mut chosen);
        assert!((chosen[0].1 - 0.9).abs() < 1e-6);
        assert!((chosen[1].1 - 0.5).abs() < 1e-6);
    }

    #[test]
    fn top_k_beyond_expert_count_is_rejected() {
        let mut c = cfg_with(4, 8, true);
        c.num_experts_per_tok = Some(8);
        c.num_experts = Some(4);
        assert!(moe_dims(&c).is_err());
    }
}
