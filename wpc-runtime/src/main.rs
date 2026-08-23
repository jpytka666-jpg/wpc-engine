use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use wpc_runtime::config::Config;
use wpc_runtime::gemma4_config::Gemma4Config;
use wpc_runtime::gemma4_model::Gemma4Model;
use wpc_runtime::model::Model;
use wpc_runtime::qwen3_model;
use wpc_runtime::qwen3_moe_model::Qwen3MoeModel;
use wpc_runtime::sampling::{argmax_banned, banned_from_env};

#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum Arch {
    /// Detect from `config.json`: presence of a top-level `text_config` key
    /// (as used by Gemma4-family multimodal configs) selects Gemma4,
    /// `model_type: "qwen3_moe"` selects Qwen3-MoE, `model_type: "qwen3"`
    /// selects Qwen3, otherwise Qwen2.
    Auto,
    Qwen2,
    Qwen3,
    /// Sparse Qwen3: a router plus `num_experts` expert MLPs per layer, of
    /// which `num_experts_per_tok` run per token (Qwen3-Coder-30B-A3B).
    Qwen3Moe,
    Gemma4,
}

/// Transformer inference engine, dense or WPC-compressed weights, for the
/// Qwen2, Qwen3 and Gemma4 text architectures.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Directory containing model.safetensors, config.json, tokenizer.json
    #[arg(long)]
    model: PathBuf,

    /// Optional directory containing WPC-compressed weights (model.wpc,
    /// model.meta, global_patterns.bin, global_residuals.bin) as written by
    /// wpc-full-compiler. When given, Linear layers and the embedding/lm_head
    /// table load through the WPC-compressed backend instead of dense
    /// safetensors; 1D tensors (norms, biases) are still read from `--model`.
    #[arg(long)]
    wpc: Option<PathBuf>,

    /// Model architecture. Defaults to auto-detecting from config.json.
    #[arg(long, value_enum, default_value_t = Arch::Auto)]
    arch: Arch,

    /// WPC compression scheme: "v3" (packed affine 6-bit, 6.25 bits/weight - recommended),
    /// "v4" (packed affine 4-bit, 4.25 bits/weight - for cards too small for v3),
    /// "v2" (byte-aligned affine, 8.25 bits/weight) or "v1" (VQ-codebook, superseded).
    /// Only used when --wpc is provided.
    #[arg(long, default_value = "v1")]
    scheme: String,

    /// Prompt text to complete.
    #[arg(long)]
    prompt: String,

    /// Number of tokens to generate.
    #[arg(long, default_value_t = 40)]
    max_tokens: usize,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let config_path = args.model.join("config.json");
    let arch = match args.arch {
        Arch::Auto => {
            if Gemma4Config::try_load(&config_path)?.is_some() {
                Arch::Gemma4
            } else {
                match Config::model_type_of(&config_path).as_deref() {
                    Some("qwen3_moe") => Arch::Qwen3Moe,
                    Some("qwen3") => Arch::Qwen3,
                    _ => Arch::Qwen2,
                }
            }
        }
        other => other,
    };
    eprintln!("architecture: {arch:?}");

    let tokenizer_path = args.model.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    match arch {
        Arch::Gemma4 => run_gemma4(&args, &config_path, &tokenizer),
        Arch::Qwen2 => run_qwen2(&args, &config_path, &tokenizer),
        Arch::Qwen3 => run_qwen3(&args, &config_path, &tokenizer),
        Arch::Qwen3Moe => run_qwen3_moe(&args, &config_path, &tokenizer),
        Arch::Auto => unreachable!("resolved above"),
    }
}

fn run_qwen2(
    args: &Args,
    config_path: &std::path::Path,
    tokenizer: &Tokenizer,
) -> anyhow::Result<()> {
    let config = Config::load(config_path)?;
    eprintln!(
        "config: hidden={} layers={} heads={} kv_heads={} vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.vocab_size
    );

    let t0 = Instant::now();
    let model = if let Some(wpc_dir) = &args.wpc {
        if args.scheme == "v3" {
            eprintln!(
                "loading model weights (WPC v3-compressed, packed) from {} (norms/biases from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Model::load_wpc_v3(&args.model, wpc_dir, config)?
        } else if args.scheme == "v2" {
            eprintln!(
                "loading model weights (WPC v2-compressed) from {} (norms/biases from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Model::load_wpc_v2(&args.model, wpc_dir, config)?
        } else {
            eprintln!(
                "loading model weights (WPC v1-compressed) from {} (norms/biases from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Model::load_wpc(&args.model, wpc_dir, config)?
        }
    } else {
        eprintln!(
            "loading model weights (dense) from {} ...",
            args.model.display()
        );
        Model::load(&args.model, config)?
    };
    eprintln!("model loaded in {:?}", t0.elapsed());

    generate(args, &model, tokenizer)
}

/// Load a Qwen3 checkpoint and generate. Qwen3 reuses the Qwen2 decoder stack
/// (`Model`/`forward_token`); only weight loading differs — explicit `head_dim`
/// from config.json, no q/k/v biases, and per-head q_norm/k_norm before RoPE.
/// See `qwen3_model.rs`.
fn run_qwen3(
    args: &Args,
    config_path: &std::path::Path,
    tokenizer: &Tokenizer,
) -> anyhow::Result<()> {
    let config = Config::load(config_path)?;
    eprintln!(
        "config: hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim(),
        config.vocab_size
    );

    let t0 = Instant::now();
    let model = if let Some(wpc_dir) = &args.wpc {
        if args.scheme == "v4" {
            eprintln!(
                "loading model weights (WPC v4-compressed, 4-bit packed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            qwen3_model::load_wpc_v4(&args.model, wpc_dir, config)?
        } else if args.scheme == "v3" {
            eprintln!(
                "loading model weights (WPC v3-compressed, packed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            qwen3_model::load_wpc_v3(&args.model, wpc_dir, config)?
        } else if args.scheme == "v2" {
            eprintln!(
                "loading model weights (WPC v2-compressed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            qwen3_model::load_wpc_v2(&args.model, wpc_dir, config)?
        } else {
            eprintln!(
                "loading model weights (WPC v1-compressed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            qwen3_model::load_wpc(&args.model, wpc_dir, config)?
        }
    } else {
        eprintln!(
            "loading model weights (dense) from {} ...",
            args.model.display()
        );
        qwen3_model::load(&args.model, config)?
    };
    eprintln!("model loaded in {:?}", t0.elapsed());

    generate(args, &model, tokenizer)
}

/// Load a Qwen3-MoE checkpoint and generate. Same attention as dense Qwen3,
/// but each layer routes to `num_experts_per_tok` of `num_experts` expert MLPs
/// and `lm_head` is untied — hence a separate model type. See
/// `qwen3_moe_model.rs`. v1 (VQ-codebook) is not offered: it is the abandoned
/// scheme and was never wired for this architecture.
fn run_qwen3_moe(
    args: &Args,
    config_path: &std::path::Path,
    tokenizer: &Tokenizer,
) -> anyhow::Result<()> {
    let config = Config::load(config_path)?;
    eprintln!(
        "config: hidden={} layers={} heads={} kv_heads={} head_dim={} vocab={} experts={:?} active={:?} expert_width={:?}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim(), config.vocab_size,
        config.num_experts, config.num_experts_per_tok, config.moe_intermediate_size
    );

    let t0 = Instant::now();
    let model = if let Some(wpc_dir) = &args.wpc {
        if args.scheme == "v4" {
            eprintln!(
                "loading MoE weights (WPC v4-compressed, 4-bit packed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Qwen3MoeModel::load_wpc_v4(&args.model, wpc_dir, config)?
        } else if args.scheme == "v3" {
            eprintln!(
                "loading MoE weights (WPC v3-compressed, packed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Qwen3MoeModel::load_wpc_v3(&args.model, wpc_dir, config)?
        } else if args.scheme == "v2" {
            eprintln!(
                "loading MoE weights (WPC v2-compressed) from {} (norms from {}) ...",
                wpc_dir.display(),
                args.model.display()
            );
            Qwen3MoeModel::load_wpc_v2(&args.model, wpc_dir, config)?
        } else {
            anyhow::bail!(
                "--scheme {} is not supported for Qwen3-MoE; use v2, v3 or v4",
                args.scheme
            );
        }
    } else {
        eprintln!(
            "loading MoE weights (dense) from {} — this needs ~4 bytes per parameter, \
             about 122 GB for a 30B model",
            args.model.display()
        );
        Qwen3MoeModel::load(&args.model, config)?
    };
    eprintln!("model loaded in {:?}", t0.elapsed());

    generate_moe(args, &model, tokenizer)
}

/// Same loop as [`generate`], against the MoE model and its own cache type.
fn generate_moe(args: &Args, model: &Qwen3MoeModel, tokenizer: &Tokenizer) -> anyhow::Result<()> {
    let prompt_ids = encode_prompt(tokenizer, &args.prompt, config_bos(&args.model))?;
    let mut cache = model.new_cache();
    let mut generated: Vec<u32> = Vec::new();
    let mut next_logits: Vec<f32> = Vec::new();

    let t1 = Instant::now();
    for &tok in &prompt_ids {
        next_logits = model.forward_token(tok, &mut cache);
    }
    eprintln!(
        "prefill ({} tokens) in {:?}",
        prompt_ids.len(),
        t1.elapsed()
    );

    let eos = config_eos_ids(&args.model);
    let banned = banned_from_env();

    let t2 = Instant::now();
    for _ in 0..args.max_tokens {
        let next_id = argmax_banned(&next_logits, &banned);
        generated.push(next_id);
        if eos.contains(&next_id) {
            eprintln!("stopped on token {next_id} (end of turn / end of text)");
            break;
        }
        next_logits = model.forward_token(next_id, &mut cache);
    }
    eprintln!("generated {} tokens in {:?}", generated.len(), t2.elapsed());

    print_result(tokenizer, &args.prompt, &generated)
}

/// Prefill the prompt, then greedily decode up to `--max-tokens`. Shared by the
/// Qwen2 and Qwen3 paths, which run the identical decoder stack.
fn generate(args: &Args, model: &Model, tokenizer: &Tokenizer) -> anyhow::Result<()> {
    let prompt_ids = encode_prompt(tokenizer, &args.prompt, config_bos(&args.model))?;
    let mut cache = model.new_cache();
    let mut generated: Vec<u32> = Vec::new();
    let mut next_logits: Vec<f32> = Vec::new();

    let t1 = Instant::now();
    for &tok in &prompt_ids {
        next_logits = model.forward_token(tok, &mut cache);
    }
    eprintln!(
        "prefill ({} tokens) in {:?}",
        prompt_ids.len(),
        t1.elapsed()
    );

    let eos = config_eos_ids(&args.model);
    let banned = banned_from_env();

    let t2 = Instant::now();
    for _ in 0..args.max_tokens {
        let next_id = argmax_banned(&next_logits, &banned);
        generated.push(next_id);
        if eos.contains(&next_id) {
            eprintln!("stopped on token {next_id} (end of turn / end of text)");
            break;
        }
        next_logits = model.forward_token(next_id, &mut cache);
    }
    eprintln!("generated {} tokens in {:?}", generated.len(), t2.elapsed());

    print_result(tokenizer, &args.prompt, &generated)
}

fn run_gemma4(
    args: &Args,
    config_path: &std::path::Path,
    tokenizer: &Tokenizer,
) -> anyhow::Result<()> {
    let config = Gemma4Config::try_load(config_path)?.ok_or_else(|| {
        anyhow::anyhow!("config.json has no `text_config` key; not a Gemma4 config")
    })?;
    eprintln!(
        "config: hidden={} layers={} heads={} kv_heads={} head_dim={} global_head_dim={} vocab={}",
        config.hidden_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.global_head_dim,
        config.vocab_size
    );

    let wpc_dir = args
        .wpc
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--wpc is required for Gemma4 (dense loading is not supported: weights don't fit in RAM)"))?;

    let t0 = Instant::now();
    let model = if args.scheme == "v3" {
        eprintln!(
            "loading model weights (WPC v3-compressed, packed) from {} (norms from {}) ...",
            wpc_dir.display(),
            args.model.display()
        );
        Gemma4Model::load_wpc_v3(&args.model, wpc_dir, config)?
    } else if args.scheme == "v2" {
        eprintln!(
            "loading model weights (WPC v2-compressed) from {} (norms from {}) ...",
            wpc_dir.display(),
            args.model.display()
        );
        Gemma4Model::load_wpc_v2(&args.model, wpc_dir, config)?
    } else {
        eprintln!(
            "loading model weights (WPC v1-compressed) from {} (norms from {}) ...",
            wpc_dir.display(),
            args.model.display()
        );
        Gemma4Model::load_wpc(&args.model, wpc_dir, config)?
    };
    eprintln!("model loaded in {:?}", t0.elapsed());

    let prompt_ids = encode_prompt(tokenizer, &args.prompt, config_bos(&args.model))?;
    let mut cache = model.new_cache();
    let mut generated: Vec<u32> = Vec::new();
    let mut next_logits: Vec<f32> = Vec::new();

    let t1 = Instant::now();
    for &tok in &prompt_ids {
        next_logits = model.forward_token(tok, &mut cache);
    }
    eprintln!(
        "prefill ({} tokens) in {:?}",
        prompt_ids.len(),
        t1.elapsed()
    );

    let mut eos = config_eos_ids(&args.model);
    if let Some(e) = model.config.eos_token_id {
        if !eos.contains(&e) {
            eos.push(e);
        }
    }
    let banned = banned_from_env();

    let t2 = Instant::now();
    for _ in 0..args.max_tokens {
        let next_id = argmax_banned(&next_logits, &banned);
        generated.push(next_id);
        if eos.contains(&next_id) {
            eprintln!("stopped on token {next_id} (end of turn / end of text)");
            break;
        }
        next_logits = model.forward_token(next_id, &mut cache);
    }
    eprintln!("generated {} tokens in {:?}", generated.len(), t2.elapsed());

    print_result(tokenizer, &args.prompt, &generated)
}

/// `bos_token_id` from the model's config, checking `text_config` first for
/// the nested Gemma4-family layout, then the top level.
fn config_bos(model_dir: &std::path::Path) -> Option<u32> {
    let text = std::fs::read_to_string(model_dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("text_config")
        .and_then(|t| t.get("bos_token_id"))
        .or_else(|| v.get("bos_token_id"))
        .and_then(|x| x.as_u64())
        .map(|x| x as u32)
}

/// Tokenize the prompt, prepending BOS when the tokenizer did not.
///
/// Whether BOS is added is a property of the tokenizer's post-processor, and
/// it differs between releases of the *same* model. google/gemma-4-12B
/// prepends `<bos>`; google/gemma-4-12B-it does not, because its chat template
/// emits `bos_token` itself. Feeding raw tokenizer output to a Gemma without
/// BOS produces pure `<pad>` -- no partial answer, no warning, just silence,
/// which reads exactly like a broken checkpoint.
///
/// So the caller states what BOS is (from config.json) and this adds it only
/// when it is genuinely missing. Passing `None`, or a prompt that already
/// starts with BOS, leaves the token stream byte-identical to before.
fn encode_prompt(
    tokenizer: &Tokenizer,
    prompt: &str,
    bos: Option<u32>,
) -> anyhow::Result<Vec<u32>> {
    let encoding = tokenizer
        .encode(prompt, true)
        .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();

    if let Some(b) = bos {
        if ids.first() != Some(&b) {
            ids.insert(0, b);
            eprintln!("prompt: tokenizer omitted BOS ({b}); prepended it");
        }
    }

    eprintln!("prompt tokens ({}): {:?}", ids.len(), ids);
    Ok(ids)
}

fn print_result(tokenizer: &Tokenizer, prompt: &str, generated: &[u32]) -> anyhow::Result<()> {
    eprintln!("generated token ids: {:?}", generated);
    let text = tokenizer
        .decode(generated, true)
        .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
    println!("{}{}", prompt, text);
    Ok(())
}

/// Collect every id that should end generation.
///
/// `eos_token_id` is a single number on base checkpoints and a list on
/// instruction-tuned ones, because those have more than one way to stop.
/// google/gemma-4-12B-it declares `[1, 106, 50]`: end-of-text, end-of-turn,
/// and "waiting for a tool result". Reading only the first means the model
/// hands the floor back and the loop keeps asking it for more -- which shows
/// up as an answer followed by unrequested rambling, not as an error.
///
/// generation_config.json is the authority and is checked first; config.json
/// (top level, then `text_config`) is the fallback for checkpoints that ship
/// without one.
fn config_eos_ids(model_dir: &std::path::Path) -> Vec<u32> {
    fn collect(v: Option<&serde_json::Value>, out: &mut Vec<u32>) {
        match v {
            Some(serde_json::Value::Number(n)) => {
                if let Some(x) = n.as_u64() {
                    out.push(x as u32);
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for e in a {
                    if let Some(x) = e.as_u64() {
                        out.push(x as u32);
                    }
                }
            }
            _ => {}
        }
    }

    let mut ids = Vec::new();

    for name in ["generation_config.json", "config.json"] {
        if let Ok(text) = std::fs::read_to_string(model_dir.join(name)) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) {
                collect(v.get("eos_token_id"), &mut ids);
                collect(
                    v.get("text_config").and_then(|t| t.get("eos_token_id")),
                    &mut ids,
                );
            }
        }
    }

    ids.sort_unstable();
    ids.dedup();
    if !ids.is_empty() {
        eprintln!("stop tokens: {ids:?}");
    }
    ids
}
