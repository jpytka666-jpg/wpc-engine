use anyhow::{bail, Context, Result};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;
use wpc_runtime::config::Config;
use wpc_runtime::qwen3_moe_model::Qwen3MoeModel;
use wpc_runtime::sampling::{argmax_banned, banned_from_env};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)] model: PathBuf,
    #[arg(long)] wpc: PathBuf,
    #[arg(long, default_value = "v4")] scheme: String,
    #[arg(long, default_value_t = 120)] max_tokens: usize,
}

#[derive(Deserialize)]
struct Request {
    prompt: String,
    #[serde(default)] max_tokens: Option<usize>,
}

#[derive(Serialize)]
struct Response {
    text: String,
    generated_tokens: usize,
}

fn bos_id(model_dir: &Path) -> Option<u32> {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("bos_token_id").and_then(|x| x.as_u64()).map(|x| x as u32))
}

fn eos_ids(model_dir: &Path) -> Vec<u32> {
    std::fs::read_to_string(model_dir.join("config.json"))
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| v.get("eos_token_id").cloned())
        .map(|v| match v {
            serde_json::Value::Array(a) => a.iter().filter_map(|x| x.as_u64().map(|n| n as u32)).collect(),
            serde_json::Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            _ => Vec::new(),
        })
        .unwrap_or_default()
}

fn encode(tokenizer: &Tokenizer, prompt: &str, bos: Option<u32>) -> Result<Vec<u32>> {
    let enc = tokenizer.encode(prompt, true).map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if let Some(bos) = bos {
        if ids.first().copied() != Some(bos) {
            ids.insert(0, bos);
        }
    }
    Ok(ids)
}

fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String> {
    tokenizer.decode(ids, true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.scheme != "v4" {
        bail!("resident runtime currently supports only --scheme v4");
    }

    let config_path = args.model.join("config.json");
    let config = Config::load(&config_path)?;
    if config.model_type.as_deref() != Some("qwen3_moe") {
        bail!("wpc-resident currently supports Qwen3-MoE only");
    }

    let tokenizer = Tokenizer::from_file(args.model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

    eprintln!("loading resident Qwen3-MoE WPC v4 model once...");
    let model = Qwen3MoeModel::load_wpc_v4(&args.model, &args.wpc, config)
        .with_context(|| "failed to load resident WPC model")?;
    eprintln!("resident model ready; weights remain loaded across requests");

    let eos = eos_ids(&args.model);
    let banned = banned_from_env();
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let req: Request = serde_json::from_str(&line).context("invalid resident request JSON")?;
        let prompt_ids = encode(&tokenizer, &req.prompt, bos_id(&args.model))?;
        let mut cache = model.new_cache();
        let mut next_logits = Vec::new();
        for tok in prompt_ids {
            next_logits = model.forward_token(tok, &mut cache);
        }

        let limit = req.max_tokens.unwrap_or(args.max_tokens);
        let mut generated = Vec::with_capacity(limit);
        for _ in 0..limit {
            let next_id = argmax_banned(&next_logits, &banned);
            generated.push(next_id);
            if eos.contains(&next_id) { break; }
            next_logits = model.forward_token(next_id, &mut cache);
        }

        let text = decode(&tokenizer, &generated)?;
        serde_json::to_writer(&mut stdout, &Response { text, generated_tokens: generated.len() })?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
    }
    Ok(())
}
