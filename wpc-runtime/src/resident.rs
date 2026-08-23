use crate::config::Config;
use crate::qwen3_moe_model::Qwen3MoeModel;
use crate::sampling::{argmax_banned, banned_from_env};
use anyhow::{bail, Context, Result};
use std::path::Path;
use tokenizers::Tokenizer;

pub struct ResidentEngine {
    model: Qwen3MoeModel,
    tokenizer: Tokenizer,
    bos: Option<u32>,
    eos: Vec<u32>,
    banned: Vec<u32>,
}

impl ResidentEngine {
    pub fn load(model_dir: &Path, wpc_dir: &Path, scheme: &str) -> Result<Self> {
        if scheme != "v4" {
            bail!("resident runtime currently supports only WPC v4");
        }
        let config_path = model_dir.join("config.json");
        let config = Config::load(&config_path)?;
        if !config.is_qwen3_moe() {
            bail!("resident runtime currently supports Qwen3-MoE only");
        }
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let model = Qwen3MoeModel::load_wpc_v4(model_dir, wpc_dir, config)
            .with_context(|| "failed to load resident WPC model")?;
        let eos = match std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| v.get("eos_token_id").cloned())
        {
            Some(serde_json::Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_u64().map(|n| n as u32))
                .collect(),
            Some(serde_json::Value::Number(n)) => {
                n.as_u64().map(|x| vec![x as u32]).unwrap_or_default()
            }
            _ => Vec::new(),
        };
        let bos = std::fs::read_to_string(&config_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .and_then(|v| {
                v.get("bos_token_id")
                    .and_then(|x| x.as_u64())
                    .map(|x| x as u32)
            });
        Ok(Self {
            model,
            tokenizer,
            bos,
            eos,
            banned: banned_from_env(),
        })
    }

    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<(String, usize)> {
        let enc = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        if let Some(bos) = self.bos {
            if ids.first().copied() != Some(bos) {
                ids.insert(0, bos);
            }
        }
        let mut cache = self.model.new_cache();
        let mut next_logits = Vec::new();
        for tok in ids {
            next_logits = self.model.forward_token(tok, &mut cache);
        }
        let mut generated = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next_id = argmax_banned(&next_logits, &self.banned);
            generated.push(next_id);
            if self.eos.contains(&next_id) {
                break;
            }
            next_logits = self.model.forward_token(next_id, &mut cache);
        }
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;
        Ok((text, generated.len()))
    }

    pub fn start_session(&self) -> ResidentSession<'_> {
        ResidentSession { engine: self, cache: self.model.new_cache(), prompt_ids: Vec::new(), prompt_logits: Vec::new() }
    }
}


pub struct ResidentSession<'a> { engine: &'a ResidentEngine, cache: crate::qwen3_moe_model::MoeKvCache, prompt_ids: Vec<u32>, prompt_logits: Vec<f32> }
impl<'a> ResidentSession<'a> {
    fn reset(&mut self) -> Result<()> { self.cache.truncate(0)?; self.prompt_ids.clear(); self.prompt_logits.clear(); Ok(()) }
    pub fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<(String,usize)> {
        let enc=self.engine.tokenizer.encode(prompt,true).map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut ids=enc.get_ids().to_vec();
        if let Some(bos)=self.engine.bos { if ids.first().copied()!=Some(bos) { ids.insert(0,bos); } }
        let reusable=ids.len()>=self.prompt_ids.len() && ids[..self.prompt_ids.len()]==self.prompt_ids[..];
        if !reusable { self.reset()?; }
        let start=self.prompt_ids.len();
        for &tok in &ids[start..] { self.prompt_logits=self.engine.model.forward_token(tok,&mut self.cache); }
        self.prompt_ids=ids; if self.prompt_logits.is_empty(){ anyhow::bail!("session prompt produced no logits"); }
        let prompt_len=self.prompt_ids.len(); let saved=self.prompt_logits.clone(); let mut next=saved.clone(); let mut generated=Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens { let id=argmax_banned(&next,&self.engine.banned); generated.push(id); if self.engine.eos.contains(&id){break;} next=self.engine.model.forward_token(id,&mut self.cache); }
        self.cache.truncate(prompt_len)?; self.prompt_logits=saved;
        let text=self.engine.tokenizer.decode(&generated,true).map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?; Ok((text,generated.len()))
    }
}
