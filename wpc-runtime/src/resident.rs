/*
 * ==========================================
 * AUTHOR: M. SZUL
 * AI MODEL: Claude Opus 5
 * TIMESTAMP: 2026-08-25 23:40:00
 * REASON FOR CREATION: The agent loop spawned a fresh engine process for every turn, so
 *   each turn paid to load the weights again and to re-read the entire transcript from
 *   the beginning. Measured on Qwen3-4B over four turns: 84.3 s, 100.1 s, 138.1 s,
 *   152.2 s -- of which the reading alone was 62.9 s, 76.2 s, 113.6 s and 123.8 s. The
 *   cost grows with every turn precisely because nothing is remembered between them.
 *   `--interactive` already solved this for chat, where turn two cost 4.2 s instead of
 *   11.2 s. This module makes the same thing available to anything else that needs a
 *   multi-turn conversation with a resident model.
 * MECHANICS: Load once, keep the model and its KV cache alive, and append each new turn
 *   to the cache instead of rebuilding it. The first turn carries the system prompt and
 *   the BOS token; later turns close the assistant's previous reply and open a new user
 *   turn, which is all the cache is missing. Turn markers are taken verbatim from the
 *   chat template shipped beside the weights.
 * SYSTEM PART: WPC runtime, resident inference lane.
 * ARCHITECTURE FUNCTION: Layer 4 of AIONS_MASTER_BUILD_PLAN.md, whose exit gate reads
 *   "resident runtime survives multi-turn sessions". Serves BOTH stacks: the dense
 *   Qwen2/Qwen3 path and Qwen3-MoE. The earlier resident engine on feature/agent-gates
 *   refused anything that was not MoE, so it could not serve the small model at all;
 *   this one refuses neither, which matters because the 30B pays 58.8 s to re-read a
 *   62-token prompt and is the model that most needs to stop doing that.
 * DEPENDENCIES/LINKS: qwen3_model::load_wpc_v4 and Qwen3MoeModel::load_wpc_v4 for the
 *   weights, model::{Model, KvCache} and qwen3_moe_model::{Qwen3MoeModel, MoeKvCache}
 *   for the forward pass and the caches, sampling::Decoder for the repetition penalty
 *   that stops long answers jamming, config::Config for the geometry.
 * TECH STACK: Rust 2021, no new crates.
 * LOCAL WORKSPACE: C:\temp\aions-multiturn-2026-08-25\wpc-runtime\src\resident.rs
 * GIT COMMIT: PENDING
 * GITHUB METADATA: jpytka666-jpg/wpc-engine, branch feature/resident-multi-turn
 * ==========================================
 */

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};
use tokenizers::Tokenizer;

use crate::config::Config;
use crate::model::{KvCache, Model};
use crate::qwen3_model;
use crate::qwen3_moe_model::{MoeKvCache, Qwen3MoeModel};
use crate::sampling::Decoder;

/// The two decoder stacks this engine can hold.
///
/// They are separate types with separate cache types, so the alternative to an enum here
/// is a second engine and a second session with the same body twice. The enum keeps one
/// session API and one valve, which is what callers actually want.
enum Weights {
    Dense(Model),
    Moe(Qwen3MoeModel),
}

enum Cache {
    Dense(KvCache),
    Moe(MoeKvCache),
}

impl Weights {
    fn new_cache(&self) -> Cache {
        match self {
            Weights::Dense(m) => Cache::Dense(m.new_cache()),
            Weights::Moe(m) => Cache::Moe(m.new_cache()),
        }
    }
}

impl Cache {
    fn len(&self) -> usize {
        match self {
            Cache::Dense(c) => c.len,
            Cache::Moe(c) => c.len,
        }
    }
    fn truncate(&mut self, len: usize) -> Result<()> {
        match self {
            Cache::Dense(c) => c.truncate(len),
            Cache::Moe(c) => c.truncate(len),
        }
    }
}

/// A model loaded once and kept.
pub struct ResidentEngine {
    model: Weights,
    tokenizer: Tokenizer,
    model_dir: PathBuf,
    bos: Option<u32>,
    eos: Vec<u32>,
}

/// What one turn cost, so a caller can report it honestly instead of guessing.
#[derive(Debug, Clone, Copy)]
pub struct TurnCost {
    pub prompt_tokens: usize,
    pub prefill: Duration,
    pub generated_tokens: usize,
    pub decode: Duration,
    pub cache_positions: usize,
}

impl ResidentEngine {
    /// Load a model from WPC v4 weights and keep it resident.
    ///
    /// Both stacks are served: the dense Qwen2/Qwen3 path and the Qwen3-MoE one. Only v4
    /// for now; refusing another scheme loudly is better than silently producing a model
    /// that cannot answer.
    pub fn load(model_dir: &Path, wpc_dir: &Path, scheme: &str) -> Result<Self> {
        if scheme != "v4" {
            bail!("the resident engine currently supports WPC v4 only, not {scheme}");
        }
        let config_path = model_dir.join("config.json");
        let config = Config::load(&config_path)?;
        let is_moe = config.is_qwen3_moe();

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;

        let t0 = Instant::now();
        let model = if is_moe {
            Weights::Moe(Qwen3MoeModel::load_wpc_v4(model_dir, wpc_dir, config)?)
        } else {
            Weights::Dense(qwen3_model::load_wpc_v4(model_dir, wpc_dir, config)?)
        };
        eprintln!(
            "resident engine: {} weights loaded in {:?}",
            if is_moe { "mixture-of-experts" } else { "dense" },
            t0.elapsed()
        );

        Ok(Self {
            bos: read_bos(&config_path),
            eos: read_eos_ids(&config_path),
            model,
            tokenizer,
            model_dir: model_dir.to_path_buf(),
        })
    }

    /// Begin a conversation. The cache lives as long as the session does.
    pub fn start_session(&self) -> ResidentSession<'_> {
        ResidentSession {
            engine: self,
            cache: self.model.new_cache(),
            history: Vec::new(),
            // Phrase blocking off: the settings without it have been measured working,
            // the settings with it have not. set_decoder() turns it on for a trial.
            decoder: Decoder::new(1.0, 0.8, 0, 64, 0.0, 0.95, 0),
            turn: 0,
            clean: 0,
        }
    }

    /// Where the weights came from, for callers that want to say so.
    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Push one token through whichever stack is loaded.
    ///
    /// A mismatched pair -- a dense cache handed to an MoE model -- is a programming
    /// mistake rather than a runtime condition, so it panics with a message that says
    /// which way round it went wrong instead of returning an error nobody can act on.
    fn forward(&self, token: u32, cache: &mut Cache) -> Vec<f32> {
        match (&self.model, cache) {
            (Weights::Dense(m), Cache::Dense(c)) => m.forward_token(token, c),
            (Weights::Moe(m), Cache::Moe(c)) => m.forward_token(token, c),
            (Weights::Dense(_), Cache::Moe(_)) => {
                panic!("resident engine: dense weights given a mixture-of-experts cache")
            }
            (Weights::Moe(_), Cache::Dense(_)) => {
                panic!("resident engine: mixture-of-experts weights given a dense cache")
            }
        }
    }

    /// True when the loaded stack is the mixture-of-experts one.
    pub fn is_mixture_of_experts(&self) -> bool {
        matches!(self.model, Weights::Moe(_))
    }
}

pub struct ResidentSession<'a> {
    engine: &'a ResidentEngine,
    cache: Cache,
    /// Everything the model has WRITTEN, across turns, and nothing it was told.
    ///
    /// Repeating turn one's answer verbatim in turn three is the same failure as
    /// repeating a word, so this outlives a single turn. It must never contain the
    /// prompt: penalising the person's own words is what makes a model asked to repeat
    /// a figure reach for a lookalike instead.
    history: Vec<u32>,
    decoder: Decoder,
    turn: usize,
    /// Position up to which the cache holds trusted input rather than the model's own
    /// output. See `mark_clean`.
    clean: usize,
}

impl ResidentSession<'_> {
    /// How many turns have been answered so far.
    pub fn turns(&self) -> usize {
        self.turn
    }

    /// How many positions the cache holds. Each one costs real memory, so a caller that
    /// runs long conversations will want to watch this.
    pub fn positions(&self) -> usize {
        self.cache.len()
    }

    /// Set the decode policy. 1.0 penalty with temperature 0 reproduces plain greedy.
    pub fn set_decoder(&mut self, decoder: Decoder) {
        self.decoder = decoder;
    }

    /// Mark the current position as clean ground.
    ///
    /// Everything up to here is trusted: the system prompt, the tool catalogue, the task
    /// as the person actually stated it. Everything after it is the model's own output,
    /// which it cannot distinguish from what it was told -- so a figure it invents is a
    /// fact to it from then on. The mark is where the valve cuts back to.
    pub fn mark_clean(&mut self) {
        self.clean = self.cache.len();
    }

    /// Where the clean ground is.
    pub fn clean_mark(&self) -> usize {
        self.clean
    }

    /// How much of the cache is the model's own output rather than trusted input.
    pub fn pressure(&self) -> usize {
        self.cache.len().saturating_sub(self.clean)
    }

    /// Open the valve: drop everything the model has said since the clean mark, then put
    /// `digest` back in its place.
    ///
    /// The caller decides what the digest says -- a summary the model wrote, the tool
    /// results alone, or simply the last question and answer. This method only knows how
    /// to release the pressure and refill with something clean; deciding what is worth
    /// keeping is a judgement, and judgement does not belong in a cache.
    ///
    /// Returns how many positions were released.
    pub fn relieve(&mut self, digest: &str) -> Result<usize> {
        let before = self.cache.len();
        if before <= self.clean {
            return Ok(0);
        }
        self.cache.truncate(self.clean)?;
        let released = before - self.clean;

        if !digest.trim().is_empty() {
            let text = format!(
                "<|im_start|>user\nEarlier in this conversation, established and confirmed:\n{}\n<|im_end|>\n<|im_start|>assistant\nUnderstood.<|im_end|>\n",
                digest.trim()
            );
            let encoding = self
                .engine
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
            for &tok in encoding.get_ids() {
                // Input, not output: it must not attract the repetition penalty.
                let _ = self.engine.forward(tok, &mut self.cache);
            }
        }
        // Whatever the model had written is gone from the cache, so it must go from the
        // penalty's memory too, or the cleaned turn would still be steered away from it.
        self.history.clear();
        Ok(released)
    }

    /// Ask, and get the answer plus what it cost.
    ///
    /// Only the new turn is prefilled. Everything said earlier is already in the cache,
    /// which is the whole point: the cost of turn N is the cost of turn N, not the cost
    /// of the transcript so far.
    pub fn ask(&mut self, prompt: &str, max_tokens: usize) -> Result<(String, TurnCost)> {
        let text = if self.turn == 0 {
            first_turn(prompt)
        } else {
            next_turn(prompt)
        };
        let bos = if self.turn == 0 { self.engine.bos } else { None };
        self.feed(&text, bos, max_tokens)
    }

    /// Append text exactly as given, adding no turn markers of any kind.
    ///
    /// For callers that already build their own conversation markers -- the agent loop
    /// writes its own system turn, tool catalogue and transcript -- wrapping again would
    /// nest one conversation inside another and the model would see a malformed
    /// transcript. Those callers send their own opening text on the first turn and only
    /// what is new on every turn after.
    pub fn feed_raw(&mut self, text: &str, max_tokens: usize) -> Result<(String, TurnCost)> {
        let bos = if self.turn == 0 { self.engine.bos } else { None };
        self.feed(text, bos, max_tokens)
    }

    fn feed(
        &mut self,
        text: &str,
        bos: Option<u32>,
        max_tokens: usize,
    ) -> Result<(String, TurnCost)> {
        let encoding = self
            .engine
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {e}"))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if let Some(b) = bos {
            if ids.first() != Some(&b) {
                ids.insert(0, b);
            }
        }

        // Each turn starts the penalty from nothing.
        //
        // The penalty exists to stop one answer jamming on itself. Carrying it between
        // turns makes it police something else entirely: repeating a fact. Asked five
        // turns later which card was mentioned, the model had already written "Quadro
        // M2000M" and "4096" in turn one, so both were on its own blacklist -- and it
        // answered "Quadro M200₀M" with "4₀96 MB", reaching for subscript lookalikes
        // exactly as it had reached for full-width ones when the prompt was penalised.
        // Repeating a fact across turns is not jamming; it is answering the question.
        self.history.clear();

        let t1 = Instant::now();
        let mut next_logits: Vec<f32> = Vec::new();
        for &tok in &ids {
            // Not pushed to `history`: the penalty applies to what the model writes, not
            // to what it is told. Seeding it with the prompt is what made a model asked
            // to repeat "4096" answer with full-width lookalike digits instead.
            next_logits = self.engine.forward(tok, &mut self.cache);
        }
        let prefill = t1.elapsed();

        let mut generated: Vec<u32> = Vec::new();
        let t2 = Instant::now();
        for _ in 0..max_tokens {
            let next_id = self.decoder.pick(&next_logits, &[], &self.history);
            generated.push(next_id);
            if self.engine.eos.contains(&next_id) {
                break;
            }
            self.history.push(next_id);
            next_logits = self.engine.forward(next_id, &mut self.cache);
        }
        let decode = t2.elapsed();

        let answer = self
            .engine
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {e}"))?;

        self.turn += 1;
        Ok((
            answer.trim().to_string(),
            TurnCost {
                prompt_tokens: ids.len(),
                prefill,
                generated_tokens: generated.len(),
                decode,
                cache_positions: self.cache.len(),
            },
        ))
    }
}

/// System turn, user turn, then an opened assistant turn for the model to complete.
/// Taken verbatim from the chat template shipped beside the weights, not guessed.
fn first_turn(prompt: &str) -> String {
    format!(
        "<|im_start|>system\nYou are Qwen, a helpful AI assistant.<|im_end|>\n\
         <|im_start|>user\n{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

/// Every turn after the first. Generation stops on `<|im_end|>` without feeding that
/// token to the model, so closing the previous turn here is what keeps the transcript
/// well-formed.
fn next_turn(prompt: &str) -> String {
    format!(
        "<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n\
         <|im_start|>assistant\n"
    )
}

fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

fn read_bos(config_path: &Path) -> Option<u32> {
    let v = read_json(config_path)?;
    v.get("text_config")
        .and_then(|t| t.get("bos_token_id"))
        .or_else(|| v.get("bos_token_id"))
        .and_then(serde_json::Value::as_u64)
        .map(|x| x as u32)
}

/// Every id that should end generation.
///
/// `eos_token_id` is a single number on base checkpoints and a list on instruction-tuned
/// ones, because those have more than one way to stop. Reading only the first means the
/// model never stops on end-of-turn and runs to the token limit every time.
fn read_eos_ids(config_path: &Path) -> Vec<u32> {
    let Some(v) = read_json(config_path) else {
        return Vec::new();
    };
    let node = v
        .get("text_config")
        .and_then(|t| t.get("eos_token_id"))
        .or_else(|| v.get("eos_token_id"));
    match node {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|x| x as u32)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_first_turn_carries_the_system_prompt() {
        let t = first_turn("hello");
        assert!(t.starts_with("<|im_start|>system"));
        assert!(t.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn a_later_turn_closes_the_previous_answer_first() {
        let t = next_turn("and then?");
        assert!(t.starts_with("<|im_end|>\n<|im_start|>user"));
        assert!(!t.contains("system"));
        assert!(t.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn a_later_turn_is_shorter_than_a_first_one() {
        // The saving is the point: everything before this turn is already resident.
        assert!(next_turn("x").len() < first_turn("x").len());
    }
}
