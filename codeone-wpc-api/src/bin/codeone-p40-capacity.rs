//! CUDA P40 24GiB capacity preflight for Qwen3.6-35B-A3B.
 //! READ-ONLY: parses original local config.json; NO GPU IO, model load, CUDA, or services.
 //! Estimates lower bound for weights and full-attention KV, PLUS explicit reserve.
use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};

const GIB: f64 = 1_073_741_824.0;
const PARAMS_QWEN36_35B: u64 = 35_951_824_188;
const REQUIRED_CONTEXT: u64 = 65_536;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Scheme {
    V4,
    V3,
}

impl Scheme {
    fn bpw(self) -> f64 {
        match self {
            Self::V4 => 4.25,
            Self::V3 => 6.25,
        }
    }
}

#[derive(Debug, Parser)]
#[command(about = "Conservative, source-bound P40 memory budget for Qwen3.6-35B-A3B (NOT GPU runtime)")]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, value_enum, default_value = "v4")]
    scheme: Scheme,
    #[arg(long, default_value_t = REQUIRED_CONTEXT)]
    context_tokens: u64,
    #[arg(long, default_value_t = 2)]
    kv_bytes: u64,
    #[arg(long, default_value_t = 24.0)]
    p40_vram_gib: f64,
    #[arg(long, default_value_t = 3.0)]
    runtime_reserve_gib: f64,
    #[arg(long, default_value_t = 0.75)]
    precision_reserve_gib: f64,
    #[arg(long, default_value_t = 1.0)]
    minimum_headroom_gib: f64,
    #[arg(long, default_value_t = PARAMS_QWEN36_35B)]
    parameter_count: u64,
}

#[derive(Deserialize)]
struct ModelConfig {
    model_type: String,
    text_config: TextConfig,
}
#[derive(Deserialize)]
struct TextConfig {
    model_type: String,
    num_hidden_layers: usize,
    layer_types: Vec<String>,
    head_dim: u64,
    num_key_value_heads: u64,
    max_position_embeddings: u64,
    num_experts: u64,
    num_experts_per_tok: u64,
}

#[derive(Debug, Serialize)]
struct Budget {
    model_id: &'static str,
    model_type: &'static str,
    status: &'static str,
    codebase_supports_inference: bool,
    gpu_full_forward_verified: bool,
    source_config_verified: bool,
    target: &'static str,
    scheme_bpw: f64,
    context_tokens: u64,
    full_attention_layers: u64,
    linear_attention_layers: u64,
    kv_bytes_per_element: u64,
    parameter_count: u64,
    packed_weight_lower_bound_gib: f64,
    full_attention_kv_gib: f64,
    precision_reserve_gib: f64,
    runtime_reserve_gib: f64,
    projected_peak_gib: f64,
    free_headroom_gib: f64,
    target_vram_gib: f64,
    minimum_headroom_gib: f64,
    memory_budget_candidate: bool,
    outstanding: Vec<&'static str>,
}

fn compute(config: &ModelConfig, cli: &Cli) -> Result<Budget> {
    if config.model_type != "qwen3_5_moe" || config.text_config.model_type != "qwen3_5_moe_text" {
        bail!("wrong model architecture; refuse use of existing qwen3_moe decoder");
    }
    let t = &config.text_config;
    if t.num_hidden_layers != t.layer_types.len() {
        bail!("layer_types length != num_hidden_layers");
    }
    if t.num_hidden_layers != 40 || t.num_key_value_heads != 2 || t.head_dim != 256 {
        bail!("not the audited Qwen3.6-35B-A3B geometry; inspect source before budgeting");
    }
    if t.num_experts != 256 || t.num_experts_per_tok != 8 {
        bail!("unexpected MoE router geometry");
    }
    if cli.context_tokens < 50_000 || cli.context_tokens > t.max_position_embeddings {
        bail!("context outside required >=50K and native model maximum");
    }
    if cli.kv_bytes != 1 && cli.kv_bytes != 2 && cli.kv_bytes != 4 {
        bail!("kv_bytes must be 1 (int8), 2 (f16) or 4 (f32)");
    }
    if !cli.p40_vram_gib.is_finite() || cli.p40_vram_gib < 1.0
        || !cli.runtime_reserve_gib.is_finite() || cli.runtime_reserve_gib < 1.0
        || !cli.precision_reserve_gib.is_finite() || cli.precision_reserve_gib < 0.0
        || !cli.minimum_headroom_gib.is_finite() || cli.minimum_headroom_gib < 0.0
        || cli.parameter_count < 30_000_000_000
    {
        bail!("invalid VRAM, reserve, headroom or model parameter argument");
    }
    let full=t.layer_types.iter().filter(|x| x.as_str()=="full_attention").count();
    let linear=t.layer_types.iter().filter(|x| x.as_str()=="linear_attention").count();
    if full + linear != t.layer_types.len() || full != 10 || linear != 30 {
        bail!("unsupported layer types or changed 3:1 hybrid pattern");
    }

    let weights_gib=(cli.parameter_count as f64) * cli.scheme.bpw() / 8.0 / GIB;
    let kv_gib=(cli.context_tokens as f64)
        * (full as f64) * 2.0 * (t.num_key_value_heads as f64)
        * (t.head_dim as f64) * (cli.kv_bytes as f64) / GIB;
    let total=weights_gib+kv_gib+cli.precision_reserve_gib+cli.runtime_reserve_gib;
    let headroom=cli.p40_vram_gib-total;

    Ok(Budget {
        model_id:"Qwen/Qwen3.6-35B-A3B",
        model_type:"qwen3_5_moe",
        status:"UNVERIFIED_SIZING_ONLY",
        codebase_supports_inference:false,
        gpu_full_forward_verified:false,
        source_config_verified:true,
        target:"NVIDIA Tesla P40 Pascal sm_61",
        scheme_bpw:cli.scheme.bpw(),
        context_tokens:cli.context_tokens,
        full_attention_layers:full as u64,
        linear_attention_layers:linear as u64,
        kv_bytes_per_element:cli.kv_bytes,
        parameter_count:cli.parameter_count,
        packed_weight_lower_bound_gib:weights_gib,
        full_attention_kv_gib:kv_gib,
        precision_reserve_gib:cli.precision_reserve_gib,
        runtime_reserve_gib:cli.runtime_reserve_gib,
        projected_peak_gib:total,
        free_headroom_gib:headroom,
        target_vram_gib:cli.p40_vram_gib,
        minimum_headroom_gib:cli.minimum_headroom_gib,
        memory_budget_candidate:headroom>=cli.minimum_headroom_gib,
        outstanding:vec![
            "Implement native qwen3_5_moe_text Gated DeltaNet + 10 gated full-attention layers; qwen3_moe is incompatible",
            "Quantize and verify real checkpoint with WPC; selective higher precision for routing/embedding and actual total bytes",
            "Add SM61 WPC4 fused GEMV to real GPU forward, tensor residency, attention, recurrent state, GPU KV",
            "Prove 65536 token context and >50000 usable input under OpenCode output/tool budget",
            "Verify real VRAM peak, prefill/decode tok/s, tool-call contract, held-out coding quality on P40 after owner GO"
        ],
    })
}

fn main() -> Result<()> {
    let cli=Cli::parse();
    let config:ModelConfig=serde_json::from_str(&fs::read_to_string(&cli.config)
        .with_context(||format!("read model config {}",cli.config.display()))?)?;
    let budget=compute(&config,&cli)?;
    println!("{}",serde_json::to_string_pretty(&budget)?);
    if !budget.memory_budget_candidate {
        bail!("model exceeds configured P40 memory headroom: NO-GO");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> ModelConfig {
        ModelConfig {
            model_type:"qwen3_5_moe".to_owned(),
            text_config:TextConfig {
                model_type:"qwen3_5_moe_text".into(),
                num_hidden_layers:40,
                layer_types:(0..40).map(|i|if i%4==3 {"full_attention".into()} else {"linear_attention".into()}).collect(),
                head_dim:256,num_key_value_heads:2,max_position_embeddings:262_144,
                num_experts:256,num_experts_per_tok:8
            }
        }
    }
    fn args() -> Cli {
        Cli {
            config:PathBuf::from("/unused/config.json"),
            scheme:Scheme::V4,
            context_tokens:65_536,
            kv_bytes:2,
            p40_vram_gib:24.0,
            runtime_reserve_gib:3.0,
            precision_reserve_gib:0.75,
            minimum_headroom_gib:1.0,
            parameter_count:PARAMS_QWEN36_35B,
        }
    }
    #[test]
    fn qwen36_64k_wpc4_fits_the_planning_budget() {
        let b=compute(&fixture(),&args()).unwrap();
        assert_eq!(b.full_attention_layers,10);
        assert_eq!(b.linear_attention_layers,30);
        assert!((b.full_attention_kv_gib-1.25).abs() < 0.000001);
        assert!(b.memory_budget_candidate);
        assert!(b.free_headroom_gib>1.0);
        assert!(!b.codebase_supports_inference);
    }
    #[test]
    fn qwen36_wpc3_does_not_fit_with_64k_context() {
        let mut c=args();
        c.scheme=Scheme::V3;
        assert!(!compute(&fixture(),&c).unwrap().memory_budget_candidate);
    }
    #[test]
    fn forbid_short_context_and_wrong_model_architecture() {
        let mut c=args();c.context_tokens=16_384;
        assert!(compute(&fixture(),&c).is_err());
        let mut f=fixture();
        f.model_type="qwen3_moe".to_owned();
        assert!(compute(&f,&args()).is_err());
    }
    #[test]
    fn source_layer_changes_fail_closed() {
        let mut f=fixture();
        f.text_config.layer_types[0]="unsupported_attention".to_owned();
        assert!(compute(&f,&args()).is_err());
        f=fixture(); f.text_config.num_hidden_layers=39;
        assert!(compute(&f,&args()).is_err());
    }
    #[test]
    fn guard_overoptimistic_memory() {
        let mut c=args();
        c.runtime_reserve_gib=6.0;
        assert!(!compute(&fixture(),&c).unwrap().memory_budget_candidate);
    }
}
