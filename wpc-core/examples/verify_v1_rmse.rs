//! RMSE verification for the v1 (VQ codebook) scheme, run against real
//! safetensors weights using the actual production `wpc_core::codebook` and
//! `wpc_core::encoder` code -- the same way `verify_v2_rmse.rs` does it for v2.
//!
//! Mirrors what wpc-compiler does for `--scheme v1`: collect every tensor of
//! one class, sample up to TARGET_SAMPLE blocks, train the 256-entry
//! PatternDict in normalized space, harvest residuals, train the residual
//! dict, then encode and measure.
//!
//! Reports an error decomposition so the loss can be attributed:
//!   base+scale only : perfect pattern, only base(f16)+scale(i8) quantization
//!   +pattern        : real 256-entry pattern lookup
//!   +residual       : full v1 decode
//!
//! Usage: verify_v1_rmse <safetensors_path> <class_substring> <target_tensor>
//! Env knobs: WPC_RESIDUAL_K (default = wpc_format::RESIDUAL_COUNT)
//!            WPC_PATTERN_K  (default = wpc_format::PATTERN_COUNT)
//!            WPC_TARGET_SAMPLE (default 262144)

use half::f16;
use memmap2::Mmap;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;
use wpc_core::codebook::{PatternDict, ResidualDict, BLOCK_DIM};
use wpc_core::encoder::{normalize_block, BlockNorm, INPUT_SCALE};
use wpc_core::quant_encoder::affine_quant_block;
use wpc_format::{BLOCK_SIZE_V2, PATTERN_COUNT, RESIDUAL_COUNT};

fn decode_dtype(view: &safetensors::tensor::TensorView) -> Vec<f32> {
    match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        safetensors::Dtype::F16 => view
            .data()
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        safetensors::Dtype::BF16 => view
            .data()
            .chunks_exact(2)
            .map(|c| {
                let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                f32::from_bits(bits << 16)
            })
            .collect(),
        other => panic!("unsupported dtype {other:?}"),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("Usage: verify_v1_rmse <safetensors_path> <class_substring> <target_tensor>");
        std::process::exit(1);
    }
    let path = &args[1];
    let class_sub = &args[2];
    let target = &args[3];

    let pattern_k = env_usize("WPC_PATTERN_K", PATTERN_COUNT);
    let residual_k_cfg = env_usize("WPC_RESIDUAL_K", RESIDUAL_COUNT);
    let target_sample = env_usize("WPC_TARGET_SAMPLE", 262144);

    let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let st = SafeTensors::deserialize(&mmap).expect("invalid safetensors");

    // ---- 1. Gather the training pool: every tensor whose name contains class_sub
    // WPC_EXCLUDE_TARGET=1 keeps the measured tensor out of the training pool, so a
    // large codebook cannot score by memorizing the very blocks it is scored on.
    let exclude_target = std::env::var("WPC_EXCLUDE_TARGET").is_ok();
    let names: Vec<String> = st
        .names()
        .into_iter()
        .filter(|n| n.contains(class_sub.as_str()))
        .filter(|n| !(exclude_target && *n == target.as_str()))
        .map(|s| s.to_string())
        .collect();
    if exclude_target {
        println!("(held-out: {target} excluded from the training pool)");
    }
    println!("class '{class_sub}': {} tensors", names.len());

    let mut pool: Vec<[f32; BLOCK_DIM]> = Vec::new();
    let mut total_blocks = 0usize;
    for n in &names {
        let v = st.tensor(n).unwrap();
        let d = decode_dtype(&v);
        total_blocks += d.len() / BLOCK_DIM;
    }
    let sample_rate = (target_sample as f64 / total_blocks as f64).min(1.0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    for n in &names {
        let v = st.tensor(n).unwrap();
        let d = decode_dtype(&v);
        let nb = d.len() / BLOCK_DIM;
        for i in 0..nb {
            if rng.gen::<f64>() > sample_rate {
                continue;
            }
            let mut b = [0.0f32; BLOCK_DIM];
            b.copy_from_slice(&d[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
            pool.push(b);
        }
    }
    println!(
        "training pool: {} blocks sampled out of {} total (rate {:.3})",
        pool.len(),
        total_blocks,
        sample_rate
    );

    // Bug-reproduction switches, to re-measure the pre-fix behaviour on demand.
    //   WPC_BUG_TRAINSPACE=1 -> train PatternDict on RAW blocks while encode still
    //                           queries it with normalized ones (the "training-space
    //                           mismatch" fixed in cf0a9fd / 5c17fb9)
    //   WPC_BUG_KMEANS=1     -> hand ResidualDict::train exactly k samples, which is
    //                           what the old `max_samples = k` cap did: kmeans takes
    //                           its n<=k path and Lloyd never runs (fixed in 35fcad3)
    let bug_trainspace = std::env::var("WPC_BUG_TRAINSPACE").is_ok();
    let bug_kmeans = std::env::var("WPC_BUG_KMEANS").is_ok();
    if bug_trainspace || bug_kmeans {
        println!("!! BUG REPRO MODE: trainspace={bug_trainspace} kmeans={bug_kmeans}");
    }

    // ---- 2. Train PatternDict in normalized space (compiler does exactly this)
    let normalized: Vec<[f32; BLOCK_DIM]> = pool.par_iter().map(|b| normalize_block(b).norm).collect();
    println!("training PatternDict k={pattern_k} ...");
    let pattern_dict = if bug_trainspace {
        PatternDict::train(&pool, pattern_k, 20)
    } else {
        PatternDict::train(&normalized, pattern_k, 20)
    };

    // ---- 3. Harvest residuals, train ResidualDict
    let residuals: Vec<[f32; BLOCK_DIM]> = pool
        .par_iter()
        .map(|b| {
            let BlockNorm { base, scale_i8, norm } = normalize_block(b);
            let (pid, _) = pattern_dict.nearest(&norm);
            let p = pattern_dict.centroids[pid as usize];
            let s = scale_i8 as f32 / INPUT_SCALE;
            let mut r = [0.0f32; BLOCK_DIM];
            for i in 0..BLOCK_DIM {
                r[i] = (b[i] - (p[i] * s + base)) * INPUT_SCALE;
            }
            r
        })
        .collect();
    let residual_k = residual_k_cfg.min(residuals.len());
    println!("training ResidualDict k={residual_k} (requested {residual_k_cfg}) ...");
    let residual_dict = if bug_kmeans {
        ResidualDict::train(&residuals[..residual_k], residual_k, 10)
    } else {
        ResidualDict::train(&residuals, residual_k, 10)
    };
    // How many of the trained centroids are actually distinct?
    let mut seen: std::collections::HashSet<[u16; BLOCK_DIM]> = std::collections::HashSet::new();
    for c in &residual_dict.centroids_f16 {
        let mut key = [0u16; BLOCK_DIM];
        for i in 0..BLOCK_DIM {
            key[i] = c[i].to_bits();
        }
        seen.insert(key);
    }
    println!("residual dict: {} distinct centroids of {}", seen.len(), residual_dict.centroids_f16.len());

    // ---- 4. Encode the target tensor and measure
    let tv = st.tensor(target).unwrap_or_else(|_| panic!("tensor {target} not found"));
    let shape = tv.shape().to_vec();
    let data = decode_dtype(&tv);
    let nb = data.len() / BLOCK_DIM;
    let usable = nb * BLOCK_DIM;

    let mut e_scale = 0.0f64; // base + scale only, perfect pattern
    let mut e_pat = 0.0f64; // + real pattern lookup
    let mut e_full = 0.0f64; // + residual
    let mut sig = 0.0f64;
    let mut zero_scale = 0usize;
    let mut scale_hist = [0usize; 128];

    for bi in 0..nb {
        let mut b = [0.0f32; BLOCK_DIM];
        b.copy_from_slice(&data[bi * BLOCK_DIM..(bi + 1) * BLOCK_DIM]);
        let BlockNorm { base, scale_i8, norm } = normalize_block(&b);
        if scale_i8 == 0 {
            zero_scale += 1;
        }
        scale_hist[(scale_i8 as i32).unsigned_abs() as usize % 128] += 1;

        let base16 = f16::from_f32(base).to_f32(); // base really goes to disk as f16
        let s = scale_i8 as f32 / INPUT_SCALE;
        let (pid, _) = pattern_dict.nearest(&norm);
        let p = pattern_dict.centroids[pid as usize];

        // residual against the f16 base actually stored
        let mut raw_res = [0.0f32; BLOCK_DIM];
        for i in 0..BLOCK_DIM {
            raw_res[i] = (b[i] - (p[i] * s + base16)) * INPUT_SCALE;
        }
        let (rid, _) = residual_dict.nearest(&raw_res);
        let r = residual_dict.centroids_f16[rid as usize];

        for i in 0..BLOCK_DIM {
            let o = b[i] as f64;
            sig += o * o;
            let a_scale = (norm[i] * s + base16) as f64;
            let a_pat = (p[i] * s + base16) as f64;
            let a_full = a_pat + (r[i].to_f32() / INPUT_SCALE) as f64;
            e_scale += (o - a_scale).powi(2);
            e_pat += (o - a_pat).powi(2);
            e_full += (o - a_full).powi(2);
        }
    }

    let sig_rms = (sig / usable as f64).sqrt();
    let pct = |e: f64| 100.0 * (e / usable as f64).sqrt() / sig_rms;

    println!("\n=== {target} shape={shape:?} blocks={nb} ===");
    println!("signal RMS                    : {sig_rms:.6}");
    println!("v1 base(f16)+scale(i8) only   : {:.2}%", pct(e_scale));
    println!("v1 + pattern (k={pattern_k})       : {:.2}%", pct(e_pat));
    println!("v1 FULL (pattern+residual)    : {:.2}%   <-- v1 reconstruction error", pct(e_full));
    println!(
        "blocks with scale_i8 == 0     : {} of {} ({:.2}%)",
        zero_scale,
        nb,
        100.0 * zero_scale as f64 / nb as f64
    );
    let mut used: Vec<(usize, usize)> = scale_hist.iter().cloned().enumerate().filter(|(_, c)| *c > 0).collect();
    used.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    used.truncate(10);
    println!("top |scale_i8| values         : {used:?}");

    // ---- 5. v2 reference on the same tensor
    let nb2 = data.len() / BLOCK_SIZE_V2;
    let mut e2 = 0.0f64;
    let mut sig2 = 0.0f64;
    for bi in 0..nb2 {
        let mut block = [0.0f32; BLOCK_SIZE_V2];
        block.copy_from_slice(&data[bi * BLOCK_SIZE_V2..(bi + 1) * BLOCK_SIZE_V2]);
        let qb = affine_quant_block(&block);
        let zp = qb.zero_point.to_f32();
        let sc = qb.scale.to_f32();
        for k in 0..BLOCK_SIZE_V2 {
            let o = block[k] as f64;
            sig2 += o * o;
            e2 += (o - (zp + qb.codes[k] as f32 * sc) as f64).powi(2);
        }
    }
    let rms2 = (sig2 / (nb2 * BLOCK_SIZE_V2) as f64).sqrt();
    println!(
        "v2/v3 reference (same tensor) : {:.2}%",
        100.0 * (e2 / (nb2 * BLOCK_SIZE_V2) as f64).sqrt() / rms2
    );

    println!(
        "\nbitrate: v1 = {:.2} bits/weight, v2 = {:.2}, v3 = {:.2}",
        6.0 * 8.0 / BLOCK_DIM as f64,
        132.0 * 8.0 / BLOCK_SIZE_V2 as f64,
        100.0 * 8.0 / BLOCK_SIZE_V2 as f64
    );
}
