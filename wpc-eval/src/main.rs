mod fused_kernel;

use std::env;
use std::time::Instant;
use wpc_core::safetensors::extract_layers;
use wpc_core::codebook::{PatternDict, BLOCK_DIM};
use wpc_core::encoder::{two_pass_encode, normalize_block, INPUT_SCALE};
use wpc_format::{CompressedBlock, BLOCK_SIZE};
use half::f16;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: wpc_eval <path_to_safetensors>");
        std::process::exit(1);
    }
    let model_path = &args[1];

    println!("Loading safetensors from: {}", model_path);
    let layers = extract_layers(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load safetensors: {e}");
        std::process::exit(1);
    });

    if layers.is_empty() {
        println!("No compatible layers found.");
        return;
    }

    println!("Found {} weight matrices.", layers.len());
    
    // Pick the first one for demonstration
    let layer = &layers[0];
    println!("Processing layer: {} (shape: {:?}, numel: {})", layer.name, layer.shape, layer.data.len());

    let n_weights = layer.data.len();
    let n_blocks = n_weights / BLOCK_SIZE;
    
    if n_blocks == 0 {
        println!("Layer too small.");
        return;
    }

    let t0 = Instant::now();
    
    // Slice data into blocks
    let mut data_blocks = Vec::with_capacity(n_blocks);
    for i in 0..n_blocks {
        let mut b = [0.0; BLOCK_DIM];
        b.copy_from_slice(&layer.data[i*BLOCK_DIM..(i+1)*BLOCK_DIM]);
        data_blocks.push(b);
    }

    // 1. Train Pattern Dictionary (L1)
    println!("Training L1 Pattern Dictionary (256 centroids, 16 iters)...");
    // Must train in the same normalized space that nearest() is queried in
    // at encode time (see wpc_core::encoder::normalize_block).
    let normalized_blocks: Vec<[f32; BLOCK_DIM]> = data_blocks.iter()
        .map(|b| normalize_block(b).norm)
        .collect();
    let pattern_dict = PatternDict::train(&normalized_blocks, 256, 16);
    
    // 2. Two-pass encode and train Residual Dictionary (L2)
    println!("Harvesting residuals and training L2 Residual Dictionary (65k centroids, 4 iters)...");
    let (blocks, residual_dict) = two_pass_encode(&layer.data[..n_blocks*BLOCK_SIZE], &pattern_dict, 4);
    
    let enc_ms = t0.elapsed().as_secs_f64() * 1e3;
    println!("Encoding complete in {:.2} ms.", enc_ms);

    // 3. Fused kernel evaluation setup
    // Synthetic activations: random N(0, 1)
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x1234_5678);
    let x: Vec<f32> = (0..n_weights).map(|_| {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos()
    }).collect();

    // Flatten dictionaries. The fused kernel expects on-disk-contract patterns
    // (pre-divided by INPUT_SCALE) since it reconstructs via `pattern * scale + base`
    // with no further division -- see wpc-format/src/lib.rs.
    let flat_patterns: Vec<f32> = pattern_dict.centroids.iter()
        .flat_map(|b| b.iter().map(|&v| v / INPUT_SCALE))
        .collect();
    let flat_residuals: Vec<f16> = residual_dict.centroids_f16.iter().flat_map(|b| b.iter().copied()).collect();

    if !is_x86_feature_detected!("avx2")
        || !is_x86_feature_detected!("fma")
        || !is_x86_feature_detected!("f16c")
    {
        eprintln!("[evaluator] CPU lacks AVX2+FMA+F16C. Aborting kernel evaluation.");
        std::process::exit(1);
    }

    // Warm-up
    let _ = unsafe {
        fused_kernel::matvec_wpc_fused(
            &blocks,
            flat_patterns.as_ptr(),
            flat_residuals.as_ptr(),
            x.as_ptr(),
            n_blocks,
        )
    };

    // Time fused kernel
    const ITERS: usize = 100; // Small number since layer is large
    let t0 = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..ITERS {
        let y = unsafe {
            fused_kernel::matvec_wpc_fused(
                &blocks,
                flat_patterns.as_ptr(),
                flat_residuals.as_ptr(),
                x.as_ptr(),
                n_blocks,
            )
        };
        sink += y;
    }
    let fused_ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    let y_fused = unsafe {
        fused_kernel::matvec_wpc_fused(
            &blocks,
            flat_patterns.as_ptr(),
            flat_residuals.as_ptr(),
            x.as_ptr(),
            n_blocks,
        )
    };

    // Time baseline
    let t0 = Instant::now();
    for _ in 0..ITERS {
        let y = unsafe { fused_kernel::matvec_fp32_baseline(layer.data.as_ptr(), x.as_ptr(), n_weights) };
        sink += y;
    }
    let baseline_ms = t0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    let y_baseline = unsafe { fused_kernel::matvec_fp32_baseline(layer.data.as_ptr(), x.as_ptr(), n_weights) };

    let abs_err = (y_fused - y_baseline).abs();
    let rel_err = abs_err / y_baseline.abs().max(1e-9);
    
    // Calculate RMSE of weights
    let mut w_err_sq = 0.0;
    for i in 0..n_blocks {
        let b = &blocks[i];
        let p = &pattern_dict.centroids[b.pattern_id as usize];
        let r = &residual_dict.centroids_f16[b.residual_id as usize];
        let base = b.base_value.to_f32();
        let scale = b.scale as f32;
        let s_decode = scale / INPUT_SCALE;

        for j in 0..BLOCK_DIM {
            let approx = p[j] * s_decode + base + r[j].to_f32() / INPUT_SCALE;
            let orig = layer.data[i * BLOCK_DIM + j];
            let d = approx - orig;
            w_err_sq += d * d;
        }
    }
    let rmse = (w_err_sq / n_weights as f32).sqrt();

    println!();
    println!("================ WPC PHASE 1 RESULTS ================");
    println!("Layer       : {}", layer.name);
    println!("Num Weights : {}", n_weights);
    println!("y_fused     = {y_fused:.6}");
    println!("y_baseline  = {y_baseline:.6}");
    println!("RMSE        = {rmse:.6}");
    println!("rel err     = {rel_err:.3e}");
    println!();
    println!("Kernel time (avg over {ITERS} iters):");
    println!("  FP32 baseline : {baseline_ms:8.4} ms/iter");
    println!("  WPC fused     : {fused_ms:8.4} ms/iter");
    println!();
    let wpc_weight_bytes = n_blocks * 6;
    let fp32_weight_bytes = n_weights * 4;
    println!("Weight-bytes touched per call:");
    println!("  FP32 baseline : {fp32_weight_bytes:>10} B ({:>6.2} KB)", fp32_weight_bytes as f64 / 1024.0);
    println!("  WPC fused     : {wpc_weight_bytes:>10} B ({:>6.2} KB)", wpc_weight_bytes as f64 / 1024.0);
    println!("  Compression   : {:.2}x", fp32_weight_bytes as f64 / wpc_weight_bytes as f64);
    println!("======================================================");
    let _ = sink;
}
