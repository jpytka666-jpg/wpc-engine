//! Diagnostic for the v1 idea itself: do 16-value weight blocks actually fall
//! into a small number of repeatable patterns?
//!
//! Three measurements, all on real weights:
//!   1. Eigenvalue spectrum of the 16x16 covariance of normalized blocks.
//!      A flat spectrum means the blocks fill the whole 16-D ball -- no
//!      low-dimensional structure for a pattern catalogue to exploit.
//!   2. Pattern-only reconstruction error vs codebook size k. Shows how much
//!      a bigger catalogue actually buys.
//!   3. Plain affine quantization at MATCHED bitrate, so v1 is compared
//!      against its real rival at the same number of bits per weight.
//!
//! Usage: diag_v1_structure <safetensors_path> <class_substring> <target_tensor>

use half::f16;
use memmap2::Mmap;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use safetensors::SafeTensors;
use std::fs::File;
use wpc_core::codebook::{PatternDict, BLOCK_DIM};
use wpc_core::encoder::{normalize_block, BlockNorm, INPUT_SCALE};

fn decode_dtype(view: &safetensors::tensor::TensorView) -> Vec<f32> {
    match view.dtype() {
        safetensors::Dtype::F32 => view.data().chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        safetensors::Dtype::F16 => view.data().chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
        safetensors::Dtype::BF16 => view.data().chunks_exact(2)
            .map(|c| f32::from_bits((u32::from(c[0]) | (u32::from(c[1]) << 8)) << 16)).collect(),
        other => panic!("unsupported dtype {other:?}"),
    }
}

/// Jacobi eigenvalue iteration for a small symmetric matrix.
fn eigenvalues(mut a: Vec<Vec<f64>>) -> Vec<f64> {
    let n = a.len();
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..n { for j in 0..n { if i != j { off += a[i][j] * a[i][j]; } } }
        if off < 1e-18 { break; }
        for p in 0..n - 1 {
            for q in p + 1..n {
                if a[p][q].abs() < 1e-15 { continue; }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let akp = a[k][p];
                    let akq = a[k][q];
                    a[k][p] = c * akp - s * akq;
                    a[k][q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p][k];
                    let aqk = a[q][k];
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
            }
        }
    }
    let mut ev: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    ev.sort_by(|x, y| y.partial_cmp(x).unwrap());
    ev
}

/// Affine (min/max) quantization at an arbitrary block size and level count.
/// Returns (relative RMSE %, bits per weight).
fn affine_rel_rmse(data: &[f32], block: usize, levels: u32) -> (f64, f64) {
    let nb = data.len() / block;
    let mut err = 0.0f64;
    let mut sig = 0.0f64;
    let lmax = (levels - 1) as f32;
    for bi in 0..nb {
        let s = &data[bi * block..(bi + 1) * block];
        let mut mn = f32::MAX;
        let mut mx = f32::MIN;
        for &v in s { if v < mn { mn = v; } if v > mx { mx = v; } }
        // header stored as f16, exactly like v2/v3/v4 do
        let zp = f16::from_f32(mn).to_f32();
        let sc = f16::from_f32((mx - mn) / lmax).to_f32();
        for &v in s {
            let code = if sc > 0.0 { (((v - zp) / sc).round()).clamp(0.0, lmax) } else { 0.0 };
            let a = (zp + code * sc) as f64;
            err += (v as f64 - a).powi(2);
            sig += (v as f64).powi(2);
        }
    }
    let n = (nb * block) as f64;
    let bits = (32.0 + block as f64 * (levels as f64).log2()) / block as f64;
    (100.0 * (err / n).sqrt() / (sig / n).sqrt(), bits)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let class_sub = &args[2];
    let target = &args[3];

    let file = File::open(path).unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };
    let st = SafeTensors::deserialize(&mmap).unwrap();

    let names: Vec<String> = st.names().into_iter()
        .filter(|n| n.contains(class_sub.as_str())).map(|s| s.to_string()).collect();

    let mut pool: Vec<[f32; BLOCK_DIM]> = Vec::new();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    for n in &names {
        let d = decode_dtype(&st.tensor(n).unwrap());
        for i in 0..d.len() / BLOCK_DIM {
            if rng.gen::<f64>() > 0.5 { continue; }
            let mut b = [0.0f32; BLOCK_DIM];
            b.copy_from_slice(&d[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
            pool.push(b);
        }
    }
    println!("pool: {} blocks from {} tensors of class '{class_sub}'", pool.len(), names.len());

    let normalized: Vec<[f32; BLOCK_DIM]> =
        pool.par_iter().map(|b| normalize_block(b).norm).collect();

    // ---- 1. covariance eigenvalues of the normalized blocks
    let n = normalized.len() as f64;
    let mut mean = [0.0f64; BLOCK_DIM];
    for v in &normalized { for i in 0..BLOCK_DIM { mean[i] += v[i] as f64; } }
    for m in mean.iter_mut() { *m /= n; }
    let mut cov = vec![vec![0.0f64; BLOCK_DIM]; BLOCK_DIM];
    for v in &normalized {
        for i in 0..BLOCK_DIM {
            let vi = v[i] as f64 - mean[i];
            for j in 0..BLOCK_DIM {
                cov[i][j] += vi * (v[j] as f64 - mean[j]);
            }
        }
    }
    for row in cov.iter_mut() { for c in row.iter_mut() { *c /= n; } }
    let ev = eigenvalues(cov);
    let tot: f64 = ev.iter().sum();
    println!("\n--- 1. eigenvalues of 16x16 covariance (normalized blocks) ---");
    print!("share of variance per axis: ");
    for e in &ev { print!("{:.1}% ", 100.0 * e / tot); }
    println!();
    let mut cum = 0.0;
    let mut d90 = BLOCK_DIM;
    for (i, e) in ev.iter().enumerate() {
        cum += e / tot;
        if cum >= 0.90 { d90 = i + 1; break; }
    }
    println!("axes needed for 90% of variance: {d90} of {BLOCK_DIM}  (16 = pure isotropic noise)");
    println!("condition number lambda_max/lambda_min: {:.2}", ev[0] / ev[BLOCK_DIM - 1]);

    // ---- 2. pattern-only error vs k
    let tv = st.tensor(target).unwrap();
    let data = decode_dtype(&tv);
    println!("\n--- 2. pattern-only error vs catalogue size (measured on {target}) ---");
    println!("{:>8} {:>10} {:>14}", "k", "bits/blk", "rel RMSE");
    for &k in &[16usize, 64, 256, 1024, 4096] {
        let dict = PatternDict::train(&normalized, k, 20);
        let nb = data.len() / BLOCK_DIM;
        let (err, sig) = (0..nb).into_par_iter().map(|bi| {
            let mut b = [0.0f32; BLOCK_DIM];
            b.copy_from_slice(&data[bi * BLOCK_DIM..(bi + 1) * BLOCK_DIM]);
            let BlockNorm { base, scale_i8, norm } = normalize_block(&b);
            let base16 = f16::from_f32(base).to_f32();
            let s = scale_i8 as f32 / INPUT_SCALE;
            // nearest over the full (possibly >256) centroid list
            let mut bd = f32::MAX;
            let mut bi2 = 0usize;
            for (i, c) in dict.centroids.iter().enumerate() {
                let mut d = 0.0f32;
                for j in 0..BLOCK_DIM { let x = norm[j] - c[j]; d += x * x; }
                if d < bd { bd = d; bi2 = i; }
            }
            let p = dict.centroids[bi2];
            let mut e = 0.0f64;
            let mut sg = 0.0f64;
            for j in 0..BLOCK_DIM {
                let o = b[j] as f64;
                e += (o - (p[j] * s + base16) as f64).powi(2);
                sg += o * o;
            }
            (e, sg)
        }).reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
        println!("{:>8} {:>10} {:>13.2}%", k, (k as f64).log2().ceil() as u32, 100.0 * (err / sig).sqrt());
    }

    // ---- 3. matched-bitrate rival
    println!("\n--- 3. plain affine quantization at matched bitrate ---");
    println!("v1 spends 6 bytes per 16 weights = 3.00 bits/weight");
    println!("{:>8} {:>8} {:>12} {:>12}", "block", "levels", "bits/weight", "rel RMSE");
    for &(b, l) in &[(16usize, 4u32), (32, 4), (64, 4), (128, 4), (128, 8), (128, 16), (128, 64)] {
        let (e, bits) = affine_rel_rmse(&data, b, l);
        println!("{:>8} {:>8} {:>12.2} {:>11.2}%", b, l, bits, e);
    }
}
