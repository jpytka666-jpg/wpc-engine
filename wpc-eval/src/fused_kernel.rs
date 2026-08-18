use std::arch::x86_64::*;
use half::f16;
use wpc_format::{CompressedBlock, BLOCK_SIZE};

/// Sum of horizontal adds of a __m256. Standard AVX2 trick.
#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn hsum_avx2(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let sh = _mm_movehdup_ps(s);
    let sums = _mm_add_ps(s, sh);
    let sh2 = _mm_movehl_ps(sh, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, sh2))
}

/// Fused-SIMD matvec: `y = W @ x` where W is stored as a sequence of
/// 6-byte WPC blocks and `x` is a contiguous f32 vector of length n_blocks*16.
///
/// This function NEVER writes the decoded 16 weights anywhere except into
/// YMM registers. Each block's 16 weights are generated and immediately
/// consumed by `_mm256_fmadd_ps` in the matvec accumulator.
#[target_feature(enable = "avx2,fma,f16c")]
pub unsafe fn matvec_wpc_fused(
    blocks: &[CompressedBlock],
    patterns: *const f32,
    residuals: *const f16,
    x: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in 0..n_blocks {
        let b = *blocks.get_unchecked(i);
        // -- 1. Load 16 activations --
        let x_ptr = x.add(i * BLOCK_SIZE);
        let x_lo = _mm256_loadu_ps(x_ptr);
        let x_hi = _mm256_loadu_ps(x_ptr.add(8));

        // -- 2. Broadcast base_value and scale --
        let v_base = _mm256_set1_ps(b.base_value.to_f32());
        let v_scl  = _mm256_set1_ps(b.scale as f32);

        // -- 3. Load pattern (pre-scaled by 1/127) and GENERATE weights --
        let p_ptr = patterns.add(b.pattern_id as usize * BLOCK_SIZE);
        let p_lo = _mm256_loadu_ps(p_ptr);
        let p_hi = _mm256_loadu_ps(p_ptr.add(8));
        // FMA #1: generated_lo = pattern_lo * scale + base
        // FMA #1': generated_hi = pattern_hi * scale + base
        let w_lo = _mm256_fmadd_ps(p_lo, v_scl, v_base);
        let w_hi = _mm256_fmadd_ps(p_hi, v_scl, v_base);

        // -- 4. Add residual (F16C widens 8 f16 -> 8 f32 per _mm256_cvtph_ps) --
        let r_ptr = residuals.add(b.residual_id as usize * BLOCK_SIZE);
        let r_packed_lo = _mm_loadu_si128(r_ptr as *const __m128i);
        let r_packed_hi = _mm_loadu_si128(r_ptr.add(8) as *const __m128i);
        // We have to divide by INPUT_SCALE to undo the encoder's pre-scaling.
        const INV_INPUT_SCALE: f32 = 1.0 / 127.0;
        let v_inv = _mm256_set1_ps(INV_INPUT_SCALE);
        let r_lo_f32 = _mm256_mul_ps(_mm256_cvtph_ps(r_packed_lo), v_inv);
        let r_hi_f32 = _mm256_mul_ps(_mm256_cvtph_ps(r_packed_hi), v_inv);
        let w_lo = _mm256_add_ps(w_lo, r_lo_f32);
        let w_hi = _mm256_add_ps(w_hi, r_hi_f32);

        // -- 5. MATMUL FMA: y_partial += activations * generated_weights --
        acc = _mm256_fmadd_ps(x_lo, w_lo, acc);
        acc = _mm256_fmadd_ps(x_hi, w_hi, acc);
    }
    hsum_avx2(acc)
}

#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_fp32_baseline(w: *const f32, x: *const f32, n: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        let wv = _mm256_loadu_ps(w.add(i));
        let xv = _mm256_loadu_ps(x.add(i));
        acc = _mm256_fmadd_ps(wv, xv, acc);
        i += 8;
    }
    let mut total = hsum_avx2(acc);
    while i < n {
        total += *w.add(i) * *x.add(i);
        i += 1;
    }
    total
}
