use std::arch::x86_64::*;
use half::f16;
use wpc_format::{CompressedBlock, QuantBlockV2, BLOCK_SIZE, BLOCK_SIZE_V2};

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

/// Fused-SIMD matvec for v2 format: `y = W @ x` where W is stored as a sequence
/// of QuantBlockV2 blocks (132 bytes each: 4 bytes zero_point+scale, 128 u8 codes).
///
/// Each block's 128 weights are decoded via: value = zero_point + code * scale,
/// then immediately consumed by FMA. Branch-free decode, no dictionary lookups.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_v2_fused(
    blocks: &[QuantBlockV2],
    x: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in 0..n_blocks {
        let b = *blocks.get_unchecked(i);
        let zp = _mm256_set1_ps(b.zero_point.to_f32());
        let scl = _mm256_set1_ps(b.scale.to_f32());

        // Process 128 codes in 16 groups of 8
        for g in 0..16 {
            let code_ptr = b.codes.as_ptr().add(g * 8);
            let x_ptr = x.add(i * BLOCK_SIZE_V2 + g * 8);

            // Load 8 u8 codes, zero-extend to i32, convert to f32
            let codes_u8 = _mm_loadl_epi64(code_ptr as *const __m128i);
            let codes_i32 = _mm256_cvtepu8_epi32(codes_u8);
            let codes_f32 = _mm256_cvtepi32_ps(codes_i32);

            // Decode: w = zero_point + code * scale
            let w = _mm256_fmadd_ps(codes_f32, scl, zp);

            // Load 8 activations and FMA
            let x_vec = _mm256_loadu_ps(x_ptr);
            acc = _mm256_fmadd_ps(w, x_vec, acc);
        }
    }
    hsum_avx2(acc)
}

/// Scalar fallback for v2 matvec (no AVX2 available).
pub fn matvec_v2_scalar(blocks: &[QuantBlockV2], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for b in blocks {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        for (code_idx, &code) in b.codes.iter().enumerate() {
            let weight = zp + code as f32 * scale;
            acc += weight * x[code_idx];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    const INV_INPUT_SCALE: f32 = 1.0 / 127.0;

    /// This is the test that would have caught the 127x contract bug: it feeds
    /// `matvec_wpc_fused` a pattern array in the documented on-disk convention
    /// (pre-divided by INPUT_SCALE) and checks the generated weight against the
    /// same formula wpc-eval's RMSE path and wpc-core's encoder use.
    #[test]
    fn fused_kernel_matches_documented_decode_formula() {
        if !is_x86_feature_detected!("avx2")
            || !is_x86_feature_detected!("fma")
            || !is_x86_feature_detected!("f16c")
        {
            eprintln!("skipping: CPU lacks AVX2+FMA+F16C");
            return;
        }

        let raw_centroid = [0.3_f32; BLOCK_SIZE]; // normalize_block()'s [-1,1] space
        let on_disk_pattern: Vec<f32> = raw_centroid.iter().map(|&v| v / 127.0).collect();
        let raw_residual = [12.5_f32; BLOCK_SIZE]; // exactly f16-representable; pre-INPUT_SCALE-multiplied as encoder.rs stores it
        let residual_f16: Vec<f16> = raw_residual.iter().map(|&v| f16::from_f32(v)).collect();

        let scale: i8 = 50;
        let base: f32 = 0.2;
        let block = CompressedBlock {
            pattern_id: 0,
            residual_id: 0,
            base_value: f16::from_f32(base),
            scale,
        };
        let x = [1.0f32; BLOCK_SIZE]; // activations: sum(w) is the expected dot product

        let y = unsafe {
            matvec_wpc_fused(
                std::slice::from_ref(&block),
                on_disk_pattern.as_ptr(),
                residual_f16.as_ptr(),
                x.as_ptr(),
                1,
            )
        };

        let mut expected = 0.0f32;
        for j in 0..BLOCK_SIZE {
            let w = raw_centroid[j] * (scale as f32 / 127.0) + base + raw_residual[j] * INV_INPUT_SCALE;
            expected += w;
        }

        let rel_err = (y - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 1e-3,
            "fused kernel y={y} vs documented-decode expected={expected} (rel_err={rel_err})"
        );
    }

    #[test]
    fn v2_matvec_varied_codes() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: CPU lacks AVX2+FMA");
            return;
        }

        // Create a v2 block with varied code values (not all constant)
        let mut codes = [0u8; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            codes[i] = (i % 64) as u8; // Use varied values 0..63
        }
        let block = QuantBlockV2 {
            zero_point: f16::from_f32(-1.0),
            scale: f16::from_f32(0.032),
            codes,
        };

        // Create varied activations
        let mut x = [0.0f32; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            x[i] = (i as f32 * 0.01 - 0.5) % 1.0; // Vary between -0.5 and 0.5
        }

        // Compute scalar reference
        let mut expected = 0.0f32;
        let zp = block.zero_point.to_f32();
        let scale = block.scale.to_f32();
        for i in 0..BLOCK_SIZE_V2 {
            let w = zp + block.codes[i] as f32 * scale;
            expected += w * x[i];
        }

        // Compute v2_fused
        let y = unsafe {
            matvec_v2_fused(std::slice::from_ref(&block), x.as_ptr(), 1)
        };

        let rel_err = (y - expected).abs() / expected.abs().max(1e-6);
        assert!(
            rel_err < 1e-2,
            "v2_matvec y={y} vs expected={expected} (rel_err={rel_err})"
        );
    }

    #[test]
    fn v2_scalar_matches_reference() {
        let mut codes = [0u8; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            codes[i] = (i % 64) as u8;
        }
        let block = QuantBlockV2 {
            zero_point: f16::from_f32(2.5),
            scale: f16::from_f32(0.05),
            codes,
        };

        let mut x = [1.0f32; BLOCK_SIZE_V2];
        for i in 0..10 {
            x[i] = 2.0;
        }

        let y = matvec_v2_scalar(std::slice::from_ref(&block), &x);

        let mut expected = 0.0f32;
        let zp = block.zero_point.to_f32();
        let scale = block.scale.to_f32();
        for i in 0..BLOCK_SIZE_V2 {
            let w = zp + block.codes[i] as f32 * scale;
            expected += w * x[i];
        }

        let rel_err = (y - expected).abs() / expected.abs().max(1e-6);
        assert!(rel_err < 1e-5, "scalar v2 y={y} vs expected={expected}");
    }
}
