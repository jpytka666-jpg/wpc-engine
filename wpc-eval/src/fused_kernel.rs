use std::arch::x86_64::*;
use half::f16;
use wpc_format::{
    CompressedBlock, QuantBlockV2, QuantBlockV3, QuantBlockV4, QuantBlockV5, BLOCK_SIZE,
    BLOCK_SIZE_V2, BLOCK_SIZE_V4, BLOCK_SIZE_V5, PACKED_BYTES_V4, PACKED_BYTES_V5,
};

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
    // `x` spans the whole row, so the activation index must advance with the
    // block rather than restart inside it. Getting this wrong is invisible in
    // a single-block test -- block 0 starts at x[0] either way -- and wrong
    // for every real tensor, where a row is thirty or more blocks wide.
    for (block_idx, b) in blocks.iter().enumerate() {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let row_base = block_idx * BLOCK_SIZE_V2;
        for (code_idx, &code) in b.codes.iter().enumerate() {
            let weight = zp + code as f32 * scale;
            acc += weight * x[row_base + code_idx];
        }
    }
    acc
}

/// Fused-SIMD v3 matvec.
///
/// The packing is undone into a 128-byte stack buffer first, then the same
/// AVX2 multiply-accumulate as v2 runs over it. Unpacking with shuffles
/// directly in registers would avoid the buffer, but the buffer is 128 bytes
/// -- it stays in L1 and never reaches memory, which is the only place this
/// workload is actually short of bandwidth. The simple version keeps the
/// bit-twiddling in one place, shared with the scalar path.
///
/// # Safety
/// Requires AVX2 and FMA. `x` must have at least `n_blocks * 128` readable
/// floats.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_v3_fused(
    blocks: &[QuantBlockV3],
    x: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in 0..n_blocks {
        let b = *blocks.get_unchecked(i);
        let codes = QuantBlockV3::unpack_codes(&b.packed);
        let zp = _mm256_set1_ps(b.zero_point.to_f32());
        let scl = _mm256_set1_ps(b.scale.to_f32());

        for g in 0..16 {
            let code_ptr = codes.as_ptr().add(g * 8);
            let x_ptr = x.add(i * BLOCK_SIZE_V2 + g * 8);

            let codes_u8 = _mm_loadl_epi64(code_ptr as *const __m128i);
            let codes_i32 = _mm256_cvtepu8_epi32(codes_u8);
            let codes_f32 = _mm256_cvtepi32_ps(codes_i32);

            let w = _mm256_fmadd_ps(codes_f32, scl, zp);
            let x_vec = _mm256_loadu_ps(x_ptr);
            acc = _mm256_fmadd_ps(w, x_vec, acc);
        }
    }
    hsum_avx2(acc)
}

/// Scalar v3 matvec: v2's arithmetic, reading bit-packed codes.
///
/// v3 stores four 6-bit codes per three bytes, so each weight costs a shift
/// and a mask to recover. That work lands on cores which are idle waiting for
/// memory anyway, and buys a 24% smaller read per token.
pub fn matvec_v3_scalar(blocks: &[QuantBlockV3], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (block_idx, b) in blocks.iter().enumerate() {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let row_base = block_idx * BLOCK_SIZE_V2;
        let packed = b.packed;
        for g in 0..BLOCK_SIZE_V2 / 4 {
            let b0 = packed[g * 3];
            let b1 = packed[g * 3 + 1];
            let b2 = packed[g * 3 + 2];
            let codes = [
                b0 & 0x3F,
                (b0 >> 6) | ((b1 & 0x0F) << 2),
                (b1 >> 4) | ((b2 & 0x03) << 4),
                b2 >> 2,
            ];
            for (lane, &code) in codes.iter().enumerate() {
                let weight = zp + code as f32 * scale;
                acc += weight * x[row_base + g * 4 + lane];
            }
        }
    }
    acc
}

/// Fused-SIMD v4 matvec: affine 4-bit codes, two per byte, decoded in
/// registers.
///
/// v3 unpacks into a 128-byte stack buffer because its four-codes-per-three-
/// bytes grouping does not line up with any lane width. v4 needs no buffer at
/// all: byte j carries code j in its low nibble and code j+64 in its high one,
/// so one 8-byte load widened by `_mm256_cvtepu8_epi32` yields, after a mask
/// and a shift, two full YMM registers of codes -- and the activations they
/// pair with, `x[j..j+8]` and `x[j+64..j+72]`, are both contiguous. No
/// shuffles, no buffer, two FMAs per 8 bytes read.
///
/// # Safety
/// Requires AVX2 and FMA. `x` must have at least `n_blocks * 128` readable
/// floats.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_v4_fused(
    blocks: &[QuantBlockV4],
    x: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let nibble_mask = _mm256_set1_epi32(0x0F);
    for i in 0..n_blocks {
        let b = blocks.get_unchecked(i);
        let zp = _mm256_set1_ps(b.zero_point.to_f32());
        let scl = _mm256_set1_ps(b.scale.to_f32());
        let packed_ptr = b.packed.as_ptr();
        let row = x.add(i * BLOCK_SIZE_V4);

        // 64 payload bytes in 8 groups of 8; each group covers 16 weights.
        for g in 0..PACKED_BYTES_V4 / 8 {
            let raw = _mm_loadl_epi64(packed_ptr.add(g * 8) as *const __m128i);
            let widened = _mm256_cvtepu8_epi32(raw);

            // Low nibbles -> codes g*8 .. g*8+8, activations at the same offset.
            let lo_codes = _mm256_cvtepi32_ps(_mm256_and_si256(widened, nibble_mask));
            let w_lo = _mm256_fmadd_ps(lo_codes, scl, zp);
            let x_lo = _mm256_loadu_ps(row.add(g * 8));
            acc = _mm256_fmadd_ps(w_lo, x_lo, acc);

            // High nibbles -> codes 64+g*8 .. 64+g*8+8. The byte was
            // zero-extended, so a plain shift leaves the nibble clean.
            let hi_codes = _mm256_cvtepi32_ps(_mm256_srli_epi32(widened, 4));
            let w_hi = _mm256_fmadd_ps(hi_codes, scl, zp);
            let x_hi = _mm256_loadu_ps(row.add(PACKED_BYTES_V4 + g * 8));
            acc = _mm256_fmadd_ps(w_hi, x_hi, acc);
        }
    }
    hsum_avx2(acc)
}

/// Scalar v4 matvec: the same arithmetic, for CPUs without AVX2+FMA.
///
/// Walks the payload in the stored order (low nibbles first, then high), which
/// is the SIMD kernel's order too, so the two agree term by term and not merely
/// in the final sum.
pub fn matvec_v4_scalar(blocks: &[QuantBlockV4], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (block_idx, b) in blocks.iter().enumerate() {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let row_base = block_idx * BLOCK_SIZE_V4;
        let packed = b.packed;
        for j in 0..PACKED_BYTES_V4 {
            let byte = packed[j];
            let lo = (byte & 0x0F) as f32;
            let hi = (byte >> 4) as f32;
            acc += (zp + lo * scale) * x[row_base + j];
            acc += (zp + hi * scale) * x[row_base + PACKED_BYTES_V4 + j];
        }
    }
    acc
}

/// Fused-SIMD v5 matvec: affine 2-bit codes, four per byte, decoded in
/// registers.
///
/// v4's trick taken one level down. Byte j carries codes j, j+32, j+64 and
/// j+96, so one 8-byte load widened by `_mm256_cvtepu8_epi32` yields four full
/// YMM registers of codes after a mask and three shifts -- and the four runs of
/// activations they pair with are each contiguous in `x`. No shuffles, no
/// buffer, four FMAs per 8 bytes read.
///
/// The top slot needs no mask: the byte was zero-extended into a 32-bit lane,
/// so `>> 6` leaves nothing above the two bits wanted.
///
/// # Safety
/// Requires AVX2 and FMA. `x` must have at least `n_blocks * 128` readable
/// floats.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matvec_v5_fused(
    blocks: &[QuantBlockV5],
    x: *const f32,
    n_blocks: usize,
) -> f32 {
    let mut acc = _mm256_setzero_ps();
    let code_mask = _mm256_set1_epi32(0x03);
    for i in 0..n_blocks {
        let b = blocks.get_unchecked(i);
        let zp = _mm256_set1_ps(b.zero_point.to_f32());
        let scl = _mm256_set1_ps(b.scale.to_f32());
        let packed_ptr = b.packed.as_ptr();
        let row = x.add(i * BLOCK_SIZE_V5);

        // 32 payload bytes in 4 groups of 8; each group covers 32 weights.
        for g in 0..PACKED_BYTES_V5 / 8 {
            let raw = _mm_loadl_epi64(packed_ptr.add(g * 8) as *const __m128i);
            let widened = _mm256_cvtepu8_epi32(raw);

            // Slot 0 -> codes g*8 .. g*8+8, activations at the same offset.
            let c0 = _mm256_cvtepi32_ps(_mm256_and_si256(widened, code_mask));
            let w0 = _mm256_fmadd_ps(c0, scl, zp);
            let x0 = _mm256_loadu_ps(row.add(g * 8));
            acc = _mm256_fmadd_ps(w0, x0, acc);

            // Slot 1 -> codes 32+g*8 ..
            let c1 = _mm256_cvtepi32_ps(_mm256_and_si256(
                _mm256_srli_epi32(widened, 2),
                code_mask,
            ));
            let w1 = _mm256_fmadd_ps(c1, scl, zp);
            let x1 = _mm256_loadu_ps(row.add(PACKED_BYTES_V5 + g * 8));
            acc = _mm256_fmadd_ps(w1, x1, acc);

            // Slot 2 -> codes 64+g*8 ..
            let c2 = _mm256_cvtepi32_ps(_mm256_and_si256(
                _mm256_srli_epi32(widened, 4),
                code_mask,
            ));
            let w2 = _mm256_fmadd_ps(c2, scl, zp);
            let x2 = _mm256_loadu_ps(row.add(2 * PACKED_BYTES_V5 + g * 8));
            acc = _mm256_fmadd_ps(w2, x2, acc);

            // Slot 3 -> codes 96+g*8 .. ; the lane holds only the byte, so the
            // shift alone leaves the two bits clean.
            let c3 = _mm256_cvtepi32_ps(_mm256_srli_epi32(widened, 6));
            let w3 = _mm256_fmadd_ps(c3, scl, zp);
            let x3 = _mm256_loadu_ps(row.add(3 * PACKED_BYTES_V5 + g * 8));
            acc = _mm256_fmadd_ps(w3, x3, acc);
        }
    }
    hsum_avx2(acc)
}

/// Scalar v5 matvec: the same arithmetic, for CPUs without AVX2+FMA.
///
/// Walks the payload in the stored order (slot by slot within each byte), which
/// is the SIMD kernel's order too, so the two agree term by term and not merely
/// in the final sum.
pub fn matvec_v5_scalar(blocks: &[QuantBlockV5], x: &[f32]) -> f32 {
    let mut acc = 0.0f32;
    for (block_idx, b) in blocks.iter().enumerate() {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let row_base = block_idx * BLOCK_SIZE_V5;
        let packed = b.packed;
        for j in 0..PACKED_BYTES_V5 {
            let byte = packed[j];
            for slot in 0..4 {
                let code = ((byte >> (2 * slot)) & 0x03) as f32;
                acc += (zp + code * scale) * x[row_base + slot * PACKED_BYTES_V5 + j];
            }
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

    /// Builds a multi-block row with a distinct value pattern per block.
    /// Single-block tests cannot see an activation-index bug, because block 0
    /// reads from x[0] whether or not the index advances.
    fn multi_block_fixture(n_blocks: usize) -> (Vec<QuantBlockV2>, Vec<f32>) {
        let mut blocks = Vec::with_capacity(n_blocks);
        for bi in 0..n_blocks {
            let mut codes = [0u8; BLOCK_SIZE_V2];
            for i in 0..BLOCK_SIZE_V2 {
                codes[i] = ((i + bi * 13) % 64) as u8;
            }
            blocks.push(QuantBlockV2 {
                zero_point: f16::from_f32(-0.4 + bi as f32 * 0.1),
                scale: f16::from_f32(0.017 + bi as f32 * 0.003),
                codes,
            });
        }
        // Activations differ per position, so reusing the wrong slice of x
        // changes the result instead of cancelling out.
        let x: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V2)
            .map(|i| ((i as f32) * 0.013).sin() + 0.5)
            .collect();
        (blocks, x)
    }

    fn reference_dot(blocks: &[QuantBlockV2], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (bi, b) in blocks.iter().enumerate() {
            let zp = b.zero_point.to_f32();
            let sc = b.scale.to_f32();
            let codes = b.codes;
            for i in 0..BLOCK_SIZE_V2 {
                acc += (zp + codes[i] as f32 * sc) * x[bi * BLOCK_SIZE_V2 + i];
            }
        }
        acc
    }

    #[test]
    fn scalar_v2_walks_the_whole_row_not_just_the_first_block() {
        let (blocks, x) = multi_block_fixture(7);
        let expected = reference_dot(&blocks, &x);
        let got = matvec_v2_scalar(&blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v2 multi-block: got {got}, expected {expected}");
    }

    #[test]
    fn scalar_v2_agrees_with_fused_v2_over_many_blocks() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (blocks, x) = multi_block_fixture(11);
        let scalar = matvec_v2_scalar(&blocks, &x);
        let fused = unsafe { matvec_v2_fused(&blocks, x.as_ptr(), blocks.len()) };
        let rel = (scalar - fused).abs() / fused.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar {scalar} vs fused {fused}");
    }

    #[test]
    fn scalar_v3_matches_scalar_v2_on_the_same_weights() {
        let (v2_blocks, x) = multi_block_fixture(9);
        let v3_blocks: Vec<QuantBlockV3> =
            v2_blocks.iter().map(QuantBlockV3::from_v2).collect();

        let v2 = matvec_v2_scalar(&v2_blocks, &x);
        let v3 = matvec_v3_scalar(&v3_blocks, &x);
        let rel = (v3 - v2).abs() / v2.abs().max(1e-6);
        assert!(rel < 1e-5, "v3 {v3} vs v2 {v2} on identical weights");
    }

    #[test]
    fn fused_v3_agrees_with_fused_v2_over_many_blocks() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (v2_blocks, x) = multi_block_fixture(13);
        let v3_blocks: Vec<QuantBlockV3> =
            v2_blocks.iter().map(QuantBlockV3::from_v2).collect();

        let v2 = unsafe { matvec_v2_fused(&v2_blocks, x.as_ptr(), v2_blocks.len()) };
        let v3 = unsafe { matvec_v3_fused(&v3_blocks, x.as_ptr(), v3_blocks.len()) };
        let rel = (v3 - v2).abs() / v2.abs().max(1e-6);
        assert!(rel < 1e-5, "fused v3 {v3} vs fused v2 {v2}");
    }

    #[test]
    fn fused_v3_agrees_with_scalar_v3() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (v2_blocks, x) = multi_block_fixture(6);
        let v3_blocks: Vec<QuantBlockV3> =
            v2_blocks.iter().map(QuantBlockV3::from_v2).collect();

        let s = matvec_v3_scalar(&v3_blocks, &x);
        let f = unsafe { matvec_v3_fused(&v3_blocks, x.as_ptr(), v3_blocks.len()) };
        let rel = (s - f).abs() / f.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v3 {s} vs fused v3 {f}");
    }

    #[test]
    fn scalar_v3_matches_the_reference_formula() {
        let (v2_blocks, x) = multi_block_fixture(5);
        let v3_blocks: Vec<QuantBlockV3> =
            v2_blocks.iter().map(QuantBlockV3::from_v2).collect();
        let expected = reference_dot(&v2_blocks, &x);
        let got = matvec_v3_scalar(&v3_blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "v3 {got} vs reference {expected}");
    }

    // -------------------------------------------------------------------------
    // v4
    // -------------------------------------------------------------------------

    /// v4 blocks with a distinct code pattern per block, and activations that
    /// differ at every position. A kernel that read the wrong half of the block
    /// -- or restarted `x` at each block -- would still pass a single-block,
    /// constant-activation test; it cannot pass this one.
    fn multi_block_fixture_v4(n_blocks: usize) -> (Vec<QuantBlockV4>, Vec<f32>) {
        let mut blocks = Vec::with_capacity(n_blocks);
        for bi in 0..n_blocks {
            let mut codes = [0u8; BLOCK_SIZE_V4];
            for i in 0..BLOCK_SIZE_V4 {
                codes[i] = ((i * 3 + bi * 5) % 16) as u8;
            }
            blocks.push(QuantBlockV4 {
                zero_point: f16::from_f32(-0.4 + bi as f32 * 0.1),
                scale: f16::from_f32(0.017 + bi as f32 * 0.003),
                packed: QuantBlockV4::pack_codes(&codes),
            });
        }
        let x: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V4)
            .map(|i| ((i as f32) * 0.013).sin() + 0.5)
            .collect();
        (blocks, x)
    }

    /// Reference dot product straight from the documented decode formula, in
    /// natural weight order -- deliberately NOT in the packed order the kernels
    /// walk, so an agreement means the layout is right and not just consistent.
    fn reference_dot_v4(blocks: &[QuantBlockV4], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (bi, b) in blocks.iter().enumerate() {
            let zp = b.zero_point.to_f32();
            let sc = b.scale.to_f32();
            let packed = b.packed;
            let codes = QuantBlockV4::unpack_codes(&packed);
            for i in 0..BLOCK_SIZE_V4 {
                acc += (zp + codes[i] as f32 * sc) * x[bi * BLOCK_SIZE_V4 + i];
            }
        }
        acc
    }

    #[test]
    fn scalar_v4_matches_the_reference_formula() {
        let (blocks, x) = multi_block_fixture_v4(5);
        let expected = reference_dot_v4(&blocks, &x);
        let got = matvec_v4_scalar(&blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v4 {got} vs reference {expected}");
    }

    #[test]
    fn scalar_v4_walks_the_whole_row_not_just_the_first_block() {
        let (blocks, x) = multi_block_fixture_v4(7);
        let expected = reference_dot_v4(&blocks, &x);
        let got = matvec_v4_scalar(&blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v4 multi-block: got {got}, expected {expected}");
    }

    #[test]
    fn fused_v4_agrees_with_scalar_v4_over_many_blocks() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: CPU lacks AVX2+FMA");
            return;
        }
        let (blocks, x) = multi_block_fixture_v4(13);
        let s = matvec_v4_scalar(&blocks, &x);
        let f = unsafe { matvec_v4_fused(&blocks, x.as_ptr(), blocks.len()) };
        let rel = (s - f).abs() / f.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v4 {s} vs fused v4 {f}");
    }

    #[test]
    fn fused_v4_matches_the_reference_formula() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (blocks, x) = multi_block_fixture_v4(9);
        let expected = reference_dot_v4(&blocks, &x);
        let got = unsafe { matvec_v4_fused(&blocks, x.as_ptr(), blocks.len()) };
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "fused v4 {got} vs reference {expected}");
    }

    /// Every code value in every nibble position, checked through the kernel
    /// rather than only through pack/unpack: a mask applied to the wrong half
    /// shows up here as a wrong dot product.
    #[test]
    fn fused_v4_handles_every_code_value() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        for v in 0u8..=15 {
            let codes = [v; BLOCK_SIZE_V4];
            let block = QuantBlockV4 {
                zero_point: f16::from_f32(-0.25),
                scale: f16::from_f32(0.03125), // exact in f16, so the check is tight
                packed: QuantBlockV4::pack_codes(&codes),
            };
            let x: Vec<f32> = (0..BLOCK_SIZE_V4).map(|i| (i as f32) * 0.01 - 0.6).collect();

            let got = unsafe { matvec_v4_fused(std::slice::from_ref(&block), x.as_ptr(), 1) };
            let expected = reference_dot_v4(std::slice::from_ref(&block), &x);
            let rel = (got - expected).abs() / expected.abs().max(1e-6);
            assert!(rel < 1e-4, "code {v}: fused {got} vs reference {expected}");
        }
    }

    /// The end-to-end shape of the claim: encode real-looking weights at v3 and
    /// at v4, and check the v4 dot product is close to the v3 one. Not equal --
    /// v4 is genuinely coarser -- but close enough that a model built on it is
    /// computing the same function.
    #[test]
    fn fused_v4_tracks_fused_v3_on_the_same_weights() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let n_blocks = 11;
        let weights: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V4)
            .map(|i| ((i as f32) * 0.011).sin() * 0.04)
            .collect();
        let x: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V4)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();

        let v3_blocks = wpc_core::quant_encoder::encode_tensor_v3(&weights);
        let v4_blocks = wpc_core::quant_encoder::encode_tensor_v4(&weights);

        let exact = unsafe { matvec_fp32_baseline(weights.as_ptr(), x.as_ptr(), weights.len()) };
        let y3 = unsafe { matvec_v3_fused(&v3_blocks, x.as_ptr(), v3_blocks.len()) };
        let y4 = unsafe { matvec_v4_fused(&v4_blocks, x.as_ptr(), v4_blocks.len()) };

        // Scaled against the magnitude of the terms rather than against the sum,
        // which is a near-cancellation of positives and negatives and so is a
        // misleading denominator.
        let scale: f32 = weights.iter().zip(x.iter()).map(|(w, a)| (w * a).abs()).sum();
        let e3 = (y3 - exact).abs() / scale;
        let e4 = (y4 - exact).abs() / scale;
        assert!(e4 < 0.02, "v4 dot product off by {e4} of term magnitude (v3: {e3})");
    }

    // -------------------------------------------------------------------------
    // v5
    // -------------------------------------------------------------------------

    /// v5 blocks with a distinct code pattern per block, and activations that
    /// differ at every position. A kernel that read the wrong 2-bit slot -- or
    /// restarted `x` at each block -- would still pass a single-block,
    /// constant-activation test; it cannot pass this one.
    fn multi_block_fixture_v5(n_blocks: usize) -> (Vec<QuantBlockV5>, Vec<f32>) {
        let mut blocks = Vec::with_capacity(n_blocks);
        for bi in 0..n_blocks {
            let mut codes = [0u8; BLOCK_SIZE_V5];
            for i in 0..BLOCK_SIZE_V5 {
                codes[i] = ((i * 3 + bi * 5 + i / 32) % 4) as u8;
            }
            blocks.push(QuantBlockV5 {
                zero_point: f16::from_f32(-0.4 + bi as f32 * 0.1),
                scale: f16::from_f32(0.017 + bi as f32 * 0.003),
                packed: QuantBlockV5::pack_codes(&codes),
            });
        }
        let x: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V5)
            .map(|i| ((i as f32) * 0.013).sin() + 0.5)
            .collect();
        (blocks, x)
    }

    /// Reference dot product straight from the documented decode formula, in
    /// natural weight order -- deliberately NOT in the packed order the kernels
    /// walk, so an agreement means the layout is right and not just consistent.
    fn reference_dot_v5(blocks: &[QuantBlockV5], x: &[f32]) -> f32 {
        let mut acc = 0.0f32;
        for (bi, b) in blocks.iter().enumerate() {
            let zp = b.zero_point.to_f32();
            let sc = b.scale.to_f32();
            let packed = b.packed;
            let codes = QuantBlockV5::unpack_codes(&packed);
            for i in 0..BLOCK_SIZE_V5 {
                acc += (zp + codes[i] as f32 * sc) * x[bi * BLOCK_SIZE_V5 + i];
            }
        }
        acc
    }

    #[test]
    fn scalar_v5_matches_the_reference_formula() {
        let (blocks, x) = multi_block_fixture_v5(5);
        let expected = reference_dot_v5(&blocks, &x);
        let got = matvec_v5_scalar(&blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v5 {got} vs reference {expected}");
    }

    #[test]
    fn scalar_v5_walks_the_whole_row_not_just_the_first_block() {
        let (blocks, x) = multi_block_fixture_v5(7);
        let expected = reference_dot_v5(&blocks, &x);
        let got = matvec_v5_scalar(&blocks, &x);
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v5 multi-block: got {got}, expected {expected}");
    }

    #[test]
    fn fused_v5_agrees_with_scalar_v5_over_many_blocks() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("skipping: CPU lacks AVX2+FMA");
            return;
        }
        let (blocks, x) = multi_block_fixture_v5(13);
        let s = matvec_v5_scalar(&blocks, &x);
        let f = unsafe { matvec_v5_fused(&blocks, x.as_ptr(), blocks.len()) };
        let rel = (s - f).abs() / f.abs().max(1e-6);
        assert!(rel < 1e-4, "scalar v5 {s} vs fused v5 {f}");
    }

    #[test]
    fn fused_v5_matches_the_reference_formula() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let (blocks, x) = multi_block_fixture_v5(9);
        let expected = reference_dot_v5(&blocks, &x);
        let got = unsafe { matvec_v5_fused(&blocks, x.as_ptr(), blocks.len()) };
        let rel = (got - expected).abs() / expected.abs().max(1e-6);
        assert!(rel < 1e-4, "fused v5 {got} vs reference {expected}");
    }

    /// Every code value in every one of the four 2-bit slots, checked through
    /// the kernel rather than only through pack/unpack: a mask or shift applied
    /// to the wrong slot shows up here as a wrong dot product.
    #[test]
    fn fused_v5_handles_every_code_value() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        for v in 0u8..=3 {
            let codes = [v; BLOCK_SIZE_V5];
            let block = QuantBlockV5 {
                zero_point: f16::from_f32(-0.25),
                scale: f16::from_f32(0.03125), // exact in f16, so the check is tight
                packed: QuantBlockV5::pack_codes(&codes),
            };
            let x: Vec<f32> = (0..BLOCK_SIZE_V5).map(|i| (i as f32) * 0.01 - 0.6).collect();

            let got = unsafe { matvec_v5_fused(std::slice::from_ref(&block), x.as_ptr(), 1) };
            let expected = reference_dot_v5(std::slice::from_ref(&block), &x);
            let rel = (got - expected).abs() / expected.abs().max(1e-6);
            assert!(rel < 1e-4, "code {v}: fused {got} vs reference {expected}");
        }
    }

    /// Each quarter of the block set to a different constant, so a kernel that
    /// paired slot 1's codes with slot 2's activations cannot pass. That swap is
    /// the specific failure the 32-apart layout exists to avoid, and a
    /// uniform-code test is blind to it.
    #[test]
    fn fused_v5_pairs_each_slot_with_the_right_activations() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let mut codes = [0u8; BLOCK_SIZE_V5];
        for i in 0..BLOCK_SIZE_V5 {
            codes[i] = (i / 32) as u8; // quarter 0 -> 0, quarter 1 -> 1, ...
        }
        let block = QuantBlockV5 {
            zero_point: f16::from_f32(0.0),
            scale: f16::from_f32(1.0),
            packed: QuantBlockV5::pack_codes(&codes),
        };
        // Activations distinct per quarter too, so a swap changes the sum.
        let x: Vec<f32> = (0..BLOCK_SIZE_V5).map(|i| (i / 32) as f32 + 1.0).collect();

        let got = unsafe { matvec_v5_fused(std::slice::from_ref(&block), x.as_ptr(), 1) };
        // Per quarter q: 32 weights of value q against activations of q+1.
        // 32 * (0*1 + 1*2 + 2*3 + 3*4) = 32 * 20 = 640.
        let expected = 640.0f32;
        assert!(
            (got - expected).abs() < 1e-3,
            "fused v5 {got} vs hand-computed {expected}: slots paired with wrong activations"
        );
    }

    /// The end-to-end shape of the claim: encode real-looking weights at v4 and
    /// at v5 and check the v5 dot product still tracks the exact one. The bound
    /// is looser than v4's on purpose -- four levels is genuinely coarse -- but
    /// a broken code path leaves the result nowhere near.
    #[test]
    fn fused_v5_tracks_the_exact_dot_product() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return;
        }
        let n_blocks = 11;
        let weights: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V5)
            .map(|i| ((i as f32) * 0.011).sin() * 0.04)
            .collect();
        let x: Vec<f32> = (0..n_blocks * BLOCK_SIZE_V5)
            .map(|i| ((i as f32) * 0.019).cos())
            .collect();

        let v4_blocks = wpc_core::quant_encoder::encode_tensor_v4(&weights);
        let v5_blocks = wpc_core::quant_encoder::encode_tensor_v5(&weights);

        let exact = unsafe { matvec_fp32_baseline(weights.as_ptr(), x.as_ptr(), weights.len()) };
        let y4 = unsafe { matvec_v4_fused(&v4_blocks, x.as_ptr(), v4_blocks.len()) };
        let y5 = unsafe { matvec_v5_fused(&v5_blocks, x.as_ptr(), v5_blocks.len()) };

        let scale: f32 = weights.iter().zip(x.iter()).map(|(w, a)| (w * a).abs()).sum();
        let e4 = (y4 - exact).abs() / scale;
        let e5 = (y5 - exact).abs() / scale;
        assert!(e5 < 0.15, "v5 dot product off by {e5} of term magnitude (v4: {e4})");
    }
}
