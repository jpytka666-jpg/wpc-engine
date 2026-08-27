use rayon::prelude::*;
use wpc_format::{
    QuantBlockV2, QuantBlockV3, QuantBlockV4, BLOCK_SIZE_V2, BLOCK_SIZE_V4, MAX_CODE_V4,
};

/// Quantize a block of 128 f32 values to v2 format (6-bit affine).
pub fn affine_quant_block(weights: &[f32; BLOCK_SIZE_V2]) -> QuantBlockV2 {
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for &w in weights {
        if w < min_v {
            min_v = w;
        }
        if w > max_v {
            max_v = w;
        }
    }
    let levels = 63.0_f32; // 2^6 - 1
    let scale = if max_v > min_v {
        (max_v - min_v) / levels
    } else {
        1.0
    };
    let mut codes = [0u8; BLOCK_SIZE_V2];
    for i in 0..BLOCK_SIZE_V2 {
        let code = ((weights[i] - min_v) / scale).round().clamp(0.0, levels);
        codes[i] = code as u8;
    }
    QuantBlockV2 {
        zero_point: half::f16::from_f32(min_v),
        scale: half::f16::from_f32(scale),
        codes,
    }
}

/// Encode a flat tensor into v2 quantized blocks.
pub fn encode_tensor_v2(data: &[f32]) -> Vec<QuantBlockV2> {
    assert_eq!(
        data.len() % BLOCK_SIZE_V2,
        0,
        "tensor length must be a multiple of {BLOCK_SIZE_V2}"
    );
    let n_blocks = data.len() / BLOCK_SIZE_V2;
    (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let mut block = [0.0f32; BLOCK_SIZE_V2];
            block.copy_from_slice(&data[i * BLOCK_SIZE_V2..(i + 1) * BLOCK_SIZE_V2]);
            affine_quant_block(&block)
        })
        .collect()
}

/// Quantize a block to v3: identical arithmetic to v2, packed layout.
///
/// Deliberately built by re-laying-out the v2 block rather than by
/// reimplementing the quantization. That makes "v3 reconstructs exactly like
/// v2" a property of the construction instead of something to keep in sync.
pub fn affine_quant_block_v3(weights: &[f32; BLOCK_SIZE_V2]) -> QuantBlockV3 {
    QuantBlockV3::from_v2(&affine_quant_block(weights))
}

/// Encode a flat tensor into v3 packed blocks.
pub fn encode_tensor_v3(data: &[f32]) -> Vec<QuantBlockV3> {
    assert_eq!(
        data.len() % BLOCK_SIZE_V2,
        0,
        "tensor length must be a multiple of {BLOCK_SIZE_V2}"
    );
    let n_blocks = data.len() / BLOCK_SIZE_V2;
    (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let mut block = [0.0f32; BLOCK_SIZE_V2];
            block.copy_from_slice(&data[i * BLOCK_SIZE_V2..(i + 1) * BLOCK_SIZE_V2]);
            affine_quant_block_v3(&block)
        })
        .collect()
}

/// Quantize a block of 128 f32 values to v4 format (4-bit affine).
///
/// The same min/max affine fit as v2 and v3, over 16 levels instead of 64.
/// Unlike v3 this is NOT a re-layout of a v2 block: fewer levels means a
/// genuinely coarser quantization, so the codes are computed from scratch and
/// the reconstruction error is expected to be larger. That trade is the whole
/// point -- 4.25 bits/weight against 6.25 puts a 4B model inside a 4 GB card.
pub fn affine_quant_block_v4(weights: &[f32; BLOCK_SIZE_V4]) -> QuantBlockV4 {
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for &w in weights {
        if w < min_v {
            min_v = w;
        }
        if w > max_v {
            max_v = w;
        }
    }
    let levels = MAX_CODE_V4 as f32; // 2^4 - 1
    let scale = if max_v > min_v {
        (max_v - min_v) / levels
    } else {
        1.0
    };
    let mut codes = [0u8; BLOCK_SIZE_V4];
    for i in 0..BLOCK_SIZE_V4 {
        let code = ((weights[i] - min_v) / scale).round().clamp(0.0, levels);
        codes[i] = code as u8;
    }
    QuantBlockV4 {
        zero_point: half::f16::from_f32(min_v),
        scale: half::f16::from_f32(scale),
        packed: QuantBlockV4::pack_codes(&codes),
    }
}

/// Encode a flat tensor into v4 packed blocks.
pub fn encode_tensor_v4(data: &[f32]) -> Vec<QuantBlockV4> {
    assert_eq!(
        data.len() % BLOCK_SIZE_V4,
        0,
        "tensor length must be a multiple of {BLOCK_SIZE_V4}"
    );
    let n_blocks = data.len() / BLOCK_SIZE_V4;
    (0..n_blocks)
        .into_par_iter()
        .map(|i| {
            let mut block = [0.0f32; BLOCK_SIZE_V4];
            block.copy_from_slice(&data[i * BLOCK_SIZE_V4..(i + 1) * BLOCK_SIZE_V4]);
            affine_quant_block_v4(&block)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v3_packing_round_trips_every_code_value() {
        // Every 6-bit value in every lane of the 4-code group, so a bad shift
        // cannot hide in a lane the test never exercises.
        let mut codes = [0u8; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            codes[i] = ((i * 7 + i / 4) % 64) as u8;
        }
        let packed = QuantBlockV3::pack_codes(&codes);
        assert_eq!(QuantBlockV3::unpack_codes(&packed), codes);

        for v in 0u8..64 {
            let uniform = [v; BLOCK_SIZE_V2];
            let p = QuantBlockV3::pack_codes(&uniform);
            assert_eq!(
                QuantBlockV3::unpack_codes(&p),
                uniform,
                "code value {v} failed"
            );
        }
    }

    #[test]
    fn v3_reconstructs_bit_identically_to_v2() {
        let mut weights = [0.0f32; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            weights[i] = (i as f32 * 0.37).sin() * 0.05;
        }
        let v2 = affine_quant_block(&weights);
        let v3 = affine_quant_block_v3(&weights);

        // Fields are copied out first: both blocks are `repr(C, packed)`, so
        // borrowing a field directly would be an unaligned reference.
        let (v2_zp, v2_sc, v2_codes) = (v2.zero_point, v2.scale, v2.codes);
        let (v3_zp, v3_sc, v3_packed) = (v3.zero_point, v3.scale, v3.packed);

        assert_eq!(v3_zp, v2_zp);
        assert_eq!(v3_sc, v2_sc);
        let unpacked = QuantBlockV3::unpack_codes(&v3_packed);
        assert_eq!(unpacked, v2_codes);

        // And the decoded values themselves, not just the codes.
        let zp = v2_zp.to_f32();
        let sc = v2_sc.to_f32();
        for i in 0..BLOCK_SIZE_V2 {
            let a = zp + v2_codes[i] as f32 * sc;
            let b = zp + unpacked[i] as f32 * sc;
            assert_eq!(a.to_bits(), b.to_bits(), "value {i} differs");
        }
    }

    #[test]
    fn v3_block_is_one_hundred_bytes() {
        assert_eq!(QuantBlockV3::SIZE, 100);
        // 100 bytes / 128 values = 6.25 bits per value, against v2's 8.25.
        assert_eq!(QuantBlockV2::SIZE, 132);
    }

    #[test]
    fn v3_survives_the_disk_round_trip() {
        let mut weights = [0.0f32; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            weights[i] = i as f32 * 0.05 - 3.2;
        }
        let block = affine_quant_block_v3(&weights);
        let bytes = block.to_le_bytes();
        assert_eq!(QuantBlockV3::from_le_bytes(&bytes), block);
    }

    #[test]
    fn test_affine_quant_block() {
        // Create a varied test block: linear ramp
        let mut weights = [0.0f32; BLOCK_SIZE_V2];
        for i in 0..BLOCK_SIZE_V2 {
            weights[i] = i as f32 * 0.05 - 3.2; // ramp from -3.2 to 3.2
        }

        let block = affine_quant_block(&weights);

        // Decode and check reconstruction error
        let mut reconstructed = [0.0f32; BLOCK_SIZE_V2];
        let zp = block.zero_point.to_f32();
        let scale = block.scale.to_f32();
        let mut sum_sq_error = 0.0f32;
        let mut sum_sq_original = 0.0f32;

        for i in 0..BLOCK_SIZE_V2 {
            let decoded = zp + block.codes[i] as f32 * scale;
            reconstructed[i] = decoded;
            let error = (decoded - weights[i]).abs();
            sum_sq_error += error * error;
            sum_sq_original += weights[i].abs().max(1e-6) * weights[i].abs().max(1e-6);
        }

        let rmse = (sum_sq_error / BLOCK_SIZE_V2 as f32).sqrt();
        let rel_rmse = rmse / (sum_sq_original / BLOCK_SIZE_V2 as f32).sqrt();

        // 6-bit quantization on a ramp should have small relative error
        assert!(
            rel_rmse < 0.05,
            "relative RMSE {rel_rmse} too high (should be <5% for this block)"
        );
    }

    // -------------------------------------------------------------------------
    // v4
    // -------------------------------------------------------------------------

    #[test]
    fn v4_block_is_sixty_eight_bytes() {
        assert_eq!(QuantBlockV4::SIZE, 68);
        // 68 bytes / 128 values = 4.25 bits per value, against v3's 6.25.
        assert_eq!(QuantBlockV3::SIZE, 100);
    }

    #[test]
    fn v4_packing_round_trips_every_code_value() {
        // Every 4-bit value in both nibble positions, plus a mixed pattern
        // where the low half and the high half of the block differ.
        for v in 0u8..=MAX_CODE_V4 {
            let uniform = [v; BLOCK_SIZE_V4];
            let p = QuantBlockV4::pack_codes(&uniform);
            assert_eq!(
                QuantBlockV4::unpack_codes(&p),
                uniform,
                "code value {v} failed"
            );
        }

        let mut codes = [0u8; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            codes[i] = ((i * 3 + i / 8) % 16) as u8;
        }
        let packed = QuantBlockV4::pack_codes(&codes);
        assert_eq!(QuantBlockV4::unpack_codes(&packed), codes);
    }

    #[test]
    fn v4_encoder_never_emits_a_code_above_fifteen() {
        // A code out of range would be silently truncated by pack_codes and
        // decode as a different weight, so the clamp is checked directly.
        let mut weights = [0.0f32; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            weights[i] = (i as f32 * 0.91).sin() * 12.5 - 3.0;
        }
        let block = affine_quant_block_v4(&weights);
        let packed = block.packed;
        for &c in QuantBlockV4::unpack_codes(&packed).iter() {
            assert!(c <= MAX_CODE_V4, "code {c} out of range");
        }
    }

    #[test]
    fn v4_reconstruction_error_stays_within_one_step() {
        // The affine contract: every weight is within half a step of its
        // reconstruction, where a step is `scale`. This is the property that
        // makes the error bounded rather than merely "usually small".
        let mut weights = [0.0f32; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            weights[i] = (i as f32 * 0.37).sin() * 0.05;
        }
        let block = affine_quant_block_v4(&weights);
        let (zp, sc, packed) = (block.zero_point, block.scale, block.packed);
        let zp = zp.to_f32();
        let sc = sc.to_f32();
        let codes = QuantBlockV4::unpack_codes(&packed);

        for i in 0..BLOCK_SIZE_V4 {
            let decoded = zp + codes[i] as f32 * sc;
            let err = (decoded - weights[i]).abs();
            // Half a step, plus slack for zero_point/scale themselves being f16.
            assert!(
                err <= sc * 0.5 + sc * 0.05 + 1e-6,
                "value {i}: |{decoded} - {}| = {err} exceeds half a step ({})",
                weights[i],
                sc * 0.5
            );
        }
    }

    #[test]
    fn v4_is_coarser_than_v3_but_still_close() {
        // The honest comparison: v4 must cost accuracy (it has 16 levels, not
        // 64) but must not collapse. Roughly 4x the step size means roughly 4x
        // the RMSE, so a factor of 8 is a generous ceiling that still fails
        // loudly if the code path is wrong rather than merely coarser.
        let mut weights = [0.0f32; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            weights[i] = (i as f32 * 0.13).sin() * 0.04 + (i as f32 * 0.7).cos() * 0.01;
        }

        let v3 = affine_quant_block_v3(&weights);
        let v4 = affine_quant_block_v4(&weights);

        let (v3_zp, v3_sc, v3_packed) = (v3.zero_point, v3.scale, v3.packed);
        let (v4_zp, v4_sc, v4_packed) = (v4.zero_point, v4.scale, v4.packed);
        let v3_codes = QuantBlockV3::unpack_codes(&v3_packed);
        let v4_codes = QuantBlockV4::unpack_codes(&v4_packed);

        let mut e3 = 0.0f32;
        let mut e4 = 0.0f32;
        for i in 0..BLOCK_SIZE_V4 {
            let d3 = v3_zp.to_f32() + v3_codes[i] as f32 * v3_sc.to_f32() - weights[i];
            let d4 = v4_zp.to_f32() + v4_codes[i] as f32 * v4_sc.to_f32() - weights[i];
            e3 += d3 * d3;
            e4 += d4 * d4;
        }
        let rmse3 = (e3 / BLOCK_SIZE_V4 as f32).sqrt();
        let rmse4 = (e4 / BLOCK_SIZE_V4 as f32).sqrt();

        assert!(
            rmse4 >= rmse3,
            "v4 RMSE {rmse4} should not beat v3's {rmse3}"
        );
        assert!(
            rmse4 < rmse3 * 8.0,
            "v4 RMSE {rmse4} is far worse than 4x v3's {rmse3}"
        );
    }

    #[test]
    fn v4_survives_the_disk_round_trip() {
        let mut weights = [0.0f32; BLOCK_SIZE_V4];
        for i in 0..BLOCK_SIZE_V4 {
            weights[i] = i as f32 * 0.05 - 3.2;
        }
        let block = affine_quant_block_v4(&weights);
        let bytes = block.to_le_bytes();
        assert_eq!(bytes.len(), 68);
        assert_eq!(QuantBlockV4::from_le_bytes(&bytes), block);
    }

    #[test]
    fn v4_tensor_encode_matches_block_encode() {
        // encode_tensor_v4 runs in parallel over rayon; a block-boundary bug
        // there would not show up in any single-block test.
        let n = BLOCK_SIZE_V4 * 5;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 0.3).collect();
        let blocks = encode_tensor_v4(&data);
        assert_eq!(blocks.len(), 5);
        for (bi, b) in blocks.iter().enumerate() {
            let mut chunk = [0.0f32; BLOCK_SIZE_V4];
            chunk.copy_from_slice(&data[bi * BLOCK_SIZE_V4..(bi + 1) * BLOCK_SIZE_V4]);
            assert_eq!(*b, affine_quant_block_v4(&chunk), "block {bi} differs");
        }
    }
}
