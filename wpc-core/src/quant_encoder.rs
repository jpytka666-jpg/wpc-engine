use wpc_format::{QuantBlockV2, BLOCK_SIZE_V2};
use rayon::prelude::*;

/// Quantize a block of 128 f32 values to v2 format (6-bit affine).
pub fn affine_quant_block(weights: &[f32; BLOCK_SIZE_V2]) -> QuantBlockV2 {
    let mut min_v = f32::MAX;
    let mut max_v = f32::MIN;
    for &w in weights {
        if w < min_v { min_v = w; }
        if w > max_v { max_v = w; }
    }
    let levels = 63.0_f32; // 2^6 - 1
    let scale = if max_v > min_v { (max_v - min_v) / levels } else { 1.0 };
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
    assert_eq!(data.len() % BLOCK_SIZE_V2, 0, "tensor length must be a multiple of {BLOCK_SIZE_V2}");
    let n_blocks = data.len() / BLOCK_SIZE_V2;
    (0..n_blocks).into_par_iter().map(|i| {
        let mut block = [0.0f32; BLOCK_SIZE_V2];
        block.copy_from_slice(&data[i*BLOCK_SIZE_V2..(i+1)*BLOCK_SIZE_V2]);
        affine_quant_block(&block)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(rel_rmse < 0.05, "relative RMSE {rel_rmse} too high (should be <5% for this block)");
    }
}
