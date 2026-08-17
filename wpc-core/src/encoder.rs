use wpc_format::CompressedBlock;
use half::f16;
use crate::codebook::{PatternDict, ResidualDict, BLOCK_DIM};

const INPUT_SCALE: f32 = 127.0;

pub fn encode_block(
    weights: &[f32; BLOCK_DIM],
    pattern_dict: &PatternDict,
    residual_dict: &ResidualDict,
) -> (CompressedBlock, [f32; BLOCK_DIM]) {
    // 1. Calculate base
    let mut sum = 0.0;
    let mut min_w = f32::MAX;
    let mut max_w = f32::MIN;
    for &w in weights {
        sum += w;
        if w < min_w { min_w = w; }
        if w > max_w { max_w = w; }
    }
    let base = sum / BLOCK_DIM as f32;

    // 2. Calculate scale and find pattern
    let mut centered = [0.0; BLOCK_DIM];
    let mut max_abs_centered = 0.0_f32;
    for i in 0..BLOCK_DIM {
        centered[i] = weights[i] - base;
        let abs_c = centered[i].abs();
        if abs_c > max_abs_centered { max_abs_centered = abs_c; }
    }

    let scale_f32 = (max_abs_centered * INPUT_SCALE).round();
    let scale_i8 = scale_f32.clamp(-127.0, 127.0) as i8;
    
    // Normalize for pattern matching
    let mut norm = [0.0; BLOCK_DIM];
    if scale_i8 != 0 {
        let inv_scale = INPUT_SCALE / scale_i8 as f32;
        for i in 0..BLOCK_DIM {
            norm[i] = centered[i] * inv_scale;
        }
    }
    
    let (pattern_id, _p_vec) = pattern_dict.nearest(&norm);
    let p_vec = pattern_dict.centroids[pattern_id as usize];

    // 3. Reconstruct without residual
    let mut approx = [0.0; BLOCK_DIM];
    let s_decode = scale_i8 as f32 / INPUT_SCALE;
    let mut block_residual = [0.0; BLOCK_DIM];
    
    for i in 0..BLOCK_DIM {
        approx[i] = p_vec[i] * s_decode + base;
        // The residual is expected to be scaled by INPUT_SCALE before dictionary lookup
        // since in the decoder we divide by INPUT_SCALE
        block_residual[i] = (weights[i] - approx[i]) * INPUT_SCALE;
    }

    // 4. Find residual
    let (residual_id, _r_vec) = residual_dict.nearest(&block_residual);

    let final_block = CompressedBlock {
        pattern_id,
        scale: scale_i8,
        residual_id: residual_id.to_le(),
        base_value: f16::from_f32(base),
    };

    // Calculate actual residual for return
    let r_vec = residual_dict.centroids_f16[residual_id as usize];
    for i in 0..BLOCK_DIM {
        block_residual[i] = r_vec[i].to_f32() / INPUT_SCALE;
    }

    (final_block, block_residual)
}

pub fn two_pass_encode(
    data: &[f32],
    pattern_dict: &PatternDict,
    train_iters: usize,
) -> (Vec<CompressedBlock>, ResidualDict) {
    let num_blocks = data.len() / BLOCK_DIM;
    let mut residuals_f32 = Vec::with_capacity(num_blocks);
    
    let dummy_rd = crate::codebook::dummy_residual_dict();
    
    // Pass 1: harvest residuals
    for i in 0..num_blocks {
        let mut block = [0.0; BLOCK_DIM];
        block.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
        let (_, r) = encode_block(&block, pattern_dict, &dummy_rd);
        residuals_f32.push(r);
    }
    
    // Train residual dict
    let k_residuals = 65536.min(num_blocks * 2); // Prevent over-allocating if tiny array
    let residual_dict = ResidualDict::train(&residuals_f32, k_residuals, train_iters);
    
    // Pass 2: encode final
    let mut blocks = Vec::with_capacity(num_blocks);
    for i in 0..num_blocks {
        let mut block = [0.0; BLOCK_DIM];
        block.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
        let (cb, _) = encode_block(&block, pattern_dict, &residual_dict);
        blocks.push(cb);
    }
    
    (blocks, residual_dict)
}
