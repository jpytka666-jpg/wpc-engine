use crate::codebook::{PatternDict, ResidualDict, BLOCK_DIM};
use half::f16;
use rayon::prelude::*;
use wpc_format::{CompressedBlock, RESIDUAL_COUNT};

pub const INPUT_SCALE: f32 = 127.0;

/// Per-block mean/scale normalization, shared by encoding and by pattern-dict
/// training. Pattern centroids are matched against `norm`, so anything that
/// trains or queries `PatternDict` must go through this same transform --
/// training on raw (un-normalized) blocks puts the dictionary in the wrong
/// space relative to what `nearest()` is queried with at encode time.
pub struct BlockNorm {
    pub base: f32,
    pub scale_i8: i8,
    pub norm: [f32; BLOCK_DIM],
}

pub fn normalize_block(weights: &[f32; BLOCK_DIM]) -> BlockNorm {
    let mut sum = 0.0;
    for &w in weights {
        sum += w;
    }
    let base = sum / BLOCK_DIM as f32;

    let mut centered = [0.0; BLOCK_DIM];
    let mut max_abs_centered = 0.0_f32;
    for i in 0..BLOCK_DIM {
        centered[i] = weights[i] - base;
        let abs_c = centered[i].abs();
        if abs_c > max_abs_centered {
            max_abs_centered = abs_c;
        }
    }

    let scale_f32 = (max_abs_centered * INPUT_SCALE).round();
    let scale_i8 = scale_f32.clamp(-127.0, 127.0) as i8;

    let mut norm = [0.0; BLOCK_DIM];
    if scale_i8 != 0 {
        let inv_scale = INPUT_SCALE / scale_i8 as f32;
        for i in 0..BLOCK_DIM {
            norm[i] = centered[i] * inv_scale;
        }
    }

    BlockNorm {
        base,
        scale_i8,
        norm,
    }
}

pub fn encode_block(
    weights: &[f32; BLOCK_DIM],
    pattern_dict: &PatternDict,
    residual_dict: &ResidualDict,
) -> (CompressedBlock, [f32; BLOCK_DIM]) {
    // 1 & 2. Base, scale, and normalized vector for pattern matching.
    let BlockNorm {
        base,
        scale_i8,
        norm,
    } = normalize_block(weights);

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
        // Native-order value: `CompressedBlock::to_le_bytes` already does the
        // explicit LE byte extraction, so pre-swapping here would double-swap
        // the value on a big-endian host.
        residual_id,
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
    let dummy_rd = crate::codebook::dummy_residual_dict();

    // Pass 1: harvest residuals (parallel; matches wpc-compiler's rayon path)
    let residuals_f32: Vec<[f32; BLOCK_DIM]> = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let mut block = [0.0; BLOCK_DIM];
            block.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
            let (_, r) = encode_block(&block, pattern_dict, &dummy_rd);
            r
        })
        .collect();

    // Train residual dict
    let k_residuals = RESIDUAL_COUNT.min(num_blocks * 2); // Prevent over-allocating if tiny array
    let residual_dict = ResidualDict::train(&residuals_f32, k_residuals, train_iters);

    // Pass 2: encode final (parallel)
    let blocks: Vec<CompressedBlock> = (0..num_blocks)
        .into_par_iter()
        .map(|i| {
            let mut block = [0.0; BLOCK_DIM];
            block.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
            let (cb, _) = encode_block(&block, pattern_dict, &residual_dict);
            cb
        })
        .collect();

    (blocks, residual_dict)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codebook::BLOCK_DIM;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;

    /// Encode a batch of synthetic blocks through the real training path
    /// (normalize -> train PatternDict in that same space -> two_pass_encode)
    /// and check the decoded weights land close to the originals. This is a
    /// regression guard for the base/scale/pattern/residual reconstruction
    /// formula agreeing with itself end to end.
    #[test]
    fn round_trip_reconstructs_close_to_original() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let num_blocks = 512;
        let mut data = vec![0.0f32; num_blocks * BLOCK_DIM];
        for v in data.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }

        let mut blocks_raw = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            let mut b = [0.0; BLOCK_DIM];
            b.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
            blocks_raw.push(b);
        }
        let normalized: Vec<[f32; BLOCK_DIM]> =
            blocks_raw.iter().map(|b| normalize_block(b).norm).collect();
        let pattern_dict = PatternDict::train(&normalized, 16, 10);

        let (blocks, residual_dict) = two_pass_encode(&data, &pattern_dict, 5);
        assert_eq!(blocks.len(), num_blocks);

        let mut sq_err = 0.0f64;
        let n = (num_blocks * BLOCK_DIM) as f64;
        for (i, cb) in blocks.iter().enumerate() {
            let p = pattern_dict.centroids[cb.pattern_id as usize];
            let r = residual_dict.centroids_f16[cb.residual_id as usize];
            let base = cb.base_value.to_f32();
            let s_decode = cb.scale as f32 / INPUT_SCALE;
            for j in 0..BLOCK_DIM {
                let approx = p[j] * s_decode + base + r[j].to_f32() / INPUT_SCALE;
                let orig = data[i * BLOCK_DIM + j];
                sq_err += ((approx - orig) as f64).powi(2);
            }
        }
        let rmse = (sq_err / n).sqrt() as f32;
        // Inputs are Uniform(-1,1), std ~0.577. A correct decode should land
        // near that even for unstructured/uncorrelated data. The old 127x
        // scale bug inflated the pattern term by ~2 orders of magnitude, which
        // would push RMSE into the tens.
        assert!(
            rmse < 1.0,
            "RMSE {rmse} is too large for inputs with std ~0.577"
        );
    }

    #[test]
    fn on_disk_pattern_convention_matches_decode() {
        // Mirrors the exact contract in wpc-format/src/lib.rs: patterns are
        // pre-divided by INPUT_SCALE on disk, so decode is `pattern*scale+base`
        // with no further division needed for the pattern term.
        let raw_centroid = 0.5_f32; // in normalize_block()'s [-1, 1] output space
        let on_disk_pattern = raw_centroid / INPUT_SCALE;
        let scale_i8: i8 = 40;
        let base = 0.1_f32;

        let decode_via_raw = raw_centroid * (scale_i8 as f32 / INPUT_SCALE) + base;
        let decode_via_on_disk = on_disk_pattern * (scale_i8 as f32) + base;

        assert!(
            (decode_via_raw - decode_via_on_disk).abs() < 1e-6,
            "on-disk (pre-divided) pattern decode must equal raw-centroid decode: \
             {decode_via_raw} vs {decode_via_on_disk}"
        );
    }
}
