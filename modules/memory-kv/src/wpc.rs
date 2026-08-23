use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use wpc_core::codebook::{dummy_residual_dict, PatternDict, ResidualDict, BLOCK_DIM};
use wpc_core::encoder::{encode_block, normalize_block};
use wpc_format::CompressedBlock;

#[derive(Debug, Clone)]
pub struct WpcKvInput {
    pub session_id: String,
    pub keys: Vec<f32>,
    pub values: Vec<f32>,
    pub vector_width: usize,
    pub pattern_count: usize,
    pub residual_count: usize,
    pub train_iters: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WpcKvMetrics {
    pub session_id: String,
    pub original_bytes_f32: usize,
    pub original_bytes_f16: usize,
    pub compressed_bytes: usize,
    pub payload_bytes: usize,
    pub dictionary_bytes: usize,
    pub compression_ratio_vs_f16: f64,
    pub payload_compression_ratio_vs_f16: f64,
    pub key_rmse: f64,
    pub value_rmse: f64,
    pub attention_output_rmse: f64,
    pub attention_score_delta: f64,
    pub generation_critical: bool,
}

#[derive(Debug, Clone)]
struct EncodedTensor {
    blocks: Vec<CompressedBlock>,
    pattern_dict: PatternDict,
    residual_dict: ResidualDict,
}

fn train_pattern_dict(values: &[f32], pattern_count: usize) -> Result<PatternDict, &'static str> {
    if values.is_empty() || values.len() % BLOCK_DIM != 0 {
        return Err("tensor length must be a non-zero multiple of WPC block dimension");
    }

    let normalized: Vec<[f32; BLOCK_DIM]> = values
        .chunks_exact(BLOCK_DIM)
        .map(|chunk| {
            let block: [f32; BLOCK_DIM] = chunk.try_into().expect("chunk width is fixed");
            normalize_block(&block).norm
        })
        .collect();

    Ok(PatternDict::train(&normalized, pattern_count, 8))
}

fn encode_tensor(
    values: &[f32],
    pattern_count: usize,
    residual_count: usize,
    train_iters: usize,
) -> Result<EncodedTensor, &'static str> {
    if values.is_empty() || values.len() % BLOCK_DIM != 0 {
        return Err("tensor length must be a non-zero multiple of WPC block dimension");
    }

    let pattern_dict = train_pattern_dict(values, pattern_count)?;
    let num_blocks = values.len() / BLOCK_DIM;
    let dummy = dummy_residual_dict();

    let residuals: Vec<[f32; BLOCK_DIM]> = (0..num_blocks)
        .into_par_iter()
        .map(|index| {
            let mut block = [0.0; BLOCK_DIM];
            block.copy_from_slice(&values[index * BLOCK_DIM..(index + 1) * BLOCK_DIM]);
            let (_, residual) = encode_block(&block, &pattern_dict, &dummy);
            residual
        })
        .collect();

    let residual_k = residual_count.min(num_blocks).max(1);
    let residual_dict = ResidualDict::train(&residuals, residual_k, train_iters);

    let blocks = (0..num_blocks)
        .into_par_iter()
        .map(|index| {
            let mut block = [0.0; BLOCK_DIM];
            block.copy_from_slice(&values[index * BLOCK_DIM..(index + 1) * BLOCK_DIM]);
            encode_block(&block, &pattern_dict, &residual_dict).0
        })
        .collect();

    Ok(EncodedTensor {
        blocks,
        pattern_dict,
        residual_dict,
    })
}

fn decode_tensor(encoded: &EncodedTensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(encoded.blocks.len() * BLOCK_DIM);
    for block in &encoded.blocks {
        let pattern = encoded.pattern_dict.centroids[block.pattern_id as usize];
        let residual = encoded.residual_dict.centroids_f16[block.residual_id as usize];
        let base = block.base_value.to_f32();
        let scale = block.scale as f32 / 127.0;
        for i in 0..BLOCK_DIM {
            out.push(pattern[i] * scale + base + residual[i].to_f32() / 127.0);
        }
    }
    out
}

fn mse_rmse(original: &[f32], reconstructed: &[f32]) -> f64 {
    let sum = original
        .iter()
        .zip(reconstructed)
        .map(|(a, b)| {
            let d = (*a as f64) - (*b as f64);
            d * d
        })
        .sum::<f64>();
    (sum / original.len() as f64).sqrt()
}

fn softmax(scores: &[f64]) -> Vec<f64> {
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum = exps.iter().sum::<f64>();
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

fn attention_output(query: &[f32], keys: &[f32], values: &[f32], vector_width: usize) -> Vec<f32> {
    let token_count = keys.len() / vector_width;
    let scale = (vector_width as f64).sqrt();
    let mut scores = Vec::with_capacity(token_count);
    for token in 0..token_count {
        let base = token * vector_width;
        let dot = (0..vector_width)
            .map(|i| query[i] as f64 * keys[base + i] as f64)
            .sum::<f64>();
        scores.push(dot / scale);
    }
    let weights = softmax(&scores);
    let mut output = vec![0.0f32; vector_width];
    for token in 0..token_count {
        let base = token * vector_width;
        for i in 0..vector_width {
            output[i] += weights[token] as f32 * values[base + i];
        }
    }
    output
}

pub fn run_wpc_kv(input: WpcKvInput) -> Result<WpcKvMetrics, &'static str> {
    if input.session_id.is_empty() {
        return Err("session id must not be empty");
    }
    if input.vector_width == 0 || input.vector_width % BLOCK_DIM != 0 {
        return Err("vector width must be a positive multiple of WPC block dimension");
    }
    if input.keys.len() != input.values.len() || input.keys.len() % input.vector_width != 0 {
        return Err("K and V must have equal lengths aligned to vector width");
    }
    if input.pattern_count == 0 || input.residual_count == 0 || input.train_iters == 0 {
        return Err("WPC training parameters must be non-zero");
    }

    let keys = encode_tensor(
        &input.keys,
        input.pattern_count,
        input.residual_count,
        input.train_iters,
    )?;
    let values = encode_tensor(
        &input.values,
        input.pattern_count,
        input.residual_count,
        input.train_iters,
    )?;

    let decoded_keys = decode_tensor(&keys);
    let decoded_values = decode_tensor(&values);

    let query = &input.keys[..input.vector_width];
    let reference_attention = attention_output(query, &input.keys, &input.values, input.vector_width);
    let compressed_attention = attention_output(
        query,
        &decoded_keys,
        &decoded_values,
        input.vector_width,
    );

    let attention_output_rmse = mse_rmse(&reference_attention, &compressed_attention);
    let reference_score = query
        .iter()
        .zip(input.keys.iter())
        .map(|(q, k)| *q as f64 * *k as f64)
        .sum::<f64>();
    let compressed_score = query
        .iter()
        .zip(decoded_keys.iter())
        .map(|(q, k)| *q as f64 * *k as f64)
        .sum::<f64>();

    let block_count = keys.blocks.len() + values.blocks.len();
    let dictionary_bytes = (keys.pattern_dict.centroids.len() + values.pattern_dict.centroids.len())
        * BLOCK_DIM
        * std::mem::size_of::<f32>()
        + (keys.residual_dict.centroids_f16.len() + values.residual_dict.centroids_f16.len())
            * BLOCK_DIM
            * std::mem::size_of::<u16>();
    let payload_bytes = block_count * CompressedBlock::SIZE;
    let compressed_bytes = dictionary_bytes + payload_bytes;

    let original_bytes_f32 = input.keys.len() * std::mem::size_of::<f32>() * 2;
    let original_bytes_f16 = input.keys.len() * std::mem::size_of::<u16>() * 2;
    let payload_reference_bytes = block_count * BLOCK_DIM * std::mem::size_of::<u16>();

    Ok(WpcKvMetrics {
        session_id: input.session_id,
        original_bytes_f32,
        original_bytes_f16,
        compressed_bytes,
        payload_bytes,
        dictionary_bytes,
        compression_ratio_vs_f16: original_bytes_f16 as f64 / compressed_bytes as f64,
        payload_compression_ratio_vs_f16: payload_reference_bytes as f64 / payload_bytes as f64,
        key_rmse: mse_rmse(&input.keys, &decoded_keys),
        value_rmse: mse_rmse(&input.values, &decoded_values),
        attention_output_rmse,
        attention_score_delta: (reference_score - compressed_score).abs(),
        generation_critical: false,
    })
}

#[cfg(test)]
mod tests {
    use super::{run_wpc_kv, WpcKvInput};

    fn synthetic_kv(tokens: usize, width: usize) -> (Vec<f32>, Vec<f32>) {
        let mut keys = Vec::with_capacity(tokens * width);
        let mut values = Vec::with_capacity(tokens * width);
        for token in 0..tokens {
            for dim in 0..width {
                let x = (token as f32 * 0.13 + dim as f32 * 0.07).sin();
                keys.push(x);
                values.push((x * 0.83 + 0.11).cos());
            }
        }
        (keys, values)
    }

    #[test]
    fn wpc_kv_round_trip_preserves_attention_reasonably() {
        let (keys, values) = synthetic_kv(128, 64);
        let metrics = run_wpc_kv(WpcKvInput {
            session_id: "test-session".into(),
            keys,
            values,
            vector_width: 64,
            pattern_count: 16,
            residual_count: 256,
            train_iters: 5,
        })
        .expect("WPC KV experiment");

        assert!(!metrics.generation_critical);
        assert!(metrics.key_rmse < 0.25, "key RMSE too high: {}", metrics.key_rmse);
        assert!(metrics.value_rmse < 0.25, "value RMSE too high: {}", metrics.value_rmse);
        assert!(metrics.attention_output_rmse < 0.25);
        assert!(metrics.payload_compression_ratio_vs_f16 > 1.0);
        assert!(metrics.compressed_bytes > 0);
    }

    #[test]
    fn rejects_misaligned_vector_width() {
        let err = run_wpc_kv(WpcKvInput {
            session_id: "test".into(),
            keys: vec![0.0; 32],
            values: vec![0.0; 32],
            vector_width: 20,
            pattern_count: 16,
            residual_count: 256,
            train_iters: 5,
        })
        .expect_err("20 is not aligned to WPC block dimension");
        assert_eq!(err, "vector width must be a positive multiple of WPC block dimension");
    }
}
