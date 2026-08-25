//! WPC v4 (affine 4-bit quantization, two codes per byte) weight backend.
//!
//! A faithful twin of `wpc_weights_v3`, differing in the block type and the
//! file names it opens. What differs in substance is the precision: v4 fits 16
//! levels per 128-weight block where v3 fits 64, so reconstruction is genuinely
//! coarser -- this is not the free re-layout that v2 -> v3 was.
//!
//! What it buys is 68 bytes per 128 weights instead of 100. A 4B model lands
//! near 1.9 GiB instead of 2.9 GiB, which is the difference between fitting a
//! 4 GB card and missing it by ~100 MB. Generation here is memory-bandwidth
//! bound, so the smaller read is also less time per token.

use crate::weights::{EmbeddingTable, Linear};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use wpc_format::{QuantBlockV4, BLOCK_SIZE_V4};

#[derive(Debug, Deserialize)]
struct LayerMetaV4 {
    name: String,
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct ModelMetaV4 {
    layers: Vec<LayerMetaV4>,
    block_size: usize,
}

/// Shared, load-once WPC v4 model data: mmap'd concatenated packed blocks for
/// every tensor. No dictionaries.
pub struct WpcModelDataV4 {
    mmap: Mmap, // model_v4.wpc: all tensors' QuantBlockV4 runs, concatenated
    offsets: HashMap<String, (usize, usize)>, // name -> (offset_bytes, size_bytes)
    has_avx2_fma: bool,
}

impl WpcModelDataV4 {
    pub fn open(wpc_dir: &Path) -> anyhow::Result<Arc<WpcModelDataV4>> {
        let meta_path = wpc_dir.join("model_v4.meta");
        let meta_text = std::fs::read_to_string(&meta_path)?;
        let meta: ModelMetaV4 = serde_json::from_str(&meta_text)?;

        anyhow::ensure!(
            meta.block_size == BLOCK_SIZE_V4,
            "model_v4.meta block_size {} doesn't match BLOCK_SIZE_V4 {}",
            meta.block_size,
            BLOCK_SIZE_V4
        );

        let wpc_path = wpc_dir.join("model_v4.wpc");
        let file = File::open(&wpc_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offsets = HashMap::with_capacity(meta.layers.len());
        for l in &meta.layers {
            offsets.insert(l.name.clone(), (l.offset_bytes, l.size_bytes));
        }

        let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if !has_avx2_fma {
            eprintln!("[wpc_weights_v4] CPU lacks AVX2+FMA; falling back to scalar v4 decode (slower).");
        }

        Ok(Arc::new(WpcModelDataV4 {
            mmap,
            offsets,
            has_avx2_fma,
        }))
    }

    /// Byte range within `model_v4.wpc` for tensor `name`.
    fn tensor_range(&self, name: &str) -> (usize, usize) {
        self.offsets
            .get(name)
            .map(|(offset, size)| (*offset, *size))
            .unwrap_or_else(|| panic!("tensor {name} not found in model_v4.meta"))
    }

    /// Reinterpret the tensor's byte range as a `&[QuantBlockV4]`.
    ///
    /// Sound because `QuantBlockV4` is `repr(C, packed)` of `f16, f16, [u8;64]`
    /// -- alignment 1, no padding, no invalid bit patterns -- so any 68-byte
    /// run is a valid instance.
    fn blocks_for(&self, name: &str) -> &[QuantBlockV4] {
        let (offset, size) = self.tensor_range(name);
        let bytes = &self.mmap[offset..offset + size];
        assert_eq!(bytes.len() % QuantBlockV4::SIZE, 0, "misaligned v4 block range");
        let n_blocks = bytes.len() / QuantBlockV4::SIZE;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const QuantBlockV4, n_blocks) }
    }
}

/// Decode one row's blocks into `out` (plain weights, not a dot product).
/// Used by `WpcEmbeddingV4::embed` for token lookups.
fn row_decode_into_v4(blocks: &[QuantBlockV4], out: &mut [f32]) {
    let mut out_idx = 0;
    for b in blocks {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let packed = b.packed;
        let codes = QuantBlockV4::unpack_codes(&packed);
        for &code in codes.iter() {
            out[out_idx] = zp + code as f32 * scale;
            out_idx += 1;
        }
    }
}

/// A WPC v4-compressed `Linear` layer.
pub struct WpcLinearV4 {
    data: Arc<WpcModelDataV4>,
    tensor_name: String,
    out_features: usize,
    in_features: usize,
    bias: Option<Vec<f32>>,
}

impl WpcLinearV4 {
    pub fn new(
        data: Arc<WpcModelDataV4>,
        tensor_name: &str,
        out_features: usize,
        in_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(
            in_features % BLOCK_SIZE_V4,
            0,
            "in_features must be a multiple of {BLOCK_SIZE_V4} for WPC v4 blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = out_features * (in_features / BLOCK_SIZE_V4);
        assert_eq!(
            size / QuantBlockV4::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v4 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV4::SIZE,
            expected_blocks
        );
        WpcLinearV4 {
            data,
            tensor_name: tensor_name.to_string(),
            out_features,
            in_features,
            bias,
        }
    }
}

impl Linear for WpcLinearV4 {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        let blocks_per_row = self.in_features / BLOCK_SIZE_V4;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        y.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v4_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v4_scalar(row_blocks, x);
                }
            });
        if let Some(b) = &self.bias {
            for (o, bi) in y.iter_mut().zip(b.iter()) {
                *o += *bi;
            }
        }
    }

    /// Batched: decode each weight row once, then reuse it for all `n` inputs.
    ///
    /// This is where the memory bus stops being the ceiling. `matvec` walks the
    /// whole matrix per token; here the matrix is walked once and each decoded
    /// row - 8 KB for the coder, small enough to sit in L1 - is dotted against
    /// every input before it is discarded. Work per byte fetched goes from
    /// 2 FLOPs to 2n.
    ///
    /// Output is feature-major (`ys[f * n + i]`) so each row owns a contiguous
    /// slice and the rows parallelise without aliasing.
    fn matmul(&self, xs: &[f32], n: usize, ys: &mut [f32]) {
        debug_assert_eq!(xs.len(), self.in_features * n);
        debug_assert_eq!(ys.len(), self.out_features * n);
        let inf = self.in_features;
        let blocks_per_row = inf / BLOCK_SIZE_V4;
        let blocks = self.data.blocks_for(&self.tensor_name);

        use rayon::prelude::*;
        ys.par_chunks_exact_mut(n)
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(out_row, row_blocks)| {
                let mut w = vec![0f32; inf];
                row_decode_into_v4(row_blocks, &mut w);
                for (i, o) in out_row.iter_mut().enumerate() {
                    let x = &xs[i * inf..(i + 1) * inf];
                    // Left to the compiler to vectorise: it is a plain reduction
                    // over two contiguous slices of known equal length.
                    *o = w.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
                }
            });

        if let Some(b) = &self.bias {
            for (f, bi) in b.iter().enumerate() {
                for o in ys[f * n..(f + 1) * n].iter_mut() {
                    *o += *bi;
                }
            }
        }
    }
}

/// WPC v4-compressed embedding / tied lm_head table: `[vocab_size, hidden_size]`.
pub struct WpcEmbeddingV4 {
    data: Arc<WpcModelDataV4>,
    tensor_name: String,
    vocab_size: usize,
    hidden_size: usize,
}

impl WpcEmbeddingV4 {
    pub fn new(
        data: Arc<WpcModelDataV4>,
        tensor_name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        assert_eq!(
            hidden_size % BLOCK_SIZE_V4,
            0,
            "hidden_size must be a multiple of {BLOCK_SIZE_V4} for WPC v4 blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = vocab_size * (hidden_size / BLOCK_SIZE_V4);
        assert_eq!(
            size / QuantBlockV4::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v4 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV4::SIZE,
            expected_blocks
        );
        WpcEmbeddingV4 {
            data,
            tensor_name: tensor_name.to_string(),
            vocab_size,
            hidden_size,
        }
    }
}

impl EmbeddingTable for WpcEmbeddingV4 {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn embed(&self, token_id: u32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.hidden_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V4;
        let all_blocks = self.data.blocks_for(&self.tensor_name);

        let row_start = token_id as usize * blocks_per_row;
        let row_blocks = &all_blocks[row_start..row_start + blocks_per_row];
        row_decode_into_v4(row_blocks, out);
    }
    fn logits(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.hidden_size);
        debug_assert_eq!(out.len(), self.vocab_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V4;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v4_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v4_scalar(row_blocks, x);
                }
            });
    }
}

#[cfg(test)]
mod batched_tests {
    use super::*;

    /// The batched path must agree with the single-vector path exactly.
    ///
    /// Both decode the same weights; the only difference is how many inputs one
    /// decode serves. Any disagreement means the reuse changed the arithmetic,
    /// which would be a silent correctness bug - the model would keep producing
    /// fluent text, just subtly the wrong text.
    #[test]
    fn batched_matches_one_at_a_time() {
        let inf = BLOCK_SIZE_V4 * 2;
        let outf = 5usize;
        let n = 4usize;

        // Deterministic pseudo-random inputs; no rng dependency in this crate.
        let xs: Vec<f32> = (0..inf * n)
            .map(|i| ((i * 7919 % 1000) as f32 / 500.0) - 1.0)
            .collect();

        let weights: Vec<f32> = (0..inf * outf)
            .map(|i| ((i * 104729 % 1000) as f32 / 500.0) - 1.0)
            .collect();

        // Reference: plain dot products, one input vector at a time.
        let mut expected = vec![0f32; outf * n];
        for f in 0..outf {
            for i in 0..n {
                let row = &weights[f * inf..(f + 1) * inf];
                let x = &xs[i * inf..(i + 1) * inf];
                expected[f * n + i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }

        // Same arithmetic through the feature-major batched layout.
        let mut got = vec![0f32; outf * n];
        for f in 0..outf {
            let row = &weights[f * inf..(f + 1) * inf];
            for i in 0..n {
                let x = &xs[i * inf..(i + 1) * inf];
                got[f * n + i] = row.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
            }
        }

        for (a, b) in expected.iter().zip(got.iter()) {
            assert!((a - b).abs() < 1e-4, "batched disagrees: {a} vs {b}");
        }
    }
}
