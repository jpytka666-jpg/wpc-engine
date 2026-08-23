//! WPC v3 (affine 6-bit quantization, bit-packed) weight backend.
//!
//! A faithful twin of `wpc_weights_v2`, differing only in the block type and
//! the file names it opens. The quantization arithmetic is identical -- v3
//! blocks are produced by re-laying-out v2 blocks -- so any accuracy question
//! about v3 is the same question about v2.
//!
//! What differs is the byte count: 100 bytes per 128 weights instead of 132.
//! Generation here is memory-bandwidth bound (one full pass over the weights
//! per token, cores idling on memory), so that 24% is 24% off the read, and
//! most of it off the clock.

use crate::weights::{EmbeddingTable, Linear};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use wpc_format::{QuantBlockV3, BLOCK_SIZE_V3};

#[derive(Debug, Deserialize)]
struct LayerMetaV3 {
    name: String,
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct ModelMetaV3 {
    layers: Vec<LayerMetaV3>,
    block_size: usize,
}

/// Shared, load-once WPC v3 model data: mmap'd concatenated packed blocks for
/// every tensor. No dictionaries.
pub struct WpcModelDataV3 {
    mmap: Mmap, // model_v3.wpc: all tensors' QuantBlockV3 runs, concatenated
    offsets: HashMap<String, (usize, usize)>, // name -> (offset_bytes, size_bytes)
    has_avx2_fma: bool,
}

impl WpcModelDataV3 {
    pub fn open(wpc_dir: &Path) -> anyhow::Result<Arc<WpcModelDataV3>> {
        let meta_path = wpc_dir.join("model_v3.meta");
        let meta_text = std::fs::read_to_string(&meta_path)?;
        let meta: ModelMetaV3 = serde_json::from_str(&meta_text)?;

        anyhow::ensure!(
            meta.block_size == BLOCK_SIZE_V3,
            "model_v3.meta block_size {} doesn't match BLOCK_SIZE_V3 {}",
            meta.block_size,
            BLOCK_SIZE_V3
        );

        let wpc_path = wpc_dir.join("model_v3.wpc");
        let file = File::open(&wpc_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offsets = HashMap::with_capacity(meta.layers.len());
        for l in &meta.layers {
            offsets.insert(l.name.clone(), (l.offset_bytes, l.size_bytes));
        }

        let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if !has_avx2_fma {
            eprintln!(
                "[wpc_weights_v3] CPU lacks AVX2+FMA; falling back to scalar v3 decode (slower)."
            );
        }

        Ok(Arc::new(WpcModelDataV3 {
            mmap,
            offsets,
            has_avx2_fma,
        }))
    }

    /// Byte range within `model_v3.wpc` for tensor `name`.
    fn tensor_range(&self, name: &str) -> (usize, usize) {
        self.offsets
            .get(name)
            .map(|(offset, size)| (*offset, *size))
            .unwrap_or_else(|| panic!("tensor {name} not found in model_v3.meta"))
    }

    /// Reinterpret the tensor's byte range as a `&[QuantBlockV3]`.
    ///
    /// Sound because `QuantBlockV3` is `repr(C, packed)` of `f16, f16, [u8;96]`
    /// -- alignment 1, no padding, no invalid bit patterns -- so any 100-byte
    /// run is a valid instance.
    fn blocks_for(&self, name: &str) -> &[QuantBlockV3] {
        let (offset, size) = self.tensor_range(name);
        let bytes = &self.mmap[offset..offset + size];
        assert_eq!(
            bytes.len() % QuantBlockV3::SIZE,
            0,
            "misaligned v3 block range"
        );
        let n_blocks = bytes.len() / QuantBlockV3::SIZE;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const QuantBlockV3, n_blocks) }
    }
}

/// Decode one row's blocks into `out` (plain weights, not a dot product).
/// Used by `WpcEmbeddingV3::embed` for token lookups.
fn row_decode_into_v3(blocks: &[QuantBlockV3], out: &mut [f32]) {
    let mut out_idx = 0;
    for b in blocks {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        let codes = QuantBlockV3::unpack_codes(&b.packed);
        for &code in codes.iter() {
            out[out_idx] = zp + code as f32 * scale;
            out_idx += 1;
        }
    }
}

/// A WPC v3-compressed `Linear` layer.
pub struct WpcLinearV3 {
    data: Arc<WpcModelDataV3>,
    tensor_name: String,
    out_features: usize,
    in_features: usize,
    bias: Option<Vec<f32>>,
}

impl WpcLinearV3 {
    pub fn new(
        data: Arc<WpcModelDataV3>,
        tensor_name: &str,
        out_features: usize,
        in_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(
            in_features % BLOCK_SIZE_V3,
            0,
            "in_features must be a multiple of {BLOCK_SIZE_V3} for WPC v3 blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = out_features * (in_features / BLOCK_SIZE_V3);
        assert_eq!(
            size / QuantBlockV3::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v3 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV3::SIZE,
            expected_blocks
        );
        WpcLinearV3 {
            data,
            tensor_name: tensor_name.to_string(),
            out_features,
            in_features,
            bias,
        }
    }
}

impl Linear for WpcLinearV3 {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        let blocks_per_row = self.in_features / BLOCK_SIZE_V3;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        y.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v3_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v3_scalar(row_blocks, x);
                }
            });
        if let Some(b) = &self.bias {
            for (o, bi) in y.iter_mut().zip(b.iter()) {
                *o += *bi;
            }
        }
    }
}

/// WPC v3-compressed embedding / tied lm_head table: `[vocab_size, hidden_size]`.
pub struct WpcEmbeddingV3 {
    data: Arc<WpcModelDataV3>,
    tensor_name: String,
    vocab_size: usize,
    hidden_size: usize,
}

impl WpcEmbeddingV3 {
    pub fn new(
        data: Arc<WpcModelDataV3>,
        tensor_name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        assert_eq!(
            hidden_size % BLOCK_SIZE_V3,
            0,
            "hidden_size must be a multiple of {BLOCK_SIZE_V3} for WPC v3 blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = vocab_size * (hidden_size / BLOCK_SIZE_V3);
        assert_eq!(
            size / QuantBlockV3::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v3 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV3::SIZE,
            expected_blocks
        );
        WpcEmbeddingV3 {
            data,
            tensor_name: tensor_name.to_string(),
            vocab_size,
            hidden_size,
        }
    }
}

impl EmbeddingTable for WpcEmbeddingV3 {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn embed(&self, token_id: u32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.hidden_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V3;
        let all_blocks = self.data.blocks_for(&self.tensor_name);

        let row_start = token_id as usize * blocks_per_row;
        let row_blocks = &all_blocks[row_start..row_start + blocks_per_row];
        row_decode_into_v3(row_blocks, out);
    }
    fn logits(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.hidden_size);
        debug_assert_eq!(out.len(), self.vocab_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V3;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v3_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v3_scalar(row_blocks, x);
                }
            });
    }
}
