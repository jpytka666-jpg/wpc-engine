//! WPC v2 (affine 6-bit quantization) weight backend.
//! Simpler than v1: no per-class dictionaries, just per-block min/max quantization.

use crate::weights::{EmbeddingTable, Linear};
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use wpc_format::{QuantBlockV2, BLOCK_SIZE_V2};

#[derive(Debug, Deserialize)]
struct LayerMetaV2 {
    name: String,
    #[allow(dead_code)]
    #[allow(dead_code)]]
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct ModelMetaV2 {
    layers: Vec<LayerMetaV2>,
    block_size: usize,
}

/// Shared, load-once WPC v2 model data: mmap'd concatenated compressed blocks
/// for every tensor. No dictionaries needed.
pub struct WpcModelDataV2 {
    mmap: Mmap, // model_v2.wpc: all tensors' QuantBlockV2 runs, concatenated
    offsets: HashMap<String, (usize, usize)>, // name -> (offset_bytes, size_bytes)
    has_avx2_fma: bool,
}

impl WpcModelDataV2 {
    pub fn open(wpc_dir: &Path) -> anyhow::Result<Arc<WpcModelDataV2>> {
        let meta_path = wpc_dir.join("model_v2.meta");
        let meta_text = std::fs::read_to_string(&meta_path)?;
        let meta: ModelMetaV2 = serde_json::from_str(&meta_text)?;

        anyhow::ensure!(
            meta.block_size == BLOCK_SIZE_V2,
            "model_v2.meta block_size {} doesn't match BLOCK_SIZE_V2 {}",
            meta.block_size,
            BLOCK_SIZE_V2
        );

        let wpc_path = wpc_dir.join("model_v2.wpc");
        let file = File::open(&wpc_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offsets = HashMap::with_capacity(meta.layers.len());
        for l in &meta.layers {
            offsets.insert(l.name.clone(), (l.offset_bytes, l.size_bytes));
        }

        let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
        if !has_avx2_fma {
            eprintln!(
                "[wpc_weights_v2] CPU lacks AVX2+FMA; falling back to scalar v2 decode (slower)."
            );
        }

        Ok(Arc::new(WpcModelDataV2 {
            mmap,
            offsets,
            has_avx2_fma,
        }))
    }

    /// Byte range within `model_v2.wpc` for tensor `name`.
    fn tensor_range(&self, name: &str) -> (usize, usize) {
        self.offsets
            .get(name)
            .map(|(offset, size)| (*offset, *size))
            .unwrap_or_else(|| panic!("tensor {name} not found in model_v2.meta"))
    }

    /// Reinterpret the tensor's byte range as a `&[QuantBlockV2]`.
    fn blocks_for(&self, name: &str) -> &[QuantBlockV2] {
        let (offset, size) = self.tensor_range(name);
        let bytes = &self.mmap[offset..offset + size];
        assert_eq!(
            bytes.len() % QuantBlockV2::SIZE,
            0,
            "misaligned v2 block range"
        );
        let n_blocks = bytes.len() / QuantBlockV2::SIZE;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const QuantBlockV2, n_blocks) }
    }
}

/// Decode one row's blocks into `out` (plain weights, not a dot product).
/// Used by `WpcEmbeddingV2::embed` for token lookups.
fn row_decode_into_v2(blocks: &[QuantBlockV2], out: &mut [f32]) {
    let mut out_idx = 0;
    for b in blocks {
        let zp = b.zero_point.to_f32();
        let scale = b.scale.to_f32();
        for code in &b.codes {
            out[out_idx] = zp + *code as f32 * scale;
            out_idx += 1;
        }
    }
}

/// A WPC v2-compressed `Linear` layer.
pub struct WpcLinearV2 {
    data: Arc<WpcModelDataV2>,
    tensor_name: String,
    out_features: usize,
    in_features: usize,
    bias: Option<Vec<f32>>,
}

impl WpcLinearV2 {
    pub fn new(
        data: Arc<WpcModelDataV2>,
        tensor_name: &str,
        out_features: usize,
        in_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(
            in_features % BLOCK_SIZE_V2,
            0,
            "in_features must be a multiple of {BLOCK_SIZE_V2} for WPC v2 blocks"
        );
        // Validate the tensor exists and has the expected number of blocks.
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = out_features * (in_features / BLOCK_SIZE_V2);
        assert_eq!(
            size / QuantBlockV2::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v2 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV2::SIZE,
            expected_blocks
        );
        WpcLinearV2 {
            data,
            tensor_name: tensor_name.to_string(),
            out_features,
            in_features,
            bias,
        }
    }
}

impl Linear for WpcLinearV2 {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        let blocks_per_row = self.in_features / BLOCK_SIZE_V2;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        y.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v2_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v2_scalar(row_blocks, x);
                }
            });
        if let Some(b) = &self.bias {
            for (o, bi) in y.iter_mut().zip(b.iter()) {
                *o += *bi;
            }
        }
    }
}

/// WPC v2-compressed embedding / tied lm_head table: `[vocab_size, hidden_size]`.
pub struct WpcEmbeddingV2 {
    data: Arc<WpcModelDataV2>,
    tensor_name: String,
    vocab_size: usize,
    hidden_size: usize,
}

impl WpcEmbeddingV2 {
    pub fn new(
        data: Arc<WpcModelDataV2>,
        tensor_name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        assert_eq!(
            hidden_size % BLOCK_SIZE_V2,
            0,
            "hidden_size must be a multiple of {BLOCK_SIZE_V2} for WPC v2 blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = vocab_size * (hidden_size / BLOCK_SIZE_V2);
        assert_eq!(
            size / QuantBlockV2::SIZE,
            expected_blocks,
            "tensor {tensor_name}: v2 block count mismatch (meta says {}, shape implies {})",
            size / QuantBlockV2::SIZE,
            expected_blocks
        );
        WpcEmbeddingV2 {
            data,
            tensor_name: tensor_name.to_string(),
            vocab_size,
            hidden_size,
        }
    }
}

impl EmbeddingTable for WpcEmbeddingV2 {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn embed(&self, token_id: u32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.hidden_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V2;
        let all_blocks = self.data.blocks_for(&self.tensor_name);

        let row_start = token_id as usize * blocks_per_row;
        let row_blocks = &all_blocks[row_start..row_start + blocks_per_row];
        row_decode_into_v2(row_blocks, out);
    }
    fn logits(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.hidden_size);
        debug_assert_eq!(out.len(), self.vocab_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE_V2;
        let blocks = self.data.blocks_for(&self.tensor_name);
        let has_avx2 = self.data.has_avx2_fma;

        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                if has_avx2 {
                    *o = unsafe {
                        wpc_eval::fused_kernel::matvec_v2_fused(
                            row_blocks,
                            x.as_ptr(),
                            row_blocks.len(),
                        )
                    }
                } else {
                    *o = wpc_eval::fused_kernel::matvec_v2_scalar(row_blocks, x);
                }
            });
    }
}
