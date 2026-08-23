//! WPC-compressed weight backend: `WpcLinear` / `WpcEmbedding` implement the
//! same `Linear` / `EmbeddingTable` traits as the dense backend in
//! `weights.rs`, but decode weights on the fly from the WPC block format
//! written by wpc-compiler (`model.wpc` + `model.meta` + the two global
//! codebook files). See wpc-format/src/lib.rs for the on-disk contract this
//! module implements, and wpc-eval/src/fused_kernel.rs for the tested AVX2
//! decode+dot-product kernel this reuses.

use crate::weights::{EmbeddingTable, Linear};
use half::f16;
use memmap2::Mmap;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use wpc_core::encoder::INPUT_SCALE;
use wpc_format::{CompressedBlock, BLOCK_SIZE, PATTERN_TABLE_BYTES, RESIDUAL_TABLE_BYTES};

#[derive(Debug, Deserialize)]
struct LayerMeta {
    name: String,
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
    dict_class: String,
}

#[derive(Debug, Deserialize)]
struct DictFiles {
    patterns: String,
    residuals: String,
}

#[derive(Debug, Deserialize)]
struct ModelMeta {
    layers: Vec<LayerMeta>,
    dictionaries: HashMap<String, DictFiles>,
}

/// Shared, load-once WPC model data: per-class codebooks + the mmap'd
/// concatenated compressed blocks for every tensor. Cheap to clone (it's an
/// `Arc`); `WpcLinear`/`WpcEmbedding` each hold a reference into it.
pub struct WpcModelData {
    dicts: HashMap<String, (Vec<f32>, Vec<f16>)>, // class -> (patterns, residuals)
    mmap: Mmap, // model.wpc: all tensors' CompressedBlock runs, concatenated
    offsets: HashMap<String, (usize, usize, String)>, // name -> (offset_bytes, size_bytes, dict_class)
    has_avx2_fma_f16c: bool,
}

impl WpcModelData {
    pub fn open(wpc_dir: &Path) -> anyhow::Result<Arc<WpcModelData>> {
        let meta_path = wpc_dir.join("model.meta");
        let meta_text = std::fs::read_to_string(&meta_path)?;
        let meta: ModelMeta = serde_json::from_str(&meta_text)?;

        // Load all per-class dictionaries
        let mut dicts: HashMap<String, (Vec<f32>, Vec<f16>)> = HashMap::new();
        for (class_name, dict_files) in &meta.dictionaries {
            let patterns_path = wpc_dir.join(&dict_files.patterns);
            let patterns_bytes = std::fs::read(&patterns_path)?;
            anyhow::ensure!(
                patterns_bytes.len() == PATTERN_TABLE_BYTES,
                "{}: expected {PATTERN_TABLE_BYTES} bytes, got {}",
                dict_files.patterns,
                patterns_bytes.len()
            );
            let patterns: Vec<f32> = patterns_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();

            let residuals_path = wpc_dir.join(&dict_files.residuals);
            let residuals_bytes = std::fs::read(&residuals_path)?;
            anyhow::ensure!(
                residuals_bytes.len() == RESIDUAL_TABLE_BYTES,
                "{}: expected {RESIDUAL_TABLE_BYTES} bytes, got {}",
                dict_files.residuals,
                residuals_bytes.len()
            );
            let residuals: Vec<f16> = residuals_bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes([c[0], c[1]]))
                .collect();

            dicts.insert(class_name.clone(), (patterns, residuals));
        }

        let wpc_path = wpc_dir.join("model.wpc");
        let file = File::open(&wpc_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        let mut offsets = HashMap::with_capacity(meta.layers.len());
        for l in &meta.layers {
            offsets.insert(
                l.name.clone(),
                (l.offset_bytes, l.size_bytes, l.dict_class.clone()),
            );
        }

        let has_avx2_fma_f16c = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        if !has_avx2_fma_f16c {
            eprintln!("[wpc_weights] CPU lacks AVX2+FMA+F16C; falling back to scalar WPC decode (slower).");
        }

        Ok(Arc::new(WpcModelData {
            dicts,
            mmap,
            offsets,
            has_avx2_fma_f16c,
        }))
    }

    /// Get dictionary class for tensor `name`.
    fn class_for(&self, name: &str) -> &str {
        self.offsets
            .get(name)
            .map(|(_, _, class)| class.as_str())
            .unwrap_or_else(|| panic!("tensor {name} not found in model.meta"))
    }

    /// Get (patterns, residuals) slices for a given class.
    fn dict_for(&self, class: &str) -> (&[f32], &[f16]) {
        let (p, r) = self
            .dicts
            .get(class)
            .unwrap_or_else(|| panic!("no dictionary for class '{class}'"));
        (p.as_slice(), r.as_slice())
    }

    /// Byte range within `model.wpc` for tensor `name`.
    fn tensor_range(&self, name: &str) -> (usize, usize) {
        self.offsets
            .get(name)
            .map(|(offset, size, _)| (*offset, *size))
            .unwrap_or_else(|| panic!("tensor {name} not found in model.meta"))
    }

    /// Reinterpret the tensor's byte range as a `&[CompressedBlock]`. Safe:
    /// `CompressedBlock` is `repr(C, packed)` so its alignment is 1, and the
    /// mmap outlives every reference we hand out (borrowed from `&self`).
    fn blocks_for(&self, name: &str) -> &[CompressedBlock] {
        let (offset, size) = self.tensor_range(name);
        let bytes = &self.mmap[offset..offset + size];
        assert_eq!(
            bytes.len() % CompressedBlock::SIZE,
            0,
            "misaligned block range"
        );
        let n_blocks = bytes.len() / CompressedBlock::SIZE;
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const CompressedBlock, n_blocks) }
    }
}

/// Decode+dot-product one row's worth of blocks (`blocks.len()` == `x.len()/16`)
/// against `x`, i.e. `sum_i decode(blocks)[i] * x[i]`. Uses the tested AVX2
/// fused kernel when available, otherwise a scalar fallback with identical
/// math (see wpc-format/src/lib.rs's decode formula).
#[inline]
fn row_dot(
    has_avx2_fma_f16c: bool,
    blocks: &[CompressedBlock],
    patterns: &[f32],
    residuals: &[f16],
    x: &[f32],
) -> f32 {
    if has_avx2_fma_f16c {
        unsafe {
            wpc_eval::fused_kernel::matvec_wpc_fused(
                blocks,
                patterns.as_ptr(),
                residuals.as_ptr(),
                x.as_ptr(),
                blocks.len(),
            )
        }
    } else {
        row_dot_scalar(blocks, patterns, residuals, x)
    }
}

fn row_dot_scalar(
    blocks: &[CompressedBlock],
    patterns: &[f32],
    residuals: &[f16],
    x: &[f32],
) -> f32 {
    let mut acc = 0.0f32;
    for (bi, b) in blocks.iter().enumerate() {
        let base = b.base_value.to_f32();
        let scale = b.scale as f32;
        let p =
            &patterns[b.pattern_id as usize * BLOCK_SIZE..(b.pattern_id as usize + 1) * BLOCK_SIZE];
        let r = &residuals
            [b.residual_id as usize * BLOCK_SIZE..(b.residual_id as usize + 1) * BLOCK_SIZE];
        let xb = &x[bi * BLOCK_SIZE..(bi + 1) * BLOCK_SIZE];
        for k in 0..BLOCK_SIZE {
            let w = p[k] * scale + base + r[k].to_f32() / INPUT_SCALE;
            acc += w * xb[k];
        }
    }
    acc
}

/// Decode one row's blocks into `out` (plain weights, not a dot product).
/// Used by `WpcEmbedding::embed` for random-row-access token lookups.
fn row_decode_into(
    blocks: &[CompressedBlock],
    patterns: &[f32],
    residuals: &[f16],
    out: &mut [f32],
) {
    for (bi, b) in blocks.iter().enumerate() {
        let base = b.base_value.to_f32();
        let scale = b.scale as f32;
        let p =
            &patterns[b.pattern_id as usize * BLOCK_SIZE..(b.pattern_id as usize + 1) * BLOCK_SIZE];
        let r = &residuals
            [b.residual_id as usize * BLOCK_SIZE..(b.residual_id as usize + 1) * BLOCK_SIZE];
        let out_b = &mut out[bi * BLOCK_SIZE..(bi + 1) * BLOCK_SIZE];
        for k in 0..BLOCK_SIZE {
            out_b[k] = p[k] * scale + base + r[k].to_f32() / INPUT_SCALE;
        }
    }
}

/// A WPC-compressed `Linear` layer: `W` is `[out_features, in_features]`,
/// stored as `out_features * (in_features/16)` consecutive `CompressedBlock`s
/// (row-major, matching the tensor's original flattened layout).
pub struct WpcLinear {
    data: Arc<WpcModelData>,
    tensor_name: String,
    out_features: usize,
    in_features: usize,
    bias: Option<Vec<f32>>,
}

impl WpcLinear {
    pub fn new(
        data: Arc<WpcModelData>,
        tensor_name: &str,
        out_features: usize,
        in_features: usize,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(
            in_features % BLOCK_SIZE,
            0,
            "in_features must be a multiple of {BLOCK_SIZE} for WPC blocks"
        );
        // Validate the tensor exists and has the expected number of blocks.
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = out_features * (in_features / BLOCK_SIZE);
        assert_eq!(
            size / CompressedBlock::SIZE,
            expected_blocks,
            "tensor {tensor_name}: block count mismatch (meta says {}, shape implies {})",
            size / CompressedBlock::SIZE,
            expected_blocks
        );
        WpcLinear {
            data,
            tensor_name: tensor_name.to_string(),
            out_features,
            in_features,
            bias,
        }
    }
}

impl Linear for WpcLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        let blocks_per_row = self.in_features / BLOCK_SIZE;
        let blocks = self.data.blocks_for(&self.tensor_name);

        // Resolve class and dictionary once per matvec call
        let class = self.data.class_for(&self.tensor_name);
        let (patterns, residuals) = self.data.dict_for(class);
        let has_avx2 = self.data.has_avx2_fma_f16c;

        use rayon::prelude::*;
        y.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                *o = row_dot(has_avx2, row_blocks, patterns, residuals, x);
            });
        if let Some(b) = &self.bias {
            for (o, bi) in y.iter_mut().zip(b.iter()) {
                *o += *bi;
            }
        }
    }
}

/// WPC-compressed embedding / tied lm_head table: `[vocab_size, hidden_size]`.
pub struct WpcEmbedding {
    data: Arc<WpcModelData>,
    tensor_name: String,
    vocab_size: usize,
    hidden_size: usize,
}

impl WpcEmbedding {
    pub fn new(
        data: Arc<WpcModelData>,
        tensor_name: &str,
        vocab_size: usize,
        hidden_size: usize,
    ) -> Self {
        assert_eq!(
            hidden_size % BLOCK_SIZE,
            0,
            "hidden_size must be a multiple of {BLOCK_SIZE} for WPC blocks"
        );
        let (_, size) = data.tensor_range(tensor_name);
        let expected_blocks = vocab_size * (hidden_size / BLOCK_SIZE);
        assert_eq!(
            size / CompressedBlock::SIZE,
            expected_blocks,
            "tensor {tensor_name}: block count mismatch (meta says {}, shape implies {})",
            size / CompressedBlock::SIZE,
            expected_blocks
        );
        WpcEmbedding {
            data,
            tensor_name: tensor_name.to_string(),
            vocab_size,
            hidden_size,
        }
    }
}

impl EmbeddingTable for WpcEmbedding {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn embed(&self, token_id: u32, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.hidden_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE;
        let all_blocks = self.data.blocks_for(&self.tensor_name);

        // Resolve class and dictionary once
        let class = self.data.class_for(&self.tensor_name);
        let (patterns, residuals) = self.data.dict_for(class);

        let row_start = token_id as usize * blocks_per_row;
        let row_blocks = &all_blocks[row_start..row_start + blocks_per_row];
        row_decode_into(row_blocks, patterns, residuals, out);
    }
    fn logits(&self, x: &[f32], out: &mut [f32]) {
        debug_assert_eq!(x.len(), self.hidden_size);
        debug_assert_eq!(out.len(), self.vocab_size);
        let blocks_per_row = self.hidden_size / BLOCK_SIZE;
        let blocks = self.data.blocks_for(&self.tensor_name);

        // Resolve class and dictionary once per logits call
        let class = self.data.class_for(&self.tensor_name);
        let (patterns, residuals) = self.data.dict_for(class);
        let has_avx2 = self.data.has_avx2_fma_f16c;

        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(blocks.par_chunks_exact(blocks_per_row))
            .for_each(|(o, row_blocks)| {
                *o = row_dot(has_avx2, row_blocks, patterns, residuals, x);
            });
    }
}
