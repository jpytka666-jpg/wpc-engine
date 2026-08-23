//! Weight access abstraction.
//!
//! Phase A only ships a dense (f32, decoded-from-bf16) backend, but every
//! consumer of weights goes through the `Linear` / `EmbeddingTable` traits
//! below rather than touching raw `&[f32]` directly. A future WPC-compressed
//! backend just needs to implement these two traits; nothing in `model.rs`
//! needs to change.

use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// A single dense linear layer: `y = W x + b`, `W` is `[out_features, in_features]`
/// row-major, matching the PyTorch / safetensors convention for `nn.Linear.weight`.
pub trait Linear: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    /// y = W x (+ bias if present). `x.len() == in_features`, `y.len() == out_features`.
    fn matvec(&self, x: &[f32], y: &mut [f32]);
}

/// Token embedding / (possibly tied) LM head table: `[vocab_size, hidden_size]`.
pub trait EmbeddingTable: Send + Sync {
    fn vocab_size(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn embed(&self, token_id: u32, out: &mut [f32]);
    /// logits = table * x  (used for the tied lm_head).
    fn logits(&self, x: &[f32], out: &mut [f32]);
}

/// Dense row-major f32 weight matrix, decoded once at load time from the
/// safetensors file (bf16 -> f32). Backing storage for Phase A.
pub struct DenseLinear {
    out_features: usize,
    in_features: usize,
    weight: Vec<f32>, // [out_features, in_features]
    bias: Option<Vec<f32>>,
}

impl DenseLinear {
    pub fn new(
        out_features: usize,
        in_features: usize,
        weight: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Self {
        assert_eq!(weight.len(), out_features * in_features);
        if let Some(b) = &bias {
            assert_eq!(b.len(), out_features);
        }
        DenseLinear {
            out_features,
            in_features,
            weight,
            bias,
        }
    }
}

impl Linear for DenseLinear {
    fn in_features(&self) -> usize {
        self.in_features
    }
    fn out_features(&self) -> usize {
        self.out_features
    }
    fn matvec(&self, x: &[f32], y: &mut [f32]) {
        debug_assert_eq!(x.len(), self.in_features);
        debug_assert_eq!(y.len(), self.out_features);
        let in_f = self.in_features;
        use rayon::prelude::*;
        y.par_iter_mut()
            .zip(self.weight.par_chunks_exact(in_f))
            .for_each(|(o, row)| {
                let mut acc = 0.0f32;
                for i in 0..in_f {
                    acc += row[i] * x[i];
                }
                *o = acc;
            });
        if let Some(b) = &self.bias {
            for (o, bi) in y.iter_mut().zip(b.iter()) {
                *o += *bi;
            }
        }
    }
}

pub struct DenseEmbedding {
    vocab_size: usize,
    hidden_size: usize,
    table: Vec<f32>, // [vocab_size, hidden_size]
}

impl DenseEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize, table: Vec<f32>) -> Self {
        assert_eq!(table.len(), vocab_size * hidden_size);
        DenseEmbedding {
            vocab_size,
            hidden_size,
            table,
        }
    }
}

impl EmbeddingTable for DenseEmbedding {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn embed(&self, token_id: u32, out: &mut [f32]) {
        let h = self.hidden_size;
        let row = &self.table[token_id as usize * h..(token_id as usize + 1) * h];
        out.copy_from_slice(row);
    }
    fn logits(&self, x: &[f32], out: &mut [f32]) {
        let h = self.hidden_size;
        debug_assert_eq!(x.len(), h);
        debug_assert_eq!(out.len(), self.vocab_size);
        use rayon::prelude::*;
        out.par_iter_mut()
            .zip(self.table.par_chunks_exact(h))
            .for_each(|(o, row)| {
                let mut acc = 0.0f32;
                for i in 0..h {
                    acc += row[i] * x[i];
                }
                *o = acc;
            });
    }
}

/// Minimal mmap + bf16->f32 safetensors reader (dense, whole-tensor decode).
/// Same bit trick as wpc-compiler's `read_tensor_f32` (upper 16 bits of f32).
pub struct SafetensorsFile {
    mmap: Mmap,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(SafetensorsFile { mmap })
    }

    fn view(&self) -> SafeTensors<'_> {
        SafeTensors::deserialize(&self.mmap).expect("invalid safetensors file")
    }

    pub fn names(&self) -> Vec<String> {
        self.view()
            .names()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        self.view()
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing tensor {name}"))
            .shape()
            .to_vec()
    }

    /// Decode a tensor (F32/F16/BF16) to a flat row-major `Vec<f32>`.
    pub fn read_f32(&self, name: &str) -> Vec<f32> {
        let st = self.view();
        let t = st
            .tensor(name)
            .unwrap_or_else(|_| panic!("missing tensor {name}"));
        match t.dtype() {
            Dtype::F32 => {
                let raw = t.data();
                raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            Dtype::F16 => {
                let raw = t.data();
                raw.chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
            Dtype::BF16 => {
                let raw = t.data();
                raw.chunks_exact(2)
                    .map(|c| {
                        let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                        f32::from_bits(bits << 16)
                    })
                    .collect()
            }
            other => panic!("unsupported dtype {other:?} for tensor {name}"),
        }
    }
}

/// A whole checkpoint directory, whether it is one `model.safetensors` or a
/// set of shards described by `model.safetensors.index.json`.
///
/// Hugging Face splits any checkpoint over ~5 GB into
/// `model-00001-of-000NN.safetensors` files plus an index that maps every
/// tensor name to the shard holding it; Qwen3-4B ships that way. Reading only
/// `model.safetensors` would fail on such a directory before a single weight
/// was touched, so this resolves names across shards. A directory that does
/// have a plain `model.safetensors` is served from that one file, which makes
/// this a drop-in superset of [`SafetensorsFile`].
pub struct ShardedSafetensors {
    shards: Vec<SafetensorsFile>,
    /// tensor name -> index into `shards`. Empty for a single-file checkpoint,
    /// where every lookup goes to shard 0.
    index: HashMap<String, usize>,
}

impl ShardedSafetensors {
    pub fn open(model_dir: &Path) -> anyhow::Result<Self> {
        let single = model_dir.join("model.safetensors");
        if single.exists() {
            return Ok(ShardedSafetensors {
                shards: vec![SafetensorsFile::open(&single)?],
                index: HashMap::new(),
            });
        }

        let index_path = model_dir.join("model.safetensors.index.json");
        if !index_path.exists() {
            anyhow::bail!(
                "{}: contains neither model.safetensors nor model.safetensors.index.json",
                model_dir.display()
            );
        }
        let text = std::fs::read_to_string(&index_path)?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let weight_map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow::anyhow!("{}: no `weight_map` object", index_path.display()))?;

        let mut shards: Vec<SafetensorsFile> = Vec::new();
        let mut shard_of_file: HashMap<String, usize> = HashMap::new();
        let mut index: HashMap<String, usize> = HashMap::new();
        for (tensor, file) in weight_map {
            let Some(file) = file.as_str() else { continue };
            let slot = match shard_of_file.get(file) {
                Some(&i) => i,
                None => {
                    shards.push(SafetensorsFile::open(&model_dir.join(file))?);
                    let i = shards.len() - 1;
                    shard_of_file.insert(file.to_string(), i);
                    i
                }
            };
            index.insert(tensor.clone(), slot);
        }
        eprintln!(
            "safetensors: {} shard(s), {} tensors indexed",
            shards.len(),
            index.len()
        );
        Ok(ShardedSafetensors { shards, index })
    }

    /// Decode a tensor to a flat row-major `Vec<f32>`, from whichever shard
    /// holds it. Panics on an unknown name, matching [`SafetensorsFile`].
    pub fn read_f32(&self, name: &str) -> Vec<f32> {
        let slot = match self.index.get(name) {
            Some(&i) => i,
            None if self.index.is_empty() => 0,
            None => panic!("missing tensor {name} (not in safetensors index)"),
        };
        self.shards[slot].read_f32(name)
    }

    /// Shape of a tensor, from whichever shard holds it.
    pub fn shape(&self, name: &str) -> Vec<usize> {
        let slot = match self.index.get(name) {
            Some(&i) => i,
            None if self.index.is_empty() => 0,
            None => panic!("missing tensor {name} (not in safetensors index)"),
        };
        self.shards[slot].shape(name)
    }
}
