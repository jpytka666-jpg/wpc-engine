//! Read-only access to an existing WPC artifact.
//!
//! This crate deliberately exposes no write, truncate, rename, delete, or
//! repack API. It opens the existing `.meta` and `.wpc` files read-only and
//! exposes validated byte slices for individual tensors. The caller owns the
//! directory; this crate never creates or modifies anything there.

use anyhow::{bail, ensure, Context, Result};
use memmap2::{Mmap, MmapOptions};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub offset_bytes: usize,
    pub size_bytes: usize,
}

#[derive(Debug, Deserialize)]
struct ModelMeta {
    layers: Vec<TensorInfo>,
    block_size: usize,
}

/// An immutable view over an already-existing WPC artifact.
///
/// The model payload is memory-mapped read-only. The bridge has intentionally
/// no method that can write back to the source files.
pub struct ReadonlyWpcArtifact {
    root: PathBuf,
    stem: String,
    mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
    block_size: usize,
}

impl ReadonlyWpcArtifact {
    /// Open `<root>/<stem>.meta` and `<root>/<stem>.wpc` strictly for reading.
    pub fn open(root: impl AsRef<Path>, stem: impl Into<String>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        let stem = stem.into();
        ensure!(!stem.is_empty(), "WPC artifact stem must not be empty");

        let meta_path = root.join(format!("{stem}.meta"));
        let model_path = root.join(format!("{stem}.wpc"));

        let meta_text = std::fs::read_to_string(&meta_path)
            .with_context(|| format!("read WPC metadata: {}", meta_path.display()))?;
        let meta: ModelMeta = serde_json::from_str(&meta_text)
            .with_context(|| format!("parse WPC metadata: {}", meta_path.display()))?;
        ensure!(meta.block_size > 0, "WPC block_size must be positive");

        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .open(&model_path)
            .with_context(|| format!("open WPC payload read-only: {}", model_path.display()))?;
        let file_len = file.metadata()?.len();
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .with_context(|| format!("mmap WPC payload read-only: {}", model_path.display()))?;

        ensure!(
            file_len == mmap.len() as u64,
            "mapped WPC length changed while opening"
        );

        let mut tensors = HashMap::with_capacity(meta.layers.len());
        let mut ranges = Vec::with_capacity(meta.layers.len());
        for tensor in meta.layers {
            let end = tensor
                .offset_bytes
                .checked_add(tensor.size_bytes)
                .ok_or_else(|| anyhow::anyhow!("tensor range overflow: {}", tensor.name))?;
            ensure!(
                end <= mmap.len(),
                "tensor {} exceeds WPC payload: offset={} size={} payload={}",
                tensor.name,
                tensor.offset_bytes,
                tensor.size_bytes,
                mmap.len()
            );
            ensure!(
                tensors.insert(tensor.name.clone(), tensor.clone()).is_none(),
                "duplicate tensor metadata: {}",
                tensor.name
            );
            ranges.push((tensor.offset_bytes, end, tensor.name));
        }

        ranges.sort_unstable_by_key(|(start, _, _)| *start);
        for pair in ranges.windows(2) {
            let (_, prev_end, prev_name) = &pair[0];
            let (next_start, _, next_name) = &pair[1];
            if prev_end > next_start {
                bail!(
                    "overlapping tensor ranges: {} overlaps {}",
                    prev_name,
                    next_name
                );
            }
        }

        Ok(Self {
            root,
            stem,
            mmap,
            tensors,
            block_size: meta.block_size,
        })
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn payload_len(&self) -> usize {
        self.mmap.len()
    }

    pub fn source_dir(&self) -> &Path {
        &self.root
    }

    pub fn artifact_stem(&self) -> &str {
        &self.stem
    }

    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Borrow the existing tensor bytes directly from the read-only mmap.
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .with_context(|| format!("tensor not found: {name}"))?;
        let end = info
            .offset_bytes
            .checked_add(info.size_bytes)
            .ok_or_else(|| anyhow::anyhow!("tensor range overflow: {name}"))?;
        Ok(&self.mmap[info.offset_bytes..end])
    }

    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn opens_existing_artifact_without_mutating_it() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("model_v3.meta"),
            r#"{
                "layers": [{"name":"k","shape":[2,4],"offset_bytes":0,"size_bytes":8}],
                "block_size":128
            }"#,
        )
        .expect("meta");
        let payload: Vec<u8> = (0..8u8).collect();
        fs::write(dir.path().join("model_v3.wpc"), &payload).expect("payload");

        let artifact = ReadonlyWpcArtifact::open(dir.path(), "model_v3").expect("open");
        assert_eq!(artifact.block_size(), 128);
        assert_eq!(artifact.payload_len(), 8);
        assert_eq!(artifact.tensor_bytes("k").expect("tensor"), payload.as_slice());
        assert!(artifact.tensor_bytes("missing").is_err());
    }

    #[test]
    fn rejects_out_of_bounds_metadata() {
        let dir = tempfile::tempdir().expect("tempdir");
        fs::write(
            dir.path().join("model_v3.meta"),
            r#"{"layers":[{"name":"bad","shape":[1],"offset_bytes":4,"size_bytes":8}],"block_size":128}"#,
        )
        .expect("meta");
        fs::write(dir.path().join("model_v3.wpc"), [0u8; 8]).expect("payload");

        let result = ReadonlyWpcArtifact::open(dir.path(), "model_v3");
        let error = match result {
            Ok(_) => panic!("must reject"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("exceeds WPC payload"));
    }
}
