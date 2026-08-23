use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressionInput {
    pub session_id: String,
    pub original_bytes: usize,
    pub run_length: usize,
}

impl CompressionInput {
    pub fn new(session_id: impl Into<String>, original_bytes: usize, run_length: usize) -> Self {
        Self {
            session_id: session_id.into(),
            original_bytes,
            run_length,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompressionResult {
    pub session_id: String,
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub compression_ratio: f64,
    pub generation_critical: bool,
}

pub struct CompressionExperiment;

impl CompressionExperiment {
    pub fn run(input: CompressionInput) -> Result<CompressionResult, &'static str> {
        if input.session_id.is_empty() {
            return Err("session id must not be empty");
        }
        if input.original_bytes == 0 {
            return Err("original size must be non-zero");
        }
        if input.run_length == 0 {
            return Err("run length must be non-zero");
        }

        let compressed_bytes = rle_probe_size(input.original_bytes, input.run_length);
        Ok(CompressionResult {
            session_id: input.session_id,
            original_bytes: input.original_bytes,
            compressed_bytes,
            compression_ratio: input.original_bytes as f64 / compressed_bytes as f64,
            generation_critical: false,
        })
    }
}

fn rle_probe_size(original_bytes: usize, run_length: usize) -> usize {
    let runs = original_bytes.div_ceil(run_length);
    runs.saturating_mul(2).max(1).min(original_bytes)
}
