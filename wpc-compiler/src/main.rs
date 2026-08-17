use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use wpc_core::codebook::{PatternDict, ResidualDict, BLOCK_DIM};
use wpc_core::encoder::encode_block;
use wpc_format::{CompressedBlock, PATTERN_COUNT, RESIDUAL_COUNT};
use safetensors::SafeTensors;
use half::f16;

#[derive(Parser, Debug)]
#[command(author, version, about = "WPC Full-Model Compiler")]
struct Args {
    /// Path to the Hugging Face model directory (containing .safetensors files)
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for the compiled model (.wpc file or directory)
    #[arg(short, long)]
    output: PathBuf,
}

#[derive(Serialize)]
struct LayerMeta {
    name: String,
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
}

#[derive(Serialize)]
struct ModelMeta {
    layers: Vec<LayerMeta>,
    global_patterns: String,
    global_residuals: String,
}

struct TensorRef {
    shard_path: PathBuf,
    name: String,
    shape: Vec<usize>,
    num_blocks: usize,
}

fn get_raw_residual(weights: &[f32; BLOCK_DIM], pattern_dict: &PatternDict) -> [f32; BLOCK_DIM] {
    let mut sum = 0.0;
    for &w in weights { sum += w; }
    let base = sum / BLOCK_DIM as f32;

    let mut centered = [0.0; BLOCK_DIM];
    let mut max_abs = 0.0_f32;
    for i in 0..BLOCK_DIM {
        centered[i] = weights[i] - base;
        let abs_c = centered[i].abs();
        if abs_c > max_abs { max_abs = abs_c; }
    }

    let scale_f32 = (max_abs * 127.0).round();
    let scale_i8 = scale_f32.clamp(-127.0, 127.0) as i8;
    
    let mut norm = [0.0; BLOCK_DIM];
    if scale_i8 != 0 {
        let inv_scale = 127.0 / scale_i8 as f32;
        for i in 0..BLOCK_DIM { norm[i] = centered[i] * inv_scale; }
    }
    
    let (pid, _) = pattern_dict.nearest(&norm);
    let p_vec = pattern_dict.centroids[pid as usize];

    let mut res = [0.0; BLOCK_DIM];
    let s_decode = scale_i8 as f32 / 127.0;
    for i in 0..BLOCK_DIM {
        let approx = p_vec[i] * s_decode + base;
        res[i] = (weights[i] - approx) * 127.0;
    }
    res
}

fn read_tensor_f32(shard_path: &Path, name: &str) -> Vec<f32> {
    let file = File::open(shard_path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file) }.unwrap();
    let st = SafeTensors::deserialize(&mmap).unwrap();
    let view = st.tensor(name).unwrap();
    
    match view.dtype() {
        safetensors::Dtype::F32 => {
            let raw = view.data();
            raw.chunks_exact(4)
               .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
               .collect()
        }
        safetensors::Dtype::F16 => {
            let raw = view.data();
            raw.chunks_exact(2)
               .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
               .collect()
        }
        safetensors::Dtype::BF16 => {
            let raw = view.data();
            raw.chunks_exact(2)
               .map(|c| {
                   let bits = u32::from(c[0]) | (u32::from(c[1]) << 8);
                   f32::from_bits(bits << 16)
               })
               .collect()
        }
        _ => panic!("Unsupported dtype {:?}", view.dtype()),
    }
}

fn main() {
    let args = Args::parse();
    
    println!("Scanning directory for .safetensors shards...");
    let mut shards = Vec::new();
    for entry in WalkDir::new(&args.input).into_iter().filter_map(|e| e.ok()) {
        if entry.path().extension().and_then(|s| s.to_str()) == Some("safetensors") {
            shards.push(entry.path().to_path_buf());
        }
    }
    
    if shards.is_empty() {
        eprintln!("No .safetensors files found in the input directory.");
        return;
    }
    
    println!("Found {} shards. Extracting tensor metadata...", shards.len());
    
    let mut all_tensors = Vec::new();
    for shard in &shards {
        let file = match File::open(shard) {
            Ok(f) => f,
            Err(_) => continue,
        };
        let mmap = match unsafe { memmap2::Mmap::map(&file) } {
            Ok(m) => m,
            Err(_) => continue,
        };
        let st = match SafeTensors::deserialize(&mmap) {
            Ok(s) => s,
            Err(_) => continue,
        };
        
        for (name, view) in st.tensors() {
            let shape = view.shape().to_vec();
            if shape.len() != 2 { continue; }
            
            let n: usize = shape.iter().product();
            if n < 4096 { continue; }
            
            match view.dtype() {
                safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {}
                _ => continue,
            }
            
            all_tensors.push(TensorRef {
                shard_path: shard.clone(),
                name: name.to_string(),
                shape,
                num_blocks: n / BLOCK_DIM,
            });
        }
    }
    
    let total_blocks: usize = all_tensors.iter().map(|t| t.num_blocks).sum();
    println!("Extracted {} valid 2D matrix tensors. Total blocks: {}", all_tensors.len(), total_blocks);
    
    if total_blocks == 0 {
        eprintln!("No valid blocks found to compress.");
        return;
    }

    let target_sample = 65536; 
    let sample_rate = (target_sample as f64 / total_blocks as f64).min(1.0);
    
    println!("Sampling blocks across all tensors (rate: {:.4})...", sample_rate);
    let mut sampled_blocks = Vec::new();
    use rand::{Rng, thread_rng};
    let mut rng = thread_rng();
    
    for t_ref in &all_tensors {
        // Read just this one tensor into memory to sample from it
        let mut t_data = None;
        for i in 0..t_ref.num_blocks {
            if rng.gen_bool(sample_rate) {
                if t_data.is_none() {
                    t_data = Some(read_tensor_f32(&t_ref.shard_path, &t_ref.name));
                }
                let data = t_data.as_ref().unwrap();
                let mut block = [0.0; BLOCK_DIM];
                block.copy_from_slice(&data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
                sampled_blocks.push(block);
            }
        }
    }
    
    if sampled_blocks.is_empty() {
        let t_ref = &all_tensors[0];
        let data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);
        let mut block = [0.0; BLOCK_DIM];
        block.copy_from_slice(&data[0..BLOCK_DIM]);
        sampled_blocks.push(block);
    }
    
    println!("Collected {} samples. Training Global Pattern Codebook...", sampled_blocks.len());
    let pattern_dict = PatternDict::train(&sampled_blocks, PATTERN_COUNT, 20);
    
    println!("Computing residuals for training...");
    let residuals: Vec<[f32; BLOCK_DIM]> = sampled_blocks.par_iter()
        .map(|b| get_raw_residual(b, &pattern_dict))
        .collect();
        
    println!("Training Global Residual Codebook...");
    let residual_k = RESIDUAL_COUNT.min(residuals.len());
    let residual_dict = ResidualDict::train(&residuals, residual_k, 10);
    
    // Free samples memory
    drop(sampled_blocks);
    drop(residuals);
    
    let mut out_dir = args.output.clone();
    if out_dir.is_file() || out_dir.extension().is_some() {
        out_dir = out_dir.parent().unwrap_or(Path::new("")).to_path_buf();
    }
    std::fs::create_dir_all(&out_dir).unwrap();
    
    let patterns_path = out_dir.join("global_patterns.bin");
    let residuals_path = out_dir.join("global_residuals.bin");
    
    let mut p_file = File::create(&patterns_path).unwrap();
    for c in &pattern_dict.centroids {
        for &val in c {
            p_file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    
    let mut r_file = File::create(&residuals_path).unwrap();
    for c in &residual_dict.centroids_f16 {
        for &val in c {
            r_file.write_all(&val.to_le_bytes()).unwrap();
        }
    }
    println!("Saved global codebooks.");

    println!("Starting multi-threaded streaming block compression...");
    let out_wpc = if args.output.extension().and_then(|e| e.to_str()) == Some("wpc") {
        args.output.clone()
    } else {
        out_dir.join("model.wpc")
    };
    
    let mut wpc_file = File::create(&out_wpc).unwrap();
    let mut current_offset = 0;
    
    let mut meta = ModelMeta {
        layers: Vec::new(),
        global_patterns: "global_patterns.bin".to_string(),
        global_residuals: "global_residuals.bin".to_string(),
    };
    
    for (idx, t_ref) in all_tensors.iter().enumerate() {
        println!("  Encoding tensor {}/{} ({})...", idx + 1, all_tensors.len(), t_ref.name);
        let t_data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);
        
        let compressed_blocks: Vec<CompressedBlock> = (0..t_ref.num_blocks)
            .into_par_iter()
            .map(|i| {
                let mut block = [0.0; BLOCK_DIM];
                block.copy_from_slice(&t_data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
                let (cb, _) = encode_block(&block, &pattern_dict, &residual_dict);
                cb
            })
            .collect();
            
        let size_bytes = compressed_blocks.len() * CompressedBlock::SIZE;
        meta.layers.push(LayerMeta {
            name: t_ref.name.clone(),
            shape: t_ref.shape.clone(),
            offset_bytes: current_offset,
            size_bytes,
        });
        
        // Write the raw bytes (this is extremely fast)
        let slice = unsafe {
            std::slice::from_raw_parts(
                compressed_blocks.as_ptr() as *const u8,
                size_bytes,
            )
        };
        wpc_file.write_all(slice).unwrap();
        
        current_offset += size_bytes;
        
        // Drop the data explicitly to keep memory flat
        drop(t_data);
    }
    
    let meta_path = out_dir.join("model.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();
    
    println!("Done! Compiled WPC model saved to {:?}", out_dir);
}
