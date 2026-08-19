use clap::Parser;
use rayon::prelude::*;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use walkdir::WalkDir;
use wpc_core::codebook::{PatternDict, ResidualDict, BLOCK_DIM};
use wpc_core::encoder::{encode_block, normalize_block, BlockNorm, INPUT_SCALE};
use wpc_core::quant_encoder;
use wpc_format::{CompressedBlock, QuantBlockV2, PATTERN_COUNT, RESIDUAL_COUNT, BLOCK_SIZE_V2};
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

    /// Compression scheme: "v1" (VQ-codebook) or "v2" (affine 6-bit)
    #[arg(long, default_value = "v1")]
    scheme: String,
}

#[derive(Serialize)]
struct LayerMeta {
    name: String,
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
    dict_class: String,
}

#[derive(Serialize)]
struct DictFiles {
    patterns: String,
    residuals: String,
}

#[derive(Serialize)]
struct ModelMeta {
    layers: Vec<LayerMeta>,
    dictionaries: HashMap<String, DictFiles>,
}

#[derive(Serialize)]
struct LayerMetaV2 {
    name: String,
    shape: Vec<usize>,
    offset_bytes: usize,
    size_bytes: usize,
}

#[derive(Serialize)]
struct ModelMetaV2 {
    layers: Vec<LayerMetaV2>,
    block_size: usize,
}

struct TensorRef {
    shard_path: PathBuf,
    name: String,
    shape: Vec<usize>,
    num_blocks: usize,
}

fn classify_tensor(name: &str) -> &'static str {
    if name.contains("embed_tokens") || name.ends_with("lm_head.weight") {
        "embed"
    } else if name.contains("q_proj") {
        "q_proj"
    } else if name.contains("k_proj") {
        "k_proj"
    } else if name.contains("v_proj") {
        "v_proj"
    } else if name.contains("o_proj") {
        "o_proj"
    } else if name.contains("gate_proj") {
        "gate_proj"
    } else if name.contains("up_proj") {
        "up_proj"
    } else if name.contains("down_proj") {
        "down_proj"
    } else {
        "other"
    }
}

fn get_raw_residual(weights: &[f32; BLOCK_DIM], pattern_dict: &PatternDict) -> [f32; BLOCK_DIM] {
    let BlockNorm { base, scale_i8, norm } = normalize_block(weights);

    let (pid, _) = pattern_dict.nearest(&norm);
    let p_vec = pattern_dict.centroids[pid as usize];

    let mut res = [0.0; BLOCK_DIM];
    let s_decode = scale_i8 as f32 / INPUT_SCALE;
    for i in 0..BLOCK_DIM {
        let approx = p_vec[i] * s_decode + base;
        res[i] = (weights[i] - approx) * INPUT_SCALE;
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

    if args.scheme != "v1" && args.scheme != "v2" {
        eprintln!("Invalid scheme '{}'. Must be 'v1' or 'v2'.", args.scheme);
        return;
    }

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

            // For v2, validate that in_features (shape[1]) is divisible by BLOCK_SIZE_V2
            if args.scheme == "v2" && shape[1] % BLOCK_SIZE_V2 != 0 {
                eprintln!("Skipping tensor '{}' with shape {:?}: in_features {} not divisible by BLOCK_SIZE_V2 ({})",
                    name, shape, shape[1], BLOCK_SIZE_V2);
                continue;
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

    let mut out_dir = args.output.clone();
    if out_dir.is_file() || out_dir.extension().is_some() {
        out_dir = out_dir.parent().unwrap_or(Path::new("")).to_path_buf();
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    if args.scheme == "v2" {
        compress_v2(&args, &all_tensors, &out_dir);
    } else {
        compress_v1(&args, &all_tensors, &out_dir);
    }
}

fn compress_v1(args: &Args, all_tensors: &[TensorRef], out_dir: &Path) {
    // Group tensors by class
    let mut class_groups: HashMap<&'static str, Vec<&TensorRef>> = HashMap::new();
    for t_ref in all_tensors {
        let class = classify_tensor(&t_ref.name);
        class_groups.entry(class).or_insert_with(Vec::new).push(t_ref);
    }

    // Train per-class codebooks
    let mut class_dicts: HashMap<&'static str, (PatternDict, ResidualDict)> = HashMap::new();
    let mut dict_files: HashMap<String, DictFiles> = HashMap::new();
    use rand::{Rng, thread_rng};

    for (class, tensors) in &class_groups {
        if tensors.is_empty() {
            continue;
        }

        let class_blocks: usize = tensors.iter().map(|t| t.num_blocks).sum();
        let target_sample = 262144;
        let sample_rate = (target_sample as f64 / class_blocks as f64).min(1.0);

        println!("Training codebook for class '{}' ({} tensors, {} blocks)...", class, tensors.len(), class_blocks);

        // Sample from this class only
        let mut sampled_blocks = Vec::new();
        let mut rng = thread_rng();

        for t_ref in tensors {
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
            let t_ref = tensors[0];
            let data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);
            let mut block = [0.0; BLOCK_DIM];
            block.copy_from_slice(&data[0..BLOCK_DIM]);
            sampled_blocks.push(block);
        }

        // Train pattern dict for this class
        let normalized_samples: Vec<[f32; BLOCK_DIM]> = sampled_blocks.iter()
            .map(|b| normalize_block(b).norm)
            .collect();
        let pattern_dict = PatternDict::train(&normalized_samples, PATTERN_COUNT, 20);

        // Compute residuals
        let residuals: Vec<[f32; BLOCK_DIM]> = sampled_blocks.par_iter()
            .map(|b| get_raw_residual(b, &pattern_dict))
            .collect();

        // Train residual dict for this class
        let residual_k = RESIDUAL_COUNT.min(residuals.len());
        let residual_dict = ResidualDict::train(&residuals, residual_k, 10);

        // Write patterns and residuals files
        let patterns_filename = format!("patterns_{}.bin", class);
        let residuals_filename = format!("residuals_{}.bin", class);
        let patterns_path = out_dir.join(&patterns_filename);
        let residuals_path = out_dir.join(&residuals_filename);

        let mut p_file = File::create(&patterns_path).unwrap();
        for c in &pattern_dict.centroids {
            for &val in c {
                p_file.write_all(&(val / INPUT_SCALE).to_le_bytes()).unwrap();
            }
        }

        let mut r_file = File::create(&residuals_path).unwrap();
        for c in &residual_dict.centroids_f16 {
            for &val in c {
                r_file.write_all(&val.to_le_bytes()).unwrap();
            }
        }

        dict_files.insert(class.to_string(), DictFiles {
            patterns: patterns_filename,
            residuals: residuals_filename,
        });

        class_dicts.insert(*class, (pattern_dict, residual_dict));
    }

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
        dictionaries: dict_files,
    };

    for (idx, t_ref) in all_tensors.iter().enumerate() {
        let class = classify_tensor(&t_ref.name);
        println!("  Encoding tensor {}/{} ({})...", idx + 1, all_tensors.len(), t_ref.name);
        let t_data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);

        let (pattern_dict, residual_dict) = &class_dicts[&class];

        let compressed_blocks: Vec<CompressedBlock> = (0..t_ref.num_blocks)
            .into_par_iter()
            .map(|i| {
                let mut block = [0.0; BLOCK_DIM];
                block.copy_from_slice(&t_data[i * BLOCK_DIM..(i + 1) * BLOCK_DIM]);
                let (cb, _) = encode_block(&block, pattern_dict, residual_dict);
                cb
            })
            .collect();

        let size_bytes = compressed_blocks.len() * CompressedBlock::SIZE;
        meta.layers.push(LayerMeta {
            name: t_ref.name.clone(),
            shape: t_ref.shape.clone(),
            offset_bytes: current_offset,
            size_bytes,
            dict_class: class.to_string(),
        });

        let bytes: Vec<u8> = compressed_blocks.iter().flat_map(|b| b.to_le_bytes()).collect();
        wpc_file.write_all(&bytes).unwrap();

        current_offset += size_bytes;

        drop(t_data);
    }

    let meta_path = out_dir.join("model.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();

    println!("Done! Compiled WPC model saved to {:?}", out_dir);
}

fn compress_v2(_args: &Args, all_tensors: &[TensorRef], out_dir: &Path) {
    println!("Starting v2 (affine 6-bit) compression...");
    let out_wpc = out_dir.join("model_v2.wpc");
    let mut wpc_file = File::create(&out_wpc).unwrap();
    let mut current_offset = 0;

    let mut meta = ModelMetaV2 {
        layers: Vec::new(),
        block_size: BLOCK_SIZE_V2,
    };

    for (idx, t_ref) in all_tensors.iter().enumerate() {
        println!("  Encoding tensor {}/{} ({})...", idx + 1, all_tensors.len(), t_ref.name);
        let t_data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);

        // Encode with v2
        let compressed_blocks: Vec<QuantBlockV2> = quant_encoder::encode_tensor_v2(&t_data);

        let size_bytes = compressed_blocks.len() * QuantBlockV2::SIZE;
        meta.layers.push(LayerMetaV2 {
            name: t_ref.name.clone(),
            shape: t_ref.shape.clone(),
            offset_bytes: current_offset,
            size_bytes,
        });

        let bytes: Vec<u8> = compressed_blocks.iter().flat_map(|b| b.to_le_bytes()).collect();
        wpc_file.write_all(&bytes).unwrap();

        current_offset += size_bytes;

        drop(t_data);
    }

    let meta_path = out_dir.join("model_v2.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();

    println!("Done! Compiled WPC v2 model saved to {:?}", out_dir);
}
