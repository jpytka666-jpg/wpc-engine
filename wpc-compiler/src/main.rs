use clap::Parser;
use half::f16;
use rayon::prelude::*;
use safetensors::SafeTensors;
use serde::Serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;
use wpc_core::codebook::{PatternDict, ResidualDict, BLOCK_DIM};
use wpc_core::encoder::{encode_block, normalize_block, BlockNorm, INPUT_SCALE};
use wpc_core::quant_encoder;
use wpc_format::{
    CompressedBlock, QuantBlockV2, QuantBlockV3, QuantBlockV4, BLOCK_SIZE_V2, PATTERN_COUNT,
    RESIDUAL_COUNT,
};

#[derive(Parser, Debug)]
#[command(author, version, about = "WPC Full-Model Compiler")]
struct Args {
    /// Path to the Hugging Face model directory (containing .safetensors files)
    #[arg(short, long)]
    input: PathBuf,

    /// Output path for the compiled model (.wpc file or directory)
    #[arg(short, long)]
    output: PathBuf,

    /// Compression scheme: "v3" (packed affine 6-bit, 6.25 bits/weight - default),
    /// "v4" (packed affine 4-bit, 4.25 bits/weight - for cards too small for v3),
    /// "v2" (byte-aligned affine, 8.25 bits/weight) or "v1" (VQ-codebook, superseded).
    #[arg(long, default_value = "v3")]
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

/// Sort key: (layer, section, expert, projection).
///
/// Tensors arrive in shard/manifest order, which is effectively random: the
/// first tensors of layer 0 were experts 99, 109, 77, 80, 47, ... and each
/// expert's gate/up/down sat far apart in the file. At generation time a layer
/// activates 8 of its 128 experts, so that layout turned every expert into
/// three seeks across a 22 GB file - 1152 random jumps per token - which the
/// hardware prefetcher cannot follow. Measured effective bandwidth was
/// 2.14 GB/s against ~10 GB/s the machine sustains on sequential reads.
///
/// Sorting costs nothing at compile time and makes one expert one contiguous
/// ~3.5 MB run. The weights themselves are untouched; only their order in the
/// file changes, so reconstruction stays bit-identical.
///
/// Non-layer tensors (embeddings, final norm) sort to the front, keeping their
/// relative order. Within a layer: attention, then the router, then experts in
/// numeric order with each expert's gate/up/down adjacent.
fn tensor_sort_key(name: &str) -> (usize, usize, usize, usize) {
    let layer = name
        .split("layers.")
        .nth(1)
        .and_then(|rest| rest.split('.').next())
        .and_then(|n| n.parse::<usize>().ok())
        .map(|l| l + 1)
        .unwrap_or(0);

    let expert = name
        .split("experts.")
        .nth(1)
        .and_then(|rest| rest.split('.').next())
        .and_then(|n| n.parse::<usize>().ok());

    let proj = if name.contains("gate_proj") {
        0
    } else if name.contains("up_proj") {
        1
    } else if name.contains("down_proj") {
        2
    } else {
        3
    };

    match expert {
        Some(e) => (layer, 2, e, proj),
        None if name.contains("mlp.gate.weight") => (layer, 1, 0, 0),
        None => (layer, 0, 0, proj),
    }
}

/// Reorder for read locality. Stable, so equal keys keep their original order.
fn sort_tensors_for_locality(tensors: &mut [TensorRef]) {
    tensors.sort_by_key(|t| tensor_sort_key(&t.name));
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
    let BlockNorm {
        base,
        scale_i8,
        norm,
    } = normalize_block(weights);

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

    if args.scheme != "v1" && args.scheme != "v2" && args.scheme != "v3" && args.scheme != "v4" {
        eprintln!(
            "Invalid scheme '{}'. Must be 'v1', 'v2', 'v3' or 'v4'.",
            args.scheme
        );
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

    println!(
        "Found {} shards. Extracting tensor metadata...",
        shards.len()
    );

    let mut all_tensors = Vec::new();
    // Tallied so the run can report what it left behind instead of dropping
    // tensors silently.
    let mut skipped_not_2d = 0usize;
    let mut skipped_too_small = 0usize;
    let mut skipped_dtype = 0usize;
    let mut skipped_router = 0usize;
    let mut skipped_bad_width = 0usize;
    let mut skipped_bytes = 0usize;
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
            if shape.len() != 2 {
                // Two very different things land here. 1D tensors are the norms
                // and biases, which are meant to stay dense and which the
                // runtime always reads from the checkpoint. 3D/4D tensors are
                // convolution kernels — and that is why a multimodal
                // checkpoint's vision tower never reaches the .wpc file at all:
                // it starts with a 4D patch convolution, so the whole tower gets
                // dropped here without ever being mentioned.
                skipped_not_2d += 1;
                skipped_bytes += view.data().len();
                continue;
            }

            let n: usize = shape.iter().product();
            if n < 4096 {
                skipped_too_small += 1;
                skipped_bytes += view.data().len();
                continue;
            }

            match view.dtype() {
                safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {}
                _ => {
                    skipped_dtype += 1;
                    skipped_bytes += view.data().len();
                    continue;
                }
            }

            // MoE routers (`mlp.gate.weight`) are deliberately left uncompressed.
            // Every other matrix computes a value, where quantization error
            // averages out over thousands of terms. The router instead makes a
            // discrete choice: when two experts score within quantization noise
            // of each other, rounding decides which one runs and the token takes
            // a different computation path — an error nothing downstream can
            // dilute. All routers together are ~0.04% of a Qwen3-MoE model, so
            // exactness here is nearly free. The runtime reads them from the
            // dense checkpoint, next to the 1D norms.
            //
            // Note the exact suffix: `mlp.gate.weight` is the router,
            // `mlp.gate_proj.weight` is an ordinary SwiGLU gate and IS compressed.
            if name.ends_with(".mlp.gate.weight") {
                skipped_router += 1;
                skipped_bytes += view.data().len();
                continue;
            }

            // For the affine schemes, in_features (shape[1]) must be a whole
            // number of blocks. v2, v3 and v4 all use 128-wide blocks.
            if (args.scheme == "v2" || args.scheme == "v3" || args.scheme == "v4")
                && shape[1] % BLOCK_SIZE_V2 != 0
            {
                eprintln!("Skipping tensor '{}' with shape {:?}: in_features {} not divisible by BLOCK_SIZE_V2 ({})",
                    name, shape, shape[1], BLOCK_SIZE_V2);
                skipped_bad_width += 1;
                skipped_bytes += view.data().len();
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
    println!(
        "Extracted {} valid 2D matrix tensors. Total blocks: {}",
        all_tensors.len(),
        total_blocks
    );

    // Say out loud what was left behind. Silently dropping tensors is how a
    // multimodal model's whole vision tower went missing from a .wpc without
    // anyone noticing until the model was asked to look at something.
    let skipped_total =
        skipped_not_2d + skipped_too_small + skipped_dtype + skipped_router + skipped_bad_width;
    if skipped_total > 0 {
        println!(
            "NOT compressed: {} tensors, {:.1} MB total",
            skipped_total,
            skipped_bytes as f64 / 1024.0 / 1024.0
        );
        if skipped_not_2d > 0 {
            println!(
                "  {skipped_not_2d:6}  not a 2D matrix — 1D norms/biases (expected), \
                 or 3D/4D conv kernels (a vision tower would land here)"
            );
        }
        if skipped_too_small > 0 {
            println!("  {skipped_too_small:6}  fewer than 4096 elements");
        }
        if skipped_dtype > 0 {
            println!("  {skipped_dtype:6}  unsupported dtype (not F32/F16/BF16)");
        }
        if skipped_router > 0 {
            println!("  {skipped_router:6}  MoE routers — deliberate, must stay exact");
        }
        if skipped_bad_width > 0 {
            println!("  {skipped_bad_width:6}  width not divisible by {BLOCK_SIZE_V2}");
        }
        println!("  The runtime must read these from the dense checkpoint.");
    }

    if total_blocks == 0 {
        eprintln!("No valid blocks found to compress.");
        return;
    }

    let mut out_dir = args.output.clone();
    if out_dir.is_file() || out_dir.extension().is_some() {
        out_dir = out_dir.parent().unwrap_or(Path::new("")).to_path_buf();
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    // Group each expert's three matrices together before writing; see
    // tensor_sort_key. This is the single largest win available on the MoE path
    // and it does not touch a single weight. Applies to every scheme.
    sort_tensors_for_locality(&mut all_tensors);

    if args.scheme == "v4" {
        compress_v4(&args, &all_tensors, &out_dir);
    } else if args.scheme == "v3" {
        compress_v3(&args, &all_tensors, &out_dir);
    } else if args.scheme == "v2" {
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
        class_groups
            .entry(class)
            .or_insert_with(Vec::new)
            .push(t_ref);
    }

    // Train per-class codebooks
    let mut class_dicts: HashMap<&'static str, (PatternDict, ResidualDict)> = HashMap::new();
    let mut dict_files: HashMap<String, DictFiles> = HashMap::new();
    use rand::{thread_rng, Rng};

    for (class, tensors) in &class_groups {
        if tensors.is_empty() {
            continue;
        }

        let class_blocks: usize = tensors.iter().map(|t| t.num_blocks).sum();
        let target_sample = 262144;
        let sample_rate = (target_sample as f64 / class_blocks as f64).min(1.0);

        println!(
            "Training codebook for class '{}' ({} tensors, {} blocks)...",
            class,
            tensors.len(),
            class_blocks
        );

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
        let normalized_samples: Vec<[f32; BLOCK_DIM]> = sampled_blocks
            .iter()
            .map(|b| normalize_block(b).norm)
            .collect();
        let pattern_dict = PatternDict::train(&normalized_samples, PATTERN_COUNT, 20);

        // Compute residuals
        let residuals: Vec<[f32; BLOCK_DIM]> = sampled_blocks
            .par_iter()
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
                p_file
                    .write_all(&(val / INPUT_SCALE).to_le_bytes())
                    .unwrap();
            }
        }

        let mut r_file = File::create(&residuals_path).unwrap();
        for c in &residual_dict.centroids_f16 {
            for &val in c {
                r_file.write_all(&val.to_le_bytes()).unwrap();
            }
        }

        dict_files.insert(
            class.to_string(),
            DictFiles {
                patterns: patterns_filename,
                residuals: residuals_filename,
            },
        );

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
        println!(
            "  Encoding tensor {}/{} ({})...",
            idx + 1,
            all_tensors.len(),
            t_ref.name
        );
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

        let bytes: Vec<u8> = compressed_blocks
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
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
        println!(
            "  Encoding tensor {}/{} ({})...",
            idx + 1,
            all_tensors.len(),
            t_ref.name
        );
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

        let bytes: Vec<u8> = compressed_blocks
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        wpc_file.write_all(&bytes).unwrap();

        current_offset += size_bytes;

        drop(t_data);
    }

    let meta_path = out_dir.join("model_v2.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();

    println!("Done! Compiled WPC v2 model saved to {:?}", out_dir);
}

/// v3: identical quantization to v2, with the 6-bit codes bit-packed.
///
/// 100 bytes per 128 weights instead of 132, so a v3 model is ~24% smaller
/// than the v2 model of the same tensors, with bit-identical reconstruction.
/// Since generation here is memory-bandwidth bound, those bytes are also
/// roughly 24% of the generation time.
///
/// The output keeps the v2 file names and metadata shape on purpose: only the
/// block layout differs, so the runtime distinguishes them by `--scheme`, not
/// by parsing a different manifest.
fn compress_v3(_args: &Args, all_tensors: &[TensorRef], out_dir: &Path) {
    println!("Starting v3 (affine 6-bit, packed) compression...");
    let out_wpc = out_dir.join("model_v3.wpc");
    let mut wpc_file = File::create(&out_wpc).unwrap();
    let mut current_offset = 0;

    let mut meta = ModelMetaV2 {
        layers: Vec::new(),
        block_size: BLOCK_SIZE_V2,
    };

    let mut v2_equivalent_bytes = 0usize;

    for (idx, t_ref) in all_tensors.iter().enumerate() {
        println!(
            "  Encoding tensor {}/{} ({})...",
            idx + 1,
            all_tensors.len(),
            t_ref.name
        );
        let t_data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);

        let compressed_blocks: Vec<QuantBlockV3> = quant_encoder::encode_tensor_v3(&t_data);

        let size_bytes = compressed_blocks.len() * QuantBlockV3::SIZE;
        v2_equivalent_bytes += compressed_blocks.len() * QuantBlockV2::SIZE;

        meta.layers.push(LayerMetaV2 {
            name: t_ref.name.clone(),
            shape: t_ref.shape.clone(),
            offset_bytes: current_offset,
            size_bytes,
        });

        let bytes: Vec<u8> = compressed_blocks
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        wpc_file.write_all(&bytes).unwrap();

        current_offset += size_bytes;

        drop(t_data);
    }

    let meta_path = out_dir.join("model_v3.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();

    let saved = v2_equivalent_bytes.saturating_sub(current_offset);
    println!(
        "Done! Compiled WPC v3 model saved to {:?}\n  \
         size: {:.2} GB (v2 would be {:.2} GB, saved {:.2} GB = {:.1}%)",
        out_dir,
        current_offset as f64 / 1024.0 / 1024.0 / 1024.0,
        v2_equivalent_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        saved as f64 / 1024.0 / 1024.0 / 1024.0,
        if v2_equivalent_bytes > 0 {
            saved as f64 * 100.0 / v2_equivalent_bytes as f64
        } else {
            0.0
        },
    );
}

/// v4: the same affine quantization at 4 bits, two codes per byte.
///
/// 68 bytes per 128 weights instead of v3's 100, so a v4 model is 32% smaller
/// than the v3 model of the same tensors. Unlike the v2 -> v3 step this is not
/// free: 16 levels per block instead of 64 is a real loss of precision, paid to
/// make a model fit a card that v3 misses.
///
/// The output uses its own file names (`model_v4.wpc` / `model_v4.meta`) so a
/// v3 and a v4 build can share one directory, and the runtime picks between
/// them by `--scheme`.
fn compress_v4(_args: &Args, all_tensors: &[TensorRef], out_dir: &Path) {
    println!("Starting v4 (affine 4-bit, packed) compression...");
    let out_wpc = out_dir.join("model_v4.wpc");
    let mut wpc_file = File::create(&out_wpc).unwrap();
    let mut current_offset = 0;

    let mut meta = ModelMetaV2 {
        layers: Vec::new(),
        block_size: BLOCK_SIZE_V2,
    };

    let mut v3_equivalent_bytes = 0usize;

    for (idx, t_ref) in all_tensors.iter().enumerate() {
        println!(
            "  Encoding tensor {}/{} ({})...",
            idx + 1,
            all_tensors.len(),
            t_ref.name
        );
        let t_data = read_tensor_f32(&t_ref.shard_path, &t_ref.name);

        let compressed_blocks: Vec<QuantBlockV4> = quant_encoder::encode_tensor_v4(&t_data);

        let size_bytes = compressed_blocks.len() * QuantBlockV4::SIZE;
        v3_equivalent_bytes += compressed_blocks.len() * QuantBlockV3::SIZE;

        meta.layers.push(LayerMetaV2 {
            name: t_ref.name.clone(),
            shape: t_ref.shape.clone(),
            offset_bytes: current_offset,
            size_bytes,
        });

        let bytes: Vec<u8> = compressed_blocks
            .iter()
            .flat_map(|b| b.to_le_bytes())
            .collect();
        wpc_file.write_all(&bytes).unwrap();

        current_offset += size_bytes;

        drop(t_data);
    }

    let meta_path = out_dir.join("model_v4.meta");
    let meta_json = serde_json::to_string_pretty(&meta).unwrap();
    std::fs::write(&meta_path, meta_json).unwrap();

    let saved = v3_equivalent_bytes.saturating_sub(current_offset);
    println!(
        "Done! Compiled WPC v4 model saved to {:?}\n  \
         size: {:.2} GB / {:.0} MiB (v3 would be {:.2} GB, saved {:.2} GB = {:.1}%)",
        out_dir,
        current_offset as f64 / 1024.0 / 1024.0 / 1024.0,
        current_offset as f64 / 1024.0 / 1024.0,
        v3_equivalent_bytes as f64 / 1024.0 / 1024.0 / 1024.0,
        saved as f64 / 1024.0 / 1024.0 / 1024.0,
        if v3_equivalent_bytes > 0 {
            saved as f64 * 100.0 / v3_equivalent_bytes as f64
        } else {
            0.0
        },
    );
}
